"""
fb_tools/weather/hrrr.py
========================
HRRR fire-hour wind extraction for FSPro/FSim wind climatology.

Builds the ``WindCellValues`` frequency table required by FSPro from
historical HRRR analysis data at fire occurrence locations. Samples
HRRR 10m wind (U, V) at each FOD point during fire-hour UTC windows
and builds a NumWindSpeeds × NumWindDirs frequency table per group.

**FSPro wind semantics:** ``WindCellValues`` is a stochastic climatology
distribution — FSPro independently draws a wind speed/direction from this
table for each simulated fire. It is *not* a schedule of daily wind values.
``CalmValue`` is the complementary percentage of historically-calm
observations and is stored separately in the FSPro input file.

**Generality:** The grouping key (``pyrome_col`` parameter) can be any
categorical column — pyrome ID, watershed ID, a constant string for a
single-area run, or any other spatial unit. The default naming reflects the
typical pyrome climatology use case but imposes no restriction.

**Single-event use:** To parameterize a specific historic fire event (e.g.,
for a deterministic FlamMap/MTT run), call ``fetch_hrrr_winds_at_fires()``
directly and use the returned raw wind DataFrame. The frequency table is
most useful for multi-fire probabilistic FSPro/FSim climatology.

**Sample size:** A well-populated 6×8 frequency table typically requires
200+ non-calm observations per group. With 3–5 test fires (~20–30 obs)
most bins will be zero — this is expected, not a bug.

HRRR archive coverage: 2016–present (AWS, ``s3://noaa-hrrr-bdp-pds/``).
Pre-2016 archive has gaps; 2014–2015 data should not be relied upon.
Access via the ``herbie-data`` library (optional dependency).

Workflow
--------
1. Filter FOD to HRRR coverage period (2016+).
2. Expand each fire to n_days × fire_hours schedule entries.
3. Deduplicate to unique (date, UTC-hour) pairs → one download per pair.
4. For each pair: download HRRR U/V 10m with herbie; extract values at
   all active fire point locations in one KD-tree lookup.
5. Group by grouping column → build WindCellValues frequency table + CalmValue.

References
----------
Benjamin et al. (2016), A North American Hourly Assimilation and Model
Forecast Cycle: The Rapid Refresh, Mon. Wea. Rev., 144, 1669–1694.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ── Module-level constants ─────────────────────────────────────────────────────
_DEFAULT_SPEED_BREAKS_MPH: list[float] = [5, 10, 15, 20, 25, 30]
_DEFAULT_DIR_BREAKS_DEG: list[float] = [45, 90, 135, 180, 225, 270, 315, 360]
_CALM_THRESHOLD_MPH: float = 2.0      # obs below this excluded from wind rose
_MS_TO_MPH: float = 2.23694
_HRRR_START_YEAR: int = 2016          # HRRR AWS archive begins mid-2014


# ── Internal helpers ───────────────────────────────────────────────────────────

def _uv_to_ws_wd(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert U/V wind components (m/s) to speed (mph) and meteorological
    direction (degrees the wind is coming FROM).

    Parameters
    ----------
    u : np.ndarray
        Eastward wind component (m/s). Positive = blowing toward east.
    v : np.ndarray
        Northward wind component (m/s). Positive = blowing toward north.

    Returns
    -------
    ws_mph : np.ndarray
    wd_deg : np.ndarray
        Direction FROM which wind is blowing, 0–360° (0/360 = north).
    """
    ws_mph = np.sqrt(u ** 2 + v ** 2) * _MS_TO_MPH
    # Meteorological FROM convention: atan2(-u, -v) + 360 mod 360
    # Verification: u=5, v=0 (eastward) → 270° (FROM west) ✓
    wd_deg = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    return ws_mph, wd_deg


def _extract_uv(ds) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract U and V 10m wind arrays from a herbie xarray Dataset.

    herbie/cfgrib can assign different short names depending on GRIB2
    parameter encoding; this helper accepts common variants.

    Returns
    -------
    u, v : np.ndarray  (2D, shape matching the HRRR grid)
    """
    dvars = list(ds.data_vars)
    u_names = {"u", "u10", "10u", "ugrd", "UGRD"}
    v_names = {"v", "v10", "10v", "vgrd", "VGRD"}
    u_var = next((v for v in dvars if v in u_names), None)
    v_var = next((v for v in dvars if v in v_names), None)
    if u_var is None or v_var is None:
        raise KeyError(
            f"Cannot identify U/V wind variables in HRRR dataset. "
            f"Available variables: {dvars}"
        )
    return ds[u_var].values, ds[v_var].values


def _build_kd_tree(lat2d: np.ndarray, lon2d: np.ndarray):
    """
    Build a scipy cKDTree from a 2D lat/lon grid.

    HRRR longitude may be 0–360; converts to -180/180 before building
    so fire point coordinates (WGS84, -180/180) match correctly.
    """
    from scipy.spatial import cKDTree
    # Normalise longitude to -180/180
    lon_180 = (lon2d + 180) % 360 - 180
    coords = np.column_stack([lat2d.ravel(), lon_180.ravel()])
    return cKDTree(coords), lon_180.ravel()


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_hrrr_winds_at_fires(
    fod_gdf,
    fire_hours_utc: list[int] | None = None,
    n_days: int = 2,
    date_col: str = "ignition_date",
    pyrome_col: str = "pyrome_id",
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Extract HRRR 10m wind at FOD fire locations during fire-hour UTC windows.

    For each unique (date, UTC-hour) across all fires + n_days window,
    downloads one HRRR analysis file and extracts U/V at all active fire
    point locations in a single KD-tree lookup.

    Parameters
    ----------
    fod_gdf : geopandas.GeoDataFrame
        Fire occurrence points. Required columns:
        - ``geometry``: Point geometry (any projected or geographic CRS).
        - ``date_col``: Ignition date (datetime-compatible).
        - ``pyrome_col``: Pyrome identifier (str or int).
    fire_hours_utc : list of int, optional
        UTC hours to extract. Defaults to ``[19, 20, 21, 22]``
        (≈ 13:00–16:00 MDT / 14:00–17:00 MST — peak fire-spread window).
    n_days : int
        Consecutive days to sample per fire (ignition day + n_days-1
        following days). Default 2 captures ignition and early spread.
    date_col : str
        Column name for ignition date in ``fod_gdf``.
    pyrome_col : str
        Column name for pyrome identifier in ``fod_gdf``.
    cache_dir : Path or str, optional
        Directory for herbie GRIB cache. Strongly recommended to avoid
        re-downloading on repeat runs (~6 GB for ~900 CO fires).

    Returns
    -------
    pd.DataFrame
        Columns: ``pyrome_id, fire_id, date, utc_hour, lon, lat,
        ws_mph, wd_deg``. One row per (fire × day × UTC-hour).

    Raises
    ------
    ImportError
        If ``herbie-data`` is not installed.
    ValueError
        If no FOD fires fall within the HRRR coverage period.
    RuntimeError
        If no wind records could be extracted (all downloads failed).
    """
    try:
        from herbie import Herbie
    except ImportError as exc:
        raise ImportError(
            "herbie-data is required for HRRR wind extraction.\n"
            "Install with:  pip install herbie-data"
        ) from exc

    if fire_hours_utc is None:
        fire_hours_utc = [19, 20, 21, 22]

    # ── Prepare FOD points ─────────────────────────────────────────────────────
    fod = fod_gdf.copy()
    fod[date_col] = pd.to_datetime(fod[date_col])

    n_total = len(fod)
    fod = fod[fod[date_col].dt.year >= _HRRR_START_YEAR].copy().reset_index(drop=False)
    if fod.empty:
        raise ValueError(
            f"No FOD fires fall within HRRR coverage period ({_HRRR_START_YEAR}+). "
            f"Input had {n_total} fires."
        )
    print(f"  {len(fod)} / {n_total} fires in HRRR period ({_HRRR_START_YEAR}+)")

    # Reproject to WGS84 for KD-tree matching
    fod_wgs84 = fod.to_crs("EPSG:4326")
    fire_lons = fod_wgs84.geometry.x.values   # -180/180
    fire_lats = fod_wgs84.geometry.y.values

    # ── Build download schedule ────────────────────────────────────────────────
    # Expand each fire to (n_days × n_fire_hours) schedule entries.
    # Group by unique (target_date, utc_hour) → list of fire positions.
    date_hour_fires: dict[tuple, list[int]] = defaultdict(list)

    for pos, row in fod.iterrows():
        base = row[date_col].normalize()
        for day in range(n_days):
            tdate = base + pd.Timedelta(days=day)
            for utc_h in fire_hours_utc:
                date_hour_fires[(tdate, utc_h)].append(pos)

    unique_pairs = sorted(date_hour_fires.keys())
    n_raw = len(fod) * n_days * len(fire_hours_utc)
    print(f"  {n_raw} raw schedule entries → {len(unique_pairs)} unique (date, hour) downloads")

    # ── herbie configuration ───────────────────────────────────────────────────
    herbie_kwargs: dict = {"model": "hrrr", "fxx": 0, "product": "sfc", "verbose": False}
    if cache_dir is not None:
        herbie_kwargs["save_dir"] = Path(cache_dir)

    searchstring = "UGRD:10 m above ground|VGRD:10 m above ground"

    kd_tree = None
    lon_flat = None
    records: list[dict] = []
    n_skip = 0

    # ── Main download loop ─────────────────────────────────────────────────────
    for i, (target_date, utc_hour) in enumerate(unique_pairs):
        dt_str = target_date.strftime("%Y-%m-%d") + f" {utc_hour:02d}:00"

        try:
            H = Herbie(dt_str, **herbie_kwargs)
            ds = H.xarray(searchstring, remove_grib=False)
        except Exception as exc:
            n_skip += 1
            if n_skip <= 10 or n_skip % 50 == 0:
                print(f"  [{i+1}/{len(unique_pairs)}] SKIP {dt_str}: {exc}")
            continue

        if i % 100 == 0:
            print(f"  [{i+1}/{len(unique_pairs)}] {dt_str}  ({len(records):,} obs so far)")

        # Build KD-tree once (HRRR grid is fixed across all analysis hours)
        if kd_tree is None:
            hrrr_lats = ds["latitude"].values
            hrrr_lons = ds["longitude"].values
            kd_tree, lon_flat = _build_kd_tree(hrrr_lats, hrrr_lons)
            lat_flat = hrrr_lats.ravel()
            print(f"  HRRR grid shape: {hrrr_lats.shape} ({hrrr_lats.size:,} grid points)")

        # Extract U and V; lookup fire points
        u_grid, v_grid = _extract_uv(ds)
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()

        fire_positions = date_hour_fires[(target_date, utc_hour)]
        query_lats = fire_lats[fire_positions]
        query_lons = fire_lons[fire_positions]

        _, grid_idx = kd_tree.query(np.column_stack([query_lats, query_lons]))

        u_vals = u_flat[grid_idx]
        v_vals = v_flat[grid_idx]
        ws_mph, wd_deg = _uv_to_ws_wd(u_vals, v_vals)

        for j, pos in enumerate(fire_positions):
            row = fod.iloc[pos]
            records.append({
                "pyrome_id": row[pyrome_col],
                "fire_id": row.get("index", pos),   # original index if preserved
                "date": target_date.date(),
                "utc_hour": utc_hour,
                "lon": float(fire_lons[pos]),
                "lat": float(fire_lats[pos]),
                "ws_mph": float(ws_mph[j]),
                "wd_deg": float(wd_deg[j]),
            })

    if n_skip > 0:
        print(f"  Warning: {n_skip} / {len(unique_pairs)} downloads skipped (file not available)")

    if not records:
        raise RuntimeError(
            "No HRRR wind records extracted. Check herbie installation, "
            "HRRR archive availability, and FOD date range."
        )

    result = pd.DataFrame(records)
    print(
        f"  Done: {len(result):,} wind observations across "
        f"{result['pyrome_id'].nunique()} pyromes"
    )
    return result


def build_wind_cells(
    ws_mph: "pd.Series | np.ndarray",
    wd_deg: "pd.Series | np.ndarray",
    speed_breaks: list[float] | None = None,
    dir_breaks: list[float] | None = None,
    calm_threshold_mph: float = _CALM_THRESHOLD_MPH,
) -> "tuple[np.ndarray, float]":
    """
    Build a NumWindSpeeds × NumWindDirs frequency table from wind observations.

    Each cell contains the percentage of non-calm observations that fell in
    that (speed, direction) bin, matching the ``WindCellValues`` format in
    FSPro input files. Calm observations are excluded from the table and
    reported separately as ``calm_pct``.

    Parameters
    ----------
    ws_mph : array-like
        Wind speed observations in mph.
    wd_deg : array-like
        Wind direction in degrees, meteorological FROM convention (0 = north).
    speed_breaks : list of float, optional
        Upper bounds of each speed bin (mph, inclusive). The last bin
        captures all observations above the second-to-last break.
        Default: ``[5, 10, 15, 20, 25, 30]`` (6 bins, matches FSPro sample).
    dir_breaks : list of float, optional
        Upper bounds of each direction bin (°, inclusive). 0° maps to the
        last (360°) bin. Default: ``[45, 90, 135, 180, 225, 270, 315, 360]``
        (8 bins, N at 360°, matches FSPro sample).
    calm_threshold_mph : float
        Observations with ws_mph below this value are classified as calm and
        excluded from the frequency table. Default 2.0 mph.

    Returns
    -------
    wind_cells : np.ndarray
        Shape ``(NumWindSpeeds, NumWindDirs)``. Values are % frequencies of
        non-calm observations (sum ≈ 100). Row order = ascending speed bin;
        column order = ascending direction azimuth bin.
    calm_pct : float
        Percentage of *all* observations (including calm) below
        ``calm_threshold_mph``. Corresponds to ``CalmValue`` in FSPro input
        files. ``wind_cells.sum() + calm_pct ≈ 100``.

    Raises
    ------
    ValueError
        If no non-calm observations remain after filtering.
    """
    if speed_breaks is None:
        speed_breaks = _DEFAULT_SPEED_BREAKS_MPH
    if dir_breaks is None:
        dir_breaks = _DEFAULT_DIR_BREAKS_DEG

    ws = np.asarray(ws_mph, dtype=float)
    wd = np.asarray(wd_deg, dtype=float)

    n_total = len(ws)
    # Remove calm / near-calm observations; track count for CalmValue
    active = ws >= calm_threshold_mph
    n_calm = int((~active).sum())
    calm_pct = 100.0 * n_calm / n_total if n_total > 0 else 0.0

    ws = ws[active]
    wd = wd[active]

    if len(ws) == 0:
        raise ValueError(
            f"All {n_total} observations have ws_mph < {calm_threshold_mph}; "
            "cannot build wind cell table."
        )

    n_speed = len(speed_breaks)
    n_dir = len(dir_breaks)
    speed_arr = np.array(speed_breaks, dtype=float)
    dir_arr = np.array(dir_breaks, dtype=float)

    # Bin speed: upper-bound inclusive; last bin catches all obs above max break
    speed_idx = np.searchsorted(speed_arr, ws, side="left")
    speed_idx = np.clip(speed_idx, 0, n_speed - 1)

    # Bin direction: upper-bound inclusive; 0° treated as 360° (north = last bin)
    wd_adj = wd.copy()
    wd_adj[wd_adj == 0.0] = 360.0
    dir_idx = np.searchsorted(dir_arr, wd_adj, side="left")
    dir_idx = np.clip(dir_idx, 0, n_dir - 1)

    counts = np.zeros((n_speed, n_dir), dtype=float)
    np.add.at(counts, (speed_idx, dir_idx), 1)

    freq_pct = counts / counts.sum() * 100.0
    return freq_pct, calm_pct


def build_pyrome_wind_cells(
    fod_gdf,
    cache_dir: "Path | str | None" = None,
    speed_breaks: "list[float] | None" = None,
    dir_breaks: "list[float] | None" = None,
    date_col: str = "ignition_date",
    pyrome_col: str = "pyrome_id",
    fire_hours_utc: "list[int] | None" = None,
    n_days: int = 5,
    out_dir: "Path | str | None" = None,
    min_obs_warn: int = 100,
) -> "dict[str, np.ndarray]":
    """
    Build per-area wind frequency tables from HRRR data at fire occurrence locations.

    Fetches HRRR 10m winds for all fires, groups observations by the column
    named by ``pyrome_col``, and builds a ``WindCellValues`` array for each
    group. Optionally writes per-group JSON cache files for reuse in FSPro
    input generation.

    The grouping key (``pyrome_col``) can be any categorical column — pyrome
    ID, watershed ID, a constant string for a single-area run, etc.  The
    default name ``"pyrome_id"`` reflects the typical pyrome climatology use
    case but imposes no restriction on the analysis unit.

    For **single-event** parameterization (e.g., recreate a specific historic
    fire for an MTT run), call ``fetch_hrrr_winds_at_fires()`` directly and
    use the raw wind DataFrame; the ``WindCellValues`` table is most useful
    for multi-fire probabilistic FSPro/FSim climatology.

    Parameters
    ----------
    fod_gdf : geopandas.GeoDataFrame
        Fire occurrence points. Requires: geometry, ``date_col``, ``pyrome_col``.
    cache_dir : Path or str, optional
        herbie GRIB cache directory (avoids re-downloading on repeat runs).
    speed_breaks : list of float, optional
        Speed bin upper bounds (mph). Default: [5, 10, 15, 20, 25, 30].
    dir_breaks : list of float, optional
        Direction bin upper bounds (°). Default: [45, 90, 135, 180, 225, 270, 315, 360].
    date_col : str
        Ignition date column in ``fod_gdf``. Default ``"ignition_date"``.
    pyrome_col : str
        Grouping column in ``fod_gdf`` (pyrome ID, watershed ID, etc.).
        Default ``"pyrome_id"``.
    fire_hours_utc : list of int, optional
        UTC hours to sample. Default: [19, 20, 21, 22].
    n_days : int
        Days per fire to sample starting from ignition date. Default 5.
    out_dir : Path or str, optional
        If provided, writes ``pyrome_{id}_wind.json`` per group to this
        directory. Load with ``load_pyrome_wind_cells()``.
    min_obs_warn : int
        Print a warning for any group whose non-calm observation count is
        below this threshold. Default 100. A well-populated 6×8 FSPro table
        typically requires 200+ non-calm observations; fewer than 100 will
        produce a sparse distribution unreliable for probabilistic runs.
        Set to 0 to suppress.

    Returns
    -------
    dict[str, np.ndarray]
        ``{group_id: wind_cells_array}`` where each array has shape
        ``(NumWindSpeeds, NumWindDirs)``.
    """
    if speed_breaks is None:
        speed_breaks = list(_DEFAULT_SPEED_BREAKS_MPH)
    if dir_breaks is None:
        dir_breaks = list(_DEFAULT_DIR_BREAKS_DEG)

    print("Building wind climatology from HRRR ...")
    wind_df = fetch_hrrr_winds_at_fires(
        fod_gdf,
        fire_hours_utc=fire_hours_utc,
        n_days=n_days,
        date_col=date_col,
        pyrome_col=pyrome_col,
        cache_dir=cache_dir,
    )

    years = pd.to_datetime(wind_df["date"]).dt.year
    year_range = f"{years.min()}–{years.max()}"

    result: dict[str, np.ndarray] = {}
    for group_id, grp in wind_df.groupby("pyrome_id"):
        pid = str(group_id)
        cells, calm_pct = build_wind_cells(
            grp["ws_mph"],
            grp["wd_deg"],
            speed_breaks=speed_breaks,
            dir_breaks=dir_breaks,
        )
        n_noncalm = int(round(len(grp) * (1.0 - calm_pct / 100.0)))
        if min_obs_warn > 0 and n_noncalm < min_obs_warn:
            print(
                f"  Warning: group '{pid}' has only {n_noncalm} non-calm observations "
                f"(calm={calm_pct:.1f}%). "
                f"Recommend ≥{min_obs_warn} for a reliable FSPro climatology table."
            )
        result[pid] = cells
        print(
            f"  Group {pid}: {len(grp):,} obs "
            f"({n_noncalm} non-calm, calm={calm_pct:.1f}%) "
            f"→ {cells.shape} wind cell table"
        )

        if out_dir is not None:
            out_path = _write_wind_cells_json(
                pyrome_id=pid,
                wind_cells=cells,
                speed_breaks=speed_breaks,
                dir_breaks=dir_breaks,
                calm_pct=calm_pct,
                n_obs=len(grp),
                years_covered=year_range,
                out_dir=Path(out_dir),
            )
            print(f"    → {out_path.name}")

    return result


def _write_wind_cells_json(
    pyrome_id: str,
    wind_cells: np.ndarray,
    speed_breaks: list[float],
    dir_breaks: list[float],
    calm_pct: float,
    n_obs: int,
    years_covered: str,
    out_dir: Path,
) -> Path:
    """Write per-area wind cells to JSON cache file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "pyrome_id": pyrome_id,
        "NumWindSpeeds": len(speed_breaks),
        "NumWindDirs": len(dir_breaks),
        "WindSpeedBreaks_mph": speed_breaks,
        "WindDirBreaks_deg": dir_breaks,
        "WindCellValues": wind_cells.tolist(),
        "CalmValue": round(calm_pct, 4),
        "n_observations": n_obs,
        "years_covered": years_covered,
    }
    out_path = out_dir / f"pyrome_{pyrome_id}_wind.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def load_pyrome_wind_cells(
    pyrome_id: "str | int",
    cache_dir: "Path | str",
    return_meta: bool = False,
) -> "np.ndarray | dict":
    """
    Load a cached wind frequency table from JSON.

    Parameters
    ----------
    pyrome_id : str or int
        Group identifier matching the JSON filename prefix (pyrome ID,
        watershed ID, etc.).
    cache_dir : Path or str
        Directory containing ``pyrome_{id}_wind.json`` files (written by
        ``build_pyrome_wind_cells()``).
    return_meta : bool
        If ``False`` (default), return only the ``(NumWindSpeeds, NumWindDirs)``
        frequency array — backward-compatible.  If ``True``, return the full
        metadata dict with keys: ``pyrome_id``, ``NumWindSpeeds``,
        ``NumWindDirs``, ``WindSpeedBreaks_mph``, ``WindDirBreaks_deg``,
        ``WindCellValues`` (np.ndarray), ``CalmValue``, ``n_observations``,
        ``years_covered``.

    Returns
    -------
    np.ndarray or dict
        Frequency table array, or full metadata dict when ``return_meta=True``.

    Raises
    ------
    FileNotFoundError
        If no cached file exists for this group.
    """
    path = Path(cache_dir) / f"pyrome_{pyrome_id}_wind.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No wind cells cache for '{pyrome_id}' in {cache_dir}.\n"
            "Run build_pyrome_wind_cells() first to generate the cache."
        )
    with open(path) as f:
        data = json.load(f)
    if not return_meta:
        return np.array(data["WindCellValues"], dtype=float)
    data["WindCellValues"] = np.array(data["WindCellValues"], dtype=float)
    return data
