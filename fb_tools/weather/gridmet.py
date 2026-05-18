"""
fb_tools/weather/gridmet.py
============================
Process GEE-exported GridMET CSV into per-pyrome FSPro weather arrays.

The expected input is a CSV exported from the ``GridMET_ERC_Climatology``
GEE notebook: one row per (pyrome, date) covering the full fire season
(April 1 – October 31, DOY 91–304, 214 days) for years 2008–2022.

Required CSV columns
--------------------
pyrome_id, date, year, doy, erc, fm100, fm1000, tmmx, rmin

- ``tmmx``: daily maximum temperature (°K) — converted to °F on load
- ``rmin``: daily minimum relative humidity (%) — used with tmmx for NFDRS FM

Outputs feed directly into ``build_fspro_inputs()`` in ``fb_tools.models.fspro``:

- ``build_historic_erc_arrays()``  → ``HistoricERCValues`` block
- ``build_erc_stats()``            → ``AvgERCValues`` / ``StdDevERCValues``
- ``build_erc_classes()``          → ``NumERCClasses`` 5-row table
- ``build_current_erc_values()``   → ``CurrentERCValues`` (scenario-based median)

JSON cache pattern
------------------
Each function that writes a cache saves ``pyrome_{id}_gridmet.json`` in
``out_dir`` / ``cache_dir``.  Matches the pattern used in ``hrrr.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fb_tools.weather.nfdrs import (
    calc_1hr_fm,
    calc_10hr_fm_instantaneous,
    calc_herb_fm,
    calc_herb_fm_gsi,
    calc_woody_fm,
    calc_woody_fm_gsi,
    calc_gsi,
    calc_vpd_pa,
    calc_daylength,
    calc_live_fm_from_dead,
    kelvin_to_fahrenheit,
)
from fb_tools.weather.fm_scenario import build_fm_timeseries

# FSPro fire-season constants (April 1 – October 31)
_N_SEASON_DAYS: int = 214
_SEASON_START_MONTH: int = 4   # April
_SEASON_END_MONTH: int = 10    # October

# Default spotting parameter rows per ERC class (high → low danger, 5 rows)
# Format: [spot_dist_ft, spot_prob, spot2]
# Matches the 416inputsfile.input example file ordering (highest ERC first).
_DEFAULT_SPOTTING: list[list[float]] = [
    [360, 0.15, 0],
    [300, 0.10, 0],
    [240, 0.05, 0],
    [180, 0.01, 0],
    [120, 0.00, 0],
]

# Scale factors: NFDRS78-lag 100-hr FM (fire-season range ~5–25%) → live FM.
# Chosen so that p25-danger fm100 (~20%) maps to actively-green herb (~130%),
# and p97-danger fm100 (~7%) maps near the dormant floor (30% herb, 60% woody).
# DOY-based phenology is unsuitable as a fallback because all ERC percentile
# bands tend to cluster at the same calendar date within a fire season.
_LIVE_FM_HERB_SCALE: float = 6.5
_LIVE_FM_WOODY_SCALE: float = 9.0


# ── Public API ─────────────────────────────────────────────────────────────────

# Variable classification for percentile selection.  "Hot" variables are
# fire-dangerous at the HIGH end of the cell distribution (select p75/p90 to
# capture the hottest cells within the fire mask).  "Wet" variables are
# fire-dangerous at the LOW end (select p10/p25 for driest fuels / lowest
# humidity).  "Neutral" variables use the median regardless of tail choice.
_HOT_VARS = ("tmmx", "tmmn", "vpd", "erc", "vs")
_WET_VARS = ("fm100", "fm1000", "rmin", "rmax")
_NEUTRAL_VARS = ("pr", "th")


def load_gridmet_csv(
    csv_path: str | Path,
    tail_percentile: int = 50,
) -> pd.DataFrame:
    """
    Load and normalize the GEE-exported GridMET fire-season CSV.

    Accepts two export formats:

    1. **Legacy flat format** — one column per variable (``erc``, ``fm100``,
       ``tmmx``, ``rmin``, …).  ``tail_percentile`` is ignored.
    2. **Percentile-suffix format** (new; exported by
       ``00a_gridMET-ERC.ipynb`` after the fire-mask + multi-percentile
       refactor) — five columns per variable, e.g., ``erc_p10, erc_p25,
       erc_p50, erc_p75, erc_p90``.  The function selects a single column per
       variable based on ``tail_percentile`` and aliases it to the flat name
       for downstream compatibility.

    Parameters
    ----------
    csv_path : str or Path
        Path to the exported CSV file.
    tail_percentile : int
        Only used for percentile-suffix CSVs.  Selects the fire-dangerous tail
        of the cell distribution per variable:

        - ``50`` (default) — cell median, matches the old single-reducer
          export behaviour.
        - ``75`` — ``p75`` of hot/dry variables (tmmx, erc, vs, vpd) paired
          with ``p25`` of moisture variables (fm100, rmin, rmax).  Approximates
          the driest quartile of cells within the fire mask.
        - ``90`` — ``p90`` / ``p10``.  Driest decile; most fire-extreme.
        - Neutral variables (pr, th) always use ``p50``.

        Must be one of ``{10, 25, 50, 75, 90}`` (the set exported by the
        GEE reducer).  Legacy flat CSVs ignore this parameter.

    Returns
    -------
    pd.DataFrame
        Flat-column DataFrame with ``pyrome``, ``date``, ``year``, ``doy``,
        ``erc``, ``fm100``, ``fm1000``, ``tmmx_f`` (°F), ``rmin``, and any
        optional variables present in the CSV (``tmmn_f``, ``rmax``, ``pr``,
        ``vpd_pa``, ``ws_mph``, ``wd_deg``).

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    KeyError
        If required columns are missing from the CSV.
    ValueError
        If ``tail_percentile`` is not in ``{10, 25, 50, 75, 90}``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"GridMET CSV not found: {csv_path}")

    allowed_pctiles = {10, 25, 50, 75, 90}
    if tail_percentile not in allowed_pctiles:
        raise ValueError(
            f"tail_percentile must be one of {sorted(allowed_pctiles)}; got {tail_percentile}"
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect format by looking for percentile-suffix columns.
    is_percentile_format = any(
        f"{v}_p{tail_percentile}" in df.columns for v in _HOT_VARS + _WET_VARS
    ) or any(f"erc_p{p}" in df.columns for p in allowed_pctiles)

    if is_percentile_format:
        df = _select_percentile_columns(df, tail_percentile)
        fmt_label = f"percentile-suffix (tail={tail_percentile})"
    else:
        fmt_label = "legacy flat"

    required = {"pyrome", "date", "erc", "fm100", "fm1000", "tmmx", "rmin"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "doy" not in df.columns:
        df["doy"] = df["date"].dt.dayofyear

    df["year"] = df["year"].astype(int)
    df["doy"] = df["doy"].astype(int)

    # Convert tmmx from °K → °F
    df["tmmx_f"] = kelvin_to_fahrenheit(df["tmmx"].values)
    df = df.drop(columns=["tmmx"])

    # Optional columns — add tmmn/rmax/pr/vs/th to GEE export as needed
    if "tmmn" in df.columns:
        df["tmmn_f"] = kelvin_to_fahrenheit(df["tmmn"].values)
        df = df.drop(columns=["tmmn"])
    for col in ("rmax", "pr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # vpd exported from GEE in kPa — store as Pa for direct use in calc_gsi()
    if "vpd" in df.columns:
        df["vpd_pa"] = pd.to_numeric(df["vpd"], errors="coerce") * 1000.0
    # GridMET wind: vs = daily mean wind speed (m/s); th = wind direction (°, met FROM)
    if "vs" in df.columns:
        df["ws_mph"] = pd.to_numeric(df["vs"], errors="coerce") * 2.23694
        df = df.drop(columns=["vs"])
    if "th" in df.columns:
        df["wd_deg"] = pd.to_numeric(df["th"], errors="coerce")
        df = df.drop(columns=["th"])

    for col in ["erc", "fm100", "fm1000", "tmmx_f", "rmin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["erc", "pyrome"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  [load_gridmet_csv] Dropped {n_before - len(df)} rows with missing erc/PYROME")

    print(f"  [load_gridmet_csv] {len(df):,} rows, {df['pyrome'].nunique()} pyromes, "
          f"years {df['year'].min()}–{df['year'].max()}  [{fmt_label}]")
    return df


def _select_percentile_columns(df: pd.DataFrame, tail_percentile: int) -> pd.DataFrame:
    """
    Collapse a multi-percentile GridMET DataFrame to flat columns.

    For each fire-danger-relevant variable, select the column matching the
    fire-dangerous tail of the cell distribution (high end for hot/dry vars,
    low end for moisture vars) and rename to the flat variable name.  All
    unused ``_pXX`` columns are dropped so downstream code sees the same
    schema as the legacy flat format.
    """
    hot_suffix = f"_p{tail_percentile}"
    wet_suffix = f"_p{100 - tail_percentile}"
    neutral_suffix = "_p50"

    renames: dict[str, str] = {}
    for v in _HOT_VARS:
        src = f"{v}{hot_suffix}"
        if src in df.columns:
            renames[src] = v
    for v in _WET_VARS:
        src = f"{v}{wet_suffix}"
        if src in df.columns:
            renames[src] = v
    for v in _NEUTRAL_VARS:
        src = f"{v}{neutral_suffix}"
        if src in df.columns:
            renames[src] = v

    # Start with meta columns + renamed tail selections; drop everything else.
    keep_meta = [c for c in ("pyrome", "date", "year", "doy", "system:index") if c in df.columns]
    keep = keep_meta + list(renames.keys())
    out = df[keep].rename(columns=renames).copy()
    return out


def build_historic_erc_arrays(
    df: pd.DataFrame,
    pyrome_col: str = "pyrome",
    n_years: int = 15,
    out_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Build per-pyrome historic ERC arrays for FSPro ``HistoricERCValues``.

    Each array has shape ``(n_years, 214)`` — one row per year, 214 columns
    representing fire-season days 1–214 (day 1 = April 1, day 214 = Oct 31).
    Years are sorted chronologically.  Missing day/year combinations are
    filled with the column (day-of-season) median across available years.

    The pivot uses **day-of-season** (1–214 anchored to April 1 of each
    calendar year) rather than raw DOY.  This makes the result leap-year-safe:
    April 1 is always season day 1 regardless of whether the year is a leap
    year, so all years produce exactly 214 columns.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_gridmet_csv`.
    pyrome_col : str
        Column name identifying the spatial grouping unit (default
        ``"pyrome"``).
    n_years : int
        Expected number of years in the record (default 15 for 2008–2022).
        Used only for the warning message; does not truncate data.
    out_dir : str or Path, optional
        If provided, write ``pyrome_{id}_gridmet.json`` cache files here.

    Returns
    -------
    dict[str, np.ndarray]
        ``{pyrome_id: ndarray(n_years, 214)}`` with ERC values.
    """
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Compute day-of-season (1 = April 1, 214 = Oct 31) anchored per year.
    # This is leap-year-safe: raw DOY shifts by 1 in leap years, but
    # (date - April_1_of_that_year).days is always consistent.
    df2 = df.copy()
    april_1 = pd.to_datetime(df2["year"].astype(str) + "-04-01")
    df2["_dos"] = (df2["date"] - april_1).dt.days + 1

    # Keep only the 214-day fire season (April 1 – Oct 31)
    df2 = df2[(df2["_dos"] >= 1) & (df2["_dos"] <= _N_SEASON_DAYS)]

    season_days = list(range(1, _N_SEASON_DAYS + 1))  # always [1 … 214]

    result: dict[str, np.ndarray] = {}

    for pyrome_id, group in df2.groupby(pyrome_col):
        # Pivot: rows = year, columns = day-of-season (1–214)
        pivot = (
            group.pivot_table(index="year", columns="_dos", values="erc", aggfunc="mean")
            .reindex(columns=season_days)
        )
        pivot = pivot.sort_index()

        # Fill missing day/year cells with column (day-of-season) median
        pivot = pivot.fillna(pivot.median(axis=0))

        arr = pivot.values.astype(float)

        if arr.shape[0] != n_years:
            print(f"  [build_historic_erc_arrays] Pyrome {pyrome_id}: "
                  f"{arr.shape[0]} years (expected {n_years})")

        result[str(pyrome_id)] = arr

        if out_dir is not None:
            cache = {
                "pyrome_id": str(pyrome_id),
                "years": list(map(int, pivot.index)),
                "season_days": season_days,   # 1-based, leap-year-safe
                "NumERCYears": int(arr.shape[0]),
                "NumWxPerYear": int(arr.shape[1]),
                "HistoricERCValues": arr.tolist(),
            }
            _write_json(cache, out_dir / f"pyrome_{pyrome_id}_gridmet.json")

    print(f"  [build_historic_erc_arrays] {len(result)} pyromes → "
          f"arrays shape ({n_years}, {_N_SEASON_DAYS})")
    return result


def build_erc_stats(
    historic_dict: dict[str, np.ndarray],
) -> dict[str, dict]:
    """
    Compute per-DOY ERC statistics for FSPro ``AvgERCValues`` / ``StdDevERCValues``.

    Parameters
    ----------
    historic_dict : dict
        Output of :func:`build_historic_erc_arrays`.

    Returns
    -------
    dict[str, dict]
        ``{pyrome_id: {"avg": ndarray(214), "std": ndarray(214)}}``.
        Both arrays are rounded to 2 decimal places to match FSPro format.
    """
    stats: dict[str, dict] = {}
    for pyrome_id, arr in historic_dict.items():
        stats[str(pyrome_id)] = {
            "avg": np.round(np.nanmean(arr, axis=0), 2),
            "std": np.round(np.nanstd(arr, axis=0, ddof=1), 2),
        }
    return stats


def build_erc_classes(
    df: pd.DataFrame,
    pyrome_col: str = "pyrome",
    n_classes: int = 5,
    erc_col: str = "erc",
    fm100_col: str = "fm100",
    tmmx_col: str = "tmmx_f",
    rmin_col: str = "rmin",
    spotting: list[list[float]] | None = None,
    lat_deg: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Build the 5-row ERC class table for FSPro ``NumERCClasses`` block.

    Each row represents one ERC percentile class (highest ERC first) with
    associated fuel moisture values derived from GridMET data.

    Row format (10 columns):
    ``[lower, upper, fm1, fm10, fm100, fm_herb, fm_woody, spot_dist, spot_prob, spot2]``

    Fuel moisture derivation:
    - ``fm1``    : median NFDRS 1-hr FM (from tmmx_f + rmin via :func:`calc_1hr_fm`)
    - ``fm10``   : median NFDRS 10-hr FM (from tmmx_f + rmin via :func:`calc_10hr_fm`)
    - ``fm100``  : median GridMET fm100 (direct)
    - ``fm_herb``: GSI-based live FM via :func:`calc_herb_fm_gsi` when ``tmmn_f``,
      ``vpd_pa``, and ``lat_deg`` are available — replicates the NFDRS 2016 /
      FireFamilyPlus method; otherwise falls back to DOY-based :func:`calc_herb_fm`
    - ``fm_woody``: same GSI path via :func:`calc_woody_fm_gsi`, or DOY fallback

    Quintile ERC bins are computed per pyrome from the full fire-season record.
    Classes are ordered highest-to-lowest ERC (class 1 = most extreme).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_gridmet_csv`.
    pyrome_col : str
        Column identifying the spatial grouping unit.
    n_classes : int
        Number of ERC classes (default 5).
    erc_col, fm100_col, tmmx_col, rmin_col : str
        Column names in ``df``.
    spotting : list of list, optional
        Override default spotting parameters. Must be ``n_classes`` rows of
        ``[spot_dist_ft, spot_prob, spot2]``. Ordered highest-to-lowest ERC.
        Defaults to :data:`_DEFAULT_SPOTTING`.
    lat_deg : float, optional
        Site latitude in decimal degrees. Required for GSI-based live FM
        (enables :func:`calc_daylength`). When None, live FM falls back to
        DOY-based phenology.

    Returns
    -------
    dict[str, np.ndarray]
        ``{pyrome_id: ndarray(n_classes, 10)}``.
    """
    if spotting is None:
        spotting = _DEFAULT_SPOTTING
    if len(spotting) != n_classes:
        raise ValueError(f"len(spotting)={len(spotting)} must equal n_classes={n_classes}")

    result: dict[str, np.ndarray] = {}

    for pyrome_id, group in df.groupby(pyrome_col):
        erc = group[erc_col].dropna().values

        # Quintile bin edges (0%, 20%, 40%, 60%, 80%, 100%)
        quantiles = np.linspace(0, 100, n_classes + 1)
        edges = np.percentile(erc, quantiles)

        # Run the NFDRS78 lag once over this pyrome's full record so
        # per-bin medians draw from time-integrated FM10/FM100 rather than
        # from independent Fosberg/GridMET snapshots (which produced the
        # non-monotonic ordering documented in plan
        # `a-couple-things-first-synthetic-grove.md`).
        precip_col = "pr" if "pr" in group.columns else None
        ts = build_fm_timeseries(
            group,
            tmax_col=tmmx_col,
            rmin_col=rmin_col,
            precip_col=precip_col,
        )

        # Build one row per class (highest ERC class first)
        rows = []
        for i in range(n_classes - 1, -1, -1):
            lower = edges[i]
            upper = edges[i + 1]

            # Mask observations in this ERC bin
            mask = (ts[erc_col] >= lower) & (ts[erc_col] <= upper)
            sub = ts[mask]

            if len(sub) == 0:
                # Fall back to all observations if bin is empty
                sub = ts

            fm1 = float(np.nanmedian(sub["FM1"].values))
            fm10 = float(np.nanmedian(sub["FM10"].values))
            fm100 = float(np.nanmedian(sub["FM100"].values))
            if "tmmn_f" in sub.columns and lat_deg is not None:
                vpd = calc_vpd_pa(sub[tmmx_col].values, sub[rmin_col].values)
                dl  = calc_daylength(sub["doy"].values, lat_deg)
                gsi = float(np.nanmedian(calc_gsi(sub["tmmn_f"].values, vpd, dl)))
                fm_herb  = float(calc_herb_fm_gsi(gsi))
                fm_woody = float(calc_woody_fm_gsi(gsi))
            else:
                median_doy = float(np.nanmedian(sub["doy"].values)) if "doy" in sub.columns else 182.0
                fm_herb  = float(calc_herb_fm(median_doy))
                fm_woody = float(calc_woody_fm(median_doy))

            spot = spotting[n_classes - 1 - i]  # highest ERC → spotting[0]
            rows.append([
                round(lower, 1), round(upper, 1),
                round(fm1, 1), round(fm10, 1), round(fm100, 1),
                round(fm_herb, 1), round(fm_woody, 1),
                float(spot[0]), float(spot[1]), float(spot[2]),
            ])

        result[str(pyrome_id)] = np.array(rows)

    print(f"  [build_erc_classes] {len(result)} pyromes × {n_classes} ERC classes")
    return result


def build_current_erc_values(
    historic_dict: dict[str, np.ndarray],
    start_doy: int,
    n_days: int = 79,
) -> dict[str, np.ndarray]:
    """
    Build ``CurrentERCValues`` from the historic-median approach.

    Computes the column-wise median across all years for a consecutive
    ``n_days`` window starting at ``start_doy`` in the fire season.  The
    DOY index is relative to the fire season (DOY 1 = April 1).

    For scenario-based (non-operational) FSPro runs this provides a
    climatologically representative current-season ERC sequence without
    requiring a real-time GEE fetch.

    Parameters
    ----------
    historic_dict : dict
        Output of :func:`build_historic_erc_arrays` —
        ``{pyrome_id: ndarray(n_years, 214)}``.
    start_doy : int
        1-based fire-season DOY (1 = April 1) at which the current-year
        window begins.
    n_days : int
        Length of the current-year ERC sequence (default 79, matching the
        416 example file).

    Returns
    -------
    dict[str, np.ndarray]
        ``{pyrome_id: ndarray(n_days)}`` of median ERC values.

    Raises
    ------
    ValueError
        If ``start_doy + n_days - 1`` exceeds the fire season length (214).
    """
    end_idx = start_doy - 1 + n_days
    if end_idx > _N_SEASON_DAYS:
        raise ValueError(
            f"start_doy={start_doy} + n_days={n_days} - 1 = {end_idx} exceeds "
            f"fire season length {_N_SEASON_DAYS}"
        )

    result: dict[str, np.ndarray] = {}
    for pyrome_id, arr in historic_dict.items():
        window = arr[:, start_doy - 1 : start_doy - 1 + n_days]
        result[str(pyrome_id)] = np.round(np.nanmedian(window, axis=0)).astype(int)
    return result


def build_flammap_fuel_moistures(
    df: pd.DataFrame,
    pyrome_id: str | int,
    scenario_doy: int,
    erc_percentile: float = 0.90,
    pyrome_col: str = "pyrome",
    fm100_col: str = "fm100",
    tmmx_col: str = "tmmx_f",
    rmin_col: str = "rmin",
    doy_window: int = 14,
    lat_deg: float | None = None,
) -> dict[str, float]:
    """
    Derive a complete FlamMap ``FUEL_MOISTURES_DATA`` row from GridMET.

    Returns a single set of fuel moisture values representative of
    high-danger conditions (``erc_percentile``) for the given day-of-year,
    drawn from the full climatological record for the specified pyrome.
    All five FM values are derived from NFDRS equations consistent with
    FireFamilyPlus:

    - fm1, fm10  : NFDRS EMC-based dead FM at peak fire-hour conditions
    - fm100      : directly from GridMET (measured 100-hr dead fuel moisture)
    - fm_herb    : DOY-based curing (:func:`~fb_tools.weather.nfdrs.calc_herb_fm`)
    - fm_woody   : seasonal sinusoidal (:func:`~fb_tools.weather.nfdrs.calc_woody_fm`)

    The returned dict is ready to pass directly to
    :func:`~fb_tools.models.flammap.run_flammap_scenarios` as ``fm_params``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_gridmet_csv`.
    pyrome_id : str or int
        Pyrome to subset.
    scenario_doy : int
        Target day-of-year (1–366) for the scenario. FM values are derived
        from a ±``doy_window`` day climatological window around this DOY.
    erc_percentile : float
        ERC percentile threshold (0–1) used to filter high-danger days
        when computing median dead FM. Default 0.90 (90th percentile).
    pyrome_col : str
        Column identifying the pyrome. Default ``"pyrome"``.
    fm100_col, tmmx_col, rmin_col : str
        Column names in ``df``.
    doy_window : int
        Half-width of the DOY window (days) used to subset the record.
        Default 14 (±2 weeks).
    lat_deg : float, optional
        Site latitude in decimal degrees.  When provided and ``tmmn_f`` is
        present in ``df``, live FM is computed via the NFDRS GSI model
        (:func:`~fb_tools.weather.nfdrs.calc_gsi`) for higher accuracy.
        If ``None`` or ``tmmn_f`` is absent, falls back to the DOY proxy.

    Returns
    -------
    dict[str, float]
        Keys: ``FM_1hr``, ``FM_10hr``, ``FM_100hr``, ``FM_herb``, ``FM_woody``.

    Raises
    ------
    ValueError
        If no records are found for the given pyrome / DOY window.
    """
    sub = df[df[pyrome_col].astype(str) == str(pyrome_id)].copy()
    if sub.empty:
        raise ValueError(f"No GridMET records for pyrome {pyrome_id!r}")

    # Wrap-around DOY window (handles year boundary near Jan 1)
    lo = scenario_doy - doy_window
    hi = scenario_doy + doy_window
    if lo < 1:
        mask = (sub["doy"] >= lo + 366) | (sub["doy"] <= hi)
    elif hi > 366:
        mask = (sub["doy"] >= lo) | (sub["doy"] <= hi - 366)
    else:
        mask = (sub["doy"] >= lo) & (sub["doy"] <= hi)

    window = sub[mask]
    if window.empty:
        raise ValueError(
            f"No records in ±{doy_window}-day window around DOY {scenario_doy} "
            f"for pyrome {pyrome_id!r}"
        )

    # Filter to high-danger days within the window
    erc_thresh = window["erc"].quantile(erc_percentile)
    high_danger = window[window["erc"] >= erc_thresh]
    if high_danger.empty:
        high_danger = window

    # FlamMap enforces 2–300% for all dead FM timelag classes; clamp before
    # writing to cache so extreme p97 conditions (EMC can drop to ~1.6%) don't
    # produce invalid input files.
    _FM_MIN, _FM_MAX = 2.0, 300.0

    # Build lag-derived FM time series for the whole pyrome record so the
    # high-danger-day medians draw from NFDRS78 time-integrated FM10/FM100
    # (Bradshaw 1984) rather than independent snapshots.
    precip_col = "pr" if "pr" in sub.columns else None
    ts_full = build_fm_timeseries(
        sub,
        tmax_col=tmmx_col,
        rmin_col=rmin_col,
        precip_col=precip_col,
    )
    high_danger_dates = pd.Index(high_danger["date"])
    ts_hd = ts_full[ts_full["date"].isin(high_danger_dates)]
    if ts_hd.empty:
        ts_hd = ts_full

    fm1 = float(np.clip(np.nanmedian(ts_hd["FM1"].values), _FM_MIN, _FM_MAX))
    fm10 = float(np.clip(np.nanmedian(ts_hd["FM10"].values), _FM_MIN, _FM_MAX))
    fm100 = float(np.clip(np.nanmedian(ts_hd["FM100"].values), _FM_MIN, _FM_MAX))
    # Cross-check: GridMET's own NFDRS-lag 100-hr product on the same days.
    fm100_gridmet = float(np.clip(np.nanmedian(high_danger[fm100_col].values), _FM_MIN, _FM_MAX))

    # Use GSI-based live FM if tmmn_f is available; fall back to dead-FM proxy.
    if "tmmn_f" in high_danger.columns and lat_deg is not None:
        vpd = calc_vpd_pa(high_danger[tmmx_col].values, high_danger[rmin_col].values)
        dl = calc_daylength(high_danger["doy"].values, lat_deg)
        gsi = float(np.nanmedian(calc_gsi(high_danger["tmmn_f"].values, vpd, dl)))
        fm_herb = float(calc_herb_fm_gsi(gsi))
        fm_woody = float(calc_woody_fm_gsi(gsi))
    else:
        # Dead-FM proxy scales monotonically with drought stress.
        # DOY alone can't differentiate ERC percentile conditions at the same date.
        _herb_arr, _woody_arr = calc_live_fm_from_dead(
            fm100,
            herb_scale=_LIVE_FM_HERB_SCALE,
            woody_scale=_LIVE_FM_WOODY_SCALE,
            herb_floor=30.0,
            woody_floor=60.0,
        )
        fm_herb = float(_herb_arr)
        fm_woody = float(_woody_arr)

    return {
        "FM_1hr": round(fm1, 1),
        "FM_10hr": round(fm10, 1),
        "FM_100hr": round(fm100, 1),
        "FM_100hr_gridmet": round(fm100_gridmet, 1),
        "FM_herb": round(fm_herb, 1),
        "FM_woody": round(fm_woody, 1),
    }


def build_flammap_weather_data(
    df: pd.DataFrame,
    pyrome_id: str | int,
    start_date: str,
    end_date: str,
    elevation_ft: int,
    pyrome_col: str = "pyrome",
    precip_start_hour: int = 600,
    precip_end_hour: int = 1200,
) -> tuple[list[tuple], int]:
    """
    Build a FlamMap ``WEATHER_DATA`` block from GridMET daily records.

    Produces one row per day in [``start_date``, ``end_date``] for the given
    pyrome, ready to embed in a FlamMap conditioning-period input file.
    This is the preferred weather input mode when using GridMET (daily data),
    as it maps directly without requiring diurnal synthesis.

    Row format (12 values per day):
    ``Mth Day Pcp mTH xTH mT xT mH xH Elv PST PET``

    Where:
    - ``Pcp``  : precipitation in hundredths of inch (``pr`` mm → hundredths)
    - ``mTH``  : hour of min temp (default 0500)
    - ``xTH``  : hour of max temp (default 1400)
    - ``mT``   : daily min temperature °F (``tmmn_f``)
    - ``xT``   : daily max temperature °F (``tmmx_f``)
    - ``mH``   : morning (max) relative humidity % (``rmax``)
    - ``xH``   : afternoon (min) relative humidity % (``rmin``)
    - ``Elv``  : station elevation in feet
    - ``PST``  : precip start time (0 when no precip)
    - ``PET``  : precip end time (0 when no precip)

    **GEE notebook prerequisite**: ``tmmn``, ``rmax``, and ``pr`` columns must
    be present in the GridMET CSV export (add to GEE notebook and re-export).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_gridmet_csv` with ``tmmn_f``, ``rmax``, and
        ``pr`` columns present.
    pyrome_id : str or int
        Pyrome to subset.
    start_date : str
        Start date inclusive (``"YYYY-MM-DD"``).
    end_date : str
        End date inclusive (``"YYYY-MM-DD"``).
    elevation_ft : int
        Elevation of the representative weather location in feet.
    pyrome_col : str
        Column identifying the pyrome. Default ``"pyrome"``.
    precip_start_hour, precip_end_hour : int
        HHMM times assigned as precip start/end when daily precip > 0.
        Defaults 0600–1200.

    Returns
    -------
    tuple[list[tuple], int]
        ``(rows, n_records)`` where each row is a 12-element tuple and
        ``n_records`` equals ``len(rows)``.

    Raises
    ------
    KeyError
        If required columns (``tmmn_f``, ``rmax``, ``pr``) are absent — update
        the GEE GridMET export to include ``tmmn``, ``rmax``, and ``pr``.
    ValueError
        If no records are found for the given pyrome / date range.
    """
    for col in ("tmmn_f", "rmax", "pr"):
        if col not in df.columns:
            raise KeyError(
                f"Column {col!r} not found. Add tmmn/rmax/pr to the GEE GridMET "
                "export and re-run load_gridmet_csv()."
            )

    sub = df[df[pyrome_col].astype(str) == str(pyrome_id)].copy()
    if sub.empty:
        raise ValueError(f"No GridMET records for pyrome {pyrome_id!r}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    sub = sub[(sub["date"] >= start) & (sub["date"] <= end)].sort_values("date")

    if sub.empty:
        raise ValueError(
            f"No records for pyrome {pyrome_id!r} in {start_date} – {end_date}"
        )

    rows: list[tuple] = []
    for _, row in sub.iterrows():
        pcp_hundredths = int(round(float(row["pr"]) * 0.03937 * 100))
        pcp_hundredths = max(0, pcp_hundredths)

        pst = precip_start_hour if pcp_hundredths > 0 else 0
        pet = precip_end_hour if pcp_hundredths > 0 else 0

        rows.append((
            int(row["date"].month),
            int(row["date"].day),
            pcp_hundredths,
            500,                             # mTH: min-temp hour default 0500
            1400,                            # xTH: max-temp hour default 1400
            int(round(float(row["tmmn_f"]))),
            int(round(float(row["tmmx_f"]))),
            int(round(float(row["rmax"]))),  # mH: morning (max) humidity
            int(round(float(row["rmin"]))),  # xH: afternoon (min) humidity
            elevation_ft,
            pst,
            pet,
        ))

    return rows, len(rows)


def build_gridmet_wind_percentiles(
    df: pd.DataFrame,
    pyrome_col: str = "pyrome",
    percentiles: list[float] | None = None,
    erc_filter_pct: float | None = None,
) -> dict[str, dict]:
    """
    Compute fire-season wind speed percentiles from GridMET ``vs``/``th``.

    Provides a gridded alternative to HRRR for wind climatology when HRRR
    records are unavailable or when spatial coverage of the pyrome is
    preferred over point-based fire-occurrence sampling.  For FSPro, HRRR
    analysis (``build_pyrome_wind_cells``) remains the primary wind source.

    GridMET variables required (added via GEE export):
    - ``vs`` → ``ws_mph``  : daily mean wind speed (m/s, converted to mph)
    - ``th`` → ``wd_deg``  : daily wind direction (degrees, meteorological FROM)

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_gridmet_csv` with ``ws_mph`` and ``wd_deg``
        columns present.
    pyrome_col : str
        Column identifying the spatial grouping unit.
    percentiles : list of float, optional
        Wind speed percentiles to compute. Default ``[0.25, 0.50, 0.75, 0.90, 0.97]``.
    erc_filter_pct : float, optional
        If provided, restrict the wind sample to days at or above this ERC
        percentile (e.g., ``0.90`` = top-10% ERC days).  Useful for computing
        wind speed representative of high fire-danger conditions only.

    Returns
    -------
    dict[str, dict]
        ``{pyrome_id: {"ws_mph": {pct_key: value}, "wd_deg_mode": value}}``.
        ``wd_deg_mode`` is the circular-mean direction on the sampled days.

    Raises
    ------
    KeyError
        If ``ws_mph`` or ``wd_deg`` columns are absent.
    """
    for col in ("ws_mph", "wd_deg"):
        if col not in df.columns:
            raise KeyError(
                f"Column {col!r} not found. Add vs/th to the GEE GridMET export."
            )

    if percentiles is None:
        percentiles = [0.25, 0.50, 0.75, 0.90, 0.97]

    season = df[(df["doy"] >= 91) & (df["doy"] <= 304)].copy()
    result: dict[str, dict] = {}

    for pyrome_id, group in season.groupby(pyrome_col):
        sub = group.dropna(subset=["ws_mph", "wd_deg"])

        if erc_filter_pct is not None and "erc" in sub.columns:
            thresh = float(sub["erc"].quantile(erc_filter_pct))
            sub = sub[sub["erc"] >= thresh]
            if sub.empty:
                sub = group.dropna(subset=["ws_mph", "wd_deg"])

        ws = sub["ws_mph"].values
        wd = sub["wd_deg"].values

        ws_pcts = {
            f"p{int(round(p * 100))}": round(float(np.nanpercentile(ws, p * 100)), 1)
            for p in percentiles
        }
        # Circular mean direction
        wd_rad = np.radians(wd)
        wd_mode = float((np.degrees(np.arctan2(np.nanmean(np.sin(wd_rad)), np.nanmean(np.cos(wd_rad)))) + 360) % 360)

        result[str(pyrome_id)] = {
            "ws_mph": ws_pcts,
            "wd_deg_mean": round(wd_mode, 1),
            "n_days": int(len(sub)),
        }

    return result


def build_flammap_scenario_cache(
    df: pd.DataFrame,
    pyrome_col: str = "pyrome",
    percentiles: list[float] | None = None,
    out_dir: str | Path | None = None,
    fm100_col: str = "fm100",
    tmmx_col: str = "tmmx_f",
    rmin_col: str = "rmin",
    lat_deg: float | None = None,
    wind_direction: int = -2,
    wind_speed_source: str = "gridmet",
    wind_percentiles: dict[str, dict] | None = None,
    erc_band: float = 0.025,
    dead_fm_source: str = "gridmet",
    live_fm_source: str = "gridmet",
    rtma_daily_df: pd.DataFrame | None = None,
    rtma_pyrome_col: str = "pyrome_id",
    rtma_date_col: str = "date",
) -> dict[str, dict]:
    """
    Build per-pyrome FlamMap scenario caches at multiple ERC percentiles.

    **Percentile semantics (RAWS / FireFamilyPlus convention):** for each
    percentile ``p`` the scenario represents conditions on days *near* the
    p-th ERC quantile — not "days at or above".  Sampling is a narrow band
    ``p ± erc_band`` of the ERC distribution.  This preserves the spread
    between low-danger (pct25) and extreme (pct97) scenarios instead of
    compressing every scenario toward the dry tail.

    For each pyrome and percentile, fuel moistures are derived from the
    median of the sampled days, and wind speed is attached either from a
    pre-computed ``wind_percentiles`` dict (preferred, e.g., from HRRR) or
    from the same day-band of GridMET ``ws_mph``.

    Results are written to ``pyrome_{id}_flammap.json`` and are directly
    usable as ``fm_params`` in
    :func:`~fb_tools.models.flammap.run_flammap_scenarios`.

    Cache JSON structure::

        {
          "pyrome_id": "42",
          "percentiles": [0.25, 0.50, 0.75, 0.90, 0.97],
          "wind_direction": -2,
          "scenarios": {
            "p25": {
              "FM_1hr": 12.3, "FM_10hr": 13.1, "FM_100hr": 11.2,
              "FM_herb": 198.3, "FM_woody": 191.2,
              "WIND_SPEED": 8, "WIND_DIRECTION": -2,
              "erc_quantile_band": [0.225, 0.275],
              "erc_center": 22.1, "scenario_doy": 162
            },
            "p97": { ... }
          }
        }

    Percentile keys use the format ``p{int(p*100)}`` (e.g., ``"p97"``).

    **Wind speed** (``WIND_SPEED``):
    - If ``wind_percentiles`` is provided, the per-percentile value is taken
      from ``wind_percentiles[pyrome_id]["ws_mph"][key]`` (preferred; use HRRR
      via :func:`fb_tools.weather.hrrr.build_hrrr_wind_percentiles` to capture
      peak-hour extremes the GridMET daily mean smooths away).
    - Else if ``wind_speed_source="gridmet"`` and ``ws_mph`` is present,
      the median GridMET wind on the sampled band days is used.
    - Else ``WIND_SPEED`` is omitted (supply at run time).

    **Wind direction** (``WIND_DIRECTION``): controlled by ``wind_direction``.
    FlamMap convention: ``-2`` = downhill (default), ``-1`` = uphill, or an
    explicit azimuth (0–360).  For FSPro, wind is derived separately via HRRR
    wind cells.

    FM derivation:
    - ``FM_1hr``, ``FM_10hr`` : NFDRS dead FM at peak-fire-hour conditions
    - ``FM_100hr``            : median GridMET 100-hr dead FM
    - ``FM_herb``             : GSI-based if ``tmmn_f`` + ``lat_deg`` available,
                               otherwise DOY-based curing (NFDRS PMS 437)
    - ``FM_woody``            : same as FM_herb choice

    Fire season: April 1 – October 31 (DOY 91–304).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_gridmet_csv`.
    pyrome_col : str
        Column identifying the spatial grouping unit. Default ``"pyrome"``.
    percentiles : list of float, optional
        ERC percentiles in [0, 1].  Default ``[0.25, 0.50, 0.75, 0.90, 0.97]``.
        Each is the *center* of the sample band (see ``erc_band``).
    out_dir : str or Path, optional
        If provided, write ``pyrome_{id}_flammap.json`` files here.
    fm100_col, tmmx_col, rmin_col : str
        Column names in ``df``.
    lat_deg : float, optional
        Site latitude (decimal degrees). When provided with ``tmmn_f``
        present, live FM uses the NFDRS GSI model.
    wind_direction : int
        Wind direction written to every scenario entry.
    wind_speed_source : str
        ``"gridmet"`` (default): median GridMET ``ws_mph`` on the sampled band.
        ``"none"``: omit ``WIND_SPEED`` from cache.
        Ignored when ``wind_percentiles`` is supplied.
    wind_percentiles : dict, optional
        Pre-computed per-pyrome wind speed percentiles with shape
        ``{pyrome_id: {"ws_mph": {"p25": v, ..., "p97": v}, ...}}``.  Output
        of :func:`fb_tools.weather.hrrr.build_hrrr_wind_percentiles` or
        :func:`build_gridmet_wind_percentiles`.  Keys must match the
        ``p{int(p*100)}`` form of ``percentiles``.  When present, overrides
        ``wind_speed_source``.
    erc_band : float
        Half-width of the ERC quantile band used to sample representative
        days for each percentile.  Default 0.025 (±2.5%, i.e. ~5% of the
        fire-season record per scenario).  For the top percentile the band
        is clipped to the upper tail (``[p - 2*band, 1.0]``) to avoid an
        empty sample on the extreme end; same trick at the low end.
    dead_fm_source : {"gridmet", "rtma"}
        Source for ``FM_1hr``, ``FM_10hr``, ``FM_100hr``.  Default
        ``"gridmet"`` (NFDRS78 lag against daily peak-hour EMC from
        :func:`build_fm_timeseries`).  ``"rtma"`` overrides dead FM with
        the median of RTMA daily-peak-hour FM rows on the same
        ERC-band days — produced via :func:`~fb_tools.weather.rtma.build_rtma_dead_fm`
        + :func:`~fb_tools.weather.rtma.collapse_to_peak_hour`.  ERC
        membership and scenario-day selection always remain GridMET-driven.
    live_fm_source : {"gridmet", "rtma"}
        Source for ``FM_herb`` and ``FM_woody``.  Default ``"gridmet"``
        — derives live FM from dead FM via
        :func:`~fb_tools.weather.nfdrs.calc_live_fm_from_dead` (preserves
        monotonic ordering with ERC percentile).  ``"rtma"`` uses median
        GSI-based herb/woody FM from RTMA daily rows on the same
        ERC-band days — see
        :func:`~fb_tools.weather.rtma.build_rtma_live_fm`.  Note: GSI
        threshold calibration (Jolly 2005) is the dominant limitation for
        semi-arid pyromes, not the input data; the RTMA path is unlikely
        to differ materially from GridMET-derived dead-FM scaling.
    rtma_daily_df : pd.DataFrame, optional
        Daily peak-hour RTMA frame required when either ``dead_fm_source``
        or ``live_fm_source`` is ``"rtma"``.  Expected to contain
        ``rtma_pyrome_col``, ``rtma_date_col``, plus the FM columns
        referenced by the chosen sources (``FM1, FM10, FM100`` for dead;
        ``FM_herb, FM_woody`` for live).
    rtma_pyrome_col, rtma_date_col : str
        Column names in ``rtma_daily_df``.

    Returns
    -------
    dict[str, dict]
        ``{pyrome_id: {"percentiles": [...], "scenarios": {key: scenario_dict}}}``.
    """
    if dead_fm_source not in ("gridmet", "rtma"):
        raise ValueError(f"dead_fm_source must be 'gridmet' or 'rtma', got {dead_fm_source!r}")
    if live_fm_source not in ("gridmet", "rtma"):
        raise ValueError(f"live_fm_source must be 'gridmet' or 'rtma', got {live_fm_source!r}")

    use_rtma = dead_fm_source == "rtma" or live_fm_source == "rtma"
    if use_rtma and rtma_daily_df is None:
        raise ValueError(
            "rtma_daily_df is required when dead_fm_source or live_fm_source is 'rtma'"
        )

    if use_rtma:
        rtma_idx = rtma_daily_df.copy()
        rtma_idx[rtma_date_col] = pd.to_datetime(rtma_idx[rtma_date_col]).dt.normalize()
        rtma_by_pyrome = {
            str(pid): grp.set_index(rtma_date_col)
            for pid, grp in rtma_idx.groupby(rtma_pyrome_col)
        }
    else:
        rtma_by_pyrome = {}
    if percentiles is None:
        percentiles = [0.25, 0.50, 0.75, 0.90, 0.97]

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    use_external_wind = wind_percentiles is not None
    has_gridmet_wind = (
        "ws_mph" in df.columns
        and wind_speed_source == "gridmet"
        and not use_external_wind
    )

    # Restrict to fire season (DOY 91–304: April 1 – October 31)
    season = df[(df["doy"] >= 91) & (df["doy"] <= 304)].copy()

    result: dict[str, dict] = {}

    for pyrome_id, group in season.groupby(pyrome_col):
        erc = group["erc"].dropna()
        scenarios: dict[str, dict] = {}
        pid_str = str(pyrome_id)

        # NFDRS78 lag once over this pyrome's full fire-season record. Day
        # selection per percentile (ERC quantile band) is unchanged; only
        # the FM derivation switches from Fosberg snapshots / native GridMET
        # fm100 to time-integrated lag values.
        precip_col = "pr" if "pr" in group.columns else None
        ts = build_fm_timeseries(
            group,
            tmax_col=tmmx_col,
            rmin_col=rmin_col,
            precip_col=precip_col,
        )

        running_max_doy: float = 0.0

        for p in percentiles:
            key = f"p{int(round(p * 100))}"

            # Quantile band around p, clipped at the tails so extreme
            # percentiles still have a non-empty sample.
            lo_q = p - erc_band
            hi_q = p + erc_band
            if lo_q < 0.0:
                lo_q, hi_q = 0.0, min(1.0, 2 * erc_band)
            elif hi_q > 1.0:
                lo_q, hi_q = max(0.0, 1.0 - 2 * erc_band), 1.0

            lo_erc = float(erc.quantile(lo_q))
            hi_erc = float(erc.quantile(hi_q))

            sample = ts[(ts["erc"] >= lo_erc) & (ts["erc"] <= hi_erc)]
            if sample.empty:
                # Fallback: single nearest-neighbor day at the target quantile
                target = float(erc.quantile(p))
                sample = ts.iloc[[(ts["erc"] - target).abs().idxmin()]]

            fm1 = float(np.nanmedian(sample["FM1"].values))
            fm10 = float(np.nanmedian(sample["FM10"].values))
            fm100 = float(np.nanmedian(sample["FM100"].values))
            fm100_gridmet = float(np.nanmedian(sample[fm100_col].values))
            erc_center = float(np.nanmedian(sample["erc"].values))

            # RTMA overrides — keep ERC selection (GridMET) but swap in
            # hourly-derived FM medians on the same band days.
            fm_herb_rtma: float | None = None
            fm_woody_rtma: float | None = None
            if use_rtma:
                pid_rtma = rtma_by_pyrome.get(pid_str)
                if pid_rtma is not None:
                    sample_dates = pd.to_datetime(sample["date"]).dt.normalize().unique()
                    rtma_sample = pid_rtma.reindex(sample_dates).dropna(how="all")
                    if not rtma_sample.empty:
                        if dead_fm_source == "rtma":
                            if "FM1" in rtma_sample.columns:
                                fm1 = float(np.nanmedian(rtma_sample["FM1"].values))
                            if "FM10" in rtma_sample.columns:
                                fm10 = float(np.nanmedian(rtma_sample["FM10"].values))
                            if "FM100" in rtma_sample.columns:
                                fm100 = float(np.nanmedian(rtma_sample["FM100"].values))
                        if live_fm_source == "rtma":
                            if "FM_herb" in rtma_sample.columns:
                                fm_herb_rtma = float(np.nanmedian(rtma_sample["FM_herb"].values))
                            if "FM_woody" in rtma_sample.columns:
                                fm_woody_rtma = float(np.nanmedian(rtma_sample["FM_woody"].values))

            # Median DOY of the sampled band (core season only) — stored for
            # reference. Running max keeps scenario_doy non-decreasing across
            # percentiles (monsoon years can invert the order).
            core_doys = sample.loc[sample["doy"] >= 152, "doy"].values
            median_doy = float(
                np.nanmedian(core_doys) if len(core_doys) > 0
                else np.nanmedian(sample["doy"].values)
            )
            effective_doy = max(median_doy, running_max_doy)
            running_max_doy = effective_doy

            # Live FM. Default path: dead-FM scaling — monotonically ordered
            # with ERC by construction. DOY-based phenology fails here
            # because all ERC percentile bands tend to cluster at the same
            # calendar date (e.g. late-August in CO), producing identical
            # live FM values across scenarios.
            _herb_arr, _woody_arr = calc_live_fm_from_dead(
                fm100,
                herb_scale=_LIVE_FM_HERB_SCALE,
                woody_scale=_LIVE_FM_WOODY_SCALE,
                herb_floor=30.0,
                woody_floor=60.0,
            )
            fm_herb = float(_herb_arr) if fm_herb_rtma is None else fm_herb_rtma
            fm_woody = float(_woody_arr) if fm_woody_rtma is None else fm_woody_rtma

            entry: dict = {
                "FM_1hr": round(fm1, 1),
                "FM_10hr": round(fm10, 1),
                "FM_100hr": round(fm100, 1),
                "FM_100hr_gridmet": round(fm100_gridmet, 1),
                "FM_herb": round(fm_herb, 1),
                "FM_woody": round(fm_woody, 1),
                "WIND_DIRECTION": wind_direction,
                "erc_quantile_band": [round(lo_q, 3), round(hi_q, 3)],
                "erc_center": round(erc_center, 1),
                "scenario_doy": int(round(effective_doy)),
                "n_sample_days": int(len(sample)),
            }

            if use_external_wind:
                ws_lookup = (
                    wind_percentiles.get(pid_str, {})
                    .get("ws_mph", {})
                    .get(key)
                )
                entry["WIND_SPEED"] = round(float(ws_lookup), 1) if ws_lookup is not None else None
            elif has_gridmet_wind:
                ws = sample["ws_mph"].dropna()
                entry["WIND_SPEED"] = round(float(np.nanmedian(ws)), 1) if len(ws) > 0 else None

            scenarios[key] = entry

        if use_external_wind:
            ws_source_label = "external"
        elif has_gridmet_wind:
            ws_source_label = "gridmet"
        else:
            ws_source_label = "none"

        cache = {
            "pyrome_id": pid_str,
            "percentiles": percentiles,
            "erc_band": erc_band,
            "wind_direction": wind_direction,
            "wind_speed_source": ws_source_label,
            "dead_fm_source": dead_fm_source,
            "live_fm_source": live_fm_source,
            "scenarios": scenarios,
        }
        result[pid_str] = cache

        if out_dir is not None:
            _write_json(cache, out_dir / f"pyrome_{pyrome_id}_flammap.json")

    print(f"  [build_flammap_scenario_cache] {len(result)} pyromes × {len(percentiles)} scenarios "
          f"(±{erc_band:.3f} ERC band, wind={ws_source_label}, "
          f"dead_fm={dead_fm_source}, live_fm={live_fm_source})")
    return result


def load_flammap_scenario_cache(
    pyrome_id: str | int,
    cache_dir: str | Path,
) -> dict:
    """
    Load a cached FlamMap scenario FM table from JSON.

    Parameters
    ----------
    pyrome_id : str or int
        Pyrome identifier matching the ``pyrome_{id}_flammap.json`` filename.
    cache_dir : str or Path
        Directory containing the JSON cache files.

    Returns
    -------
    dict
        Full cache dict with keys ``pyrome_id``, ``percentiles``, and
        ``scenarios``.  Each scenario entry has keys ``FM_1hr``, ``FM_10hr``,
        ``FM_100hr``, ``FM_herb``, ``FM_woody``, ``erc_threshold``,
        ``scenario_doy``.

    Raises
    ------
    FileNotFoundError
        If no cache file exists for this pyrome.

    Examples
    --------
    >>> cache = load_flammap_scenario_cache(42, "cache/")
    >>> fm_params = cache["scenarios"]["p97"]
    >>> fm_params["FM_herb"]  # live herbaceous FM at 97th percentile ERC
    """
    path = Path(cache_dir) / f"pyrome_{pyrome_id}_flammap.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No FlamMap scenario cache for '{pyrome_id}' in {cache_dir}.\n"
            "Run build_flammap_scenario_cache() first."
        )
    with open(path) as f:
        return json.load(f)


def load_gridmet_pyrome_cache(
    pyrome_id: str | int,
    cache_dir: str | Path,
    return_meta: bool = False,
) -> np.ndarray | dict:
    """
    Load cached GridMET historic ERC array from JSON.

    Parameters
    ----------
    pyrome_id : str or int
        Pyrome identifier matching the ``pyrome_{id}_gridmet.json`` filename.
    cache_dir : str or Path
        Directory containing the JSON cache files.
    return_meta : bool
        If False (default), return only the ``HistoricERCValues`` array.
        If True, return the full metadata dictionary.

    Returns
    -------
    np.ndarray or dict
        Shape ``(NumERCYears, 214)`` array, or full dict if ``return_meta=True``.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    """
    cache_path = Path(cache_dir) / f"pyrome_{pyrome_id}_gridmet.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"GridMET cache not found: {cache_path}")
    with open(cache_path) as f:
        meta = json.load(f)
    if return_meta:
        return meta
    return np.array(meta["HistoricERCValues"])


def save_erc_classes_to_cache(
    gridmet_csv: str | Path,
    cache_dir: str | Path,
    pyrome_col: str = "pyrome",
    n_classes: int = 5,
) -> list[str]:
    """
    Enrich existing ``pyrome_{id}_gridmet.json`` cache files with ERC classes.

    Reads the GEE-exported GridMET CSV, builds the 5-row ERC class table for
    each pyrome via :func:`build_erc_classes`, and writes the result into the
    ``"ERCClasses"`` key of each matching cache file in *cache_dir*.  Only
    cache files that already exist (written by :func:`build_historic_erc_arrays`)
    are updated; no new files are created.

    Call this once after running ``build_historic_erc_arrays()`` to make ERC
    classes available to :func:`~fb_tools.models.container.prepare_container_fspro`
    without needing to pass the raw CSV on every run.

    Parameters
    ----------
    gridmet_csv : str or Path
        Path to the GEE-exported GridMET fire-season CSV
        (input to :func:`load_gridmet_csv`).
    cache_dir : str or Path
        Directory containing ``pyrome_{id}_gridmet.json`` files to update.
    pyrome_col : str
        Column in the CSV identifying the pyrome (default ``"pyrome"``).
    n_classes : int
        Number of ERC quantile classes (default 5).

    Returns
    -------
    list[str]
        Pyrome IDs whose cache files were successfully updated.

    Examples
    --------
    >>> from fb_tools import save_erc_classes_to_cache
    >>> save_erc_classes_to_cache(
    ...     "data/weather/gridmet_clima_co_pyromes.csv",
    ...     "data/weather/pyrome_erc/",
    ... )
    """
    cache_dir = Path(cache_dir)
    df = load_gridmet_csv(gridmet_csv)
    classes_dict = build_erc_classes(df, pyrome_col=pyrome_col, n_classes=n_classes)

    updated: list[str] = []
    for pid, classes_arr in classes_dict.items():
        cache_path = cache_dir / f"pyrome_{pid}_gridmet.json"
        if not cache_path.exists():
            print(f"  [save_erc_classes_to_cache] Skipping pyrome {pid} — "
                  f"cache file not found: {cache_path.name}")
            continue
        with open(cache_path) as f:
            meta = json.load(f)
        meta["ERCClasses"] = classes_arr.tolist()
        with open(cache_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [save_erc_classes_to_cache] Updated {cache_path.name}")
        updated.append(pid)

    print(f"  [save_erc_classes_to_cache] Done — {len(updated)} cache files updated.")
    return updated


# ── Internal helpers ───────────────────────────────────────────────────────────

def _write_json(data: dict, path: Path) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [gridmet] Wrote {path.name}")
