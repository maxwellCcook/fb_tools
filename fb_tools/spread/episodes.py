"""
Weather episodes for pyretechnics spread runs.

An *episode* is a real historical window (e.g. 21 days from 2020-08-13) chosen
at a named ERC severity. Nothing is scaled or offset: each episode is nameable
and keeps its diurnal cycle and day-to-day autocorrelation. The hourly frame is
built from the cached pyrome-mean RTMA record (T/RH -> NFDRS dead FM; u/v wind)
and converted to the units pyretechnics documents.

Engine-free: nothing here imports pyretechnics, so it runs and tests in the
``fb_tools`` env.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..weather.gridmet import load_gridmet_pyrome_cache
from ..weather.hrrr import longest_true_run, uv_to_speed_dir
from ..weather.nfdrs import calc_live_fm_from_dead
from ..weather.rtma import build_rtma_dead_fm, load_rtma_csv

SEASON_START = (4, 1)          # fire-season day 1 = April 1 (214-day season)
SEASON_DAYS = 214

#: Cube name -> hourly-frame column. Every value is already in engine units.
WEATHER_CUBES = {
    "temperature":                   "tmp_c",
    "wind_speed_10m":                "ws_kmh",
    "upwind_direction":              "wd_from_deg",
    "fuel_moisture_dead_1hr":        "fm1",
    "fuel_moisture_dead_10hr":       "fm10",
    "fuel_moisture_dead_100hr":      "fm100",
    "fuel_moisture_live_herbaceous": "fm_herb",
    "fuel_moisture_live_woody":      "fm_woody",
}


def season_day_to_date(year, season_day):
    """Season day (1 = April 1) -> calendar date."""
    return date(year, *SEASON_START) + timedelta(days=int(season_day) - 1)


def _longest_gap_h(rtma_df, start_date, n_days):
    """Longest run of missing hours in ``rtma_df`` over ``n_days`` from ``start_date``."""
    idx = pd.date_range(pd.Timestamp(start_date), periods=n_days * 24, freq="h")
    return longest_true_run(~idx.isin(rtma_df["datetime_utc"]))


def select_episodes(pyrome_id, erc_cache_dir, *, severity="p97", n=5, duration_days=21,
                    years=None, rtma_df=None, max_gap_h=3, seed=0, dates=None, rank="random"):
    """
    Choose historical windows to burn under.

    The **pool** is either every season day of the pyrome's GridMET ERC
    climatology that meets ``severity`` (default), or explicit ``dates`` —
    the discovery dates of large fires (:func:`dates_from_fod`), or a named
    event such as an ignition date. The **rank** is how ``n`` windows are
    picked from the pool: a seeded random draw, or the top ``n`` by day-1 or
    window-mean ERC. Windows never overlap.

    Parameters
    ----------
    pyrome_id : int or str
    erc_cache_dir : Path
        Directory of ``pyrome_<id>_gridmet.json`` (``data/weather/pyrome_erc``).
    severity : str, float or None
        ``"p97"`` style label or a percentile in (0, 100): day-1 ERC must reach
        this percentile of every (year, season-day) value in the climatology.
        ``None`` applies no ERC filter (a named event is kept whatever its ERC).
    n : int or None
        Windows to return; ``None`` keeps every non-overlapping pool window.
    duration_days : int
    years : iterable of int, optional
        Restrict the climatology pool to these start years (ignored for ``dates``).
    rtma_df : pandas.DataFrame, optional
        Hourly record from :func:`load_rtma_hourly`. Windows with a run of more
        than ``max_gap_h`` missing hours are dropped (also windows that run past
        the record, e.g. past October 31).
    max_gap_h : int
    seed : int
        For ``rank="random"``.
    dates : iterable of date-like, optional
        Explicit start dates. Dates outside the climatology's years or the
        April–October season, and dates whose window would run past October 31,
        are dropped and counted.
    rank : {"random", "erc_day1", "erc_mean"}
        Random draw, or highest day-1 / window-mean ERC first.

    Returns
    -------
    pandas.DataFrame
        One row per episode: ``episode_id, year, season_day, start_date,
        end_date, erc_day1, erc_mean, erc_pct, erc_threshold, source``
        (``source`` is ``climatology`` or ``dates``). Sorted by start date.
    """
    meta = load_gridmet_pyrome_cache(pyrome_id, erc_cache_dir, return_meta=True)
    erc = np.asarray(meta["HistoricERCValues"], dtype=float)
    yrs = np.asarray(meta["years"], dtype=int)
    threshold = None if severity is None else float(np.nanpercentile(erc, float(str(severity).lstrip("pP"))))
    finite = erc[np.isfinite(erc)]
    last_start = SEASON_DAYS - duration_days + 1
    dropped = {}

    if dates is None:
        keep = np.ones(len(yrs), dtype=bool) if years is None else np.isin(yrs, list(years))
        cands = [(y, d) for yi, y in enumerate(yrs) if keep[yi]
                 for d in range(1, last_start + 1) if threshold is None or erc[yi, d - 1] >= threshold]
        source = "climatology"
    else:
        cands = []
        for ts in pd.to_datetime(pd.Series(list(dates))).dropna().dt.normalize().drop_duplicates().sort_values():
            y, d = ts.year, (ts - pd.Timestamp(year=ts.year, month=4, day=1)).days + 1
            if y not in yrs or not 1 <= d <= SEASON_DAYS:
                dropped["outside the ERC climatology"] = dropped.get("outside the ERC climatology", 0) + 1
            elif d > last_start:
                dropped["window runs past the season"] = dropped.get("window runs past the season", 0) + 1
            elif threshold is not None and erc[int(np.where(yrs == y)[0][0]), d - 1] < threshold:
                dropped[f"below {severity}"] = dropped.get(f"below {severity}", 0) + 1
            else:
                cands.append((y, d))
        source = "dates"
    n_pool = len(cands)
    if rtma_df is not None:
        cands = [(y, d) for y, d in cands
                 if _longest_gap_h(rtma_df, season_day_to_date(y, d), duration_days + 1) <= max_gap_h]
        if n_pool - len(cands):
            dropped["RTMA gap or record end"] = n_pool - len(cands)
    print(f"[select_episodes] pool: {n_pool} {source} window(s)" + (f" at {severity}" if threshold is not None else "")
          + (f"; dropped {dropped}" if dropped else "") + f"; rank = {rank}")
    if not cands:
        raise ValueError(f"No candidate {duration_days}-day window (severity {severity}, years {years}, dates {dates is not None})")

    def _stats(y, d):
        yi = int(np.where(yrs == y)[0][0])
        win = erc[yi, d - 1:d - 1 + duration_days]
        return float(erc[yi, d - 1]), float(np.nanmean(win))

    if rank == "random":
        order = np.random.default_rng(seed).permutation(len(cands))
    elif rank in ("erc_day1", "erc_mean"):
        key = [_stats(y, d)[0 if rank == "erc_day1" else 1] for y, d in cands]
        order = np.argsort(-np.asarray(key), kind="stable")
    else:
        raise ValueError(f"rank must be 'random', 'erc_day1' or 'erc_mean', not {rank!r}")
    chosen = []
    for k in order:
        y, d = cands[k]
        if all(y != cy or abs(d - cd) >= duration_days for cy, cd in chosen):
            chosen.append((y, d))
        if n is not None and len(chosen) == n:
            break
    if n is not None and len(chosen) < n:
        raise ValueError(f"Only {len(chosen)} non-overlapping {duration_days}-day "
                         f"episodes available (severity {severity}); asked for {n}")

    rows = []
    for y, d in sorted(chosen):
        day1, mean = _stats(y, d)
        start = pd.Timestamp(season_day_to_date(y, d))
        rows.append({"year": y, "season_day": d,
                     "start_date": start, "end_date": start + pd.Timedelta(days=duration_days - 1),
                     "erc_day1": round(day1, 1), "erc_mean": round(mean, 1),
                     "erc_pct": round(float((finite <= day1).mean() * 100), 1),
                     "erc_threshold": None if threshold is None else round(threshold, 1),
                     "source": source})
    df = pd.DataFrame(rows)
    df.insert(0, "episode_id", [f"ep_{i:03d}" for i in range(len(df))])
    return df


FAST_FIRE_GROWTH_HA = 1620.0     # Balch et al. (2025, Science): a "fast fire" grows >= 1,620 ha in one day
#: Fast-fires lookup shipped with the package (FIRED v2025var, CONUS, Oct 2000 – Feb 2026). Rebuild with
#: :func:`build_fast_fires_table` on a new FIRED release and replace this file.
FAST_FIRES_CSV = Path(__file__).resolve().parents[1] / "data" / "fast_fires_fired_v2025var.csv"


def build_fast_fires_table(fired_events_fp, out_csv, *, min_growth_ha=FAST_FIRE_GROWTH_HA):
    """
    Lookup table of FIRED events whose maximum single-day growth reaches the
    fast-fire threshold. Rebuild it when a new FIRED release lands.

    Parameters
    ----------
    fired_events_fp : Path
        FIRED *events* GeoPackage (``mx_grw_km2``, ``mx_grw_dte``, ``ig_date``, ...).
    out_csv : Path
    min_growth_ha : float

    Returns
    -------
    pandas.DataFrame
        One row per event: ``fired_id, ig_date, last_date, mx_grw_dte, mx_grw_ha,
        tot_ar_ha, event_dur, lc_name, lon, lat`` (event ignition centroid, WGS84).
    """
    import pyogrio
    cols = ["id", "ig_date", "last_date", "mx_grw_dte", "mx_grw_km2", "tot_ar_km2", "event_dur", "lc_name"]
    g = pyogrio.read_dataframe(fired_events_fp, where=f"mx_grw_km2 >= {min_growth_ha / 100.0}", columns=cols)
    c = g.geometry.centroid.to_crs(4326)
    out = pd.DataFrame({"fired_id": g["id"].values, "ig_date": pd.to_datetime(g["ig_date"]).dt.date,
                        "last_date": pd.to_datetime(g["last_date"]).dt.date,
                        "mx_grw_dte": pd.to_datetime(g["mx_grw_dte"]).dt.date,
                        "mx_grw_ha": (g["mx_grw_km2"] * 100).round(1).values,
                        "tot_ar_ha": (g["tot_ar_km2"] * 100).round(1).values,
                        "event_dur": g["event_dur"].values, "lc_name": g["lc_name"].values,
                        "lon": c.x.round(4).values, "lat": c.y.round(4).values}).sort_values("mx_grw_dte")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  [build_fast_fires_table] {len(out)} events with max daily growth >= {min_growth_ha:,.0f} ha "
          f"from {Path(fired_events_fp).name} -> {out_csv}")
    return out


def dates_from_fast_fires(table=FAST_FIRES_CSV, *, geom=None, bbox=None, days_before=3, min_growth_ha=FAST_FIRE_GROWTH_HA):
    """
    Episode start dates that put a fast fire's biggest day inside the window.

    Parameters
    ----------
    table : DataFrame or Path
        From :func:`build_fast_fires_table` (or its CSV); default the bundled ``FAST_FIRES_CSV``.
    geom : shapely geometry, GeoSeries or GeoDataFrame, optional
        Keep events whose ignition centroid falls inside (any CRS).
    bbox : tuple, optional
        ``(lon_min, lat_min, lon_max, lat_max)`` alternative to ``geom``.
    days_before : int
        The window starts this many days before the maximum-growth day, so
        with ``duration_days = days_before + 1 + days_after`` in
        :func:`select_episodes` the run-up, the blow-up day and its aftermath
        are all simulated.
    min_growth_ha : float
        Re-threshold the table (it is built at the fast-fire threshold).

    Returns
    -------
    pandas.DataFrame
        One row per event: ``date`` (window start), ``fired_id, mx_grw_dte,
        mx_grw_ha, tot_ar_ha``. Pass ``dates=pool.date``.
    """
    import geopandas as gpd
    t = pd.read_csv(table) if not isinstance(table, pd.DataFrame) else table.copy()
    t = t[t.mx_grw_ha >= min_growth_ha]
    pts = gpd.GeoSeries(gpd.points_from_xy(t.lon, t.lat), index=t.index, crs=4326)
    if bbox is not None:
        t = t.loc[pts.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].index]
    elif geom is not None:
        if isinstance(geom, (gpd.GeoSeries, gpd.GeoDataFrame)):
            geom = geom.to_crs(4326).union_all()
        t = t[pts.within(geom).values]
    t = t.assign(mx_grw_dte=pd.to_datetime(t.mx_grw_dte))
    t["date"] = t.mx_grw_dte - pd.Timedelta(days=days_before)
    print(f"  [dates_from_fast_fires] {len(t)} fast fire(s) (>= {min_growth_ha:,.0f} ha in a day) in the area; "
          f"windows start {days_before} d before the biggest day")
    return t[["date", "fired_id", "mx_grw_dte", "mx_grw_ha", "tot_ar_ha"]].sort_values("date").reset_index(drop=True)


def dates_from_fod(fod_gdf, *, min_class="D"):
    """
    Discovery dates of large fires, as an episode pool for :func:`select_episodes`.

    Parameters
    ----------
    fod_gdf : GeoDataFrame
        FPA-FOD records (``DISCOVERY_DATE``, ``FIRE_SIZE_CLASS``, ``FIRE_SIZE``,
        ``FIRE_NAME``), already restricted to the area of interest.
    min_class : str
        Smallest size class kept (``"D"`` = 100 ac+).

    Returns
    -------
    pandas.DataFrame
        One row per date: ``date, n_fires, largest_ac, fire_name`` (the largest fire
        that day). Pass ``dates=pool.date``.
    """
    f = fod_gdf[fod_gdf["FIRE_SIZE_CLASS"] >= min_class].copy()
    f["date"] = pd.to_datetime(f["DISCOVERY_DATE"]).dt.normalize()
    f = f.sort_values("FIRE_SIZE", ascending=False)
    g = f.groupby("date").agg(n_fires=("FIRE_SIZE", "size"), largest_ac=("FIRE_SIZE", "max"),
                             fire_name=("FIRE_NAME", "first")).reset_index().sort_values("date")
    print(f"  [dates_from_fod] {len(f)} fires of class >= {min_class} on {len(g)} distinct dates")
    return g.reset_index(drop=True)


def _year_files(directory, prefix, years):
    """``<prefix>_*_<year>.csv`` and ``<prefix>_*_<year>_<mm>.csv`` under ``directory``, sorted."""
    directory = Path(directory)
    years = sorted({int(y) for y in years})
    files = []
    for y in years:
        files += directory.glob(f"{prefix}_*_{y}.csv")
        files += directory.glob(f"{prefix}_*_{y}_[0-1][0-9].csv")
    if not files:
        raise FileNotFoundError(f"No {prefix}_*_<year>[_<mm>].csv for {years} in {directory}")
    return sorted(set(files))


def load_rtma_hourly(rtma_dir, pyrome_id, years, *, wind_dir=None):
    """
    Load the cached hourly RTMA record for one pyrome with dead FM spun up.

    Parameters
    ----------
    rtma_dir : Path
        Directory of ``rtma_hourly_*_<year>.csv`` or ``rtma_hourly_*_<year>_<mm>.csv``
        (the ``00b`` per-month export). Both spellings load; they may be mixed.
    pyrome_id : int
    years : iterable of int
    wind_dir : Path, optional
        Directory of ``rtma_wind_*_<year>[_<mm>].csv`` (``u_ms, v_ms, ws_ms``) from
        the wind export in ``00b_RTMA-EMC.ipynb``. Merged on time when given.

    Returns
    -------
    pandas.DataFrame
        Hourly rows for the pyrome with ``FM1, FM10, FM100`` appended by
        :func:`~fb_tools.weather.rtma.build_rtma_dead_fm`. The lag is run over
        the full record so every episode window is spun up.
    """
    files = _year_files(rtma_dir, "rtma_hourly", years)
    df = load_rtma_csv(files)
    df = df[df["pyrome_id"].astype(int) == int(pyrome_id)].copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = build_rtma_dead_fm(df)

    if wind_dir is not None:
        parts = [pd.read_csv(p) for p in _year_files(wind_dir, "rtma_wind", years)]
        w = pd.concat([x for x in parts if len(x)], ignore_index=True)   # header-only months poison dtypes
        w = w[w["pyrome_id"].astype(int) == int(pyrome_id)].copy()
        w["datetime_utc"] = pd.to_datetime(w["datetime_utc"])
        w[["u_ms", "v_ms", "ws_ms"]] = w[["u_ms", "v_ms", "ws_ms"]].astype(float)
        w = w.drop_duplicates("datetime_utc")
        df = df.merge(w[["datetime_utc", "u_ms", "v_ms", "ws_ms"]], on="datetime_utc", how="left")
    return df.sort_values("datetime_utc").reset_index(drop=True)


def episode_hourly_frame(episode, rtma_df, *, ignition_hour_local=12, utc_offset_h=-6,
                         herb_scale=6.5, woody_scale=9.0, max_gap_h=3):
    """
    Materialize one episode as an hourly driver frame in pyretechnics units.

    Parameters
    ----------
    episode : Mapping
        A row of :func:`select_episodes` (needs ``start_date, end_date``).
    rtma_df : pandas.DataFrame
        From :func:`load_rtma_hourly` with ``wind_dir=`` (needs ``u_ms, v_ms, ws_ms``).
    ignition_hour_local, utc_offset_h : int
        Hour 0 of the cube is the ignition hour in local time (MDT = -6).
    herb_scale, woody_scale : float
        Live FM from FM100, as ``build_erc_classes`` uses.
    max_gap_h : int
        Longest run of missing hours to interpolate; longer raises.

    Returns
    -------
    pandas.DataFrame
        ``duration_days * 24`` rows: ``hour, datetime_utc, tmp_c, rh_pct, ws_kmh,
        wd_from_deg, fm1, fm10, fm100, fm_herb, fm_woody`` (moistures as
        fractions). The frame's wind is the pyrome-mean RTMA wind; an HRRR cube
        replaces it at run time (:func:`weather_arrays`).
    """
    start = datetime.combine(pd.Timestamp(episode["start_date"]).date(), datetime.min.time())
    start += timedelta(hours=ignition_hour_local - utc_offset_h)
    n_days = (pd.Timestamp(episode["end_date"]) - pd.Timestamp(episode["start_date"])).days + 1
    idx = pd.date_range(start, periods=n_days * 24, freq="h")

    src = rtma_df.set_index("datetime_utc").reindex(idx)
    missing = [c for c in ("u_ms", "v_ms", "ws_ms") if c not in src]
    if missing:
        raise ValueError(f"rtma_df lacks {missing}; load it with load_rtma_hourly(..., wind_dir=)")
    if "rh_pct" not in src:
        src["rh_pct"] = np.nan                       # older caches; RH is report-only, never an engine input
    src = src[["tmp_f", "rh_pct", "FM1", "FM10", "FM100", "u_ms", "v_ms", "ws_ms"]].astype(float)
    gap = longest_true_run(src.drop(columns="rh_pct").isna().any(axis=1).to_numpy())
    if gap > max_gap_h:
        raise ValueError(f"RTMA gap of {gap} h in episode starting {idx[0]} (max_gap_h={max_gap_h})")
    src = src.interpolate(limit_direction="both")

    out = pd.DataFrame({"hour": np.arange(len(idx)), "datetime_utc": idx})
    out["tmp_c"] = (src["tmp_f"].values - 32.0) / 1.8
    out["rh_pct"] = src["rh_pct"].values
    out["ws_kmh"], out["wd_from_deg"] = uv_to_speed_dir(src["u_ms"].values, src["v_ms"].values, "km/h")
    out["fm1"] = src["FM1"].values / 100.0
    out["fm10"] = src["FM10"].values / 100.0
    out["fm100"] = src["FM100"].values / 100.0
    herb, woody = calc_live_fm_from_dead(src["FM100"].values, herb_scale, woody_scale)
    out["fm_herb"], out["fm_woody"] = herb / 100.0, woody / 100.0
    out.attrs["wind_source"] = "rtma"
    out.attrs["episode_id"] = episode.get("episode_id")
    return out


def weather_from_scenario(scenario, *, wind_from_deg, tmp_c=30.0, n_hours=1):
    """
    A FlamMap scenario-cache entry (``pyrome_<id>_flammap.json`` ``scenarios[pNN]``)
    as engine-unit weather, so a single-run map is parameterized exactly like
    the FlamMap runs it sits beside.

    ``WIND_SPEED`` there is HRRR 10 m mph -> km/h; FM columns are percent ->
    fraction. ``n_hours`` repeats the state into that many bands.

    Returns
    -------
    dict
        ``{cube_name: float32 array of n_hours}``.
    """
    vals = {
        "temperature": tmp_c,
        "wind_speed_10m": scenario["WIND_SPEED"] * 1.609344,
        "upwind_direction": wind_from_deg,
        "fuel_moisture_dead_1hr": scenario["FM_1hr"] / 100.0,
        "fuel_moisture_dead_10hr": scenario["FM_10hr"] / 100.0,
        "fuel_moisture_dead_100hr": scenario["FM_100hr"] / 100.0,
        "fuel_moisture_live_herbaceous": scenario["FM_herb"] / 100.0,
        "fuel_moisture_live_woody": scenario["FM_woody"] / 100.0,
    }
    return {k: np.full(n_hours, v, dtype="float32") for k, v in vals.items()}


def weather_arrays(frame, wind_cube=None):
    """
    Hourly frame -> ``{cube_name: float32 array}`` for :func:`~fb_tools.models.pyre.build_cubes`.

    Parameters
    ----------
    frame : pandas.DataFrame
        From :func:`episode_hourly_frame`.
    wind_cube : Path or dict, optional
        A :func:`episode_wind_cube` result (or its ``.npz``). Replaces the
        frame's 1-D wind with the 3-D ``(n_hours, rows/ky, cols/kx)`` arrays.
    """
    out = {cube: frame[col].to_numpy(dtype="float32") for cube, col in WEATHER_CUBES.items()}
    if wind_cube is not None:
        wc = load_wind_cube(wind_cube) if isinstance(wind_cube, (str, Path)) else wind_cube
        for k in WIND_CUBES:
            a = np.asarray(wc[k], dtype="float32")
            if a.shape[0] != len(frame):
                raise ValueError(f"{k} has {a.shape[0]} hours; frame has {len(frame)}")
            out[k] = a
    return out


# ── Local wind: HRRR 3-D cubes on a coarse block grid ───────────────────────

WIND_CUBES = ("wind_speed_10m", "upwind_direction")


def layer_bounds_lonlat(layers_dir, *, margin_km=5.0, n_edge=25):
    """Lon/lat box ``(min_lon, min_lat, max_lon, max_lat)`` of a ``destack_lcp`` layer set plus a margin."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(Path(layers_dir) / "slope.tif") as src:
        b, crs = src.bounds, src.crs
    m = margin_km * 1000.0
    xs = np.linspace(b.left - m, b.right + m, n_edge)
    ys = np.linspace(b.bottom - m, b.top + m, n_edge)
    ex = np.concatenate([xs, xs, np.full(n_edge, xs[0]), np.full(n_edge, xs[-1])])
    ey = np.concatenate([np.full(n_edge, ys[0]), np.full(n_edge, ys[-1]), ys, ys])
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(ex, ey)
    return (float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat)))


def block_factors(shape, target=100):
    """Largest divisor of each dimension not above ``target`` — a block the engine can repeat exactly."""
    return tuple(max(k for k in range(1, min(int(n), int(target)) + 1) if n % k == 0) for n in shape)


def episode_wind_cube(hrrr, transform, crs, shape, *, block=None, target_block=100):
    """
    Nearest-HRRR-cell wind on a coarse block grid aligned to the landscape.

    Parameters
    ----------
    hrrr : dict
        From :func:`~fb_tools.weather.hrrr.fetch_hrrr_wind_grid` (``u, v, lat, lon``).
    transform, crs : affine.Affine, CRS
        Landscape grid (from :func:`~fb_tools.models.pyre.load_static_layers`).
    shape : tuple
        Landscape ``(rows, cols)``.
    block : tuple, optional
        ``(ky, kx)`` cells per block; must divide ``shape``. Default
        :func:`block_factors` with ``target_block``.

    Returns
    -------
    dict
        ``wind_speed_10m`` km/h and ``upwind_direction`` degrees (met FROM), each
        float32 ``(n_hours, rows/ky, cols/kx)``; ``block``, ``block_m``, ``source``,
        ``n_hrrr_cells``.
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    rows, cols = int(shape[0]), int(shape[1])
    ky, kx = block or block_factors((rows, cols), target_block)
    if rows % ky or cols % kx:
        raise ValueError(f"block {ky}x{kx} does not divide the landscape {rows}x{cols}")
    ny, nx = rows // ky, cols // kx
    cc, rr = np.meshgrid((np.arange(nx) + 0.5) * kx, (np.arange(ny) + 0.5) * ky)
    x, y = transform * (cc.ravel(), rr.ravel())
    hx, hy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        np.asarray(hrrr["lon"]).ravel(), np.asarray(hrrr["lat"]).ravel())
    _, idx = cKDTree(np.column_stack([hx, hy])).query(np.column_stack([x, y]))
    n_t = hrrr["u"].shape[0]
    u = np.asarray(hrrr["u"]).reshape(n_t, -1)[:, idx].reshape(n_t, ny, nx)
    v = np.asarray(hrrr["v"]).reshape(n_t, -1)[:, idx].reshape(n_t, ny, nx)
    res = abs(float(transform.a))
    ws, wd = uv_to_speed_dir(u, v, "km/h")
    return {
        "wind_speed_10m": ws.astype("float32"),
        "upwind_direction": wd.astype("float32"),
        "block": (ky, kx), "block_m": (ky * res, kx * res),
        "source": "hrrr_10m_nearest", "n_hrrr_cells": int(len(np.unique(idx))),
    }


def write_wind_cube(path, cube):
    """Save an :func:`episode_wind_cube` result as ``.npz`` (readable without cfgrib/netCDF)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, wind_speed_10m=cube["wind_speed_10m"], upwind_direction=cube["upwind_direction"],
                        block=np.asarray(cube["block"]), block_m=np.asarray(cube["block_m"]),
                        source=np.asarray(cube["source"]), n_hrrr_cells=np.asarray(cube["n_hrrr_cells"]))
    return path


def load_wind_cube(path):
    """Inverse of :func:`write_wind_cube`."""
    with np.load(path) as z:
        return {"wind_speed_10m": z["wind_speed_10m"], "upwind_direction": z["upwind_direction"],
                "block": tuple(int(b) for b in z["block"]), "block_m": tuple(float(b) for b in z["block_m"]),
                "source": str(z["source"]), "n_hrrr_cells": int(z["n_hrrr_cells"])}


def wind_cube_summary(cube):
    """Per-hour domain mean / max speed (km/h) and vector-mean direction of a wind cube, for plots."""
    ws, wd = cube["wind_speed_10m"], np.radians(cube["upwind_direction"])
    n_t = ws.shape[0]
    u = -(ws * np.sin(wd)).reshape(n_t, -1).mean(axis=1)
    v = -(ws * np.cos(wd)).reshape(n_t, -1).mean(axis=1)
    return pd.DataFrame({"hour": np.arange(n_t), "ws_mean": ws.reshape(n_t, -1).mean(axis=1),
                         "ws_p90": np.percentile(ws.reshape(n_t, -1), 90, axis=1),
                         "ws_max": ws.reshape(n_t, -1).max(axis=1),
                         "wd_vecmean": uv_to_speed_dir(u, v)[1]})


def write_episode_frames(episodes, rtma_df, out_dir, **kwargs):
    """Write one ``<episode_id>.csv`` per episode; returns ``{episode_id: Path}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for _, ep in episodes.iterrows():
        fp = out_dir / f"{ep['episode_id']}.csv"
        episode_hourly_frame(ep, rtma_df, **kwargs).to_csv(fp, index=False)
        paths[ep["episode_id"]] = fp
    return paths


def episode_weather_report(episodes, frames, wind_cubes=None, *, fire_hours_utc=(19, 20, 21, 22)):
    """
    One row per episode describing the fire weather it carries, for a report.

    Parameters
    ----------
    episodes : DataFrame
        From :func:`select_episodes`.
    frames : dict
        ``{episode_id: csv path or DataFrame}`` hourly frames.
    wind_cubes : dict, optional
        ``{episode_id: npz path}``; adds the local HRRR wind (fire-hour domain
        mean and windiest-cell percentiles) beside the pyrome-mean RTMA wind.
    fire_hours_utc : tuple of int
        Afternoon hours the wind statistics are taken over (13–16 MDT).

    Returns
    -------
    DataFrame
        ``episode_id, start_date, end_date, erc_day1, erc_mean, tmp_max_c,
        rh_min_pct, fm1_min_pct, ws_rtma_p50/p97_kmh, ws_hrrr_mean_p50/p97_kmh,
        ws_hrrr_max_p97_kmh, ws_hrrr_max_kmh, wd_dominant_deg``.
    """
    rows = []
    for _, ep in episodes.iterrows():
        eid = ep["episode_id"]
        f = frames[eid]
        f = pd.read_csv(f, parse_dates=["datetime_utc"]) if not isinstance(f, pd.DataFrame) else f
        fh = f[pd.to_datetime(f.datetime_utc).dt.hour.isin(fire_hours_utc)]
        row = {"episode_id": eid, "start_date": pd.Timestamp(ep["start_date"]).date(),
               "end_date": pd.Timestamp(ep["end_date"]).date(),
               "erc_day1": ep["erc_day1"], "erc_mean": ep["erc_mean"],
               "erc_pct": ep.get("erc_pct"), "source": ep.get("source"),
               "tmp_max_c": round(float(f.tmp_c.max()), 1),
               "rh_min_pct": round(float(f.rh_pct.min()), 1) if "rh_pct" in f else np.nan,
               "fm1_min_pct": round(float(f.fm1.min() * 100), 1),
               "ws_rtma_p50_kmh": round(float(fh.ws_kmh.quantile(0.5)), 1),
               "ws_rtma_p97_kmh": round(float(fh.ws_kmh.quantile(0.97)), 1)}
        if wind_cubes and eid in wind_cubes:
            c = load_wind_cube(wind_cubes[eid])
            s = wind_cube_summary(c)
            s = s[pd.to_datetime(f.datetime_utc).dt.hour.reindex(s.hour).isin(fire_hours_utc).values]
            u = -(c["wind_speed_10m"] * np.sin(np.radians(c["upwind_direction"]))).mean()
            v = -(c["wind_speed_10m"] * np.cos(np.radians(c["upwind_direction"]))).mean()
            row.update({"ws_hrrr_mean_p50_kmh": round(float(s.ws_mean.quantile(0.5)), 1),
                        "ws_hrrr_mean_p97_kmh": round(float(s.ws_mean.quantile(0.97)), 1),
                        "ws_hrrr_max_p97_kmh": round(float(s.ws_max.quantile(0.97)), 1),
                        "ws_hrrr_max_kmh": round(float(s.ws_max.max()), 1),
                        "wd_dominant_deg": round(float(uv_to_speed_dir(np.array([u]), np.array([v]))[1][0]))})
        rows.append(row)
    return pd.DataFrame(rows)
