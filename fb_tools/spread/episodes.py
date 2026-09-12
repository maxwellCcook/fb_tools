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
    present = rtma_df["datetime_utc"].isin(idx)
    have = pd.Series(False, index=idx)
    have[rtma_df.loc[present, "datetime_utc"]] = True
    miss = (~have).astype(int)
    runs = miss.groupby((miss != miss.shift()).cumsum()).sum()
    return int(runs.max()) if len(runs) else 0


def select_episodes(pyrome_id, erc_cache_dir, *, severity="p97", n=5, duration_days=21,
                    years=None, rtma_df=None, max_gap_h=3, seed=0):
    """
    Choose ``n`` non-overlapping historical windows whose day-1 ERC meets a
    severity percentile of the pyrome's GridMET climatology.

    Parameters
    ----------
    pyrome_id : int or str
    erc_cache_dir : Path
        Directory of ``pyrome_<id>_gridmet.json`` (``data/weather/pyrome_erc``).
    severity : str or float
        ``"p97"`` style label or a percentile in (0, 100). Threshold is taken
        over every (year, season-day) ERC value in the climatology.
    n, duration_days : int
    years : iterable of int, optional
        Restrict start years.
    rtma_df : pandas.DataFrame, optional
        Hourly record from :func:`load_rtma_hourly`. When given, windows with a
        run of more than ``max_gap_h`` missing hours are not candidates, so an
        episode can never fail later in :func:`episode_hourly_frame`.
    max_gap_h : int
    seed : int

    Returns
    -------
    pandas.DataFrame
        One row per episode: ``episode_id, year, season_day, start_date,
        end_date, erc_day1, erc_mean, erc_threshold``. Sorted by start date.
    """
    meta = load_gridmet_pyrome_cache(pyrome_id, erc_cache_dir, return_meta=True)
    erc = np.asarray(meta["HistoricERCValues"], dtype=float)
    yrs = np.asarray(meta["years"], dtype=int)
    pct = float(str(severity).lstrip("pP"))
    threshold = float(np.nanpercentile(erc, pct))

    keep = np.ones(len(yrs), dtype=bool) if years is None else np.isin(yrs, list(years))
    last_start = SEASON_DAYS - duration_days + 1
    cands = [(y, d) for yi, y in enumerate(yrs) if keep[yi]
             for d in range(1, last_start + 1) if erc[yi, d - 1] >= threshold]
    if rtma_df is not None:
        n_erc = len(cands)
        cands = [(y, d) for y, d in cands
                 if _longest_gap_h(rtma_df, season_day_to_date(y, d), duration_days + 1) <= max_gap_h]
        print(f"[select_episodes] {n_erc} windows meet {severity}; "
              f"{len(cands)} also have RTMA coverage (gaps <= {max_gap_h} h)")
    if not cands:
        raise ValueError(f"No season day reaches ERC {threshold:.1f} ({severity}) "
                         f"in years {sorted(yrs[keep])}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cands))
    chosen = []
    for k in order:
        y, d = cands[k]
        if all(y != cy or abs(d - cd) >= duration_days for cy, cd in chosen):
            chosen.append((y, d))
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(f"Only {len(chosen)} non-overlapping {duration_days}-day "
                         f"episodes available at {severity}; asked for {n}")

    rows = []
    for y, d in sorted(chosen):
        yi = int(np.where(yrs == y)[0][0])
        start = pd.Timestamp(season_day_to_date(y, d))
        rows.append({
            "year": y, "season_day": d,
            "start_date": start, "end_date": start + pd.Timedelta(days=duration_days - 1),
            "erc_day1": round(float(erc[yi, d - 1]), 1),
            "erc_mean": round(float(erc[yi, d - 1:d - 1 + duration_days].mean()), 1),
            "erc_threshold": round(threshold, 1),
        })
    df = pd.DataFrame(rows)
    df.insert(0, "episode_id", [f"ep_{i:03d}" for i in range(len(df))])
    return df


def load_rtma_hourly(rtma_dir, pyrome_id, years, *, wind_dir=None):
    """
    Load the cached hourly RTMA record for one pyrome with dead FM spun up.

    Parameters
    ----------
    rtma_dir : Path
        Directory of ``rtma_hourly_*_pyromes_<year>.csv``.
    pyrome_id : int
    years : iterable of int
    wind_dir : Path, optional
        Directory of ``rtma_wind_*_pyromes_<year>.csv`` (``u_ms, v_ms, ws_ms``)
        from the wind export in ``00b_RTMA-EMC.ipynb``. Merged on time when given.

    Returns
    -------
    pandas.DataFrame
        Hourly rows for the pyrome with ``FM1, FM10, FM100`` appended by
        :func:`~fb_tools.weather.rtma.build_rtma_dead_fm`. The lag is run over
        the full record so every episode window is spun up.
    """
    rtma_dir = Path(rtma_dir)
    files = [p for y in years for p in sorted(rtma_dir.glob(f"rtma_hourly_*_{y}.csv"))]
    if not files:
        raise FileNotFoundError(f"No rtma_hourly_*_<year>.csv for {sorted(years)} in {rtma_dir}")
    df = load_rtma_csv(files)
    df = df[df["pyrome_id"].astype(int) == int(pyrome_id)].copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = build_rtma_dead_fm(df)

    if wind_dir is not None:
        wfiles = [p for y in years for p in sorted(Path(wind_dir).glob(f"rtma_wind_*_{y}.csv"))]
        if not wfiles:
            raise FileNotFoundError(f"No rtma_wind_*_<year>.csv for {sorted(years)} in {wind_dir}")
        w = pd.concat(pd.read_csv(p) for p in wfiles)
        w = w[w["pyrome_id"].astype(int) == int(pyrome_id)]
        w["datetime_utc"] = pd.to_datetime(w["datetime_utc"])
        df = df.merge(w[["datetime_utc", "u_ms", "v_ms", "ws_ms"]], on="datetime_utc", how="left")
    return df.sort_values("datetime_utc").reset_index(drop=True)


def episode_hourly_frame(episode, rtma_df, *, wind=None, ignition_hour_local=12,
                         utc_offset_h=-6, herb_scale=6.5, woody_scale=9.0, max_gap_h=3):
    """
    Materialize one episode as an hourly driver frame in pyretechnics units.

    Parameters
    ----------
    episode : Mapping
        A row of :func:`select_episodes` (needs ``start_date, end_date``).
    rtma_df : pandas.DataFrame
        From :func:`load_rtma_hourly`.
    wind : tuple or None
        ``(speed_kmh, from_deg)`` constant wind, used when the RTMA record has
        no ``u_ms/v_ms/ws_ms`` columns. ``None`` requires them.
    ignition_hour_local, utc_offset_h : int
        Hour 0 of the cube is the ignition hour in local time (MDT = -6).
    herb_scale, woody_scale : float
        Live FM from FM100, as ``build_erc_classes`` uses.
    max_gap_h : int
        Longest run of missing hours to interpolate; longer raises.

    Returns
    -------
    pandas.DataFrame
        ``duration_days * 24`` rows: ``hour, datetime_utc, tmp_c, ws_kmh,
        wd_from_deg, fm1, fm10, fm100, fm_herb, fm_woody`` (moistures as
        fractions). ``frame.attrs["wind_source"]`` is ``"rtma"`` or ``"constant"``.
    """
    start = datetime.combine(pd.Timestamp(episode["start_date"]).date(), datetime.min.time())
    start += timedelta(hours=ignition_hour_local - utc_offset_h)
    n_days = (pd.Timestamp(episode["end_date"]) - pd.Timestamp(episode["start_date"])).days + 1
    idx = pd.date_range(start, periods=n_days * 24, freq="h")

    src = rtma_df.set_index("datetime_utc").reindex(idx)
    cols = ["tmp_f", "FM1", "FM10", "FM100"]
    have_wind = wind is None
    if have_wind:
        missing = [c for c in ("u_ms", "v_ms", "ws_ms") if c not in src]
        if missing:
            raise ValueError(f"rtma_df lacks {missing}; pass wind=(speed_kmh, from_deg) "
                             "or load with wind_dir=")
        cols += ["u_ms", "v_ms", "ws_ms"]
    src = src[cols]
    gap = src.isna().any(axis=1).astype(int)
    run = gap.groupby((gap != gap.shift()).cumsum()).transform("sum") * gap
    if run.max() > max_gap_h:
        raise ValueError(f"RTMA gap of {int(run.max())} h in episode starting {idx[0]} "
                         f"(max_gap_h={max_gap_h})")
    src = src.interpolate(limit_direction="both")

    out = pd.DataFrame({"hour": np.arange(len(idx)), "datetime_utc": idx})
    out["tmp_c"] = (src["tmp_f"].values - 32.0) / 1.8
    if have_wind:
        out["ws_kmh"] = src["ws_ms"].values * 3.6
        out["wd_from_deg"] = (np.degrees(np.arctan2(-src["u_ms"].values, -src["v_ms"].values)) + 360) % 360
    else:
        out["ws_kmh"], out["wd_from_deg"] = float(wind[0]), float(wind[1])
    out["fm1"] = src["FM1"].values / 100.0
    out["fm10"] = src["FM10"].values / 100.0
    out["fm100"] = src["FM100"].values / 100.0
    herb, woody = calc_live_fm_from_dead(src["FM100"].values, herb_scale, woody_scale)
    out["fm_herb"], out["fm_woody"] = herb / 100.0, woody / 100.0
    out.attrs["wind_source"] = "rtma" if have_wind else "constant"
    out.attrs["episode_id"] = episode.get("episode_id")
    return out


def weather_from_scenario(scenario, *, wind_from_deg, tmp_c=30.0, n_hours=1, herb_scale=None):
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


def weather_arrays(frame):
    """Hourly frame -> ``{cube_name: float32 1-D array}`` for :func:`~fb_tools.models.pyre.build_cubes`."""
    return {cube: frame[col].to_numpy(dtype="float32") for cube, col in WEATHER_CUBES.items()}


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
