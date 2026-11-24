"""
NASA FIRMS active-fire detections: source catalog, cached per-fire fetch, shared schema.

Area CSV API::

    {FIRMS_BASE}/area/csv/{MAP_KEY}/{SOURCE}/{W,S,E,N}/{DAY_RANGE}/{DATE}

A free MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/) is read from the
``map_key`` argument, then ``$FIRMS_MAP_KEY``; never from a file. The limit is about
5,000 transactions per 10 minutes.

FRP is kept per sensor and is not comparable across instruments (pixel size, bands,
retrieval differ). Never pool it across ``instrument``.

Ported from AspenAcres ``firms_nrt.FIRMSAccess``.
"""
import hashlib
import io
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api"

# Day-range cap per request: 5 days for most sources, 2 for the high-cadence GOES product.
DEFAULT_MAX_DAYS = 5
SOURCE_MAX_DAYS = {"GOES_NRT": 2}

# source: (satellite label, instrument, nominal pixel in m, NRT | SP)
SOURCE_META = {
    "VIIRS_SNPP_NRT":   ("VIIRS S-NPP",   "VIIRS", 375,  "NRT"),
    "VIIRS_SNPP_SP":    ("VIIRS S-NPP",   "VIIRS", 375,  "SP"),
    "VIIRS_NOAA20_NRT": ("VIIRS NOAA-20", "VIIRS", 375,  "NRT"),
    "VIIRS_NOAA20_SP":  ("VIIRS NOAA-20", "VIIRS", 375,  "SP"),
    "VIIRS_NOAA21_NRT": ("VIIRS NOAA-21", "VIIRS", 375,  "NRT"),
    "MODIS_NRT":        ("MODIS",         "MODIS", 1000, "NRT"),
    "MODIS_SP":         ("MODIS",         "MODIS", 1000, "SP"),
    "LANDSAT_NRT":      ("Landsat",       "OLI",   30,   "NRT"),
    "GOES_NRT":         ("GOES",          "ABI",   2000, "NRT"),
}
VIIRS_NRT = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
VIIRS_SP = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")

# Shared detection schema (column order). Swath rows (``swath.py``) use the same columns.
STD_COLS = [
    "detection_id", "longitude", "latitude", "frp", "acq_datetime", "acq_date", "acq_time",
    "confidence_norm", "confidence_raw", "daynight", "satellite", "instrument", "source",
    "product", "scan", "track", "type", "version", "bright_ti4", "bright_ti5", "geometry",
]
_CONF_LETTERS = {"l": "low", "n": "nominal", "h": "high", "m": "nominal"}


# ── Key, cache, requests ───────────────────────────────────────────────────────

def _get_map_key(map_key=None) -> str:
    key = map_key or os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError(
            "No FIRMS MAP_KEY found. Set $FIRMS_MAP_KEY (free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/) or pass map_key=."
        )
    return key


def default_firms_cache(cache_dir=None) -> Path:
    """``cache_dir``, else ``$FB_FIRMS_CACHE``, else ``./firms_cache`` in the working directory."""
    return Path(cache_dir or os.environ.get("FB_FIRMS_CACHE") or Path.cwd() / "firms_cache")


def _read_firms_csv(url: str, map_key: str = ""):
    """
    GET a FIRMS CSV endpoint; DataFrame, or None when there are no records.

    FIRMS reports errors as short plain-text bodies (e.g. 'Invalid MAP_KEY'), often
    without an HTTP error, so the payload is checked before ``raise_for_status``.
    The key is redacted from any message raised.
    """
    resp = requests.get(url, timeout=120)
    text = resp.text.strip()
    low = text.lower()
    if "invalid" in low[:40] or "map_key" in low[:40] or low.startswith(("<", "error")):
        shown = url.replace(map_key, "<MAP_KEY>") if map_key else url
        raise RuntimeError(f"FIRMS API error for {shown}: {text[:300]}")
    resp.raise_for_status()
    if not text:
        return None
    df = pd.read_csv(io.StringIO(text))
    return None if df.empty else df


def _availability(map_key: str):
    """Live ``data_availability`` table (``data_id, min_date, max_date``), or None if unreadable."""
    try:
        df = _read_firms_csv(f"{FIRMS_BASE}/data_availability/csv/{map_key}/all", map_key)
    except (RuntimeError, requests.RequestException) as e:
        print(f"  ! FIRMS availability not read: {e}")
        return None
    if df is None or "data_id" not in df.columns:
        return None
    for c in ("min_date", "max_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df.set_index("data_id")


def firms_sources(map_key=None, *, print_table: bool = True) -> pd.DataFrame:
    """
    Catalog of FIRMS active-fire sources.

    Parameters
    ----------
    map_key : str, optional
        With a key (argument only; ``$FIRMS_MAP_KEY`` is not read here, so the call
        stays offline by default), live ``min_date`` / ``max_date`` are added.
    print_table : bool

    Returns
    -------
    DataFrame
        ``source, satellite, instrument, pixel_m, latency`` (+ availability dates).
    """
    cat = pd.DataFrame(
        [(s, *m) for s, m in SOURCE_META.items()],
        columns=["source", "satellite", "instrument", "pixel_m", "latency"],
    )
    if map_key:
        avail = _availability(map_key)
        if avail is not None:
            cat = cat.join(avail[["min_date", "max_date"]], on="source")
    if print_table:
        print(cat.to_string(index=False))
    return cat


# ── Schema ─────────────────────────────────────────────────────────────────────

def _norm_confidence(conf: pd.Series) -> pd.Series:
    """Per-sensor confidence → low / nominal / high (letters, or MODIS 0–100: <30, ≤80, >80)."""
    s = conf.astype("string").str.strip().str.lower()
    v = pd.to_numeric(s, errors="coerce").to_numpy(float, na_value=np.nan)
    num = pd.Series(np.select([v < 30, v <= 80, v > 80], ["low", "nominal", "high"], None), conf.index)
    return s.map(_CONF_LETTERS).astype("object").fillna(num)


def _empty_detections() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=STD_COLS, geometry="geometry", crs="EPSG:4326")


def standardize_firms(raw: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Raw FIRMS rows (any mix of sources, with a ``source`` column) → the shared schema.

    Column names are unified; values are not cross-calibrated. Negative FRP (the GOES
    −999 fill) becomes NaN. ``acq_datetime`` is tz-aware UTC. Exact repeats of
    (longitude, latitude, acq_datetime, source) are dropped.

    Returns
    -------
    GeoDataFrame
        ``STD_COLS``, EPSG:4326 points. ``detection_id`` is built from source, time and
        location, so it is stable across fetches.
    """
    if raw is None or len(raw) == 0:
        return _empty_detections()
    df = raw.copy()

    hhmm = df["acq_time"].astype("Int64").astype(str).str.zfill(4)
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + hhmm, format="%Y-%m-%d %H%M", utc=True, errors="coerce"
    )
    meta = df["source"].map(SOURCE_META)
    df["satellite"] = meta.map(lambda t: t[0] if isinstance(t, tuple) else None)
    df["instrument"] = meta.map(lambda t: t[1] if isinstance(t, tuple) else None)
    df["product"] = "FIRMS"

    conf = df["confidence"] if "confidence" in df else pd.Series(pd.NA, index=df.index)
    df["confidence_raw"] = conf
    df["confidence_norm"] = _norm_confidence(conf)

    frp = pd.to_numeric(df["frp"], errors="coerce") if "frp" in df else pd.Series(np.nan, df.index)
    df["frp"] = frp.where(frp >= 0)   # Landsat has no FRP; GOES fills with -999

    df = df.drop_duplicates(["longitude", "latitude", "acq_datetime", "source"]).reset_index(drop=True)
    df["detection_id"] = (
        df["source"] + "_" + df["acq_datetime"].dt.strftime("%Y%m%dT%H%M")
        + "_" + df["longitude"].map("{:.5f}".format) + "_" + df["latitude"].map("{:.5f}".format)
    )
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )
    for c in STD_COLS:
        if c not in gdf.columns:
            gdf[c] = pd.NA
    return gdf[STD_COLS]


# ── Fetch ──────────────────────────────────────────────────────────────────────

def _bbox_wgs84(geoms: gpd.GeoSeries, buffer_m: float) -> tuple:
    """(W, S, E, N) in lon/lat, buffered in metres and rounded outward to 1e-4°."""
    if buffer_m:
        geoms = geoms.to_crs(geoms.estimate_utm_crs()).buffer(buffer_m)
    w, s, e, n = geoms.to_crs("EPSG:4326").total_bounds
    return (math.floor(w * 1e4) / 1e4, math.floor(s * 1e4) / 1e4,
            math.ceil(e * 1e4) / 1e4, math.ceil(n * 1e4) / 1e4)


def _resolve_events(aoi, start, end, start_col, end_col, id_col, buffer_m) -> list:
    """Normalize the input to a list of ``(event_id | None, bbox, first_day, last_day)``."""
    def window(s, e):
        if pd.isna(s) or pd.isna(e):
            raise ValueError("Every fetch needs a start and an end date.")
        s, e = pd.Timestamp(s).date(), pd.Timestamp(e).date()
        if e < s:
            raise ValueError(f"end ({e}) is before start ({s}).")
        return s, e

    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        from shapely.geometry import box
        aoi = gpd.GeoSeries([box(*map(float, aoi))], crs="EPSG:4326")
    elif hasattr(aoi, "geom_type") and not isinstance(aoi, (gpd.GeoSeries, gpd.GeoDataFrame)):
        aoi = gpd.GeoSeries([aoi], crs="EPSG:4326")  # a bare geometry is taken as lon/lat
    if isinstance(aoi, gpd.GeoSeries):
        return [(None, _bbox_wgs84(aoi, buffer_m), *window(start, end))]
    if not isinstance(aoi, gpd.GeoDataFrame):
        raise ValueError("aoi_or_events must be a (W, S, E, N) bbox, a geometry or a GeoDataFrame.")
    if not (id_col or start_col or end_col):      # one AOI: the frame's whole extent
        return [(None, _bbox_wgs84(aoi.geometry, buffer_m), *window(start, end))]

    ids = aoi[id_col] if id_col else aoi.index
    if ids.duplicated().any():
        raise ValueError("Event ids must be unique.")
    return [
        (eid, _bbox_wgs84(aoi.geometry.iloc[[i]], buffer_m),
         *window(aoi[start_col].iloc[i] if start_col else start,
                 aoi[end_col].iloc[i] if end_col else end))
        for i, eid in enumerate(ids)
    ]


def _day_runs(days: list, max_days: int):
    """Group sorted dates into (first_day, n_days) requests of consecutive days, ≤ max_days each."""
    run = []
    for d in days:
        if run and ((d - run[-1]).days != 1 or len(run) == max_days):
            yield run[0], len(run)
            run = []
        run.append(d)
    if run:
        yield run[0], len(run)


def _is_final(fp: Path, day, refetch_recent_days: int) -> bool:
    """A cached day is final once it was written ≥ ``refetch_recent_days`` after the day began."""
    if not fp.exists():
        return False
    written = pd.Timestamp(fp.stat().st_mtime, unit="s", tz="UTC")
    return written >= pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=refetch_recent_days)


def _write_day(fp: Path, df: pd.DataFrame) -> None:
    tmp = fp.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(fp)


def fetch_firms(
    aoi_or_events,
    start=None,
    end=None,
    *,
    sources=VIIRS_NRT,
    start_col=None,
    end_col=None,
    id_col=None,
    buffer_m: float = 0,
    map_key=None,
    cache_dir=None,
    request_pause: float = 1.0,
    refetch_recent_days: int = 1,
) -> gpd.GeoDataFrame:
    """
    Fetch FIRMS detections for one AOI or a set of events, through a per-day cache.

    Fires are requested one at a time, never in parallel. Each fire's buffered bbox ×
    date window is split into requests of ≤ 5 days (2 for GOES), with ``request_pause``
    seconds between requests. Detections are returned for the **bbox**, not the
    polygon — clip with ``events.associate_detections``.

    Parameters
    ----------
    aoi_or_events : tuple, shapely geometry, GeoSeries or GeoDataFrame
        A lon/lat ``(W, S, E, N)`` bbox, a geometry (lon/lat if bare), or a
        GeoDataFrame. A GeoDataFrame is one AOI (its total extent) unless ``id_col``,
        ``start_col`` or ``end_col`` is given; then every row is an event.
    start, end : date-like, optional
        Inclusive UTC days. Used for every event that has no ``start_col``/``end_col``.
    sources : sequence of str
        FIRMS source names (see ``firms_sources``). NRT and SP are separate sources.
    start_col, end_col, id_col : str, optional
        Per-event date and id columns. Without ``id_col`` the index is the id.
    buffer_m : float
        Buffer around each AOI/event, metres.
    map_key : str, optional
        Falls back to ``$FIRMS_MAP_KEY``.
    cache_dir : path-like, optional
        See ``default_firms_cache``. One Parquet per (event/bbox, source, UTC day);
        a day with no detections is cached as an empty file.
    request_pause : float
        Seconds slept after each request.
    refetch_recent_days : int
        A cached day is re-requested until it has been written at least this many days
        after the day began, so today's partial day (default 1) is always re-pulled.

    Returns
    -------
    GeoDataFrame
        ``STD_COLS``, plus ``event_id`` when events were given.

    Notes
    -----
    Days outside a source's live availability are skipped and not cached, so a
    Standard Product day requested before it is published is fetched once it exists.
    A failed request is reported and skipped (not cached); re-run to fill it.
    """
    key = _get_map_key(map_key)
    cache = default_firms_cache(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    sources = [sources] if isinstance(sources, str) else list(sources)
    events = _resolve_events(aoi_or_events, start, end, start_col, end_col, id_col, buffer_m)
    tagged = events[0][0] is not None
    avail = None          # read once, only if something has to be requested
    out = []

    for k, (eid, bbox, d0, d1) in enumerate(events, 1):
        area = ",".join(f"{v:.4f}" for v in bbox)
        stem = hashlib.sha1(area.encode()).hexdigest()[:10]
        if tagged:
            stem = f"{re.sub(r'[^A-Za-z0-9-]+', '-', str(eid))}_{stem}"
        days = [d.date() for d in pd.date_range(d0, d1)]
        n_req = n_cached = n_failed = 0
        frames = []

        for src in sources:
            fps = {d: cache / f"{stem}_{src}_{d:%Y%m%d}.parquet" for d in days}
            need = [d for d in days if not _is_final(fps[d], d, refetch_recent_days)]
            if need and avail is None:
                avail = _availability(key)
                avail = avail if avail is not None else pd.DataFrame()
                time.sleep(request_pause)
            if need and src in avail.index and avail.loc[src, ["min_date", "max_date"]].notna().all():
                lo, hi = avail.loc[src, ["min_date", "max_date"]]
                skipped = [d for d in need if not lo <= d <= hi]
                if skipped:
                    print(f"  {src}: {len(skipped)} day(s) outside availability {lo}..{hi}, skipped")
                    need = [d for d in need if lo <= d <= hi]
            n_cached += len(days) - len(need)

            for first, n in _day_runs(need, SOURCE_MAX_DAYS.get(src, DEFAULT_MAX_DAYS)):
                url = f"{FIRMS_BASE}/area/csv/{key}/{src}/{area}/{n}/{first.isoformat()}"
                n_req += 1
                try:
                    df = _read_firms_csv(url, key)
                except (RuntimeError, requests.RequestException) as e:
                    msg = str(e).replace(key, "<MAP_KEY>")
                    print(f"  ! {src} {first} ({n} d) failed, skipped: {msg}")
                    n_failed += 1
                    continue
                finally:
                    time.sleep(request_pause)
                df = pd.DataFrame({"acq_date": []}) if df is None else df.assign(source=src)
                by_day = df["acq_date"].astype(str)
                for d in (first + pd.Timedelta(days=i) for i in range(n)):
                    _write_day(fps[d], df[by_day == d.isoformat()])

            parts = [pd.read_parquet(fp) for fp in fps.values() if fp.exists()]
            frames += [p for p in parts if len(p)]

        gdf = standardize_firms(pd.concat(frames, ignore_index=True) if frames else None)
        if tagged:
            gdf.insert(0, "event_id", eid)
        label = f"{eid}: " if tagged else ""
        fail = f", {n_failed} failed" if n_failed else ""
        print(f"[{k}/{len(events)}] {label}{len(gdf)} detections "
              f"({n_req} requests{fail}, {n_cached} source-days from cache)")
        out.append(gdf)

    out = [g for g in out if len(g)] or out[:1]
    return pd.concat(out, ignore_index=True)


# ── Per-fire processing ────────────────────────────────────────────────────────

def map_events(func, detections, events=None, *, id_col=None, n_workers: int = 1, **kw):
    """
    Apply a per-fire function to each fire's detections and concatenate the results.

    Processing only: fetching stays sequential in ``fetch_firms``.

    Parameters
    ----------
    func : callable
        ``func(detections_k, **kw)``, or ``func(detections_k, event_k, **kw)`` when
        ``events`` is given (``event_k`` is that fire's one-row GeoDataFrame). Must
        return a DataFrame / GeoDataFrame. With ``n_workers > 1`` it must be importable
        (module-level), not defined in a notebook cell.
    detections : GeoDataFrame
        With an ``event_id`` column.
    events : GeoDataFrame, optional
        Matched to ``event_id`` through ``id_col`` (the index when omitted).
    n_workers : int
        Fires processed in parallel processes when > 1.

    Returns
    -------
    DataFrame or GeoDataFrame
        Results in ``event_id`` order, tagged with ``event_id`` if ``func`` dropped it.
    """
    if "event_id" not in detections.columns:
        raise ValueError("detections has no 'event_id' column.")
    groups = list(detections.groupby("event_id", sort=True))
    if events is not None:
        ev = events.set_index(id_col) if id_col else events
        args = [(g, ev.loc[[eid]]) for eid, g in groups]
    else:
        args = [(g,) for _, g in groups]

    if n_workers > 1:
        with ProcessPoolExecutor(n_workers) as pool:
            futures = [pool.submit(func, *a, **kw) for a in args]
            results = [f.result() for f in futures]
    else:
        results = [func(*a, **kw) for a in args]

    results = [r if "event_id" in r.columns else r.assign(event_id=eid)
               for (eid, _), r in zip(groups, results)]
    return pd.concat(results, ignore_index=True)
