"""
Quality screening for active-fire detections in the shared schema (``firms.STD_COLS``).

Every step returns ``(kept, dropped)``, keeps the input's index and row order, and
prints what it dropped.
"""
import numpy as np
import pandas as pd

VIIRS_INTERSCAN_RADIUS_M = 187.5   # half the nominal 375 m pixel
VIIRS_INTERSCAN_TOL_S = 7


def _split(gdf, drop, label: str):
    drop = np.asarray(drop, dtype=bool)
    print(f"{label}: dropped {drop.sum()} of {len(gdf)}")
    return gdf[~drop].copy(), gdf[drop].copy()


def drop_low_confidence(gdf):
    """Drop detections whose ``confidence_norm`` is ``low`` (VIIRS ``l``, MODIS < 30)."""
    return _split(gdf, gdf["confidence_norm"].eq("low").fillna(False), "low confidence")


def drop_static_sources(gdf):
    """
    Keep ``type == 0`` (presumed vegetation fire); drop 1 volcano, 2 other static land
    source, 3 offshore. NRT rows carry no ``type`` and cannot be screened — they are
    kept, and the count is printed.
    """
    typ = pd.to_numeric(gdf["type"], errors="coerce") if "type" in gdf else pd.Series(np.nan, gdf.index)
    if typ.isna().any():
        print(f"static sources: {typ.isna().sum()} detections have no `type` (NRT); not screened")
    return _split(gdf, typ.notna() & (typ != 0), "static sources")


def remove_interscan_duplicates(
    gdf,
    *,
    method=None,
    radius_m: float = VIIRS_INTERSCAN_RADIUS_M,
    time_tol_s: float = VIIRS_INTERSCAN_TOL_S,
    group_cols=None,
    crs="EPSG:5070",
):
    """
    Remove duplicate detections from overlapping adjacent scans, keeping the higher FRP.

    ``method="firms"`` follows potr5 ``correct_interscan_duplicates_firms``: within a
    group, two detections are duplicates when they share ``scan`` and ``track``, are
    ≤ ``time_tol_s`` apart and ≤ ``radius_m`` apart. Pairs are found with
    ``cKDTree.query_pairs`` and resolved in time order; a detection already dropped
    cannot drop another. FRP is never summed or averaged.

    Only VIIRS is screened. Other instruments pass through, with a printed count.

    Parameters
    ----------
    gdf : GeoDataFrame
    method : {"firms", "swath"}, optional
        Default is chosen from ``product``. ``"swath"`` arrives with ``swath.py``.
    radius_m, time_tol_s : float
    group_cols : sequence of str, optional
        Default ``("satellite", "event_id")`` when ``event_id`` is present, else
        ``("satellite",)``.
    crs : any
        Projected CRS the distances are measured in.

    Returns
    -------
    kept, dropped : GeoDataFrame
    """
    from scipy.spatial import cKDTree

    if method is None:
        method = "firms" if gdf["product"].eq("FIRMS").all() else "swath"
    if method != "firms":
        raise NotImplementedError("method='swath' is not built yet (swath.py).")
    if group_cols is None:
        group_cols = ("satellite", "event_id") if "event_id" in gdf else ("satellite",)

    viirs = gdf["instrument"].eq("VIIRS").to_numpy()
    if (~viirs).any():
        print(f"interscan: {(~viirs).sum()} non-VIIRS detections not screened")

    pts = gdf.geometry.to_crs(crs)
    xy = np.column_stack([pts.x, pts.y])
    t = gdf["acq_datetime"].to_numpy("datetime64[ns]").astype("int64") / 1e9
    scan, track = gdf["scan"].to_numpy(float), gdf["track"].to_numpy(float)
    frp = gdf["frp"].fillna(-np.inf).to_numpy(float)   # a missing FRP always loses
    drop = np.zeros(len(gdf), dtype=bool)

    pos = pd.Series(np.arange(len(gdf)))[viirs]
    keys = [gdf[c].to_numpy()[viirs] for c in group_cols]
    for _, g in pos.groupby(keys, dropna=False):
        if len(g) < 2:
            continue
        ix = g.to_numpy()
        ix = ix[np.argsort(t[ix], kind="stable")]          # time order; pairs come out i < j
        pairs = cKDTree(xy[ix]).query_pairs(radius_m, output_type="ndarray")
        if not len(pairs):
            continue
        i, j = ix[pairs[:, 0]], ix[pairs[:, 1]]
        dup = (scan[i] == scan[j]) & (track[i] == track[j]) & (np.abs(t[i] - t[j]) <= time_tol_s)
        order = np.lexsort((pairs[dup, 1], pairs[dup, 0]))
        for a, b in zip(i[dup][order], j[dup][order]):
            if not (drop[a] or drop[b]):
                drop[b if frp[a] >= frp[b] else a] = True

    return _split(gdf, drop, "interscan duplicates")


def apply_qa(gdf, *, low_conf: bool = True, interscan: bool = True, static: bool = True, **interscan_kw):
    """
    Chain the screens: low confidence → interscan duplicates → static sources.

    Returns
    -------
    kept, dropped : GeoDataFrame
        ``dropped`` carries a ``qa_reason`` column.
    """
    steps = [
        (low_conf, "low_confidence", drop_low_confidence, {}),
        (interscan, "interscan", remove_interscan_duplicates, interscan_kw),
        (static, "static_source", drop_static_sources, {}),
    ]
    kept, dropped = gdf, [gdf.iloc[:0].assign(qa_reason=pd.Series(dtype="object"))]
    for on, reason, step, kw in steps:
        if on:
            kept, out = step(kept, **kw)
            dropped.append(out.assign(qa_reason=reason))
    return kept.copy(), pd.concat(dropped)
