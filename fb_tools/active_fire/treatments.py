"""
Active-fire summaries per treatment polygon (CFT / TWIG records or interaction atoms).

Caveats that this module documents but does not fix:

- CFT planning boundaries can be far larger than the area treated (``ac_flag`` in
  ``treatment_interactions``); screen them before summarising.
- Passing raw CFT and TWIG records together double counts their overlaps. The atomic
  interactions layer does not overlap.
- TWIG's fallback fields leave pile vs broadcast burning uncertain.
- A VIIRS pixel (~375 m, ~35 acres at nadir) is larger than many units, so a detection
  "on" a small unit says fire was within a pixel of it, not that the unit burned.
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from .footprints import footprint_boxes
from .timeutil import assign_overpass


def _stat(name: str):
    """A pandas aggregation for ``"mean"``, ``"max"``, … or a percentile such as ``"p90"``."""
    if name.startswith("p") and name[1:].isdigit():
        q = int(name[1:]) / 100
        return lambda s: s.quantile(q)
    return name


def treatment_detections(
    treatments,
    detections,
    *,
    id_col="SOURCE_UID",
    keep_cols=("ACTIVITY", "YEAR_COMP"),
    method: str = "footprint",
    buffer_m: float = 0,
    start=None,
    end=None,
    frp_stats=("mean",),
):
    """
    Summarise the detections on each treatment polygon.

    Treatments are assumed to predate the fire: the detections passed in (optionally
    limited by ``start`` / ``end``) define the period, and treatment dates set no
    window.

    Parameters
    ----------
    treatments : GeoDataFrame
        Polygons. The defaults match the harmonized CFT/TWIG schema of
        ``treatment_interactions``; any table works through the column arguments.
    detections : GeoDataFrame
        Detections in the shared schema (points), or ``footprint_boxes`` output.
    id_col : str or None
        Polygon id column; the index when None or absent.
    keep_cols : sequence of str
        Treatment attributes carried through unchanged (those present).
    method : {"footprint", "centre"}
        ``"footprint"``: a detection counts when its scan × track box overlaps the
        polygon. ``"centre"``: only when its centre falls inside.
    buffer_m : float
        Buffer applied to the polygons first.
    start, end : date-like, optional
        Inclusive UTC days limiting the detections.
    frp_stats : sequence of str
        Statistics of the attributed detections' FRP, per instrument: pandas names
        (``"mean"``, ``"max"``, ``"sum"``, ``"median"``) or percentiles (``"p90"``).
        FRP is the detection's whole-pixel value, not weighted by overlap.

    Returns
    -------
    GeoDataFrame
        One row per polygon, in the treatments' CRS (UTM if that is geographic):
        id, ``keep_cols``, ``detected``, ``n_detections``, ``n_overpasses``,
        ``first_detection`` / ``last_detection`` (UTC), ``satellites``,
        ``frp_<stat>_<instrument>``, and for ``"footprint"`` ``cover_fraction`` — the
        share of the (buffered) polygon under the union of the boxes.
    """
    import shapely

    if method not in ("footprint", "centre"):
        raise ValueError(f"method must be 'footprint' or 'centre'; got {method!r}.")
    trt = treatments.to_crs(treatments.estimate_utm_crs()) if treatments.crs.is_geographic else treatments
    zones = (trt.geometry.buffer(buffer_m) if buffer_m else trt.geometry).to_numpy()

    det = detections
    t = det["acq_datetime"]
    if start is not None:
        det = det[t >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        det = det[det["acq_datetime"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    if "overpass_id" not in det:
        det = assign_overpass(det)
    is_point = det.geom_type.eq("Point").all()
    if method == "footprint":
        det = footprint_boxes(det, crs=trt.crs, instruments=None) if is_point else det.to_crs(trt.crs)
    else:
        det = det.to_crs(trt.crs) if is_point else det.set_geometry(det.geometry.centroid).to_crs(trt.crs)

    dgeom = det.geometry.to_numpy()
    iz, idt = det.sindex.query(zones, predicate="intersects")
    if method == "footprint":
        inter = shapely.intersection(zones[iz], dgeom[idt])
        keep = shapely.area(inter) > 0
        iz, idt, inter = iz[keep], idt[keep], inter[keep]

    d = det.iloc[idt]
    pairs = pd.DataFrame({
        "pos": iz, "t": d["acq_datetime"].to_numpy(), "overpass_id": d["overpass_id"].to_numpy(),
        "satellite": d["satellite"].to_numpy(), "instrument": d["instrument"].to_numpy(),
        "frp": pd.to_numeric(d["frp"], errors="coerce").to_numpy(float),
    })
    g = pairs.groupby("pos")
    out = g.agg(n_detections=("t", "size"), n_overpasses=("overpass_id", "nunique"),
                first_detection=("t", "min"), last_detection=("t", "max"),
                satellites=("satellite", lambda s: "|".join(sorted(set(s.dropna())))))
    if method == "footprint":
        area = pd.Series(inter, index=iz).groupby(level=0).agg(lambda s: shapely.union_all(s.to_numpy()).area)
        out["cover_fraction"] = area / shapely.area(zones)[area.index]
    frp = pairs.groupby(["pos", "instrument"])["frp"].agg([_stat(s) for s in frp_stats])
    frp.columns = list(frp_stats)
    frp = frp.unstack("instrument")
    frp.columns = [f"frp_{stat}_{inst}" for stat, inst in frp.columns]
    out = out.join(frp).reindex(np.arange(len(trt)))

    out["n_detections"] = out["n_detections"].fillna(0).astype(int)
    out["n_overpasses"] = out["n_overpasses"].fillna(0).astype(int)
    out.insert(0, "detected", out["n_detections"] > 0)
    if method == "footprint":
        out["cover_fraction"] = out["cover_fraction"].fillna(0.0)

    has_id = id_col is not None and id_col in trt.columns
    head = pd.DataFrame({id_col if has_id else "treatment_id": (trt[id_col] if has_id else trt.index).to_numpy()})
    for c in keep_cols:
        if c in trt.columns and c not in head:
            head[c] = trt[c].to_numpy()
    res = gpd.GeoDataFrame(pd.concat([head, out.reset_index(drop=True)], axis=1),
                           geometry=trt.geometry.to_numpy(), crs=trt.crs)
    print(f"treatments: {int(res['detected'].sum())} of {len(res)} polygons with detections ({method})")
    return res
