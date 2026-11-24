"""
Regular grids for active-fire footprints: overlap-weighted FRP and first detection.

Ported from potr5-fb ``01c_VIIRS-Grid.ipynb`` (``aligned_grid``, ``aggregate_frp``).
The FRP allocation is potr5's, kept until it is evaluated against the alternatives
(docs/ACTIVE_FIRE_PLAN.md, note A).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ..utils.geo import snap_origin, snapped_grid
from ..utils.io import raster_write_kwargs
from .timeutil import assign_overpass, day_of


def afd_grid(aoi, res_m: float = 480, crs="EPSG:5070", snap=None, clip: bool = True):
    """
    Regular grid of square cells covering an AOI.

    Parameters
    ----------
    aoi : GeoDataFrame or GeoSeries
    res_m : float
        Cell size, metres. 480 = 16 LANDFIRE/TreeMap pixels, close to the median
        off-nadir VIIRS pixel length.
    crs : any
        Projected grid CRS. Ignored when ``snap`` is a raster (its CRS is used).
    snap : None, "landfire" or path-like
        Where cell edges fall: multiples of ``res_m``; LANDFIRE pixel edges
        (EPSG:5070); or the pixel edges of a raster such as TreeMap.
    clip : bool
        Keep only cells that intersect the AOI.

    Returns
    -------
    GeoDataFrame
        ``grid_id, row, col, geometry``. ``row`` / ``col`` index the unclipped grid
        (row 0 at the top) and are what :func:`to_raster` uses.
    """
    import shapely

    crs, origin = snap_origin(snap, crs, res_m)
    aoi = aoi.geometry.to_crs(crs)
    transform, (nrows, ncols) = snapped_grid(aoi, crs, float(res_m), origin)
    row, col = np.divmod(np.arange(nrows * ncols), ncols)
    x0, y1 = transform.c + col * res_m, transform.f - row * res_m
    grid = gpd.GeoDataFrame({"row": row, "col": col},
                            geometry=shapely.box(x0, y1 - res_m, x0 + res_m, y1), crs=crs)
    if clip:
        grid = grid.iloc[np.sort(grid.sindex.query(aoi.union_all(), predicate="intersects"))]
    grid = grid.reset_index(drop=True)
    grid.insert(0, "grid_id", np.arange(len(grid)))
    return grid


def _pieces(boxes, grid):
    """Footprint × cell intersections: box position, cell position, intersection geometry (area > 0)."""
    import shapely

    if boxes.crs != grid.crs:
        boxes = boxes.to_crs(grid.crs)
    ib, ic = grid.sindex.query(boxes.geometry, predicate="intersects")
    geom = shapely.intersection(boxes.geometry.values[ib], grid.geometry.values[ic])
    keep = shapely.area(geom) > 0          # drops edge and corner touches
    return ib[keep], ic[keep], geom[keep]


def gridded_frp(boxes, grid, *, tz=None, drop_zero_frp: bool = True, percentiles=(90, 95, 97, 99)):
    """
    Overlap-weighted FRP statistics per grid cell (potr5 ``aggregate_frp``).

    For every footprint piece inside a cell::

        fraction = overlap_area / pixel_area
        frp_fr   = frp_wkm2 * fraction          # frp_wkm2 = frp / pixel_area

    and ``frp_fr`` is summarised per cell. Pass one instrument at a time: FRP is not
    comparable across instruments.

    Parameters
    ----------
    boxes : GeoDataFrame
        From ``footprint_boxes`` (needs ``pix_area_km2``, ``frp_wkm2``).
    grid : GeoDataFrame
        From ``afd_grid``.
    tz : str, optional
        Time zone for the day columns; UTC days if omitted.
    drop_zero_frp : bool
        Drop detections with FRP ≤ 0 or missing before gridding, as potr5 did.
        When False, a missing FRP contributes ``frp_fr = 0``.
    percentiles : sequence of int

    Returns
    -------
    GeoDataFrame
        Cells with at least one detection: ``afd_count``, ``unique_days``, ``overlap``
        (sum of fractions), ``frp_csum/max/min/mean/pXX/first``, ``day_max_frp`` and
        ``dt_max_frp`` (day and time of the largest ``frp_fr``), ``first_obs_date``,
        ``last_obs_date``, ``t4_*`` / ``t5_*`` brightness temperatures, and the
        day/night split (``day_count``, ``frp_max_day``, …; 0 where there is none).
    """
    if drop_zero_frp:
        ok = boxes["frp"].gt(0).fillna(False)
        print(f"gridded_frp: dropped {(~ok).sum()} of {len(boxes)} detections with FRP <= 0 or missing")
        boxes = boxes[ok]
    import shapely

    ib, ic, geom = _pieces(boxes, grid)
    b = boxes.iloc[ib]
    fraction = shapely.area(geom) / (b["pix_area_km2"].to_numpy(float) * 1e6)
    o = pd.DataFrame({
        "grid_id": grid["grid_id"].to_numpy()[ic],
        "fraction": fraction,
        "frp_fr": np.nan_to_num(b["frp_wkm2"].to_numpy(float) * fraction),
        "t": b["acq_datetime"].to_numpy(),
        "day": day_of(b["acq_datetime"], tz).to_numpy(),
        "dn": b["daynight"].astype(str).str[:1].str.upper().to_numpy(),
        "t4": pd.to_numeric(b["bright_ti4"], errors="coerce").to_numpy(float),
        "t5": pd.to_numeric(b["bright_ti5"], errors="coerce").to_numpy(float),
    }).sort_values("t", kind="stable", ignore_index=True)      # time order, so "first" is the earliest

    def frp_stats(g, suffix=""):
        s = g["frp_fr"].agg(["sum", "max", "min", "mean", "first"])
        s.columns = [f"frp_{'csum' if c == 'sum' else c}{suffix}" for c in s.columns]
        qs = [p / 100 for p in percentiles]
        q = g["frp_fr"].quantile(qs).unstack().reindex(columns=qs)     # reindex: an empty group has no columns
        q.columns = [f"frp_p{p}{suffix}" for p in percentiles]
        return s.join(q)

    g = o.groupby("grid_id")
    agg = g.agg(afd_count=("frp_fr", "count"), unique_days=("day", "nunique"), overlap=("fraction", "sum"))
    agg = agg.join(frp_stats(g))
    imax = g["frp_fr"].idxmax()
    agg["day_max_frp"] = o.loc[imax, "day"].to_numpy()
    agg["dt_max_frp"] = o.loc[imax, "t"].to_numpy()
    agg = agg.join(g.agg(first_obs_date=("day", "min"), last_obs_date=("day", "max"),
                         t4_max=("t4", "max"), t4_mean=("t4", "mean"),
                         t5_max=("t5", "max"), t5_mean=("t5", "mean")))
    for code, name in (("D", "day"), ("N", "night")):
        gd = o[o["dn"] == code].groupby("grid_id")
        part = gd.size().rename(f"{name}_count").to_frame().join(frp_stats(gd, f"_{name}").drop(columns=f"frp_min_{name}"))
        agg = agg.join(part.reindex(agg.index, fill_value=0))
    return grid.merge(agg.reset_index(), on="grid_id", how="inner")


def first_detection_grid(boxes, grid, rule: str = "fraction", min_fraction: float = 0.5, gap_min: float = 30):
    """
    Earliest overpass that detected fire in each cell. Not interpolated.

    Parameters
    ----------
    boxes : GeoDataFrame
        From ``footprint_boxes``. Overpasses come from ``assign_overpass`` (per
        satellite, split at gaps > ``gap_min``) unless ``overpass_id`` is present.
    grid : GeoDataFrame
    rule : {"fraction", "any", "centre"}
        ``"fraction"``: the earliest overpass whose unioned boxes cover at least
        ``min_fraction`` of the cell (one 375 m box covers at most ~61% of a 480 m
        cell). ``"any"``: any overlap, the potr5 ``first_obs_date`` rule.
        ``"centre"``: the cell holding the detection centre.
    min_fraction : float
    gap_min : float

    Returns
    -------
    GeoDataFrame
        Cells with a first detection: ``first_detection`` (UTC overpass time),
        ``overpass_id``, ``satellite`` and, for ``"fraction"``, the covered
        ``fraction``.
    """
    import shapely

    if rule not in ("fraction", "any", "centre"):
        raise ValueError(f"rule must be 'fraction', 'any' or 'centre'; got {rule!r}.")
    if "overpass_id" not in boxes:
        boxes = assign_overpass(boxes, gap_min=gap_min)
    if boxes.crs != grid.crs:
        boxes = boxes.to_crs(grid.crs)

    if rule == "centre":
        ib, ic = grid.sindex.query(boxes.geometry.centroid, predicate="intersects")
        geom = None
    else:
        ib, ic, geom = _pieces(boxes, grid)
    o = pd.DataFrame({"grid_id": grid["grid_id"].to_numpy()[ic],
                      "overpass_id": boxes["overpass_id"].to_numpy()[ib]})

    if rule == "fraction":
        o["area"] = shapely.area(geom)
        cell_area = pd.Series(grid.geometry.area.to_numpy(), grid["grid_id"].to_numpy())
        g = o.groupby(["grid_id", "overpass_id"])["area"]
        cover = g.sum().to_frame("area").assign(n=g.size())
        need = min_fraction * cell_area.reindex(cover.index.get_level_values(0)).to_numpy()
        # Boxes of one overpass overlap, so a summed area overstates coverage: union the
        # multi-piece groups, but only those whose sum could reach the threshold.
        multi = cover.index[(cover["n"] > 1) & (cover["area"] >= need)]
        if len(multi):
            key = pd.MultiIndex.from_frame(o[["grid_id", "overpass_id"]])
            sel = key.isin(multi)
            cover.loc[multi, "area"] = (pd.Series(geom[sel], index=key[sel]).groupby(level=[0, 1])
                                        .agg(lambda s: shapely.union_all(s.to_numpy()).area).reindex(multi))
        cover["fraction"] = cover["area"] / (need / min_fraction)
        o = cover[cover["area"] >= need - 1e-6].reset_index()[["grid_id", "overpass_id", "fraction"]]

    op = boxes.groupby("overpass_id").agg(first_detection=("overpass_time", "first"), satellite=("satellite", "first"))
    o = o.join(op, on="overpass_id").sort_values("first_detection", kind="stable")
    first = o.drop_duplicates("grid_id")
    cols = ["grid_id", "first_detection", "overpass_id", "satellite"] + (["fraction"] if rule == "fraction" else [])
    return grid.merge(first[cols], on="grid_id", how="inner")


def to_raster(grid_gdf, column: str, out_fp, *, nodata=np.nan, t0=None) -> Path:
    """
    Write one column of a grid GeoDataFrame to a GeoTIFF.

    The raster spans the cells present (their ``row`` / ``col`` extent). A datetime
    column is written as float days since ``t0`` (default: the column's first UTC
    midnight), with ``t0`` recorded in the ``T0`` tag.

    Returns
    -------
    Path
    """
    import rasterio
    from rasterio.transform import from_origin

    out_fp = Path(out_fp)
    vals = grid_gdf[column]
    tags = {}
    if pd.api.types.is_datetime64_any_dtype(vals):
        vals = vals.dt.tz_convert("UTC").dt.tz_localize(None) if vals.dt.tz is not None else vals
        t0 = pd.Timestamp(t0).tz_localize(None) if t0 is not None else vals.min().normalize()
        vals, tags = (vals - t0) / pd.Timedelta(days=1), {"T0": t0.isoformat(), "UNITS": f"days since {t0.isoformat()}"}
    row, col = grid_gdf["row"].to_numpy(), grid_gdf["col"].to_numpy()
    b = grid_gdf.geometry.bounds
    res = float(b["maxx"].iloc[0] - b["minx"].iloc[0])
    arr = np.full((row.max() - row.min() + 1, col.max() - col.min() + 1), nodata, dtype="float64")
    arr[row - row.min(), col - col.min()] = pd.to_numeric(vals, errors="coerce").to_numpy(float)
    arr[np.isnan(arr)] = nodata
    with rasterio.open(out_fp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float64", crs=grid_gdf.crs, nodata=nodata,
                       transform=from_origin(b["minx"].min(), b["maxy"].max(), res, res),
                       **raster_write_kwargs()) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, column)
        dst.update_tags(**tags)
    return out_fp
