"""
Zonal statistics helpers for treatment-level fire behavior analysis.

Uses rasterize() from utils.geo to burn treatment polygon IDs onto the
raster grid, then applies numpy-based groupby operations to compute
per-zone statistics without requiring rasterstats.
"""

import gc

import numpy as np
import pandas as pd

from ..utils.geo import rasterize


def zonal_categorical(zones_gdf, raster_arr, reference_da, id_col, nodata=-9999):
    """
    Compute per-zone pixel counts and percent cover for each categorical class.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Treatment polygons. Must include *id_col* and geometry.
    raster_arr : numpy.ndarray
        2-D integer array of categorical values (e.g., binned FL class,
        crown state code). Shape must match *reference_da*.
    reference_da : xarray.DataArray
        Single-band 2-D DataArray used as the rasterization reference grid.
    id_col : str
        Column in *zones_gdf* to use as zone identifier. Values must be
        numeric (int or float) so they can be burned as raster pixel values.
    nodata : int
        Pixel value in the zone grid (and in *raster_arr*) that indicates
        outside-polygon or no-data (default -9999).

    Returns
    -------
    pandas.DataFrame
        Long-form with columns: [id_col, 'class_val', 'pixel_count', 'pct_cover'].
        One row per (zone_id, class_val) combination. Zones with zero valid
        pixels are omitted.
    """
    zone_da = rasterize(zones_gdf[[id_col, "geometry"]], to_img=reference_da,
                        attr=id_col, fill_val=nodata)
    zone_arr = zone_da.values.squeeze().astype(np.int64)
    del zone_da
    gc.collect()

    val_arr = np.asarray(raster_arr).squeeze()

    # Mask: pixel is inside a polygon AND value raster is not nodata
    valid_mask = (zone_arr != nodata) & (val_arr != nodata)

    rows = []
    for zone_id in np.unique(zone_arr[zone_arr != nodata]):
        zone_mask = (zone_arr == zone_id) & valid_mask
        vals = val_arr[zone_mask]
        total = len(vals)
        if total == 0:
            continue
        classes, counts = np.unique(vals, return_counts=True)
        for cls, cnt in zip(classes, counts):
            rows.append({
                id_col: int(zone_id),
                "class_val": int(cls),
                "pixel_count": int(cnt),
                "pct_cover": float(cnt / total * 100),
            })

    return pd.DataFrame(rows)


def zonal_continuous(zones_gdf, raster_arr, reference_da, id_col,
                     stat="mean", nodata_raster=None, scale_factor=1.0):
    """
    Compute a per-zone summary statistic for a continuous raster.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Treatment polygons. Must include *id_col* and geometry.
    raster_arr : numpy.ndarray
        2-D float array of continuous values (e.g., SDI × 100, flame length).
        Shape must match *reference_da*.
    reference_da : xarray.DataArray
        Single-band 2-D DataArray used as the rasterization reference grid.
    id_col : str
        Column in *zones_gdf* to use as zone identifier (numeric).
    stat : str
        Statistic to compute per zone: ``'mean'``, ``'sum'``, ``'count'``,
        or ``'std'`` (default ``'mean'``).
    nodata_raster : scalar, optional
        Value in *raster_arr* to exclude before computing stats. NaN values
        are always excluded regardless of this setting.
    scale_factor : float
        Divide raster values by this factor before computing the statistic.
        Use ``100.0`` to convert SDI × 100 int16 storage back to float SDI.

    Returns
    -------
    pandas.DataFrame
        One row per zone with columns [id_col, stat].
    """
    _FILL = -9999

    zone_da = rasterize(zones_gdf[[id_col, "geometry"]], to_img=reference_da,
                        attr=id_col, fill_val=_FILL)
    zone_arr = zone_da.values.squeeze().astype(np.int64)
    del zone_da
    gc.collect()

    val_arr = np.asarray(raster_arr, dtype=np.float32).squeeze()
    if scale_factor != 1.0:
        val_arr = val_arr / scale_factor

    # Mark nodata as NaN
    if nodata_raster is not None:
        val_arr = np.where(val_arr == nodata_raster, np.nan, val_arr)

    _STAT_FUNCS = {
        "mean":  np.nanmean,
        "sum":   np.nansum,
        "count": lambda x: np.sum(~np.isnan(x)),
        "std":   np.nanstd,
    }
    if stat not in _STAT_FUNCS:
        raise ValueError(f"stat must be one of {list(_STAT_FUNCS)}; got {stat!r}")
    fn = _STAT_FUNCS[stat]

    rows = []
    for zone_id in np.unique(zone_arr[zone_arr != _FILL]):
        zone_mask = zone_arr == zone_id
        vals = val_arr[zone_mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        rows.append({id_col: int(zone_id), stat: float(fn(vals))})

    return pd.DataFrame(rows)
