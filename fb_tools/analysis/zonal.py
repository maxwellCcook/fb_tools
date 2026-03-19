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


def _make_zone_arr(zones_gdf, reference_da, id_col, fill_val=-9999):
    """
    Burn zone polygon IDs onto the reference grid and return a raw int64 array.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Polygons with *id_col* and geometry. Must already be in the same CRS
        as *reference_da* (call geom_to_raster_crs() beforehand if needed).
    reference_da : xarray.DataArray
        Reference grid for rasterization.
    id_col : str
        Numeric zone-ID column to burn.
    fill_val : int
        Fill value for outside-polygon pixels (default -9999).

    Returns
    -------
    numpy.ndarray
        int64 array, shape (H, W), values = zone IDs or fill_val.
    """
    zone_da = rasterize(zones_gdf[[id_col, "geometry"]], to_img=reference_da,
                        attr=id_col, fill_val=fill_val)
    arr = zone_da.values.squeeze().astype(np.int64)
    del zone_da
    gc.collect()
    return arr


def zonal_categorical(zones_gdf, raster_arr, reference_da, id_col,
                      nodata=-9999, zone_arr=None):
    """
    Compute per-zone pixel counts and percent cover for each categorical class.

    Parameters
    ----------
    zones_gdf : GeoDataFrame or None
        Treatment polygons. Must include *id_col* and geometry. Ignored when
        *zone_arr* is provided; pass ``None`` in that case.
    raster_arr : numpy.ndarray
        2-D integer array of categorical values (e.g., binned FL class,
        crown state code). Shape must match *reference_da* (and *zone_arr*
        when provided).
    reference_da : xarray.DataArray
        Single-band 2-D DataArray used as the rasterization reference grid.
        Ignored when *zone_arr* is provided.
    id_col : str
        Column in *zones_gdf* to use as zone identifier. Values must be
        numeric (int or float) so they can be burned as raster pixel values.
    nodata : int
        Pixel value in the zone grid (and in *raster_arr*) that indicates
        outside-polygon or no-data (default -9999).
    zone_arr : numpy.ndarray, optional
        Pre-computed int64 zone grid (H×W) from a prior :func:`_make_zone_arr`
        call. When provided, the rasterization step is skipped entirely —
        *zones_gdf* and *reference_da* are not used.

    Returns
    -------
    pandas.DataFrame
        Long-form with columns: [id_col, 'class_val', 'pixel_count', 'pct_cover'].
        One row per (zone_id, class_val) combination. Zones with zero valid
        pixels are omitted.
    """
    if zone_arr is None:
        zone_da = rasterize(zones_gdf[[id_col, "geometry"]], to_img=reference_da,
                            attr=id_col, fill_val=nodata)
        zone_arr = zone_da.values.squeeze().astype(np.int64)
        del zone_da
        gc.collect()
    else:
        zone_arr = np.asarray(zone_arr, dtype=np.int64)

    val_arr = np.asarray(raster_arr).squeeze()

    # Flatten to only the pixels that are inside a polygon AND not nodata in
    # the value raster. For large landscapes with sparse treatment polygons
    # this dramatically reduces the per-zone loop iterations.
    valid_mask = (zone_arr != nodata) & (val_arr != nodata)
    valid_zones = zone_arr[valid_mask]
    valid_vals  = val_arr[valid_mask]

    rows = []
    for zone_id in np.unique(valid_zones):
        zone_mask = valid_zones == zone_id
        vals  = valid_vals[zone_mask]
        total = len(vals)
        if total == 0:
            continue
        classes, counts = np.unique(vals, return_counts=True)
        for cls, cnt in zip(classes, counts):
            rows.append({
                id_col:        int(zone_id),
                "class_val":   int(cls),
                "pixel_count": int(cnt),
                "pct_cover":   float(cnt / total * 100),
            })

    return pd.DataFrame(rows)


def zonal_continuous(zones_gdf, raster_arr, reference_da, id_col,
                     stat="mean", nodata_raster=None, scale_factor=1.0,
                     zone_arr=None):
    """
    Compute a per-zone summary statistic for a continuous raster.

    Parameters
    ----------
    zones_gdf : GeoDataFrame or None
        Treatment polygons. Must include *id_col* and geometry. Ignored when
        *zone_arr* is provided; pass ``None`` in that case.
    raster_arr : numpy.ndarray
        2-D float array of continuous values (e.g., SDI × 100, flame length).
        Shape must match *reference_da* (and *zone_arr* when provided).
    reference_da : xarray.DataArray
        Single-band 2-D DataArray used as the rasterization reference grid.
        Ignored when *zone_arr* is provided.
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
    zone_arr : numpy.ndarray, optional
        Pre-computed int64 zone grid (H×W) from a prior :func:`_make_zone_arr`
        call. When provided, the rasterization step is skipped entirely —
        *zones_gdf* and *reference_da* are not used.

    Returns
    -------
    pandas.DataFrame
        One row per zone with columns [id_col, stat].
    """
    _FILL = -9999

    if zone_arr is None:
        zone_da = rasterize(zones_gdf[[id_col, "geometry"]], to_img=reference_da,
                            attr=id_col, fill_val=_FILL)
        zone_arr = zone_da.values.squeeze().astype(np.int64)
        del zone_da
        gc.collect()
    else:
        zone_arr = np.asarray(zone_arr, dtype=np.int64)

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

    # Flatten to only polygon pixels — skips the large nodata regions
    valid_mask  = zone_arr != _FILL
    valid_zones = zone_arr[valid_mask]
    valid_vals  = val_arr[valid_mask]

    rows = []
    for zone_id in np.unique(valid_zones):
        zone_mask = valid_zones == zone_id
        vals = valid_vals[zone_mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        rows.append({id_col: int(zone_id), stat: float(fn(vals))})

    return pd.DataFrame(rows)
