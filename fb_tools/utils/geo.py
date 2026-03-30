"""
Geospatial utility functions for fb_tools.
"""

import numpy as np
import rasterio as rio
import rioxarray as rxr

from geocube.api.core import make_geocube
from rasterio.mask import mask as rio_mask


def is_valid_geom(g):
    """
    Return True if *g* is a non-None, non-empty geometry.

    Parameters
    ----------
    g : shapely geometry
    """
    return g is not None and not g.is_empty


def mask_raster(raster_path, geom, nodata_val=None):
    """
    Clip a raster to a geometry and return the first band as a numpy array.

    Parameters
    ----------
    raster_path : str or Path
        Path to the source raster.
    geom : shapely geometry
        Mask geometry (must be in the same CRS as the raster).
    nodata_val : scalar, optional
        Value to replace with NaN.

    Returns
    -------
    numpy.ndarray
    """
    with rio.open(raster_path) as src:
        out_image, _ = rio_mask(src, [geom], crop=True)
        arr = out_image[0]
        if nodata_val is not None:
            arr = np.ma.masked_equal(arr, nodata_val)
            arr = arr.filled(np.nan)
        return arr


def geom_to_raster_crs(zones_gdf, raster_fp):
    """
    Reproject a GeoDataFrame to match the CRS of a raster file.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
    raster_fp : str or Path

    Returns
    -------
    GeoDataFrame
        Reprojected (or unchanged) GeoDataFrame.
    """
    if isinstance(raster_fp, str):
        with rxr.open_rasterio(raster_fp, masked=True) as da:
            r_crs = da.rio.crs
    else:
        r_crs = raster_fp.rio.crs
    if zones_gdf.crs != r_crs:
        return zones_gdf.to_crs(r_crs)
    return zones_gdf


def clip_raster_inplace(path, mask_gdf):
    """
    Clip a GeoTIFF to the union of *mask_gdf* geometries, overwriting the
    file in place.

    Parameters
    ----------
    path : str or Path
        GeoTIFF to clip.  Must be writable.
    mask_gdf : GeoDataFrame
        Clip geometry.  Reprojected to the raster CRS automatically.

    Returns
    -------
    Path
        Same path that was passed in.
    """
    from pathlib import Path as _Path
    path = _Path(path)
    with rio.open(path) as src:
        shapes = list(mask_gdf.to_crs(src.crs).geometry)
        clipped, clipped_transform = rio_mask(src, shapes, crop=True)
        profile = src.profile.copy()
        descriptions = src.descriptions

    profile.update(
        width=clipped.shape[2],
        height=clipped.shape[1],
        transform=clipped_transform,
    )

    with rio.open(path, "w", **profile) as dst:
        dst.write(clipped)
        dst.descriptions = descriptions

    return path


def lookup_pyrome(container_geom, pyromes_gdf, pyrome_col="Pyrome_ID"):
    """
    Return the pyrome ID with the greatest overlap area with *container_geom*.

    Parameters
    ----------
    container_geom : shapely geometry
        The spatial container (e.g. HUC12, fireshed, POD boundary).
        Must be in the same CRS as *pyromes_gdf*.
    pyromes_gdf : GeoDataFrame
        Pyrome polygons with a pyrome ID column.
    pyrome_col : str
        Column in *pyromes_gdf* holding the pyrome identifier
        (default ``"Pyrome_ID"``).

    Returns
    -------
    scalar
        The pyrome ID (type matches *pyrome_col* dtype) of the dominant pyrome.

    Raises
    ------
    ValueError
        If *container_geom* does not intersect any pyrome.
    """
    import geopandas as gpd

    container = gpd.GeoDataFrame(geometry=[container_geom], crs=pyromes_gdf.crs)
    clipped = gpd.overlay(
        container,
        pyromes_gdf[[pyrome_col, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if clipped.empty:
        raise ValueError("Container does not intersect any features in pyromes_gdf.")

    clipped["_area"] = clipped.geometry.area
    return clipped.loc[clipped["_area"].idxmax(), pyrome_col]


def rasterize(zones, to_img, attr="id", fill_val=-9999):
    """
    Rasterize polygon features onto the grid of a reference raster.

    Parameters
    ----------
    zones : GeoDataFrame
        Input polygon features to rasterize.
    to_img : xarray.DataArray
        Reference raster that defines the output grid and CRS.
    attr : str
        Column in *zones* to burn as pixel values (default ``"id"``).
    fill_val : int or float
        Fill value for pixels outside any polygon (default ``-9999``).

    Returns
    -------
    xarray.DataArray
    """
    zones = zones.to_crs(to_img.rio.crs)

    rasterized = make_geocube(
        vector_data=zones,
        measurements=[attr],
        like=to_img,
        fill=fill_val,
    )

    da = rasterized[attr]
    da = da.rio.write_crs(to_img.rio.crs)
    da = da.rio.write_transform(to_img.rio.transform())
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    return da
