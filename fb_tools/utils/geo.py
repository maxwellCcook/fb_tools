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


def get_pyrome_id(
    location,
    pyromes,
    pyrome_col: str = "Pyrome_ID",
) -> "str | int":
    """
    Return the pyrome ID for a given point location or spatial container.

    Convenience wrapper around :func:`lookup_pyrome` that accepts flexible
    input types for both the location and the pyromes layer — no pre-loading
    or CRS wrangling required.

    Parameters
    ----------
    location : tuple, str, Path, GeoDataFrame, or shapely geometry
        The location to look up.  Accepted forms:

        ``(lat, lon)``
            A ``(latitude, longitude)`` tuple in **WGS84** decimal degrees.
            The function queries which pyrome the point falls within.  If the
            point lies on a shared boundary, the first intersecting pyrome is
            returned.  If it falls in a gap, the nearest pyrome is used.

        ``str`` or ``pathlib.Path``
            Path to a vector file (shapefile, GeoPackage, etc.) to read as
            the area of interest.  The file is dissolved to a single polygon
            and the dominant pyrome (greatest overlap area) is returned.

        ``GeoDataFrame``
            An already-loaded polygon GeoDataFrame.  Dissolved to a single
            polygon before the overlap query.  Any CRS; reprojected
            automatically.

        ``shapely Point`` or polygon geometry
            A raw Shapely geometry.  Assumed to be in **WGS84** unless the
            pyromes layer is also in WGS84 (which it typically is).  For a
            Point, the containment query is used; for a polygon, the greatest-
            overlap query is used.

    pyromes : str, Path, or GeoDataFrame
        NIFC pyrome polygons.  Accepted forms:

        ``str`` or ``pathlib.Path``
            Path to a vector file (shapefile, GeoPackage, etc.) that will be
            read with ``geopandas.read_file()``.

        ``GeoDataFrame``
            An already-loaded pyrome layer.

    pyrome_col : str
        Column in the pyromes layer holding the pyrome identifier.
        Default ``"Pyrome_ID"``.

    Returns
    -------
    str or int
        Pyrome ID (type matches the ``pyrome_col`` column dtype).

    Raises
    ------
    ValueError
        If the location does not intersect or fall within any pyrome, and no
        nearest fallback can be found.
    TypeError
        If ``location`` is not one of the recognised input types.

    Examples
    --------
    Point lookup by lat/lon:

    >>> get_pyrome_id((39.7, -105.2), "data/pyromes/pyromes.shp")
    42

    AOI lookup from a shapefile:

    >>> get_pyrome_id("data/aoi/huc12_030601.shp", pyromes_gdf)
    42

    AOI from a GeoDataFrame:

    >>> get_pyrome_id(my_huc_gdf, "data/pyromes/pyromes.shp")
    42
    """
    import geopandas as gpd
    from pathlib import Path as _Path
    from shapely.geometry import Point, mapping

    # ── Load pyromes ──────────────────────────────────────────────────────────
    if isinstance(pyromes, (str, _Path)):
        pyromes_gdf = gpd.read_file(pyromes)
    elif hasattr(pyromes, "geometry"):          # GeoDataFrame duck-type check
        pyromes_gdf = pyromes
    else:
        raise TypeError(
            f"'pyromes' must be a file path or GeoDataFrame, got {type(pyromes).__name__}"
        )

    # ── Normalise location → (geom, is_point, crs_or_None) ───────────────────
    is_point  = False
    input_crs = None

    if isinstance(location, (list, tuple)) and len(location) == 2:
        # (lat, lon) → WGS84 Point
        lat, lon = float(location[0]), float(location[1])
        geom      = Point(lon, lat)
        is_point  = True
        input_crs = "EPSG:4326"

    elif isinstance(location, (str, _Path)):
        aoi = gpd.read_file(location)
        try:
            geom = aoi.dissolve().geometry.iloc[0]
        except AttributeError:
            geom = aoi.geometry.unary_union
        input_crs = aoi.crs
        is_point  = False

    elif hasattr(location, "geometry"):         # GeoDataFrame
        try:
            geom = location.dissolve().geometry.iloc[0]
        except AttributeError:
            geom = location.geometry.unary_union
        input_crs = location.crs
        is_point  = geom.geom_type == "Point"

    elif hasattr(location, "geom_type"):        # raw Shapely geometry
        geom      = location
        input_crs = "EPSG:4326"                 # assume WGS84 for bare geometries
        is_point  = geom.geom_type == "Point"

    else:
        raise TypeError(
            f"Unrecognised 'location' type: {type(location).__name__}. "
            "Pass a (lat, lon) tuple, file path, GeoDataFrame, or Shapely geometry."
        )

    # ── Reproject to pyromes CRS ──────────────────────────────────────────────
    if input_crs is not None and pyromes_gdf.crs is not None:
        temp_gdf = gpd.GeoDataFrame(geometry=[geom], crs=input_crs)
        geom = temp_gdf.to_crs(pyromes_gdf.crs).geometry.iloc[0]

    # ── Query ─────────────────────────────────────────────────────────────────
    if is_point:
        # Direct containment check — fast, no area calculation needed
        hits = pyromes_gdf[pyromes_gdf.geometry.contains(geom)]

        if hits.empty:
            # Point on boundary or in a gap → try intersects
            hits = pyromes_gdf[pyromes_gdf.geometry.intersects(geom)]

        if hits.empty:
            # Last resort: nearest pyrome (handles gaps / coastal points)
            import warnings
            warnings.warn(
                "Point does not fall within any pyrome — returning nearest pyrome.",
                stacklevel=2,
            )
            pyromes_gdf = pyromes_gdf.copy()
            pyromes_gdf["_dist"] = pyromes_gdf.geometry.distance(geom)
            return pyromes_gdf.loc[pyromes_gdf["_dist"].idxmin(), pyrome_col]

        return hits.iloc[0][pyrome_col]

    else:
        # Polygon / AOI — greatest overlap area
        return lookup_pyrome(geom, pyromes_gdf, pyrome_col=pyrome_col)


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
