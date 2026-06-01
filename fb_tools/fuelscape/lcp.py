"""
LCP / fuelscape raster utilities.

Covers:
  - Stacking individual FlamMap output bands into a single multi-band GeoTIFF.
  - Creating an ASCII ignition grid from point features.
  - Selecting a band from a multi-band DataArray by its long_name attribute.
"""

import os
import gc
from pathlib import Path

import pandas as pd
import rioxarray as rxr
import xarray as xr


def get_band_by_longname(da, long_name_value):
    """
    Select a band from a multi-band DataArray by its ``long_name`` attribute.

    Parameters
    ----------
    da : xarray.DataArray
        Multi-band rioxarray DataArray with a ``long_name`` attribute list.
    long_name_value : str
        The name to search for (e.g. ``"FBFM40"``).

    Returns
    -------
    xarray.DataArray
        Single-band DataArray.

    Raises
    ------
    ValueError
        If no ``long_name`` attribute is present or the value is not found.
    """
    longnames = da.attrs.get("long_name", [])
    if not longnames:
        raise ValueError("No long_name attribute found in DataArray.")
    if long_name_value not in longnames:
        raise ValueError(f"{long_name_value!r} not found in long_name list: {longnames}")
    idx = list(longnames).index(long_name_value) + 1  # bands are 1-based in xarray
    return da.sel(band=idx)


def stack_rasters(in_dir, tag=None, out_dir=None, cleanup=True):
    """
    Stack individual single-band GeoTIFFs in *in_dir* into one multi-band file.

    Expects files named ``<scenario>_<bandname>.tif``
    (e.g. ``PCT25_FLAMELENGTH.tif``).  The stacked output is written to
    ``<out_dir>/<SCENARIO>_<TAG>.tif``.

    Parameters
    ----------
    in_dir : str or Path
        Directory containing single-band TIFFs.
    tag : str, optional
        Label for the fuelscape (derived from ``in_dir.name`` if omitted).
    out_dir : str or Path, optional
        Destination directory (defaults to *in_dir*).
    cleanup : bool
        Delete the input single-band TIFFs after stacking (default True).
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir) if out_dir else in_dir

    parts = in_dir.name.split("_", 1)
    if tag is None:
        tag = parts[1].upper() if len(parts) > 1 else in_dir.name.upper()

    file_prefix = os.path.basename(in_dir) # scenario or file name, needs attention
    print(file_prefix)
    # Compute output path first so we can exclude it from the input glob.
    # Without this, re-runs pick up the previously stacked file and fail
    # with a band-dimension mismatch (stacked file has N bands; squeeze()
    # doesn't reduce it; concat sees N+1+1+… bands vs. N+2 names).
    out_fp = out_dir / f"{file_prefix}_{tag.upper()}.tif"
    tifs = sorted(t for t in in_dir.glob("*.tif") if t.resolve() != out_fp.resolve())
    if not tifs:
        raise FileNotFoundError(f"No TIFFs found in {in_dir}")

    bands, band_names = [], []
    for tif in tifs:
        band_name = tif.stem.rsplit("_", 1)[-1]
        band_names.append(band_name)
        da = rxr.open_rasterio(tif).squeeze().load()  # .load() pulls into memory
        bands.append(da)

    print(band_names)
    stack = xr.concat(bands, dim=xr.Variable("band", band_names))
    stack.attrs["long_name"] = band_names
    print(out_fp)
    stack.rio.to_raster(out_fp, compress="deflate")
    print(f"Stacked {len(tifs)} rasters → {out_fp}")

    del stack, bands
    gc.collect()

    if cleanup:
        for tif in tifs:
            tif.unlink()
            aux = tif.with_suffix(".tif.aux")
            aux_xml = tif.with_suffix(".tif.aux.xml")
            if aux.exists():
                aux.unlink()
            if aux_xml.exists():
                aux_xml.unlink()


# FBFM40 non-burnable codes: 0 (NoData), 91-93/98/99 (NB1-NB5).
_NB_CODES = {0, 91, 92, 93, 98, 99}


def _burnable_pixel_points(lcp_fp, mask_geom=None):
    """
    Return burnable pixel-centre points from a landscape raster.

    Locates the FBFM40 band by description (falls back to band 4), masks out
    non-burnable codes and NoData, and returns the centre of every burnable
    pixel as a point GeoDataFrame in the LCP CRS.

    Parameters
    ----------
    lcp_fp : str or Path
        Path to the landscape raster.
    mask_geom : shapely geometry, optional
        If provided (in the LCP CRS), only pixel centres inside this geometry
        are returned.

    Returns
    -------
    tuple
        ``(points_gdf, buffer_dist)`` — a point GeoDataFrame of burnable pixel
        centres and half the minimum pixel dimension (suitable as an ignition
        buffer radius).
    """
    import numpy as np
    import rasterio
    from rasterio.transform import xy as rio_xy
    import geopandas as gpd

    with rasterio.open(lcp_fp) as src:
        crs = src.crs
        res_x, res_y = src.res
        transform = src.transform
        nodata = src.nodata

        fbfm_band = 4
        for i, desc in enumerate(src.descriptions, start=1):
            if desc and "FBFM" in desc.upper():
                fbfm_band = i
                break
        data = src.read(fbfm_band)

    burnable = ~np.isin(data, list(_NB_CODES))
    if nodata is not None:
        burnable &= data != nodata

    rows, cols = np.where(burnable)
    xs, ys = rio_xy(transform, rows, cols)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs)

    if mask_geom is not None:
        pts = pts[pts.within(mask_geom)].reset_index(drop=True)

    return pts, min(res_x, res_y) / 2.0


def create_container_ignition(container_gdf, out_path):
    """
    Save a dissolved container polygon as an FSPro ignition shapefile.

    FSPro accepts polygon or polyline ignitions via the ``IgnitionFile`` field.
    This function dissolves all features in *container_gdf* to a single polygon
    (the container boundary) and writes it as a shapefile.

    Parameters
    ----------
    container_gdf : GeoDataFrame
        Spatial container features (HUC12, fireshed, POD, etc.).
        May have any CRS; it is written as-is.
    out_path : str or Path
        Destination shapefile path (e.g. ``"data/ignitions/huc12_ign.shp"``).
        Parent directory is created if it does not exist.

    Returns
    -------
    Path
        Absolute path to the written shapefile.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dissolved = container_gdf.dissolve()[["geometry"]].reset_index(drop=True)
    dissolved.to_file(out_path)

    print(f"  [create_container_ignition] Saved ignition → {out_path.name} "
          f"({len(dissolved)} feature(s))")
    return out_path.resolve()


def create_random_ignitions(
    container_gdf,
    n_points: int,
    lcp_fp,
    out_path,
    seed=None,
):
    """
    Generate spatially-distributed random ignition polygons within a container.

    Samples *n_points* random pixel centers from burnable cells in *lcp_fp*
    that fall inside *container_gdf*, then buffers each by half a pixel to
    produce a set of small circle polygons suitable for FSPro's
    ``IgnitionFile``.  Using many small ignition polygons instead of the full
    container boundary forces FSPro to start each simulated fire from a
    realistic, spatially-variable location rather than sampling uniformly from
    the entire analysis area.

    Non-burnable FBFM40 codes excluded from sampling: 0 (NoData) and
    91–99 (NB1–NB5).  The FBFM40 band is located by searching raster band
    descriptions for "FBFM"; band 4 is used as a fallback for standard
    LANDFIRE LCP band order.

    Parameters
    ----------
    container_gdf : GeoDataFrame
        Spatial container (HUC12, fireshed, POD, etc.).  May be in any CRS;
        it is reprojected to the LCP CRS internally.
    n_points : int
        Target number of ignition points.  If fewer burnable pixels exist
        inside the container, all burnable pixels are used and a warning is
        printed.
    lcp_fp : str or Path
        Path to the landscape raster.  Used to determine CRS, pixel
        resolution, and burnable/non-burnable pixel locations.
    out_path : str or Path
        Destination shapefile path.  Parent directory is created if needed.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Path
        Absolute path to the written shapefile.

    Raises
    ------
    ValueError
        If no burnable pixels are found inside the container.
    """
    import numpy as np
    import rasterio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    with rasterio.open(lcp_fp) as src:
        crs = src.crs

    # Container union in LCP CRS, used to mask burnable pixel centres
    container_proj = container_gdf.to_crs(crs)
    try:
        container_union = container_proj.geometry.union_all()
    except AttributeError:
        container_union = container_proj.geometry.unary_union

    inside_pts, buffer_dist = _burnable_pixel_points(lcp_fp, mask_geom=container_union)

    if len(inside_pts) == 0:
        raise ValueError(
            "No burnable pixels found within the container boundary. "
            "Check that the LCP overlaps the container and contains burnable fuels."
        )

    if len(inside_pts) < n_points:
        print(
            f"  [create_random_ignitions] Warning: only {len(inside_pts)} burnable "
            f"pixels inside container (requested {n_points}); using all of them."
        )
        sampled = inside_pts
    else:
        idx = rng.choice(len(inside_pts), size=n_points, replace=False)
        sampled = inside_pts.iloc[idx].reset_index(drop=True)

    sampled = sampled.copy()
    sampled["geometry"] = sampled.geometry.buffer(buffer_dist)
    sampled[["geometry"]].to_file(out_path)

    print(
        f"  [create_random_ignitions] {len(sampled)} ignition circles "
        f"(buffer={buffer_dist:.1f} m, seed={seed}) → {out_path.name}"
    )
    return out_path.resolve()


def _bearing_deg(p_from, p_to):
    """Azimuth in degrees (0-360, from north) from point *p_from* to *p_to*."""
    import numpy as np
    dx = p_to.x - p_from.x
    dy = p_to.y - p_from.y
    return float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)


def _annular_sector(cx, cy, center_az, sector_deg, r_in, r_out, n_arc=48):
    """
    Build an annular-wedge polygon around (cx, cy).

    Parameters
    ----------
    cx, cy : float
        Sector apex coordinates (projected metres).
    center_az : float
        Azimuth (degrees from north) the wedge is centred on.
    sector_deg : float
        Full angular width of the wedge.
    r_in, r_out : float
        Inner and outer radii in metres.
    n_arc : int
        Number of points sampled along each arc.

    Returns
    -------
    shapely.geometry.Polygon
    """
    import numpy as np
    from shapely.geometry import Polygon

    half = sector_deg / 2.0
    azimuths = np.linspace(center_az - half, center_az + half, n_arc)
    # Azimuth (from north) → math angle (CCW from east)
    math_ang = np.radians(90.0 - azimuths)

    outer = [(cx + r_out * np.cos(a), cy + r_out * np.sin(a)) for a in math_ang]
    inner = [(cx + r_in * np.cos(a), cy + r_in * np.sin(a)) for a in math_ang[::-1]]
    return Polygon(outer + inner)


def create_directional_ignitions(
    treatments_gdf,
    values_gdf,
    wind_from_deg,
    lcp_fp,
    out_dir,
    n_ignitions=15,
    dist_band_km=(2.0, 10.0),
    sector_deg=45.0,
    require_treatment_intersect=True,
    buffer_m=None,
    seed=None,
):
    """
    Generate single-feature ignition shapefiles upwind of a treatment cluster.

    Implements an "upwind toward values" experimental design.  Ignitions are
    placed upwind of the treatment cluster (using the meteorological FROM wind
    direction) so that, when fire spreads downwind, the geometry is
    *ignition → treatments → values*.  Each ignition is written as its own
    single-feature shapefile so it can be run as a **separate** FSPro fire
    (FSPro treats one ``IgnitionFile`` as one fire's ignition zone).

    Parameters
    ----------
    treatments_gdf : GeoDataFrame
        Treatment polygons (the treatment group under test).  Any CRS.
    values_gdf : GeoDataFrame
        Values / assets to protect (WUI, communities, structures, POD).
        Any CRS.  Used only to verify the wind/placement geometry.
    wind_from_deg : float
        Dominant wind direction in degrees FROM (meteorological convention,
        0/360 = north).  See :func:`~fb_tools.weather.dominant_wind_direction`.
    lcp_fp : str or Path
        Landscape raster — supplies CRS, pixel size, and burnable fuels.
    out_dir : str or Path
        Directory for the per-ignition shapefiles.  Created if absent.
    n_ignitions : int
        Number of ignition points to generate.  Default 15.
    dist_band_km : tuple of float
        ``(min, max)`` distance band from the treatment-cluster centroid, in
        km, within which ignitions are placed.  Choose so fire growth reaches
        the treatments within the desired early window.  Default ``(2, 10)``.
    sector_deg : float
        Full angular width of the upwind placement wedge.  Default 45.
    require_treatment_intersect : bool
        If ``True`` (default), keep only candidate ignitions whose straight
        downwind ray intersects the treatment footprint, so every ignition
        genuinely tests treatment interaction.
    buffer_m : float, optional
        Ignition circle radius in LCP units.  Defaults to half the LCP pixel.
    seed : int, optional
        Random seed for reproducible sampling.

    Returns
    -------
    dict
        ``"ignition_shapefiles"`` : list of Path
            One single-feature shapefile per ignition (``ign_000.shp`` …).
        ``"ignitions_gdf"`` : GeoDataFrame
            One row per ignition: ``ign_id``, ``dist_m`` (to treatment
            centroid), ``az_to_trt`` (azimuth toward treatment centroid),
            ``shp_path``, and point geometry (LCP CRS).
        ``"wind_from_deg"`` : float
        ``"downwind_az"`` : float

    Raises
    ------
    ValueError
        If no burnable candidate pixels are found in the placement zone.

    Notes
    -----
    The placement wedge is built upwind of the treatment centroid (azimuth
    ``wind_from_deg``).  ``downwind_az = (wind_from_deg + 180) % 360`` is the
    direction fire travels.  A warning is printed if the values layer is not
    downwind of the treatments, which would indicate the design geometry is
    inconsistent with the prevailing wind.
    """
    import numpy as np
    import geopandas as gpd
    import rasterio
    from shapely.geometry import LineString

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(lcp_fp) as src:
        crs = src.crs
        res_x, res_y = src.res
    if buffer_m is None:
        buffer_m = min(res_x, res_y) / 2.0

    trt = treatments_gdf.to_crs(crs)
    val = values_gdf.to_crs(crs)
    trt_union = trt.geometry.union_all()
    val_union = val.geometry.union_all()
    trt_centroid = trt_union.centroid
    val_centroid = val_union.centroid

    downwind_az = (float(wind_from_deg) + 180.0) % 360.0

    # Geometry check: values should be downwind of the treatments
    trt_to_val_az = _bearing_deg(trt_centroid, val_centroid)
    ang_diff = abs((trt_to_val_az - downwind_az + 180.0) % 360.0 - 180.0)
    if ang_diff > 90.0:
        print(
            f"  [create_directional_ignitions] Warning: values are not downwind "
            f"of treatments (treatment→values azimuth {trt_to_val_az:.0f}°, "
            f"downwind azimuth {downwind_az:.0f}°, off by {ang_diff:.0f}°). "
            f"The ignition→treatments→values geometry may be inconsistent with "
            f"the prevailing wind."
        )

    # Upwind placement wedge centred on the treatment centroid
    r_in, r_out = dist_band_km[0] * 1000.0, dist_band_km[1] * 1000.0
    wedge = _annular_sector(
        trt_centroid.x, trt_centroid.y,
        center_az=float(wind_from_deg),
        sector_deg=sector_deg,
        r_in=r_in, r_out=r_out,
    )

    candidates, _ = _burnable_pixel_points(lcp_fp, mask_geom=wedge)
    if len(candidates) == 0:
        raise ValueError(
            "No burnable pixels found in the upwind placement zone. "
            "Check dist_band_km, sector_deg, and that the LCP covers the zone."
        )

    # Keep candidates whose downwind ray crosses the treatment footprint
    if require_treatment_intersect:
        ray_len = r_out + 2000.0
        math_ang = np.radians(90.0 - downwind_az)
        dx, dy = np.cos(math_ang), np.sin(math_ang)
        keep = []
        for pt in candidates.geometry:
            ray = LineString([
                (pt.x, pt.y),
                (pt.x + dx * ray_len, pt.y + dy * ray_len),
            ])
            keep.append(ray.intersects(trt_union))
        candidates = candidates[np.array(keep)].reset_index(drop=True)
        if len(candidates) == 0:
            raise ValueError(
                "No candidate ignition crosses the treatment footprint along "
                "the downwind axis. Widen sector_deg or set "
                "require_treatment_intersect=False."
            )

    rng = np.random.default_rng(seed)
    if len(candidates) < n_ignitions:
        print(
            f"  [create_directional_ignitions] Warning: only {len(candidates)} "
            f"candidate pixels (requested {n_ignitions}); using all of them."
        )
        sampled = candidates
    else:
        idx = rng.choice(len(candidates), size=n_ignitions, replace=False)
        sampled = candidates.iloc[idx].reset_index(drop=True)

    shp_paths = []
    rows = []
    for i, pt in enumerate(sampled.geometry):
        shp_path = out_dir / f"ign_{i:03d}.shp"
        circle = gpd.GeoDataFrame(geometry=[pt.buffer(buffer_m)], crs=crs)
        circle.to_file(shp_path)
        shp_paths.append(shp_path.resolve())
        rows.append({
            "ign_id":    i,
            "dist_m":    round(pt.distance(trt_centroid), 1),
            "az_to_trt": round(_bearing_deg(pt, trt_centroid), 1),
            "shp_path":  str(shp_path.resolve()),
            "geometry":  pt,
        })

    ignitions_gdf = gpd.GeoDataFrame(rows, crs=crs)

    print(
        f"  [create_directional_ignitions] {len(sampled)} ignition(s) "
        f"upwind of treatments (wind FROM {wind_from_deg:.0f}°, "
        f"band {dist_band_km[0]}-{dist_band_km[1]} km) → {out_dir}"
    )
    return {
        "ignition_shapefiles": shp_paths,
        "ignitions_gdf":       ignitions_gdf,
        "wind_from_deg":       float(wind_from_deg),
        "downwind_az":         downwind_az,
    }


def create_fod_ignitions(
    container_gdf,
    fod_gdf,
    lcp_fp,
    out_path,
    buffer_m=None,
):
    """
    Build an FSPro ignition shapefile from historical FPA-FOD point locations.

    Clips *fod_gdf* to *container_gdf*, reprojects to the LCP CRS, and
    buffers each point by half a pixel (or *buffer_m*) to produce a set of
    small circle polygons.  Using historical ignition locations grounds the
    FSPro simulation in where fires have actually started within the analysis
    unit rather than drawing uniformly from the full container area.

    Parameters
    ----------
    container_gdf : GeoDataFrame
        Spatial container (HUC12, fireshed, POD, etc.).
    fod_gdf : GeoDataFrame
        FPA-FOD (or equivalent) point features.  Should be pre-filtered to
        the relevant years, fire-size classes, and/or cause codes before
        calling this function.  Any CRS is accepted; it is reprojected
        internally.
    lcp_fp : str or Path
        Path to the landscape raster.  Used to determine CRS and default
        buffer distance.
    out_path : str or Path
        Destination shapefile path.  Parent directory is created if needed.
    buffer_m : float, optional
        Buffer radius in the LCP's linear units (usually metres).  Defaults
        to half the minimum pixel dimension so each circle covers one pixel.

    Returns
    -------
    Path
        Absolute path to the written shapefile.

    Raises
    ------
    ValueError
        If no FPA-FOD points fall inside the container.

    Notes
    -----
    Overlapping circles are intentional — densely-ignited areas will receive
    proportionally more simulated fire starts, reflecting the historical
    ignition density.
    """
    import rasterio
    import geopandas as gpd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(lcp_fp) as src:
        crs = src.crs
        res_x, res_y = src.res
        default_buffer = min(res_x, res_y) / 2.0

    if buffer_m is None:
        buffer_m = default_buffer

    container_proj = container_gdf.to_crs(crs)
    fod_proj = fod_gdf.to_crs(crs)

    fod_clipped = gpd.clip(fod_proj, container_proj)

    if len(fod_clipped) == 0:
        raise ValueError(
            "No FPA-FOD ignitions found within the container boundary. "
            "Check that fod_gdf covers the study area and is not over-filtered."
        )

    result = fod_clipped[["geometry"]].copy().reset_index(drop=True)
    result["geometry"] = result.geometry.buffer(buffer_m)
    result.to_file(out_path)

    print(
        f"  [create_fod_ignitions] {len(result)} historical ignition circles "
        f"(buffer={buffer_m:.1f} m) → {out_path.name}"
    )
    return out_path.resolve()


def create_ignition_ascii(ign_gdf, ref_img_fp, out_ascii_fp):
    """
    Rasterize ignition points to an ASCII grid (.asc) snapped to a reference raster.

    Parameters
    ----------
    ign_gdf : GeoDataFrame
        Ignition point features (must be in the same CRS as the reference raster).
    ref_img_fp : str or Path
        Path to the reference raster (e.g. your landscape .tif).
    out_ascii_fp : str or Path
        Output path for the ASCII grid file.
    """
    from ..utils.geo import rasterize  # avoid circular at module level

    ign_gdf = ign_gdf.copy()
    ign_gdf["burn"] = 1

    ref_img = rxr.open_rasterio(ref_img_fp)[0]
    ign_grid = rasterize(ign_gdf, to_img=ref_img, attr="burn", fill_val=0)
    ign_grid.rio.to_raster(str(out_ascii_fp), driver="AAIGrid", nodata=0)

    del ref_img
    print(f"Saved ignition grid → {out_ascii_fp}")
