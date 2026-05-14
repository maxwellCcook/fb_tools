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
    from rasterio.transform import xy as rio_xy
    import geopandas as gpd
    from shapely.geometry import Point

    _NB_CODES = {0, 91, 92, 93, 98, 99}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    with rasterio.open(lcp_fp) as src:
        crs = src.crs
        res_x, res_y = src.res
        buffer_dist = min(res_x, res_y) / 2.0

        # Locate FBFM40 band by description; fall back to band 4
        fbfm_band = 4
        for i, desc in enumerate(src.descriptions, start=1):
            if desc and "FBFM" in desc.upper():
                fbfm_band = i
                break

        data = src.read(fbfm_band)
        nodata = src.nodata

        burnable = ~np.isin(data, list(_NB_CODES))
        if nodata is not None:
            burnable &= data != nodata

        rows, cols = np.where(burnable)
        all_xs, all_ys = rio_xy(src.transform, rows, cols)

    # Vectorised containment check against container in LCP CRS
    container_proj = container_gdf.to_crs(crs)
    try:
        container_union = container_proj.geometry.union_all()
    except AttributeError:
        container_union = container_proj.geometry.unary_union

    pts_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(all_xs, all_ys), crs=crs
    )
    inside_mask = pts_gdf.within(container_union)
    inside_pts = pts_gdf[inside_mask].reset_index(drop=True)

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
