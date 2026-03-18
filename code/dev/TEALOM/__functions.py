"""
TEALOM helper functions
"""

import os, shutil, gc, glob, re
import time, json, io, zipfile
import requests
import shapely
import tempfile
import subprocess
import hashlib

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import rasterio as rio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from geocube.api.core import make_geocube
from pathlib import Path
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from rasterstats import zonal_stats
from rasterio.mask import mask as rio_mask
from shapely import set_precision


def is_valid_geom(g):
    """
    Check if a GeoDataFrame is valid (has rows and geometry)
    :param g: GeoDataFrame
    :return: Boolean
    """
    return g is not None and not g.is_empty

def get_geom(gdf, label):
    try:
        geom = gdf.loc[label, 'geometry']
        return geom if is_valid_geom(geom) else None
    except KeyError:
        print(f"Missing geometry for: {label}")
        return None


def make_valid_nonempty(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Drop empty geometries """
    gdf["geometry"] = gdf.geometry.make_valid().copy()
    return gdf.loc[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()


def iqr(x):
    """Return the IQR of a given column"""
    return x.quantile(0.75) - x.quantile(0.25)


def list_files(path, ext, recursive=True):
    """
    Find file names recursively for a given string match

    :param path: the directory to search
    :param ext: the file extension to return
    :param recursive: search recursively or not, default to True
    :return:
    """
    if recursive is True:
        return glob.glob(os.path.join(path, '**', '*{}'.format(ext)), recursive=recursive)
    else:
        return glob.glob(os.path.join(path, '*{}'.format(ext)), recursive=recursive)


def make_geom_key(g, grid=0.01):
    """
    Creates a unique ID from a geometry column

    :param g:
    :param grid:
    :return: 32 character hash string unique to the geometry.
    """
    if g is None or g.is_empty:
        return np.nan
    g = set_precision(g, grid)
    g = shapely.normalize(g)   # stable vertex/ring ordering
    if g.is_empty:
        return np.nan
    return hashlib.md5(g.wkb).hexdigest()


def plot_color_swatch(cmap_dict, title="Color swatches"):
    """
    Creates a color swatch plot helpful for testing different color combinations
    :param cmap_dict: A dictionary with the categorical values
    :param title: Plot title
    :return: Color palette plot
    """
    keys = list(cmap_dict.keys())
    colors = [cmap_dict[t] for t in keys]

    fig, ax = plt.subplots(figsize=(5, len(keys) * 0.3))

    for i, (label, color) in enumerate(zip(keys, colors)):
        ax.barh(i, 1, color=color, edgecolor='black')
        ax.text(1.02, i, label, va='center', ha='left', fontsize=10)

    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.5, len(keys) - 0.5)
    ax.axis('off')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def rasterize_it(zones, to_img, attr="id", fill_val=-9999):
    """
    Rasterize input polygon features based on a specified column.
    :param zones: The input geometry to be rasterized (required).
    :param to_img: Snap raster (required).
    :param attr: The attribute to be used (required).
    :param fill_val: Pixel fill value (optional).
    :return: Rasterized polygon features.
    """

    # ensure CRS matches
    zones = zones.to_crs(to_img.rio.crs)

    # rasterize
    rastered = make_geocube(
        vector_data=zones,
        measurements=[attr],
        like=to_img,  # snap to the raster grid
        fill=fill_val
    )

    da = rastered[attr]

    # attach CRS and transform directly here
    da = da.rio.write_crs(to_img.rio.crs)
    da = da.rio.write_transform(to_img.rio.transform())
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    return da


def extract_by_region(gdf, aoi: gpd.GeoDataFrame,
                      uid: str, out_col='UNIT_ID',
                      thresh: float = 0.5,
                      clip: bool = False):
    """

    :param gdf:
    :param aoi:
    :param uid:
    :param out_col:
    :param thresh:
    :param clip:
    :return:
    """
    if isinstance(gdf, str) or isinstance(gdf, io.IOBase):
        regions = (gpd.read_file(gdf)
                  .to_crs(aoi.crs))  # read in if file path
    else:
        regions = gdf.to_crs(aoi.crs)  # else already a geodataframe

    # Force polygon-only before overlay to avoid sliver artifacts
    regions = regions[regions.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    aoi = aoi[aoi.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    regions['geometry'] = shapely.force_2d(regions.geometry.values)
    aoi['geometry'] = shapely.force_2d(aoi.geometry.values)

    regions['unit_ac'] = regions.geometry.area

    # Run extract by location (with overlap acres)
    nb = regions.copy()
    nb['geometry'] = nb.geometry.buffer(-2)
    overlap = gpd.overlay(nb, aoi, how='intersection')
    overlap['overlap_ac'] = nb.geometry.area  # overlap area
    overlap['unit_pct'] = (overlap['overlap_ac'] / overlap['unit_ac']) # calculate percent overlap
    overlap[out_col] = overlap[uid].astype(str)  # force a new region ID column

    if clip is True:
        # return the clipped portion
        return overlap[[out_col, 'unit_pct', 'geometry']]

    else:
        regions = regions[regions[uid].isin(overlap[uid].unique())] # filter the original regions
        regions = regions.merge(overlap[[uid,'overlap_ac','unit_pct']], how='left', on=uid) # merge to get overlap acres
        regions[out_col] = regions[uid].astype(str) # force a new region ID column

        if thresh is not None:
            regions = regions[regions['unit_pct'] >= thresh]

        return regions[[out_col, 'unit_pct', 'geometry']]


def assign_majority(gdf: gpd.GeoDataFrame, regions: gpd.GeoDataFrame,
                    id_col: str = 'TRT_ID', region_col: str = 'RegionID',
                    out_col: str = 'Region', thresh: float = None):
    """

    :param gdf:
    :param regions:
    :param id_col:
    :param region_col:
    :param out_col:
    :param thresh:
    :return:
    """
    # Ensure projection matches
    regions = regions.to_crs(gdf.crs)
    regions['region_acres'] = regions.geometry.area * 0.000247105
    # Run the spatial intersection
    overlap = gpd.overlay(gdf, regions, how='intersection')
    # Calculate overlap area in acres
    overlap['overlap_area'] = overlap.geometry.area * 0.000247105

    # Find the region with majority overlap
    majority = (
        overlap.loc[
            overlap.groupby(id_col)['overlap_area'].idxmax(),
            [id_col, region_col]
        ]
        .rename(columns={region_col: out_col})
        .reset_index(drop=True)
    )

    # Filter if requested
    if thresh is not None:
        majority['overlap_pct'] = majority['overlap_pct'] / majority['region_acres']
        majority = majority[majority['overlap_pct'] >= thresh]

    majority = majority[[id_col, out_col]]

    # Merge majority region back to original GeoDataFrame
    result = gdf.merge(majority, on=id_col, how='left')
    del overlap, majority

    return result


def unit_acres(units_gdfs: dict, crs_proj: str = None) -> pd.DataFrame:
    """
    Summarize GIS acres by unit type and unit ID from a dictionary of GeoDataFrames.

    Parameters
    ----------
    units_gdfs : dict
        Dictionary of {unit_name: GeoDataFrame} with a 'UNIT_ID' column
    crs_proj : str, optional
        EPSG code or proj string for area calculation. If None, uses each GDF's native CRS.
        Recommended: an equal-area projection (e.g., 'EPSG:5070' for CONUS Albers)

    Returns
    -------
    pd.DataFrame with columns: [unit_type, UNIT_ID, acres]
    """
    records = []
    for unit_name, gdf in units_gdfs.items():
        g = gdf.to_crs(crs_proj) if crs_proj else gdf

        # Handle GDFs that may not have UNIT_ID (e.g., NCFC as a single polygon)
        id_col = 'UNIT_ID' if 'UNIT_ID' in g.columns else None

        if id_col:
            for uid, row in g.set_index(id_col).iterrows():
                acres = row.geometry.area / 4046.856  # m² → acres
                records.append({'UNIT_NAME': unit_name, 'UNIT_ID': uid, 'UNIT_ACRES': round(acres, 2)})
        else:
            # Single-feature units (e.g., NCFC boundary)
            total_acres = g.geometry.area.sum() / 4046.856
            records.append({'UNIT_NAME': unit_name, 'UNIT_ID': unit_name, 'UNIT_ACRES': round(total_acres, 2)})

    return pd.DataFrame(records)


def get_band_by_longname(da, long_name_value):
    """
    Select a band from a multiband rioxarray DataArray using its long_name attribute list.
    """
    longnames = da.attrs.get("long_name", [])

    if not longnames:
        raise ValueError("No long_name attribute found in DataArray.")
    if long_name_value not in longnames:
        raise ValueError(f"{long_name_value} not found in long_name list: {longnames}")

    idx = longnames.index(long_name_value) + 1  # +1 because bands are 1-based in xarray
    return da.sel(band=idx)

def lfps_request(
    aoi_gdf,
    out_dir,
    layer_codes,
    edit_rules=None,
    rename=None,
    email="Maxwell.Cook@colostate.edu",
    out_crs=5070
):
    """
    Submit and extract LANDFIRE products using the LFPS API (restful service)

    :param aoi_gdf: The area of interest (geospatial)
    :param out_dir: Where to store the outputs form the requested area
    :param layer_codes: Which layers to access from LANDFIRE (https://lfps.usgs.gov/products)
    :param edit_rules: (Optional) edit rules to be applied to the layers
    :param rename: Output file name (multiband raster)
    :param email: Email required to access the download link
    :param out_crs: The project CRS
    :return: Multi-band image stack with requested layers
    """
    print(f"Requesting from LFPS: {layer_codes}")

    # Format bounding box
    aoi = aoi_gdf.to_crs(4326) # geographic CRS needed for bounding box
    bbox = aoi.total_bounds # create the bounding box
    bbox_str = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"

    # Set up parameters
    layer_list = ";".join(layer_codes)
    params = {
        "Layer_List": layer_list,
        "Area_of_Interest": bbox_str,
        "Output_Projection": out_crs,
        "Email": email,
    }

    if edit_rules:
        params["Edit_Rule"] = json.dumps(edit_rules)

    # Submit job
    job_url = "https://lfps.usgs.gov/api/job/submit"
    r = requests.get(job_url, params=params)
    job_id = r.json()["jobId"]
    print(f"Job submitted. ID: {job_id}")

    # Poll for status
    status_url = f"https://lfps.usgs.gov/api/job/status?JobId={job_id}"
    while True:
        resp = requests.get(status_url).json()
        status = resp.get("status", "Unknown")
        print(f"[{time.ctime()}] Status: {status}")
        if status == "Succeeded":
            break
        elif status in ["Failed", "Cancelled"]:
            raise RuntimeError(f"LFPS job failed: {resp.get('messages', '')}")
        time.sleep(30)

    # Download ZIP
    download_url = resp["outputFile"]
    print(f"\tDownload: {download_url}")
    r = requests.get(download_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Clean and extract
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    z.extractall(out_dir)
    print(f"\tDownloaded to {out_dir}\n")

    # Optional: Rename files
    if rename:
        tif_path = next(Path(out_dir).glob("*.tif"))
        base_old = tif_path.stem
        base_new = rename

        # Define file types to rename
        extensions = [".tif", ".tfw", ".tif.aux.xml", ".tif.xml"]
        for ext in extensions:
            old_path = Path(out_dir) / f"{base_old}{ext}"
            new_path = Path(out_dir) / f"{base_new}{ext}"
            if old_path.exists():
                old_path.rename(new_path)
        print(f"\t\tRenamed files to: {base_new}.*\n")

def min_overlap_fraction(geom_a, geom_b, min_frac: float = 0.0) -> float:
    """
    Symmetric overlap fraction: intersection / min(area_a, area_b).
    Useful for catching near-exact matches even if one polygon is slightly
    larger (e.g. digitizing slop).
    """
    inter = geom_a.intersection(geom_b).area
    denom = min(geom_a.area, geom_b.area)
    return inter / denom if denom > 0 else 0.0


# Classify thinning as either manual or mechanical to match the forest tracker
def classify_thin(row):
    """
    Categorizes forest thinning treatments into Manual or Mechanical
    :param row:
    :return:
    """

    ttype = str(row.get("type") or "").strip().lower()
    twig_type = str(row.get("twig_type") or "").strip().lower()
    activity = str(row.get("activity") or "").strip().lower()

    # is this a thinning treatment??
    is_thin = False
    # a. direct attribution with TWIG type
    is_thin = (ttype == "thinning") # based on treatment type attribute
    # b. thin keyword in activity code
    if "thin" in activity:
        is_thin = True
    # c. including "cut" activities
    thin_cut = [
        'sanitation cut','improvement cut','group selection cut',
        'overstory removal cut','salvage cut','recreation removal',
        'salvage'
    ]
    if any(k in activity for k in thin_cut):
        is_thin = True

    # if not thinning, ignore
    if not is_thin:
        return np.nan

    # --- Manual vs. Mechanical logic

    # use the equipment and method columns
    eq  = str(row.get("equipment") or "").strip().lower()
    mth = str(row.get("method") or "").strip().lower()

    # Equipment-based rules (highest priority)
    manual_eq = {
        "chain saw", "hand work", "hand saw", "manual logging"
    }
    mech_eq = {
        "feller buncher", "tree shear", "dozer", "masticator",
        "rubber tired skidder logging", "helicopter logging -medium",
        "helicopter logging -small",
    }

    if eq in manual_eq:
        return "Manual"
    if eq in mech_eq:
        return "Mechanical"

    # Method-based rules (if equipment is missing/ambiguous)
    manual_m = {
        "manual", "power hand", "manual logging", "cut trees and brush",
    }

    mech_m = {
        "mechanical", "tractor logging", "logging methods",
        "helicopter", "removal",
    }

    if mth in manual_m:
        return "Manual"
    if mth in mech_m:
        return "Mechanical"

    # Default to mechanical
    return "Mechanical"


def canopy_flattened(treatments):
    """
    Build Hand/Mech/Rx with 'Complete Hand' and 'Complete Mech' for a single period.
    Returns GeoDataFrame with columns: ['CanEff','period','geometry'] (may be empty).
    """

    # dissolve by CanEff within period (union per category)
    flat = treatments[treatments['CanEff'] != 'None']

    # isolate each treatment type
    hand = get_geom(flat, 'Hand Thin')
    mech = get_geom(flat, 'Mech Thin')
    mast = get_geom(flat, 'Masticate')
    rx   = get_geom(flat, 'RxFire')

    # instantiate columns for complete hand/mech
    complete_hand = None
    complete_mech = None

    # run intersections
    if is_valid_geom(hand) and is_valid_geom(rx):
        complete_hand = hand.intersection(rx)
        if complete_hand and not complete_hand.is_empty:
            hand = hand.difference(complete_hand)
            rx   = rx.difference(complete_hand)

    if is_valid_geom(mech) and is_valid_geom(rx):
        complete_mech = mech.intersection(rx)
        if complete_mech and not complete_mech.is_empty:
            mech = mech.difference(complete_mech)
            rx   = rx.difference(complete_mech)

    # prioritize Mech Thin over Hand Thin
    if is_valid_geom(hand) and is_valid_geom(mech):
        hand = hand.difference(mech)

    rows = []
    if is_valid_geom(hand):
        rows.append({'CanEff': 'Hand Thin', 'geometry': hand})
    if is_valid_geom(mech):
        rows.append({'CanEff': 'Mech Thin', 'geometry': mech})
    if is_valid_geom(complete_hand):
        rows.append({'CanEff': 'Complete Hand', 'geometry': complete_hand})
    if is_valid_geom(complete_mech):
        rows.append({'CanEff': 'Complete Mech', 'geometry': complete_mech})
    if is_valid_geom(mast):
        rows.append({'CanEff': 'Masticate', 'geometry': mast})
    if is_valid_geom(rx):
        rows.append({'CanEff': 'RxFire', 'geometry': rx})

    if not rows:
        return gpd.GeoDataFrame(columns=['CanEff','geometry'], geometry='geometry', crs=treatments.crs)

    return gpd.GeoDataFrame(rows, geometry='geometry', crs=treatments.crs)

def surface_flattened(treatments):
    """
    Build Manage/Rearrange/RxFire with priorities for a single period.
    RxFire > Manage > Rearrange. Returns GeoDataFrame ['SurfEff','period','geometry'].
    """

    # dissolve by CanEff within period (union per category)
    flat = treatments[treatments['SurfEff'] != 'None'].dissolve(by='SurfEff')  # index is CanEff, geometry is union

    manage = flat.loc['Manage', 'geometry'] if 'Manage' in flat.index else None
    rearrange = flat.loc['Rearrange', 'geometry'] if 'Rearrange' in flat.index else None
    rxs = flat.loc['RxFire', 'geometry'] if 'RxFire' in flat.index else None

    # priorities: RxFire trumps all; Manage trumps Rearrange
    if is_valid_geom(manage) and is_valid_geom(rxs):
        manage = manage.difference(rxs)
    if is_valid_geom(rearrange) and is_valid_geom(rxs):
        rearrange = rearrange.difference(rxs)
    if is_valid_geom(rearrange) and is_valid_geom(manage):
        rearrange = rearrange.difference(manage)

    rows = []
    if is_valid_geom(manage):
        rows.append({'SurfEff': 'Manage', 'geometry': manage})
    if is_valid_geom(rearrange):
        rows.append({'SurfEff': 'Rearrange', 'geometry': rearrange})
    if is_valid_geom(rxs):
        rows.append({'SurfEff': 'RxFire', 'geometry': rxs})

    return gpd.GeoDataFrame(rows, geometry='geometry', crs=treatments.crs) if rows else \
           gpd.GeoDataFrame(columns=['SurfEff','geometry'], geometry='geometry', crs=treatments.crs)


def _mask_raster(raster_path, geom, nodata_val=None):
    with rio.open(raster_path) as src:
        out_image, _ = rio_mask(src, [geom], crop=True)
        arr = out_image[0]
        if nodata_val is not None:
            arr = np.ma.masked_equal(arr, nodata_val)
            arr = arr.filled(np.nan)
        return arr


def _crown_change(trt_id, geom, baseline_fp, treated_fp, nodata_val=-9999):
    try:
        baseline = _mask_raster(baseline_fp, geom, nodata_val)
        treated = _mask_raster(treated_fp, geom, nodata_val)
        valid = (baseline > 0) & (treated > 0)
        n_valid = valid.sum()
        n_reduced = ((treated < baseline) & valid).sum()
        return {
            "TRT_ID": trt_id,
            "n_valid": int(n_valid),
            "n_reduced": int(n_reduced),
            "prop_reduced": n_reduced / n_valid if n_valid > 0 else None,
        }
    except Exception as e:
        print(f"Error with {trt_id}: {e}")
        return None


def _single_union(gdf):
    """ Union all geometries safely (returns None if empty). """
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return None
    # fix small topology issues
    gdf['geometry'] = gdf.geometry.apply(make_valid)
    geoms = [g for g in gdf.geometry if g and not g.is_empty]
    return unary_union(geoms) if geoms else None


def force_multipolygon(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    # if GeometryCollection etc: keep only polygonal parts
    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        # flatten MultiPolygons inside list
        flat = []
        for p in polys:
            if isinstance(p, Polygon):
                flat.append(p)
            else:
                flat.extend(p.geoms)
        return MultiPolygon(flat)
    return None


def compute_band_stats(
    geoms,
    image_da,
    id_col,
    attr=None,
    stats=None,
    ztype='categorical',
    transform=None,
    nodata=None
):
    """
    Compute zonal statistics for geometries and a single-band raster input.

    Args:
        geoms (GeoDataFrame): Polygons for zonal stats.
        image_da (str | Path | np.ndarray | xr.DataArray): Raster input (filepath, numpy array, or xarray).
        id_col (str): Unique ID column in geoms.
        attr (str): Base attribute name for output columns.
        stats (list): List of stats to compute (for continuous mode).
        ztype (str): 'categorical' or 'continuous'.
        transform (Affine): Affine transform if image_da is an array.
        nodata (int | float): Value to treat as NoData.

    Returns:
        pd.DataFrame: Table of zonal stats by geometry ID.
    """
    attr = attr or 'value'

    # Determine how to handle raster input
    if isinstance(image_da, (str, Path)):
        raster_input = str(image_da)
        affine_kwarg = {}
    elif isinstance(image_da, xr.DataArray):
        raster_input = image_da
        if transform is None:
            transform = image_da.rio.transform()
        if nodata is None and image_da.rio.nodata is not None:
            nodata = image_da.rio.nodata
        affine_kwarg = {"affine": transform}
    elif isinstance(image_da, np.ndarray):
        if transform is None:
            raise ValueError("Affine transform must be provided with numpy array input.")
        raster_input = image_da
        affine_kwarg = {"affine": transform}
    else:
        raise TypeError("Unsupported image_da type. Must be file path, xarray.DataArray, or numpy.ndarray.")

    # Run zonal stats
    zs = zonal_stats(
        vectors=geoms[[id_col, 'geometry']],
        raster=raster_input,
        stats=stats if ztype == 'continuous' else None,
        categorical=(ztype == 'categorical'),
        all_touched=True,
        geojson_out=True,
        nodata=nodata,
        **affine_kwarg
    )

    stats_df = pd.DataFrame(zs)
    stats_df[id_col] = stats_df['properties'].apply(lambda x: x.get(id_col))

    if ztype == 'categorical':
        stats_df['props_list'] = stats_df['properties'].apply(
            lambda x: [(k, v) for k, v in x.items() if k != id_col]
        )
        props = stats_df.explode('props_list').reset_index(drop=True)
        props[[attr, f'{attr}_count']] = pd.DataFrame(props['props_list'].tolist(), index=props.index)
        props.dropna(subset=[attr], inplace=True)
        props[attr] = props[attr].astype(int)

        total = props.groupby(id_col)[f'{attr}_count'].transform('sum')
        props[f'{attr}_pct_cover'] = (props[f'{attr}_count'] / total) * 100

        result = props[[id_col, attr, f'{attr}_count', f'{attr}_pct_cover']]
        if f'{attr}_area_acres' in props.columns:
            result[f'{attr}_area_acres'] = props[f'{attr}_area_acres']
        return result.reset_index(drop=True)

    elif ztype == 'continuous':
        if not stats:
            raise ValueError("For continuous mode, provide a list of stats (e.g., ['sum', 'mean']).")
        for stat in stats:
            stats_df[f'{attr}_{stat}'] = stats_df['properties'].apply(lambda x: x.get(stat))
        return stats_df[[id_col] + [f'{attr}_{stat}' for stat in stats]].copy()

    else:
        raise ValueError("ztype must be 'categorical' or 'continuous'.")


def create_ignition_ascii(ign_gdf, ref_img_fp, out_ascii_fp):
    """
    Rasterize ignition points to an ASCII grid (.asc) using reference raster.

    Parameters
    ----------
    ign_gdf : GeoDataFrame
        Ignition points (must be in same CRS as reference raster)
    ref_img_fp : str
        Path to reference raster (e.g., your .lcp file or related tif)
    out_ascii_fp : str
        Output path for ASCII grid
    """
    # add a binary value for rasterization
    ign_gdf["burn"] = 1

    # load the reference image, first band
    ref_img = rxr.open_rasterio(ref_img_fp)[0]

    # rasterize the ignition points
    ign_grid = rasterize_it(ign_gdf, to_img=ref_img, attr="burn", fill_val=0)

    # save the raster out as an ascii file
    ign_grid.rio.to_raster(out_ascii_fp, driver="AAIGrid", nodata=0)

    del ref_img
    print(f"Saved ignition grid to: {out_ascii_fp}")


def stack_rasters(in_dir, tag=None, out_dir=None, cleanup=True):
    """
    Stack rasters in a specified directory.
    Parameters
    ----------
    in_dir
    tag
    out_dir
    cleanup
    file_suffix

    Returns
    -------
    """
    # situate the directories
    in_dir = Path(in_dir)
    if out_dir is None:
        out_dir = in_dir
    else:
        out_dir = Path(out_dir)

    # tag is the fuelscape we're using (e.g., baseline)
    if tag is None:
        tag = in_dir.name.split("_")[1].upper()  # assumes naming like scenario_tag

    # file_prefix is the scenario (e.g., 'pct25')
    file_prefix = in_dir.name.split("_")[0].upper()

    # find the raster files for the current scenario or raise an error
    tifs = list(Path(in_dir).glob('*.tif'))
    if not tifs:
        raise FileNotFoundError(f"No tifs found in {in_dir}")
    tifs.sort()

    # Load and stack rasters
    bands = []
    band_names = []
    for tif in tifs:
        # get the band naming convention
        band_name = str(tif.stem).split("_")[1]
        band_names.append(band_name)  # use filename stem as band name
        # open the band, name it
        with rxr.open_rasterio(tif) as da:
            da = da.squeeze()  # single band
            da.attrs["long_name"] = f"{file_prefix}_{band_name}"
            bands.append(da)  # append the band

    # stack the individual bands, assign band names
    stack = xr.concat(bands, dim=pd.Index(band_names, name="band"))

    # save the tif file out
    out_fp = out_dir / f"{file_prefix}_{tag.upper()}.tif"
    stack.rio.to_raster(out_fp, compress="deflate")
    print(f"Stacked {len(tifs)} rasters to: {out_fp}")

    del stack, bands
    gc.collect() # clean up

    # cleanup the directories if specified (default True)
    if cleanup:
        for tif in tifs:
            tif.unlink()
            aux_fp = tif.with_suffix(".tif.aux")
            if aux_fp.exists():
                aux_fp.unlink() # delete the FlamMap tifs and aux files


def cli_flammap_scenarios(
        fm_exe, lcp_fp, fm_params, output_directory, n_process=1,
        tag=None, stack_out=False, cleanup=False
):
    """
    Initiates a FlamMap CLI run for a given fire weather scenario and landscape file.

    :param fm_exe: the executable for TestFlamMap
    :param lcp_fp: landscape/fuelscape file
    :param fm_params: scenario table defining the model inputs
    :param output_directory: Where to save the outputs
    :param n_process: number of cores to use (I think?)
    :param tag: tag to identify this run (optional)
    :param stack_out: Stack rasters into output folder
    :param cleanup: Clean up the output folder
    :return: All requested FlamMap outputs for each fire scenario.
    """

    # get the fuelscape name as the tag
    if tag is None:
        tag = fm_params['LCPType']

    # initiate a FlamMap input file and command file
    input_file = output_directory / "FlamMap.input"
    command_file = output_directory / "FMcommand.txt"

    # write contents to the input file
    with open(input_file, "w") as f:
        # write the header
        f.write("ShortTerm-Inputs-File-Version-1\n\n")

        # specify fuel moisture and other methods
        f.write("FUEL_MOISTURES_DATA: 1\n")
        f.write(f"0 {fm_params['FM_1hr']} {fm_params['FM_10hr']} "
                f"{fm_params['FM_100hr']} {fm_params['FM_herb']} {fm_params['FM_woody']}\n")
        f.write("FOLIAR_MOISTURE_CONTENT: 100\n")
        # specify the crown fire methodology
        f.write(f"CROWN_FIRE_METHOD: {str(fm_params['CROWN_FIRE_METHOD'])}\n")
        # processors (could experiment with higher values for greater computation?)
        f.write(f"NUMBER_PROCESSORS: {n_process}\n") # default number of processors to 1
        # specify wind information (speed and direction)
        f.write(f"WIND_SPEED: {fm_params['WIND_SPEED']}\n") # wind speed
        f.write(f"WIND_DIRECTION: {fm_params['WIND_DIRECTION']}\n") # note that -1 is uphill and -2 is downhill
        # WindNinja settings:
        if fm_params["GRIDDED_WINDS_GENERATE"] == "Yes":
            # turn on WindNinja
            f.write("GRIDDED_WINDS_GENERATE: Yes\n")
            # set the resolution
            f.write(f"GRIDDED_WINDS_RESOLUTION: {fm_params['GRIDDED_WINDS_RESOLUTION']}\n")

        # specify the desired outputs
        for output_type in str(fm_params["Outputs"]).split(", "):
            f.write(f"{output_type}: 1\n")

    # open and edit the command file (tells FlamMap what to run)
    with open(command_file, "w") as f:
        f.write(f'{str(lcp_fp)} {str(input_file.name)} . 2\n')

    # Run FlamMap scenario
    # log the outputs
    with open(output_directory / "flammap_run.log", "w") as log_file:
        subprocess.run(
            [str(fm_exe), str(command_file.name)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=output_directory
        )

    # stack the rasters into a multi-band output, if specified
    if stack_out is True:
        stack_rasters(
            in_dir=output_directory,
            cleanup=cleanup
        )


def get_feature_service_gdf(url, geo=None, qry='1=1', layer=0):
    """
    GeoDataFrame from a Feature Service from url and optional bounding geometry and where clause

    :param url:
    :param geo:
    :param qry:
    :param layer:
    :return:
    """

    # Gather info from the Feature Service
    s_info = requests.get(url + '?f=pjson').json()  # json metadata
    srn = s_info['spatialReference']['wkid']  # spatial reference
    sr = 'EPSG:' + str(srn)

    # Handle the bounding geometry if provided, else return all
    if geo is not None:
        # Check on the instance type for the bounding geometry
        if isinstance(geo, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geo = geo.to_crs(sr).total_bounds
        elif isinstance(geo, shapely.geometry.base.BaseGeometry):
            geo = gpd.GeoSeries([geo], crs='EPSG:4326').to_crs(sr).total_bounds
        elif isinstance(geo, (list, tuple, np.ndarray)) and len(geo) == 4:
            geo = geo
        else:
            raise ValueError("Invalid geometry input.")
        # return the bounding geometry tuple
        geo = ','.join(np.array(geo).astype(str))
    else:
        geo = None  # otherwise return none for bounding

    # Extract the correct URL for the Feature Service layer
    url1 = url + '/' + str(layer)  # adds the layer identifier (eg, 0)
    # Get the Feature Service metadata information
    l_info_resp = requests.get(url1 + '?f=pjson')
    try:
        l_info = l_info_resp.json()
        maxrcn = int(l_info.get('maxRecordCount', 1000))  # fallback to 1000
    except Exception as e:
        print(f"[Warning] Could not parse layer metadata from {url1}")
        print(f"Response text: {l_info_resp.text[:300]}...")
        maxrcn = 1000  # fallback default
    # set the base url
    url2 = url1 + '/query?'  # base URL for service requests
    # Get a list of Object IDs (OIDs) for features matching the filter
    o_info = requests.get(
        url2, {
            'where': qry,
            'geometry': geo,
            'geometryType': 'esriGeometryEnvelope',
            'returnIdsOnly': 'True',
            'f': 'pjson'
        }).json()
    # Gather the OIDs
    oid_name = o_info['objectIdFieldName']
    oids = o_info['objectIds']
    numrec = len(oids)  # number of records returned

    # Gather the list of features
    fslist = []
    for i in range(0, numrec, maxrcn):
        objectIds = oids[i:i + maxrcn]
        idstr = oid_name + ' in (' + str(objectIds)[1:-1] + ')'
        prm = {
            'where': idstr,
            'outFields': '*',
            'returnGeometry': 'true',
            'outSR': srn,
            'f': 'pgeojson',
        }
        response = requests.post(url2, data=prm)

        # Fallback to standard geojson if pgeojson fails
        try:
            ftrs = response.json()['features']
        except (requests.exceptions.JSONDecodeError, KeyError):
            prm['f'] = 'geojson'
            response = requests.post(url2, data=prm)
            try:
                ftrs = response.json()['features']
            except Exception as e:
                raise RuntimeError(
                    f"Failed to retrieve features from {url2}\nResponse text: {response.text[:300]}...") from e

        fslist.append(gpd.GeoDataFrame.from_features(ftrs, crs=sr))

    valid_fslist = [gdf for gdf in fslist if not gdf.empty]
    if not valid_fslist:
        return gpd.GeoDataFrame(columns=['geometry'])  # or however you want to handle the empty case

    return gpd.pd.concat([gdf.dropna(axis=1, how='all') for gdf in valid_fslist], ignore_index=True)


def check_service_type(url):
    """
    Checks for either Map/Feature or Image in a service URL.

    Parameters
    ----------
    url : TYPE
        REST service URL.

    Returns
    -------
    STRING.

    """
    if 'ImageServer' in url:
        return 'Image'
    elif 'FeatureServer' in url:
        return 'Feature'
    elif 'MapServer' in url:
        return 'Feature'
    else:
        return 'Unknown'


def download_file(img_url, outname, chunk_size=1024):
    try:
        with requests.get(img_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(outname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
        raise


def get_image_service_array(url, ply, out_prefix, res=30, outSR=3857,
                            outdir=None, cleanup=True, plot=False):
    """
    Downloads raster imagery from an ArcGIS REST ImageService for a given polygon.

    Parameters:
    ----------
    url : str
        REST endpoint of the ImageServer
    ply : GeoDataFrame or GeoSeries
        Area of interest (single polygon or feature)
    out_prefix : str
        Prefix for naming downloaded image tiles
    res : int
        Pixel resolution (in meters)
    outSR : str or int
        Output spatial reference (e.g., 5070); default uses service SR
    output_dir : str or None
        Directory to save tiles. If None, uses a temporary directory.
    cleanup : bool
        Whether to delete files after processing (if using temp directory)

    Returns:
    --------
    xarray.DataArray
        Combined raster from all tiles (or single tile)
    """

    # Query the metadata from the service (url)
    meta = requests.get(url + '?f=pjson').json()

    if 'spatialReference' not in meta:
        # Try parent layer for spatial reference
        print(f"[WARNING] No spatialReference found for {url}, checking parent...")
        parent_url = '/'.join(url.strip('/').split('/')[:-1])
        parent_meta = requests.get(parent_url + '?f=pjson').json()
        if 'spatialReference' not in parent_meta:
            raise ValueError(f"No spatial reference found in metadata for {url} or its parent.")
        spr = parent_meta['spatialReference']
    else:
        spr = meta['spatialReference']

    epsg = spr.get('latestWkid') or spr.get('wkid')
    if epsg is None:
        raise ValueError("Unable to determine EPSG code from spatialReference.")

    # --- map ESRI WKIDs to real EPSG if needed
    # avoid pyproj error for 102039 by using EPSG:5070 (CONUS Albers)
    esri_to_epsg = {102039: 5070, 102003: 5070}
    epsg_use = esri_to_epsg.get(epsg, epsg)

    # Use default max dimensions if missing
    max_w = meta.get('maxImageWidth', 4096)
    max_h = meta.get('maxImageHeight', 4096)

    # Reproject the region of interest to the correct CRS
    # Extract the bounding geometry (optional buffer)
    if ply.crs is None:
        raise ValueError("Input geometry must have a defined CRS.")
    ply_proj = ply.to_crs(epsg_use)
    xmin, ymin, xmax, ymax = ply_proj.total_bounds

    # Computing the tiling extents
    wcells = int((xmax - xmin) / res)
    hcells = int((ymax - ymin) / res)
    tile_w = min(wcells, max_w)
    tile_h = min(hcells, max_h)
    wcells_l = np.arange(0, wcells, tile_w)
    hcells_l = np.arange(0, hcells, tile_h)

    # Prepare output directory
    temp_dir = None
    if outdir is None:
        temp_dir = tempfile.mkdtemp()
        outdir = temp_dir
    os.makedirs(outdir, exist_ok=True)

    # Decide export endpoint and layer handling
    # detect MapServer layer vs ImageServer
    url_parts = url.strip('/').split('/')
    is_mapserver_layer = 'MapServer' in url_parts and url_parts[-2] == 'MapServer' and url_parts[-1].isdigit()
    if is_mapserver_layer:
        parent_url = '/'.join(url_parts[:-1])  # .../MapServer
        layer_id = url_parts[-1]               # e.g., "0" for 2024
        export_url = parent_url + '/export'
    else:
        # assume ImageServer
        export_url = url + '/exportImage'

    # Download the tiles
    rasters = []
    tile = 1
    for w in wcells_l:
        for h in hcells_l:
            xmin2 = xmin + w * res
            xmax2 = min(xmin + (w + tile_w) * res, xmax)
            ymin2 = ymin + h * res
            ymax2 = min(ymin + (h + tile_h) * res, ymax)

            if is_mapserver_layer:
                # <<< CHANGED: MapServer export params (no 'where'; use layers=show:<id>)
                params = {
                    'f': 'json',
                    'bbox': f"{xmin2},{ymin2},{xmax2},{ymax2}",
                    'bboxSR': epsg_use,  # <<< CHANGED
                    'imageSR': outSR,
                    'size': f"{tile_w},{tile_h}",
                    'format': 'tiff',
                    'transparent': 'false',
                    'layers': f"show:{layer_id}"  # <<< CHANGED
                }
                qry = export_url
            else:
                # ImageServer exportImage (your original flow)
                params = {
                    'f': 'json',
                    # 'where': 'Year = 2020',   # <<< CHANGED: remove; not valid for MapServer layer
                    'bbox': f"{xmin2},{ymin2},{xmax2},{ymax2}",
                    'size': f"{tile_w},{tile_h}",
                    'imageSR': outSR,
                    'bboxSR': epsg_use,  # <<< CHANGED: include bboxSR too
                    'format': 'tiff'
                }
                qry = export_url

            resp = requests.get(qry, params, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                img_url = result.get('href') or result.get('imageUrl')

                if img_url is None:
                    print(f"[WARNING] Tile {tile} returned no image URL.")
                    continue  # Skip to next tile

                out_fp = os.path.join(outdir, f"{out_prefix}_{tile}.tif")
                download_file(img_url, out_fp)
                rasters.append(rxr.open_rasterio(out_fp, masked=True).squeeze())
                tile += 1
            else:
                print(f"[WARNING] Tile {tile} failed: HTTP {resp.status_code}")

    if not rasters:
        raise ValueError("No tiles were successfully downloaded.")

    # 6. Merge tiles
    result = rasters[0] if len(rasters) == 1 else xr.combine_by_coords(rasters)

    # 7. Cleanup
    if cleanup and temp_dir is not None:
        shutil.rmtree(temp_dir)

    # 8. Plot
    if plot is True:
        result.plot(cmap='viridis')
        # plt.title(f"({key})")
        plt.show()

    return result

def fuelscape_adjustment(fs,
                         evt_code=7050,
                         cbh_adjust=0.70,
                         fbfm_to=185):
    """
    Adjusts a fuelscape (multi-band geotiff) for a given EVT code.
    For the specified code, apply an adjustment to individual or multiple bands.

    :param fs: Input fuelscape (multi-band geotiff)
    :param evt_code: The EVT code to make adjustments to
    :param cbh_adjust: Canopy base height adjustment (% of baseline)
    :param fbfm_to: Fuel model adjustment (int, fuel code to assign)
    :return: modified multiband geotiff
    """

    fs = fs.copy()

    # create a binary mask for the EVT code
    evt_mask = fs['EVT'] == evt_code

    # make sure there are some pixels of that EVT code
    # run the adjustments on the appropriate bands
    if evt_mask.any():

        # adjust the canopy base height
        if 'CBH' in fs and cbh_adjust is not None:
            fs['CBH'] = xr.where(evt_mask, fs['CBH'] * 0.7, fs['CBH'])
        else:
            print("CBH not found or adjustment factor not given!")

        # adjust the fuel model
        if 'FBFM40' in fs and fbfm_to is not None:
            fs['FBFM40'] = xr.where(evt_mask, 185, fs['FBFM40'])
        else:
            print("FBFM40 not found or adjustment factor not given!")

    else:
        print(f"No pixels of EVT code [{evt_code}] found in fuelscape.")

        # def delta_envc_landscape(self,
        #                          units: dict,
        #                          region_col: str = "RegionID",
        #                          aoi: gpd.GeoDataFrame = None):
        #     """
        #
        #     :param units:
        #     :param region_col:
        #     :param aoi:
        #     :return:
        #     """
        #
        #     envc_files = self.dd["eNVC"] # eNVC file paths
        #     feas_files = self.dd["Feas"] # treatment feasibility files
        #
        #     # normalize scenario keys (longest-first prevents prefix collisions with complete treatments)
        #     keys_norm = [(k, k.lower()) for k in sorted(self.trts_keys, key=len, reverse=True)]
        #     scenario_alias = {
        #         "patch_sm": ["patch"],  # patch_feas.tif, patch_cost.tif, etc.
        #         # add more if needed:
        #         # "mechRxFire": ["mechRxFire", "mech_rxfire", "mechRx"],
        #     }
        #
        #     def _match_scenario(fp: str):
        #         stem = Path(fp).stem.lower()
        #         # 1) alias match: scenario name -> prefix list
        #         for sc, prefixes in scenario_alias.items():
        #             for pref in prefixes:
        #                 pref = pref.lower()
        #                 if stem.startswith(pref + "_") or stem == pref or stem.startswith(pref):
        #                     return sc
        #         # 2) default match: scenario key itself
        #         return next(
        #             (orig for orig, low in keys_norm if stem.startswith(low + "_") or stem == low),
        #             None
        #         )
        #
        #     # build scenario -> feas raster lookup (keep first match if multiple)
        #     feas_by_scenario = {}
        #     for fp in feas_files:
        #         sc = _match_scenario(fp)
        #         if sc is None:
        #             continue
        #         feas_by_scenario.setdefault(sc, fp)
        #
        #     if not feas_by_scenario:
        #         raise ValueError("! Could not match any feasibility rasters to treatment scenario keys.")
        #     print(f"{feas_by_scenario}\n")
        #
        #     results = []
        #     for hvra in self.hvras:
        #         cat_bl = "Composite" if hvra in ("Total", "Composite") else hvra
        #         # baseline eNVC raster for this HVRA
        #         bl_candidates = [
        #             f for f in envc_files
        #             if "baseline" in f.lower() and cat_bl.lower() in basename(f).lower()
        #         ]
        #         if not bl_candidates:
        #             print(f"! No baseline eNVC raster found for HVRA={hvra}. Skipping.")
        #             continue
        #
        #         baseline_path = bl_candidates[0]
        #         print(f"HVRA={hvra} baseline={baseline_path}")
        #
        #         # load baseline raster once (for mask alignment + nodata)
        #         baseline_da = rxr.open_rasterio(baseline_path).squeeze()
        #         raster_crs = baseline_da.rio.crs
        #         nodata_val = baseline_da.rio.nodata
        #         transform = baseline_da.rio.transform()
        #
        #         # ensure float if nodata is NaN
        #         if nodata_val is not None and np.isnan(nodata_val):
        #             baseline_da = baseline_da.astype("float32")
        #
        #         for unit_type, gdf in units.items():
        #             if region_col not in gdf.columns:
        #                 raise KeyError(f"{unit_type} units missing required column: {region_col}")
        #
        #             # aoi: GeoDataFrame or GeoSeries passed into the function
        #             aoi_proj = aoi.to_crs(raster_crs)
        #             aoi_geom = aoi_proj.geometry.unary_union  # single geometry
        #
        #             # dissolve the regions
        #             gdf_proj = gdf.to_crs(raster_crs)
        #             gdf_proj = (
        #                 gdf_proj[[region_col, "geometry"]]
        #                 .dissolve(by=region_col)
        #                 .reset_index()
        #             ) # make sure we have a single unit
        #
        #             # calculate the overlap area within the AOI
        #             gdf_proj["geometry"] = gdf_proj.geometry.intersection(aoi_geom).copy()
        #             gdf_proj = gdf_proj[~gdf_proj.geometry.is_empty] # drop empties
        #
        #             # compute acres inside AOI
        #             region_acres = gdf_proj[[region_col]].copy()
        #             region_acres["RegionAcres"] = gdf_proj.geometry.area / 4046.86
        #
        #             # baseline total eNVC (not scenario-specific)
        #             total_envc = compute_band_stats(
        #                 gdf_proj, baseline_path,
        #                 id_col=region_col,
        #                 attr="eNVC_total",
        #                 stats=["sum"],
        #                 ztype="continuous"
        #             )
        #             total_envc = pd.merge(total_envc, region_acres, on=region_col, how="left")
        #
        #             # scenario-specific feasible eNVC
        #             for scenario, feas_fp in feas_by_scenario.items():
        #                 # load + align feasibility mask once per scenario per HVRA/unit_type
        #                 feas_da = rxr.open_rasterio(feas_fp).squeeze()
        #                 feas_da = feas_da.rio.reproject_match(baseline_da)
        #
        #                 # apply feasibility (assumes feasible==1)
        #                 masked_da = baseline_da.where(feas_da == 1)
        #
        #                 feas_envc = compute_band_stats(
        #                     gdf_proj, masked_da.data,
        #                     id_col=region_col,
        #                     attr="eNVC_feasible",
        #                     stats=["sum"],
        #                     ztype="continuous",
        #                     nodata=nodata_val,
        #                     transform=transform
        #                 )
        #
        #                 # tidy columns
        #                 df = pd.merge(total_envc, feas_envc, on=region_col, how="left")
        #                 df.columns = [c.replace("_sum", "") for c in df.columns]
        #
        #                 df["RegionType"] = unit_type
        #                 df["RegionID"] = df[region_col]
        #                 df["HVRA"] = hvra
        #                 df["scenario"] = scenario
        #
        #                 results.append(df[[
        #                     "RegionType", "RegionID", "RegionAcres", "HVRA",
        #                     "scenario", "eNVC_total", "eNVC_feasible"
        #                 ]])
        #
        #     landscape_envc = pd.concat(results, ignore_index=True)
        #
        #     return landscape_envc