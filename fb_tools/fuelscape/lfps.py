"""
LANDFIRE Product Service (LFPS) API client.

Submits a job to the LFPS REST API, polls until complete, downloads the
resulting ZIP, and optionally renames the output files.  For spatially
dispersed inputs (multiple SFAs, treatment clusters) use ``lfps_mosaic``
to download one tile per feature and merge into a single GeoTIFF.

Reference: https://lfps.usgs.gov/LFProductsServiceUserGuide.pdf
"""

import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import requests


_LFPS_SUBMIT_URL = "https://lfps.usgs.gov/api/job/submit"
_LFPS_STATUS_URL = "https://lfps.usgs.gov/api/job/status"

# Add entries here as new versions are verified against the LFPS products table:
# https://lfps.usgs.gov/products


def _read_band_names(src):
    """Return a tuple of band name strings from an open rasterio dataset.

    Tries ``src.descriptions`` (rasterio band descriptions) first.  Falls
    back to the per-band ``long_name`` metadata tag written by rioxarray,
    because LFPS GeoTIFFs store layer names as per-band tags rather than
    as GDAL band descriptions.
    """
    descriptions = src.descriptions
    if any(descriptions):
        return descriptions
    return tuple(
        src.tags(i).get("long_name") for i in range(1, src.count + 1)
    )


def lfps_request(
    region,
    out_dir,
    lf_year,
    layer_list=None,
    edit_rules=None,
    lodgepole_adjust=False,
    rename=None,
    email="Maxwell.Cook@colostate.edu",
    out_crs=5070,
    poll_interval=30,
    clip=False,
    max_retries=3,
):
    """
    Submit and download a LANDFIRE Products Service (LFPS) job.

    Downloads a multi-band GeoTIFF containing standard landscape layers
    (elevation, slope, aspect, FBFM40, canopy cover/height/base height/
    bulk density, and EVT) for the bounding box of *region*.

    Parameters
    ----------
    region : GeoDataFrame
        Area of interest. The bounding box (in WGS-84) is sent to LFPS.
    out_dir : str or Path
        Directory where extracted files are written.
    lf_year : str
        LANDFIRE year (e.g., 2023).
    layer_list : str, optional
        Semi-colon-delimited LFPS layer string.  Defaults to the standard
        8-layer landscape stack (topo + fuel + EVT) using *lf_version*.
    edit_rules : dict, optional
        Custom LFPS edit-rule JSON.  Mutually exclusive with
        *lodgepole_adjust*.
    lodgepole_adjust : bool
        Apply the standard CFRI lodgepole-pine fuel adjustment (FBFM 181/183
        → 185; CBH × 0.70) for EVT code 7050.  Ignored when *edit_rules*
        is provided.
    rename : str, optional
        Base filename (no extension) to rename the downloaded files.
    email : str
        Email address required by the LFPS API.
    out_crs : int
        EPSG code for the output projection (default ``5070``, Albers).
    poll_interval : int
        Seconds to wait between status checks (default ``30``).
    clip : bool
        If ``True``, clip the downloaded raster to the union of *region*
        geometries after download.  Removes excess coverage outside the AOI
        that the bounding-box download always includes.  Default ``False``.
    max_retries : int
        Number of times to resubmit the job if the LFPS service returns a
        ``"Failed"`` or ``"Cancelled"`` status.  The LFPS API occasionally
        throws a transient server-side error (e.g. ``'GPEnvironment' object
        has no attribute 'pyramid'``) that resolves on retry.  Default ``3``.
    """
    codes = {
        "FBFM40": f"LF{lf_year}_FBFM40",
        "CC":     f"LF{lf_year}_CC",
        "CH":     f"LF{lf_year}_CH",
        "CBH":    f"LF{lf_year}_CBH",
        "CBD":    f"LF{lf_year}_CBD",
        "EVT":    f"LF{lf_year}_EVT",
    }
    # --- 1. Build the layer list for the requested vintage
    if layer_list is None:
        layer_list = (
            f"LF2020_Elev;LF2020_SlpD;LF2020_Asp;" # default 2020 topography
            f"{codes['FBFM40']};"
            f"{codes['CC']};"
            f"{codes['CH']};"
            f"{codes['CBH']};"
            f"{codes['CBD']};"
            f"{codes['EVT']}"
        )
    print(f"Requesting from LFPS: {layer_list}")

    # --- 2. Lodgepole adjustment rule (standard CFRI rule)
    lp_adjustment_rule = {
        "edit": [
            {
                "condition": [
                    {"product": codes["EVT"],   "operator": "EQ", "value": 7050},
                    {"product": codes["FBFM40"], "operator": "EQ", "value": 181},
                ],
                "change": [{"product": codes["FBFM40"], "operator": "ST", "value": 185}],
            },
            {
                "condition": [
                    {"product": codes["EVT"],   "operator": "EQ", "value": 7050},
                    {"product": codes["FBFM40"], "operator": "EQ", "value": 183},
                ],
                "change": [{"product": codes["FBFM40"], "operator": "ST", "value": 185}],
            },
            {
                "condition": [
                    {"product": codes["EVT"], "operator": "EQ", "value": 7050},
                ],
                "change": [{"product": codes["CBH"], "operator": "MB", "value": 0.7}],
            },
        ]
    }

    # --- 3. Format bounding box (LFPS requires WGS-84 lon/lat)
    aoi = region.to_crs(4326)
    bbox = aoi.total_bounds
    bbox_str = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"

    # --- 4. Build request params
    params = {
        "Layer_List": layer_list,
        "Area_of_Interest": bbox_str,
        "Output_Projection": out_crs,
        "Email": email,
    }
    if edit_rules:
        params["Edit_Rule"] = json.dumps(edit_rules)
    elif lodgepole_adjust:
        params["Edit_Rule"] = json.dumps(lp_adjustment_rule)

    # --- 5–6. Submit job and poll (with retry on transient server failures)
    data = {}
    for attempt in range(1, max_retries + 1):
        # Submit fresh job each attempt
        r = requests.get(_LFPS_SUBMIT_URL, params=params)
        r.raise_for_status()
        job_id = r.json()["jobId"]
        print(f"Job submitted (attempt {attempt}/{max_retries}). ID: {job_id}")

        # Poll
        status_url = f"{_LFPS_STATUS_URL}?JobId={job_id}"
        job_failed = False
        while True:
            try:
                resp = requests.get(status_url)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "Unknown")
                print(f"[{time.ctime()}] Status: {status}")

                if status == "Succeeded":
                    break
                elif status in ("Failed", "Cancelled"):
                    msgs = data.get("messages", [])
                    print(f"[lfps_request] Job {status} on attempt {attempt}.")
                    for m in msgs:
                        if m.get("type", "").endswith("Error"):
                            print(f"  ERROR: {m['description']}")
                    job_failed = True
                    break
            except requests.exceptions.RequestException as exc:
                print(f"[{time.ctime()}] Request error: {exc} — retrying in {poll_interval}s")

            time.sleep(poll_interval)

        if not job_failed:
            break  # Succeeded — exit retry loop

        if attempt < max_retries:
            print(f"[lfps_request] Retrying in {poll_interval}s ...")
            time.sleep(poll_interval)
        else:
            raise RuntimeError(
                f"LFPS job failed after {max_retries} attempt(s). "
                f"Last messages: {data.get('messages', '')}"
            )

    # --- 7. Download and extract ZIP
    download_url = data["outputFile"]
    print(f"Downloading: {download_url}")
    r = requests.get(download_url)
    r.raise_for_status()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out_dir)
    print(f"Extracted to: {out_dir}")

    # --- 8. Optional rename
    if rename:
        tif_path = next(out_dir.glob("*.tif"))
        base_old = tif_path.stem
        for ext in (".tif", ".tfw", ".tif.aux.xml", ".tif.xml"):
            old = out_dir / f"{base_old}{ext}"
            new = out_dir / f"{rename}{ext}"
            if old.exists():
                old.rename(new)
        print(f"Renamed files to: {rename}.*")
        out_path = out_dir / f"{rename}.tif"
    else:
        tif_path = next(out_dir.glob("*.tif"))
        out_path = out_dir / tif_path.name

    # --- 9. Optional clip to region geometry
    if clip:
        import rasterio
        from rasterio.mask import mask as rio_mask

        with rasterio.open(out_path) as src:
            shapes = list(region.to_crs(src.crs).geometry)
            clipped, clipped_transform = rio_mask(src, shapes, crop=True)
            profile = src.profile.copy()
            descriptions = _read_band_names(src)

        profile.update(
            width=clipped.shape[2],
            height=clipped.shape[1],
            transform=clipped_transform,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(clipped)
            dst.descriptions = descriptions

        print(f"Clipped output to region boundary: {out_path}")

    return out_path


def lfps_mosaic(
    regions_gdf,
    out_dir,
    lf_year,
    layer_list=None,
    edit_rules=None,
    lodgepole_adjust=False,
    rename=None,
    email="Maxwell.Cook@colostate.edu",
    out_crs=5070,
    poll_interval=30,
    keep_tiles=False,
    clip=False,
    clip_to=None,
    max_retries=3,
):
    """
    Download one LFPS tile per feature in *regions_gdf* and merge into a
    single GeoTIFF.

    Use this instead of ``lfps_request`` when your area of interest is a
    multipolygon or a set of dispersed polygons (e.g., multiple SFAs or
    treatment clusters).  Each feature is downloaded individually using
    its own bounding box, avoiding the large empty extents that
    ``total_bounds`` would produce for scattered geometries.  The tiles
    are then merged with ``rasterio.merge`` and written as a single
    output file.

    Parameters
    ----------
    regions_gdf : GeoDataFrame
        One or more polygons defining the areas of interest.  Each row
        triggers a separate LFPS download.
    out_dir : str or Path
        Directory where the merged output (and optionally tiles) are written.
    lf_year : str
        LANDFIRE year (e.g., ``"2023"``).
    layer_list : str, optional
        Semi-colon-delimited LFPS layer string.  Passed through to
        ``lfps_request`` unchanged.
    edit_rules : dict, optional
        Custom LFPS edit-rule JSON.  Mutually exclusive with
        *lodgepole_adjust*.
    lodgepole_adjust : bool
        Apply the standard CFRI lodgepole-pine fuel adjustment.
    rename : str, optional
        Base filename (no extension) for the merged output.  Defaults to
        ``"mosaic"``.
    email : str
        Email address required by the LFPS API.
    out_crs : int
        EPSG code for the output projection (default ``5070``, Albers).
    poll_interval : int
        Seconds to wait between LFPS status checks (default ``30``).
    keep_tiles : bool
        If ``True``, keep the per-feature tile directories under
        ``out_dir/tiles/``.  If ``False`` (default), the tile directory
        is deleted after the merge.
    clip : bool
        Passed through to ``lfps_request`` for each tile.  When ``True``,
        each tile is clipped to its feature polygon before merging, so the
        final mosaic has no excess coverage.  Default ``False``.
    clip_to : GeoDataFrame, optional
        If provided, clip the final merged mosaic to the union of these
        geometries after all tiles have been combined.  Useful when the
        clip boundary (e.g., a project area or SFA outline) differs from
        the per-tile download polygons in *regions_gdf*.  Geometries are
        reprojected to the mosaic CRS automatically.  Default ``None``.
    max_retries : int
        Passed through to ``lfps_request`` for each tile.  Default ``3``.

    Returns
    -------
    Path
        Path to the merged GeoTIFF.
    """
    import rasterio
    from rasterio.merge import merge as rio_merge

    out_dir = Path(out_dir)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    out_name = rename or "mosaic"

    # --- 1. Download one tile per feature
    tile_paths = []
    for idx, row in regions_gdf.reset_index(drop=True).iterrows():
        tile_subdir = tiles_dir / f"tile_{idx:03d}"
        single = regions_gdf.iloc[[idx]]  # keep as GeoDataFrame (1-row)
        print(f"\n[lfps_mosaic] Tile {idx + 1}/{len(regions_gdf)} ...")
        tile_path = lfps_request(
            region=single,
            out_dir=tile_subdir,
            lf_year=lf_year,
            layer_list=layer_list,
            edit_rules=edit_rules,
            lodgepole_adjust=lodgepole_adjust,
            rename=f"tile_{idx:03d}",
            email=email,
            out_crs=out_crs,
            poll_interval=poll_interval,
            clip=clip,
            max_retries=max_retries,
        )
        tile_paths.append(tile_path)

    # --- 2. Merge tiles into a single GeoTIFF
    print(f"\n[lfps_mosaic] Merging {len(tile_paths)} tile(s) ...")
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_arr, mosaic_transform = rio_merge(datasets)
        profile = datasets[0].profile.copy()
    finally:
        for ds in datasets:
            ds.close()

    profile.update(
        width=mosaic_arr.shape[2],
        height=mosaic_arr.shape[1],
        transform=mosaic_transform,
    )

    # Propagate band names from the first tile
    with rasterio.open(tile_paths[0]) as src_check:
        descriptions = _read_band_names(src_check)

    out_path = out_dir / f"{out_name}.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic_arr)
        dst.descriptions = descriptions

    print(f"[lfps_mosaic] Merged output: {out_path}")

    # --- 3. Optional clip of final mosaic to a separate region
    if clip_to is not None:
        from rasterio.mask import mask as rio_mask

        with rasterio.open(out_path) as src:
            shapes = list(clip_to.to_crs(src.crs).geometry)
            clipped, clipped_transform = rio_mask(src, shapes, crop=True)
            profile = src.profile.copy()
            descriptions = _read_band_names(src)

        profile.update(
            width=clipped.shape[2],
            height=clipped.shape[1],
            transform=clipped_transform,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(clipped)
            dst.descriptions = descriptions

        print(f"[lfps_mosaic] Clipped final mosaic to clip_to region: {out_path}")

    # --- 4. Optionally remove tile directories
    if not keep_tiles:
        shutil.rmtree(tiles_dir)
        print(f"[lfps_mosaic] Tile directory removed.")

    return out_path
