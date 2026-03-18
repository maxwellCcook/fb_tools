"""
LANDFIRE Product Service (LFPS) API client.

Submits a job to the LFPS REST API, polls until complete, downloads the
resulting ZIP, and optionally renames the output files.

Reference: https://lfps.usgs.gov/LFProductsServiceUserGuide.pdf
"""

import io
import json
import time
import zipfile
from pathlib import Path

import requests


_LFPS_SUBMIT_URL = "https://lfps.usgs.gov/api/job/submit"
_LFPS_STATUS_URL = "https://lfps.usgs.gov/api/job/status"

# Add entries here as new versions are verified against the LFPS products table:
# https://lfps.usgs.gov/products

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

    # --- 5. Submit job
    r = requests.get(_LFPS_SUBMIT_URL, params=params)
    r.raise_for_status()
    job_id = r.json()["jobId"]
    print(f"Job submitted. ID: {job_id}")

    # --- 6. Poll for completion
    status_url = f"{_LFPS_STATUS_URL}?JobId={job_id}"
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
                raise RuntimeError(
                    f"LFPS job {status}. Messages: {data.get('messages', '')}"
                )
        except requests.exceptions.RequestException as exc:
            print(f"[{time.ctime()}] Request error: {exc} — retrying in {poll_interval}s")

        time.sleep(poll_interval)

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

        return out_dir / f"{rename}.tif"
    else:
        tif_path = next(out_dir.glob("*.tif"))
        base_old = tif_path.stem
        return out_dir / f"{base_old}.tif"
