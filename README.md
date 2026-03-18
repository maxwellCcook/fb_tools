# fb_tools

Python wrapper for CLI fire behavior models (FlamMap, FSPro, FSim, MTT) with supporting tools for landscape fuel preparation, fire behavior simulation, and suppression difficulty analysis.

---

## Overview

`fb_tools` is organized around a typical fire modeling workflow:

1. **Fuelscape** — download LANDFIRE layers via the LFPS API, build and adjust LCP rasters
2. **Models** — run FlamMap scenarios in batch; organize outputs
3. **Suppression** — compute Suppression Difficulty Index (SDI) from fire behavior outputs and road/trail networks
4. **Utils** — shared geospatial helpers used across sub-packages

Fire behavior model executables (FlamMap, FSPro, FSim, MTT) are Windows-only. Data preparation and fuelscape workflows run on macOS or Windows. The package is used on macOS (data prep) and a Windows VM via Parallels (model runs).

---

## Installation

The geospatial stack must be installed via conda before the package itself.

```bash
# 1. Create the conda environment (run from repo root)
conda env create -f environment.yml
conda activate fb_tools

# 2. Install fb_tools in editable mode
#    --no-deps is critical: prevents pip from overwriting conda's GDAL/rasterio DLLs
pip install -e . --no-deps
```

> **Windows note:** Always create and activate the environment from **Anaconda Prompt**, not PowerShell. This ensures the correct DLL search path for GDAL/PROJ/rasterio.

---

## Package Structure

```
fb_tools/
├── fuelscape/
│   ├── lfps.py       LANDFIRE Product Service REST API client
│   ├── lcp.py        LCP raster stacking and ignition grid utilities
│   └── adjust.py     Fuel treatment adjustments (canopy, surface fuel)
├── models/
│   ├── base.py       Shared subprocess runner for CLI executables
│   ├── flammap.py    FlamMap scenario wrapper
│   └── scenarios.py  Scenario loading, building, and batch execution
├── suppression/
│   ├── roads.py      OSM road and trail network extraction
│   └── sdi.py        Suppression Difficulty Index (SDI) calculator
├── utils/
│   ├── geo.py        Geospatial helpers (mask, reproject, rasterize)
│   ├── io.py         File listing utility
│   └── plot.py       Multi-band raster plotting
└── weather/          (stub — GridMET and RAWS planned)
```

All public functions are importable directly from `fb_tools`:

```python
from fb_tools import (
    # fuelscape
    lfps_request, stack_rasters, create_ignition_ascii, get_band_by_longname,
    adjust_lcp, apply_treatment, build_surface_lut,
    # models
    run_flammap_scenarios, load_scenarios, build_scenarios, run_batch,
    # suppression
    fetch_osm_roads, calculate_sdi, calculate_delta_sdi,
    # utils
    is_valid_geom, mask_raster, geom_to_raster_crs, rasterize,
    list_files, plot_bands,
)
```

---

## Sub-packages

### `fb_tools.fuelscape` — Landscape Fuel Preparation

Download LANDFIRE layers and build adjusted LCP rasters for fire behavior modeling.

#### Download from LFPS

```python
from fb_tools import lfps_request

# aoi_gdf: GeoDataFrame with project boundary (any CRS)
lcp_path = lfps_request(
    region=aoi_gdf,
    out_dir="data/spatial/lcp/",
    lf_version="200",              # LANDFIRE 2.0.0
    lodgepole_adjust=True,         # optional lodgepole pine treatment
    out_crs=26913,                 # NAD83 UTM Zone 13N
)
```

Layers downloaded by default for LF 2.0.0: ELEV, SLP, ASP, FBFM40, CC, CH, CBH, CBD, EVT.

#### Stack single-band rasters into a multi-band LCP

```python
from fb_tools import stack_rasters

stack_rasters(in_dir="data/spatial/lcp/raw/", tag="BASELINE", cleanup=True)
# → writes data/spatial/lcp/raw/PCT90_BASELINE.tif
```

#### Apply fuel treatments

```python
from fb_tools import apply_treatment

lcp_treated = apply_treatment(
    lcp="data/lcp/baseline.tif",
    canopy_df=canopy_treatments,   # DataFrame with adjustment factors per scenario
    surface_df=surface_treatments, # DataFrame with FBFM40 remapping per scenario
    scenario={"canopy": "mech_thin", "surface": "mech_thin"},
    mask=treatment_mask,           # optional spatial mask (DataArray)
)
```

---

### `fb_tools.models` — Fire Behavior Simulation

Run FlamMap scenarios individually or in batch. Executables are Windows-only; paths to `.exe` files are typically on a mapped network drive or Parallels shared folder.

#### Define scenarios from a CSV

```python
from fb_tools import load_scenarios, run_batch

scenarios = load_scenarios("data/fire_scenarios.csv", lcp_dir="data/lcp/")
results   = run_batch(
    fm_exe="Z:/code/FB/TestFlamMap/TestFlamMap.exe",
    scenarios_df=scenarios,
    output_root="data/outputs/flammap/",
    n_process=8,
    stack_out=True,
)
print(results[["Scenario", "LCP", "status"]])
```

#### Build scenarios programmatically

```python
from fb_tools import build_scenarios, run_batch
import pandas as pd

conditions = pd.DataFrame({
    "Scenario":    ["pct90_wind270", "pct90_wind315"],
    "WIND_SPEED":  [30, 30],
    "WIND_DIRECTION": [270, 315],
    "FM_1hr": [3], "FM_10hr": [4], "FM_100hr": [6],
    "FM_herb": [60], "FM_woody": [60],
    "CROWN_FIRE_METHOD": ["ScottReinhardt"],
    "Outputs": ["FLAMELENGTH, HEATAREA, CROWNSTATE"],
})

scenarios = build_scenarios(conditions, lcps=["baseline.lcp", "mech_thin.lcp"])
results   = run_batch(fm_exe=..., scenarios_df=scenarios, output_root=...)
```

**Required CSV columns:** `Scenario`, `LCP`, `WIND_SPEED`, `WIND_DIRECTION`, `FM_1hr`, `FM_10hr`, `FM_100hr`, `FM_herb`, `FM_woody`, `CROWN_FIRE_METHOD`, `Outputs`

**Output organization:** `output_root/{lcp_stem}/{scenario_name}/`

---

### `fb_tools.suppression` — Suppression Difficulty Index

Implements the Rodriguez y Silva et al. (2014) SDI framework in pure Python (numpy/scipy/geopandas), replacing the original ArcPy workflow.

SDI combines five sub-indices:
- **Accessibility** — Euclidean distance to the nearest road
- **Penetrability** — slope, aspect, trail density, and fuel control difficulty
- **Energy Behavior** — harmonic mean of flame length and heat unit area
- **Fireline Opening** — fuel difficulty adjusted for hand and machine crew slope limits
- **KO Slope Mobility Hazard** — added for areas beyond road access

**Delta SDI** (baseline − treatment) quantifies how much a fuel treatment improves suppression conditions.

#### Extract OSM road and trail networks

```python
from fb_tools import fetch_osm_roads

# aoi_gdf: project boundary GeoDataFrame (any CRS)
roads_gdf, trails_gdf = fetch_osm_roads(aoi_gdf, out_crs=26913)
# roads_gdf: highway_code 1–7 used for accessibility (motorway → residential)
# trails_gdf: path/footway/bridleway/track used for penetrability density
```

#### Calculate SDI

```python
from fb_tools import calculate_sdi, calculate_delta_sdi

rtc_path = "code/dev/SDI/08_RTC_lookup_SDIwt_westernUS_2021_update.txt"

sdi_baseline = calculate_sdi(
    lcp="data/lcp/baseline.tif",
    flame_length="data/outputs/baseline/FLAMELENGTH.tif",
    heat_area="data/outputs/baseline/HEATAREA.tif",
    roads_gdf=roads_gdf,
    trails_gdf=trails_gdf,
    rtc_path=rtc_path,
    out_path="data/sdi/baseline_sdi.tif",
)

sdi_treated = calculate_sdi(
    lcp="data/lcp/mech_thin.tif",
    flame_length="data/outputs/mech_thin/FLAMELENGTH.tif",
    heat_area="data/outputs/mech_thin/HEATAREA.tif",
    roads_gdf=roads_gdf,
    trails_gdf=trails_gdf,
    rtc_path=rtc_path,
    out_path="data/sdi/mech_thin_sdi.tif",
)

delta_sdi = calculate_delta_sdi(
    baseline_sdi=sdi_baseline,
    treatment_sdi=sdi_treated,
    out_path="data/sdi/delta_mech_thin.tif",
)
```

SDI outputs are stored as `int16` rasters scaled by 100 (divide by 100 for the raw index value). Roads and trails only need to be fetched once per project area and can be reused across all scenario SDI calculations.

#### Required input data

| Input | Source |
|-------|--------|
| LCP (SLP, ASP, FBFM40 bands) | LANDFIRE via `lfps_request`, or existing LCP |
| FLAMELENGTH, HEATAREA rasters | FlamMap outputs via `run_batch` |
| Roads and trails | OSM via `fetch_osm_roads` |
| RTC lookup table | `08_RTC_lookup_SDIwt_westernUS_2021_update.txt` (western US 2021) |

---

### `fb_tools.utils` — Shared Utilities

```python
from fb_tools import mask_raster, rasterize, geom_to_raster_crs, plot_bands

# Mask a raster to a geometry
arr = mask_raster("elev.tif", aoi_geometry, nodata_val=-9999)

# Rasterize a vector layer onto a reference grid
da = rasterize(zones_gdf, reference_da, attr="zone_id")

# Plot all bands of a multi-band raster
fig, axes = plot_bands("data/lcp/baseline.tif", cols=3)
```

---

## Notebooks

Jupyter notebooks in `code/Python/` demonstrate end-to-end workflows:

| Notebook | Purpose |
|----------|---------|
| `00-LCP_Baseline.ipynb` | LFPS download and baseline fuelscape preparation |
| `01a-LCP_Treated_Landscape.ipynb` | Landscape-scale fuel treatment application |
| `01b-LCP_Treated_InSitu.ipynb` | In-situ treatment variants |
| `02-Batch_FlamMap.ipynb` | Batch FlamMap scenario runs using `run_batch` |

---

## Data Sources

- **LANDFIRE:** [lfps.usgs.gov](https://lfps.usgs.gov) — fuel, canopy, topographic layers
- **OpenStreetMap:** via [osmnx](https://osmnx.readthedocs.io) — road and trail networks
- **RTC lookup table:** Rodriguez y Silva et al. (2014), western US adaptation (2021 update)

---

## Platform Notes

| Task | Platform |
|------|----------|
| LFPS download, LCP prep, SDI | macOS or Windows |
| FlamMap / FSPro / FSim / MTT runs | Windows only (Parallels Pro VM) |
| Notebook development | macOS (JupyterLab) |

Output paths for model runs should use Windows-style paths or UNC paths when calling `run_batch` from macOS via a shared drive.

---

## Author

Max C. Cook — [maxwell.cook@colostate.edu](mailto:maxwell.cook@colostate.edu)
