# fb_tools

Python wrapper for CLI fire behavior models (FlamMap, FSPro, FSim, MTT) with tools for landscape fuel preparation, weather processing, burn probability analysis, and suppression difficulty.

---

## Installation

The geospatial stack must come from conda — do not let pip manage GDAL/rasterio.

```bash
# Clone and set up the environment
git clone https://github.com/maxwellCcook/fb_tools.git
cd fb_tools

conda env create -f environment.yml
conda activate fb_tools

# Install in editable mode — --no-deps is critical
pip install -e . --no-deps
```

`--no-deps` prevents pip from overwriting conda's GDAL/PROJ/rasterio builds. Omitting it will break raster I/O.

**Optional** (not in conda-forge):
```bash
pip install herbie-data   # required for HRRR wind extraction
```

**Platform split:** Data prep and weather processing run on macOS. Fire behavior model executables (FlamMap, FSPro, FSim, MTT) are Windows-only — run those on a Windows VM or machine.

> **Windows:** Always create and activate the environment from Anaconda Prompt, not PowerShell.

---

## Package Structure

```
fb_tools/
├── fuelscape/     LANDFIRE download, LCP raster stacking, fuel treatment adjustments
├── models/        FlamMap, FSPro, MTT scenario config and batch execution
├── weather/       HRRR wind extraction, GridMET ERC climatology for FSPro inputs
├── spread/        Burn probability analysis and treatment effect summaries
├── suppression/   Suppression Difficulty Index (SDI) via Rodriguez y Silva
└── utils/         Shared geo helpers: mask, clip, rasterize, plot
```

### Key modules

| Module | Key functions |
|--------|--------------|
| `fuelscape/lfps.py` | `lfps_request()` — LANDFIRE REST API download |
| `fuelscape/lcp.py` | `stack_rasters()`, `get_band_by_longname()` |
| `fuelscape/adjust.py` | `adjust_lcp()`, `apply_treatment()` |
| `models/flammap.py` | `run_flammap_scenarios()` |
| `models/scenarios.py` | `load_scenarios()`, `build_scenarios()`, `run_batch()` |
| `models/fspro.py` | `build_fspro_inputs()`, `run_fspro_batch()` |
| `models/mtt.py` | `run_mtt()`, `run_mtt_batch()` |
| `weather/hrrr.py` | HRRR fire-hour wind extraction, pyrome wind climatology |
| `weather/gridmet.py` | GridMET ERC arrays, classes, stats for FSPro |
| `spread/bp.py` | `delta_burn_probability()`, `summarize_bp_treatments()` |
| `suppression/sdi.py` | `calculate_sdi()`, `calculate_delta_sdi()` |
| `utils/geo.py` | `mask_raster()`, `clip_raster_inplace()`, `rasterize()` |

---

## Author

Max C. Cook — [maxwell.cook@colostate.edu](mailto:maxwell.cook@colostate.edu)
