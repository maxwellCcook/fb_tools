# fb_tools

Python wrapper for CLI fire behavior models (FlamMap, FSPro, FSim, MTT) with tools for landscape fuel preparation, weather processing (HRRR winds · GridMET ERC climatology · RTMA hourly fuel moisture), burn probability analysis, and suppression difficulty.

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
├── weather/       HRRR winds · GridMET ERC climatology · RTMA hourly fuel moisture
├── spread/        Burn probability analysis, treatment effects, FSPro perimeter growth
├── suppression/   Suppression Difficulty Index (SDI) via Rodriguez y Silva
├── analysis/      Zonal statistics and treatment-level fire behavior summaries
└── utils/         Shared geo helpers: mask, clip, rasterize, plot
```

### Key modules

| Module | Key functions |
|--------|--------------|
| `fuelscape/lfps.py` | `lfps_request()` — LANDFIRE REST API download |
| `fuelscape/lcp.py` | `stack_rasters()`, `get_band_by_longname()` |
| `fuelscape/adjust.py` | `adjust_lcp()`, `apply_treatment()` |
| `models/flammap.py` | `run_flammap_scenarios()` |
| `models/scenarios.py` | `load_scenarios()`, `build_scenarios()`, `run_batch()` (supports `skip_existing`) |
| `models/fspro.py` | `build_fspro_inputs()`, `run_fspro_batch()` |
| `models/mtt.py` | `run_mtt()`, `run_mtt_batch()` |
| `weather/hrrr.py` | HRRR fire-hour wind extraction, pyrome wind climatology |
| `weather/gridmet.py` | GridMET ERC arrays, classes, and FlamMap scenario cache |
| `weather/rtma.py` | `build_rtma_dead_fm()`, `build_rtma_live_fm()`, `build_rtma_scenario_fm()` — hourly NFDRS78 FM from NWS RTMA |
| `spread/bp.py` | `delta_burn_probability()`, `summarize_bp_treatments()` |
| `spread/perimeters.py` | `load_fspro_perimeters()`, `summarize_early_growth()`, `compare_growth()` |
| `suppression/sdi.py` | `calculate_sdi()`, `calculate_delta_sdi()` |
| `analysis/zonal.py` | Zonal statistics helpers |
| `utils/geo.py` | `mask_raster()`, `clip_raster_inplace()`, `rasterize()` |

---

## Weather data sources

| Source | Cadence | Variables | Module |
|--------|---------|-----------|--------|
| **HRRR** | Hourly (via Herbie) | Wind speed & direction | `weather/hrrr.py` |
| **GridMET** | Daily (via GEE export) | ERC climatology, ERC classes | `weather/gridmet.py` |
| **NWS RTMA** | Hourly (via GEE export) | Dead FM (FM1/FM10/FM100), live FM (herb/woody) | `weather/rtma.py` |

ERC stays GridMET-derived — the FSPro/FSim daily-climatology contract is unchanged. RTMA upgrades the dead-FM and live-FM legs of FlamMap percentile scenarios using NFDRS78 time-lag recursion and GSI at 1-hour time steps.

---

## Author

Max C. Cook — [maxwell.cook@colostate.edu](mailto:maxwell.cook@colostate.edu)
