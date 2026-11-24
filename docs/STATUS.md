# fb_tools — status

What each part of the package does and where it stands. Updated 2026-11-24.
History and decisions: [DEVELOPMENT.md](DEVELOPMENT.md). Install and layout: [README](../README.md).

**States** — *working*: used on real data and tested. *prototype*: runs end to end, interface still moving.
*partial*: some of the planned pieces exist. *stub*: placeholder that raises. *broken*: known not to work.

## Fire models

| Module | Does | State | Tests | Known gaps |
|---|---|---|---|---|
| `models/flammap.py`, `scenarios.py` | FlamMap batch runs from scenario tables (Windows VM) | working | `test_wind_sweep.py` | — |
| `models/mtt.py` | Minimum travel time runs (Windows VM) | partial | — | Not run end to end on the VM as of 2026-08; not rechecked since |
| `models/fspro.py`, `fspro_validate.py`, `container.py` | FSPro inputs, validation, N-arm counterfactual experiments, batch runs (Windows VM) | working | `test_fspro_*.py`, `test_experiment_api.py`, `test_current_erc.py`, `test_erc_classes.py` | Runs are not exactly paired (seed reaches spotting only), so every difference needs a null band. Analysis layer beyond pre-flight (transmission matrix, exposure) not built. Runtime grid and footprint-size experiments not run |
| `models/pyre.py` | pyretechnics spread runs: one realization, stepped runs with a containment rule, paired batches (`fb_pyre` env, Mac or Linux) | prototype | `test_pyre_engine.py`, `test_pyre_runner.py`, `test_pyre_stepped.py` | Pinned to 2026.8.10 (CFRI build); upgrade to 2026.11.20 untested. Audit in progress — see DEVELOPMENT.md |
| `models/cell2fire.py` | — | stub | — | Not planned at present |
| FSim | — | not in package | — | `01d_FSimBuild.ipynb` exists; no wrapper module |
| ELMFIRE | — | not in package | — | Spike planned before any wrapper is written |

## Landscape and ignitions

| Module | Does | State | Tests | Known gaps |
|---|---|---|---|---|
| `fuelscape/lfps.py` | LANDFIRE download (LFPS REST) | working | — | Product codes for LF 220 / 230 noted as unconfirmed in 2026-08 |
| `fuelscape/lcp.py` | Stack rasters, clip to a domain, ignition files | working | `test_ignitions.py` | — |
| `fuelscape/adjust.py` | Fuel treatments on an LCP (`apply_treatment`), corridors along lines (`fuel_break`), masking | working | `test_combination_lcp.py`, `test_fuel_break.py`, `test_mask_lcp.py` | Effects tables come from the tealom package, not from here |
| `fuelscape/synthetic.py` | Synthetic LCPs; combination LCP (FlamMap once per unique fuel combination) | working | `test_combination_lcp.py` | Valid only for uniform weather and upslope wind |
| `fuelscape/ignitions.py` | Footprints, FPA-FOD density surface and sampling, design fires, clustering test, density grids by cause | working | `test_ignitions.py`, `test_ignition_density.py` | Clustering test is not masked to burnable fuel |

## Fire spread products

| Module | Does | State | Tests | Known gaps |
|---|---|---|---|---|
| `spread/bp.py`, `noise.py` | Burn-probability differences, ignition-weighted aggregation, zonal summaries, null bands (FSPro) | working | `test_bp_delta.py`, `test_noise_floor.py` | — |
| `spread/fspro_outputs.py` | Daily growth record, domain adequacy | working | `test_fspro_outputs.py` | — |
| `spread/perimeters.py` | FSPro perimeter growth | broken | — | FSPro perimeter shapefiles cannot be loaded; use `fspro_outputs` |
| `spread/episodes.py` | Weather episodes for pyretechnics: selection, hourly frames, HRRR wind cubes | prototype | `test_pyre_episodes.py`, `test_hrrr_wind.py` | April–October only; winter and late-October events cannot be run |
| `spread/pyre_outputs.py` | Per-run summaries, paired differences, ensemble rasters, pruning | prototype | `test_pyre_episodes.py` | No ensemble difference raster or null band yet |
| `spread/transmission.py` | Treatment shadow, structure blocks, lead time, near-field profile | prototype | `test_pyre_transmission.py` | Shadow cannot follow embers; validated on one fire only |
| `spread/convert.py` | LCP to pyretechnics layers (`destack_lcp`, `raster_to_layers`) | working | `test_pyre_destack.py`, `test_raster_to_layers.py` | Cell2Fire converters are stubs |

## Weather

| Module | Does | State | Tests | Known gaps |
|---|---|---|---|---|
| `weather/gridmet.py` | ERC climatology, ERC classes, FSPro and FlamMap weather inputs | working | `test_erc_classes.py`, `test_current_erc.py` | Source export is April–October, 2008–2022 |
| `weather/rtma.py`, `nfdrs.py`, `fm_scenario.py` | Hourly RTMA to dead and live fuel moisture; scenario fuel moistures | working | `test_rtma_load.py`, `test_fm_timeseries.py` | Months 04–10 only; live fuel moisture is scaled from dead, not measured |
| `weather/hrrr.py` | HRRR fire-hour wind tables; per-episode 3-D wind cubes | working | `test_wind_cells.py`, `test_hrrr_wind.py` | Archive reliable from 2016. Wind components are not rotated from the model grid (open question) |

## Suppression, active fire, analysis

| Module | Does | State | Tests | Known gaps |
|---|---|---|---|---|
| `suppression/sdi.py`, `roads.py` | Suppression Difficulty Index; OSM roads | working | — | No unit tests |
| `suppression/containment.py` | Containment rules for stepped pyretechnics runs | prototype | `test_pyre_stepped.py` | `pyre_firelines` (engine's own functions) is the supported rule; `elmfire0` is experimental and not to be extended |
| `active_fire/` | FIRMS access, QA, event association, footprints, gridded FRP, first-detection grid, treatment summaries, plots | partial | `test_active_fire.py` | Swath access not built; remaining steps in `ACTIVE_FIRE_PLAN.md` |
| `analysis/zonal.py` | Zonal statistics | working | — | Treatment summaries moved to the tealom package |
| `plotting/treatments.py` | Flame-length, crown-state and SDI figures | working | — | — |
| `utils/` | Raster masking, rasterizing, grid snapping, pyrome lookup, I/O | working | `test_mask_lcp.py` | — |

## Tests and environments

- 575 tests in 27 files as of 2026-11-24. Engine-free tests run in `fb_tools`; `test_pyre_engine.py`,
  `test_pyre_runner.py` and `test_pyre_stepped.py` need `fb_pyre`. Tests that need model output or cached
  weather skip when it is absent.
- `environment.yml` → `fb_tools` (everything except the pyretechnics engine).
  `environment-pyre.yml` → `fb_pyre` (pyretechnics, Python 3.11).
- FlamMap, MTT and FSPro executables run on the Windows VM; data preparation and pyretechnics run on the Mac.

## Notebooks (`code/notebooks/`)

| Notebook | Shows |
|---|---|
| `00a`–`00d` | Weather inputs: GridMET ERC, RTMA moisture, HRRR wind tables |
| `01a`–`01c` | Building FlamMap, MTT and FSPro runs |
| `01f_PyreBuild.ipynb` | pyretechnics from landscape to ensemble, with a fuel-break counterfactual over 20 seeds |
| `ignitions/` | Ignition density by cause and the ArcGIS Online build |
| `active_fire/01_FIRMS.ipynb` | Active-fire detections |
