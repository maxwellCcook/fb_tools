# fb_tools — Claude Code Reference

Python wrapper for CLI fire behavior models (FlamMap, FSPro, FSim, MTT).
GitHub: https://github.com/maxwellCcook/fb_tools

## Platform split
- **Mac**: data prep, fuelscape work, weather data processing (this machine)
- **Windows VM (Parallels Pro)**: running model executables (TestFlamMap.exe, TestFSPro.exe, etc.)

## Install
```
conda env create -f environment.yml && conda activate fb_tools
pip install -e . --no-deps   # --no-deps is critical: conda manages GDAL/rasterio DLLs
```

## Package layout (fb_tools/)
```
fuelscape/
  lfps.py      — LFPS REST API client
  lcp.py       — stack_rasters, clip_lcp, ignition ASCII/shapefile helpers, get_band_by_longname
  ignitions.py — footprints, FPA-FOD density surface + sampling, clustering test, downwind cone,
                 design ignitions; load_fod, ignition_density_grid, convert_density_units (IDG by cause)
  adjust.py    — adjust_lcp, apply_treatment, fuel_break, build_surface_lut, mask_lcp
  synthetic.py — create_synthetic_lcp; create_combination_lcp, expand_combinations, combination_spread
models/
  base.py      — run_cli(), _write_shortterm_inputs()
  flammap.py   — run_flammap_scenarios()
  scenarios.py — load_scenarios, build_scenarios, run_batch, build_mtt_scenarios
  mtt.py       — run_mtt(), run_mtt_batch()
  fspro.py     — build_fspro_inputs(), build_treatment_pair(), run_fspro(), run_fspro_batch()
  fspro_validate.py — parse_fspro_input(), validate_fspro_input(), assert_valid_fspro_input()
  container.py — prepare_fspro_experiment() and the FSPro counterfactual set-up
  pyre.py      — pyretechnics runner: run_realization, burn_landscape, build_runs, run_batch
  cell2fire.py — stub (NotImplementedError)
spread/
  bp.py        — delta_burn_probability, aggregate_ignition_bp, summarize_bp_treatments,
                 downwind_treatment_effect
  noise.py     — null bands: bp_noise_floor, area_noise_floor, required_num_fires, annotate_noise_floor
  perimeters.py — BROKEN (_Perimeters.shp has malformed LinearRings) — use fspro_outputs
  fspro_outputs.py — read_daily_acres(), check_domain_adequacy()
  episodes.py  — pyre weather episodes, hourly frames, wind cubes
  pyre_outputs.py — per-run summaries, paired deltas, ensemble rasters, grid checks, pruning
  transmission.py — treatment shadow, structure blocks, lead time, near-field profile
  convert.py   — destack_lcp, raster_to_layers; Cell2Fire stubs
suppression/
  roads.py     — fetch_osm_roads() via osmnx
  sdi.py       — calculate_sdi(), calculate_delta_sdi()
  containment.py — containment rules for stepped pyre runs (PyreFirelines; SectorContainment experimental)
active_fire/   — FIRMS access, QA, events, footprints, grids, treatments, plots
utils/
  geo.py       — mask_raster, rasterize, clip_raster_inplace, lookup_pyrome, snapped_grid, is_valid_geom
  io.py        — list_files(), raster_write_kwargs()
  plot.py      — plot_bands()
analysis/
  zonal.py     — zonal statistics helpers
  treatments.py — deprecated stub → import from tealom.analyses
plotting/
  treatments.py — flame-length, crown-state and SDI figures
weather/
  hrrr.py      — HRRR fire-hour wind extraction; pyrome wind climatology; per-episode wind grids
  gridmet.py   — GEE GridMET CSV → ERC arrays, classes, stats; FlamMap scenario cache
  nfdrs.py     — NFDRS fuel moisture (EMC, 1-hr, 10-hr)
  rtma.py      — NWS RTMA → NFDRS78 dead FM (FM1/FM10/FM100) + GSI live FM; build_rtma_scenario_fm
  fm_scenario.py — build_fm_timeseries, extract_scenario_fm
```

## Code conventions
- NumPy docstring format (Parameters / Returns / Raises)
- `pathlib.Path` throughout; convert `str → Path` at function boundary
- Copy-not-mutate: return new objects, never modify inputs in place
- `print()` for progress; no `logging` module
- Deferred imports for heavy/optional deps (`osmnx`, `pyretechnics`) so `import fb_tools` never fails

## Status, history and planned work

- `docs/STATUS.md` — what each module does and its state. Update it when a module's state changes.
- `docs/DEVELOPMENT.md` — dated development notes, work in progress, planned work, and the detailed FSPro
  phase records. Add a dated entry there for any decision or finding; keep this file to what a session needs
  to work correctly.
- `docs/pyre_decisions.md` (pyretechnics) and `docs/ACTIVE_FIRE_PLAN.md` (`fb_tools.active_fire`) are the
  detailed logs for those two areas.

## pyretechnics transmission (branch `pyre-transmission`) — prototype stage
COSWAP transmission moves from FSPro to `pyretechnics` (pinned 2026.8.10, CFRI build `+cfri1`) for
mechanistic spotting, exact pairing (common random numbers), and native arrival time. **Decisions and
open items: `docs/pyre_decisions.md`** (dated; wins over the older plan in `docs/`). Do not re-litigate
what is logged there.

- **Env**: engine runs only in `fb_pyre` (`environment-pyre.yml`; py 3.11 / numpy 1.26). `fb_tools`
  is never downgraded. `import fb_tools` works in both; engine imports are deferred.
- **Modules**: `models/pyre.py` (version/gate, `build_cubes`, `ignition_matrix`, `seed_for`,
  `run_realization` — one call, or stepped with `step_h` and a containment rule via `suppression` (`suppression/containment.py::SectorContainment` = ELMFIRE model 0, reads `layers_dir/sdi.tif`, writes `held_line.tif`; every arm of a stepped experiment must be stepped), `build_runs`, `run_batch`), `spread/episodes.py` (`select_episodes`,
  `load_rtma_hourly`, `episode_hourly_frame`, `weather_arrays`, `episode_weather_report`; `select_episodes` takes a
  pool — climatology days at a severity (default), `dates=` from `dates_from_fod`, `dates_from_fast_fires` (FIRED
  events ≥ 1,620 ha/day; the lookup `fb_tools/data/fast_fires_fired_v2025var.csv` ships with the package, rebuild it with
  `build_fast_fires_table` on a new FIRED release) or a named event — and a
  `rank` — seeded random (default) or top-n by `erc_day1`/`erc_mean`),
  `spread/pyre_outputs.py` (`summarize_realization`, `paired_deltas`, `suppression_summary`/`_batch`, `aggregate_realizations`,
  `firebrand_production`, `edge_contact`, `prune_realizations` — drops per-run rasters once summaries and ensemble rasters are written), `spread/transmission.py` (`structure_blocks`, `treatment_shadow`,
  `realization_transmission`, `summarize_blocks`, `shadow_probability` — the location product: share of the
  fire reaching each cell that passed through a treatment, MFD-weighted, one triangular solve; and the
  near field: `changed_cells` (the treatment as the engine sees it), `distance_bands`, `near_field_zones`
  (mask zones — `all_touched` polygons a few cells wide leak into their own collar), `ember_descent` (burn the shadow cannot trace), `treatment_entry_time` (the `lead_h` clock), `shadow_agreement` (shadow vs the paired change; the shadow is a baseline-arm product), `lee_profile` /
  `lee_profile_batch` / `summarize_lee_profile` / `near_field_headlines` — paired deltas by distance from
  the treatment, split by the baseline run's shadow; recovery distance, carried delay, hauling-chart classes),
  `spread/convert.py` (`destack_lcp`, `raster_to_layers` — a statewide raster windowed onto a layers grid), `fuelscape/lcp.py::clip_lcp` (SFA + buffer window, 100-cell
  multiples), `fuelscape/adjust.py::fuel_break` (treated corridor along lines via `apply_treatment` + tealom
  effects tables → LCP for `destack_lcp`; notebook §5b is the fuel-break counterfactual: one named fire, three arms, 20 spotting seeds at 14 days, breakout rate and `shadow_agreement`), `fuelscape/ignitions.py::sample_density_ignitions` (FSim-style sites ∝ FOD density, uniform
  `w_i`). A fire that never leaves its footprint is `stop_condition == "no_spread"`, not an error. **Example**: `code/notebooks/01f_PyreBuild.ipynb` on the
  `front_range` test AOI (LFPS download in `data/spatial/mod/fuelscapes/front_range/`, 3000×1000 @ 30 m,
  pyrome 46): destack → potential map → 3 episodes + HRRR cubes → 30 IDG sites (FOD class ≥ C, 10 km
  kernel; cells drawn ∝ density, never the FOD points) → one burn → 90-run
  ensemble → `BurnProb`, `ArrivalProb_d{d}`, `AvgTime`, `Firebrands`. The COSWAP arms/design-fire/paired-Δ
  work lives in the project folder (`COSWAP/…/fire_behavior/code/04_PyreTransmission.ipynb`, a private repo with `scripts/run.sh` for CyVerse), not here.
  Outputs that skip on an existing manifest are named by their inputs so a changed parameter cannot reuse
  a stale product.
- **Units the engine wants** (each unit-tested): wind km/h, temperature °C, direction met FROM,
  moistures as fractions, slope rise/run (LCP `SLPD` degrees → `tan`; never `load_landfire`),
  `time_of_arrival` minutes (`AvgTime.tif` written in days). `ignite_cells` window is −1 inside.
- **Pairing**: seed = `seed_for(base, ignition, episode, condition)`, never the arm; ignition
  rasterized once per footprint; `assert_crn_contract` before any run. Sham arm gives Δ = 0 exactly.
- **Physics settings, stamped in every manifest**: crown gate (production 0.0; stock 0.4 is a
  sensitivity subset), `max_cells_per_timestep` 0.4, `buffer_width` 3, spot config.
- **Production resolution 30 m** (90 m destroys fuel breaks, 3.8× area). ~250 s per 21-day
  realization with spotting; 0.55 GB per worker.
- **Spotting is a heavy tail**: report breakout rate + conditional magnitudes, never a mean alone.
- **Weather**: T/RH → NFDRS dead FM from the pyrome-mean RTMA monthly files
  (`data/weather/pyrome_emc/RTMA_MONTHLY`; not the older `RTMA_EMC`, different fire mask).
  **Wind is a 3-D HRRR cube per episode** (`weather/hrrr.py::fetch_hrrr_wind_grid`, byte-range from
  AWS, no herbie; `episodes.episode_wind_cube` → `.npz`; `build_runs(wind_cubes=...)`). **All HRRR fetches
  share one cache**, `default_hrrr_cache()` = `$FB_HRRR_CACHE` or `~/fb_scratch/hrrr_cache`. Any spatial
  mean of wind (pyrome or domain) erases the design wind — see the log. Episodes 2016+.
  `RTMA_WIND_MONTHLY` is the pyrome-mean fallback and the frame's 1-D wind.
- **Domain**: the 52 km North Cottonwood LCP saturates by day 3–5 under any weather; a wider LCP
  (~100 km, rows/cols multiples of 100) is pending. Duration is set by `check_domain_adequacy`.
- **Scratch**: `~/fb_scratch/{coswap_pyre,pyre_example}` (`$FB_PYRE_SCRATCH`); only ensemble products go to Box.
- **Shared helpers, defined once**: `hrrr.uv_to_speed_dir` (met FROM; m/s, km/h, mph), `hrrr.longest_true_run`,
  `pyre_outputs.REALIZATION_LAYERS` / `rasterize_mask`. Do not re-derive these inline.
- **Diagnostics** (`code/dev/pyre/`): `g0`–`g1c` crown/units checks, `b1`–`b3` benchmark and
  spotting variance, `build_patched_engine.py`, `pyre_source.py` (pure-Python engine for cfunc access).
- **Tests**: `test_pyre_destack.py`, `test_pyre_episodes.py`, `test_pyre_transmission.py`, `test_raster_to_layers.py` (engine-free, run in
  `fb_tools`); `test_pyre_engine.py`, `test_pyre_runner.py`, `test_pyre_stepped.py` (need `fb_pyre`).

## Model CLI invocation
- **FlamMap**: command file approach via `run_cli()`
- **MTT command file**: `{lcp} {input_file} {ignition_shp} {barrier_or_0} {output_base_path} {output_type}`
  - Output flag: `FLAMELENGTH:` (no value) vs FlamMap's `FLAMELENGTH: 1`
- **FSPro CLI**: `TestFSPro {lcp} {input_file} {output_base}` (3 direct args, no command file)
- Platform guards in `run_mtt()` and `run_fspro()` raise `RuntimeError` on non-Windows

## FSPro input format (FSPRO-Inputs-File-Version-4)
Key sections and expected shapes:
- `NumERCYears` / `NumWxPerYear`: typically 15 × 214 (April 1 – Oct 31)
- `HistoricERCValues`: one row per year, 214 space-separated floats
- `WindCellValues`: NumWindSpeeds rows × NumWindDirs cols (% frequency table)
- ERC classes: 5 rows, **descending ERC**, each
  `MinERC MaxERC FM1 FM10 FM100 FMHerb FMWoody Duration SpotProbability SpotDelay` (spec p.4)
  - **Column 8 is the daily burn period in minutes** (spec name `Duration`, distinct from the
    run-level `Duration:` switch; echoed as `burnPeriod` in `_DayTypes.txt`). The `360/300/240/180/120`
    ladder is 6h→2h and is a first-order control on daily growth. Historically mislabelled `spot_dist`.
  - Fuel moisture must **fall as ERC rises** — every FM column non-decreasing down the rows
- `CurrentERCValues`: the season-to-date stream, **always starting at fire-season day 1 (April 1)**
  and ending the day before ignition — FSPro reads it positionally, so an interior slice shifts the
  whole stream (defect #3, fixed in P1.2). Build with
  `build_current_erc_values(ignition_season_day=N, mode=...)`; length is `N − 1`.
  Modes: `analog_year` (default — a real year's observed sequence, keeps variance/autocorrelation
  and is nameable in a writeup), `percentile`, `median` (the old mild reference), `observed`.
  The pre-P1.2 `start_doy=` keyword now raises.
- Spec constraints: `MaxLag ≤ NumWxCurrYear < NumWxPerYear − Duration`; `NumForecast ∈ [0, Duration−1]`
  with that many rows following; `PolyDegree ∈ [4,15]`; `CROWN_FIRE_METHOD ∈ {"Finney","ScottRheinhardt"}`
  (**not** "Scott/Reinhardt"); forecast row order is `ERC WindSpeed WindDirection`
- `WindCellValues` sums to ~100 **on its own**; `CalmValue` is stored separately, not subtracted
- `IgnitionFile`: polygon/polyline shapefile (NOT points) — it is a **starting fire perimeter**, not a
  sampling domain. BP ≈ 1.0 everywhere inside it by construction, so it must be masked out of every
  Δ statistic, and a whole container must never be used as the ignition.

`build_fspro_inputs()` validates what it wrote and raises on any violation (P1.6). A rejected file is
moved aside to `*.input.invalid` so it cannot be run by accident — FSPro does not validate, it just
misreads. Pass `validate=False` to opt out. `NumForecast` is always synced to the number of forecast
rows actually written (including `None → 0`); leaving it set without rows made FSPro consume
`BarrierFill` / `SavePerimeters` / `IgnitionFile` as forecast records (defect #8).

### FSPro counterfactual workflow
Spatial container (HUC12/fireshed/POD) as analysis unit; arms share one input file. Full records of each
phase (measurements, defects, reasoning) are in `docs/DEVELOPMENT.md` under "Detailed records".

- **`SPOTTING_SEED` does not give common random numbers** (P0.1, 2026-08-05): pairing is statistical, never
  exact, and **every reported Δ needs a null band** (`spread/noise.py`, calibrated on P0.1 and scaled 1/√N;
  `area_noise_floor(for_difference=True)`; `annotate_noise_floor(metric="area")` needs a `total_col`).
  Prefer `Σ BP × cell_area` over pixel ΔBP. ~1.22 s/fire at Duration 7 on 740×783 @ 90 m.
- **`prepare_fspro_experiment()` is the entry point** (`prepare_container_fspro` /
  `prepare_counterfactual_fspro` are frozen). N arms via `lcps={...}` + `contrasts`; `runs.csv` carries
  `Arm` and `w_i`. Grid congruence across arms is asserted before any run. `domain_gdf` never clips.
  `ignition_mode` has no default. One shapefile per ignition (`create_*_ignitions` return `list[Path]`,
  take `out_dir`). Default footprint 10 acres.
- **Design fires**: downwind cone (half-angle = narrowest arc holding 50% of fire-hour wind; ±38° for
  pyrome 47), stratified bearing × distance, Horvitz–Thompson weights. Colorado ignitions are clustered
  (P2.2), so sampling is density-weighted. `ignition_density_surface` smooths on a coarsened grid. The public
  clustering function is `check_ignition_clustering`, never `test_*`.
- **Phase 3a API**: `summarize_bp_treatments(zones_gdf, delta_bp=…)` or `(…, baseline_bp=…, treatment_bp=…)`;
  `delta_burn_probability` writes `DELTA_NODATA = -32768` and checks the grid (`strict_grid=True`);
  `aggregate_ignition_bp` uses the design weights — mask BP surfaces with `n_burned`, not `n_ignitions`;
  the ignition footprint is masked from every Δ statistic (`ignition=` / `ignitions=`);
  `run_fspro(check=True)` fails unless the run exits 0 and writes output.
- P0.2 (runtime grid) and P0.3 (footprint size) are unrun — `code/dev/FSPro/p0_experiments.py`.

### FPA-FOD on disk — mind the vintage
- **Authoritative, through 2024**: `/Users/mcc/Library/CloudStorage/Box-Box/MCC/data/
  wildfire/FPA_FOD/RDS-2013-0009/Data/FPA_FOD_20260615.gpkg` — layer `Fires`, 2.66 M
  records, EPSG:4269. Read with a `where=` clause (Class D–G over 7 states ≈ 1 s).
- Everything under `data/spatial/raw/fpa_fod/` is **older**: the `SHP/Fires_ClassDEFG_*`
  layers stop at **2022**, and `DRAFT_7th/FPA_FOD_DRAFT_7thED_points.shp` is a truncated
  export holding only **1992–2009**. The `.accdb` beside it is the full draft 7th ed.
- `_NB_CODES = {0, 91, 92, 93, 98, 99}` in `lcp.py` is **correct** — Scott & Burgan FBFM40
  defines NB1/2/3/8/9 only; 94–97 are unassigned. (Plan flagged this as needing a check.)

### FSPro outputs
- `_DailyAcres.txt` — `day,acres`, **daily increments**, one block of `Duration` rows per fire.
  This is the growth record; read with `read_daily_acres()`.
- `_Perimeters.shp` — **cannot be loaded** (malformed LinearRings) and its DBF has no time field.
- `_Ignitions.asc` — the ignition footprint already rasterized onto the output grid.
- Also unread: `_FireStreams.txt`, `_ArrivalDistribution.shp`, `_DayTypes.txt`, `_ContainSummary.txt`,
  `_EventCoverage.txt`, `_Suppression.asc`.
- `Σ BP × cell_area` over the domain ≈ mean simulated fire size (verified within 7% on the p47 run) —
  the identity behind the Ager `TF_ij` transmission estimator.

## Ignition density grids by cause (`fuelscape/ignitions.py`, 2026-11-12)
`load_fod(fp, aoi=, buffer_m=, years=, size_classes=)` pushes bbox + where into the read (sub-second for a county
set) and adds `cause` ∈ human/natural/unknown. `ignition_density_grid(fod, aoi, res_m, bandwidth_m)` → xarray
Dataset in **fires/yr/100 km²**: one layer per cause, `all` (incl. unknown), `human_share` (NaN where the kernel
holds < `share_min_fires`, effective count `rate × years × 4πσ²`), `difference`. Any projected `crs`; in
EPSG:5070 edges snap to LANDFIRE pixel edges (15 m mod 30), elsewhere to multiples of `res_m`; or `template=` a raster. `adaptive=True` is Abramson's square-root
law (σ_i = σ₀·√(g/f̃), clipped to [σ₀/`trim`, σ₀·`trim`]); σ is set from **all fires pooled** and shared by every
cause (Davies et al. 2016 symmetric estimator) so `human_share` compares like with like — the lightning layer stays
spotty on the plains for that reason. Per-fire σ is binned into `n_bandwidth_bins` log classes, each smoothed once
(~4 s at 90 m for the Larimer AOI); adds `bandwidth_local` (kernel-weighted σ, m), which also feeds the share mask.
Not burnable-masked — `ignition_density_surface` is the LCP sampling frame. Both share `_smooth_points_onto_grid`; the refactor fixed a resampling offset that put
the surface ~σ/8 up-left of the points. Notebook `code/notebooks/ignitions/IgnitionDensity.ipynb` (was 00e) (Larimer + 6
touching counties, UTM 13N, 90 m, adaptive σ₀ 2.5 km (2 km left 18% of the share map blank, 2.5 km 4%), all sizes, ≥1,000 ac fires as points by cause, light reference
layers — OSM motorway/trunk/primary + place points cached in `data/spatial/raw/reference/`, TIGER 2020 urban areas,
county-seat labels; pyrome/west/CONUS cells off by default). Overpass: set `ox.settings.max_query_area_size` large
for a filtered query — osmnx's default subdivision (~20 sub-queries here) gets rate-limited for >10 min. A fixed
σ ≤ 2 km turns lightning into a dot field and blanks most of the share map; that is why the adaptive kernel exists.
FOD classes a fire Human when lightning is ruled out — 38% of CO+WY human-classed fires have an undetermined
`NWCG_GENERAL_CAUSE` (48% of those ≥1,000 ac, e.g. Cameron Peak).
`convert_density_units(ds, per_year=, area=)` re-expresses the layers (km2/100km2/mi2/100mi2, per year or over the
record); 00e maps and GeoTIFFs use **ignitions per km² over `YEARS` = 2013–2024** on a shared classed log scale (1-3-10
edges, so the ~16× people-vs-lightning gap shows as classes) with plain-language labels for a general audience, an ignitions-per-year-by-county chart and a federal-agency-only 1992–2024 series (§5), PNGs in
`OUT/<aoi>/figures/`. Records named as prescribed/pile burns (`RX_PATTERN`, 7 here) are dropped; FPA-FOD already
screens prescribed fire out. **FOD reporting changes over time here** (always tabulate spatially — `FIPS_NAME` is
blank on many records): 1992–2009 is almost all federal-agency reports (federal land, mountains); state & local
(`NWCG_REPORTING_AGENCY == "ST/C&L"`: ST-NASF to 2020, InFORM from 2021) begins 2010, but 2011 is ~1,400 no-cause
records and 2012 only ~30 — complete from **2013**. Federal agencies moved from FS-FIRESTAT/DOI-WFMI to IRWIN (2020)
and InFORM (2021+), so split federal vs state by `NWCG_REPORTING_AGENCY`, never `SOURCE_SYSTEM_TYPE`. **Cause detail**: `NWCG_GENERAL_CAUSE` is the finest cause field; in 2021–2024 InFORM codes state & local fires with no specific cause as "Other causes" (~3,650 here) where earlier years used "undetermined" — read the two together (~70% of ignitions have no specific cause). Tests: `tests/test_ignition_density.py`.
**AGOL** (`code/notebooks/ignitions/IgnitionAGOL.ipynb`, run after IgnitionDensity, on the Mac; publish with `IgnitionAGOL_Publish.ipynb` in ArcGIS Pro on the VM — Pro's Python only, no `fb_tools`, the pods-atlas build/publish split): one GeoPackage → one hosted feature layer —
`ignitions` (points, 2013–2024; public FOD services stop at 2020), `density_human`/`density_lightning`/`share_human`
(the GeoTIFFs as classed polygons at `CLASS_EDGES`, coverage-simplified; hosted rasters need the Image extension),
`counties`. Perimeters (NIFC history, WFIGS) and WFIGS current incidents are streamed in the web map, not published.
Updates **overwrite** the item (keeps id/URL; same layers and fields); new datasets go in new items.

## Combination LCP (`fuelscape/synthetic.py`, 2026-09-28)
FlamMap once per unique fuel combination instead of once per pixel. Valid when a cell's output
depends only on its own bands: uniform weather, upslope wind (`WIND_DIRECTION: -1`), no gridded
winds, no conditioning. `combination_spread(lcp, output)` is the check (all zeros). A fixed wind
direction adds `Asp` to `keys`; conditioning adds `Elev` and `Asp`; gridded winds rule it out.
`create_combination_lcp(baseline, out, treatments)` writes block 0 = each unique baseline
combination once (a real pixel's band row), then one block per treatment (``fn(DataArray)``, e.g.
a `partial` of `apply_treatment`) applied to block 0 — cell *j* of every block traces to baseline
combination *j*, so no treated rasters or lookup table are needed. Layout and keys live in GeoTIFF
tags. `expand_combinations(combo_lcp, output, baseline, treatment)` rebuilds a full-grid stack
(tagged `EXPANDED_TAG` when written). First used on treatment_effects: 22 arms + baseline in a
4,393 x 192 LCP, 3 runs instead of 69. Tests: `tests/test_combination_lcp.py` (engine-free).

## Tests
`pytest` from the repo root (conda env `fb_tools`). Tests needing model output or cached weather
**skip** when absent, since `data/` is gitignored. `tests/conftest.py` holds the data paths.
575 tests as of 2026-11-24 (335 at Phase 3a). FSPro and weather files: `test_fspro_input.py`, `test_fspro_outputs.py`, `test_erc_classes.py`,
`test_current_erc.py`, `test_wind_cells.py`, `test_fspro_write_validation.py`,
`test_fm_timeseries.py`, `test_ignitions.py`, `test_experiment_api.py`, plus Phase 3a's
`test_bp_delta.py`, `test_noise_floor.py`, `test_fspro_runcheck.py`.
The Phase 2/3a files depend on nothing in `data/` — `conftest.py` builds a synthetic
heterogeneous LCP (`synth_lcp`) and a clustered point set (`synth_fod`) in-process, and
`test_bp_delta.py` writes its own small BP rasters.
`_assert_fspro_succeeded` is tested directly since `TestFSPro.exe` is Windows-only.

## LANDFIRE layers
Topo: `ELEV2020`, `SLPD2020`, `ASP2020`
Version "200" fuels: `F40_20`, `CC_20`, `CH_20`, `CBH_20`, `CBD_20`, `EVT`
Band normalization (`adjust.py`, `plot.py`): strip region prefix + LF version/year suffix
→ canonical names: `ELEV`, `SLP`, `ASP`, `FBFM40`, `CC`, `CH`, `CBH`, `CBD`, `EVT`

## HRRR wind climatology — non-obvious constraints
- Reliable archive starts **2016** (NOT 2014 — gaps pre-2016)
- HRRR longitude grid is **0–360**; must normalize to −180/180 for WGS84 fire points
- Fire hours UTC: `[19, 20, 21, 22]` = 13:00–16:00 MDT / 12:00–15:00 MST
- Wind direction (met FROM): `wd_deg = (degrees(arctan2(-u, -v)) + 360) % 360`
- KD-tree built once per HRRR file; vectorized query across all fire points
- `build_pyrome_wind_cells(season_months=(4, 10))` filters to the fire season (P1.5) — before this
  the only temporal filter was `year >= 2016`
- NaN winds are **missing, not calm**: `ws >= threshold` is False for NaN, so archive gaps used to
  inflate `CalmValue`. Missing obs are now dropped from both the table and the calm denominator.
- `k_neighbors=9` samples a 3×3 (~9 km) neighbourhood — these are pseudo-replicates of one wind
  observation, so `min_obs_warn` is applied to `n_noncalm / k_neighbors` (200 raw ≈ 22 independent)
- `WindSpeedBreaks_mph` / `WindDirBreaks_deg` from the cache are forwarded into the input file;
  they used to be dropped, so a cache with custom breaks contradicted its own matrix

## GEE assets (project: cfri-ee)
- Pyromes: `projects/cfri-ee/assets/weather/Pyromes_CONUS_20200206`
- FOD: `projects/cfri-ee/assets/weather/Fires_ClassDEFG_CO_Pyromes`
- CO analysis extent: pyromes `42, 43, 45, 46, 47, 52, 53, 56, 128`

## GridMET ERC climatology
- Source: GEE-exported CSV, pyrome mean per day, April 1–Oct 31, 2008–2022
- CSV columns: `pyrome, date, year, doy, erc, fm100, fm1000, tmmx, tmmn` (K→°F on load), `rmin, vpd` (kPa→Pa)
- Day-of-season pivot anchored to April 1 (1–214), leap-year-safe
- ERC class row: `[min_erc, max_erc, fm1, fm10, fm100, fm_herb, fm_woody, burn_period_min, spot_prob, spot_delay]`
- **`build_erc_classes` defaults (Phase 1)** — `dead_fm_source="rtma"`, `live_fm_source="dead_fm"`,
  `class_percentiles=[0, 60, 80, 90, 97, 100]`. This combination is monotonic in all 9 cached CO
  pyromes; the validator rejects anything that is not.
  - **dead FM** from the RTMA daily peak-hour cache (`flammap_rtma/rtma_daily_fm.parquet`), median
    over each ERC band's dates. `"gridmet"` falls back to the NFDRS78 lag; the two agree within ~1%
    at the extreme class and diverge only in mild classes.
  - **live FM** scaled from that FM100 via `calc_live_fm_from_dead(herb_scale=6.5, woody_scale=9.0)`.
    Extreme class lands at 32–69% herb / 60–95% woody — Ager et al. 2014 Table 7 is 40–60 / 60–90.
  - `live_fm_source="gsi"` (RTMA GSI columns) is **measured non-monotonic in 4 of 9 pyromes** and
    pins 6 of 9 extreme classes at the 30/60 dormant floor. `"doy"` reproduces defect #2. Both are
    selectable for comparison only.
  - Tail-weighted bands matter: quintiles gave pyrome 47 a top class of ERC 67–94; the default now
    resolves 84–94.
- FlamMap scenarios (`build_flammap_fuel_moistures`) still use the DOY/GSI live-FM path — the
  `lat_deg` GSI branch lives there, not in `build_erc_classes`.
  - NOTE: GSI with pyrome-mean GridMET VPD does not work for semi-arid western US fire season —
    Jolly et al. (2005) thresholds calibrated for temperate/boreal, not CO conditions
- `build_fm_timeseries(max_gap_days=45)` restarts the NFDRS78 lag at each season break. Integrating
  through the Oct 31 → Apr 1 gap carried the October end-state into spring, leaving April FM100
  **2.5–7.2 pp too moist** across the 9 CO pyromes, converging by season day 18–22.
