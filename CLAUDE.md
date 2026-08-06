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
# Optional (not in conda-forge):
pip install herbie-data
```

## Package layout (fb_tools/)
```
fuelscape/
  lfps.py      — LFPS REST API client
  lcp.py       — stack_rasters, ignition ASCII/shapefile helpers, get_band_by_longname
  adjust.py    — adjust_lcp, apply_treatment, build_surface_lut
models/
  base.py      — run_cli(), _write_shortterm_inputs()
  flammap.py   — run_flammap_scenarios()
  scenarios.py — load_scenarios, build_scenarios, run_batch, build_mtt_scenarios
  mtt.py       — run_mtt(), run_mtt_batch()
  fspro.py     — build_fspro_inputs(), build_treatment_pair(), run_fspro(), run_fspro_batch()
  fspro_validate.py — parse_fspro_input(), validate_fspro_input(), assert_valid_fspro_input()
  cell2fire.py — stub (NotImplementedError)
spread/
  bp.py        — delta_burn_probability, summarize_bp_treatments, downwind_treatment_effect
  perimeters.py — load_fspro_perimeters, summarize_early_growth, compare_growth
                  (BROKEN: _Perimeters.shp has malformed LinearRings — use fspro_outputs)
  fspro_outputs.py — read_daily_acres(), check_domain_adequacy()
  convert.py   — stub: lcp_to_cell2fire, build_cell2fire_weather
suppression/
  roads.py     — fetch_osm_roads() via osmnx
  sdi.py       — calculate_sdi(), calculate_delta_sdi()
utils/
  geo.py       — mask_raster, rasterize, clip_raster_inplace, lookup_pyrome, is_valid_geom
  io.py        — list_files()
  plot.py      — plot_bands()
analysis/
  zonal.py     — zonal statistics helpers
  treatments.py — deprecated stub → import from tealom.analyses
weather/
  hrrr.py      — HRRR fire-hour wind extraction; pyrome wind climatology
  gridmet.py   — GEE GridMET CSV → ERC arrays, classes, stats; FlamMap scenario cache
  nfdrs.py     — NFDRS fuel moisture (EMC, 1-hr, 10-hr)
  rtma.py      — NWS RTMA → NFDRS78 dead FM (FM1/FM10/FM100) + GSI live FM; build_rtma_scenario_fm
```

## Code conventions
- NumPy docstring format (Parameters / Returns / Raises)
- `pathlib.Path` throughout; convert `str → Path` at function boundary
- Copy-not-mutate: return new objects, never modify inputs in place
- `print()` for progress; no `logging` module
- Deferred imports for heavy/optional deps (`osmnx`, `herbie`) so `import fb_tools` never fails

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
Spatial container (HUC12/fireshed/POD) as analysis unit. Baseline vs. treated LCP runs share
one input file and the same `SPOTTING_SEED` → paired comparison via `delta_burn_probability()`.
Whether that pairing is *exact* (common random numbers) or merely statistical is the P0.1 question —
`code/dev/FSPro/p0_experiments.py`, run on the VM.

### FSPro outputs
- `_DailyAcres.txt` — `day,acres`, **daily increments**, one block of `Duration` rows per fire.
  This is the growth record; read with `read_daily_acres()`.
- `_Perimeters.shp` — **cannot be loaded** (malformed LinearRings) and its DBF has no time field.
- `_Ignitions.asc` — the ignition footprint already rasterized onto the output grid.
- Also unread: `_FireStreams.txt`, `_ArrivalDistribution.shp`, `_DayTypes.txt`, `_ContainSummary.txt`,
  `_EventCoverage.txt`, `_Suppression.asc`.
- `Σ BP × cell_area` over the domain ≈ mean simulated fire size (verified within 7% on the p47 run) —
  the identity behind the Ager `TF_ij` transmission estimator.

## Tests
`pytest` from the repo root (conda env `fb_tools`). Tests needing model output or cached weather
**skip** when absent, since `data/` is gitignored. `tests/conftest.py` holds the data paths.
183 tests as of Phase 1: `test_fspro_input.py`, `test_fspro_outputs.py`, `test_erc_classes.py`,
`test_current_erc.py`, `test_wind_cells.py`, `test_fspro_write_validation.py`, `test_fm_timeseries.py`.

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
