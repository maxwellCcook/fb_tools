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
  ignitions.py — footprint sizing, FPA-FOD density surface, clustering test,
                 downwind cone, select_design_ignitions  (Phase 2)
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
  bp.py        — delta_burn_probability, aggregate_ignition_bp, summarize_bp_treatments,
                 downwind_treatment_effect  (all reworked in Phase 3a)
  noise.py     — P0.1 Monte Carlo null bands: bp_noise_floor, area_noise_floor,
                 required_num_fires, annotate_noise_floor, describe_noise_floor
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

**P0.1 answered (2026-08-05, VM, 416 sample @ 100 fires / Duration 7): `SPOTTING_SEED` does NOT
give common random numbers.** Two runs at the *same* seed differ on 66% of burned cells
(|ΔBP| p95 = 0.07, max 0.18). Same-seed and different-seed noise are statistically
indistinguishable (p95 0.07 vs 0.05) — the seed reaches only spotting, not the ERC stream or wind
draws, so it buys **no variance reduction at all**. Consequences:
- Pairing is statistical, never exact. **Every reported Δ needs a null band.**
- Per-pixel ΔBP noise scales as 1/√N: p95 ≈ 0.022 at N=1,000, ≈ 0.011 at N=4,000.
  Resolving a 0.01 BP effect needs **N ≈ 4,900**.
- **`Σ BP × cell_area` (the Ager TF_ij estimator) is far more robust** — pixel errors average out.
  Expected area burned varies only 1.46% run-to-run at N=100, ≈ 0.23% at N=4,000. Prefer
  area-integrated transmission metrics over pixel-level ΔBP wherever the question allows.
- Runtime measured at **1.22 s/fire at Duration 7** on this 740×783 @ 90 m grid → ~4.1 h per run at
  `Duration=21, NumFires=4000`; 30 runs ≈ 5.1 days serial, ≈ 1.3 days at 4 concurrent processes.

P0.2 (full runtime grid) and P0.3 (ignition footprint size) remain unrun — see
`code/dev/FSPro/p0_experiments.py`.

### Phase 2 — domain, arms, and design fires
**`prepare_fspro_experiment()` is the entry point.** `prepare_container_fspro` and
`prepare_counterfactual_fspro` are frozen (one/two arms, one ignition, no new features);
`prepare_counterfactual_ignition_set` is a deprecated alias that raises on the removed
`sector_deg` / `require_treatment_intersect` kwargs.

- **N arms**: `lcps={"untreated": …, "background": …, "coswap": …}` + `contrasts`
  (default = every ordered pair, so the Phase 3 additivity check has all three).
  Two-arm `baseline_lcp_path`/`treated_lcp_path` shim still works; mixing the two raises.
  `runs.csv` gains an `Arm` column and a `w_i` column; `output_basename` is the arm name.
- **Grid congruence is asserted before any run** — CRS, transform, and shape must match
  across arms, because `xr.align(join="left")` in `delta_burn_probability` would silently
  paper over a mismatch and return a plausible, wrong Δ surface.
- **`domain_gdf` is never used to clip** — validation and manifest provenance only. It
  warns when the LCPs don't cover the domain. `container_gdf` in
  `postprocess_fspro_outputs` is documented display-only; Δ statistics run unclipped.
- **`ignition_mode` has no default** (defect #1). Omitting it raises with an explanation.
  `"container"` is kept only for reproducing an observed fire from its real perimeter.
- **One shapefile per ignition** (P2.5). `create_random_ignitions` and
  `create_fod_ignitions` now return `list[Path]` and take an `out_dir`, not an `out_path`.
  Both used to write all N circles into one file, which FSPro reads as one fire starting
  simultaneously at all N points. `create_fod_ignitions`' old docstring claim that
  overlapping circles weight dense areas proportionally was false for the same reason.
- **Ignition footprint** defaults to a **10-acre** day-1 perimeter (`footprint_acres`);
  `None` gives the old half-pixel circle. P0.3 has not been run — revisit when it is.
  `footprint_ac` is measured off the written polygon (≈9.999 for a nominal 10), not the
  nominal circle.
- **Downwind cone replaces the zero-width ray test.** Measured on the real LCP: the ray
  kept 7,219 of 20,732 candidates where the cone keeps 20,656 — **the ray discarded 65%
  of sources whose fire would burn straight through the treatment.** Half-angle defaults
  to the narrowest arc holding 50% of non-calm fire-hour wind frequency
  (`wind_cone_half_angle`) — **±38° for pyrome 47**, vs ±16° at 25% coverage.
- **Design fires** are stratified over bearing × distance (default 3×2), allocated
  proportional to each stratum's density mass with ≥1 per non-empty stratum, and drawn
  within a stratum ∝ density. Weights are Horvitz–Thompson,
  `w_i = stratum_mass / draws_in_stratum`, normalized to sum 1.
- `ignition_density_surface` smooths on a **coarsened working grid** (σ ≈ 4 cells) and
  resamples back. Smoothing at LCP resolution with a 20 km bandwidth is a 667-cell σ over
  a heavily padded array and does not terminate.
- **The public clustering function is `check_ignition_clustering`, not `test_*`** — a
  public `test_`-prefixed name gets collected by pytest in any suite that imports it.

### P2.2 ANSWERED — Ager's uniform-ignition assumption FAILS for Colorado
`code/dev/FSPro/p2_ignition_clustering.{py,json}` (committed; `data/` is gitignored).
FPA-FOD Class D–G, **1992–2024**, CO pyromes, 199 sims. Mean nearest-neighbour distance,
observed ÷ CSR null:

| subset | n | obs NN | null NN | ratio | p | L(r)−r @ 50 km |
|---|---|---|---|---|---|---|
| all     | 2,864 | 5,561 m | 7,715 m  | 0.721 | ≤0.01 | +11,074 |
| natural | 1,355 | 7,322 m | 11,320 m | **0.647** | ≤0.01 | **+27,158** |
| human   | 1,184 | 8,835 m | 12,118 m | 0.729 | ≤0.01 | +12,497 |

All clustered; `L(r)−r` is outside the envelope at every radius 5–50 km. **The Natural
(lightning) subset is the most clustered** — exactly the population Ager described as
*"lightning caused and randomly located"*. Density-weighted sampling is therefore a real
methodological improvement here, not a refinement.

**Caveat to state when reporting:** the null is CSR over the pyrome polygons, *not*
restricted to burnable fuel, because no pyrome-wide FBFM40 is on disk. Some of the
clustering is fuel availability rather than ignition process. `mask_burnable=True` with
an `lcp_fp` runs the stronger test once such a landscape exists.

### FPA-FOD on disk — mind the vintage
- **Authoritative, through 2024**: `/Users/mcc/Library/CloudStorage/Box-Box/MCC/data/
  wildfire/FPA_FOD/RDS-2013-0009/Data/FPA_FOD_20260615.gpkg` — layer `Fires`, 2.66 M
  records, EPSG:4269. Read with a `where=` clause (Class D–G over 7 states ≈ 1 s).
- Everything under `data/spatial/raw/fpa_fod/` is **older**: the `SHP/Fires_ClassDEFG_*`
  layers stop at **2022**, and `DRAFT_7th/FPA_FOD_DRAFT_7thED_points.shp` is a truncated
  export holding only **1992–2009**. The `.accdb` beside it is the full draft 7th ed.
- `_NB_CODES = {0, 91, 92, 93, 98, 99}` in `lcp.py` is **correct** — Scott & Burgan FBFM40
  defines NB1/2/3/8/9 only; 94–97 are unassigned. (Plan flagged this as needing a check.)

### Phase 3a — pre-flight correctness (branch `fspro-phase3a`)
Scoped ahead of the COSWAP production test: fix what would silently corrupt results or
waste a multi-day campaign. The analysis layer (P3.1 remainder, P3.3 transmission matrix,
P3.4 exposure, P3.5 decomposition) is deliberately deferred to Phase 3b, to be built
against real output.

- **`summarize_bp_treatments` is a rewrite, not a patch — its signature changed.** The old
  one called `geom_to_raster_crs` with **three** args against a two-arg signature, so it
  raised `TypeError` on the first polygon and **had never run**. It now takes
  `(zones_gdf, delta_bp=…)` or `(zones_gdf, baseline_bp=…, treatment_bp=…)` and computes
  zonal stats on the **aligned** Δ raster. Returns `dBP_sum_ha` — the area-integrated
  ΔTF_ij estimator — alongside the pixel means.
- **`delta_burn_probability` writes explicit nodata** (`DELTA_NODATA = -32768`). Masked
  pixels used to descale to a plausible-looking **−327.68**. Overflow is now rejected.
- **Grid congruence is enforced at difference time** (`strict_grid=True`), not just in
  `prepare_fspro_experiment`. `xr.align(join="left")` silently papers over a shape or
  half-pixel transform mismatch and returns a plausible, wrong Δ surface.
- **`aggregate_ignition_bp` consumes the design weights.** Phase 2 wrote `w_i` to
  `runs.csv` and nothing read it. Weights are resolved against the **full** input list
  before dropping ignitions with missing output, so a dropped arm cannot shift the
  weight-to-ignition correspondence. `delta_std` is now written to disk.
- **`n_ignitions` cannot detect co-burn for BP** — unburned interior pixels are `0.0`, not
  nodata, so it equals the ignition count almost everywhere. **Use the new `n_burned`** to
  mask BP surfaces; `n_ignitions` remains correct for FL/arrival grids, which are NaN
  outside the burn.
- **The ignition footprint is masked from every Δ statistic** (`ignition=` on all four
  entry points; `ignitions=` takes one footprint per design fire).
- **`downwind_treatment_effect`**: `src_crs` was hard-set to `None` on both branches, so
  the reprojection was unreachable and a WGS84 polygon returned all-NaN. Now resolves CRS
  from the object, then `src_crs`. FlamMap's `-1`/`-2` slope sentinels are rejected.
- **`run_fspro(check=True)` is the highest-value fix for production.** `subprocess.run`
  had no `check`, the return code was never read, and `run_fspro_batch` marked any
  non-raising run `"success"`. A run now fails loudly unless it exits `0` **and** writes
  output; the batch prints a failure summary and carries `Arm`/`w_i` through.

**Noise floors (`spread/noise.py`).** P0.1 settled that every reported Δ needs a null band
and nothing computed one. Calibrated on the P0.1 measurement (N=100, Duration 7: pixel
|ΔBP| p95 0.07, area CV 1.46%) and scaled by 1/√N. `area_noise_floor(for_difference=True)`
applies the √2 two-run inflation; the area calibration is a **CV on the total**, so
`annotate_noise_floor(metric="area")` requires a `total_col`, not just the delta.

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
335 tests as of Phase 3a: `test_fspro_input.py`, `test_fspro_outputs.py`, `test_erc_classes.py`,
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
