# pyretechnics transmission — decision log

Dated record of what was decided and why, so `CLAUDE.md` can stay short. Stage narrative for
Stages 0–2 (engine install, de-stacker, crown gate, benchmark, spotting tail) is in the git history
of `CLAUDE.md` before 2026-09-12 and in `code/dev/pyre/*.py` docstrings. The plan is
`pyretechnics_transmission_plan.md`; where this log and the plan disagree, this log wins.

## Standing facts (measured, each has a unit test)

| fact | where enforced |
|---|---|
| `wind_speed_10m` is km/h, `temperature` °C, `upwind_direction` met FROM, FM as fractions | `test_pyre_episodes.py::test_frame_units_match_engine` |
| `time_of_arrival` is minutes; `AvgTime.tif` is written in **days** | `test_aggregate_and_paired_deltas` |
| LCP slope is `SLPD` degrees; `destack_lcp` converts to rise/run and refuses unknown units | `test_pyre_destack.py` |
| Engine pinned at `2026.8.10`; CFRI build `+cfri1` exposes the crown gate; every manifest stamps version, gate, solver knobs | `test_pyre_engine.py` |
| Same seed → bitwise-identical output; sham arm → Δ = 0 everywhere; worker count irrelevant | `test_pyre_runner.py` |
| `ignite_cells` wants a float window with **−1 inside** the footprint | `test_ignition_window_marks_footprint_negative` |

## Decisions

**2026-09-10** — Port transmission from FSPro to pyretechnics (plan approved). Reasons: mechanistic
spotting, exact pairing via common random numbers, native arrival time.

**2026-09-10** — pyretechnics cannot build in `fb_tools` (numpy 2 / Cython 3.1+). Second env `fb_pyre`
(py 3.11, numpy 1.26, Cython 3.0.12). No subprocess bridge.

**2026-09-10** — G1 (FlamMap cross-check) demoted to a diagnostic; exact unit tests are the stronger
check. FL correlates r = 0.91 with FlamMap; crown fraction diverges (21% vs 51%).

**2026-09-11** — Crown divergence is a hard-coded `canopy_cover > 0.4` gate in pyretechnics, at two
sites. Not a units bug. **Production runs at gate 0.0**; stock 0.4 is a small sensitivity subset.
Gate-off overstates intensity (all crown classed active), gate-on understates (no crown below CC 0.4);
the pair brackets the truth. Passive crown never appears at p97 at any threshold.

**2026-09-11** — Produce at **30 m**. 90 m burns 3.8× the area because mode-resampling destroys fuel
breaks. ~250 s per 21-day realization with spotting, 0.55 GB/worker.

**2026-09-11** — The 52 km domain fails `check_domain_adequacy` under constant p97 wind. Deferred:
re-check on real hourly episodes before any production campaign.

**2026-09-11** — Spotting is a heavy tail (breakout in ~30% of seeds adds +83–145% area; otherwise
< 1.4%). Report breakout rate plus conditional magnitudes, never a mean alone. Solver knobs pinned at
defaults (cfl 0.4, buf 3) and stamped.

**2026-09-12** — Direction reset with Max, superseding the plan's Stage 3–7 layout:
- **Prototype first, notebook-driven.** Two arms (untreated vs coswap), 3 existing design fires,
  5 episodes, central flux, gate 0.0 = 30 realizations. `code/notebooks/01f_PyreBuild.ipynb` exposes
  every step; methods stay small so the notebook can plot and iterate between them. The plan's
  3,600-realization design is a later scale-up on the same table.
- **Three modules, not six**: `models/pyre.py` (engine, cubes, ignition, runner, batch),
  `spread/episodes.py` (episode selection, hourly frame), `spread/pyre_outputs.py` (read, summarize,
  paired deltas, ensemble rasters). No `prepare_pyre_experiment` refactor of `container.py`; the
  runner consumes the Phase 2 design fires already on disk.
- **Wind from RTMA via GEE**, not HRRR via herbie. The hourly RTMA cache already comes from GEE;
  its `UGRD/VGRD` bands give hourly wind on the same grid and years as T/RH with no new dependency
  (`00b_RTMA-EMC.ipynb`, wind cells at the end). HRRR stays a future `weather/` integration. Until the
  export is run, `episode_hourly_frame(wind=(kmh, deg))` is a constant-wind placeholder and the frame
  says so in `attrs["wind_source"]`.
- **Pyrome-mean weather from the existing cache.** Focus-area-local RTMA/HRRR extraction is a later,
  general `weather/` feature (it serves FlamMap/FSim too).
- **Stage 1b folded into outputs.** `pyre_outputs.firebrand_production` is the engine's
  `expected_firebrand_production` vectorized (`f · 60 I / R · cell area`), read off each realization's
  intensity and spread-rate rasters per zone. No separate `burn_cells` product. **Firebrand production
  and cast are a key analysis direction to get right for counterfactual and risk work** — revisit
  the flux calibration and the jump-distance check (plan Stage 7) once real-wind runs exist.
- Episodes span 2011+ (RTMA), not 2016+ (the HRRR floor no longer binds). **But the RTMA EMC cache
  has export holes** (pyrome 45: 2012–2015 and 2024 carry day-scale gaps, 2013 is missing 844 h;
  2016–2023 and 2025 are complete to within one hour). `select_episodes(rtma_df=...)` therefore
  refuses windows with a gap > 3 h, and `episode_hourly_frame` raises rather than interpolate a
  longer one. Re-exporting the failed months in `00b` would widen the pool.

**2026-09-12** — Engine API audit against the pinned release's literate source
(`org/pyretechnics.org` @ 2026.8.10). What fb_tools should call rather than rewrite, and what it does:

| engine capability | Python-callable in the compiled build? | fb_tools |
|---|---|---|
| `burn_cells.burn_all_cells_as_head_fire` / `_toward_azimuth` — potential fire behavior, no spread | yes | **`pyre.burn_landscape`** (single-run product; FlamMap-style baseline map) |
| `burn_cells.burn_cells_by_time_of_arrival` — behavior along an arrival raster | yes | not needed: spread already writes behavior per cell |
| `eulerian_level_set.spread_fire_with_phi_field` (+ `spot_config`, `spot_ignitions`, `cube_refresh_rates`) | yes | `pyre.run_realization` |
| `SpreadState.copy()` + `start_time=stop_time` — pause / fork / continue | yes | not wrapped yet; the route to realized spot-ignition attribution and mid-run treatment forks |
| `SpaceTimeCube` 0-D/1-D/2-D/3-D; `LazySpaceTimeCube` | yes | `pyre.build_cubes` (1-D weather, 2-D fuels); Lazy not needed at 3 M cells |
| `fuel_spread_adjustment` (2-D) / `weather_spread_adjustment` (0-D) cubes | yes | `build_cubes(adjustments=...)` — the hook for `canopy_effects.csv`-style treatment multipliers |
| `spot_fire.schroeder_ignition_probability`, `firebrand_flight_survival_probability` | yes (the only two) | **`pyre.spot_ignition_probability`** (receptivity) |
| `spot_fire.expected_firebrand_production`, jump sampling, `spread_firebrands` | no (`cfunc`) | `pyre_outputs.firebrand_production` = `f·60I/R·A`, the engine's formula via `surface_fire.calc_areal_heat_output`; jump/cast stay inside the spread run |
| `fuel_models.moisturize / get_fuel_model / list_fuel_models`, `surface_fire.calc_areal_heat_output` | yes | unused; useful for fuel-model-level diagnostics (docs §7.2.1) |
| `crown_fire.van_wagner_crown_fire_initiation` | yes | used in tests only; the gate patch is the real interface |
| `suppression` (pure Python): `order_frontier_cells`, `score_frontier_cells`, `select_frontier_cells_for_suppression`, `build_firelines`, **`make_sdi_suppression_priority_fn(sdi_matrix)`** | yes | unused — direct hook for `suppression/sdi.py` output; next-session candidate |
| `load_landfire` | yes | deliberately not used (SLPP vs SLPD); its multiplier table matches `destack_lcp` exactly |
| docs §9.2.4 low-res spread then `copy_with_new_shape` upsample | yes | unused; possible cost saver but G2 showed 90 m destroys fuel breaks, so not for production |

Accuracy confirmations from the audit: unit table in `destack_lcp` equals the engine's own
`landfire_array_conversions`; `firebrand_production` equals `expected_firebrand_production` with the
engine's `calc_areal_heat_output`; spotting defaults in `SPOT_CENTRAL` equal the engine's
(`firebrands_per_unit_heat` 5e-11, `downwind_distance_mean` 11.7, exponents 0.3/0.6, variance ratio
600, crosswind sd 10, decay 500). Note the docs' own worked example uses `firebrands_per_unit_heat = 1e-6`,
2e4× the default — the flux axis is genuinely unconstrained upstream.

**Single-run vs delta are now separate entry points.** `burn_landscape` and `run_realization` take one
`layers_dir` and one weather state or episode; nothing in them knows about arms. Pairing lives only
in `build_runs` / `run_batch` / `paired_deltas` / `aggregate_realizations`, which consume those
single-run outputs. Fire-behavior options (`use_wind_limit`, `surface_lw_ratio_model`,
`crown_max_lw_ratio`) are explicit kwargs and stamped, with the engine defaults.

**2026-09-12 (later)** — RTMA wind export landed: `data/weather/pyrome_emc/RTMA_WIND_MONTHLY/
rtma_wind_InterWest_pyromes_<yyyy>_<mm>.csv`, 12 InterWest pyromes, Apr–Oct 2011–2025, one file per
month. `load_rtma_hourly` now globs both `*_<year>.csv` and `*_<year>_<mm>.csv`. Facts measured on
pyrome 45:
- The wind files carry exactly the same missing hours as the T/RH record (both come from the same
  RTMA images): 2016–2023 and 2025 complete, 2012–2015 and 2024-05 gappy, 2013-09 empty.
- **Use `RTMA_MONTHLY` (InterWest, fire-masked) with the wind, not `RTMA_EMC` (older CO yearly).**
  The two disagree on the same hours (2020-08: mean −2.0 °F, up to 3.9 °F / 5.9 % RH) because the
  monthly export applies the fire mask. EMC differs by < 0.1 pp on average, so dead FM is barely
  affected; the wind export is only coherent with the monthly record. The FlamMap/FSPro RTMA
  caches still derive from `RTMA_EMC`.
- **Pyrome-mean wind is calmer than the design wind.** Fire-hour (13–16 MDT) speed, Jul–Sep 2016+:
  p50 9.4, p90 14.3, p97 16.7 mph, vs the FlamMap `p97` scenario's 23.1 mph (point-scale, 20 ft).
  Spatial averaging over the pyrome damps the peaks; direction stays coherent (|mean vector| /
  mean speed 0.78 at fire hours; 65 % of fire hours from WSW–WNW).
- Episode pool at 21 days with gaps ≤ 3 h: p97 supports at most 11 non-overlapping episodes,
  p90 ≥ 20. Five p97 episodes is fine for the prototype.

**2026-09-12 (evening)** — Two decisions with Max, both measured on ep_003 (2020-08-18, pyrome 45):

*Wind: HRRR, local, spatially resolved.* Fire-hour speed over the 504 h episode, mph:

| | p50 | p97 | max |
|---|---|---|---|
| RTMA pyrome mean (frame) | 10.2 | 17.0 | 18.8 |
| HRRR mean over the 52 km domain | 11.7 | 17.7 | 19.3 |
| HRRR windiest 3 km cell in the domain | 21.0 | 29.4 | 31.9 |

The domain mean is the pyrome mean (ratio 1.06, r = 0.69): changing the *source* buys nothing, any
spatial *mean* erases the design wind. 139 of 504 hours have a cell above 20 mph; no hour's mean does.
Local direction differs from the pyrome-mean direction by a median 46° at fire hours (64° all hours).
So each episode gets a 3-D wind cube: `weather.hrrr.fetch_hrrr_wind_grid` (byte-range GET of the two
10 m U/V messages from the AWS `.idx`, ~2 MB/h, cached; **no herbie** — its index parser fails on every
pre-2019 file, and the archive starts 2014-07-30) → `spread.episodes.episode_wind_cube` (nearest HRRR
cell per block on a block grid that divides the landscape exactly; the engine repeats integer factors,
and **does not check divisibility for 3-D bases**, so `build_cubes` does) → `.npz` beside the frame →
`build_runs(wind_cubes=...)`. T/RH → NFDRS FM stay on the RTMA pyrome-mean frame. Episodes are
restricted to 2016+ (`YEARS` in the notebook). Manifests stamp `wind_source` and `wind_cube_shape`.
Block for the current 1738×1730 grid is 79×10 cells (2.4 km × 0.3 km) — a re-cut LCP should have
row and column counts that are multiples of 100 so the block is 3 km square.

*Domain: the 52 km landscape fails on real weather too — Max re-cuts a wider LCP (~100 km).*
Six realizations (3 design fires × 2 arms, RTMA wind, gate 0.0, central spotting, 21 d) all burned
31.7–32.4 k ha, reached the domain edge on day 3.4–5.5, and filled 44–45 % of the 72 k ha connected
fuel patch, which itself touches the edge. The same 32.4 k ha ceiling appears in the constant-wind
21-day benchmark. Under this geometry area Δ is zero by construction and arrival Δ is noise
(mean ≈ 0 h, p5/p95 ≈ −13/+17 h); runtime was 65 s per realization because the fire runs out of room.
Duration is re-decided by `check_domain_adequacy` on the new domain, not fixed at 21 d.

*Provenance:* the FlamMap/FSPro RTMA caches stay on `RTMA_EMC`; the pyre path uses `RTMA_MONTHLY`.

**2026-09-12 (night)** — Housekeeping decided with Max:
- **herbie is gone.** Its `.idx` parser fails on every pre-2019 HRRR file (an empty forecast field
  becomes a missing value under pandas' pyarrow string dtype and breaks the join). Both HRRR paths —
  `fetch_hrrr_winds_at_fires` (fire-point climatology) and `fetch_hrrr_wind_grid` (episode cubes) —
  now share `hrrr_uv10`: byte-range GET from the AWS index, cfgrib decode, one cache. `cfgrib` is
  declared in both environment files (installed into `fb_pyre` so the notebook can fetch in-kernel).
- **Notebook split.** `code/notebooks/01f_PyreBuild.ipynb` is the generic "burn a landscape" example
  (14 cells: destack → potential map → one episode → HRRR cube → one burn). The COSWAP arms/design
  fires/paired-Δ work moved to the project folder as `fire_behavior/code/05_PyreTransmission.ipynb`.
- **Example landscape: the `front_range` test AOI** (`data/spatial/raw/test_aois.gpkg`, ~2,060 km²,
  pyrome 46) via `lfps_request`. The treatment-effects `LCP_p46_Baseline.tif` was tried first and is
  89 % nodata (masked to the study's forest types), so no contiguous window exists in it.
- Scratch: the North Cottonwood RTMA/HRRR pair runs were deleted after being logged above;
  `layers/`, `episodes/`, `hrrr_cache/` under `~/fb_scratch/coswap_pyre` are kept for the project
  notebook. The example uses `~/fb_scratch/pyre_example`.

**2026-09-13** — Review pass with Max: consistency, duplication, and the example notebook's direction.
- **Goal restated.** Before COSWAP arms: a baseline probabilistic workflow — deterministic behavior map,
  point ignitions, arrival/flame-length/ember outputs, then an *ensemble* large enough to make a
  probabilistic arrival surface. `01f_PyreBuild.ipynb` now ends with that ensemble (section 6).
- **Ignitions follow FSim's ignition-density-grid step.** `fuelscape.ignitions.sample_density_ignitions`
  draws sites ∝ `ignition_density_surface` without replacement (so `w_i` is uniform), one footprint
  shapefile each. The example builds the IDG from the authoritative FOD (through 2024), size class ≥ B
  (class A is sub-¼-acre starts, mostly human and caught at once), 5 km kernel. `front_range` holds 433
  class-B+ fires (1,260 within 20 km) vs only 25 class D–G — too few for a surface. What the example does
  **not** copy from FSim: the ERC-conditioned number of fires per day, and season-long ignition timing
  (all sites start on day 1 of a severe episode). Both are the next FSim-alignment steps if wanted.
- **Ensemble size for the example: 30 sites × 3 `p97` episodes = 90 realizations** at 7 days, spotting on,
  4 workers. Measured headless 2026-09-13 on front_range: 90/90 ok, 23.6 min compute, **6.7 min wall**
  (~16 s per realization; fires are larger than the earlier 6 s single run). Fire size mean 11,374 ha,
  median 8,365, sd ≈ mean, min 454 — the heavy tail is the ensemble, not a nuisance. `Σ BP × area` equals
  the mean fire size to the hectare. Episodes drawn: 2018-06-14, 2020-09-06, 2020-10-09; the 2020-09
  episode's ERC drops to a 45.7 mean after a day-1 80 — a real front passage the design keeps.
- **New ensemble products** in `pyre_outputs.aggregate_realizations` (streamed, one run in memory at a
  time): `ArrivalProb_d{d}` = fraction of realizations reaching a cell by day *d* (the probabilistic
  arrival surface; marks past the longest run are dropped) and `Firebrands` = expected firebrands per
  cell averaged over realizations (`firebrand_production` per run; a spot-fire *source* map — realized
  landings remain unexposed by the engine). `BurnProb`, `AvgTime` (days), `AvgFlameLength`, `n_burned`
  unchanged.
- **One HRRR cache.** `weather.hrrr.default_hrrr_cache()` = `$FB_HRRR_CACHE` or
  `~/fb_scratch/hrrr_cache`, the default for `hrrr_uv10`, `fetch_hrrr_wind_grid` and
  `fetch_hrrr_winds_at_fires`. The two per-project caches (5.7 GB + 357 MB, 48 hours duplicated) were
  merged into it (2,309 hours). Files are CONUS-wide two-message subsets, so every consumer shares them.
- **Duplication removed.** One `uv_to_speed_dir(u, v, units)` (was written 4×, mph in hrrr.py and km/h
  three times in episodes.py); one `longest_true_run` (was 3×); `REALIZATION_LAYERS` defined once in
  `pyre_outputs` and imported by `models.pyre`; one `rasterize_mask` behind `ignition_matrix` and
  `zone_masks`. Archive constants are now the pair `HRRR_ARCHIVE_START` (2014-07-30, refuses earlier) and
  `HRRR_RELIABLE_YEAR` (2016, the climatology filter) — the old `_HRRR_START_YEAR` was the second one.
- **Stale text fixed:** `convert.py` docstring (`build_static_cubes`), `environment-pyre.yml`
  (`03b_Transmitted_Pyre`), README's empty "Optional" block, `00c`'s "Requires herbie-data" comment; the
  plan doc carries a superseded banner. `burn_landscape` no longer breaks on a 3-D wind cube in its
  manifest stamp.
- **Cache hygiene is now enforced in code, not by naming.** Re-running `01f` with the class-C ignition
  grid reproduced the class-B ensemble to the hectare in 0.0 min: the run dirs are keyed by ignition *id*
  and `run_realization` returned any existing manifest. `run_realization` and `burn_landscape` now stamp a
  `request_hash` (sha256 of seed, duration, layers dir, ignition window, weather arrays, spot config, gate,
  solver knobs, adjustments, behavior options) and re-run when the stored hash differs or is absent
  (`test_cached_manifest_is_reused_only_for_the_identical_request`). Pre-fingerprint manifests are treated
  as stale. Wind cubes are still named by episode start date because `.npz` files carry no manifest.
- **IDG from larger fires (Max, later the same day): size class ≥ C (10 ac+), kernel 10 km.** The
  front_range 0.6° window holds 494 class-C+ fires (70 inside the LCP, 148 within 15 km) vs 159 class-D+
  (25 inside), so C is the smallest class that still gives a surface rather than spikes at 10 km. Sites are
  **grid cells drawn ∝ the smoothed surface, never the FOD points themselves**
  (`test_sample_density_ignitions_draws_cells_not_fod_points`).
- **Production surfaces need more than 90 runs.** Cost on this grid ≈ 4.5 s wall per 7-day realization on
  4 workers (1,000 ≈ 75 min, 5,000 ≈ 6 h). Raise sites first, then episodes; judge convergence by a
  split-half `BurnProb` comparison. The notebook's "Next" carries this guidance.
- **Stale code removed:** `episode_hourly_frame(wind=...)` constant-wind placeholder (RTMA wind is on disk
  and HRRR cubes are the design; the frame now requires `u_ms/v_ms/ws_ms`), the unused `herb_scale` kwarg of
  `weather_from_scenario`, the unused `transform=/crs=` overrides of `run_realization`, and the unused
  `models.run_pyre_batch` alias (call `pyre.run_batch`). `load_static_layers` reads `fuel_model` as
  float32 — the engine copied int64 to float32 on every run with a warning.
- Scratch: `~/fb_scratch/coswap_pyre/{pyretechnics-src,benchmark,smoke,runs,potential,ensemble}` and
  the old `pyre_example` outputs are approved for deletion (Max runs the `rm`; the sandbox refused it).
- Pre-existing, unrelated: `tests/test_mask_lcp.py::test_output_is_lzw_compressed_and_tiled_by_default`
  fails on `main` too (writer defaults to deflate, test expects lzw).

**2026-09-13 (late)** — The class-C ensemble lost 4 of 90 realizations and `run_batch` hung.
- **Engine bug, measured.** Four `SIGSEGV` crash reports (`~/Library/Logs/DiagnosticReports/python3.11-*`),
  identical stacks: `spread_fire_with_phi_field → spread_one_timestep → route_cell_to_diff →
  is_frontier_cell → spot_fire.is_burnable_cell → SpaceTimeCube.get`. Cause: on the final step the engine
  indexes cubes at `int(stop_time // band_duration)` (`eulerian_level_set.py` ~L2690), which equals the
  band count when a run ends exactly at `max_duration`; `check_start_and_stop_times` rejects only `>`.
  For a 2-D base that is `data[1, y, x]` on a `(1, rows, cols)` array — an out-of-bounds read with
  `boundscheck(False)`: garbage on most heaps, a fault on some. The same row ran fine alone (0.6 s), so the
  crash is memory-layout dependent, not input dependent, and the "same seed → bitwise identical" contract
  was only holding because the garbage was usually benign. **Report upstream.**
- **fb_tools fix:** `pyre.pad_bands` gives every weather cube `ceil(duration_h) + 1` bands (last hour
  repeated), so the final lookup is a real band. Stamped into `request_hash`, so all cached runs re-run.
- **`run_batch` no longer hangs on a dead worker.** `multiprocessing.Pool.map` waits forever for a lost
  task. `_run_parallel` feeds a `ProcessPoolExecutor` (`spawn`, `max_tasks_per_child=1`) at most
  `n_workers` rows at a time; when a worker dies, the rows in flight are re-run one per isolated process
  at the end, so a crash names its row and only that row is reported `status == "error"`
  (`test_parallel_batch_survives_a_dying_worker`). Progress prints every 10 %.
- Class-C ensemble before the fix, 86 completed: 26.4 min compute, per-run wall 0.1–89 s (median 12),
  fire size mean 11,675 ha; 14 of 86 fires touched a domain edge — the front_range window is not wide
  enough for a 7-day p97 fire from the interior, so `BurnProb` is edge-truncated there.

- **Final class-C ensemble after both fixes (headless, 2026-09-13): 90/90 ok, 0 errors, 28.2 min compute,
  8.1 min wall on 4 workers, no crash reports.** Fire size mean 11,386 ha, median 10,194, sd 9,290, range
  108–39,866; `Σ BP × area` = mean size. `BurnProb > 0` covers 171,780 ha of the 270,000 ha window.

**2026-10-08** — Single-realization treatment counterfactual in `01f_PyreBuild.ipynb` §5b (Max's choices):
- **Feature:** the OSM tertiary road the baseline fire crossed (Overland Road / James Canyon Drive), clipped to
  the baseline perimeter + 500 m — not the whole road, not a road network. Roads come from `fetch_osm_roads`
  within 2 km of the perimeter, cached in scratch; **osmnx is now in `environment-pyre.yml`** for this.
  The baseline perimeter is itself held by LANDFIRE's non-burnable road cells on three sides (primary, trunk,
  secondary): a 30 m NB line stops the level set, and central spotting did not carry it over in this draw.
- **Prescriptions:** two treated arms from the tealom effects tables at the same width — *shaded fuel break*
  (`Mech Thin` canopy × `Complete` surface) and *clear cut* (`Patch` × `Patch_SM` → GS1). No non-burnable arm:
  the existing road cells already show what a hard barrier does.
- **Width:** 120 m total (4 cells), fixed; a sweep is one more arm per width, not built.
- **Mechanics:** `fuelscape.adjust.fuel_break(lcp, lines, width_m, canopy_df, surface_df, scenario, out_path)`
  rasterizes the corridor (valid cells only), calls `apply_treatment` with the mask, writes an int16 LCP with
  band descriptions kept; `destack_lcp` per arm; `changed_cells` = treatment footprint = `trt` zone; the
  three arms go through `build_runs`/`run_batch` so the baseline row reproduces the single run bitwise.
  Products kept minimal: arrival maps per arm, Δ arrival + lost-burn map, `paired_deltas`, treatment shadow
  (share of baseline burn behind the break and how much of it still burned), in-break intensity against the
  hauling chart. Tests: `tests/test_fuel_break.py` (engine-free).
- **Measured (ep_000 × ign_000, 21 d, central spotting, 1.6 min for 3 arms):** footprint unchanged — 18,197 /
  18,211 / 18,223 ha; corridor reached at 23 h in every arm; 100 % of the 11,094 ha behind the break (shadow
  ≥ 0.5) burns in both treated arms, median arrival 0.4–0.6 h *earlier* (GS1 spreads faster than TL/TU litter).
  In-corridor mean intensity 1,423 → 863 (shaded) → 35 kW/m (clear cut, hand class); collar unchanged
  (~2,350). Reading: an unsuppressed 21-day fire fills the road-bounded block regardless; the break changes the
  fire where a crew would stand, not where it ends. Next, if wanted: the same arms under the stepped
  containment rule, and the arms through the §6 ensemble for breakout rates. Not built.

## Open items

- **Realized spot ignitions are not exposed** by the engine's final matrices; `spot_ignitions` in the
  output is only the pending set at stop time. To attribute cells to embers, step the run in chunks
  (`max_duration`, carrying `spot_ignitions` and `random_generator`) and diff the pending set.
- ~~Domain adequacy on real episodes~~ — fails (2026-09-12 evening); waiting on the wider LCP.
- **Next session**: (a) pause/fork wrapper over `SpreadState.copy()` for spot attribution and mid-run
  treatment forks (note: the RNG is re-seeded per call, so chunked ≠ single-run bitwise, but stays
  deterministic); (b) ~~`suppression.make_sdi_suppression_priority_fn` with `calculate_sdi` output~~ (done: `PyreFirelines`, see 2026-11-23) — see the
  `sdi_static_layers` / `SDI_NODATA` commit on this branch (treatment-invariant SDI inputs cached
  once, SDI written on the slope band), which is the raster that hook would consume;
  (c) `fuel_spread_adjustment` from the COSWAP canopy-effects table; (d) per-pixel receptivity map
  once a spatial FM/temperature surface exists.
- Passive crown at p75/p90 — does it appear where the Cruz rate no longer dominates?
- `SharedMemory` for static cubes if worker RSS becomes binding at scale (not needed at 30 runs).

**2026-09-13 (review with Max)** — What the two ensembles actually showed, and the redesign.
- **`01f` front_range, 100 sites × 5 p97 episodes × 21 d (495 ok):** 87 % of fires reached the domain edge
  (35 % on day 1; the AOI is a 30 × 90 km strip), site explains 85 % of log fire-size variance and episode 2 %,
  split-half `BurnProb` by site disagrees by |Δ| p90 0.12 against a median BP of 0.10. The "blocky" BurnProb is
  ~100 truncated 21-day footprints each stamped 5×; more sites would smooth it only after the truncation is
  fixed. front_range stays a QA/parameter landscape only.
- **`05` North Cottonwood, 30 paired realizations:** 18/30 sit on the 32.4 kha domain ceiling; in 4 of 15 pairs
  one arm spots into a second fuel patch and the other does not (same seed) → ±27 kha area deltas of random
  sign. **Exact pairing does not make far-field or total-area deltas small when spotting is on**; the stable
  signal is near-field (intensity in the treatment fell in 14/15 pairs, firebrands in 11/15, median |Δ area|
  on day 1 = 54 ha).
- **Redesign, one SFA at a time (Upper Arkansas RMRI first, pyrome 45):** two products. *Location* — one arm
  (`background` = sc1c, every treatment except COSWAP), density-drawn ignitions inside the SFA + 10 km ring,
  7-day p97 fires, and per fire the **treatment shadow** (`spread/transmission.py`: BFS from the COSWAP cells
  the fire burned along 8-neighbour edges that only run forward in arrival time). Value blocks are 1 km cells
  with ≥ 10 Overture structures. Per block: `p_reached`, `p_intercepted`, lead time; per project: fires
  reaching it and the share that then reached a block behind it; rasters `BurnProb`, `ShadowGivenBurn`.
  *Impact* — only the fires that burned into a COSWAP atom are re-run on `coswap` (sc2c); paired near-field
  deltas (intensity, crown share, firebrands in the atoms and a 500 m collar, day-1/3 area, arrival delay at
  intercepted blocks) plus a breakout rate for |Δ area| > 1,000 ha. Total-area means are not reported.
- **Arms are `background` vs `coswap`, not `untreated`.** The 657 UA "treatment atoms" in the CFT table are
  every recorded treatment; only 42 atoms / 255 ha / 6 projects are COSWAP-funded
  (`Focus_Area_analysis/Data/spatial/mod/COSWAP_treatment_interactions.gpkg`, SFA from `EVENTS`). sc2c − sc1c
  differs on 0.05 % of UA cells; that is the COSWAP effect being tested.
- **Domain:** `fuelscape.clip_lcp` cuts the statewide arm to the SFA + 15 km, padded to 100-cell multiples
  (UA: 5000 × 2900 = 14.5 M cells, 41 s and 3.7 GB per 7-day realization with spotting → ~600 runs/h on
  6 workers). Duration is judged by `pyre_outputs.edge_contact` (first edge-contact day per run), not
  `check_domain_adequacy`.
- **Shadow validated on front_range:** a 1 km treatment in the head-fire path shadows 42 % of a burn, one on
  a flank ~0–6 %; every shadow cell arrives after the treatment entry. Strict monotone edges (no tolerance):
  a tolerance leaks the shadow upstream through level-set wrinkles.
- **Realized spot ignitions cannot be read as arrival-time local minima.** With spotting *off* the level set
  still leaves single-cell minima (gaps up to 23 min to their neighbours) where fronts merge; on/off runs
  give the same set. The chunked-stepping route (open item) stands. Spotting on/off comparisons stay on
  front_range.
- **A fire that never leaves its footprint is an outcome** (`stop_condition == "no_spread"`, layers written),
  not an error — front_range site 54 sat in an urban block and failed all 5 episodes as "errors".
- Episode frames now carry `rh_pct`; `episodes.episode_weather_report` is the per-episode weather table for
  the report (ERC, T max, RH min, FM1 min, RTMA and HRRR fire-hour wind percentiles, dominant direction).
  `summarize_realization` adds `{zone}_crown_frac`.
- Severity stays p97 only (matches the in-situ analysis); a mixture is a later option.

**2026-09-13 (pilot result, Upper Arkansas RMRI)** — 200 sites × 3 p97 episodes × 7 d on `background`
(600/600 ok, 80 min wall on 6 workers; 3 no-spread; 0 % reached the edge; fire size median 20.7 kha,
p90 40.8 kha), then 186 paired `coswap` fires (the ones that burned into a COSWAP atom; 372 rows, 0 errors).
- *Location:* 31 % of fires burned into a COSWAP atom and 170 of those 186 went on to reach a structure block
  behind it. Mesa Antero / Mount Princeton / Weldon Creek each met ~110 fires (≈ 20 % of all fires),
  North Cottonwood 41, Feddema 15, Home Stake 11. 183 of 384 blocks were reached through a COSWAP treatment at
  least once; among blocks reached by ≥ 30 fires, 85 have `p_intercepted` ≥ 0.25 (3,001 structures) and 5 have
  ≥ 0.5. Median lead time treatment → block 52 h (p10 10 h, p90 107 h). Split-half convergence on
  `p_intercepted` (302 blocks, ≥ 10 fires per half): |Δ| median 0.013, p90 0.10, r = 0.94 — 200 sites is
  enough to rank blocks; 400 would tighten the tail.
- *Impact (coswap − background, medians over 186 pairs):* intensity in the atoms −2,040 kW/m (99 % of pairs
  negative; background median 2,400 kW/m), crown share −0.58 (background 0.67), firebrands in the atoms −0.78
  and in the 500 m collar −0.44 (96 % negative), day-1 area 0 (58 % identical), day-3 area −0.2 ha (50 % neg /
  29 % pos), arrival at blocks behind the treatment +0.4 h median (73 % delayed, p10/p90 −1.1/+5.7 h; 190 block
  arrivals disappear on the treated landscape). Breakout (|Δ total area| > 1,000 ha) in 25 % of pairs.
  Reading: the treatments reliably change *how* the fire burns through them and what they emit, and barely
  change *whether* or *when* it reaches the blocks behind them at 7 days — 255 ha against 20 kha fires.

**2026-09-13 (episode control, Max)** — `select_episodes` now separates the *pool* from the *rank*, both exposed
as notebook parameters (`EPISODE_POOL`, `EPISODE_RANK`) with the previous behaviour as the default (verified: the
same three Upper Arkansas p97 windows are drawn, so cached runs stay valid). Pool = climatology days at
`SEVERITY` in `YEARS` (default), or explicit `dates=` — `dates_from_fod` (discovery dates of FOD fires ≥ a size
class in the window; the observed-fire pool) or a named ignition date such as East Troublesome 2020-10-14 with
`severity=None`. Rank = seeded random (default) or top-n by `erc_day1` / `erc_mean`. Every row reports
`erc_pct` (where day 1 sits in the climatology) and `source`, carried into the weather report. Limits: dates must
fall in the ERC climatology years and the April–October season; windows that run past the RTMA record
(Oct 31) or hold a > 3 h gap are dropped with a count; HRRR cubes need 2014-07-30 or later. Satellite growth
records (MTBS/FIRED) as a pool are a later add — the interface is the same (`dates=`).

**2026-09-13 (fast-fire episodes, Max)** — Third episode pool: FIRED events whose maximum single-day growth
reaches **1,620 ha** (the fast-fire threshold of Balch et al. 2025, Science). `episodes.build_fast_fires_table`
reads the FIRED *events* GeoPackage once (`MCC/data/wildfire/FIRED/raw/fired_v2025var_…_events.gpkg`, 2000–
Feb 2026) into a lookup that **ships with the package**, `fb_tools/data/fast_fires_fired_v2025var.csv` (2,169 CONUS events, 188 KB: id, ignition /
last / max-growth dates, max-day ha, total ha, duration, land cover, ignition centroid lon/lat); rebuild it on a
new FIRED release. `dates_from_fast_fires(table, geom=|bbox=, days_before=3)` returns window starts
`days_before` the biggest day, so `duration_days = before + 1 + after` brackets the blow-up. Weather then comes
from the real record for those dates as for every pool. Counts inside the RTMA window (Apr–Oct 2011–2025):
pyrome 45 has 7 usable fast fires (12 total; largest day 10.9 kha), 46 has 12, 47 has 13, 53 has 7 (largest
192 kha/day, the 2018 grass fires). Only one fast fire sits inside the Upper Arkansas landscape itself (Hayden
Pass 2016, 3.5 kha/day), so the notebook's default area for this pool is the **pyrome**, consistent with the
pyrome-mean T/RH/ERC frames. East Troublesome's biggest day (2020-11-01) is outside the RTMA season and is
dropped with a count. ICS-209-PLUS can feed the same `dates=` interface later.

**2026-09-13 (near field)** — Review of the Upper Arkansas pilot's impact metrics
(`COSWAP/…/fire_behavior/docs/transmission_nearfield_review.md`):
- The 500 m collar effect was **zone leakage**: impact zones were `all_touched` polygons, COSWAP atoms are
  3–4 cells wide, so 50 % of treated cells sat in their own collar mask (6.6 % of collar cells;
  6.6 % × −2,036 kW/m ≈ the −119 kW/m collar median). Zones are now boolean masks from
  `changed_cells` + `distance_bands` (`near_field_zones`); `zone_masks` passes dicts through.
- Zone `_arrival_h` is first arrival = the windward edge, so `d_trt_arrival_h` / `d_collar_arrival_h` ≈ 0
  by construction. The timing effect lives in the lee: `lee_profile` reads paired deltas by distance band
  × shadow class (baseline shadow ≥ 0.5). Pilot: intensity/crown effect recovers < 100 m; arrival delay
  ~+1 h median (p90 3.5–5 h) carried flat to 8 km in the shadow, exactly 0 outside it.
- Hauling-chart classes (350 / 1,700 / 3,500 kW/m) are reported per band and arm: the bridge to
  suppression. Next: stepped runner (`SpreadState` continue, per-step `seed_for(seed, step)` in every arm,
  pending-spot logging), then the paired 2 × 2 {background, coswap} × {none, suppression} with ELMFIRE's
  area-growth containment (model 0) before direct attack. `fuel_spread_adjustment` vs crown spread is open.

**2026-09-15 (stepped runner + rule S1, Max)** — `run_realization(step_h=, suppression=)`:
- **Stepping.** `step_h` drives `spread_fire_with_phi_field` in windows; the engine's `spot_ignitions`
  (a `SortedDict`, handed back as a `dict`) carry over, the spotting RNG is re-seeded per window with
  `seed_for(seed, k)`, so pairing across arms stays exact (sham Δ = 0, tested). Stepped ≠ unstepped bitwise
  but indistinguishable in practice: one 7-day Upper Arkansas fire burned 13,726 ha in one call, 13,727 ha at
  1 h steps (Jaccard 1.000) and 13,728 ha at 3 h; engine time 11.6 → 17.5 → 13.3 s. **Every condition of an
  experiment is stepped the same way** (the notebook steps all four at 1 h). `step_h`/`suppression` enter
  `request_hash` only when set, so pre-existing manifests keep answering.
- **Rule S1** (`fb_tools/suppression/containment.py::SectorContainment`) = ELMFIRE containment model 0: target
  `dC = 0.01·diurnal·MAX_CONTAINMENT_PER_DAY·(1 − rate_ac_day/AREA_NO_CONTAINMENT_CHANGE)·exp(−B_SDI·SDĪ)·dt/24`,
  clipped to [0, 1]; the perimeter is split into 36 sectors around the ignition, the slowest sectors are held
  first, a held sector stays held. Held cells = the engine's unburned frontier cells in held sectors, set to fuel
  91 in the live `fuel_model` cube (`np.asarray(cube.data)` aliases the base). Held cells leave the engine's
  frontier, so the rule keeps its own count of them as perimeter — without that it froze every sector within
  three days. SDI is the package's `SDI97_*.tif` (raw SDI × 100, `sdi_scale` 100; clipped 0–3 like ELMFIRE),
  read from `layers_dir/sdi.tif` (`spread.convert.raster_to_layers`), NaN filled with the median of the known
  cells (25.7 % of the UA domain is known; 98.5 % of the pilot fire's burned cells). Written per run:
  `held_line.tif` (step index), manifest `suppression_log`, `held_ha`, `containment_target`;
  `pyre_outputs.suppression_summary/_batch` add `held_in_changed_share` (line on treated ground).
- **What the defaults do at this scale**: the same fire burned 12,974 ha with S1 (−5.5 %, 117 ha of line, all
  perimeter held by hour 108) and 6,251 ha at 3× `max_containment_per_day` (358 ha of line). Fires growing
  > 10,000 ac/day drive `dC` negative, so ELMFIRE's model is nearly inert during the run; the notebook exposes
  `PYRE_CONTAINMENT_PER_DAY` and reports the parameter with every number. Uncalibrated for Colorado; a
  sensitivity axis like the crown gate, never a containment prediction.
- **Ember lottery under suppression**: with the accounting bug the held line reshaped the front, the identical
  ember stream cast differently, and a second 18,177 ha fire started at hour 16 (10 of 3,558 held cells burned,
  all by embers). Not a level-set leak — but the reason the factorial has a spotting-off column and reports
  breakout rates. Direct attack (S2, `build_firelines`) is still not started; `fuel_spread_adjustment` vs crown
  spread still open.

**2026-11-23 (audit + ELMFIRE scoping, Max)** — Code-reading QA/QC of the pyre path, prompted by ELMFIRE 1.1
(2026-08-24). Full plan with file-level findings: the session plan "fb_tools pyretechnics audit, ELMFIRE
scoping, and burn-probability options"; this entry is the record.
- **Decided.** (1) ELMFIRE is a *spike first* — different tools for different jobs; compare with pyretechnics
  only where a comparison means something; no `fb_tools.models.elmfire` until the spike reports.
  (2) **Suppression stays out-of-the-box**: use what each engine ships, write no rules of our own for now.
  (3) Test the pyretechnics upgrade **in `fb_pyre`** (no new env), pin kept in `environment-pyre.yml`.
  (4) Tests stay light — small crop, short duration, few ignitions. (5) Correctness fixes before clean-up
  before ELMFIRE/BP. (6) Work one item at a time; anything that changes a reported number is asked first.
- **Log was behind the code.** The engine's own rule is implemented: `containment.py::PyreFirelines`
  (`"pyre_firelines"`, `make_rule`'s default) drives `pyretechnics.suppression` (`order_frontier_cells` →
  `score_frontier_cells` → `select_frontier_cells_for_suppression`, shipped SDI / flame-length priority).
  So "S2 not started" above and open item (b) are closed. **Ours, not the engine's**, in that rule: the constant
  construction rate (2,000 m/h), the optional flame-length gate on the rate, the median fill of unknown SDI,
  SDI ÷ 100, the hourly cadence. `"elmfire0"` (rule S1) is our re-implementation of ELMFIRE model 0 —
  **experimental from here, not extended.** Pending-spot logging per step is still not built (open item 1 stands).
- **Found by reading, not yet fixed** (fix order = plan Phase 1a): `ignition_matrix` never reprojects (docstring
  says it does); `pad_bands` repeats the last hour over any shortfall; `crown_gate=None` is hashed as `None`
  while the process-global gate may be whatever the previous row set; `run_batch` rasterizes on the first arm's
  grid without checking the others; `aggregate_realizations` pools arms; `summarize_realization` reports day
  marks past the run; `destack_lcp(overwrite=False)` can mix grids; `select_episodes(dates=)` skips `last_start`;
  `apply_treatment` overwrites `band_map`; pyre output grid checks are shape-only.
- **Open questions for Max** (each changes a number): HRRR 10 m U/V are used unrotated (grid-relative on the
  Lambert grid, ~5° in Colorado — confirm from the GRIB flag first); prevented arrivals drop out of
  `paired_deltas` as NaN; delta sign differs between `bp.py` (baseline − treated) and `paired_deltas`
  (treated − baseline); pyre `BurnProb` includes the ignition footprint; `Firebrands.tif` skips the slope
  correction; rounding before differencing. Shadow: spotting roots a new path at 0, and only treated cells
  that *burned* carry share — "passed through", not "affected by" — to be confirmed on synthetic cases.
- **Upstream.** pyretechnics 2026.10.20 fixes underpredicted directional spread, intensity and flame length
  (up to 2×) and changes the license (dual community/commercial); latest is 2026.11.20. Everything measured in
  this log is on 2026.8.10. In the pinned build and untried here: `burn_cells_by_time_of_arrival`,
  `SpreadState.render_at_new_resolution`.
- **ELMFIRE 1.1, as documented**: initial attack and extended attack models 0/1 (SDI, PCL lines), WU-E building
  spread (experimental), Monte Carlo BP with flame-length bins and pyrome tables, MPI. One global seed — ember
  draws diverge between arms, so pairing is statistical (null bands), not exact. Linux/Docker; AGPL v3 + Commons
  Clause. No Python wrapper exists; it is a namelist + GeoTIFF CLI, so the FlamMap/FSPro wrapper pattern applies.
- **Burn probability**: undecided. Uses are QWRA inputs and landscape-scale treatment evaluation; regional /
  national FSim may supply annual BP. The ensemble here is conditional exposure, not annual BP.
- **Audit fixes applied (2026-11-23)**: `ignition_matrix` reprojects to the grid CRS; `pad_bands` raises on short
  weather; `crown_gate` is required and set on every run; `run_batch` checks all arms sit on one grid;
  `aggregate_realizations` is one arm per call (`arm=`) and takes per-ignition `weights`. **Weights rule (Max)**:
  an ignition keeps its full `w_i`, split equally over the runs it has, so a failed run does not shrink its share;
  a weighted ignition with no run raises. `AvgTime` / `AvgFlameLength` are weighted means where burned;
  `n_burned` stays a count of runs.
- **Year-round weather record — planned phase (Max, 2026-11-23)**: the April–October limit is in the extracts, not
  the engine: GridMET ERC is Apr 1–Oct 31, 2008–2022 (214-day pivot); `RTMA_MONTHLY` holds months 04–10, 2011–2025.
  HRRR cubes are already any-date. So Marshall (30 Dec 2021) and late-October 2020 at long durations cannot be
  burned. To do, after the QA/QC items and before the ELMFIRE spike: (1) Max exports full-calendar GridMET ERC
  through 2025 and RTMA months 11–03; (2) a calendar-day episode path beside the 214-day one (FSPro inputs keep
  their season); (3) QC — winter RTMA gaps, NFDRS dead FM under snow/freezing, live FM out of season (cured
  grass); (4) HRRR wind rotation settled before a downslope-wind event is run. **Open**: what a severity
  percentile means over a full year. Until then `select_episodes` drops and counts a date whose window passes Oct 31.
- **Shadow review (2026-11-24, Max)** — scripts `code/dev/pyre/s1`–`s5`. Exact for a full-width treatment and for
  merging fires. Decided: (1) **bridge 3 cells** — the engine's arrival raster skips cells (a cell burns before all
  8 neighbours with fire 2 cells away, spotting on or off); such a cell rooted its own path at 0 and 6–66% of the
  burn in the example runs descended from them. A cell with no earlier neighbour now inherits from earlier cells
  on the nearest ring out to `bridge_cells=3`. (2) **Ember landings stay 0** (the engine does not record an
  ember's source); `ember_descent` / `ember_descent.tif` / `n_ember_starts`, `ember_descent_ha` report the burn
  the shadow cannot trace (0–5.5% in the six runs after bridging). Shadows written before this date are stale.
  **Open, not changed**: a treatment the fire flanks or that does not burn casts no shadow (the stronger the
  treatment, the smaller the shadow); the 8-neighbour rule blurs a narrow shadow sideways with distance (centre
  0.85 at 3 km, 0.70 at 6 km behind a 630 m block) and is direction-dependent; `lead_h` is clocked from the first
  treatment touched anywhere; why the engine's raster skips cells is not traced.
- **Shadow is a baseline-arm product (2026-11-24, Max)**: defined as where the *untreated* fire passed over the
  treatment footprint. On a treated arm it shrinks as the treatment works — synthetic flanked block
  (`s6_shadow_flanked_baseline.py`): the baseline shadow held 100% of the cells the block delayed, the
  treated-arm shadow 81% / 3% / 0% for a block burning at 50% / 20% / 5% speed and 0% for one that did not burn.
  `transmission_batch` and `shadow_probability` now take `arm=` and raise on mixed arms (`pyre_outputs.one_arm`,
  shared with `aggregate_realizations`); shadow rasters carry an `arm` tag. The baseline shadow is a superset of
  the delayed cells (the delay recovers with distance; the shadow does not), so it marks ground the treatment
  could act on, not ground it changed — the paired deltas say that.
- **Shadow validated against the paired change (2026-11-24)** — `shadow_agreement` (recall / precision / AUC of the
  baseline shadow ≥ 0.5 for cells whose arrival differs between arms; treated cells excluded),
  `code/dev/pyre/s7_shadow_agreement.py`. One 21-day fire, 300 m road corridor, front_range example:
  **shaded** (burned area ≈ baseline): every shadow cell changed (precision 1.00); at a 10 min tolerance recall
  0.88, precision 0.88, AUC 0.86. 91% of all burned cells differ by *something* (any nonzero difference), a third
  of them outside the shadow — small changes propagate everywhere, so "changed" needs a tolerance.
  **clearcut** (burned 45,890 ha vs 18,197 baseline): recall 0.25 — most changed cells burned only in the treated
  arm, which a baseline-arm shadow cannot see. The weight exponent (1 → 50, i.e. single steepest path) moved no
  score by more than 0.02, so the sideways blur is not what limits the shadow here; rule left at `power=1`.
  One fire only; the COSWAP run folders in `coswap_pyre/results.csv` were not on disk to repeat it there.
  **Standing reading**: paired deltas are the effect; the shadow is the stratifier and the siting product.
- **Clear-cut arm of the 01f fuel-break example burns 2.5× the baseline (2026-11-24)** —
  `code/dev/pyre/s8_clearcut_divergence.py`, one realization. Established: the prescription sets every corridor
  cell to GS1 (121) with zero canopy; inside the corridor it behaves as intended (intensity p90 4,489 → 103 kW/m,
  100% surface fire, 59% of cells reached > 1 h later). The arms track each other to day 10 (4,784 vs 4,931 ha),
  then split in a wind run at h 264–300: clear cut 32,260 ha by day 14 vs 16,577. The extra 27,692 ha is not
  ember-started (2.2%); 98% of it descends from fire that crossed the corridor, breaking out 4.3 km north of it
  at h 269 on ground the baseline burned 6% of. 17% of corridor cells burned > 1 h *earlier* in the clear cut.
  Reading (not proven): a timing difference at the corridor placed the clear-cut front where the wind event could
  run it. All arms touch the domain edge on day 13. **Open for Max**: whether GS1 / no canopy is the intended
  clear-cut fuel; one realization cannot rank prescriptions — this needs the ensemble.
- **Lead-time clock (2026-11-24, Max)**: `lead_h` was block arrival minus the fire's first touch of *any* treated
  cell, so with several treatments every block took the earliest one's clock (11 h where 2 h is right in the
  two-strip test). Now clocked from the **first arrival at the treatment unit the block's fire came through**
  (`treatment_entry_time`: units = `atom_ids`, else connected treated patches; several upstream units are
  weighted by their share), read at the block's first mostly-shadowed cell. `trt_arrival_h` in the fire summary is
  unchanged (first touch of any treatment). Block tables written before this date carry the old clock.
  This closes the shadow review: bridged gaps, ember share reported, baseline-arm product, validated against the
  paired change, lead clock. The sideways blur stays as measured, rule unchanged.
- **01f §5b rebuilt around seeds (2026-11-24, Max)**: one named fire (`CF_XY`, `CF_START` 2018-06-15), three arms,
  **20 spotting seeds at 14 days** — each seed a condition id, so every seed is an exact pair across arms. Baseline
  seeds run first and the corridor is cut on the union of their perimeters. Reports growth by seed, paired Δ final
  area, breakout rate (final area > 1.5× the arm's own median **and** > 1.5× the baseline median — the second is the
  one that compares arms), edge contact, per-arm share-of-seeds maps, `shadow_agreement`. Only the seed varies;
  ignition point, start hour and episode are the next experiment. **Storage**: summaries and ensemble rasters are
  written once under input-named paths and read back; `prune_realizations` (§8) keeps arrival time and flame length
  per run and drops the rest (~60%). Smoke-tested at 2 seeds × 2 days; the 20 × 14 run is Max's.
