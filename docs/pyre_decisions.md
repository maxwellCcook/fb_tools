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

## Open items

- **Realized spot ignitions are not exposed** by the engine's final matrices; `spot_ignitions` in the
  output is only the pending set at stop time. To attribute cells to embers, step the run in chunks
  (`max_duration`, carrying `spot_ignitions` and `random_generator`) and diff the pending set.
- ~~Domain adequacy on real episodes~~ — fails (2026-09-12 evening); waiting on the wider LCP.
- **Next session**: (a) pause/fork wrapper over `SpreadState.copy()` for spot attribution and mid-run
  treatment forks (note: the RNG is re-seeded per call, so chunked ≠ single-run bitwise, but stays
  deterministic); (b) `suppression.make_sdi_suppression_priority_fn` with `calculate_sdi` output — see the
  `sdi_static_layers` / `SDI_NODATA` commit on this branch (treatment-invariant SDI inputs cached
  once, SDI written on the slope band), which is the raster that hook would consume;
  (c) `fuel_spread_adjustment` from the COSWAP canopy-effects table; (d) per-pixel receptivity map
  once a spatial FM/temperature surface exists.
- Passive crown at p75/p90 — does it appear where the Cruz rate no longer dominates?
- `SharedMemory` for static cubes if worker RSS becomes binding at scale (not needed at 30 runs).
