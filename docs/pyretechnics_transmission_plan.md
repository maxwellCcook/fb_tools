> **Superseded.** This is the original plan of 2026-09-10, kept for reference. Where it disagrees with
> [pyre_decisions.md](pyre_decisions.md), the decision log is right. Current state: [STATUS.md](STATUS.md).

<!--
SUPERSEDED IN PART: docs/pyre_decisions.md (dated log) wins wherever the two disagree —
notably herbie is gone, the module layout is three modules, and wind is HRRR 3-D cubes.

Status: APPROVED 2026-09-10. Revised 2026-09-10 after scope review (v2).
Scope: replaces FSPro as the COSWAP transmission engine with pyretechnics.
Entry point for a fresh session: Stage 0. G0+G1+G2 are the <=4-day kill point.
Nothing in fb_tools/ has been modified yet.

v2 decisions (recorded so they are not re-litigated):
  - Mechanistic spotting is the primary motivation. Firebrand flux is the reported
    sweep axis; hourly HRRR is critical path, not optional.
  - v1 production runs at p97 only. The severity API is built; the sweep is v2.
  - Campaign allocated toward ignitions: 12 ignitions x ~30 episodes.
  - Episodes restricted to 2016+ so every episode carries real hourly wind.
  - Nothing serving FlamMap / FSPro / FSim / MTT / Cell2Fire is removed.
-->

# Replace FSPro with pyretechnics for the COSWAP transmission analysis

## Why

`code/03_Transmitted.ipynb` asks three questions — does treatment reduce burn probability along
pathways to values, does it delay fire arrival, and does it change ember/spot transmission — and FSPro
can only answer the first, badly.

| Problem | Evidence |
|---|---|
| **No ember physics.** One scalar `spot_probability` per ERC class (0.00–0.15), `spot_delay = 0`, no distance term. This is the question we most need answered. | `fspro_inputs/ign_000.input`; `gridmet.py:72-78` |
| **Arms are not paired.** `SPOTTING_SEED` reaches spotting only, not the ERC stream or wind draws: two same-seed runs differ on 66% of burned cells. Every Δ then needs a 1/√N band, and ΔBP = 0.01 needs **N ≈ 4,900 fires/run** — against a treatment that is 0.63 km² of a 2,700 km² landscape. Empirically the treated arm burns **+5.9%** vs untreated on the one completed triplet: wrong sign, consistent with pure noise. | `CLAUDE.md:110-121` (P0.1); cell 13; `runs/*/ign_000/*_DailyAcres.txt` |
| **Arrival/growth branch is dead.** `_Perimeters.shp` is malformed and final-state-only; `SKIP_Q2_PERIMETERS = True`. "Delay" is a ×24 rescale of a mean raster. | `CLAUDE.md:244`; cell 19; `spread/perimeters.py` |
| Windows-only `TestFSPro.exe` on a Parallels VM, Mac→Windows path patching, serial multi-hour runs. | `coswap_Timings.txt`; `container.py:1866` |
| Fire size set by an opaque "burn period minutes/day" column, no explicit diurnal cycle. Crown fire runs Finney while the in-situ FlamMap side deliberately chose Scott & Reinhardt. | `CLAUDE.md:81-83`; `COSWAP_FB_Approach_draft.md:326` |

`pyretechnics` (Spatial Informatics Group, EPL-2.0, 2026.8.10) is a native Python/Cython
reimplementation of the equations behind ELMFIRE, GridFire, FlamMap, FARSITE and FSim — not a wrapper.
In-process, no executable, runs on macOS. `eulerian_level_set.spread_fire_with_phi_field(...)` drives a
`SpreadState` carrying `phi, fire_type, spread_rate, spread_direction, fireline_intensity, flame_length,
time_of_arrival` as native 2D rasters.

**The reason we are doing this** is `spot_fire.py`: real firebrand physics, with
`SpotConfig{random_seed, firebrands_per_unit_heat, downwind_distance_mean,
fireline_intensity_exponent, wind_speed_exponent, downwind_variance_mean_ratio,
crosswind_distance_stdev, decay_distance}`, `expected_firebrand_production`, lognormal jumps scaled by
fireline intensity and 20-ft wind, Himoto crosswind σ_y, Albini firebrand height/t_max,
`firebrand_flight_survival_probability`, `schroeder_ignition_probability(temperature,
fine_fuel_moisture)`, and spot-ignited cells that cast their own firebrands. Every input is 0D/1D/2D/3D
via `SpaceTimeCube`, so weather is an explicit hourly stream rather than a parameterized burn period —
which is also what makes exact arm pairing possible.

It has **no Monte Carlo layer** — and that is the point. `fb_tools` already has the better one:
density-weighted stratified design fires with Horvitz–Thompson weights, runs manifests, paired ensemble
aggregation, area-integrated ΔTF, zonal summaries. That layer is engine-agnostic, which is why this port
is cheap.

**Decision: move transmission onto pyretechnics and retire FSPro for this analysis. Nothing is deleted.**
`models/fspro.py`, `fspro_validate.py`, the ERC/wind/FM builders, the HRRR rose, and the Cell2Fire stubs
all stay — they serve FlamMap, FSim, MTT and Cell2Fire, and the rose feeds FSPro `WindCellValues`. FSPro
simply stops being the transmission engine, and no further FSPro runs are made. Validation is therefore
built on what is already on disk — FlamMap rasters and observed fire sizes — not on new FSPro campaigns.

---

## Verified constraints (measured on this machine)

```
fb_tools env: python 3.12.13, numpy 2.4.3, scipy 1.17.1, rasterio 1.4.2
              Cython and sortedcontainers NOT installed
clang 21.0.0 arm64, CommandLineTools present
pyretechnics 2026.8.10: sdist only, NO wheels, not on conda-forge
  requires-python ~=3.11 (3.12 allowed)
  deps  numpy~=1.26.2  scipy~=1.12.0  rasterio~=1.4.3  sortedcontainers~=2.4.0  Cython~=3.0.11
  build-system.requires also pins numpy~=1.26.2
```

It compiles locally, and its numpy-1.26 build pin collides with the env's numpy 2.4.3. `CLAUDE.md:13`
forbids pip touching the geospatial stack, so Stage 0 is deliberate.

**LCP band audit** on `transmission/lcp/North_Cottonwood/coswap.tif` (1738×1730, EPSG:5070, 30 m, int32,
nodata −9999):

```
1 LF2020_Elev_CONUS   2218 … 4380   m                 -> as-is
2 LF2020_SlpD_CONUS      0 … 70     DEGREES           -> tan(radians(x))   [NOT x*0.01]
3 LF2020_Asp_CONUS      -1 … 359    deg CW from N     -> resolve the -1 flat sentinel
4 LF2016_FBFM40_CONUS   91 … 201    FBFM40 codes      -> as-is; NB codes + nodata -> 91
5 LF2016_CC_CONUS        0 … 85     percent           -> /100
6 LF2016_CH_CONUS        0 … 506    decimetres        -> /10  -> m
7 LF2016_CBH_CONUS       0 … 143    decimetres        -> /10  -> m
8 LF2016_CBD_CONUS       0 … 45     kg/m3 * 100       -> /100
9 LF2016_EVT_CONUS                  (not a pyretechnics input)
```

**The slope trap.** `pyretechnics.load_landfire` applies `slope * 0.01` because it expects LANDFIRE
**SLPP** (percent); our stack carries **SLPD** (degrees). A 20° slope becomes 0.20 rise/run = 11.3° — a
silent 43% under-slope on exactly the terrain that drives spread. The fix belongs in the de-stacker, not
upstream: `SLPD` is correct for FlamMap, FSPro and `sdi.py::_classify_slope_pen`, and a 10th `SLPP` band
would break `synthetic.py::_BAND_ORDER` and `adjust.py`'s band maps. The real defect is that units are
implied by the band name, so the de-stacker converts **once**, **requires the band description to say
`SlpD` or `SlpP`** and raises otherwise, never calls
`read_landfire_rasters_as_space_time_cubes` (so the scaling path is unreachable), and has every table
entry unit-tested — each one produces a plausible wrong answer.

---

## Plan

### Stage 0 — Install, pinned (~0.5 day)

Try the cheap thing first; it is a 20-minute experiment, not a design exercise.

1. **In the existing `fb_tools` env.** Add `cython` and `sortedcontainers` (conda-forge) to
   `environment.yml`, then `pip install "pyretechnics==2026.8.10" --no-deps --no-build-isolation`.
   `--no-build-isolation` is load-bearing: with isolation, pip honours the `build-system.requires`
   numpy-1.26 pin and compiles against numpy 1.x headers, which then will not import under numpy 2.4.
   `--no-deps` keeps the promise that pip installs one package and touches nothing else.
2. **Only if that fails:** a second env `environment-pyre.yml` named `fb_pyre` — `python=3.11`,
   `numpy=1.26.*`, `scipy=1.12.*`, `rasterio=1.4.*`, `sortedcontainers`, `cython`, the same
   `gdal>=3.6,<3.10` pins, then the same pip line plus `pip install -e . --no-deps`. Conda pins chosen to
   *satisfy* pyretechnics' declared constraints so `--no-deps` is not a lie. `fb_tools` is never
   downgraded; the transmission driver just runs in `fb_pyre`.

**No subprocess/IPC fallback.** Option 2 resolves any env conflict completely, so a `_pyre_worker.py` +
JSON-spec boundary would only be needed to mix two envs in one process — which we never need — and it
would compete with Stage 3's `multiprocessing.Pool` for the same job. One parallelism mechanism.

Pin the version: `PYRETECHNICS_VERSION = "2026.8.10"`, stamped into every manifest by
`pyre_version_info()`. Fifteen releases in 23 months with API additions (`spot_ignitions`, spot-from-spot
in 2025.4.8, crown physics retuned 2026.8.10) means an unpinned campaign is not reproducible. Document
the env situation in `CLAUDE.md`.

**G0:** `import pyretechnics.eulerian_level_set` succeeds in a conda env pip never modified, and a
200×200 synthetic fire runs to completion.

### Stage 1 — De-stacker, and the FlamMap cross-check (~2–3 days)

Rewrite `fb_tools/spread/convert.py` — today a 32-line stub documenting `lcp_to_cell2fire` — as the
engine-agnostic de-stacker it was always meant to be. **Keep the Cell2Fire stubs.**

```python
#: pyretechnics layer -> (canonical fb_tools band, multiplier, unit)
PYRE_LAYERS = {
    "elevation":           ("ELEV",   1.0,  "m"),
    "slope":               ("SLP",    None, "rise/run"),   # special: degrees -> tan()
    "aspect":              ("ASP",    1.0,  "deg"),
    "fuel_model":          ("FBFM40", 1,    "code"),
    "canopy_cover":        ("CC",     0.01, "fraction"),
    "canopy_height":       ("CH",     0.1,  "m"),
    "canopy_base_height":  ("CBH",    0.1,  "m"),
    "canopy_bulk_density": ("CBD",    0.01, "kg/m3"),
}

def destack_lcp(lcp_fp, out_dir, *, layers=None, resolution=None, resample="nearest",
                slope_units="auto", nb_fill=91, flat_aspect=0.0,
                overwrite=False, verbose=True):
    """De-stack a multi-band LANDFIRE LCP into per-variable rasters for pyretechnics.

    slope_units : "auto" resolves from the band description (SlpD -> degrees,
        SlpP -> percent) and raises if it matches neither. Degrees -> tan(radians(x)).
    nb_fill : FBFM40 code written where the source is nodata (91 = NB1). pyretechnics
        has no nodata concept, so an unfilled domain is undefined behaviour.
    resolution : resampling happens here, once per arm, so every arm lands on one
        congruent grid. Categorical layers always use nearest/mode.

    Raises ValueError on a missing band, unresolvable slope units, a converted layer
    outside its physical range, or an FBFM40 code absent from Scott & Burgan 40.
    """
```

Reuses `get_band_by_longname` (`fuelscape/lcp.py:21`), the canonical-name normalization in
`adjust.py:18`, and `_NB_CODES` (`lcp.py:117`).

**G1 — the validation that costs nothing and decides whether the adapter is right.** Run
`pyretechnics.burn_cells` on the North Cottonwood fuelscape under the same p97 conditions as
`01b_Run_FlamMap_Batch.ipynb`, and compare flame length and crown state against the FlamMap rasters
**already on disk** at `data/spatial/mod/flammap/outputs/{baseline_lcp,lcp_sc1b,lcp_sc2b}/p97/*.tif`
(13430×14783 @ 30 m, bands `CROWNSTATE, FLAMELENGTH, HEATAREA, DIR, DX`).

*Reconciliation window:* compare on the **LCP extent**, windowed-read from the FlamMap rasters via
`rasterio.windows.from_bounds` on the LCP bounds, with transforms asserted equal (both 30 m EPSG:5070)
rather than resampled. Two independent Rothermel implementations on identical fuels and weather should
agree closely; where they do not, the cause is ours (suspects in order: slope units, canopy scaling, fuel
adjustment factors, live FM). This also settles the crown-fire inconsistency at
`COSWAP_FB_Approach_draft.md:326` — pyretechnics' crown module is CBH/CBD-driven, consistent with the
in-situ Scott & Reinhardt rationale, unlike FSPro's Finney default.

**No-go if flame length does not reconcile.** The adapter is wrong and everything downstream inherits it.

### Stage 1b — Static firebrand production: the first spotting product (~1 day)

Pulled forward from Stage 7 because it needs **`burn_cells` only — no spread run, no weather cubes, no
CRN machinery** — so it lands right after G1 and unparks the Phase III spotting question ~3 weeks before
the rest of the campaign. It is also a spotting-side sanity check on the Stage 1 adapter.

```python
# fb_tools/spread/spotting.py  (created here, completed in Stage 7)
def delta_firebrand_production(ref_layers, trt_layers, zones_gdf, *, conditions,
                               id_col="TRT_ID", out_csv=None, verbose=True):
    """Paired Δ in expected firebrand production from treated pixels.

    Deterministic: expected_firebrand_production is a function of fireline intensity
    and the spot parameters, so this is a static raster difference with no Monte
    Carlo term and no pairing problem. The mechanistic counterpart of the in-situ
    FL/CS story — treatment lowers fireline intensity inside the unit, so the unit
    casts fewer and shorter-range embers.
    """
```

Deliverable: per-unit Δ firebrand production at p97 for all three arms, as a table and a raster.

### Stage 2 — Benchmark against the analysis budget (~0.5 day)

The question is not how pyretechnics compares to FSPro — it is **what campaign we can afford.** Measure
with `code/dev/pyre/b1_benchmark.py` (mirroring `code/dev/FSPro/p0_experiments.py`) on the real arms:

| cell size | grid | record |
|---|---|---|
| 90 m | 577×579 | wall s/realization, peak RSS, burned ha, spot-ignition count |
| 30 m | 1731×1737 | same |

each at `duration_days ∈ {7, 21}`, **spotting on as the default case** plus one spotting-off reference
(spot-from-spot cascades are a runtime risk and the primary deliverable depends on them), single process,
with a small sweep on `max_cells_per_timestep` and `buffer_width` (the two runtime/accuracy knobs).

**G2 — pick resolution from the measurement.** The v1 campaign is fixed by the scope decisions:

```
3 arms x 12 ignitions x 30 episodes x p97 x 3 flux levels = 3,240 realizations
```

- **≤ 30 s/realization @ 90 m** → production at **30 m**; ~9× cells ≈ 12 h on 8 workers. 30 m is the
  prize — it makes transmission commensurate with the 30 m in-situ FlamMap analysis.
- **30–120 s @ 90 m** → production at **90 m**, 30 m kept for a single-ignition illustration.
  ≈ 11–22 h on 8 workers. Expected case.
- **120–600 s @ 90 m** → hold the flux axis and the ignition count, cut episodes to ~15, and state the
  budget as the binding constraint.
- **> 600 s @ 90 m** → stop and re-scope: shorten duration, shrink the window (the measured burned
  footprint is far smaller than the 52 km buffer assumed), or switch to ELMFIRE (see Risks).

**Allocation rule, applied in that order: flux levels ≥ ignitions > episodes > severity levels.**
Firebrand flux is the reported sensitivity axis and the reason for the port, so it cannot be bootstrapped
after the fact. Ignitions are the limiting independent unit for inference — CRN collapses per-pixel Monte
Carlo noise but **12 design fires are still 12 independent units**, and CRN buys nothing for
generalization across ignition geometry. Episodes mostly tighten an interval CRN has already narrowed,
hence 30 and not 150. Severity is deferred to v2 by decision, with the API built in Stage 4 so the sweep
is a re-run rather than a rewrite.

Also call `check_domain_adequacy` (`spread/fspro_outputs.py:235` — engine-agnostic). It exists and was
never called while the burned footprint filled 72–92% of each axis.

**Output location matters.** `data/` is under Box CloudStorage, and a multi-thousand-realization campaign
writing tens of thousands of small rasters into Box will be slow and sync-hostile. Run outputs go to a
local scratch root; only aggregated ensemble products land in `data/spatial/mod/transmission/`.
`run_pyre_batch(out_root=...)` documents this.

### Stage 3 — Exact pairing and the runner (~3 days) — the methodological core

**A realization is `(ignition i, episode w, condition c)` with seed `s` derived from
`(base_seed, i, w, c)` only — never from the arm.** A *condition* is one point on the reported sweep
axes: `{severity, flux_level}` in v1, extensible without touching the contract.

```
for i in ignitions:                      # existing select_design_ignitions + HT weights
  ign_matrix_i = rasterize(footprint_i)  # built ONCE, shared by all arms
  for w in episodes:
    wx_cubes = build_weather_cubes(episode_w)   # built ONCE, arm-independent
    for c in conditions:
      s = seed_for(base_seed, i, w, c)          # arm-independent
      for arm in ("untreated", "background", "coswap"):
        static = static_cubes[arm]       # ONLY the 8 topo/fuel layers differ
        state  = SpreadState(cube_shape).set_resolution(res).set_start_time(t0)
        state.ignite_cells(lower_left_corner=ll_i, ignition_matrix=ign_matrix_i)
        out = spread_fire_with_phi_field({**static, **wx_cubes}, state, ...,
                                         spot_config=spot_cfg[c] | {"random_seed": s})
```

Rasterize the ignition **once** and reuse the array across arms — a per-arm rasterization risks a
one-cell difference that would break the pairing invisibly.

Two invariants, asserted in code:

1. `_assert_crn_contract(runs_df)` — every `(ConditionId, IgnitionId, EpisodeId)` group has exactly one
   distinct `Seed` and one `EpisodeFile`, and `len(group) == n_arms`. Raised in `prepare_pyre_experiment`,
   before any run.
2. `_assert_grid_congruence(arms)` — **reuse `container.py:196` unchanged.**

A third cube-hashing invariant was considered and dropped: `wx_cubes` is built outside the arm loop, so
hashing it inside asserts only that Python did not rebind a variable, and the sham-arm test already
catches arm-dependent weather end-to-end.

**G3a, as pytest not notebook cells:**

- **Repeatability** — same arm, same seed, twice → `np.array_equal` on `time_of_arrival` *including the
  NaN mask*, `fire_type`, `fireline_intensity`, and the spot-ignition set. Bitwise, not `allclose`.
- **Sham arm** — a byte-identical copy of `untreated` as a fourth arm. Δ must be **exactly 0 at every
  pixel, every realization** (`(delta != 0).sum() == 0`). This is the test FSPro failed at 66% of burned
  cells, and it is the whole claim.
- **Negative control** — a different seed must differ, so the above can actually fail.
- **Worker-count irrelevance** — `n_workers=1` and `n_workers=4` bitwise identical.

**The caveat that belongs in the methods section.** CRN gives common randomness *up to divergence*: once
the treated arm behaves differently it consumes RNG draws at different points and the streams
desynchronize. So pairing is exact in the null — verified by the sham-arm test — and exact up to the
first treatment-caused difference; beyond that the residual *is* the treatment effect. That is precisely
what the 1/√N band was paying for: **Δ = 0 wherever the treatment does nothing, by construction.**

**Module layout.** Name everything `pyre`, not `pyretechnics` — a module shadowing the library name
breaks `python -m` and reads badly beside `import pyretechnics`.

```
fb_tools/
  models/pyre.py          NEW  run_pyre_realization, run_pyre_batch, pyre_version_info
  models/container.py     MOD  prepare_pyre_experiment; extract _prepare_experiment_common()
  spread/convert.py       REWRITE  destack_lcp, PYRE_LAYERS (Cell2Fire stubs kept)
  spread/cubes.py         NEW  build_static_cubes, build_weather_cubes, cube_memory_report
  spread/episodes.py      NEW  build_weather_episodes, episode_hourly_frame
  spread/pyre_outputs.py  NEW  write_realization_rasters, aggregate_realization_bp,
                               postprocess_pyre_outputs
  spread/growth.py        NEW  summarize_growth, compare_growth_paired, time_to_reach_values
  spread/spotting.py      NEW  Stage 1b + Stage 7
  spread/bp.py            MOD  _find_bp_tif "pyre" entry (one line)
  spread/noise.py         MOD  paired_effect_interval + a docstring scope note
  spread/perimeters.py    DEPRECATE -> growth.py
  weather/rtma.py         MOD  build_hourly_fm_timeseries   (additive)
  weather/hrrr.py         MOD  fetch_hrrr_hourly_winds      (additive, herbie-gated)
  analysis/zonal.py       MOD  build_treatment_zones
```

`import pyretechnics` must be **deferred to call time** inside these modules, per the existing
`osmnx`/`herbie` convention — `fb_tools/__init__.py` imports eagerly from every subpackage.

**Refactor, do not fork.** `prepare_fspro_experiment` (`container.py:1424`) is 415 lines, of which ~175
is docstring; of the ~200 lines of code, roughly 100 are engine-agnostic — arm congruence, domain
validation, pyrome resolution, wind/cone, density surface, `select_design_ignitions`, footprint writing,
manifest. The weather block routes through `_load_weather_for_pyrome`, which is FSPro-specific (ERC
classes, current-ERC stream, wind cells) and stays behind. So `_prepare_experiment_common(...) -> dict`
is a ~100-line extraction and **half a day, not the bulk of this stage**. `tests/test_experiment_api.py`
is the safety net — run it before and after and require identical output. **Its own commit**, so a
regression is bisectable.

`prepare_pyre_experiment` mirrors the FSPro signature and returns the same shape plus `layers` (arm →
{layer: Path}), `episodes_df`, `conditions_df`, and a `runs_df` of one row per realization (`Scenario,
Arm, ConditionId, IgnitionId, EpisodeId, Seed, w_i, layers_dir, episode_csv, output_dir,
spot_config_json`).

`run_pyre_batch` mirrors `run_fspro_batch` — per-row `status`/`log_path`, `Arm` and `w_i` carried
through, printed failure summary, `skip_existing=True` for restartable campaigns, and `check=True`
meaning a run that errors **or produces no output** is recorded as an error, not success (the Phase 3a
fix that mattered most). `run_pyre_realization` raises if a realization returns no burned cells from an
ignition on burnable fuel — a silent no-op would otherwise average into the ensemble as a zero.

**Parallelism** — the first real Python parallelism in `fb_tools`. Per pyretechnics'
`README.multiprocessing.md`: native `multiprocessing.Pool` with `"spawn"`, chunking, float32 arrays, and
`SharedMemory` for the static cubes (~5× memory reduction); dask measured ~20% slower, Ray worse, threads
GIL-bound. `"spawn"` always — `fork` with GDAL/PROJ handles open is how you get non-deterministic
segfaults, and non-determinism is the one thing this design cannot tolerate. Default `n_workers=1` so the
serial-batch idiom still holds for debugging. Seeds derive in the parent, so worker count cannot affect
results.

### Stage 4 — Explicit weather cubes, hourly wind, severity API (~5 days)

**Hourly HRRR is critical path, not an optional 4b.** Spotting is driven by the wind tail, it is the
primary deliverable, and G7 cannot be met without it.

**Dimensionality is a memory decision, not a style one.** 21 days × 24 h = 504 steps. A full 3D float32
cube is 504×579×577×4 B = **673 MB per layer at 90 m**, **6.1 GB at 30 m**. So:

| layer | dim | source |
|---|---|---|
| elevation, slope, aspect | 2D | `destack_lcp` |
| fuel_model, CC, CH, CBH, CBD | 2D | `destack_lcp` — **the only layers that differ across arms** |
| wind_speed_10m, upwind_direction | 1D (504) | real hourly HRRR for the episode window |
| fm_dead_1hr / 10hr / 100hr | 1D (504) | hourly RTMA T/RH → `nfdrs.calc_lagged_fm` |
| fm_live_herbaceous / woody | 1D, daily step | `calc_live_fm_from_dead(6.5, 9.0)` — GSI is measured non-monotonic in 4 of 9 CO pyromes (`CLAUDE.md:302`) |
| temperature | 1D (504) | hourly RTMA `tmp_f` — **required for spotting** (Schroeder) |
| foliar_moisture | 0D | sampled per episode, 0.85–1.20 |
| fuel_spread_adjustment | 2D | `code/adjustments.py` / `canopy_effects.csv` |
| weather_spread_adjustment | 0D = 1.0 | do not tune it |

1D cubes are 504 floats — free. Per realization ~100 MB static + ~100 MB `SpreadState`, which is what
makes 8 spawn workers viable. Go 3D only for wind, only if G4 shows 1D wind is the binding fidelity
problem, and then coarsely (3-hourly = 168 steps ≈ 224 MB/layer at 90 m; infeasible at 30 m — another
argument for 90 m production).

**Hourly T/RH is already cached:** `data/weather/pyrome_emc/RTMA_EMC/rtma_hourly_CO_pyromes_{2011..2025}.csv`
(`pyrome_id, datetime_utc, tmp_f, rh_pct, emc_pct, vpd_pa, pcp_mm_hr`) — fifteen years per pyrome: the
diurnal cycle, the dead-FM driver, and the spotting temperature cube. **Hourly wind is not** —
`pyrome_wind/pyrome_45_wind.json` holds only the 6×8 HRRR fire-hour rose (UTC 19–22, 2016–2023, n=803,
CalmValue 1.868), so 24-hour wind needs new extraction. Both additions are **purely additive**; the
daily-peak-hour collapse and the rose builders are untouched because FlamMap, FSPro, FSim and MTT consume
them.

```python
# fb_tools/weather/rtma.py — new, beside collapse_to_peak_hour
def build_hourly_fm_timeseries(df_hourly, pyrome_id, start, end, *,
                               max_gap_hours=72, init_fm=None, verbose=True):
    """Hourly NFDRS78 dead FM (1/10/100 hr) for one pyrome over a date window.

    The daily-peak-hour collapse stays unchanged and still serves FlamMap/FSPro.
    pyretechnics consumes an hourly cube, so the lag integration stays at its native
    hourly step. Reuses nfdrs.calc_lagged_fm unchanged and restarts the lag across
    gaps — the same defence as build_fm_timeseries(max_gap_days=45).
    """

# fb_tools/weather/hrrr.py — new, beside fetch_hrrr_winds_at_fires (herbie-gated)
def fetch_hrrr_hourly_winds(point, start, end, *, hours=None, cache_dir=None,
                            k_neighbors=9, verbose=True):
    """All-hours HRRR 10 m winds at one point.

    fetch_hrrr_winds_at_fires filters to UTC 19-22 by design and keeps doing so for
    the rose that feeds FSPro WindCellValues; cube construction needs all 24. Same
    KD-tree, same 0-360 longitude normalization, same
    wd_deg = (degrees(arctan2(-u, -v)) + 360) % 360.
    """
```

**Episode sampling — a block bootstrap over observed weather**, in `fb_tools/spread/episodes.py`:

```python
def build_weather_episodes(pyrome_id, weather_dir, *, severity="p97", n_episodes=30,
                           duration_days=21, min_start_year=2016, analog_years=None,
                           fod_gdf=None, rtma_dir=None, allow_overlap=False,
                           seed=617327, verbose=True):
    """Select N real historical weather windows at a named severity level.

    severity : {"p75","p90","p97","large_fire_days"} or an explicit (low, high) ERC
        percentile tuple. The percentile options use the pyrome's GridMET ERC
        climatology and the same bands as build_erc_classes; "large_fire_days"
        instead selects days on which Class D-G fires were discovered and growing in
        this pyrome.
    min_start_year : earliest episode start, default 2016 — the first reliable HRRR
        year, so EVERY episode carries real hourly wind and real hourly T/RH from the
        same window. There is no rose-sampled fallback and no wind_source column: a
        2016-2025 pool of ten fire seasons is ample at p90/p97, and a synthetic-wind
        episode could not be published for spotting anyway.
    analog_years : restrict starts to named years; must be >= min_start_year.
    allow_overlap : when False, starts are separated by at least duration_days, so
        draws are near-independent.

    Returns one row per episode: episode_id, start_date, end_date, year, severity,
    erc_day1, erc_pctl_day1, erc_mean, n_large_fires_day1.

    Episodes are observed records, never scaled or offset, so each is nameable in a
    write-up ("the 21 days from 2020-08-13") and preserves the diurnal cycle,
    day-to-day autocorrelation, and frontal passages that a stochastic per-day ERC
    draw destroys.
    """

def episode_hourly_frame(episode_row, pyrome_id, weather_dir, *, foliar_moisture=0.9,
                         herb_scale=6.5, woody_scale=9.0, verbose=True):
    """Materialize one episode as a duration_days x 24 hourly driver frame:
    datetime, tmp_f, rh_pct, ws_mph_10m, wd_from_deg, fm_1hr, fm_10hr, fm_100hr,
    fm_herb, fm_woody, foliar_moisture — indexed by hour-from-ignition.
    """
```

**The severity dial is episode selection: which real historical days the weather stream comes from.**
Nothing is scaled or offset, so every scenario is nameable and defensible. The percentile labels
deliberately match the FlamMap side (`flammap/outputs/<lcp>/p97/`,
`class_percentiles=[0, 60, 80, 90, 97, 100]`), so in-situ and transmission results are comparable at the
same severity. `"large_fire_days"` conditions on days the landscape *actually* produced large fires, and
costs nothing because FPA-FOD is already loaded for ignition sampling.

**v1 runs at p97 only.** The API is built and tested now because building it later would mean re-running
the campaign, but the reported sweep axis in v1 is firebrand flux (Stage 7). G4.3 therefore validates the
dial on a cheap 3-episode smoke test rather than as a production axis, and the severity sweep is a v2
re-run with `severity=` varied and nothing else changed — the `ConditionId` key already has room for it.
Two rules keep it honest whenever it is run: the level is applied **identically to every arm**, so
pairing is untouched; and a severity-conditional headline is reported as a sweep, never at one level.

What improves regardless of the dial: the diurnal cycle *is* the hourly FM and wind series, so overnight
slowing comes from recovering 1-hr fuel moisture and falling wind rather than a hard clock, and daily
growth is a consequence of weather instead of a minutes-per-day table.

**G4 — weather fidelity, validated against observation, not against FSPro:**

1. Simulated final fire-size distribution, untreated arm, vs **observed FPA-FOD Class D–G sizes in
   pyrome 45** (`FPA_FOD_20260615.gpkg`, `CLAUDE.md:191-193`): medians within a factor of ~2, sane upper
   tail.
2. Diurnal growth curve (ha/hour by hour-of-day) shows the canonical afternoon peak and a near-zero
   overnight floor.
3. **The severity dial demonstrably moves fire size** — 3 episodes each at p75/p90/p97: median final
   size and ΣBP increase monotonically, and `"large_fire_days"` lands at or above p90. A dial that does
   not respond is a broken dial; this is the cheap test for it before v2 depends on it.
4. `check_domain_adequacy` passes on the aggregated p97 BP, with a pyretechnics daily-area frame standing
   in for `_DailyAcres.txt`.

### Stage 5 — Aggregation: reuse the inference layer (~2 days)

The key architectural move: **collapse per-realization output into exactly the three rasters the existing
pipeline already consumes**, so `bp.py`, `zonal.py` and notebook cells 15–21 work untouched.

```
per realization  ->  burned mask, time_of_arrival, flame_length, fireline_intensity,
                     fire_type, spot ignitions
        | aggregate_realization_bp over episodes, per (arm, ignition, condition)
BurnProb.tif        = mean over episodes of burned indicator      [0-1, defined everywhere]
AvgFlameLength.tif  = mean over episodes where burned             [NaN outside burn]
AvgTime.tif         = mean time_of_arrival, in DAYS               [NaN outside burn]
        |
aggregate_ignition_bp(...)     <- UNCHANGED, weights=w_i, ignitions=footprints, paired=True
summarize_bp_treatments(...)   <- UNCHANGED, dBP_sum_ha stays the headline
downwind_treatment_effect(...) <- UNCHANGED
```

Verified against the code: `aggregate_ignition_bp` already exposes `paired=`, `weights=`,
`ignition_ids=`, `ignitions=`, `strict_grid=`; `summarize_bp_treatments` already takes `id_col=`. **No
signature changes are needed.** `AvgTime` in **days** is deliberate — it keeps cell 15's `* 24.0` and the
deliberately-swapped `_agg(T["at"], R["at"])` sign convention valid verbatim.

Reused unchanged: `delta_burn_probability`, `aggregate_ignition_bp` (HT weights, `delta_std`, `n_burned`,
footprint masking, `DELTA_NODATA`, strict-grid guards), `summarize_bp_treatments` — keeping
**`dBP_sum_ha`, the area-integrated ΔTF_ij, as the headline BP metric**, never a domain-mean ΔBP
(`CLAUDE.md:118-120`; COSWAP is 0.07% of the landscape, so a domain mean is structurally
indistinguishable from zero regardless of engine) — `downwind_treatment_effect`, `analysis/zonal.py`,
`check_domain_adequacy`. The only `bp.py` edit is a `"pyre"` entry in `_find_bp_tif` (`bp.py:72`), for
intent rather than necessity.

**Move the notebook's hand-rolled geometry under test.** Cell 18 builds `in_treatment / around_0_500m /
downwind` with a local `_downwind_sector` helper and cell 21 samples points by hand, duplicating
`summarize_bp_treatments` and `downwind_treatment_effect`. Add
`zonal.build_treatment_zones(treatments_gdf, wind_from_deg, collar_m=500, downwind_km=5.0,
sector_deg=60.0) -> GeoDataFrame` with `zone` and `TRT_ID` columns, then drive
`summarize_bp_treatments(zones, delta_bp=..., id_col="zone")`. Kills ~60 lines of notebook code.

**What `noise.py` becomes.** `P01_REFERENCE` is an FSPro measurement of an FSPro pathology; applying its
1/√N band to a CRN-paired Δ would be wrong in the conservative direction, declaring real effects
unresolvable. Add a docstring paragraph to `bp_noise_floor` / `area_noise_floor` / `required_num_fires` /
`annotate_noise_floor` restricting them to FSPro, and add the replacement:

```python
def paired_effect_interval(deltas, *, weights=None, by=None, alpha=0.05,
                           n_boot=10_000, seed=0, statistic="mean", annotate=None):
    """Bootstrap CI for an exactly-paired treatment effect.

    With common random numbers the estimator is the mean of per-realization paired
    differences, so its uncertainty is the variance of the *effect* across
    realizations — measured here rather than extrapolated from a calibration on a
    different engine, landscape, and duration.

    by : cluster labels. Realizations within an ignition are not independent, so
        resample ignitions as the primary unit. At 12 design fires a 12-cluster
        bootstrap is usable but not generous: report per-ignition effects as primary
        and the pooled estimate as secondary.
    annotate : when given, attach {col}_ci_low / _ci_high / _resolvable to that
        DataFrame and return it — the reporting analogue of annotate_noise_floor, so
        a results table still cannot be read without its uncertainty.
    """
```

No `engine=` guard is threaded through the four FSPro functions: after Stage 8 nothing in the
transmission path calls them, so the parameter and its test would protect a route nobody takes. The
docstring scope note is the honest, cheap version.

### Stage 6 — Arrival time and growth as a real product (~2 days)

`time_of_arrival` is native per realization, which resurrects the dead branch. `perimeters.py`'s *logic* —
per-fire area at day marks, arrival day at treatments/values, moderate-vs-extreme regime split, paired
delta — is sound; its *data source* is not. **Write `spread/growth.py` fresh and deprecate
`perimeters.py`** rather than patching, so the FSPro-specific column sniffing and the `elapsed = 1.0`
fallback do not survive. This also **removes the planned `perimeters.py` rebuild on `read_daily_acres`
from the backlog** — a concrete saving the port buys immediately.

```python
def summarize_growth(toa, *, ignition=None, treatments_gdf=None, values_gdf=None,
                     days=(1, 2, 3, 5, 7, 14, 21), extreme_quantile=0.67, res_m=None):
    """Per-realization growth metrics: area_d{D}_ha, growth_d2_ha_per_day,
    max_spread_km, trt_arrival_h, trt_frac_burned, val_arrival_h, n_values_reached,
    regime. Day marks come straight off the arrival raster — area_d2 is
    count(toa <= 48 h) * cell_area — so they are exact, not inferred from a snapshot.
    """

def compare_growth_paired(run_pairs, *, treatments_gdf=None, values_gdf=None,
                          days=(2, 3, 5), weights=None, by=None, alpha=0.05,
                          n_boot=10_000, seed=0, out_dir=None, verbose=True):
    """Exactly-paired baseline-vs-treated growth comparison across realizations.

    Each pair's manifests are read and asserted to share
    (ConditionId, IgnitionId, EpisodeId, Seed). The FSPro-era docstring claimed fires
    paired 1:1 because both runs shared SPOTTING_SEED; P0.1 disproved it. Here the
    claim is checked. Positive = delayed / smaller treated, matching cell 15.
    """

def time_to_reach_values(toa, values_gdf, *, id_col=None, max_h=None):
    """Per-structure arrival hour (NaN = never reached this realization). Pooled
    across realizations this gives, per structure, an exceedance curve
    P(reached by hour h) — the product that replaces a single mean delay number.
    """
```

Iso-arrival perimeters stay a **plotting concern, not a library function**: `rasterio.features.shapes` on
`toa <= level` is ~5 notebook lines and `summarize_growth` already gives exact day-mark areas off the
raster, so a tested `arrival_contours` with `simplify_m` / `min_area_ha` / `out_path` would earn nothing.
The maps still get drawn — that is the figure the notebook currently cannot produce.

Products: per-structure time-to-arrival exceedance curves with exactly-paired CIs ("delays arrival at
these 312 structures by a median of 6.4 h, 95% CI 3.1–9.8, and reduces P(reached within 48 h) from 0.41
to 0.29"); Δ arrival-hour distributions at the treatment and at values; growth rate toward values by
regime; real iso-arrival perimeter maps. Report Δ arrival with its co-burn support count — it is only
defined where both arms burn.

### Stage 7 — Mechanistic spotting: the primary deliverable (~4 days)

This is why the port is happening. Treat the eight `SpotConfig` fields as three kinds of parameter:

- **Physics shape** — `fireline_intensity_exponent`, `wind_speed_exponent`,
  `downwind_variance_mean_ratio`, `crosswind_distance_stdev`: leave at published defaults. We have no
  data to re-derive the lognormal jump form and should not pretend to.
- **Magnitude** — `downwind_distance_mean`, `decay_distance`: **checked against**, not fitted to, FlamMap
  `DX`, on disk for all three arms at 30 m (band 5, metres). The parked in-situ analysis measured
  per-unit **p95 DX 965–1,390 ft (294–424 m)**, per-unit max 1,008–1,809 ft, and a treatment effect of
  −21 to −130 ft on p95. *Why a check and not a calibration:* DX is a deterministic **potential maximum**
  per pixel with no survival or ignition-probability term, so it is an upper envelope on a different
  quantity from a realized jump distance. Fitting magnitude parameters to it would dress a weak
  constraint as a strong one. So compare jump p95 to DX p95 as a diagnostic, keep published defaults, and
  let the flux axis carry the uncertainty honestly.
- **Flux** — `firebrands_per_unit_heat`: the least constrained and the one that most determines whether
  spotting dominates. It is the **reported sweep axis** of the v1 campaign: low / central / high as three
  `ConditionId` levels (plus an off reference), with the treatment conclusion reported across all of them.

```python
def build_spot_config(random_seed, *, preset="central", **overrides):
    """Validated SpotConfig for spread_fire_with_phi_field.

    random_seed must be the realization seed, identical across arms — spotting is the
    largest stochastic term in the model, so an arm-dependent spot seed destroys the
    pairing Stage 3 exists to establish. preset="off" returns None so callers need no
    branch. Presets: off, low_flux, central, high_flux.
    """

def compare_jump_distance_to_dx(dx_raster, spot_config, *, zones_gdf=None,
                                percentile=95, wind_speed_mph=None,
                                fireline_intensity=None, out_csv=None):
    """Diagnostic: simulated jump-distance p95 vs FlamMap per-pixel max spot distance.
    DX bounds rather than equals the jump distribution. Reported, not fitted.
    """

def summarize_spot_ignitions(realization_dirs, *, treatments_gdf=None, values_gdf=None,
                             wind_from_deg=None,
                             distance_bins_m=(0, 100, 250, 500, 1000, 2000),
                             weights=None, out_dir=None, verbose=True):
    """Per-realization table (n_spots, n_spots_downwind_trt, spot_dist_p50/p95/max_m,
    frac_area_from_spots), a spot-ignition density raster per 100 ha, and a binned
    distance histogram of *successful* ignitions.
    """

def first_arrival_attribution(run_dir, values_gdf, *, link_radius_m=None):
    """Classify each value's first arrival as flaming-front or spot ignition."""

def breaching_diagnostics(run_pairs, treatments_gdf, *, weights=None, out_dir=None):
    """Realizations where the fire crosses the treatment *only* via spotting: count,
    rate, and a map of crossing locations — the honest answer to "do fuel treatments
    work under extreme conditions", reportable per flux level.
    """
```

Transmission metrics, none possible in FSPro:

1. **Spot-ignition density downwind of the treatment**, per arm — does the treatment reduce ember seeding
   of the downwind landscape?
2. **Distance distribution of successful spot ignitions**, and Δ in its p95 — the mechanism the in-situ
   DX analysis could only describe as potential.
3. **Fraction of structures first reached by an ember rather than the front**. If that fraction is high
   it reframes the whole effectiveness argument: a treatment that stops a front but not embers is doing
   much less than ΔBP suggests. With a scalar `spot_probability ∈ [0, 0.15]`, `spot_delay = 0` and no
   distance term, FSPro cannot address this even in principle.
4. **Δ firebrand production per treatment unit**, paired — delivered early at Stage 1b.
5. **Breaching rate**, per flux level.

**G7:** jump p95 within ~±25% of per-unit DX p95 under comparable wind and intensity, shifting in the
same direction between arms; spotting off plus low/central/high flux reported for area burned, structures
reached, `dBP_sum_ha`, and breaching rate. **If the treatment conclusion flips between low and high flux,
that is the result** and must be published as such, not tuned away. Depends on Stage 4's hourly HRRR,
which is why that is critical path.

### Stage 8 — Notebook, memo, retirement (~3–4 days)

- `code/03b_Transmitted_Pyre.ipynb` — **new notebook; do not edit `03_Transmitted.ipynb` in place.**
  Cells 0–12 (paths, SFA/project selection, wind, values-in-cone, design fires) carry over nearly
  unchanged; cells 13–21 swap FSPro calls for the `pyre` equivalents, which were designed to keep the
  same shape.
- `docs/transmission_engine_comparison.md` — the methods memo: the failure-mode table above, what each
  engine can and cannot answer, the CRN argument and its up-to-divergence caveat, G1's FlamMap
  reconciliation, G4's fire-size validation, G2's cost numbers, and the risk list. It slots into
  `COSWAP_FB_Approach_draft.md:321-324`, whose TODO placeholder is waiting for exactly this.
- **Retire FSPro for transmission now, not after a parallel campaign.** Mark the FSPro transmission path
  in `CLAUDE.md` as superseded, freeze FSPro feature development (no `perimeters.py` rebuild), archive
  `03_Transmitted.ipynb` in `code/archive/`. Nothing is deleted, per the Why section.
- **State the one capability genuinely dropped:** FSPro modelled containment/suppression probability
  (`_ContainSummary.txt`); we will not. `suppression/sdi.py` covers difficulty, not containment
  probability. For "what does this treatment buy", omitting suppression biases the answer in an unknown
  direction — plausibly *against* the treatment, since treatments likely improve suppression
  effectiveness. Declare it as a known gap.
- Production campaign: **3 arms × 12 ignitions × 30 episodes × p97 × 3 flux levels = 3,240
  realizations** at the G2 resolution, parallel, with `skip_existing` restartability.
- **Headline product: treatment effect as a function of firebrand flux.** `dBP_sum_ha`, arrival delay,
  structures reached, spot-ignition density, first-arrival attribution, and breaching rate each reported
  across off / low / central / high with paired CIs. The severity sweep is the v2 re-run on the same
  machinery.

### Testing (~1–2 days, alongside)

In `tests/`, following the existing style — data-free, built on `synth_lcp` and `synth_fod`
(`tests/conftest.py:63,113`), guarded with `pytest.importorskip("pyretechnics")` so the suite still
passes wherever the engine is absent. New fixtures: `synth_pyre_layers` (de-stacked), `synth_episode`
(deterministic 21 d × 24 h frame), `synth_arm_pair` (a byte-identical copy plus a genuinely treated copy).

- **`test_pyre_destack.py`** (no engine) — the conversion table cell by cell: **`SLPD = 20° →
  tan(20°) = 0.3640`, not 0.20** (explicit regression test for the silent-underslope bug); `CC 45 →
  0.45`; `CH 150 → 15.0 m`; `CBH 15 → 1.5 m`; `CBD 10 → 0.10`; elevation unchanged; NB codes survive;
  `-9999 → 91`; no NaN in `fuel_model`; grid congruent with source; the `resolution=90` path invents no
  fuel codes; raises on a missing band, an unresolvable slope unit, a fabricated FBFM code, and an
  out-of-range conversion.
- **`test_pyre_cubes.py`** — 1D length == `duration_days × 24`; 0D broadcast; `cube_memory_report`
  matches hand arithmetic; an oversized 3D request raises with the measured MB in the message; FM out of
  range raises; `upwind_direction` round-trips the 292.5°-FROM convention.
- **`test_pyre_pairing.py`** — the keystone: bitwise identity for same arm + same seed; **sham-arm Δ
  exactly zero at every pixel**; different seed differs; the treated copy differs and the first
  divergence is at or downwind of the modified patch; `_assert_crn_contract` rejects an arm-dependent
  seed; `n_workers=1` vs `4` bitwise identical.
- **`test_pyre_spotting.py`** — high-wind high-intensity cube yields ≥1 spot ignition, `preset="off"`
  yields 0; ignitions lie downwind within the crosswind envelope; deterministic for a fixed seed;
  `build_spot_config` raises on a negative seed, non-positive `decay_distance`, unknown preset;
  `delta_firebrand_production` is exactly zero between two identical layer sets; `breaching_diagnostics`
  counts a hand-built spot-only crossing.
- **`test_pyre_growth.py`** (no engine) — radial-ramp arrival raster: `area_d2_ha == count(toa<=48h) ×
  cell_area` exactly; `time_to_reach_values` matches hand computation and NaNs the unreached; sign
  convention positive = delayed; `compare_growth_paired` raises when a pair's manifests disagree on
  `Seed`/`EpisodeId`/`ConditionId`.
- **`test_pyre_ensemble.py`** (no engine) — 10 fabricated burn masks give BP exactly `k/10`; output is
  found by `_find_bp_tif(model="pyre")` and consumed by `aggregate_ignition_bp` unchanged; `AvgTime` is in
  days; raises when episode- and ignition-level weights are both supplied.
- **`test_pyre_episodes.py`** (no engine) — `min_start_year=2016` enforced and an earlier `analog_years`
  raises; `allow_overlap=False` separates starts by `duration_days`; the frame is `duration_days × 24`
  rows with no NaN in any driver column.
- **`test_paired_interval.py`** — CI coverage on known deltas; `alpha` monotone; zero-variance deltas give
  a degenerate CI without dividing by zero; clustering `by=ignition` widens the interval; `annotate=`
  round-trips the columns.

Plus `tests/test_experiment_api.py` must stay green across the `_prepare_experiment_common` extraction —
that is the refactor's gate.

---

## Effort and gates

Gate criteria are stated once, in their stage. This table is the scan surface and the verification
checklist both.

| Stage | Deliverable | Gate | Days |
|---|---|---|---|
| **0** Install | pinned 2026.8.10, env documented | **G0** | 0.5 |
| **1** De-stacker | `convert.py` rewritten + unit tests | **G1** (no-go if FL does not reconcile) | 2–3 |
| **1b** Static firebrands | `delta_firebrand_production`, per-unit Δ table + raster | — early spotting deliverable | 1 |
| **2** Benchmark | `code/dev/pyre/b1_benchmark.py`, s/realization and RSS | **G2** resolution chosen from measurement | 0.5 |
| **3** CRN engine | `models/pyre.py`, `prepare_pyre_experiment`, common-helper extraction | **G3a** + `test_experiment_api.py` green | 3 |
| **4** Cubes + hourly HRRR + severity API | `cubes.py`, `episodes.py`, hourly FM and winds | **G4** | 5 |
| **5** Aggregation | `pyre_outputs.py`, `bp.py` one-liner, `build_treatment_zones`, `paired_effect_interval` | **G5** headline `dBP_sum_ha` CI excludes 0, or the required budget is quantified from observed paired SD | 2 |
| **6** Arrival/growth | `growth.py`, `perimeters.py` deprecated, exceedance curves | — | 2 |
| **7** Spotting | `spotting.py` completed, DX diagnostic, flux sweep, breaching | **G7** | 4 |
| **8** Notebook, memo, campaign | `03b_Transmitted_Pyre.ipynb`, methods memo, 3,240-realization run, effect-vs-flux curves | — | 3–4 |
| Tests | alongside | `pytest` — ~335 existing + 7 new files | 1–2 |
| **Total** | | | **~24–28** |

**G0 + G1 + G2 cost ≤4 days and either justify or kill the effort.** Stage 1b gives a real spotting
deliverable inside the first week. Do not start Stage 4 before G2 — the cube design depends on the
resolution decision. Two gates are re-checked on real data rather than only on fixtures: **G3a's sham-arm
Δ = 0 on the full North Cottonwood window**, and **G2's wall clock and peak RSS recorded in the
manifest** alongside the chosen resolution.

---

## Risks and honest limitations

1. **Beta, fast-moving, no agency pedigree.** Fifteen releases in 23 months; `SpreadState` refactor in
   2025.3.27, crown physics retuned 2026.8.10. Our results will be "a physics-based reimplementation,
   configured by us", not "the USFS product". Mitigations: pin and stamp the version; lean on G1 (an
   independent Rothermel implementation on identical inputs) and G4 (observed fire sizes) as the
   credibility instruments. The counter-argument is real, though: the reason FSPro's pairing flaw went
   unnoticed for years is that nobody could see inside it.
2. **Spotting parameters are weakly constrained** — `firebrands_per_unit_heat` especially, and DX bounds
   rather than measures jump distance. This is the primary deliverable, so the uncertainty is reported as
   the flux sweep rather than absorbed into a fitted constant. A conclusion that flips across flux is a
   publishable finding, not a failure.
3. **We own the Monte Carlo design, the validation, and the blame.** No built-in ensemble orchestration,
   weather sampler, or BP aggregator. The block-bootstrap-of-real-episodes approach is more physically
   faithful than a stochastic per-day ERC draw but has no fire-transmission precedent to cite — defend it
   on its merits and show the size validation.
4. **Packaging fragility** (measured, not speculated): sdist-only, no conda-forge package, a numpy-1.26
   build pin against an env on numpy 2.4.3. Expect this to be the likeliest Stage 0 failure and an
   ongoing maintenance tax.
5. **Containment/suppression probability is dropped** (Stage 8). Declare it.
6. **Unit-convention traps** — slope degrees-vs-percent worst because it is silent, plus FBFM nodata,
   canopy scaling, wind-direction convention, arrival-time units. Each has an explicit test because each
   yields a plausible wrong answer.
7. **CRN pairs only up to divergence, and 12 design fires are still 12 independent units** (Stages 2, 3).
   Cluster the bootstrap on ignition; report per-ignition effects as primary.
8. **The 2016+ episode restriction** shrinks the pool to ten fire seasons and excludes 2012 — a year
   worth naming in a Colorado write-up. Accepted deliberately: a pre-2016 episode would carry synthetic
   wind, and no spotting result from synthetic wind is publishable. If 2012 becomes a required scenario it
   is a separate non-spotting run with its provenance stated, not a silent mode inside the episode builder.
9. **New parallelism in a repo with none** — spawn+GDAL interactions, shared-memory lifetimes,
   partial-campaign recovery. Determinism is non-negotiable: seeds derive in the parent, and a test
   asserts worker count cannot change results.
10. **The engine may simply be too slow at 30 m.** Unknown until G2; the plan branches explicitly.

**When ELMFIRE (`lautenberger/elmfire`) would be the better choice** — the Fortran/MPI operational model
with its own Monte Carlo driver, ember transport, and building-to-building spread: when scale dominates
(≥10⁴–10⁵ realizations or a statewide 30 m domain, i.e. if G2 lands in the ">600 s" branch); when the
deliverable must read "ELMFIRE-based" to an agency or insurer; or when structure-to-structure spread
becomes the question (plausible here — 147 NSI structures sit within 1 km of North Cottonwood).

Prefer **pyretechnics**, as here, for two properties ELMFIRE cannot offer: **inspectable, configurable
ember physics** in the language the rest of the analysis is written in, and **exact pairing of
counterfactual arms** via in-process RNG control. That second property is the difference between "COSWAP
burned 5.9% *more*, consistent with pure Monte Carlo noise" and an interpretable result. ELMFIRE's MPI
driver with its own seeding would reproduce the same pairing problem in a faster language, and it costs a
Linux/Docker/MPI toolchain and a second platform split, having just escaped the Windows VM. Hold it as
the named fallback if G2 fails on cost.

## End-to-end acceptance

`code/03b_Transmitted_Pyre.ipynb` produces, for the three contrasts across 12 ignitions and 4 flux
levels at p97: ΔBP / ΔFL / Δarrival surfaces; `dBP_sum_ha` per zone with paired bootstrap CIs;
iso-arrival perimeter maps; per-structure arrival exceedance curves; spot-ignition density and distance
distributions; first-arrival attribution; breaching rates; Δ firebrand production per unit; and the
treatment-effect-vs-flux curves.
