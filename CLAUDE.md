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

## In progress: pyretechnics replaces FSPro for transmission
`docs/pyretechnics_transmission_plan.md` — approved 2026-09-10, scope-revised to **v2** the same day.
**Stages 0, 1 and 2 complete; G2 decided (30 m). Stage 1b / 3 next.** Moves the COSWAP transmission analysis off FSPro onto `pyretechnics` (native
Python/Cython, EPL-2.0, pinned 2026.8.10) for three reasons FSPro cannot fix: **mechanistic
ember/spotting physics** replaces a scalar `spot_probability` (the primary motivation), arms become
**exactly paired** (in-process RNG + explicit weather cubes, vs P0.1's finding that `SPOTTING_SEED`
buys no variance reduction), and `time_of_arrival` is a native raster so the dead
`spread/perimeters.py` branch becomes a real product. The `fb_tools` design-fire / aggregation /
zonal layers are reused unchanged — that is the point. New code goes in `models/pyre.py` +
`spread/{cubes,episodes,pyre_outputs,growth,spotting}.py`; `spread/convert.py` becomes the LCP
de-stacker. **Nothing is deleted**: FSPro's wrapper, the ERC/wind/FM builders, the HRRR fire-hour
rose and the Cell2Fire stubs all stay, because FlamMap/FSim/MTT/Cell2Fire consume them.

**Gotcha the plan exists to prevent**: our LCP carries LANDFIRE `SLPD` (degrees) while
`pyretechnics.load_landfire` expects `SLPP` (percent), so that loader must not be used — it silently
flattens a 20° slope to 11.3°. `destack_lcp` reads the band description and raises unless it says
`SlpD` or `SlpP`.

v2 scope decisions (in the plan's header comment; do not re-litigate):
- Firebrand flux (`firebrands_per_unit_heat` low/central/high) is the **reported sweep axis**;
  hourly HRRR is critical path because spotting rides the wind tail. Severity (p75/p90/p97) ships as
  an API but v1 production runs **p97 only**; the sweep is a v2 re-run on the same machinery.
- Campaign: 3 arms × **12 ignitions × 30 episodes** × p97 × 3 flux = 3,240 realizations. Allocation
  rule: flux ≥ ignitions > episodes > severity. CRN collapses per-pixel noise but 12 design fires are
  still 12 independent units, so ignitions beat episodes.
- Episodes restricted to **2016+** (first reliable HRRR year) so every episode carries real hourly
  wind — no synthetic-wind fallback, no `wind_source` column. Excludes 2012; accepted.
- Install is two tiers (`fb_tools` env, else a `fb_pyre` env), **no subprocess/IPC fallback** — it
  would duplicate the `multiprocessing.Pool` path for no gain.
- `delta_firebrand_production` is pulled forward to Stage 1b: `burn_cells` only, no spread run, so a
  real spotting deliverable lands in week 1.

### Stage 0 — DONE. pyretechnics runs in a second env, `fb_pyre`, not in `fb_tools`
**Option 1 of the plan failed, as Risk 4 predicted.** There is no (Cython, numpy) pair that builds
pyretechnics 2026.8.10 inside `fb_tools` (python 3.12, numpy 2.4.3) — measured, both directions:
- **Cython 3.1.8 / 3.2 / 3.3 → cythonization fails**: `Function signature does not match previous
  declaration` at `burn_cells.py:39,257,357` (stricter `@cy.ccall` vs `.pxd` matching).
- **Cython 3.0.12 + numpy 2 headers → C compile fails**: `no member named 'subarray' in
  'struct _PyArray_Descr'` — Cython 3.0's bundled `numpy.pxd` `PyDataType_SHAPE` touches a field
  numpy 2 moved to `_PyArray_LegacyDescr`.

So `environment-pyre.yml` (Option 2) is the install: **`fb_pyre`** = python 3.11, numpy 1.26.4,
scipy 1.12.0, rasterio 1.4.2, Cython 3.0.12, sortedcontainers 2.4. `fb_tools` is untouched and is
**never downgraded**; it keeps serving FlamMap/FSPro/FSim/MTT and every notebook. Only the
transmission driver runs in `fb_pyre`.
```
conda env create -f environment-pyre.yml && conda activate fb_pyre
pip install -e . --no-deps
pip install "pyretechnics==2026.8.10" --no-deps --no-build-isolation
```
`--no-build-isolation` is load-bearing: with isolation pip builds against a throwaway numpy-1.26 env
and we lose control of which headers the extension compiles against.

`models/pyre.py` holds `PYRETECHNICS_VERSION`, `require_pyretechnics()` and `pyre_version_info()`
(stamped into every campaign manifest: engine, python, **numpy and scipy** — the engine is compiled
against the numpy C-API present at build time, so the same pinned source yields different binaries in
different envs). `import fb_tools` still succeeds where the engine is absent; the import is deferred.

**G0 PASS** — `code/dev/pyre/g0_smoke.py`, 200×200 @ 30 m, 5 h:
- runs to completion, 7,544 cells with spotting off;
- **bitwise reproducible** at a fixed seed (`time_of_arrival` incl. the NaN mask, and `fire_type`) —
  the property the entire Stage 3 CRN design rests on, and the one FSPro does not have;
- **spotting is mechanically active**: a 180 m non-burnable barrier stops the head fire dead at the
  barrier edge with spotting off (`max_x = 99`, barrier at 100), and is crossed with spotting on;
- a different seed changes the run, so the determinism test can actually fail.

**Engine facts measured at Stage 0 that contradict the plan's draft tables** — each would produce a
plausible wrong answer, so they are listed here and each gets a unit test:
- `wind_speed_10m` is **km/hr** (not m/s, not mph). HRRR gives m/s.
- `temperature` is **degrees Celsius** (not °F). The RTMA cache is `tmp_f`.
- `upwind_direction` is degrees clockwise from North, meteorological **FROM** — matches
  `hrrr.py`'s `wd_deg` convention directly.
- `time_of_arrival` is in **minutes**; `AvgTime.tif` must be written in **days** (Stage 5) to keep
  `03_Transmitted.ipynb` cell 15's `* 24.0` valid.
- `SpaceTimeCube` warns and **copies** unless the array is float32 — build cubes as float32.
- `spot_fire.expected_firebrand_production` is a **`@cy.cfunc`, not callable from Python** in the
  compiled build. Stage 1b cannot call it; it must be reimplemented vectorized in
  `spread/spotting.py` and cross-checked against the sdist's pure-Python-mode source (pyretechnics is
  pure-Python-mode Cython, so `spot_fire.py` runs as plain Python with the `cython` shadow module).
  Only `firebrand_flight_survival_probability` and `schroeder_ignition_probability` are exported.
- `spread_fire_with_phi_field` returns `random_generator` only when spotting is used — a cheap
  assertion that the spotting path was actually exercised.

### Stage 1 — DONE. `spread/convert.py` is the de-stacker; G1 demoted to a diagnostic
`destack_lcp(lcp_fp, out_dir, ...)` + `PYRE_LAYERS` write the eight pyretechnics inputs in the units
the engine documents. Cell2Fire stubs kept. Exported from `fb_tools` and `fb_tools.spread`.
Outputs are float32 (`fuel_model` int16), so `SpaceTimeCube` does not copy; units, source band and
resolved slope units are written to each file's tags.

**`tests/test_pyre_destack.py` — 38 tests, no engine import**, so they run in the `fb_tools` env.
They pin every conversion exactly: `SLPD 20 deg -> tan(20) = 0.3640` (**not** 0.20), `CC 45 -> 0.45`,
`CH 150 -> 15.0 m`, `CBH 15 -> 1.5 m`, `CBD 10 -> 0.10`, aspect `-1 -> flat_aspect`, nodata `-> 91`,
plus raises on a missing band, an unresolvable slope unit, a fabricated FBFM40 code (94-97 are
unassigned), an out-of-range conversion, and bilinear resampling inventing fuel codes. Two arms
de-stacked separately at the same `resolution` are asserted byte-congruent — resampling is done
**once per arm here**, never per realization.

**G1 was deliberately demoted from the plan's no-go gate to a one-shot diagnostic** (decision
2026-09-10): the unit tests above are a *stronger* check for the silent-units class of bug than a
raster comparison, because they are exact rather than "within 20%". What the FlamMap run still buys
is separating adapter error from weather-sampler error at G4. `code/dev/pyre/g1_flammap_diagnostic.py`,
North Cottonwood, p97 pyrome 45, 3.0 M cells, 1.4 s per arm:

| arm | FL median pyre / FlamMap | ratio | r | crown frac pyre / FlamMap | agree |
|---|---|---|---|---|---|
| untreated  | 1.80 / 3.75 m | 0.71 | 0.913 | 21.4% / 51.3% | 70.0% |
| background | 1.79 / 3.49 m | 0.71 | 0.914 | 21.1% / 50.7% | 70.4% |
| coswap     | 1.79 / 3.42 m | 0.71 | 0.914 | 21.1% / 50.6% | 70.5% |

Flame length correlates at r = 0.91 and runs ~29% below FlamMap. **Crown fire is the real divergence**
— pyretechnics calls 21% of burnable pixels crown where FlamMap (ScottReinhardt) calls 51%. Recorded,
not chased, per the timebox. **Revisit it if Stage 4's simulated fire sizes come in low against
FPA-FOD**: less crown fire means less fireline intensity, which means shorter ember jumps, and
spotting is the primary deliverable.

Two traps hit while building it, both worth knowing:
- **FlamMap `CROWNSTATE` is coded 0/1/2/3 — identical to pyretechnics `fire_type`** (measured:
  median FL 0.00 / 0.85 / 5.78 / 27.24 m). Assuming FlamMap's 0/1/2 produced a fake "78% vs 22%"
  crown disagreement out of nothing but a lookup table.
- **Load the p97 scenario from `data/weather/flammap_rtma/pyrome_<id>_flammap.json`, never transcribe
  it.** Hardcoding it once silently picked up pyrome 128 (`FM_herb` 30) instead of 45 (59.3).

**A finding about the in-situ Phase 1 FlamMap results, not about this port**: the scenario cache fills
`WIND_SPEED` from HRRR's **10 m** `ws_mph` and FlamMap reads that field as a **20-ft** wind. The true
20-ft equivalent of a 23.1 mph 10 m observation is 20.1 mph, so the published FlamMap runs are ~15%
hot on wind. G1 reproduces what FlamMap actually ran (`wind_speed_10m = mph x 1.15 x 1.609` km/hr, so
pyretechnics' internal `/1.15` lands back on 23.1 mph at 20 ft) rather than correcting it.

Scratch root is **`~/fb_scratch/coswap_pyre`** (`$FB_PYRE_SCRATCH`), outside Box: only aggregated
ensemble products belong in `data/spatial/mod/transmission/`.

### The crown-fire divergence is a hard-coded `canopy_cover > 0.4` gate — OPEN DECISION
`code/dev/pyre/g1b_crown_divergence.py`. G1's 21%-vs-51% crown gap is **not an adapter bug**.
`crown_fire.van_wagner_crown_fire_initiation` requires `canopy_cover > 0.4` — a term Van Wagner (1977)
does not have and FlamMap's ScottReinhardt path does not apply:
```
surface_fireline_intensity > 0.0  and  canopy_cover > 0.4  and  I >= I0
```
| | cells | pyre crown | FlamMap crown |
|---|---|---|---|
| **above** CC 0.4 | 405,011 (28.7%) | 74.6% | 75.2% |
| **below** CC 0.4 | 1,007,975 (71.3%) | **0.0%** | 41.8% |

**Where the gate lets them, the two crown modules agree to within 0.6 pp.** Below it, pyretechnics
returns zero crown fire although the *intensity* criterion is satisfied on **82.3%** of those cells.
North Cottonwood is 71.3% below the threshold.

Three consequences, all of which land on the spotting deliverable (firebrand production scales with
fireline intensity, jump distance with `I^0.3`):

1. **There is no passive-crown regime anywhere** — `fire_type == 2` is zero over all 3.0 M cells.
   Below the gate crowning is blocked; above it, at p97 (42.8 km/hr, 4% 1-hr FM) the Cruz active rate
   (median 107.8 m/min) clears van Wagner's `3/CBD` (median 27.3) on 90.8% of cells, so it goes
   straight to ACTIVE. **Torching in open ponderosa / dry mixed conifer is the characteristic crown
   behaviour of this forest type and the dominant short-range ember source, and the model does not
   produce it at all.**
2. **The modelled COSWAP canopy effect is a threshold artifact.** Of the 2,330 pixels COSWAP changes
   (background → coswap): **89.0%** are already below 0.4 in *both* arms, so the canopy work is
   invisible to the crown model; **10.7%** cross 0.4 downward as a step function; and **0.3% — eight
   pixels — stay above 0.4 in both arms**, the only place the treatment acts as the continuous canopy
   reduction it physically is. Nearly the whole modelled crown effect would ride on pixels tipping
   over one hard-coded constant.
3. It also explains most of G1's flame-length gap: FL = f(I), and suppressed crown fire suppresses I.

**It is not a units problem** — checked three ways. pyretechnics' own `load_landfire.py` applies
`canopy_cover * 0.01 -> [0-0.95]`, "canopy area / total area", so a **fraction** is what the engine
wants and `destack_lcp`'s `CC 45 -> 0.45` matches it (same table confirms `CH x0.10`, `CBH x0.10`,
`CBD x0.01`, and `slope x0.01` — the SLPP trap). Deliberately passing CC in percent *looks* like a
fix (crown 21.4% -> 42.8%) but only because `45 > 0.4` makes every canopied pixel clear the gate; it
degrades above-gate agreement with FlamMap from **99.4% -> 94.6%**, because CC also drives the
wind-adjustment factor. A units error is not a physics knob.

### Gate sweep — and the second, opposite problem
`code/dev/pyre/g1c_crown_gate_sweep.py`, 40,000-cell sample, threshold patched in pyretechnics'
**pure-Python-mode source** (CC correct everywhere else). FlamMap on the same cells: crown 51.6%
(passive 29.8%, active 21.8%).

| gate | crown | passive | active | FL median | agree w/ FlamMap |
|---|---|---|---|---|---|
| CC > 0.4 (default) | 21.3% | 0.0% | 21.3% | 1.81 m | 69.7% |
| CC > 0.3 | 26.6% | 0.0% | 26.6% | 1.93 m | 74.9% |
| CC > 0.2 | 36.3% | 0.1% | 36.1% | 2.50 m | 84.6% |
| CC > 0.1 | 51.1% | 0.5% | 50.7% | 4.97 m | **99.5%** |
| CC > 0.0 | 51.1% | 0.5% | 50.7% | 4.97 m | 99.5% |

**Removing the gate aligns crown extent almost perfectly** — 99.5% pixel agreement on
surface-vs-crown and crown fraction to 0.5 pp. Two independently written crown modules do not agree
at 99.5% by luck; the gate was the entire difference. (0.1 and 0.0 tie because LANDFIRE CC has no
values in (0, 0.1].)

**But passive crown never exceeds 0.5% at any threshold, against FlamMap's 29.8%.** At p97 the Cruz
active rate (median 107.8 m/min) clears van Wagner's `3/CBD` (median 27.3) on 90.8% of canopied
cells, so anything that crowns goes straight to ACTIVE. So flame length **overshoots** with the gate
off (4.97 vs 3.75 m) exactly as it **undershoots** with it on (1.81 m) — pyretechnics is calling
torching-grade fire full active crown fire.

That reframes the constant: the CC > 0.4 gate is plausibly a **crude stand-in for the missing passive
regime**, not an arbitrary number. Removing it trades one bias for its opposite, and both land on the
spotting deliverable:
```
gate ON   too little crown fire  -> too little fireline intensity -> UNDER-states embers
gate OFF  right crown extent, all classed active -> too much intensity -> OVER-states embers
```
**Neither setting is simply correct.** DECIDED 2026-09-11, superseding the 09-10 "balanced sweep axis"
decision:

- **Production runs at gate 0.0** — crowning allowed below CC 0.4. Err extreme: for a
  treatment-effectiveness claim, overstating ember production is the conservative direction, and the
  71% of this landscape below CC 0.4 must be allowed to crown at all.
- **The stock 0.4 gate is an UNBALANCED sensitivity subset**, not a second full axis. Balanced would
  double the campaign to 6,480. Instead: 3 arms x 12 ignitions x **10** episodes x **central flux
  only** = 360, i.e. **3,240 + 360 = 3,600, +11% not +100%**. Episodes are the axis to cut (the
  plan's own rule is `flux >= ignitions > episodes > severity`) because CRN already collapses
  per-pixel noise, while 12 design fires stay the limiting inferential unit — so a paired,
  clustered-by-ignition statement survives at both settings.

**Correction to the 09-10 framing, which matters when writing this up.** Gate-off does **not** buy a
torching regime — passive crown (`fire_type == 2`) stays at or below **0.5% at every threshold**
(0.4 / 0.2 / 0.1 / 0.0). What it allows below CC 0.4 is full **active** crown fire. And the absence of
passive is probably not a defect: at p97 the Cruz active rate (~107 m/min) clears van Wagner's
`3/CBD` (~27–60 m/min), so pyretechnics is saying these canopies *can* carry an active fire at this
severity. FlamMap's ScottReinhardt splits more into passive because it uses a crown-fraction-burned
formulation — a genuine model difference, not one side being broken. So: gate-off overshoots
intensity (active where torching belongs), gate-on undershoots (nothing where torching belongs), and
the subset brackets it. **Open check:** whether passive appears at p75/p90, where the Cruz rate no
longer dominates.

### The patched engine: `pyretechnics 2026.8.10+cfri1`
`code/dev/pyre/build_patched_engine.py` builds the pinned sdist with the gate exposed as a runtime
parameter and installs it over stock in `fb_pyre`. Running the low end in production needs a compiled
build — the pure-Python source is ~100x too slow for a campaign.

**The gate is hard-coded TWICE, on two different code paths**, and patching only one does nothing:
| site | function | consumed by |
|---|---|---|
| `crown_fire.py:83` | `canopy_cover > 0.4` in `van_wagner_crown_fire_initiation` | `burn_cells` (what G1 measured) |
| `crown_fire.py:54` | `if canopy_cover <= 0.4: return inf` in `van_wagner_crowning_spread_rate` | **`spread_fire_with_phi_field` — the campaign path** |

Safety properties, all enforced in the build script:
- **The patch asserts its target text** and fails loudly if absent — an unpatched build that reports
  itself as patched is the worst outcome available. It also deletes the sdist's pre-generated `.c`
  files, which would otherwise be used in preference to re-cythonizing the patched `.py`.
- **The default is inert.** At threshold 0.4 the patched build reproduces stock **exactly** on all
  three arms — crown fraction to 4 dp, FL median to 6 dp, fireline-intensity sums over 3.0 M cells,
  and exact integer active-crown counts (baseline in `$FB_PYRE_SCRATCH/stock_engine_baseline.json`).
- **The build is identifiable.** PEP 440 local label, so `pyre_version_info()` reports
  `pyretechnics_version: 2026.8.10+cfri1`, `pyretechnics_patched: true`, and
  **`crown_cover_threshold`** — recorded on every run because a results table that does not say which
  end of the sweep produced it cannot be interpreted.

API in `models/pyre.py`: `is_patched_engine()`, `crown_cover_threshold()`,
`set_crown_cover_threshold(v)`. The setter raises on a stock build naming the remedy;
`require_pyretechnics` accepts a local label but still pins the base version.
**It is process-global** — set it once per condition level *before* spawning a worker pool, since a
`"spawn"` worker re-imports the module and gets the default.

**What the gate is worth, measured.** Uniform CC 0.25 test landscape: 34x area burned, 67x fireline
intensity. Real North Cottonwood, one 24 h spread run from a central ignition, wind 292.5 FROM,
spotting off:

| gate | burned ha | crown | FL p95 | I p95 kW/m | wall s | peak RSS |
|---|---|---|---|---|---|---|
| 0.4 (stock) | 4,292 | 7.1% | 2.53 m | 1,953 | 3.7 | 0.49 GB |
| 0.0 (no gate) | 6,033 | 38.8% | 5.62 m | 11,101 | 5.3 | 0.56 GB |

Area brackets at only **1.4x** — narrow enough to report. But **I p95 brackets at 5.7x**, and that is
the quantity spotting rides on: firebrand production scales with `I`, jump distance with `I^0.3`
(≈1.7x). Wide, but that is what the sweep is for.

**Incidental Stage 2 preview**: 3.7–5.3 s per 24 h realization at **30 m** on 3.0 M cells, 0.5 GB peak
RSS, single process, spotting off. G2's best branch is "≤30 s at 90 m → produce at 30 m". Not yet a
benchmark — 21 days and spotting on both cost more — but 30 m production looks reachable.

`tests/test_pyre_engine.py` — 18 tests. The engine-free half runs in `fb_tools`; the rest skip there
and skip on a stock build. The keystone drives a **real spread run** at both thresholds, because the
campaign-path site is a `cy.cfunc` unreachable from Python, so end-to-end is the only assertion
available and the one that matters.

### Diagnostic tooling: `code/dev/pyre/pyre_source.py`
`ensure_pyre_source()` / `import_pure_python()` fetch the pinned sdist into the scratch root and swap
the process onto pyretechnics' pure-Python modules. Two uses, both load-bearing:
- **`cy.cfunc` functions become callable**, including `spot_fire.expected_firebrand_production` —
  which the compiled build does not export, and which Stage 1b is built on. A vectorized
  reimplementation can now be checked against the reference implementation rather than a docstring.
- **Constants can be patched for an experiment** without shipping a fork.
Diagnostic only: nothing under `fb_tools/` imports it, and production runs use the compiled, pinned,
version-stamped engine. `import_pure_python()` is irreversible in-process — do compiled work first.

### Stage 2 / G2 — DONE. Produce at **30 m**. And the domain is too small.
`code/dev/pyre/b1_benchmark.py`; CSVs in `$FB_PYRE_SCRATCH/benchmark/`. Every run is its own
**`spawn` subprocess** (so `ru_maxrss` is per-run, and the Stage 3 parallelism mechanism is exercised
early), weather is a **1D hourly cube** (constant p97 values but `days*24` bands, so per-band
cube-refresh cost is measured rather than skipped), and the ignition is a **real Phase 2 design fire**
(`ign_000`, cell 737,639).

| res | days | spot | gate | grid | wall s | RSS | burned ha | %dom | crown |
|---|---|---|---|---|---|---|---|---|---|
| 90 | 21 | on | 0.4 | 580×577 | 41.2 | 0.26 GB | 122,521 | 45.2% | 20.1% |
| 90 | 21 | on | 0.0 | 580×577 | 42.5 | 0.26 GB | 123,371 | 45.5% | 48.2% |
| 30 | 21 | on | 0.4 | 1738×1730 | 159.6 | 0.53 GB | 32,484 | 12.0% | 30.7% |
| 30 | 21 | on | 0.0 | 1738×1730 | **252.2** | 0.57 GB | **59,389** | 21.9% | 47.3% |

**G2: 30 m.** The 3,600-realization design costs ~30 h on 8 workers at 30 m vs ~5 h at 90 m — both
affordable, so cost stopped being the deciding factor. **The plan's "30–120 s @ 90 m → produce at
90 m" branch is deliberately overridden**, because the measurement it existed to inform showed
something the plan did not anticipate: **90 m burns 3.8x the area of 30 m** (123,371 vs 32,484 ha).
That is resampling artifact, not physics — mode-resampling FBFM40 to 90 m destroys the fuel-break
structure (roads, riparian, rock) that stops fires at 30 m. 90 m also has worse edge contact and
36 km max spread vs 28 km. RSS is only ~0.55 GB/worker at 30 m, so 8 workers is comfortable.

**Spotting only fires when crown fire supplies the intensity** — a clean internal-consistency result
that also justifies the gate-0.0 production choice. At gate 0.4, spotting adds **14 ha**
(32,470 → 32,484, i.e. nothing). At gate 0.0 it adds **27,000 ha (+83%)**: spotting is what breaks
the fire out of its fuel-connected basin. **At the stock gate the primary deliverable would have been
approximately nil.**

### OPEN GATE — the 52 km domain fails `check_domain_adequacy` on all 16 configs
It was never called in the FSPro era (`spread/fspro_outputs.py:235`). It should have been.
- `edge_ok` and `headroom_ok` **fail in every configuration**. Spread headroom 0.535–0.856 (needs
  `> 1`), so fires run past the nearest boundary every time. Max spread **22.4–27.7 km** at 30 m,
  34–36 km at 90 m.
- **The downwind corridor from the treatment centroid exits the domain after 22.7 km** — the same
  scale as the spread. Treated pixels are a median 11.9 km from the edge, but p5 is 4.5 km and the
  minimum is **0.0 km**.
- Final-day growth share is ~0.0000, so **duration is not what binds these fires — the domain is.**

Mitigating: edge BP mass is only 0.3–0.7%, and the planned reporting zones are near-field
(in-treatment, 500 m collar, ~5 km downwind), which sits well inside the window.

**DECIDED 2026-09-11: defer, do not enlarge the domain yet.** This benchmark used **constant p97 with
no diurnal cycle** — conservative for runtime, but simultaneously the worst possible case for domain
adequacy. Stage 4's real hourly episodes bring overnight wind drop and fuel-moisture recovery and
should shrink fires materially.

**Gate, not a note — Stage 4 must re-run `check_domain_adequacy` on the first real episodes before
any production campaign, and G4's fire-size validation against FPA-FOD must state that simulated
sizes are domain-bounded until that check passes.**

### Spotting is a HEAVY TAIL, not an increment — and never quote a single realization
`code/dev/pyre/{b2_spot_variance,b3_buffer_diagnosis}.py`. The knob sweep first looked like solver
fragility: at 30 m / 21 d / gate 0.0 the "+83% spotting effect" appeared only at `buf 3, cfl <= 0.4`
and 32,550 ha (the spotting-off answer) everywhere else. It is not. Holding the solver fixed and
varying **only the seed**, pooled over 10 seeds per buffer:

| | spotting off | on, mean | breakouts | CV of burned area |
|---|---|---|---|---|
| buf 3 | 32,463 ha | 41,655 ha (+28%) | **4/10** | 30.1% |
| buf 6 | 32,546 ha | 44,915 ha (+38%) | **2/10** | 41.0% |

Fisher exact on 4/10 vs 2/10 gives **p = 0.63 — indistinguishable**, and spotting-off agrees between
buffers to 0.26%. So **`buffer_width` is innocent**; the knob-sweep pattern was small-sample noise in
a bimodal outcome, not a solver effect.

**The shape of the distribution is the finding.** Per realization, spotting adds **+0.01% to +1.4%**
in the common case and **+83% to +145%** in the tail (32.5k / 59.4k / 79.8k ha are distinct modes —
one ember reaching a new fuel-connected basin, or two). The *mean* effect is therefore entirely a
function of **breakout frequency**, and the mean of a bimodal variable is not a typical outcome.
Consequences that bind the analysis:
- **Report a breakout RATE plus the conditional magnitudes, never a mean area alone.**
- **Breakout-rate precision**: +/-16 pp at 30 episodes (one ignition), +/-5 pp at 360 (pooled per arm
  per condition). So **per-ignition breakout rates are too noisy to report** — the plan's "per-ignition
  effects as primary" guidance holds for paired Δ but NOT for the breakout rate, which needs pooling.
- Breakout runs cost 270–437 s vs 165–245 s, so campaign wall-clock depends on breakout frequency too;
  the ~252 s/realization G2 figure already averages over it.
- `paired_effect_interval` (Stage 5) must cluster on ignition **and** cope with a bimodal outcome —
  a bootstrap of a heavy-tailed mean needs the clustered resample, not a normal approximation.

**Two process lessons, both mine.** (1) My first verdict function concluded "the solver is innocent"
from buf-3 bimodality alone while ignoring buf 6's 0.1% CV; it now requires *both* buffers to show
seed variance. (2) Four seeds was underpowered and I read a systematic effect into it — buf 6's 0/4
with CV 0.1% was chance, and six more seeds reversed it outright. **n=4 is not enough to characterise
a ~30% tail; nothing in Stage 2 is settled from four runs.**

**Pin and stamp the solver knobs.** `max_cells_per_timestep` and `buffer_width` must go into every
manifest beside `crown_cover_threshold`. Production: **cfl 0.4, buf 3** (the pyretechnics defaults),
chosen because they are the defaults and the evidence does not distinguish them, not because they were
tuned. `cfl 0.8` is 3.1x faster and is NOT free — it suppressed the tail in the one knob run, which on
this evidence is small-sample noise but is untested at n=10.

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
