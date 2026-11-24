# fb_tools — development notes

A dated record of what was built and decided, newest first, followed by work in progress, planned work, and
the detailed records that used to live in `CLAUDE.md`. Current state by module: [STATUS.md](STATUS.md).

Detailed logs kept separately:
- [pyre_decisions.md](pyre_decisions.md) — every pyretechnics decision and finding, dated.
- [ACTIVE_FIRE_PLAN.md](ACTIVE_FIRE_PLAN.md) — the active-fire module plan, with what was built at each step.
- [pyretechnics_transmission_plan.md](pyretechnics_transmission_plan.md) — the original pyretechnics plan
  (2026-09-10); superseded by the decision log where they disagree.

## Log

**2026-11-23 / 24 — pyretechnics audit, shadow review, example notebook**
- Audit prompted by the ELMFIRE 1.1 release. Ten fixes for errors that could give a wrong result silently
  (ignition reprojection, short weather arrays, required crown gate, cross-arm grid checks, one-arm and
  weighted ensembles, day marks, layer reuse, season-end limit, `apply_treatment` band map).
- Treatment shadow reviewed on synthetic cases and example runs: gaps of up to 3 cells bridged; ember-started
  fire reported separately; the shadow defined as a baseline-arm product; lead time clocked from the upstream
  treatment unit; `shadow_agreement` added to score it against the paired change.
- Position reached: paired differences are the measure of a treatment's effect; the shadow is the stratifier
  and the siting tool.
- `01f_PyreBuild.ipynb` §5b rebuilt as one named fire over 20 spotting seeds at 14 days; `prune_realizations`
  added so per-run rasters are not kept.
- The April–October weather limit was recognized as a constraint (Marshall, late-season 2020).
- HRRR downloads now retry dropped connections.

**2026-11-13 / 14 — active fire module**
- `fb_tools.active_fire` scoped and build steps 1–2 plus treatment summaries completed, ported from the
  potr5-fb and AspenAcres project code.

**2026-11-12 — ignition density grids by cause**
- `load_fod`, `ignition_density_grid` (fixed or adaptive kernel), unit conversion, the Larimer-area notebook
  and the ArcGIS Online build and publish notebooks. FPA-FOD reporting is complete for state and local
  agencies only from 2013.

**2026-11-09 — urban fuel models noted** (not started; see Planned).

**2026-10-08 — fuel break**
- `fuel_break` (treated corridor along line features) and the first single-fire counterfactual in 01f.

**2026-09-28 — combination LCP**
- FlamMap once per unique fuel combination; 22 arms plus baseline in 3 runs instead of 69.

**2026-09-10 to 09-15 — pyretechnics adopted for COSWAP transmission**
- Installed and pinned in a second environment; LCP de-stacker with unit tests.
- Crown-fire divergence from FlamMap traced to a hard-coded canopy-cover gate; patched into a runtime
  parameter (CFRI build).
- Production resolution set at 30 m; spotting found to be a heavy tail; the 52 km domain fails adequacy.
- Episodes, the paired runner, HRRR 3-D wind cubes (herbie removed), FSim-style ignition sites, transmission
  products, the near-field profile, the stepped runner and the first containment rule.

**2026-08-05 to 08-07 — FSPro risk-transmission phases**
- Phase 0: input validator, domain diagnostic, first test suite.
- P0.1 measured on the VM: the FSPro seed does not give paired runs.
- Phase 1: weather stream corrections (ERC classes, current ERC stream, fuel-moisture time series, wind cells).
- Phase 2: simulation domain, N-arm experiments, design-fire selection; Colorado ignitions shown to be clustered.
- Phase 3a: pre-flight correctness before the COSWAP production run; null bands.
- Wind sweep and LCP masking added.

**2026-05 to 2026-06 — RTMA and FSPro counterfactuals**
- Hourly RTMA pipeline for fuel moisture (GEE export, fire mask, comparison with RAWS).
- FSPro container counterfactuals; treatment summaries moved to the tealom package.

**2026-03 to 2026-04 — package started**
- Initial commit 2026-03-18: FlamMap batch runs, SDI, draft treatment summaries.
- FSPro wrapper, GridMET ERC climatology, HRRR wind cells, the weather module.

## In progress

The audit plan from 2026-11-23, in order. Items marked *(decision)* wait on Max.
1. Suppression review — keep to the engines' built-in functionality; *(decision)* on the flame-length gate and
   the median SDI fill, which are our additions.
2. Items that change a reported number *(decision each)*: HRRR wind rotation; arrivals never reached in
   `paired_deltas`; sign of differences (FSPro is baseline − treated, pyretechnics treated − baseline);
   ignition footprint in pyretechnics `BurnProb`; firebrand slope correction; rounding before differencing.
3. Minor fixes found in the audit.
4. pyretechnics upgrade test (2026.8.10 → 2026.11.20) in the existing `fb_pyre` environment, on a small
   reference case. The 2026.10.20 release corrects under-predicted spread and intensity and changes the license.
5. Clean-up: an engine-neutral core, one raster writer, one mask rasterizer, one ensemble aggregator, notebook
   glue moved into the package.
6. Year-round weather record: full-calendar GridMET ERC and RTMA months 11–03 *(exports by Max)*, a
   calendar-day episode path, QC of winter fuel moisture.
7. ELMFIRE spike in Docker: suppression, building spread, Monte Carlo burn probability, pairing noise.
8. Burn probability *(decision)*: conditional only, anchored to FSim, scaled by fire occurrence, or calibrated annual.

Open questions recorded with the findings: whether GS1 with no canopy is the intended post-clear-cut fuel;
why the engine's arrival raster skips cells.

## Planned, not started

### Urban fuel models (built environment as burnable fuel)

**Status:** not started; not time sensitive. Noted 2026-11-09 (Max), from the Gunnison TEALOM work.

**Goal.** An optional, shared set of urban fuel models that projects can pull in for testing and for new
approaches that treat the built environment as burnable fuel. Best suited to spread-based work
(FSim, MTT, transmission).

**Source.** Moran, Seielstad and Pietruszka (2025), *International Journal of Wildland Fire* 34, WF24132,
doi:10.1071/WF24132, open access (CC BY). Calibrated to a target rate of spread of 0.81 km/h from seven urban
conflagrations. Confirm this is the paper intended.

| model | 1-h | 10-h | 100-h | herb | woody (Mg/ha) | type | SAV 1-h / herb / woody (m⁻¹) | depth (cm) | Mx (%) | heat (J/g) |
|---|---|---|---|---|---|---|---|---|---|---|
| BU1 | 5.00 | 0 | 0 | 0.50 | 27.0 | static | 13,120 / 13,120 / 13,120 | 20 | 12 | 20,000 |
| FM9 | 6.54 | 0.93 | 0.34 | 0 | 0 | static | 8,202 / – / – | 6 | 25 | 18,622 |

BU0 is excluded: its 30,000 m⁻¹ SAV exceeds what current systems allow. Values are from the paper's Table 1.
FM9 is Anderson model 9, so it needs no custom definition.

**Scope: small.** About one session of code plus one short VM run.

1. Package data: the table above, with units and source.
2. `models/flammap.py`, `run_flammap_scenarios`: optional custom fuels in `FlamMap.input`. `FlamMap.dll` reads
   `CUSTOM_FUELS_FILE`, `CUSTOM_FUELS_UNITS` and `CUSTOM_FUELS_DATA`; the record format in the DLL is
   `%d %s %f %f %f %f %f %s %f %f %f %f %f %f %f %s`. Confirm the syntax against FlamMap's documentation.
3. A recode function: LANDFIRE EVT 7296 (Developed-Low) and 7297 (Developed-Medium) to a chosen model,
   with roads (7299) left alone. Check how code 9 behaves when mixed with 40-model codes on one landscape.
4. Tests, and a short section in `CLAUDE.md`.

**Check before any project uses it.** On a small synthetic landscape, reproduce the paper's curves on the current
`TestFlamMap`: open wind 0–80 km/h, under its very low (3, 4, 5, 30, 60 %) and low (6, 7, 8, 60, 90 %)
moisture scenarios.

**Resolve first.**
- **BU1 and the wind limit.** BU1 is tuned to be capped near 0.8 km/h by the effective wind limit. FlamMap's help
  lists that limit as no longer used from the August 30, 2024 version of 6.2, so BU1 is unlikely to behave as
  published on the current build. Quantify the difference. This may decide between BU1 and FM9.
- **BU1 1-h load.** Table 1 gives 5.00 Mg/ha; the text says 6.54 (FM9's value). Find which reproduces the curves.
- **What is validated.** Spread rate only. Flame length is not, so per-pixel flame-length risk workflows should
  treat these as exploratory.
- **What it does not cover.** Roads and the non-burnable fill used in RADS products. In Gunnison, Developed-Roads
  holds most of the filled pixels (see that project's `docs/decision_log.md`, 2026-11-09).

### Smaller items
- `weather/gridmet.py`: a current-season ERC fetch for operational `CurrentERCValues`.
- MTT wrapper: end-to-end run on the Windows VM.
- LFPS: confirm product codes for LANDFIRE versions 220 and 230.
- Active fire: swath access and the remaining build steps in `ACTIVE_FIRE_PLAN.md`.

## Detailed records

Moved from `CLAUDE.md` on 2026-11-24, unchanged.

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
