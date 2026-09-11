"""
Why does pyretechnics call 21% of burnable pixels crown fire where FlamMap
(ScottReinhardt) calls 51%?

ANSWER (measured on North Cottonwood, p97 pyrome 45, 2026-09-10):
``crown_fire.van_wagner_crown_fire_initiation`` carries a **hard-coded
``canopy_cover > 0.4`` gate** that Van Wagner (1977) does not have:

    surface_fireline_intensity > 0.0
    and canopy_cover > 0.4                    <-- this
    and surface_fireline_intensity >= van_wagner_critical_fireline_intensity(...)

Above the gate the two models agree almost exactly (pyre 74.6% crown vs FlamMap
75.2%). Below it pyretechnics returns 0.0% against FlamMap's 41.8%. **This is not
an adapter bug** — our inputs are right and the intensity criterion is satisfied
on 82.3% of the blocked cells; it is a physics choice in the engine meeting a
landscape that is 71.3% below the threshold.

Two consequences, both of which matter more than the headline number:

1. **Passive crown fire does not occur anywhere on this landscape.**
   ``fire_type == 2`` count is zero over all 3.0 M cells. Below CC 0.4 the gate
   blocks crowning outright; above it, at p97 wind (42.8 km/hr) and 4% 1-hr FM,
   Cruz active spread rate (~107 m/min) clears van Wagner's critical rate
   (3/CBD ~ 30 m/min) almost everywhere, so it jumps straight to active. The
   model has no torching regime — and torching in open ponderosa / dry mixed
   conifer is the characteristic crown behaviour of exactly this forest type and
   the dominant short-range ember source.

2. **The modelled COSWAP canopy effect is a threshold artifact.** Of the 2,330
   pixels COSWAP changes between the background and coswap arms, 89.0% are
   already below CC 0.4 in *both* arms (so the canopy work is invisible to the
   crown model), 10.7% cross 0.4 downward as a step function, and **0.3% — eight
   pixels — stay above 0.4 in both arms**, which is the only place the treatment
   acts as the continuous canopy reduction it physically is. Essentially the
   entire modelled crown-fire treatment effect would come from pixels tipping
   over one hard-coded constant.

Spotting is the primary deliverable of this port. Firebrand production scales
with fireline intensity and jump distance with I^0.3, so a crown-fire regime this
bimodal propagates straight into the ember result. Read this before designing
Stage 7.

Run:  conda activate fb_pyre && python code/dev/pyre/g1b_crown_divergence.py
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from g1_flammap_diagnostic import (
    ARMS, FLAMMAP_OUT, FOLIAR_MOISTURE, LAYER_DIR, LCP_DIR, P97, PYROME_ID,
    build_cubes,
)
from fb_tools.spread.convert import destack_lcp

NB = (0, 91, 92, 93, 98, 99)
FIRE_TYPE = {0: "unburned", 1: "surface", 2: "passive", 3: "active"}

#: The gate, transcribed from pyretechnics crown_fire.py. Not configurable there.
CC_GATE = 0.4


def van_wagner_i0(cbh_m, foliar_moisture=FOLIAR_MOISTURE):
    """Critical fireline intensity for crown initiation (kW/m).

    Transcribed from pyretechnics' own crown_fire.py so the comparison uses the
    engine's constants: H = 460 + 2600 * FMC with FMC as a fraction, which equals
    the familiar 460 + 25.9 * FMC_percent to within 0.3%.
    """
    h = 460.0 + 2600.0 * foliar_moisture
    v = 0.01 * cbh_m * h
    return v ** 1.5


def van_wagner_critical_spread_rate(cbd):
    """m/min. 3.0 / CBD, the active-crown criterion."""
    return 3.0 / np.where(cbd > 0, cbd, np.nan)


def cruz_active_spread_rate(ws_kmh, cbd, fm_1hr):
    """m/min. Cruz et al. active crown fire spread rate, as pyretechnics has it."""
    return 11.02 * ws_kmh ** 0.90 * np.where(cbd > 0, cbd, np.nan) ** 0.19 * np.exp(-17.0 * fm_1hr)


def read_flammap(scenario, bounds, shape):
    fp = FLAMMAP_OUT / scenario / "p97" / f"{scenario}.tif"
    with rasterio.open(fp) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        names = list(src.descriptions)
        cs = src.read(names.index("CROWNSTATE") + 1, window=win, masked=True)
    cs = np.ma.filled(cs.astype("float64"), np.nan)
    assert cs.shape == shape, (cs.shape, shape)
    return cs


def pct(x):
    return f"{100 * x:.1f}%"


def crosstab(pyre_ft, fm_cs, mask):
    print("    pyretechnics \\ FlamMap      0        1        2        3")
    for p in (0, 1, 2, 3):
        row = [int(((pyre_ft == p) & (fm_cs == f) & mask).sum()) for f in (0, 1, 2, 3)]
        print(f"    {FIRE_TYPE[p]:>12s} ({p})  " + "".join(f"{v:>9,}" for v in row))


def diagnose_arm(arm, scenario):
    import pyretechnics.burn_cells as bc

    print(f"{'=' * 74}\n{arm}  (vs FlamMap {scenario}/p97)\n{'=' * 74}")
    layers = destack_lcp(LCP_DIR / f"{arm}.tif", LAYER_DIR / arm, verbose=False)
    cubes, arrays, profile, bounds, ws_kmh = build_cubes(layers)

    r = bc.burn_all_cells_as_head_fire(cubes, t=0)
    ft = r["fire_type"].astype("int32")
    pyre_i = r["fireline_intensity"].astype("float64")

    cs = read_flammap(scenario, bounds, arrays["slope"].shape)
    cs_i = np.where(np.isfinite(cs), cs, -1).astype("int32")

    fuel = arrays["fuel_model"].astype("int32")
    cbh = arrays["canopy_base_height"].astype("float64")
    cbd = arrays["canopy_bulk_density"].astype("float64")
    cc = arrays["canopy_cover"].astype("float64")

    burnable = ~np.isin(fuel, NB)
    comp = burnable & np.isfinite(cs) & (cs_i > 0) & (pyre_i > 0)
    n_comp = int(comp.sum())

    print(f"\n  [1] cross-tab, {n_comp:,} comparable burnable cells")
    crosstab(ft, cs_i, comp)
    print(f"      passive crown (fire_type == 2) anywhere in the grid: "
          f"{int((ft == 2).sum()):,}")

    # --- 2. the gate ------------------------------------------------------
    print(f"\n  [2] crown fraction by canopy cover — the gate is CC > {CC_GATE}")
    print(f"      {'CC band':>12s} {'cells':>10s} {'share':>7s} "
          f"{'pyre crown':>11s} {'FlamMap':>9s}")
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)]:
        m = comp & (cc >= lo) & (cc < hi)
        if not m.sum():
            continue
        flag = "   <- gate" if lo == 0.4 else ""
        print(f"      {lo:.1f}-{hi:>4.1f}    {int(m.sum()):>10,} {pct(m.sum()/n_comp):>7s} "
              f"{pct((ft[m] >= 2).mean()):>11s} {pct((cs_i[m] >= 2).mean()):>9s}{flag}")

    above = comp & (cc > CC_GATE)
    below = comp & (cc <= CC_GATE)
    print(f"\n      ABOVE the gate: {int(above.sum()):,} ({pct(above.sum()/n_comp)}) — "
          f"pyre {pct((ft[above] >= 2).mean())} vs FlamMap {pct((cs_i[above] >= 2).mean())}"
          "  <- the crown modules AGREE")
    print(f"      BELOW the gate: {int(below.sum()):,} ({pct(below.sum()/n_comp)}) — "
          f"pyre {pct((ft[below] >= 2).mean())} vs FlamMap {pct((cs_i[below] >= 2).mean())}")
    would = float((pyre_i[below] >= van_wagner_i0(cbh[below])).mean())
    print(f"      of the blocked cells, the INTENSITY criterion is met on {pct(would)} "
          "— so the gate, not the physics, is what blocks them")

    # --- 3. why no passive regime ----------------------------------------
    m = above & (cbd > 0)
    act = cruz_active_spread_rate(ws_kmh, cbd[m], P97["FM_1hr"] / 100.0)
    crit = van_wagner_critical_spread_rate(cbd[m])
    print(f"\n  [3] why no passive regime, on cells that pass the gate")
    print(f"      Cruz active spread rate median {np.nanmedian(act):.1f} m/min "
          f"vs van Wagner critical 3/CBD median {np.nanmedian(crit):.1f} m/min")
    print(f"      active >= critical on {pct(float(np.nanmean(act >= crit)))} of them "
          "— so crowning goes straight from none to ACTIVE, with no torching in between")
    return arrays


def treatment_crossings(cc_by_arm, fuel_by_arm):
    """How much of the treatment's canopy work the crown model can even see."""
    print(f"\n{'=' * 74}\n  [4] does the treatment act as a gradient, or as a step over CC "
          f"{CC_GATE}?\n{'=' * 74}")
    for ref, trt in (("untreated", "coswap"), ("background", "coswap"),
                     ("untreated", "background")):
        changed = (cc_by_arm[ref] != cc_by_arm[trt]) | (fuel_by_arm[ref] != fuel_by_arm[trt])
        n = int(changed.sum())
        if not n:
            print(f"  {ref} -> {trt}: no pixels differ")
            continue
        a, b = cc_by_arm[ref][changed], cc_by_arm[trt][changed]
        down = int(((a > CC_GATE) & (b <= CC_GATE)).sum())
        up = int(((a <= CC_GATE) & (b > CC_GATE)).sum())
        both_hi = int(((a > CC_GATE) & (b > CC_GATE)).sum())
        both_lo = int(((a <= CC_GATE) & (b <= CC_GATE)).sum())
        print(f"\n  {ref} -> {trt}: {n:,} pixels differ ({n * 0.09:,.0f} ha), "
              f"CC median {np.median(a):.2f} -> {np.median(b):.2f}")
        print(f"    both already BELOW {CC_GATE} (canopy work invisible to the crown "
              f"model): {both_lo:>7,} {pct(both_lo/n):>7s}")
        print(f"    cross {CC_GATE} downward  (step function, not a gradient)          : "
              f"{down:>7,} {pct(down/n):>7s}")
        print(f"    cross {CC_GATE} upward                                             : "
              f"{up:>7,} {pct(up/n):>7s}")
        print(f"    both ABOVE {CC_GATE} (the only genuine continuous effect)          : "
              f"{both_hi:>7,} {pct(both_hi/n):>7s}")


def main():
    print(f"p97 pyrome {PYROME_ID}: FM_1hr {P97['FM_1hr']}%, WIND_SPEED "
          f"{P97['WIND_SPEED']} mph (20 ft), foliar moisture {FOLIAR_MOISTURE}")
    print(f"Van Wagner I0: CBH 1 m -> {van_wagner_i0(1.0):,.0f} kW/m, "
          f"2 m -> {van_wagner_i0(2.0):,.0f}, 5 m -> {van_wagner_i0(5.0):,.0f}\n")

    # One arm is enough to diagnose a landscape-wide gap: the arms differ only
    # inside the treated polygons, 0.07% of the landscape.
    diagnose_arm("untreated", ARMS["untreated"])

    cc_by_arm, fuel_by_arm = {}, {}
    for arm in ARMS:
        L = destack_lcp(LCP_DIR / f"{arm}.tif", LAYER_DIR / arm, verbose=False)
        with rasterio.open(L["canopy_cover"]) as s:
            cc_by_arm[arm] = s.read(1).astype("float64")
        with rasterio.open(L["fuel_model"]) as s:
            fuel_by_arm[arm] = s.read(1).astype("int32")
    treatment_crossings(cc_by_arm, fuel_by_arm)

    print(f"\n{'=' * 74}")
    print("  CONCLUSION: not an adapter bug. pyretechnics' hard-coded CC > 0.4 crown")
    print("  gate meets a landscape that is 71% below it. The crown modules agree")
    print("  where the gate lets them. Decide how to handle it before Stage 7 —")
    print("  spotting rides on fireline intensity, which rides on crown fire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
