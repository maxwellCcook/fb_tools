"""
What does pyretechnics' hard-coded ``canopy_cover > 0.4`` crown gate cost us?

``g1b`` established the gate is the whole source of the 21%-vs-51% crown gap.
This sweeps the threshold on the real landscape, with canopy cover **correct
everywhere else** — the distinction matters, because deliberately passing CC in
percent instead of as a fraction *looks* like it fixes the crown fraction
(21.4% -> 42.8%) while actually just making every canopied pixel clear the gate,
and it degrades above-gate agreement with FlamMap from 99.4% to 94.6% because CC
also drives the wind-adjustment factor. A units error is not a physics knob.

Run:  conda activate fb_pyre && python code/dev/pyre/g1c_crown_gate_sweep.py

RESULT (40,000-cell random sample, North Cottonwood untreated, p97 pyrome 45;
FlamMap crown on the same cells: 51.6%):

    gate      crown   passive   active   FL median   agree w/ FlamMap
    CC > 0.4  21.3%      0.0%    21.3%      1.81 m              69.7%
    CC > 0.3  26.6%      0.0%    26.6%      1.93 m              74.9%
    CC > 0.2  36.3%      0.1%    36.1%      2.50 m              84.6%
    CC > 0.1  51.1%      0.5%    50.7%      4.97 m              99.5%
    CC > 0.0  51.1%      0.5%    50.7%      4.97 m              99.5%

Two readings, and they point opposite ways:

**Removing the gate aligns crown EXTENT almost perfectly.** At CC > 0.1 the two
models agree on surface-vs-crown at **99.5% of pixels** and on crown fraction to
0.5 pp. Two independently written crown modules do not agree at 99.5% by luck:
the physics is the same and the gate was the entire difference. (0.1 and 0.0 are
identical because LANDFIRE CC has no values in (0, 0.1] — cells below 0.1 are
bare 0.)

**But pyretechnics has no passive-crown regime either way.** Passive stays <= 0.5%
at every threshold, while FlamMap splits its crown cells 58% passive / 42% active
(417,822 vs 307,558 on the full window). At p97 — 42.8 km/hr, 4% 1-hr FM — the
Cruz active spread rate (median 107.8 m/min) clears van Wagner's 3/CBD (median
27.3) on 90.8% of canopied cells, so anything that crowns goes straight to ACTIVE.
So with the gate off, flame length **overshoots** FlamMap (4.97 vs 3.75 m) exactly
as it **undershoots** with the gate on (1.81 m): pyretechnics is calling
torching-grade fire full active crown fire.

That reframes the gate. It is plausibly a **crude stand-in for the missing passive
regime** — "only fairly dense canopy sustains active crown fire" — rather than an
arbitrary constant. Removing it without adding a torching regime trades one bias
for its opposite:

    gate ON   too little crown fire  -> too little fireline intensity
                                     -> UNDER-states ember production
    gate OFF  right crown extent, all of it classed active
                                     -> too much intensity
                                     -> OVER-states ember production

Both matter directly: firebrand production scales with fireline intensity and jump
distance with I^0.3. Neither setting is simply "correct", which is the finding.
"""

import time

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# Compiled-engine work first: import_pure_python() is irreversible in-process.
from g1_flammap_diagnostic import (
    FLAMMAP_OUT, FOLIAR_MOISTURE, LAYER_DIR, LCP_DIR, P97, PYROME_ID, build_cubes,
)
from g1b_crown_divergence import NB
from fb_tools.spread.convert import destack_lcp

N_SAMPLE = 40_000
SEED = 617327
GATES = (0.4, 0.3, 0.2, 0.1, 0.0)


def prepare():
    """De-stack, read the FlamMap reference, and draw the comparison sample."""
    layers = destack_lcp(LCP_DIR / "untreated.tif", LAYER_DIR / "untreated", verbose=False)
    _, arrays, profile, bounds, ws_kmh = build_cubes(layers)

    fp = FLAMMAP_OUT / "baseline_lcp" / "p97" / "baseline_lcp.tif"
    with rasterio.open(fp) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        names = list(src.descriptions)
        cs = src.read(names.index("CROWNSTATE") + 1, window=win, masked=True)
    cs = np.ma.filled(cs.astype("float64"), np.nan)

    burnable = ~np.isin(arrays["fuel_model"].astype("int32"), NB)
    comparable = burnable & np.isfinite(cs) & (cs > 0)

    rng = np.random.default_rng(SEED)
    flat = np.flatnonzero(comparable.ravel())
    samp = rng.choice(flat, size=min(N_SAMPLE, flat.size), replace=False)
    ys, xs = np.unravel_index(samp, comparable.shape)
    return arrays, ws_kmh, ys, xs, (cs.ravel()[samp] >= 2), cs.ravel()[samp]


def build_pure_cubes(arrays, ws_kmh, SpaceTimeCube):
    shape = (1,) + arrays["slope"].shape
    f = np.float32
    return {
        "slope":               SpaceTimeCube(shape, arrays["slope"]),
        "aspect":              SpaceTimeCube(shape, arrays["aspect"]),
        "fuel_model":          SpaceTimeCube(shape, arrays["fuel_model"].astype("int64")),
        "canopy_cover":        SpaceTimeCube(shape, arrays["canopy_cover"]),
        "canopy_height":       SpaceTimeCube(shape, arrays["canopy_height"]),
        "canopy_base_height":  SpaceTimeCube(shape, arrays["canopy_base_height"]),
        "canopy_bulk_density": SpaceTimeCube(shape, arrays["canopy_bulk_density"]),
        "wind_speed_10m":      SpaceTimeCube(shape, f(ws_kmh)),
        "upwind_direction":    SpaceTimeCube(shape, arrays["aspect"]),
        "fuel_moisture_dead_1hr":        SpaceTimeCube(shape, f(P97["FM_1hr"] / 100.0)),
        "fuel_moisture_dead_10hr":       SpaceTimeCube(shape, f(P97["FM_10hr"] / 100.0)),
        "fuel_moisture_dead_100hr":      SpaceTimeCube(shape, f(P97["FM_100hr"] / 100.0)),
        "fuel_moisture_live_herbaceous": SpaceTimeCube(shape, f(P97["FM_herb"] / 100.0)),
        "fuel_moisture_live_woody":      SpaceTimeCube(shape, f(P97["FM_woody"] / 100.0)),
        "foliar_moisture":               SpaceTimeCube(shape, f(FOLIAR_MOISTURE)),
    }


def main():
    arrays, ws_kmh, ys, xs, fm_crown, fm_cs = prepare()

    from pyre_source import import_pure_python
    import_pure_python()
    import pyretechnics.burn_cells as pbc
    import pyretechnics.crown_fire as cf
    from pyretechnics.space_time_cube import SpaceTimeCube

    cubes = build_pure_cubes(arrays, ws_kmh, SpaceTimeCube)

    print(f"\n{len(ys):,}-cell random sample (seed {SEED}), North Cottonwood untreated, "
          f"p97 pyrome {PYROME_ID}")
    print(f"FlamMap on these cells: crown {100 * fm_crown.mean():.1f}%  "
          f"(passive {100 * (fm_cs == 2).mean():.1f}%, active {100 * (fm_cs == 3).mean():.1f}%)\n")
    print(f"  {'gate':>10s} {'crown':>7s} {'passive':>8s} {'active':>7s} "
          f"{'FL median':>10s} {'agree':>7s}")

    rows = []
    for gate in GATES:
        def patched(intensity, canopy_cover, cbh, foliar, _g=gate):
            return bool(
                intensity > 0.0
                and canopy_cover > _g
                and intensity >= cf.van_wagner_critical_fireline_intensity(cbh, foliar)
            )
        # Patch both the defining module and the name burn_cells resolved at import.
        cf.van_wagner_crown_fire_initiation = patched
        pbc.cf.van_wagner_crown_fire_initiation = patched

        t0 = time.perf_counter()
        ft = np.empty(len(ys), dtype="int32")
        fl = np.empty(len(ys), dtype="float64")
        for i, (y, x) in enumerate(zip(ys, xs)):
            r = pbc.burn_cell_as_head_fire(cubes, (0, int(y), int(x)))
            ft[i] = r["fire_type"]
            fl[i] = r["flame_length"]

        crown = ft >= 2
        row = dict(
            gate=gate,
            crown=float(crown.mean()),
            passive=float((ft == 2).mean()),
            active=float((ft == 3).mean()),
            fl_median=float(np.median(fl[fl > 0])),
            agreement=float((crown == fm_crown).mean()),
            seconds=time.perf_counter() - t0,
        )
        rows.append(row)
        print(f"  {'CC > ' + format(gate, '.1f'):>10s} {100*row['crown']:>6.1f}% "
              f"{100*row['passive']:>7.1f}% {100*row['active']:>6.1f}% "
              f"{row['fl_median']:>8.2f} m {100*row['agreement']:>6.1f}%")

    best = max(rows, key=lambda r: r["agreement"])
    print(f"\n  Best surface-vs-crown agreement with FlamMap: CC > {best['gate']:.1f} "
          f"at {100 * best['agreement']:.1f}%")
    print(f"  Passive crown never exceeds {100 * max(r['passive'] for r in rows):.1f}% at any "
          f"threshold, against FlamMap's {100 * (fm_cs == 2).mean():.1f}% — the missing torching")
    print("  regime is a separate problem from the gate, and it is why flame length")
    print("  overshoots with the gate off and undershoots with it on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
