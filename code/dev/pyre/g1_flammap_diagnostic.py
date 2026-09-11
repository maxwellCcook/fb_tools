"""
G1 — one-shot FlamMap diagnostic for the Stage 1 de-stacker.

**Deliberately demoted from the plan's no-go gate to a diagnostic** (decision
2026-09-10). The 38 unit tests in ``tests/test_pyre_destack.py`` pin every unit
conversion *exactly* — tan(20 deg) = 0.3640 and not 0.20, CC 45 -> 0.45,
CH 150 -> 15.0 m, CBD 10 -> 0.10, plus raises on a fabricated FBFM40 code and on
a slope band that declares neither SlpD nor SlpP. That is a stronger check for
the silent-units class of bug than any raster comparison, because it is exact
rather than "within 20%".

What this script is still worth, then, is one thing the unit tests cannot give:
it separates **adapter error from weather-sampler error later**. If Stage 4's
simulated fire sizes come out 3x off against FPA-FOD, this run tells us whether
the landscape inputs were already wrong. Run it once, record the number, move on.

No tuning. No second pass. No iterating on residuals: matching a static model we
are replacing has poor returns past the sanity check.

Run:  conda activate fb_pyre && python code/dev/pyre/g1_flammap_diagnostic.py

Settings, all chosen to reproduce what the FlamMap p97 runs actually did
(confirmed with the user, 2026-09-10):

  wind direction   FlamMap ran WIND_DIRECTION: -1 (upslope everywhere).
                   pyretechnics has no upslope mode, so upwind_direction is a
                   2D cube = aspect: wind blowing upslope arrives FROM the
                   direction the slope faces.
  wind speed       FlamMap read WIND_SPEED: 24.5 as a 20-ft wind. pyretechnics
                   takes a 10 m wind and divides by 1.15, so we pass
                   24.5 * 1.15 mph in km/hr and its internal conversion lands
                   back on 24.5 mph at 20 ft.
                   NOTE, and it is not this port's problem to fix: the cache
                   populates WIND_SPEED from HRRR's *10 m* ws_mph, so the
                   FlamMap runs are ~15% hot on 20-ft wind. That is a finding
                   about the published in-situ Phase 1 results.
  foliar moisture  FlamMap ran FOLIAR_MOISTURE_CONTENT: 100 -> 1.00.
  fuel adjustment  FlamMap used no adjustment file, and COSWAP treatments are
                   stamped into the LCP bands -> 1.0 everywhere. Applying
                   canopy_effects.csv on top would double-count the treatment.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from fb_tools.spread.convert import destack_lcp
from fb_tools.models.pyre import pyre_version_info


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

COSWAP = Path(
    "/Users/mcc/Library/CloudStorage/Box-Box/CFRI/All Projects/COSWAP"
    "/Focus_Area_analysis/Analysis/fire_behavior"
)
LCP_DIR = COSWAP / "data/spatial/mod/transmission/lcp/North_Cottonwood"
FLAMMAP_OUT = COSWAP / "data/spatial/mod/flammap/outputs"

# Local scratch, outside Box: Box syncs every intermediate and a campaign writes
# tens of thousands of small rasters.
SCRATCH = Path(os.environ.get("FB_PYRE_SCRATCH", Path.home() / "fb_scratch/coswap_pyre"))
LAYER_DIR = SCRATCH / "layers/North_Cottonwood"

#: LCP arm -> the FlamMap output whose p97 raster was run on the same fuels.
#: 'b' scenarios are the canopy-thinning-only variants that 03_Transmitted.ipynb
#: uses; sc1b = CFT projects only, sc2b = CFT + COSWAP.
ARMS = {
    "untreated":  "baseline_lcp",
    "background": "lcp_sc1b",
    "coswap":     "lcp_sc2b",
}

#: North Cottonwood's pyrome, from the FSPro run manifest.
PYROME_ID = 45

#: The FlamMap scenario cache the p97 runs were driven from. Loaded rather than
#: transcribed: hardcoding it here once picked up pyrome 128's numbers by
#: accident (FM_herb 30 vs 45's 59.3), and nothing downstream would have caught it.
FLAMMAP_CACHE = Path(__file__).resolve().parents[3] / "data/weather/flammap_rtma"

FOLIAR_MOISTURE = 1.00          # FOLIAR_MOISTURE_CONTENT: 100
CROWN_FIRE_METHOD = "ScottReinhardt"   # scenarios.py default; what the runs used


def load_p97(pyrome_id=PYROME_ID):
    """The p97 scenario dict FlamMap was actually driven with, for this pyrome."""
    fp = FLAMMAP_CACHE / f"pyrome_{pyrome_id}_flammap.json"
    with open(fp) as fh:
        cache = json.load(fh)
    p97 = cache["scenarios"]["p97"]
    if p97.get("WIND_DIRECTION") != -1:
        raise ValueError(
            f"This script reproduces FlamMap's upslope wind (WIND_DIRECTION -1); "
            f"the cache says {p97.get('WIND_DIRECTION')}."
        )
    return p97


P97 = load_p97()

MPH_TO_KMH = 1.609344
WIND_10M_TO_20FT = 1.15

#: FlamMap CROWNSTATE -> pyretechnics fire_type. MEASURED on the North Cottonwood
#: window of baseline_lcp/p97, not assumed: CROWNSTATE takes values {0,1,2,3} with
#: median flame lengths 0.00 / 0.85 / 5.78 / 27.24 m, which is exactly
#: pyretechnics' 0=unburned, 1=surface, 2=passive_crown, 3=active_crown. The two
#: codings are identical and the map is the identity.
#:
#: This was wrong on the first pass (assumed FlamMap used 0=surface, 1=passive,
#: 2=active) and the off-by-one reported 78% crown fire against pyretechnics' 22%
#: -- a fake physics disagreement produced entirely by a lookup table. Exactly the
#: class of silent error the de-stacker's unit tests exist to prevent, arriving
#: through the one path they do not cover.
CROWNSTATE_TO_FIRE_TYPE = {0: 0, 1: 1, 2: 2, 3: 3}


def build_cubes(layer_paths, shape_hint=None):
    """Read the de-stacked layers and wrap them as pyretechnics SpaceTimeCubes.

    All static layers are 2D; weather is 0D except upwind_direction, which is a
    2D cube derived from aspect to reproduce FlamMap's upslope wind.
    """
    from pyretechnics.space_time_cube import SpaceTimeCube

    arrays = {}
    for name, fp in layer_paths.items():
        with rasterio.open(fp) as src:
            arrays[name] = src.read(1).astype("float32")
            profile = src.profile
            bounds = src.bounds

    rows, cols = arrays["slope"].shape
    shape = (1, rows, cols)   # one band: burn_cells is a static calculation

    ws_kmh = P97["WIND_SPEED"] * WIND_10M_TO_20FT * MPH_TO_KMH

    cubes = {
        "slope":               SpaceTimeCube(shape, arrays["slope"]),
        "aspect":              SpaceTimeCube(shape, arrays["aspect"]),
        "fuel_model":          SpaceTimeCube(shape, arrays["fuel_model"].astype("int64")),
        "canopy_cover":        SpaceTimeCube(shape, arrays["canopy_cover"]),
        "canopy_height":       SpaceTimeCube(shape, arrays["canopy_height"]),
        "canopy_base_height":  SpaceTimeCube(shape, arrays["canopy_base_height"]),
        "canopy_bulk_density": SpaceTimeCube(shape, arrays["canopy_bulk_density"]),
        "wind_speed_10m":      SpaceTimeCube(shape, np.float32(ws_kmh)),
        # Upslope wind: it arrives FROM the direction the slope faces.
        "upwind_direction":    SpaceTimeCube(shape, arrays["aspect"]),
        "fuel_moisture_dead_1hr":        SpaceTimeCube(shape, np.float32(P97["FM_1hr"] / 100.0)),
        "fuel_moisture_dead_10hr":       SpaceTimeCube(shape, np.float32(P97["FM_10hr"] / 100.0)),
        "fuel_moisture_dead_100hr":      SpaceTimeCube(shape, np.float32(P97["FM_100hr"] / 100.0)),
        "fuel_moisture_live_herbaceous": SpaceTimeCube(shape, np.float32(P97["FM_herb"] / 100.0)),
        "fuel_moisture_live_woody":      SpaceTimeCube(shape, np.float32(P97["FM_woody"] / 100.0)),
        "foliar_moisture":               SpaceTimeCube(shape, np.float32(FOLIAR_MOISTURE)),
    }
    return cubes, arrays, profile, bounds, ws_kmh


def read_flammap_window(scenario, bounds, ref_transform, ref_shape):
    """Window-read the FlamMap p97 stack onto the LCP footprint.

    Transforms are asserted equal rather than resampled — both grids are 30 m
    EPSG:5070, so any offset here is a real misalignment, not something to
    paper over with a resample.
    """
    fp = FLAMMAP_OUT / scenario / "p97" / f"{scenario}.tif"
    if not fp.exists():
        raise FileNotFoundError(fp)
    with rasterio.open(fp) as src:
        win = from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        wt = src.window_transform(win)
        if not np.allclose(
            [wt.a, wt.b, wt.c, wt.d, wt.e, wt.f],
            [ref_transform.a, ref_transform.b, ref_transform.c,
             ref_transform.d, ref_transform.e, ref_transform.f],
            atol=1e-6,
        ):
            raise ValueError(
                f"FlamMap window transform {wt} != LCP transform {ref_transform}. "
                "These grids must align exactly; resampling would hide a real offset."
            )
        names = list(src.descriptions)
        out = {}
        for band in ("CROWNSTATE", "FLAMELENGTH", "DX"):
            if band in names:
                arr = src.read(names.index(band) + 1, window=win, masked=True)
                out[band] = np.ma.filled(arr.astype("float64"), np.nan)
        if out["FLAMELENGTH"].shape != ref_shape:
            raise ValueError(
                f"FlamMap window {out['FLAMELENGTH'].shape} != LCP {ref_shape}"
            )
    return out


def compare(arm, pyre, fm, fuel):
    """Flame-length magnitude and crown-state agreement on co-burnable pixels."""
    burnable = ~np.isin(fuel.astype("int32"), [0, 91, 92, 93, 98, 99])

    pfl = pyre["flame_length"].astype("float64")
    ffl = fm["FLAMELENGTH"]

    both = burnable & np.isfinite(ffl) & (ffl > 0) & (pfl > 0)
    n = int(both.sum())
    if n == 0:
        print(f"  {arm}: no co-burnable pixels — comparison impossible")
        return {}

    a, b = pfl[both], ffl[both]
    ratio = a / b
    r = float(np.corrcoef(a, b)[0, 1])

    pft = pyre["fire_type"][both].astype("int32")
    fcs = fm["CROWNSTATE"][both]
    mapped = np.select(
        [fcs == k for k in CROWNSTATE_TO_FIRE_TYPE],
        [CROWNSTATE_TO_FIRE_TYPE[k] for k in CROWNSTATE_TO_FIRE_TYPE],
        default=0,
    ).astype("int32")
    agree = float((pft == mapped).mean())
    # Collapse to surface-vs-crown, which is the distinction the analysis uses.
    agree_binary = float(((pft >= 2) == (mapped >= 2)).mean())

    res = {
        "arm": arm,
        "n_pixels": n,
        "fl_pyre_median_m": float(np.median(a)),
        "fl_flammap_median_m": float(np.median(b)),
        "fl_median_ratio": float(np.median(ratio)),
        "fl_ratio_p10": float(np.percentile(ratio, 10)),
        "fl_ratio_p90": float(np.percentile(ratio, 90)),
        "fl_pearson_r": r,
        "crown_state_agreement": agree,
        "crown_binary_agreement": agree_binary,
        "pyre_crown_frac": float((pft >= 2).mean()),
        "flammap_crown_frac": float((mapped >= 2).mean()),
    }
    print(f"  {arm:11s} n={n:>9,}  FL median pyre {res['fl_pyre_median_m']:5.2f} m vs "
          f"FlamMap {res['fl_flammap_median_m']:5.2f} m  ratio {res['fl_median_ratio']:.2f} "
          f"[{res['fl_ratio_p10']:.2f}-{res['fl_ratio_p90']:.2f}]  r={r:.3f}")
    print(f"  {'':11s} crown: pyre {res['pyre_crown_frac']:.1%} vs FlamMap "
          f"{res['flammap_crown_frac']:.1%}  agree(surface/crown) "
          f"{res['crown_binary_agreement']:.1%}  agree(3-class) {res['crown_state_agreement']:.1%}")
    return res


def main():
    import pyretechnics.burn_cells as bc

    print(json.dumps(pyre_version_info(), indent=2))
    print(f"\nscratch: {SCRATCH}")
    print(f"p97 pyrome {PYROME_ID} (from the FlamMap cache): {P97}")
    print(f"crown fire: FlamMap {CROWN_FIRE_METHOD} vs pyretechnics' own CBH/CBD module")
    print(f"  -> wind_speed_10m = {P97['WIND_SPEED']} mph x {WIND_10M_TO_20FT} "
          f"x {MPH_TO_KMH:.4f} = "
          f"{P97['WIND_SPEED'] * WIND_10M_TO_20FT * MPH_TO_KMH:.2f} km/hr "
          f"(-> {P97['WIND_SPEED']:.1f} mph at 20 ft inside pyretechnics)\n")

    results = []
    for arm, scenario in ARMS.items():
        lcp = LCP_DIR / f"{arm}.tif"
        print(f"=== {arm}  ({lcp.name} vs FlamMap {scenario}/p97) ===")

        layers = destack_lcp(lcp, LAYER_DIR / arm, verbose=True)
        cubes, arrays, profile, bounds, ws = build_cubes(layers)

        t0 = time.perf_counter()
        pyre = bc.burn_all_cells_as_head_fire(cubes, t=0)
        print(f"  burn_all_cells_as_head_fire: {time.perf_counter() - t0:.1f}s "
              f"over {arrays['slope'].size:,} cells")

        fm = read_flammap_window(
            scenario, bounds, profile["transform"], arrays["slope"].shape
        )
        res = compare(arm, pyre, fm, arrays["fuel_model"])
        if res:
            res["scenario"] = scenario
            results.append(res)
        print()

    if results:
        out_csv = SCRATCH / "g1_flammap_diagnostic.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        print(f"wrote {out_csv}")

        ratios = [r["fl_median_ratio"] for r in results]
        worst = max(abs(np.log(x)) for x in ratios)
        print(f"\nG1 diagnostic: FL median ratio {min(ratios):.2f}-{max(ratios):.2f} "
              f"across {len(results)} arms")
        if worst > np.log(2.0):
            print("  >2x apart on at least one arm — that is a band or units error, "
                  "not a physics difference. STOP and find it before Stage 2.")
            return 1
        print("  Within 2x. Recorded; proceeding to Stage 1b / 2 without further "
              "reconciliation, per the timeboxed-diagnostic decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
