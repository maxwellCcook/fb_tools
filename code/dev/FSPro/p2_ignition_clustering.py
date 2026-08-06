"""
P2.2 — is Ager's uniform-ignition assumption defensible for Colorado?

Ager et al. 2014 (For. Ecol. Manage. 334:377-390) placed ignitions uniformly at
random, reporting *"no evidence of spatial correlation in the ignition locations
of large fires"* and describing them as *"lightning caused and randomly
located"*.  Our Part A design weights design fires by an FPA-FOD ignition
density surface instead.  That is only worth doing — and only reportable as a
methodological improvement — if the uniform assumption actually fails here.

This script tests it three ways: all large-fire ignitions, the Natural
(lightning) subset that Ager's wording describes, and the Human subset.

The null is complete spatial randomness restricted to the CO pyrome polygons.
It is deliberately *not* restricted to burnable fuel, because no pyrome-wide
FBFM40 raster is on disk; `check_ignition_clustering(mask_burnable=True,
lcp_fp=...)` runs the stronger test once one exists.  Some of the measured
clustering is therefore fuel availability rather than ignition process — state
that caveat when reporting.

Run (Mac, ~2 min):
    python code/dev/FSPro/p2_ignition_clustering.py

Results as of 2026-08-05 are committed alongside as
``p2_ignition_clustering.json`` — ``data/`` is gitignored, so that file is the
durable record.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd

from fb_tools.fuelscape.ignitions import check_ignition_clustering

REPO_ROOT = Path(__file__).resolve().parents[3]

#: FPA-FOD 20260615 — the full record through 2024.  The copies under
#: ``data/spatial/raw/fpa_fod/`` all stop at 2022 (or 2009 for the truncated
#: shapefile export), so this external path is the authoritative source.
FOD_GPKG = Path(
    "/Users/mcc/Library/CloudStorage/Box-Box/MCC/data/wildfire/FPA_FOD/"
    "RDS-2013-0009/Data/FPA_FOD_20260615.gpkg"
)
PYROMES = REPO_ROOT / "data" / "spatial" / "raw" / "boundaries" / "CO_Pyromes.gpkg"
OUT_JSON = Path(__file__).with_suffix(".json")

#: Pyrome polygons spill into neighbouring states, so pull a wider net than CO.
STATES = ("CO", "WY", "UT", "NM", "KS", "NE", "OK")
RADII_M = [5_000, 10_000, 25_000, 50_000]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sim", type=int, default=199,
                    help="Monte-Carlo realizations (199 → p floor 0.01)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fod", type=Path, default=FOD_GPKG)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    if not args.fod.exists():
        raise SystemExit(f"FPA-FOD not found: {args.fod}")

    pyromes = gpd.read_file(PYROMES)
    states = ", ".join(f"'{s}'" for s in STATES)
    fod = gpd.read_file(
        args.fod, layer="Fires",
        where=(f"FIRE_SIZE_CLASS IN ('D','E','F','G') AND STATE IN ({states})"),
        columns=["FOD_ID", "FIRE_YEAR", "FIRE_SIZE", "FIRE_SIZE_CLASS",
                 "NWCG_CAUSE_CLASSIFICATION", "NWCG_GENERAL_CAUSE"],
    )
    print(f"Loaded {len(fod):,} Class D-G ignitions, "
          f"{fod.FIRE_YEAR.min()}-{fod.FIRE_YEAR.max()}")

    subsets = {
        "all":     fod,
        "natural": fod[fod.NWCG_CAUSE_CLASSIFICATION == "Natural"],
        "human":   fod[fod.NWCG_CAUSE_CLASSIFICATION == "Human"],
    }

    results = {}
    for name, sub in subsets.items():
        print(f"\n{'=' * 62}\n{name.upper()}")
        r = check_ignition_clustering(
            sub, domain_gdf=pyromes, n_sim=args.n_sim, seed=args.seed,
            mask_burnable=False, radii_m=RADII_M,
        )
        r["nn_ratio_obs_over_null"] = r["mean_nn_m"] / r["null_mean_nn_m"]
        results[name] = r
        print(f"  mean NN {r['mean_nn_m']:,.0f} m vs null "
              f"{r['null_mean_nn_m']:,.0f} m "
              f"(ratio {r['nn_ratio_obs_over_null']:.3f}), "
              f"p={r['nn_p_value']:.4f} → {r['verdict']}")
        for rr, l, lo, hi in zip(r["radii_m"], r["L_minus_r"],
                                 r["L_null_lo"], r["L_null_hi"]):
            flag = "OUTSIDE" if (l < lo or l > hi) else "inside "
            print(f"    L(r)-r  r={rr / 1000:5.0f} km  {l:+9,.0f}  "
                  f"null[{lo:+8,.0f}, {hi:+8,.0f}]  {flag}")

    payload = {
        "source":      str(args.fod),
        "domain":      "CO pyromes (CO_Pyromes.gpkg, 9 polygons)",
        "size_class":  "D-G",
        "years":       [int(fod.FIRE_YEAR.min()), int(fod.FIRE_YEAR.max())],
        "n_sim":       args.n_sim,
        "seed":        args.seed,
        "null":        "CSR over pyrome polygons; NOT restricted to burnable fuel",
        "results":     results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
