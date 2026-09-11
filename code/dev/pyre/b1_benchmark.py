"""
Stage 2 / G2 — what campaign can we afford, and at what resolution?

The question is NOT how pyretechnics compares to FSPro. It is what the analysis
budget buys. Mirrors ``code/dev/FSPro/p0_experiments.py`` in spirit.

Run:
    conda activate fb_pyre
    python code/dev/pyre/b1_benchmark.py                  # everything
    python code/dev/pyre/b1_benchmark.py --grid           # main factorial only
    python code/dev/pyre/b1_benchmark.py --knobs          # accuracy/runtime knobs
    python code/dev/pyre/b1_benchmark.py --ignitions      # ignition-to-ignition spread

Design notes, each of which changes the number
----------------------------------------------
**Every run is its own ``spawn`` subprocess.** ``ru_maxrss`` is a process
high-water mark that never decreases, so measuring several configs in one process
reports the largest, not each. Spawning also exercises the exact mechanism Stage 3
will parallelise with, early, where a GDAL/PROJ fork problem would be cheap to
find.

**Weather is a 1D hourly cube, not 0D.** Constant p97 values, but shaped as
``duration_days * 24`` bands so the engine's per-band cube-refresh machinery runs
exactly as it will in production. A 0D cube would skip that cost and flatter the
benchmark.

**Constant weather is conservative for runtime.** Real hourly HRRR/RTMA brings
overnight wind drop and fuel-moisture recovery, so fires burn less area and the
engine tracks fewer cells. Stage 4's episodes should run at or below these times.

**The crown gate is now a benchmark axis.** It was decided (2026-09-10) as a
reported sweep level, and it is worth 1.4x area / 5.7x fireline-intensity p95 on
this landscape, so it changes both cost and the campaign size. It also DOUBLES the
realization count the plan budgeted for — see ``report_budget``.

**Spot-ignition counts here are not a spotting deliverable.** The engine returns
``spot_ignitions`` as a schedule of *pending* future ignitions at stop time, not a
cumulative log, so it is recorded as ``pending_spot_cells`` with that caveat.
Proper accounting is Stage 7's ``summarize_spot_ignitions``.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path

import numpy as np
import rasterio

from fb_tools.spread.convert import destack_lcp

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

COSWAP = Path(
    "/Users/mcc/Library/CloudStorage/Box-Box/CFRI/All Projects/COSWAP"
    "/Focus_Area_analysis/Analysis/fire_behavior"
)
LCP_DIR = COSWAP / "data/spatial/mod/transmission/lcp/North_Cottonwood"
IGNITION_DIR = COSWAP / "data/spatial/mod/transmission/ignitions"

SCRATCH = Path(os.environ.get("FB_PYRE_SCRATCH", Path.home() / "fb_scratch/coswap_pyre"))
OUT_DIR = SCRATCH / "benchmark"

#: p97, pyrome 45 — loaded rather than transcribed (see g1_flammap_diagnostic).
FLAMMAP_CACHE = Path(__file__).resolve().parents[3] / "data/weather/flammap_rtma"
PYROME_ID = 45

#: The real modal fire-hour wind for pyrome 45, from the FSPro run manifest.
#: A spread run needs a real direction; G1's upslope wind is a static-comparison
#: device only.
WIND_FROM_DEG = 292.5
FOLIAR_MOISTURE = 0.90          # campaign value, not G1's FlamMap-matching 1.00
TEMPERATURE_C = 30.0            # required for spotting (Schroeder ignition prob)

MPH_TO_KMH = 1.609344
WIND_10M_TO_20FT = 1.15

#: The v1 campaign as decided 2026-09-11. Production runs at crown gate **0.0**
#: (crowning allowed below CC 0.4); the stock 0.4 gate is carried as an
#: **unbalanced sensitivity subset** rather than a full second axis.
#:
#: Why unbalanced. A balanced gate axis would double the campaign to 6,480. The
#: plan's own allocation rule is `flux >= ignitions > episodes > severity`, and
#: episodes are the cheapest axis to cut because CRN already collapses per-pixel
#: noise -- 30 episodes mostly tighten an interval pairing has narrowed, while 12
#: design fires remain the limiting inferential unit. So the subset keeps all 12
#: ignitions and cuts episodes and flux instead: +11% cost, and a paired,
#: clustered-by-ignition statement is still available at both gate settings.
CAMPAIGN = {
    "production": {
        "arms": 3, "ignitions": 12, "episodes": 30, "severity": 1,
        "flux_levels": 3, "crown_gate": 0.0,
    },
    "gate_sensitivity": {
        "arms": 3, "ignitions": 12, "episodes": 10, "severity": 1,
        "flux_levels": 1, "crown_gate": 0.4,
    },
}


def _n_realizations(block):
    return (block["arms"] * block["ignitions"] * block["episodes"]
            * block["severity"] * block["flux_levels"])

#: Stage 7's flux presets. "central" is pyretechnics' own default.
SPOT_PRESETS = {
    "off": None,
    "central": {
        "firebrands_per_unit_heat": 5e-11,
        "downwind_distance_mean": 11.7,
        "fireline_intensity_exponent": 0.3,
        "wind_speed_exponent": 0.6,
        "downwind_variance_mean_ratio": 600.0,
        "crosswind_distance_stdev": 10.0,
        "decay_distance": 500.0,
    },
}

BASE_SEED = 617327


def load_p97():
    with open(FLAMMAP_CACHE / f"pyrome_{PYROME_ID}_flammap.json") as fh:
        return json.load(fh)["scenarios"]["p97"]


# ---------------------------------------------------------------------------
# the worker — runs in a spawned child so its ru_maxrss is its own
# ---------------------------------------------------------------------------

def _run_one(spec, q):
    """One realization. Reports timings and outcome through *q*."""
    try:
        from fb_tools.models.pyre import set_crown_cover_threshold, pyre_version_info
        import pyretechnics.eulerian_level_set as els
        from pyretechnics.space_time_cube import SpaceTimeCube

        p97 = load_p97()
        res_m = spec["resolution"]

        t_setup = time.perf_counter()
        layers = destack_lcp(
            LCP_DIR / f"{spec['arm']}.tif",
            SCRATCH / "layers/North_Cottonwood" / (
                spec["arm"] if res_m == 30 else f"{spec['arm']}_{int(res_m)}m"),
            resolution=None if res_m == 30 else res_m,
            verbose=False,
        )
        arrays, transform, crs = {}, None, None
        for name, fp in layers.items():
            with rasterio.open(fp) as src:
                arrays[name] = src.read(1).astype("float32")
                transform, crs, nodata = src.transform, src.crs, src.nodata
        rows, cols = arrays["slope"].shape
        bands = spec["duration_days"] * 24
        shape = (bands, rows, cols)
        setup_s = time.perf_counter() - t_setup

        f = np.float32
        ws_kmh = f(p97["WIND_SPEED"] * WIND_10M_TO_20FT * MPH_TO_KMH)

        def hourly(value):
            """1D hourly cube so per-band refresh cost is measured, not skipped."""
            return SpaceTimeCube(shape, np.full(bands, value, dtype="float32"))

        cubes = {
            "slope":               SpaceTimeCube(shape, arrays["slope"]),
            "aspect":              SpaceTimeCube(shape, arrays["aspect"]),
            "fuel_model":          SpaceTimeCube(shape, arrays["fuel_model"].astype("int64")),
            "canopy_cover":        SpaceTimeCube(shape, arrays["canopy_cover"]),
            "canopy_height":       SpaceTimeCube(shape, arrays["canopy_height"]),
            "canopy_base_height":  SpaceTimeCube(shape, arrays["canopy_base_height"]),
            "canopy_bulk_density": SpaceTimeCube(shape, arrays["canopy_bulk_density"]),
            "wind_speed_10m":      hourly(ws_kmh),
            "upwind_direction":    hourly(WIND_FROM_DEG),
            "fuel_moisture_dead_1hr":        hourly(p97["FM_1hr"] / 100.0),
            "fuel_moisture_dead_10hr":       hourly(p97["FM_10hr"] / 100.0),
            "fuel_moisture_dead_100hr":      hourly(p97["FM_100hr"] / 100.0),
            "fuel_moisture_live_herbaceous": hourly(p97["FM_herb"] / 100.0),
            "fuel_moisture_live_woody":      hourly(p97["FM_woody"] / 100.0),
            "foliar_moisture":               SpaceTimeCube(shape, f(FOLIAR_MOISTURE)),
            "temperature":                   hourly(TEMPERATURE_C),
        }

        set_crown_cover_threshold(spec["crown_gate"])

        iy, ix = spec["ignition_yx"]
        spot_cfg = SPOT_PRESETS[spec["spotting"]]
        if spot_cfg is not None:
            spot_cfg = {**spot_cfg, "random_seed": spec["seed"]}

        state = (els.SpreadState(shape)
                 .set_resolution((60.0, res_m, res_m))
                 .set_start_time(0.0))
        state.ignite_cell((int(iy), int(ix)))

        t0 = time.perf_counter()
        out = els.spread_fire_with_phi_field(
            cubes, state,
            max_duration=spec["duration_days"] * 24 * 60.0,
            max_cells_per_timestep=spec["max_cells_per_timestep"],
            buffer_width=spec["buffer_width"],
            spot_config=spot_cfg,
        )
        wall_s = time.perf_counter() - t0

        m = out["spread_state"].get_full_matrices()
        ft = np.asarray(m["fire_type"])
        fl = np.asarray(m["flame_length"]).astype("float64")
        fi = np.asarray(m["fireline_intensity"]).astype("float64")
        toa = np.asarray(m["time_of_arrival"]).astype("float64")
        burned = ft > 0
        n_burned = int(burned.sum())
        cell_ha = res_m * res_m / 1e4

        pending = out.get("spot_ignitions") or {}
        pending_cells = sum(len(v) for v in pending.values())

        rec = {
            **{k: v for k, v in spec.items() if k != "ignition_yx"},
            "rows": rows, "cols": cols, "cells": rows * cols, "bands": bands,
            "setup_s": round(setup_s, 2),
            "wall_s": round(wall_s, 2),
            "peak_rss_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 3),
            "burned_ha": round(n_burned * cell_ha, 1),
            "burned_frac_domain": round(n_burned / (rows * cols), 4),
            "crown_frac": round(float((ft[burned] >= 2).mean()) if n_burned else 0.0, 4),
            "fl_p95_m": round(float(np.percentile(fl[burned], 95)) if n_burned else 0.0, 2),
            "fi_p95_kw_m": round(float(np.percentile(fi[burned], 95)) if n_burned else 0.0, 0),
            "pending_spot_cells": pending_cells,
            "stop_condition": out["stop_condition"],
            "stop_time_min": round(float(out["stop_time"]), 1),
            "num_tracked_cells": int(out["num_tracked_cells"]),
            "error": None,
        }

        # Domain adequacy, on this realization's burn mask. BP from a single
        # realization is the burn indicator, which is what the edge-mass and
        # headroom signals need; growth saturation comes from time_of_arrival.
        try:
            rec.update(_domain_checks(burned, toa, (iy, ix), transform, crs,
                                      res_m, spec["duration_days"]))
        except Exception as exc:                                   # pragma: no cover
            rec["domain_error"] = f"{type(exc).__name__}: {exc}"

        q.put(rec)
    except Exception as exc:
        import traceback
        q.put({**{k: v for k, v in spec.items() if k != "ignition_yx"},
               "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-2000:]})


def _domain_checks(burned, toa, ign_yx, transform, crs, res_m, duration_days):
    """Run the engine-agnostic domain-adequacy check on one realization.

    ``check_domain_adequacy`` was written for FSPro and never called while the
    burned footprint filled 72-92% of each axis. It needs a BP grid, an ignition
    footprint on that grid, and a daily-area frame. The first two are direct; the
    third is synthesized from ``time_of_arrival``, which is *exact* here rather
    than parsed from ``_DailyAcres.txt`` — a day's area is
    ``count(day-1 < toa/1440 <= day) * cell_area``.
    """
    import pandas as pd
    import xarray as xr
    from fb_tools.spread.fspro_outputs import check_domain_adequacy

    bp = xr.DataArray(
        burned.astype("float32"), dims=("y", "x"),
        coords={
            "y": transform.f + (np.arange(burned.shape[0]) + 0.5) * transform.e,
            "x": transform.c + (np.arange(burned.shape[1]) + 0.5) * transform.a,
        },
    )
    bp.rio.write_crs(crs, inplace=True)
    bp.rio.write_transform(transform, inplace=True)

    ign = np.zeros_like(burned, dtype=bool)
    ign[int(ign_yx[0]), int(ign_yx[1])] = True

    cell_ac = (res_m * res_m) / 4046.8564224
    days = np.ceil(np.where(np.isfinite(toa), toa, np.nan) / 1440.0)
    rows = []
    for d in range(1, duration_days + 1):
        rows.append({"fire_id": 0, "day": d,
                     "acres_day": float((days == d).sum()) * cell_ac})
    daily = pd.DataFrame(rows)
    daily["acres_cum"] = daily["acres_day"].cumsum()

    res = check_domain_adequacy(bp, ignition=ign, daily_acres=daily, verbose=False)
    return {
        "domain_passed": res["passed"],
        "edge_bp_fraction": round(float(res["edge_bp_fraction"]), 5),
        "edge_ok": res["edge_ok"],
        "spread_headroom": (None if res["spread_headroom"] is None
                            else round(float(res["spread_headroom"]), 3)),
        "headroom_ok": res["headroom_ok"],
        "max_spread_km": (None if res["max_spread_km"] is None
                          else round(float(res["max_spread_km"]), 2)),
        "final_day_growth_share": round(float(res["final_day_growth_share"]), 4),
        "growth_ok": res["growth_ok"],
    }


def run_spec(spec, timeout_s=3600):
    """Run one spec in a spawned child and return its record."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_one, args=(spec, q))
    t0 = time.perf_counter()
    p.start()
    try:
        rec = q.get(timeout=timeout_s)
    except Exception:
        p.terminate()
        p.join(10)
        return {**{k: v for k, v in spec.items() if k != "ignition_yx"},
                "error": f"timeout after {timeout_s}s"}
    p.join(30)
    rec["child_total_s"] = round(time.perf_counter() - t0, 2)
    return rec


# ---------------------------------------------------------------------------
# ignition selection
# ---------------------------------------------------------------------------

def pick_ignitions(n=3):
    """Real design-fire ignition cells, on the 30 m grid.

    Uses the Phase 2 density-weighted stratified design fires already on disk
    (``ign_000..002.shp``), so runtime reflects real ignition placement -- fires
    started in different fuels and terrain cost very different amounts. Falls back
    to the domain centre if the shapefiles are absent.
    """
    import geopandas as gpd

    layers = destack_lcp(LCP_DIR / "untreated.tif",
                         SCRATCH / "layers/North_Cottonwood/untreated", verbose=False)
    with rasterio.open(layers["fuel_model"]) as src:
        fuel = src.read(1).astype("int32")
        transform, crs = src.transform, src.crs
    burnable = ~np.isin(fuel, [0, 91, 92, 93, 98, 99])

    shps = sorted(IGNITION_DIR.glob("ign_*.shp"))[:n]
    picks = []
    for shp in shps:
        g = gpd.read_file(shp).to_crs(crs)
        cx, cy = g.geometry.union_all().centroid.coords[0]
        col = int((cx - transform.c) / transform.a)
        row = int((cy - transform.f) / transform.e)
        if not (0 <= row < fuel.shape[0] and 0 <= col < fuel.shape[1]):
            continue
        if not burnable[row, col]:
            # Snap to the nearest burnable cell rather than igniting rock.
            d = np.argwhere(burnable)
            k = np.argmin(np.abs(d[:, 0] - row) + np.abs(d[:, 1] - col))
            row, col = int(d[k][0]), int(d[k][1])
        picks.append((shp.stem, (row, col)))

    if not picks:
        r, c = fuel.shape[0] // 2, fuel.shape[1] // 2
        d = np.argwhere(burnable)
        k = np.argmin(np.abs(d[:, 0] - r) + np.abs(d[:, 1] - c))
        picks = [("centre", (int(d[k][0]), int(d[k][1])))]
    return picks


def scale_yx(yx, from_res, to_res):
    """Map a cell index between resolutions sharing an upper-left origin."""
    k = from_res / to_res
    return (int(yx[0] * k), int(yx[1] * k))


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------

def base_spec(**kw):
    spec = dict(arm="untreated", resolution=30, duration_days=7, spotting="central",
                crown_gate=0.4, max_cells_per_timestep=0.4, buffer_width=3,
                seed=BASE_SEED, ignition="ign_000")
    spec.update(kw)
    return spec


def main_grid(ign_name, ign_yx_30):
    """resolution x duration x spotting x crown gate = 16 runs."""
    specs = []
    for res_m in (90, 30):
        yx = ign_yx_30 if res_m == 30 else scale_yx(ign_yx_30, 30, res_m)
        for days in (7, 21):
            for spotting in ("off", "central"):
                for gate in (0.4, 0.0):
                    specs.append(base_spec(
                        resolution=res_m, duration_days=days, spotting=spotting,
                        crown_gate=gate, ignition=ign_name, ignition_yx=yx))
    return specs


def knob_grid(ign_name, ign_yx_30, resolution):
    """The two runtime/accuracy knobs, at the worst realistic config.

    ``max_cells_per_timestep`` is the CFL condition and ``buffer_width`` the
    narrow-band width: both trade accuracy for speed, so burned area is recorded
    beside runtime. A knob that halves runtime and moves area 0.2% is free; one
    that moves area 10% is not a knob, it is a different model.
    """
    yx = ign_yx_30 if resolution == 30 else scale_yx(ign_yx_30, 30, resolution)
    specs = []
    for mcpt in (0.2, 0.4, 0.8):
        for bw in (3, 6):
            specs.append(base_spec(
                resolution=resolution, duration_days=21, spotting="central",
                crown_gate=0.0, max_cells_per_timestep=mcpt, buffer_width=bw,
                ignition=ign_name, ignition_yx=yx))
    return specs


def ignition_grid(picks, resolution):
    """Ignition-to-ignition runtime spread -- what actually sets the campaign cost."""
    specs = []
    for name, yx30 in picks:
        yx = yx30 if resolution == 30 else scale_yx(yx30, 30, resolution)
        for gate in (0.4, 0.0):
            specs.append(base_spec(
                resolution=resolution, duration_days=21, spotting="central",
                crown_gate=gate, ignition=name, ignition_yx=yx))
    return specs


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

_COLS = ["resolution", "duration_days", "spotting", "crown_gate",
         "max_cells_per_timestep", "buffer_width", "ignition",
         "rows", "cols", "wall_s", "peak_rss_gb", "burned_ha",
         "burned_frac_domain", "crown_frac", "fl_p95_m", "fi_p95_kw_m",
         "pending_spot_cells", "stop_condition", "num_tracked_cells",
         "domain_passed", "edge_bp_fraction", "spread_headroom",
         "final_day_growth_share", "setup_s", "error"]


def print_rows(rows, title):
    print(f"\n{'=' * 118}\n{title}\n{'=' * 118}")
    hdr = (f"{'res':>4s} {'days':>5s} {'spot':>8s} {'gate':>5s} {'cfl':>5s} {'buf':>4s} "
           f"{'grid':>11s} {'wall s':>9s} {'RSS GB':>7s} {'burned ha':>11s} "
           f"{'%dom':>6s} {'crown':>7s} {'I p95':>8s} {'stop':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("error"):
            print(f"{r['resolution']:>4} {r['duration_days']:>5} {r['spotting']:>8s} "
                  f"{r['crown_gate']:>5.1f} {'':>5s} {'':>4s} {'':>11s} "
                  f"  ERROR  {r['error'][:60]}")
            continue
        grid = "{}x{}".format(r["rows"], r["cols"])
        print(f"{r['resolution']:>4} {r['duration_days']:>5} {r['spotting']:>8s} "
              f"{r['crown_gate']:>5.1f} {r['max_cells_per_timestep']:>5.1f} "
              f"{r['buffer_width']:>4} {grid:>11s} "
              f"{r['wall_s']:>9.1f} {r['peak_rss_gb']:>7.2f} {r['burned_ha']:>11,.0f} "
              f"{100*r['burned_frac_domain']:>5.1f}% {100*r['crown_frac']:>6.1f}% "
              f"{r['fi_p95_kw_m']:>8,.0f} {r['stop_condition'][:10]:>10s}")


def report_budget(rows):
    """G2 — translate s/realization into the campaign decision."""
    prod, sens = CAMPAIGN["production"], CAMPAIGN["gate_sensitivity"]
    n_prod, n_sens = _n_realizations(prod), _n_realizations(sens)
    n_with_gate = n_prod + n_sens

    print(f"\n{'=' * 118}\nG2 — campaign budget\n{'=' * 118}")
    print(f"  production      gate {prod['crown_gate']:.1f}: {prod['arms']} arms x "
          f"{prod['ignitions']} ign x {prod['episodes']} episodes x p97 x "
          f"{prod['flux_levels']} flux = {n_prod:,}")
    print(f"  gate sensitivity gate {sens['crown_gate']:.1f}: {sens['arms']} arms x "
          f"{sens['ignitions']} ign x {sens['episodes']} episodes x p97 x "
          f"{sens['flux_levels']} flux = {n_sens:,}")
    print(f"  total = {n_with_gate:,} realizations "
          f"(+{100 * n_sens / n_prod:.0f}% over production alone; a balanced gate axis "
          f"would have been {2 * n_prod:,})")

    ok = [r for r in rows if not r.get("error")]
    for res_m in (90, 30):
        cand = [r for r in ok if r["resolution"] == res_m and r["duration_days"] == 21
                and r["spotting"] == "central"]
        if not cand:
            continue
        worst = max(cand, key=lambda r: r["wall_s"])
        print(f"\n  {res_m} m, 21 d, spotting on, worst gate: "
              f"{worst['wall_s']:.1f} s/realization ({worst['crown_gate']:.1f} gate)")
        for workers in (1, 4, 8):
            hrs = n_with_gate * worst["wall_s"] / workers / 3600
            print(f"      {n_with_gate:,} realizations / {workers} worker(s): "
                  f"{hrs:7.1f} h ({hrs/24:5.1f} d)")

    g90 = [r for r in ok if r["resolution"] == 90 and r["duration_days"] == 21
           and r["spotting"] == "central"]
    if g90:
        s90 = max(r["wall_s"] for r in g90)
        print(f"\n  G2 branch, keyed on {s90:.1f} s/realization at 90 m (21 d, spotting on):")
        if s90 <= 30:
            print("    <= 30 s  -> PRODUCE AT 30 m. 30 m makes transmission commensurate")
            print("                with the 30 m in-situ FlamMap analysis; that is the prize.")
        elif s90 <= 120:
            print("    30-120 s -> produce at 90 m, keep 30 m for a single-ignition illustration.")
        elif s90 <= 600:
            print("    120-600 s-> hold flux and ignitions, cut episodes to ~15, and state the")
            print("                budget as the binding constraint.")
        else:
            print("    > 600 s  -> STOP and re-scope (shorter duration, smaller window, ELMFIRE).")


def write_csv(rows, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / name
    keys = list(dict.fromkeys(_COLS + [k for r in rows for k in r]))
    with open(fp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {fp}")
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--knobs", action="store_true")
    ap.add_argument("--ignitions", action="store_true")
    ap.add_argument("--production-res", type=int, default=30,
                    help="resolution for the knob and ignition suites")
    args = ap.parse_args()
    everything = not (args.grid or args.knobs or args.ignitions)

    from fb_tools.models.pyre import pyre_version_info
    print(json.dumps(pyre_version_info(), indent=2))

    picks = pick_ignitions(3)
    print(f"\nignition cells (30 m grid): {picks}")
    name0, yx0 = picks[0]

    all_rows = []
    if everything or args.grid:
        rows = [run_spec(s) for s in main_grid(name0, yx0)]
        print_rows(rows, "MAIN GRID — resolution x duration x spotting x crown gate")
        report_budget(rows)
        write_csv(rows, "b1_main_grid.csv")
        all_rows += rows

    if everything or args.knobs:
        rows = [run_spec(s) for s in knob_grid(name0, yx0, args.production_res)]
        print_rows(rows, f"KNOBS — max_cells_per_timestep x buffer_width @ "
                         f"{args.production_res} m, 21 d, spotting, gate 0.0")
        ok = [r for r in rows if not r.get("error")]
        if ok:
            ref = min(ok, key=lambda r: (abs(r["max_cells_per_timestep"] - 0.4),
                                         abs(r["buffer_width"] - 3)))
            print(f"\n  accuracy drift vs the default (cfl 0.4, buf 3): "
                  f"{ref['burned_ha']:,.0f} ha reference")
            for r in sorted(ok, key=lambda r: r["wall_s"]):
                d = 100 * (r["burned_ha"] - ref["burned_ha"]) / max(ref["burned_ha"], 1)
                sp = ref["wall_s"] / max(r["wall_s"], 1e-9)
                print(f"    cfl {r['max_cells_per_timestep']:.1f} buf {r['buffer_width']}: "
                      f"{r['wall_s']:7.1f} s ({sp:4.2f}x) area {d:+6.2f}%")
        write_csv(rows, "b1_knobs.csv")
        all_rows += rows

    if everything or args.ignitions:
        rows = [run_spec(s) for s in ignition_grid(picks, args.production_res)]
        print_rows(rows, f"IGNITIONS — real design fires @ {args.production_res} m, "
                         f"21 d, spotting")
        ok = [r for r in rows if not r.get("error")]
        if ok:
            w = [r["wall_s"] for r in ok]
            print(f"\n  wall s across {len(ok)} runs: min {min(w):.1f}, "
                  f"median {np.median(w):.1f}, max {max(w):.1f} "
                  f"({max(w)/max(min(w), 1e-9):.1f}x spread)")
            print("  Campaign cost is set by the MEAN, but a single worker's tail is set")
            print("  by the max -- chunk the pool so a slow ignition cannot serialise the run.")
        write_csv(rows, "b1_ignitions.csv")
        all_rows += rows

    if all_rows:
        write_csv(all_rows, "b1_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
