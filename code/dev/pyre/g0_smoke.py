"""
G0 — Stage 0 gate for the pyretechnics transmission port.

Proves, in the environment pip built the extension against, that:
  1. ``import pyretechnics.eulerian_level_set`` succeeds;
  2. a 200x200 synthetic fire runs to completion and burns cells;
  3. spotting is reachable AND mechanically active — the landscape carries a
     non-burnable barrier downwind of the ignition, so cells beyond it can only
     burn via an ember jump. Without this the check is vacuous: on an open
     landscape the fire reaches every spot-ignited cell anyway, so spotting
     leaves no trace;
  4. the engine is bit-for-bit reproducible at a fixed seed — the property the
     whole CRN design in Stage 3 rests on.

Run:  conda activate fb_pyre && python code/dev/pyre/g0_smoke.py

Units, taken from ``spread_fire_with_phi_field``'s own docstring (they differ
from the plan's draft table in two places, both noted here because each would
produce a plausible wrong answer):
  wind_speed_10m  km/hr   (NOT m/s and NOT mph)
  temperature     Celsius (NOT Fahrenheit)
  upwind_direction / aspect / spread_direction  degrees clockwise from North
  slope           rise/run
  time_of_arrival minutes
"""

import time

import numpy as np

from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.eulerian_level_set as els


GRID = 200
BANDS = 24          # 24 hourly bands
BAND_MINUTES = 60.0
CELL_M = 30.0
MAX_DURATION = 5 * 60.0   # minutes

# A non-burnable strip the head fire cannot cross. Anything burning east of it
# got there by ember. 6 cells @ 30 m = 180 m, beyond the fire's own reach but
# comfortably inside a plausible jump distance.
BARRIER_X0, BARRIER_X1 = 100, 106


def build_cubes(seed=0):
    """A synthetic heterogeneous landscape: sloping, patchy grass/timber fuel."""
    shape = (BANDS, GRID, GRID)
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:GRID, 0:GRID]

    # A 15% slope rising to the east, so spread is not radially symmetric.
    slope = np.full((GRID, GRID), 0.15, dtype="float32")
    aspect = np.full((GRID, GRID), 90.0, dtype="float32")   # faces east

    # Patchy fuels: GR2 (102) background with TU5 (165) timber patches.
    fuel = np.full((GRID, GRID), 102, dtype="int64")
    patch = ((xx // 25 + yy // 25) % 3) == 0
    fuel[patch] = 165

    # Non-burnable barrier (NB1 = 91) spanning the full height of the domain.
    barrier = (xx >= BARRIER_X0) & (xx < BARRIER_X1)
    fuel[barrier] = 91
    patch = patch & ~barrier

    canopy_cover = np.where(patch, 0.55, 0.0).astype("float32")
    canopy_height = np.where(patch, 18.0, 0.0).astype("float32")
    canopy_base_height = np.where(patch, 2.0, 0.0).astype("float32")
    canopy_bulk_density = np.where(patch, 0.12, 0.0).astype("float32")

    cubes = {
        "slope":               SpaceTimeCube(shape, slope),
        "aspect":              SpaceTimeCube(shape, aspect),
        "fuel_model":          SpaceTimeCube(shape, fuel),
        "canopy_cover":        SpaceTimeCube(shape, canopy_cover),
        "canopy_height":       SpaceTimeCube(shape, canopy_height),
        "canopy_base_height":  SpaceTimeCube(shape, canopy_base_height),
        "canopy_bulk_density": SpaceTimeCube(shape, canopy_bulk_density),
        # 1D hourly weather — the shape the Stage 4 cubes will take.
        "wind_speed_10m":                SpaceTimeCube(shape, np.full(BANDS, 30.0, dtype="float32")),   # km/hr
        "upwind_direction":              SpaceTimeCube(shape, np.full(BANDS, 270.0, dtype="float32")),  # FROM the west
        "fuel_moisture_dead_1hr":        SpaceTimeCube(shape, np.full(BANDS, 0.03, dtype="float32")),
        "fuel_moisture_dead_10hr":       SpaceTimeCube(shape, np.full(BANDS, 0.04, dtype="float32")),
        "fuel_moisture_dead_100hr":      SpaceTimeCube(shape, np.full(BANDS, 0.05, dtype="float32")),
        "fuel_moisture_live_herbaceous": SpaceTimeCube(shape, np.full(BANDS, 0.30, dtype="float32")),
        "fuel_moisture_live_woody":      SpaceTimeCube(shape, np.full(BANDS, 0.60, dtype="float32")),
        "foliar_moisture":               SpaceTimeCube(shape, 0.90),
        "temperature":                   SpaceTimeCube(shape, np.full(BANDS, 30.0, dtype="float32")),   # Celsius
    }
    del rng
    return shape, cubes


def run_once(shape, cubes, spot_config=None):
    state = els.SpreadState(shape).set_resolution((BAND_MINUTES, CELL_M, CELL_M)).set_start_time(0.0)
    state.ignite_cell((GRID // 2, GRID // 4))   # west-centre, so it runs upslope and downwind
    t0 = time.perf_counter()
    out = els.spread_fire_with_phi_field(
        cubes, state,
        max_duration=MAX_DURATION,
        spot_config=spot_config,
    )
    return out, time.perf_counter() - t0


def summarize(out, label, elapsed):
    state = out["spread_state"]
    m = state.get_full_matrices()
    burned = np.asarray(m["fire_type"]) > 0
    toa = np.asarray(m["time_of_arrival"])
    fl = np.asarray(m["flame_length"])
    n = int(burned.sum())
    ha = n * (CELL_M ** 2) / 10_000.0
    print(f"  {label:16s} {elapsed:6.2f}s  burned {n:6d} cells ({ha:8.1f} ha)  "
          f"stop={out['stop_condition']!r} t={out['stop_time']:.0f}min  "
          f"FL p95={np.nanpercentile(fl[burned], 95) if n else float('nan'):.2f}m  "
          f"crown={int((np.asarray(m['fire_type'])[burned] >= 2).sum())}  "
          f"max_x={int(np.nonzero(burned.any(axis=0))[0].max()) if n else -1}")
    return {"burned": burned, "toa": toa, "fl": fl,
            "fire_type": np.asarray(m["fire_type"]), "n": n,
            "spot_ignitions": out.get("spot_ignitions")}


def main():
    import pyretechnics
    print(f"pyretechnics {getattr(pyretechnics, '__version__', 'unknown')} "
          f"({pyretechnics.__file__})")
    print(f"grid {GRID}x{GRID} @ {CELL_M:.0f} m, {BANDS} bands x {BAND_MINUTES:.0f} min, "
          f"max_duration {MAX_DURATION:.0f} min\n")

    shape, cubes = build_cubes()

    # --- 1. no spotting -----------------------------------------------------
    out, el = run_once(shape, cubes, spot_config=None)
    a = summarize(out, "no spotting", el)
    assert a["n"] > 0, "G0 FAIL: no cells burned without spotting"
    beyond_a = int(a["burned"][:, BARRIER_X1:].sum())
    assert beyond_a == 0, (
        f"G0 FAIL: {beyond_a} cells burned past the barrier with spotting OFF — "
        "the barrier is not a barrier, so the spotting test below means nothing")

    # --- 2. spotting on -----------------------------------------------------
    spot_cfg = {
        "random_seed": 617327,
        # Deliberately high: this is a reachability check, not a calibration.
        # At the 5e-11 default, spot ignitions are rare enough on a 200x200 grid
        # that a single seed can produce none at all.
        "firebrands_per_unit_heat": 1e-8,
        "downwind_distance_mean": 11.7,
        "fireline_intensity_exponent": 0.3,
        "wind_speed_exponent": 0.6,
        "downwind_variance_mean_ratio": 600.0,
        "crosswind_distance_stdev": 10.0,
        "decay_distance": 500.0,
    }
    out_s1, el_s1 = run_once(shape, cubes, spot_config=spot_cfg)
    b = summarize(out_s1, "spotting seed A", el_s1)
    assert "random_generator" in out_s1, "G0 FAIL: spotting path not exercised"
    assert b["n"] > 0, "G0 FAIL: no cells burned with spotting"
    beyond_b = int(b["burned"][:, BARRIER_X1:].sum())
    print(f"  {'':16s} cells past the barrier: off={beyond_a}, on={beyond_b}")

    # --- 3. determinism at a fixed seed ------------------------------------
    out_s2, el_s2 = run_once(shape, cubes, spot_config=spot_cfg)
    c = summarize(out_s2, "spotting seed A'", el_s2)

    toa_same = np.array_equal(b["toa"], c["toa"], equal_nan=True)
    ft_same = np.array_equal(b["fire_type"], c["fire_type"])
    print(f"\n  repeat @ same seed : time_of_arrival identical={toa_same}  "
          f"fire_type identical={ft_same}")

    # --- 4. a different seed must actually differ --------------------------
    out_s3, el_s3 = run_once(shape, cubes, spot_config={**spot_cfg, "random_seed": 99991})
    d = summarize(out_s3, "spotting seed B", el_s3)
    differs = not np.array_equal(b["toa"], d["toa"], equal_nan=True)
    print(f"  different seed     : differs={differs}")

    spotting_works = beyond_b > 0
    ok = toa_same and ft_same and spotting_works and differs
    print("\n  determinism (same seed, bitwise) :", "PASS" if (toa_same and ft_same) else "FAIL")
    print("  spotting crosses the barrier     :",
          f"PASS ({beyond_b} cells)" if spotting_works else "FAIL (0 cells past barrier)")
    print("  a different seed changes the run :", "PASS" if differs else "FAIL")
    print("\nG0:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
