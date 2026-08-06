"""
Phase 0 verification experiments — P0.1, P0.2, P0.3.

**Run this on the Windows VM.**  ``TestFSPro.exe`` is Windows-only; everything
else in Phase 0 (the input validator, the domain-adequacy diagnostic) runs on
the Mac.  The script writes ``p0_results.json`` next to the run directories;
copy that back to the Mac and the numbers can be read without re-running.

Each experiment answers a question that gates later phases.

P0.1 — seed determinism  *(highest-value single experiment)*
    Run the same LCP with the same input file twice and diff ``_BurnProb.asc``.
    The spec calls ``SPOTTING_SEED`` "the seed used to initialize the random
    number generator for spotting" — it may not seed the ERC stream generator
    or the wind draws.

    * **Bit-identical** — common random numbers hold, the paired difference
      cancels weather variance, and ``NumFires`` can stay near 1,000.
    * **Differs** — pairing is statistical, not exact; ``NumFires`` must rise
      to 4,000+ and every reported delta needs a null band.  The percentiles of
      the run-to-run |ΔBP| distribution *are* that null band: **no treatment
      delta below the noise floor is reportable.**

    A third run with a different seed shows how much of the variation the seed
    controls at all.

P0.2 — runtime budget
    Wall-clock across a ``Duration`` × ``NumFires`` grid, cheapest first, so
    the cost of the production parameterisation can be extrapolated to the
    24–30 paired runs Phase 4 needs.  ``ThreadsPerFire`` is deprecated and
    always 1, so process-level parallelism is the only lever.

P0.3 — ignition semantics
    ``IgnitionFile`` is a *starting fire perimeter*, not a sampling domain:
    burn probability is ~1.0 everywhere inside it by construction.  Re-runs the
    416 sample with circular ignitions of 1 / 10 / 100 acres in place of the
    vendor's 630-acre IR perimeter, and reports the interior burn-probability
    saturation and the resulting fire-size distribution for each.  Decides
    point-ignition vs day-1-perimeter (P2.4).

Usage
-----
::

    python p0_experiments.py                        # everything
    python p0_experiments.py --experiments p01      # one experiment
    python p0_experiments.py --dry-run              # build + validate, no runs
    python p0_experiments.py --num-fires 500        # cheaper determinism test

The runtime grid is ordered cheapest-first and can be cut short at any point;
whatever finished is already in ``p0_results.json``.
"""

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Repo root: .../FM_PythonWrapper/code/dev/FSPro/p0_experiments.py
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fb_tools.models.fspro import run_fspro                        # noqa: E402
from fb_tools.models.fspro_validate import (                       # noqa: E402
    assert_valid_fspro_input,
    parse_fspro_input,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

VENDOR_DIR = REPO_ROOT / "code" / "FB" / "TestFSPro" / "SampleData"
VENDOR_INPUT = VENDOR_DIR / "416inputsfile.input"
VENDOR_LCP = VENDOR_DIR / "416lcp.lcp"
VENDOR_IGN = VENDOR_DIR / "416ign.shp"

# The FB toolchain ships as one bin/ directory holding every executable plus the
# DLLs they link against; TestFSPro.exe must stay beside them.  This matches the
# vendor's own RunFSPro.bat, which invokes ``..\..\bin\TestFSPro``.
DEFAULT_EXE = REPO_ROOT / "code" / "FB" / "bin" / "TestFSPro.exe"

WORK_DIR = REPO_ROOT / "data" / "fspro_test" / "p0_experiments"
RUNS_DIR = WORK_DIR / "runs"
RESULTS_JSON = WORK_DIR / "p0_results.json"

# Circular ignitions built on the Mac by the P0.3 preparation step.
ALT_IGNITIONS = {
    "perimeter_630ac": VENDOR_IGN,
    "circle_100ac":    WORK_DIR / "416ign_100ac.shp",
    "circle_10ac":     WORK_DIR / "416ign_10ac.shp",
    "circle_1ac":      WORK_DIR / "416ign_halfpixel.shp",
}

# Baseline parameterisation shared by P0.1 and P0.3.
BASE_DURATION = 7
BASE_NUM_FIRES = 1000
SEED_A = 617327
SEED_B = 987654

# P0.2 grid, cheapest first.  (Duration, NumFires)
RUNTIME_GRID = [(7, 1000), (21, 1000), (7, 4000), (21, 4000)]

# Phase 4 needs 3 arms x 8-12 design fires.
PHASE4_RUNS = 30


# ── Input file construction ───────────────────────────────────────────────────

def _set_switch(text: str, switch: str, value) -> str:
    """Replace the value of a scalar switch, preserving line order."""
    out, found = [], False
    for line in text.splitlines():
        if line.startswith(f"{switch}:"):
            out.append(f"{switch}: {value}")
            found = True
        else:
            out.append(line)
    if not found:
        raise KeyError(f"switch {switch!r} not present in the input file")
    return "\n".join(out) + "\n"


def build_variant(name, *, duration, num_fires, seed, ignition, out_dir):
    """
    Write a validated variant of the vendor input file.

    ``IgnitionFile`` is written as an **absolute** path.  The vendor file ships
    with ``.\\416ign.shp``, which only resolves when FSPro is launched from
    ``SampleData``; :func:`run_fspro` sets the working directory to the output
    directory instead, so a relative path would silently fail to load.
    """
    text = VENDOR_INPUT.read_text()
    text = _set_switch(text, "Duration", duration)
    text = _set_switch(text, "NumFires", num_fires)
    text = _set_switch(text, "SPOTTING_SEED", seed)
    text = _set_switch(text, "IgnitionFile", str(Path(ignition).resolve()))

    # NumForecast must stay <= Duration - 1; the vendor file carries 3 rows.
    parsed_dur = duration
    if parsed_dur - 1 < 3:
        raise ValueError(
            f"Duration={duration} cannot carry the vendor's 3 forecast rows"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.input"
    path.write_text(text)

    warnings = assert_valid_fspro_input(path, warn=False)
    return path, warnings


# ── Output reading (numpy only — no GDAL dependency on the VM) ────────────────

def read_asc(path: Path):
    """Read an ESRI ASCII grid into ``(array, header dict)``.

    Nodata cells become ``NaN``.  Deliberately dependency-free so this script
    runs even if the VM's GDAL stack is unhappy.
    """
    header, values = {}, []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            key = parts[0].lower()
            if key in ("ncols", "nrows", "xllcorner", "yllcorner", "xllcenter",
                       "yllcenter", "cellsize", "nodata_value"):
                header[key] = float(parts[1])
                continue
            values.append([float(v) for v in parts])
    arr = np.asarray(values, dtype="float64")
    nodata = header.get("nodata_value")
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, header


def read_daily_acres_simple(path: Path):
    """Per-fire total acres from ``_DailyAcres.txt`` without pandas."""
    totals, current, last_day = [], 0.0, 0
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            day, acres = int(float(parts[0])), float(parts[1])
        except ValueError:
            continue
        if day <= last_day and current > 0:
            totals.append(current)
            current = 0.0
        current += acres
        last_day = day
    if current > 0:
        totals.append(current)
    return totals


def fire_size_stats(run_dir: Path, basename: str):
    """Summarize the simulated fire-size distribution, or ``None`` if absent."""
    path = run_dir / f"{basename}_DailyAcres.txt"
    if not path.exists():
        return None
    sizes = read_daily_acres_simple(path)
    if not sizes:
        return None
    sizes_sorted = sorted(sizes)
    return {
        "n_fires":   len(sizes),
        "mean_ac":   statistics.mean(sizes),
        "median_ac": statistics.median(sizes),
        "p90_ac":    sizes_sorted[int(0.90 * (len(sizes) - 1))],
        "max_ac":    max(sizes),
        "mean_ha":   statistics.mean(sizes) * 0.404686,
    }


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_bp(path_a: Path, path_b: Path) -> dict:
    """
    Diff two burn-probability grids and characterise the difference.

    The percentiles of ``|ΔBP|`` over cells burned in either run are the noise
    floor: a treatment effect smaller than this cannot be distinguished from
    run-to-run stochasticity.
    """
    a, _ = read_asc(path_a)
    b, _ = read_asc(path_b)
    if a.shape != b.shape:
        return {"error": f"shape mismatch {a.shape} vs {b.shape}"}

    a0 = np.nan_to_num(a, nan=0.0)
    b0 = np.nan_to_num(b, nan=0.0)
    diff = a0 - b0
    burned = (a0 > 0) | (b0 > 0)
    d_burned = np.abs(diff[burned])

    out = {
        "identical":          bool(np.array_equal(a0, b0)),
        "n_cells_burned":     int(burned.sum()),
        "n_cells_differing":  int((diff != 0).sum()),
        "frac_cells_differing": (
            float((diff[burned] != 0).mean()) if burned.any() else 0.0
        ),
        "max_abs_delta":      float(d_burned.max()) if burned.any() else 0.0,
        "mean_abs_delta":     float(d_burned.mean()) if burned.any() else 0.0,
        "rms_delta":          float(np.sqrt((diff[burned] ** 2).mean()))
                              if burned.any() else 0.0,
        "expected_area_delta_frac": (
            float(abs(a0.sum() - b0.sum()) / a0.sum()) if a0.sum() > 0 else 0.0
        ),
    }
    if burned.any():
        for q in (50, 90, 95, 99):
            out[f"abs_delta_p{q}"] = float(np.percentile(d_burned, q))
    return out


def ignition_saturation(bp_path: Path, ign_asc: Path) -> "dict | None":
    """
    Burn-probability saturation inside the ignition footprint.

    FSPro rasterizes the ignition to ``_Ignitions.asc`` on the output grid, so
    no reprojection is needed.  If nearly every interior cell sits at BP ≈ 1,
    the ignition is a starting perimeter, not a sampling domain — and every
    delta statistic must mask it out, since it is structurally zero-delta.
    """
    if not ign_asc.exists() or not bp_path.exists():
        return None
    bp, _ = read_asc(bp_path)
    ign, _ = read_asc(ign_asc)
    if bp.shape != ign.shape:
        return None

    inside = np.nan_to_num(ign, nan=0.0) > 0
    if not inside.any():
        return None
    bp0 = np.nan_to_num(bp, nan=0.0)
    grid_burned = bp0 > 0

    return {
        "ignition_cells":     int(inside.sum()),
        "ignition_acres":     float(inside.sum()) * (90.0 * 90.0) / 4046.86,
        "frac_bp_ge_0999":    float((bp0[inside] >= 0.999).mean()),
        "mean_bp_inside":     float(bp0[inside].mean()),
        "grid_frac_bp_ge_0999": float((bp0[grid_burned] >= 0.999).mean())
                                if grid_burned.any() else 0.0,
    }


# ── Run driver ────────────────────────────────────────────────────────────────

def do_run(exe, name, *, duration, num_fires, seed, ignition, dry_run):
    """Build, validate, and execute one configuration.  Returns a result dict."""
    run_dir = RUNS_DIR / name
    input_path, warnings = build_variant(
        name, duration=duration, num_fires=num_fires, seed=seed,
        ignition=ignition, out_dir=run_dir,
    )

    record = {
        "name": name,
        "duration": duration,
        "num_fires": num_fires,
        "seed": seed,
        "ignition": str(Path(ignition).name),
        "input_file": str(input_path),
        "run_dir": str(run_dir),
        "validator_warnings": warnings,
    }

    if dry_run:
        record["status"] = "dry-run"
        print(f"  [{name}] validated, not run "
              f"({len(warnings)} warning(s))")
        return record

    print(f"\n[{name}] Duration={duration} NumFires={num_fires} seed={seed} "
          f"ignition={Path(ignition).name}")
    t0 = time.time()
    try:
        proc = run_fspro(
            fspro_exe=exe,
            lcp_fp=VENDOR_LCP,
            input_file=input_path,
            output_directory=run_dir,
            output_basename=name,
            num_fires_warn=0,
            verbose=True,
        )
        record["returncode"] = int(getattr(proc, "returncode", -1))
        record["status"] = "ok" if record["returncode"] == 0 else "nonzero-returncode"
    except Exception as exc:
        record["status"] = f"error: {exc}"
        record["returncode"] = None

    record["wall_seconds"] = time.time() - t0
    record["bp_path"] = str(run_dir / f"{name}_BurnProb.asc")
    record["fire_sizes"] = fire_size_stats(run_dir, name)
    record["ignition_saturation"] = ignition_saturation(
        run_dir / f"{name}_BurnProb.asc", run_dir / f"{name}_Ignitions.asc"
    )
    print(f"  [{name}] {record['status']} in {record['wall_seconds']:.0f}s")
    return record


# ── Experiments ───────────────────────────────────────────────────────────────

def experiment_p01(exe, args, results):
    """Seed determinism and the run-to-run noise floor."""
    print("\n" + "=" * 78)
    print("P0.1  Seed determinism  —  does SPOTTING_SEED give common random numbers?")
    print("=" * 78)

    n = args.num_fires
    runs = {
        "p01_seedA_run1": SEED_A,
        "p01_seedA_run2": SEED_A,
        "p01_seedB":      SEED_B,
    }
    for name, seed in runs.items():
        results["runs"][name] = do_run(
            exe, name, duration=BASE_DURATION, num_fires=n, seed=seed,
            ignition=VENDOR_IGN, dry_run=args.dry_run,
        )
    if args.dry_run:
        return

    bp = {k: Path(results["runs"][k]["bp_path"]) for k in runs}
    same_seed = compare_bp(bp["p01_seedA_run1"], bp["p01_seedA_run2"])
    diff_seed = compare_bp(bp["p01_seedA_run1"], bp["p01_seedB"])

    results["p01"] = {
        "num_fires": n,
        "duration": BASE_DURATION,
        "same_seed": same_seed,
        "different_seed": diff_seed,
    }

    print("\n  --- P0.1 result ---")
    if same_seed.get("identical"):
        print("  Two runs at the SAME seed are BIT-IDENTICAL.")
        print("  => Common random numbers hold. The paired baseline/treated")
        print(f"     difference cancels weather variance; NumFires can stay near {n}.")
        print("     Noise floor is exactly zero — any non-zero delta is signal.")
    else:
        floor = same_seed.get("abs_delta_p95", float("nan"))
        print("  Two runs at the SAME seed DIFFER.")
        print(f"    cells differing : {same_seed['frac_cells_differing'] * 100:.2f}% "
              "of burned cells")
        print(f"    |dBP| p50/p95/p99/max : "
              f"{same_seed.get('abs_delta_p50', float('nan')):.4f} / "
              f"{floor:.4f} / "
              f"{same_seed.get('abs_delta_p99', float('nan')):.4f} / "
              f"{same_seed['max_abs_delta']:.4f}")
        print(f"    expected area burned differs by "
              f"{same_seed['expected_area_delta_frac'] * 100:.2f}%")
        print("  => Pairing is statistical, not exact. Raise NumFires to 4000+")
        print(f"     and report every treatment delta against a null band of "
              f"~{floor:.4f} BP.")
        n_needed = n * (floor / 0.01) ** 2 if floor > 0 else n
        print(f"     To resolve a 0.01 BP effect at this floor, NumFires must be "
              f"roughly {n_needed:,.0f} (variance scales as 1/N).")
    print(f"  A DIFFERENT seed changes |dBP| p95 to "
          f"{diff_seed.get('abs_delta_p95', float('nan')):.4f} "
          f"(max {diff_seed['max_abs_delta']:.4f}) — how much the seed controls.")


def experiment_p02(exe, args, results):
    """Wall-clock across the Duration x NumFires grid."""
    print("\n" + "=" * 78)
    print("P0.2  Runtime budget  —  what does the production parameterisation cost?")
    print("=" * 78)

    rows = []
    for duration, num_fires in RUNTIME_GRID:
        name = f"p02_d{duration}_n{num_fires}"
        rec = do_run(
            exe, name, duration=duration, num_fires=num_fires, seed=SEED_A,
            ignition=VENDOR_IGN, dry_run=args.dry_run,
        )
        results["runs"][name] = rec
        if not args.dry_run:
            rows.append((duration, num_fires, rec.get("wall_seconds"),
                         rec.get("status"), rec.get("fire_sizes")))

    if args.dry_run or not rows:
        return

    results["p02"] = {
        "grid": [
            {"duration": d, "num_fires": n, "wall_seconds": w, "status": s,
             "mean_fire_ac": (f or {}).get("mean_ac")}
            for d, n, w, s, f in rows
        ],
        "phase4_runs": PHASE4_RUNS,
    }

    print("\n  --- P0.2 result ---")
    print(f"  {'Duration':>8} {'NumFires':>9} {'wall (s)':>10} {'wall (min)':>11} "
          f"{'mean fire (ac)':>15} {'x' + str(PHASE4_RUNS) + ' runs (h)':>16}")
    for d, n, w, s, f in rows:
        if w is None:
            continue
        mean_ac = (f or {}).get("mean_ac")
        print(f"  {d:>8} {n:>9} {w:>10.0f} {w / 60:>11.1f} "
              f"{(f'{mean_ac:,.0f}' if mean_ac else '-'):>15} "
              f"{w * PHASE4_RUNS / 3600:>16.1f}")
    print("  Note: ThreadsPerFire is deprecated (always 1), so process-level")
    print("  parallelism across runs is the only lever. Divide the last column")
    print("  by the number of concurrent VM processes.")


def experiment_p03(exe, args, results):
    """Ignition semantics — perimeter vs small circles."""
    print("\n" + "=" * 78)
    print("P0.3  Ignition semantics  —  is IgnitionFile a perimeter or a domain?")
    print("=" * 78)

    missing = [k for k, v in ALT_IGNITIONS.items() if not Path(v).exists()]
    if missing:
        print(f"  Missing ignition shapefiles: {missing}")
        print("  Build them on the Mac first (see the P0.3 preparation step).")
        return

    rows = []
    for label, ign in ALT_IGNITIONS.items():
        name = f"p03_{label}"
        rec = do_run(
            exe, name, duration=BASE_DURATION, num_fires=args.num_fires,
            seed=SEED_A, ignition=ign, dry_run=args.dry_run,
        )
        results["runs"][name] = rec
        if not args.dry_run:
            rows.append((label, rec.get("ignition_saturation"),
                         rec.get("fire_sizes")))

    if args.dry_run or not rows:
        return

    results["p03"] = [
        {"ignition": label, "saturation": sat, "fire_sizes": sizes}
        for label, sat, sizes in rows
    ]

    print("\n  --- P0.3 result ---")
    print(f"  {'ignition':>16} {'ign ac':>9} {'BP>=0.999 inside':>18} "
          f"{'mean BP inside':>15} {'mean fire ac':>13} {'median ac':>11}")
    for label, sat, sizes in rows:
        if sat is None:
            print(f"  {label:>16}   (no _Ignitions.asc written)")
            continue
        print(f"  {label:>16} {sat['ignition_acres']:>9.1f} "
              f"{sat['frac_bp_ge_0999'] * 100:>17.1f}% "
              f"{sat['mean_bp_inside']:>15.3f} "
              f"{(sizes or {}).get('mean_ac', float('nan')):>13,.0f} "
              f"{(sizes or {}).get('median_ac', float('nan')):>11,.0f}")
    print("  Grid-wide BP >= 0.999 for reference: "
          + ", ".join(
              f"{label} {sat['grid_frac_bp_ge_0999'] * 100:.2f}%"
              for label, sat, _ in rows if sat
          ))
    print("  If saturation inside the footprint stays near 100% at every size,")
    print("  IgnitionFile is unambiguously a starting perimeter: mask it out of")
    print("  every delta statistic, and never use a whole container as ignition.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE,
                    help="path to TestFSPro.exe")
    ap.add_argument("--experiments", nargs="+", default=["p01", "p02", "p03"],
                    choices=["p01", "p02", "p03"],
                    help="which experiments to run (default: all)")
    ap.add_argument("--num-fires", type=int, default=BASE_NUM_FIRES,
                    help=f"NumFires for P0.1 and P0.3 (default {BASE_NUM_FIRES})")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate the input files without running FSPro")
    ap.add_argument("--results", type=Path, default=RESULTS_JSON,
                    help="where to write the JSON results")
    args = ap.parse_args(argv)

    if not args.dry_run and platform.system() != "Windows":
        ap.error(
            "TestFSPro.exe is Windows-only — run this on the Parallels VM, or "
            "pass --dry-run to build and validate the input files on the Mac."
        )
    if not VENDOR_INPUT.exists():
        ap.error(f"vendor sample input not found: {VENDOR_INPUT}")
    if not args.dry_run and not args.exe.exists():
        ap.error(f"TestFSPro.exe not found: {args.exe} (pass --exe)")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    vendor = parse_fspro_input(VENDOR_INPUT)
    results = {
        "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exe": str(args.exe),
        "lcp": str(VENDOR_LCP),
        "vendor_input": str(VENDOR_INPUT),
        "vendor_scalars": {
            k: vendor.get(k) for k in
            ("Resolution", "Duration", "NumFires", "MaxLag", "PolyDegree",
             "SPOTTING_SEED", "CROWN_FIRE_METHOD", "NumWxCurrYear", "NumForecast")
        },
        "runs": {},
    }

    dispatch = {"p01": experiment_p01, "p02": experiment_p02, "p03": experiment_p03}
    try:
        for key in args.experiments:
            dispatch[key](args.exe, args, results)
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results.")
    finally:
        results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults written to {args.results}")
        print("Copy this file back to the Mac — the Phase 0 conclusions read "
              "off it without re-running.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
