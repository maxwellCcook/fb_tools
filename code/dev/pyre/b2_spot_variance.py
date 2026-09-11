"""
Is the spotting "breakout" solver fragility, or a real heavy tail?

Stage 2's knob sweep threw up something that has to be resolved before Stage 3,
because it decides whether the primary deliverable is trustworthy. At 30 m, 21 d,
spotting on, crown gate 0.0, ignition ign_000:

    cfl 0.2  buf 3   59,372 ha     cfl 0.2  buf 6   32,557 ha
    cfl 0.4  buf 3   59,389 ha     cfl 0.4  buf 6   32,553 ha
    cfl 0.8  buf 3   32,548 ha     cfl 0.8  buf 6   32,551 ha

The 59,000 ha answer appears at `buf 3` with `cfl <= 0.4` and nowhere else, and
32,550 ha is exactly the no-spotting answer. So the entire +83% spotting effect
switches on and off with solver settings. Worse, in the ignition suite at
identical settings, ign_000 broke out to 59,389 ha while ign_001 and ign_002 gave
32,520 and 32,480 ha.

Two competing explanations, with opposite consequences:

**(A) Solver fragility.** `buffer_width` is the Chebyshev distance from frontier
cells included in the narrow band, so a *larger* buffer tracks *more* cells and
should be more accurate, not less. If buf 6 is right and buf 3 is spuriously
breaking out, the breakout is a numerical artifact and the deliverable is broken.

**(B) A real heavy tail, sampled differently.** Spotting draws from an RNG once
per timestep per burning cell, so changing `cfl` or `buffer_width` changes the
timestep sequence, which changes *which embers land where*. That would make these
six runs six different realizations of a bimodal outcome rather than six
approximations of one number -- most fires stay inside their fuel-connected basin
(~32,500 ha), a minority have one ember reach a new basin and roughly double.

The discriminating experiment: **hold the solver fixed and vary only the seed.**
If the outcome is bimodal across seeds at fixed `cfl`/`buf`, (B) is confirmed and
the solver is innocent. If every seed at buf 3 breaks out and every seed at buf 6
does not, it is (A) and Stage 3 stops until it is understood.

Note what is NOT in question: determinism. Same settings and same seed reproduce
bitwise (G0), which is what CRN pairing needs. The question here is convergence
in the solver parameters, which is a different property.

Run:  conda activate fb_pyre && python code/dev/pyre/b2_spot_variance.py
"""

import csv
import json

import numpy as np

from b1_benchmark import OUT_DIR, base_spec, pick_ignitions, run_spec

#: Fixed solver, varying seed. Four seeds is enough to distinguish "always" from
#: "sometimes"; it is not enough to estimate a breakout rate, which is the
#: campaign's job.
SEEDS = (617327, 11, 20260911, 99991)
BUFFERS = (3, 6)


def main():
    from fb_tools.models.pyre import pyre_version_info
    print(json.dumps(pyre_version_info(), indent=2))

    picks = pick_ignitions(3)
    name0, yx0 = picks[0]
    print(f"\nignition {name0} at {yx0}; 30 m, 21 d, spotting central, crown gate 0.0")
    print("holding cfl at 0.4 and varying ONLY the seed\n")

    rows = []
    print(f"  {'buf':>4s} {'seed':>10s} {'burned ha':>11s} {'crown':>7s} "
          f"{'I p95':>8s} {'wall s':>8s} {'breakout':>9s}")
    for buf in BUFFERS:
        for seed in SEEDS:
            rec = run_spec(base_spec(
                resolution=30, duration_days=21, spotting="central",
                crown_gate=0.0, max_cells_per_timestep=0.4, buffer_width=buf,
                seed=seed, ignition=name0, ignition_yx=yx0))
            if rec.get("error"):
                print(f"  {buf:>4} {seed:>10} ERROR {rec['error'][:60]}")
                rows.append(rec)
                continue
            # The no-spotting basin is ~32,500 ha; treat >40,000 as a breakout.
            breakout = rec["burned_ha"] > 40_000
            rec["breakout"] = breakout
            rows.append(rec)
            print(f"  {buf:>4} {seed:>10} {rec['burned_ha']:>11,.0f} "
                  f"{100*rec['crown_frac']:>6.1f}% {rec['fi_p95_kw_m']:>8,.0f} "
                  f"{rec['wall_s']:>8.1f} {str(breakout):>9s}")

    ok = [r for r in rows if not r.get("error")]
    print(f"\n{'=' * 74}")
    for buf in BUFFERS:
        sub = [r for r in ok if r["buffer_width"] == buf]
        if not sub:
            continue
        n_b = sum(r["breakout"] for r in sub)
        ha = [r["burned_ha"] for r in sub]
        print(f"  buffer_width {buf}: {n_b}/{len(sub)} seeds broke out; "
              f"area {min(ha):,.0f}-{max(ha):,.0f} ha "
              f"(mean {np.mean(ha):,.0f}, CV {100*np.std(ha)/np.mean(ha):.1f}%)")

    b3 = [r for r in ok if r["buffer_width"] == 3]
    b6 = [r for r in ok if r["buffer_width"] == 6]
    n3, n6 = sum(r["breakout"] for r in b3), sum(r["breakout"] for r in b6)

    def cv(rs):
        ha = [r["burned_ha"] for r in rs]
        return 100 * np.std(ha) / np.mean(ha) if rs else float("nan")

    cv3, cv6 = cv(b3), cv(b6)
    print()
    # (B) requires BOTH buffers to show seed-driven variance. Bimodality at one
    # buffer while the other is flat is not evidence that the solver is innocent
    # -- it is evidence that the solver setting controls whether the stochastic
    # term does anything at all. The first version of this check concluded (B)
    # off the buf-3 bimodality alone and ignored buf 6's 0.1% CV.
    seedy3 = 0 < n3 < len(b3) or cv3 > 5.0
    seedy6 = 0 < n6 < len(b6) or cv6 > 5.0

    if seedy3 and seedy6:
        print("  VERDICT (B): both buffers show seed-driven bimodality, so these are")
        print("  realizations of a heavy tail rather than approximations of one number.")
        print("  The solver is innocent; the campaign must report a breakout RATE and no")
        print("  single realization may be quoted.")
    elif seedy3 and not seedy6:
        print(f"  VERDICT (A+B): buf 3 is genuinely stochastic (CV {cv3:.1f}%, {n3}/{len(b3)}")
        print(f"  breakouts) but buf 6 is nearly deterministic (CV {cv6:.1f}%, {n6}/{len(b6)}).")
        print("  So there IS a real heavy tail AND the solver setting controls whether it")
        print("  can express itself. buffer_width is not an accuracy knob here -- it gates")
        print("  the primary deliverable. Resolve which setting is correct before Stage 3:")
        print("  run spotting OFF at buf 6. If it matches spotting ON, spotting is inert")
        print("  at buf 6 and that is an artifact, not a physics choice.")
    elif not seedy3 and not seedy6:
        print("  Neither buffer shows seed variance. The knob-sweep spread was then NOT")
        print("  stochastic, so it was solver-driven -- verdict (A). STOP before Stage 3.")
    else:
        print(f"  INCONCLUSIVE: buf 3 {n3}/{len(b3)} breakouts (CV {cv3:.1f}%), "
              f"buf 6 {n6}/{len(b6)} (CV {cv6:.1f}%).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / "b2_spot_variance.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(fp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
