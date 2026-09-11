"""
Is spotting inert at buffer_width=6?

b2 found buf 3 genuinely stochastic (CV 29.3%, 2/4 seeds breaking out) while
buf 6 is nearly deterministic (CV 0.1%, 0/4). That is not an accuracy trade-off
-- it means the solver setting controls whether the stochastic term does anything
at all, and spotting is the primary deliverable of this port.

The decisive test is one line of reasoning: run spotting OFF at each buffer. If
spotting-on equals spotting-off at buf 6, spotting is INERT there and buf 6 is an
artifact rather than a defensible setting. Extra seeds firm up whether buf 6's
breakout rate is genuinely zero or merely low.

`spot_from_burned_cell` is called ONCE PER CELL at the moment it ignites (it takes
`time_of_arrival`), and `expected_firebrand_production` returns the firebrands for
"an entire cell when it burns" -- so total firebrand production is dt-independent
by construction. That rules out the naive "more timesteps -> more embers"
explanation and makes the buf-6 result genuinely puzzling, which is why it needs
measuring rather than reasoning about.
"""
import csv, json
import numpy as np
from b1_benchmark import OUT_DIR, base_spec, pick_ignitions, run_spec

EXTRA_SEEDS = (2027, 424242, 8675309, 13, 1066, 31337)

def main():
    from fb_tools.models.pyre import pyre_version_info
    print(json.dumps(pyre_version_info(), indent=2))
    name0, yx0 = pick_ignitions(3)[0]
    print(f"\nignition {name0} at {yx0}; 30 m, 21 d, crown gate 0.0, cfl 0.4\n")

    rows = []

    print("--- [1] is spotting inert at each buffer? (spotting OFF reference) ---")
    print(f"  {'buf':>4s} {'spotting':>9s} {'burned ha':>11s} {'wall s':>8s}")
    off = {}
    for buf in (3, 6):
        rec = run_spec(base_spec(resolution=30, duration_days=21, spotting="off",
                                 crown_gate=0.0, buffer_width=buf, ignition=name0,
                                 ignition_yx=yx0))
        rows.append(rec)
        off[buf] = rec["burned_ha"]
        print(f"  {buf:>4} {'off':>9s} {rec['burned_ha']:>11,.0f} {rec['wall_s']:>8.1f}")

    print("\n--- [2] more seeds, spotting ON, to pin the breakout rate ---")
    print(f"  {'buf':>4s} {'seed':>10s} {'burned ha':>11s} {'vs off':>9s} {'wall s':>8s}")
    on = {3: [], 6: []}
    for buf in (3, 6):
        for seed in EXTRA_SEEDS:
            rec = run_spec(base_spec(resolution=30, duration_days=21, spotting="central",
                                     crown_gate=0.0, buffer_width=buf, seed=seed,
                                     ignition=name0, ignition_yx=yx0))
            rows.append(rec)
            if rec.get("error"):
                print(f"  {buf:>4} {seed:>10} ERROR {rec['error'][:50]}"); continue
            on[buf].append(rec["burned_ha"])
            d = 100 * (rec["burned_ha"] - off[buf]) / off[buf]
            print(f"  {buf:>4} {seed:>10} {rec['burned_ha']:>11,.0f} {d:>+8.2f}% "
                  f"{rec['wall_s']:>8.1f}")

    print(f"\n{'=' * 74}")
    for buf in (3, 6):
        if not on[buf]:
            continue
        ha = np.array(on[buf])
        lift = 100 * (ha.mean() - off[buf]) / off[buf]
        n_b = int((ha > 40_000).sum())
        print(f"  buf {buf}: spotting off {off[buf]:,.0f} ha | on mean {ha.mean():,.0f} ha "
              f"({lift:+.2f}%), {n_b}/{len(ha)} breakouts, CV {100*ha.std()/ha.mean():.1f}%")
    if on[6]:
        ha6 = np.array(on[6])
        inert = abs(ha6.mean() - off[6]) / off[6] < 0.01 and (ha6 > 40_000).sum() == 0
        print()
        if inert:
            print("  -> SPOTTING IS INERT AT buf 6 (within 1% of spotting-off, no breakouts).")
            print("     buf 6 is an artifact, not a conservative choice. Production must use")
            print("     buf 3, and buffer_width must be pinned and stamped in every manifest.")
        else:
            print("  -> spotting is ACTIVE at buf 6, just with a lower breakout rate. Then")
            print("     buffer_width is a genuine accuracy knob and the correct setting has")
            print("     to be argued, not assumed.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / "b3_buffer_diagnosis.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(fp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {fp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
