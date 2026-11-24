"""Why does the clear-cut arm of the 01f fuel-break example burn more than the baseline? Read-only."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

from fb_tools.spread.transmission import changed_cells

root = Path(sys.argv[1]).expanduser()
P = root / "runs/fuel_break/central/ign_000/ep_000"
arms = ("baseline", "shaded", "clearcut")
man = {a: json.loads((P / a / "manifest.json").read_text()) for a in arms}
def rd(fp):
    with rasterio.open(fp) as s:
        return s.read(1)
lay = {a: {k: rd(root / man[a]["layers_dir"] / f"{k}.tif") for k in
           ("fuel_model", "canopy_cover", "canopy_height", "canopy_base_height", "canopy_bulk_density")} for a in arms}
run = {a: {k: rd(P / a / f"{k}.tif") for k in ("time_of_arrival", "spread_rate", "fireline_intensity", "flame_length", "fire_type")} for a in arms}
cell_ha = man["baseline"]["resolution_m"] ** 2 / 1e4

print("== 1. fuels inside the corridor")
for a in ("shaded", "clearcut"):
    trt = changed_cells(lay["baseline"], lay[a])
    print(f"-- {a}: {trt.sum()} changed cells ({trt.sum() * cell_ha:.0f} ha)")
    pairs = Counter(zip(lay["baseline"]["fuel_model"][trt].tolist(), lay[a]["fuel_model"][trt].tolist()))
    print("   fuel model before -> after (cells): " + ", ".join(f"{b}->{c}: {n}" for (b, c), n in pairs.most_common(8)))
    for k in ("canopy_cover", "canopy_height", "canopy_base_height", "canopy_bulk_density"):
        print(f"   {k:20s} mean {lay['baseline'][k][trt].mean():.3f} -> {lay[a][k][trt].mean():.3f}")
trt = changed_cells(lay["baseline"], lay["clearcut"])

print("\n== 2. burned area by day (ha) and spot/config")
print("   " + "  ".join(f"{a}: spot={man[a].get('spot_config') is not None} gate={man[a].get('crown_gate')} seed={man[a].get('seed')}" for a in arms))
for d in (0.5, 1, 1.5, 2, 3, 5, 7, 10, 14, 21):
    print(f"   day {d:>4}: " + "  ".join(f"{a} {(run[a]['time_of_arrival'] <= d * 1440).sum() * cell_ha:>8.0f}" for a in arms))

print("\n== 3. behaviour inside the corridor, cells burned in all arms")
both = trt & np.all([np.isfinite(run[a]["time_of_arrival"]) for a in arms], axis=0)
print(f"   corridor cells burned: " + ", ".join(f"{a} {(np.isfinite(run[a]['time_of_arrival']) & trt).sum()}" for a in arms) + f"; in all three {both.sum()}")
for k in ("spread_rate", "fireline_intensity", "flame_length"):
    print(f"   {k:20s} median: " + "  ".join(f"{a} {np.nanmedian(run[a][k][both]):.1f}" for a in arms)
          + "   p90: " + "  ".join(f"{a} {np.nanpercentile(run[a][k][both], 90):.1f}" for a in arms))
print("   fire type shares (1 surface, 2 passive, 3 active): " + "  ".join(
    f"{a} " + "/".join(f"{(run[a]['fire_type'][both] == v).mean():.0%}" for v in (1, 2, 3)) for a in arms))
print("   corridor first reached (h): " + "  ".join(f"{a} {np.nanmin(run[a]['time_of_arrival'][trt]) / 60:.1f}" for a in arms))
print("   time to cross: corridor cells' arrival, p10/p50/p90 (h): " + "  ".join(
    f"{a} " + "/".join(f"{np.nanpercentile(run[a]['time_of_arrival'][trt], q) / 60:.0f}" for q in (10, 50, 90)) for a in arms))

print("\n== 4. where the extra clear-cut area is")
tb, tc = run["baseline"]["time_of_arrival"], run["clearcut"]["time_of_arrival"]
extra = np.isfinite(tc) & ~np.isfinite(tb)
rr, cc = np.where(np.isfinite(tb)); er, ec = np.where(extra); kr, kc = np.where(trt)
print(f"   extra cells {extra.sum()} ({extra.sum() * cell_ha:.0f} ha); baseline centroid (row, col) ({rr.mean():.0f}, {cc.mean():.0f}); "
      f"extra centroid ({er.mean():.0f}, {ec.mean():.0f}); corridor centroid ({kr.mean():.0f}, {kc.mean():.0f}); grid {tb.shape}")
print(f"   earliest extra cell burns at {np.nanmin(tc[extra]) / 60:.1f} h; baseline max arrival {np.nanmax(tb) / 60:.0f} h")
# first cells where clear cut is >1 h EARLIER than baseline, or burns where baseline never does
with np.errstate(invalid="ignore"):
    early = (np.isfinite(tc) & np.isfinite(tb) & (tc < tb - 60)) | extra
t_first = np.nanmin(np.where(early, tc, np.nan))
fr, fc = np.where(early & (tc <= t_first + 120))
d2 = ndi.distance_transform_edt(~trt) * man["baseline"]["resolution_m"]
print(f"   first cells running ahead of the baseline: at {t_first / 60:.1f} h, {len(fr)} cells within 2 h, "
      f"median distance from the corridor {np.median(d2[fr, fc]):.0f} m (min {d2[fr, fc].min():.0f} m), inside corridor {trt[fr, fc].mean():.0%}")
edge = 3
for a in arms:
    t = run[a]["time_of_arrival"]
    b = np.concatenate([t[:edge].ravel(), t[-edge:].ravel(), t[:, :edge].ravel(), t[:, -edge:].ravel()])
    b = b[np.isfinite(b)]
    print(f"   {a}: first touched the domain edge at {'never' if not b.size else f'{b.min() / 1440:.1f} d'}")

print("\n== 5. is the extra area ember-started?")
from fb_tools.spread.transmission import ember_descent, _propagate   # noqa: E402
res = man["baseline"]["resolution_m"]
starts = {}
for a in arms:
    t = run[a]["time_of_arrival"]
    r0, c0, h, w = man[a]["ignition_window"]
    ign = np.zeros(t.shape, bool); ign[max(r0 - 1, 0):r0 + h + 1, max(c0 - 1, 0):c0 + w + 1] = True
    share, k = ember_descent(t, ign)
    _, st = _propagate(t, np.zeros(t.shape, bool)); st &= ~ign
    starts[a] = st
    sr, sc = np.where(st)
    o = np.argsort(t[sr, sc])
    print(f"   {a}: {k} ember starts; burn descending from them {share[np.isfinite(t)].mean():.1%}; "
          + "; ".join(f"({sr[i]},{sc[i]}) at {t[sr[i], sc[i]] / 60:.0f} h" for i in o))
    if a == "clearcut":
        print(f"      share of the extra area descending from an ember start: {share[extra].mean():.1%}")
        for i in o:                                              # each start: how far from fire burning at that time, was it burned in baseline, what grew from it
            r, c = sr[i], sc[i]
            one = np.zeros(t.shape, bool); one[r, c] = True
            s1, _ = _propagate(t, one)
            prior = np.isfinite(t) & (t < t[r, c])
            pr, pc = np.where(prior)
            gap = np.hypot(pr - r, pc - c).min() * res
            print(f"      start ({r},{c}) at {t[r, c] / 60:.0f} h: {gap:.0f} m from fire already burning; baseline arrival there "
                  f"{tb[r, c] / 60 if np.isfinite(tb[r, c]) else float('nan'):.0f} h; area descending from it {s1.sum() * cell_ha:.0f} ha, of the extra area {s1[extra].sum() / extra.sum():.0%}; "
                  f"distance from the corridor {d2[r, c]:.0f} m")
print("\n== 6. hourly growth through the blow-up (ha burned in the hour)")
for h0 in range(255, 300, 3):
    print(f"   h {h0}-{h0 + 3}: " + "  ".join(f"{a} {((run[a]['time_of_arrival'] > h0 * 60) & (run[a]['time_of_arrival'] <= (h0 + 3) * 60)).sum() * cell_ha:>7.0f}" for a in arms))

print("\n== 7. did the extra area come through the corridor?")
from fb_tools.spread.transmission import treatment_shadow   # noqa: E402
sh_c = treatment_shadow(tc, trt)
print(f"   clear-cut arm: share of the extra area whose fire passed through corridor cells {sh_c[extra].mean():.0%} "
      f"(cells >= 0.5: {(sh_c[extra] >= 0.5).mean():.0%}); same for the baseline-burned area {sh_c[np.isfinite(tb) & np.isfinite(tc) & ~trt].mean():.0%}")
nb = np.isin(lay["baseline"]["fuel_model"], (91, 92, 93, 98, 99))
print(f"   corridor footprint (300 m buffer of the road): baseline non-burnable cells inside the changed set {int((nb & trt).sum())}; "
      f"non-burnable in baseline within 150 m of changed cells {int((nb & (d2 <= 150)).sum())}")
with np.errstate(invalid="ignore"):
    lead = (tb - tc)[trt]                                   # + = clear cut burned the corridor cell earlier
print(f"   corridor cells: clear cut earlier than baseline by >1 h: {(lead > 60).mean():.0%}, later by >1 h: {(lead < -60).mean():.0%}; "
      f"lead p10/p50/p90 (h): {np.nanpercentile(lead, 10) / 60:.1f}/{np.nanpercentile(lead, 50) / 60:.1f}/{np.nanpercentile(lead, 90) / 60:.1f}")
# the breakout: first extra cells and the corridor cells feeding them
t0 = np.nanmin(tc[extra]); fr, fc = np.where(extra & (tc <= t0 + 180))
print(f"   breakout: {len(fr)} extra cells in the first 3 h from {t0 / 60:.0f} h, centred (row, col) ({fr.mean():.0f}, {fc.mean():.0f}), "
      f"{np.median(d2[fr, fc]):.0f} m from the corridor (min {d2[fr, fc].min():.0f} m); shadow share there {sh_c[fr, fc].mean():.0%}")
r, c = int(fr.mean()), int(fc.mean()); w = 40
loc = (slice(max(r - w, 0), r + w), slice(max(c - w, 0), c + w))
print(f"   around the breakout (±{w} cells): baseline fuel models {Counter(lay['baseline']['fuel_model'][loc].ravel().tolist()).most_common(6)}")
print(f"      baseline burned {np.isfinite(tb[loc]).mean():.0%} of that window (latest {np.nanmax(tb[loc]) / 60 if np.isfinite(tb[loc]).any() else float('nan'):.0f} h); "
      f"clear cut burned {np.isfinite(tc[loc]).mean():.0%}; corridor cells in window {int(trt[loc].sum())}")
