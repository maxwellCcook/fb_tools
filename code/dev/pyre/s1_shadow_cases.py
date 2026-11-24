"""Synthetic review of treatment_shadow. Expected values are stated before each result."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import erf
from fb_tools.spread import transmission as tr
from fb_tools.spread.transmission import treatment_shadow

RES, ROS = 30.0, 10.0            # m, m/min  -> 3 min per cell
H, W = 201, 300
DT = RES / ROS

def plane(h=H, w=W, angle_deg=0.0):
    r, c = np.mgrid[:h, :w]
    a = np.radians(angle_deg)
    return (c * np.cos(a) + r * np.sin(a)) * DT

def travel_time(speed, sources):
    """Dijkstra arrival time (min) on the 8-neighbour lattice; speed m/min, 0 = non-burnable."""
    h, w = speed.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals = [], [], []
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        a, a2 = slice(max(dr, 0), h + min(dr, 0)), slice(max(-dr, 0), h + min(-dr, 0))
        k, k2 = slice(max(dc, 0), w + min(dc, 0)), slice(max(-dc, 0), w + min(-dc, 0))
        ok = (speed[a, k] > 0) & (speed[a2, k2] > 0)
        cost = RES * np.hypot(dr, dc) * 0.5 * (1 / np.where(speed[a, k] > 0, speed[a, k], 1) + 1 / np.where(speed[a2, k2] > 0, speed[a2, k2], 1))
        rows.append(idx[a2, k2][ok]); cols.append(idx[a, k][ok]); vals.append(cost[ok])
    g = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w))
    d = dijkstra(g, indices=[idx[s] for s in sources], min_only=True).reshape(h, w)
    return np.where(np.isfinite(d), d, np.nan)

print("== A. plane front, full-height strip (cols 50-53)")
print("   expected: 1.0 everywhere east of the strip, 0.0 west")
toa = plane(); trt = np.zeros((H, W), bool); trt[:, 50:54] = True
sh = treatment_shadow(toa, trt)
print(f"   got: east min {sh[:, 54:].min():.4f} max {sh[:, 54:].max():.4f}; west max {sh[:, :50].max():.4f}")

print("== B. plane front, 21-row block (rows 90-110, cols 50-53); straight spread")
print("   expected (geometry): 1.0 in rows 90-110 east of the block at any distance, 0.0 outside those rows")
trt = np.zeros((H, W), bool); trt[90:111, 50:54] = True
for p in (1.0, 2.0, 4.0, 8.0):
    sh = treatment_shadow(toa, trt, power=p)
    wn = np.array([1.0, 2 ** (-p / 2), 2 ** (-p / 2)]); wn /= wn.sum()
    var = wn[1] + wn[2]                                   # lateral variance per column, cells^2
    line = []
    for d in (10, 50, 100, 200):
        col = 53 + d
        pred = erf(10.5 / np.sqrt(2 * var * d))
        inside = sh[90:111, col].sum(); total = sh[:, col].sum()
        line.append(f"d={d * RES / 1000:.1f}km centre {sh[100, col]:.2f} (random-walk {pred:.2f}) leak {1 - inside / total:.0%}")
    print(f"   power={p:g}: " + "; ".join(line) + f"; column sum conserved {sh[:, 250].sum():.1f}/21")

print("== B2. same block, front rotated (anisotropy of the 8-neighbour rule), power=1")
print("   expected (geometry): band of constant width behind the block along the travel direction")
for ang in (0.0, 22.5, 45.0):
    h = w = 301
    toa2 = plane(h, w, ang); trt2 = np.zeros((h, w), bool)
    r, c = np.mgrid[:h, :w]; a = np.radians(ang)
    along = (c - 60) * np.cos(a) + (r - 60) * np.sin(a); across = -(c - 60) * np.sin(a) + (r - 60) * np.cos(a)
    trt2[(np.abs(along) <= 2) & (np.abs(across) <= 10)] = True
    sh2 = treatment_shadow(toa2, trt2)
    out = []
    for d in (50, 150):
        m = np.abs(along - d) <= 0.5
        prof_c = sh2[m & (np.abs(across) <= 0.75)].mean()
        mom = np.sqrt((sh2[m] * across[m] ** 2).sum() / sh2[m].sum())
        out.append(f"d={d} cells: centre {prof_c:.2f}, lateral RMS {mom:.1f} cells (block alone = 6.1)")
    print(f"   angle {ang:>4}: " + "; ".join(out))

print("== C. obstacle the fire flanks around (Dijkstra travel time, point ignition west)")
speed = np.full((H, W), ROS); blk = np.zeros((H, W), bool); blk[80:121, 100:104] = True
src = [(100, 20)]
base = travel_time(speed, src)
print("   C1 treated block does not burn (non-burnable / held): expected by the code: shadow 0 everywhere")
s1 = speed.copy(); s1[blk] = 0; t1 = travel_time(s1, src)
sh = treatment_shadow(t1, blk)
lee = np.zeros((H, W), bool); lee[90:111, 104:160] = True
print(f"        got: shadow max {sh.max():.2f}; arrival delay in the lee vs untreated: median {np.nanmedian((t1 - base)[lee]):.0f} min")
for slow in (0.5, 0.2, 0.05):
    s2 = speed.copy(); s2[blk] = ROS * slow; t2 = travel_time(s2, src)
    sh = treatment_shadow(t2, blk)
    d = (t2 - base)
    delayed = (d > 1.0) & ~blk
    print(f"   C2 block burns at {slow:.0%} speed: cells delayed >1 min outside the block: {delayed.sum()}; "
          f"of those shadow>=0.5: {(sh[delayed] >= 0.5).mean():.0%}, shadow<0.1: {(sh[delayed] < 0.1).mean():.0%}; "
          f"median delay there {np.median(d[delayed]):.0f} min")

print("== D. spot fire landing beyond a full-height strip")
print("   expected by the code: the spot-grown area is 0 although the ember was lofted after the fire crossed the strip")
toa = plane(); trt = np.zeros((H, W), bool); trt[:, 50:54] = True
r, c = np.mgrid[:H, :W]
t_spot = toa[100, 60] + 5.0                                # lofted from col 60 (already beyond the strip), lands col 120
spot = t_spot + np.hypot(r - 100, c - 120) * DT
toa_s = np.minimum(toa, spot); sh = treatment_shadow(toa_s, trt)
east = np.zeros((H, W), bool); east[:, 54:] = True
print(f"        got: area east of the strip with shadow < 0.5: {(sh[east] < 0.5).mean():.0%} "
      f"(spot-ignited cells are {(spot < toa)[east].mean():.0%} of it); without the spot: {(treatment_shadow(toa, trt)[east] < 0.5).mean():.0%}")

print("== E. two fires merging, one through a strip (A at col 30 with strip at col 45, B at col 269)")
print("   expected: 1 between the strip and the meeting line (col ~150), 0 on B's side, ~0.5 on the line")
toa_m = np.minimum(np.abs(c - 30), np.abs(c - 269)) * DT
trt = np.zeros((H, W), bool); trt[:, 45:49] = True
sh = treatment_shadow(toa_m, trt)
print(f"        got: cols 49-148 min {sh[:, 49:149].min():.2f}; cols 149-150 {sh[100, 149]:.2f}, {sh[100, 150]:.2f}; cols 151+ max {sh[:, 151:].max():.2f}; west of A max {sh[:, :30].max():.2f}")

print("== F. tied arrival times (a plateau 5 columns wide at cols 70-74, east of a full strip)")
print("   expected by intent: 1 east of the strip. Expected by the code: plateau interior has no strictly earlier neighbour -> roots -> 0 downstream")
toa_t = plane().copy(); toa_t[:, 70:75] = toa_t[:, 70:71]
sh = treatment_shadow(toa_t, np.pad(np.ones((H, 4), bool), ((0, 0), (50, W - 54))))
print(f"        got: col 69 {sh[100, 69]:.2f}, col 70 {sh[100, 70]:.2f}, col 72 {sh[100, 72]:.2f}, col 80 {sh[100, 80]:.2f}, col 200 {sh[100, 200]:.2f}")
toa_q = np.round(plane(angle_deg=22.5))                     # whole-minute arrival times on an oblique front
trtq = np.zeros((H, W), bool); trtq[:, 50:54] = True
shq = treatment_shadow(toa_q, trtq)
print(f"   whole-minute rounding, oblique front, full strip: east-of-strip mean {shq[:, 60:].mean():.3f} (1.0 expected)")

print("== G. raw solver values before the 0-1 clip (case B, power 1)")
rec = {}
orig = np.clip
def spy(x, lo, hi, *a, **k):
    rec["min"], rec["max"] = float(np.min(x)), float(np.max(x)); return orig(x, lo, hi, *a, **k)
tr.np.clip = spy
trt = np.zeros((H, W), bool); trt[90:111, 50:54] = True
treatment_shadow(plane(), trt)
tr.np.clip = orig
print(f"        raw min {rec['min']:.3e}, raw max {rec['max']:.6f}")
