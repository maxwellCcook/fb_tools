"""For each late root: distance to the nearest earlier-burning cell, and the burn share descending from
roots that remain after bridging gaps of up to R cells. Read-only experiment; nothing in fb_tools changes."""
import sys, json
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve_triangular

def roots_of(toa):
    b = np.isfinite(toa); t = np.where(b, toa, np.inf)
    return b & (ndi.minimum_filter(t, size=3, mode="constant", cval=np.inf) >= t)

def nearest_earlier(toa, r, c, win=120):
    r0, c0 = max(r - win, 0), max(c - win, 0)
    w = toa[r0:r + win + 1, c0:c + win + 1]
    rr, cc = np.where(np.isfinite(w) & (w < toa[r, c]))
    return np.inf if rr.size == 0 else float(np.hypot(rr + r0 - r, cc + c0 - c).min())

def shadow(toa, seed_mask, bridge=0):
    """treatment_shadow with roots allowed inflow from earlier cells within `bridge` cells (0 = package behaviour)."""
    burned = np.isfinite(toa); H, W = toa.shape
    n = int(burned.sum()); t = np.where(burned, toa, np.inf)
    rank = np.full((H, W), -1, dtype=np.int64)
    rank[burned] = np.argsort(np.argsort(toa[burned], kind="stable"), kind="stable")
    rows, cols, vals = [], [], []
    def add(offsets, only=None):
        for dr, dc in offsets:
            a, a2 = slice(max(dr, 0), H + min(dr, 0)), slice(max(-dr, 0), H + min(-dr, 0))
            k, k2 = slice(max(dc, 0), W + min(dc, 0)), slice(max(-dc, 0), W + min(-dc, 0))
            ok = burned[a, k] & burned[a2, k2] & (t[a2, k2] < t[a, k])
            if only is not None:
                ok &= only[a, k]
            with np.errstate(invalid="ignore"):
                w = (t[a, k] - t[a2, k2]) / np.hypot(dr, dc)
            rows.append(rank[a, k][ok]); cols.append(rank[a2, k2][ok]); vals.append(w[ok])
    add([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)])
    if bridge > 1:
        rt = roots_of(toa)
        for R in range(2, bridge + 1):                      # widen ring by ring; a root bridged at R stops there
            ring = [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1) if max(abs(dr), abs(dc)) == R]
            n0 = sum(len(x) for x in rows); add(ring, only=rt)
            got = np.zeros(n, bool); got[np.concatenate(rows[-len(ring):])] = True if sum(len(x) for x in rows) > n0 else False
            g2 = np.zeros((H, W), bool); g2[burned] = got[rank[burned]]; rt = rt & ~g2
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    s = np.zeros(n); s[rank[seed_mask & burned]] = 1.0
    keep = s[rows] == 0
    inflow = csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))
    rs = np.asarray(inflow.sum(axis=1)).ravel(); rs[rs == 0] = 1.0
    inflow.data /= np.repeat(rs, np.diff(inflow.indptr))
    x = spsolve_triangular((identity(n, format="csr") - inflow).tocsr(), s, lower=True, unit_diagonal=True)
    out = np.zeros((H, W), "float32"); out[burned] = np.clip(x[rank[burned]], 0, 1)
    return out

for rd in map(Path, sys.argv[1:]):
    man = json.loads((rd / "manifest.json").read_text())
    with rasterio.open(rd / "time_of_arrival.tif") as s:
        toa = s.read(1)
    b = np.isfinite(toa); rt = roots_of(toa); late = rt & (toa > 60.0)
    d = np.array([nearest_earlier(toa, r, c) for r, c in zip(*np.where(late))])
    line = f"{rd.parent.name}/{rd.name} (spot {'on' if man.get('spot_config') else 'off'}): late roots {late.sum()}; gap to nearest earlier cell, cells: {np.sort(np.round(d, 1)).tolist()}"
    print(line)
    for R in (0, 2, 3, 5):
        far = late.copy()
        if R:
            rr, cc = np.where(late); far[rr[d <= R * np.sqrt(2)], cc[d <= R * np.sqrt(2)]] = False
        sh = shadow(toa, far, bridge=R)
        print(f"     bridge {R}: roots left {far.sum():>2}; burn descending from them {sh[b].mean():.1%}")
