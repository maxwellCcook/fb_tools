"""Package ember_descent on real runs: fresh starts left after bridging and the burn share descending from them. Read-only."""
import sys, json, time
from pathlib import Path
import numpy as np, rasterio
from fb_tools.spread.transmission import ember_descent
for rd in map(Path, sys.argv[1:]):
    man = json.loads((rd / "manifest.json").read_text())
    with rasterio.open(rd / "time_of_arrival.tif") as s:
        toa = s.read(1)
    b = np.isfinite(toa); ign = np.zeros(toa.shape, bool)
    r0, c0, h, w = man["ignition_window"]; ign[max(r0 - 1, 0):r0 + h + 1, max(c0 - 1, 0):c0 + w + 1] = True
    out = []
    for bridge in (1, 3):
        t0 = time.time(); share, k = ember_descent(toa, ign, bridge_cells=bridge)
        out.append(f"bridge {bridge}: {k} starts, {share[b].mean():.1%} of burn ({time.time() - t0:.1f} s)")
    print(f"{rd.parent.name}/{rd.name} (spot {'on' if man.get('spot_config') else 'off'}, {b.sum():,} cells): " + "; ".join(out))
