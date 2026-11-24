"""Share of each real burn that descends from a spot landing (a root later than the first hour). Read-only."""
import sys
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage as ndi
from fb_tools.spread.transmission import treatment_shadow
for rd in map(Path, sys.argv[1:]):
    with rasterio.open(rd / "time_of_arrival.tif") as s:
        toa = s.read(1)
    b = np.isfinite(toa); t = np.where(b, toa, np.inf)
    root = b & (ndi.minimum_filter(t, size=3, mode="constant", cval=np.inf) >= t)
    spot = root & (toa > 60.0)
    sh = treatment_shadow(toa, spot)                        # "treated" = spot landings -> share of fire descending from a spot
    print(f"{rd.parent.name}/{rd.name}: roots {root.sum()} (first-hour {(root & ~spot).sum()}, later {spot.sum()}); "
          f"burned area descending from a later root: {sh[b].mean():.1%} (cells >=0.5: {(sh[b] >= 0.5).mean():.1%}); "
          f"latest root at {toa[spot].max() / 60 if spot.any() else 0:.0f} h of {np.nanmax(toa) / 60:.0f} h")
