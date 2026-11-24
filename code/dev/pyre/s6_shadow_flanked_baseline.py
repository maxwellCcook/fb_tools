"""Flanked treatment: does the shadow read off the BASELINE arm cover the cells the treatment delayed?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from s1_shadow_cases import travel_time, H, W, ROS          # noqa: E402  (prints s1's cases on import)
from fb_tools.spread.transmission import treatment_shadow   # noqa: E402

print("\n== flanked block, shadow from the baseline arm vs from the treated arm")
speed = np.full((H, W), ROS); blk = np.zeros((H, W), bool); blk[80:121, 100:104] = True
src = [(100, 20)]
base = travel_time(speed, src)
sh_base = treatment_shadow(base, blk)
for label, v in (("50% speed", 0.5), ("20% speed", 0.2), ("5% speed", 0.05), ("does not burn", 0.0)):
    s = speed.copy(); s[blk] = ROS * v
    t = travel_time(s, src)
    d = t - base
    delayed = (d > 1.0) & ~blk & np.isfinite(t)
    sh_trt = treatment_shadow(t, blk)
    lee = (sh_base >= 0.5) & ~blk
    print(f"   block {label:>13}: delayed cells {delayed.sum():>5}; in baseline shadow {(sh_base[delayed] >= 0.5).mean():.0%}, "
          f"in treated-arm shadow {(sh_trt[delayed] >= 0.5).mean():.0%}; baseline-shadow cells not delayed {(~delayed[lee]).mean():.0%}; "
          f"median delay inside baseline shadow {np.median(d[lee & np.isfinite(t)]):.0f} min, outside {np.median(d[~lee & ~blk & np.isfinite(t)]):.0f} min")
