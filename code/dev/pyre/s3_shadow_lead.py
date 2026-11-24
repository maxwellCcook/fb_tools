"""Lead time with two treatments: which treatment's arrival is the clock started from?"""
import json, numpy as np, rasterio, tempfile
from pathlib import Path
from rasterio.transform import from_origin
from fb_tools.spread.transmission import realization_transmission
H, W, DT = 200, 300, 3.0
r, c = np.mgrid[:H, :W]; toa = (c * DT).astype("float32")
trt = np.zeros((H, W), bool); trt[:100, 20:24] = True; trt[100:, 200:204] = True     # north strip early, south strip late
blocks = np.zeros((H, W), "int32"); blocks[30:60, 240:260] = 1; blocks[140:170, 240:260] = 2
d = Path(tempfile.mkdtemp())
with rasterio.open(d / "time_of_arrival.tif", "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                   crs="EPSG:32613", transform=from_origin(0, H * 30, 30, 30)) as dst:
    dst.write(toa, 1)
(d / "manifest.json").write_text(json.dumps({"arm": "a"}))
s, b, _ = realization_transmission(d, treated=trt, block_ids=blocks, write_shadow=False)
print("expected lead (block arrival minus arrival at the strip upstream of it): block 1 (north) = (240-20)*3/60 = 11.0 h; block 2 (south) = (240-200)*3/60 = 2.0 h")
print(b[["block_id", "arrival_h", "shadow_frac", "lead_h"]].round(2).to_string(index=False))
