"""fuel_break: a treated corridor along line features, engine-free."""
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import LineString

gpd = pytest.importorskip("geopandas")

from fb_tools.fuelscape import fuel_break
from fb_tools.spread import destack_lcp

CANOPY = pd.DataFrame({"Treatment": ["Patch", "Thin"], "cbd_AF": [0, 0.6], "cbh_AF": [0, 1.2],
                       "cc_AF": [0, 0.7], "ch_AF": [0, 1.2]})
SURFACE = pd.DataFrame({"Code": ["TU5"], "FBFM40": [165], "Patch_SM": [121], "Complete": [161]})


def _line(synth_lcp):
    with rasterio.open(synth_lcp) as src:
        b, crs = src.bounds, src.crs
    # north-south line down the middle, through the water stripe
    x = (b.left + b.right) / 2
    return gpd.GeoSeries([LineString([(x, b.bottom), (x, b.top)])], crs=crs)


def test_clear_cut_corridor_changes_only_the_corridor(synth_lcp, tmp_path):
    out_fp = tmp_path / "break.tif"
    out, mask = fuel_break(synth_lcp, _line(synth_lcp), 120.0, CANOPY, SURFACE,
                           {"canopy": "Patch", "surface": "Patch_SM"}, out_path=out_fp)
    m = mask.values
    assert 3 <= np.unique(m.sum(axis=1)).min() <= 5           # 120 m / 30 m cells wide, every row
    with rasterio.open(synth_lcp) as a, rasterio.open(out_fp) as b:
        assert b.descriptions == a.descriptions and b.nodata == a.nodata and b.dtypes == a.dtypes
        base, trt = a.read(), b.read()
    changed = (base != trt).any(axis=0)
    assert changed.sum() > 0 and not (changed & ~m).any()  # nothing outside the corridor moved
    fm_in = trt[3][m]
    assert set(np.unique(fm_in)) <= {121, 98}                # TU5 -> GS1; water stays water
    assert (trt[4:8][:, m] == 0).all()                       # canopy zeroed in the corridor
    assert (trt[3][~m] == base[3][~m]).all()


def test_shaded_break_keeps_canopy(synth_lcp, tmp_path):
    out, mask = fuel_break(synth_lcp, _line(synth_lcp), 120.0, CANOPY, SURFACE,
                           {"canopy": "Thin", "surface": "Complete"})
    m = mask.values
    cc = out.sel(band=5).values
    assert (cc[m & (cc > 0)] == np.floor(45 * 0.7)).all()
    assert set(np.unique(out.sel(band=4).values[m])) <= {161, 98}


def test_corridor_destacks_for_the_engine(synth_lcp, tmp_path):
    out_fp = tmp_path / "break.tif"
    fuel_break(synth_lcp, _line(synth_lcp), 120.0, CANOPY, SURFACE,
               {"canopy": "Patch", "surface": "Patch_SM"}, out_path=out_fp)
    layers = destack_lcp(out_fp, tmp_path / "layers", verbose=False)
    with rasterio.open(tmp_path / "layers" / "fuel_model.tif") as src:
        assert 121 in np.unique(src.read(1))


def test_rejects_bad_inputs(synth_lcp):
    with pytest.raises(ValueError, match="positive"):
        fuel_break(synth_lcp, _line(synth_lcp), 0, CANOPY, SURFACE, {"canopy": "Patch", "surface": "Patch_SM"})
    with pytest.raises(ValueError, match="CRS"):
        fuel_break(synth_lcp, _line(synth_lcp).set_crs(None, allow_override=True), 120, CANOPY, SURFACE,
                   {"canopy": "Patch", "surface": "Patch_SM"})
