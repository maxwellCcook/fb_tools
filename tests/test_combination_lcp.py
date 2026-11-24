"""Combination LCP round trip: baseline + treatments -> (fake FlamMap) -> expand, engine-free."""

from functools import partial

import numpy as np
import pandas as pd
import pytest
import rasterio
import rioxarray as rxr
from rasterio.transform import from_origin

from fb_tools.fuelscape.adjust import apply_treatment
from fb_tools.fuelscape.synthetic import (EXPANDED_TAG, combination_spread,
                                          create_combination_lcp, expand_combinations)

BANDS = ["Elev", "SlpD", "Asp", "FBFM40", "CC", "CH", "CBH", "CBD", "EVT"]
DESC = tuple(f"LF2016_{b}_CONUS" for b in BANDS)
OUT_DESC = ("CROWNSTATE", "FLAMELENGTH", "INTENSITY")
NODATA = -9999
KEYS = ("FBFM40", "SlpD", "CC", "CH", "CBH", "CBD")

CANOPY = pd.DataFrame(dict(Treatment=["Thin"], cc_AF=[0.47], ch_AF=[1.08], cbh_AF=[1.24],
                           cbd_AF=[0.52]))
SURFACE = pd.DataFrame(dict(FBFM40=[102, 165, 183], Manage=[102, 165, 183], RxFire=[101, 161, 181]))
TREATMENTS = {
    "thin": partial(apply_treatment, canopy_df=CANOPY, surface_df=SURFACE,
                    scenario={"canopy": "Thin", "surface": "Manage"}),
    "thin_rx_wall": partial(apply_treatment, canopy_df=CANOPY, surface_df=SURFACE,
                            scenario={"canopy": "Thin", "surface": "RxFire"}, cc_floor=10),
}


def _write_lcp(fp, seed, shape=(12, 10)):
    """Random LCP from small value sets so combinations repeat; a nodata top row."""
    rng = np.random.default_rng(seed)
    sets = dict(Elev=range(1800, 3000), SlpD=[0, 5, 12], Asp=range(360), FBFM40=[102, 165, 183],
                CC=[15, 25, 45], CH=[100, 150], CBH=[3, 10], CBD=[5, 12], EVT=[7011, 7050])
    a = np.stack([rng.choice(list(sets[b]), shape) for b in BANDS]).astype(np.int16)
    a[:, 0, :] = NODATA
    with rasterio.open(fp, "w", driver="GTiff", dtype="int16", count=9, height=shape[0],
                       width=shape[1], crs="EPSG:32613", transform=from_origin(0, 360, 30, 30),
                       nodata=NODATA) as dst:
        dst.write(a)
        dst.descriptions = DESC
    return a


def _fake_flammap(lcp, depends_on=KEYS):
    """Deterministic 3-band 'output' of the named bands; 0 where Elev is nodata, as FlamMap writes."""
    v = sum((i + 1) * 0.37 * lcp[BANDS.index(b)].astype(np.float32) for i, b in enumerate(depends_on))
    out = np.stack([v % 4, v / 100, v * 3]).astype(np.float32)
    out[:, lcp[0] == NODATA] = 0
    return out


def _write_out(fp, out, like):
    with rasterio.open(like) as src:
        prof = dict(src.profile, dtype="float32", count=3, height=out.shape[1], width=out.shape[2])
    with rasterio.open(fp, "w", **prof) as dst:
        dst.write(out)
        dst.descriptions = OUT_DESC


@pytest.fixture
def base(tmp_path):
    fp = tmp_path / "base.tif"
    return fp, _write_lcp(fp, 1)


@pytest.fixture
def combo(tmp_path, base):
    fp = tmp_path / "combo.tif"
    summary = create_combination_lcp(base[0], fp, treatments=TREATMENTS)
    with rasterio.open(fp) as src:
        _write_out(tmp_path / "combo_out.tif", _fake_flammap(src.read()), fp)
    return fp, tmp_path / "combo_out.tif", summary


def test_baseline_and_treatments_reproduce_full_grid_runs(tmp_path, base, combo):
    base_fp, a = base
    combo_fp, out_fp, summary = combo
    assert list(summary.block) == ["baseline", *TREATMENTS]
    assert np.array_equal(expand_combinations(combo_fp, out_fp, base_fp), _fake_flammap(a))
    full = rxr.open_rasterio(base_fp, masked=False)
    for name, fn in TREATMENTS.items():
        treated = np.asarray(fn(full)).astype(np.int16)
        got = expand_combinations(combo_fp, out_fp, base_fp, treatment=name,
                                  out_path=tmp_path / f"{name}.tif")
        assert np.array_equal(got, _fake_flammap(treated)), name
        with rasterio.open(tmp_path / f"{name}.tif") as src:
            assert np.array_equal(src.read(), got)
            assert src.descriptions == OUT_DESC and src.tags()["source"] == EXPANDED_TAG["source"]


def test_baseline_cells_are_real_pixel_rows(base, combo):
    _, a = base
    with rasterio.open(combo[0]) as src:
        h = int(src.tags()["combination_block_height"])
        block = src.read()[:, :h].reshape(9, -1)
    real = {tuple(r) for r in a.reshape(9, -1).T if r[0] != NODATA}
    cells = [tuple(r) for r in block.T if r[0] != NODATA]
    assert cells and all(c in real for c in cells)


def test_unknown_combination_or_block_raises(tmp_path, combo):
    combo_fp, out_fp, _ = combo
    other = tmp_path / "other.tif"
    _write_lcp(other, 2)
    with pytest.raises(ValueError, match="not in the baseline block|outside"):
        expand_combinations(combo_fp, out_fp, other)
    with pytest.raises(ValueError, match="no block"):
        expand_combinations(combo_fp, out_fp, combo_fp, treatment="nope")


def test_spread_detects_a_missing_key(tmp_path, base):
    fp, a = base
    _write_out(tmp_path / "ok.tif", _fake_flammap(a), fp)
    _write_out(tmp_path / "asp.tif", _fake_flammap(a, depends_on=("FBFM40", "SlpD", "CC", "Asp")), fp)
    assert all(v == 0 for d, v in combination_spread(fp, tmp_path / "ok.tif").items() if d in OUT_DESC)
    assert combination_spread(fp, tmp_path / "asp.tif")["FLAMELENGTH"] > 0


def test_cc_wall_never_raises_cc(tmp_path):
    """cc_floor holds thinned CC at the wall; a cell already below it keeps all its canopy (CC = 1 is not cleared)."""
    pre = np.array([0, 1, 3, 9, 15, 40, 100], np.int16)
    a = np.ones((len(BANDS), 1, len(pre)), np.int16)
    a[BANDS.index("FBFM40")], a[BANDS.index("CC")] = 165, pre
    fp = tmp_path / "wall.tif"
    with rasterio.open(fp, "w", driver="GTiff", dtype="int16", count=len(BANDS), height=1, width=len(pre),
                       crs="EPSG:32613", transform=from_origin(0, 30, 30, 30), nodata=NODATA) as dst:
        dst.write(a)
        dst.descriptions = DESC
    half = CANOPY.assign(cc_AF=0.5)
    out = apply_treatment(rxr.open_rasterio(fp, masked=True), half, SURFACE,
                          {"canopy": "Thin", "surface": "Manage"}, cc_floor=10)
    assert out.sel(band=BANDS.index("CC") + 1).values.ravel().tolist() == [0, 1, 3, 9, 10, 20, 50]
    below = (pre > 0) & (pre < 10)
    for b in ("CH", "CBH", "CBD"):
        assert (out.sel(band=BANDS.index(b) + 1).values.ravel()[below] == a[BANDS.index(b), 0, below]).all(), b


def test_apply_treatment_honors_a_passed_band_map(tmp_path):
    _write_lcp(tmp_path / "lcp.tif", seed=3)
    lcp = rxr.open_rasterio(tmp_path / "lcp.tif", masked=True)
    kw = dict(canopy_df=CANOPY, surface_df=SURFACE, scenario={"canopy": "Thin", "surface": "RxFire"})
    auto = apply_treatment(lcp, **kw)
    bare = lcp.copy()
    bare.attrs.pop("long_name")                                     # an LCP with no band names
    with pytest.raises(ValueError, match="needs band"):
        apply_treatment(bare, **kw)
    named = apply_treatment(bare, band_map={b: i for i, b in enumerate(BANDS, start=1)}, **kw)
    np.testing.assert_array_equal(named.values, auto.values)
    # the passed map wins over the names: canopy-only map leaves CH/CBH/CBD untouched
    part = apply_treatment(lcp, band_map={"CC": 5, "FBFM40": 4}, **kw)
    np.testing.assert_array_equal(part.sel(band=5).values, auto.sel(band=5).values)
    np.testing.assert_array_equal(part.sel(band=6).values, lcp.sel(band=6).values)
