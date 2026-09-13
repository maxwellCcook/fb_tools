"""Engine-free tests for spread/transmission.py (treatment shadow, value blocks) and edge_contact."""
import numpy as np
import pandas as pd
import pytest

from fb_tools.spread.transmission import treatment_shadow, structure_blocks, _min_by_zone, summarize_blocks


def radial_fire(n=400, ros_m_min=10.0, res_m=30.0):
    rr, cc = np.mgrid[:n, :n]
    return np.hypot(rr - n // 2, cc - n // 2) * res_m / ros_m_min      # minutes


def test_shadow_is_the_wedge_behind_the_treatment():
    toa = radial_fire()
    trt = np.zeros(toa.shape, bool)
    trt[190:210, 260:280] = True                  # 20-cell square east of the ignition
    sh = treatment_shadow(toa, trt)
    assert (sh[trt] == 1).all(), "treated cells carry a share of 1"
    assert sh[:, :260].sum() < 0.01 * sh.sum(), "no more than a lattice leak of the shadow lies upwind of the treatment"
    assert np.nanmin(toa[sh > 0]) >= toa[trt].min(), "every shadowed cell arrives after the fire entered the treatment"
    # a symmetric wedge behind the treatment, close to the geometric one (10/70 half-angle beyond r = 80 cells)
    r, c = np.mgrid[:400, :400]
    wedge = (np.abs(np.degrees(np.arctan2(200 - r, c - 200))) <= np.degrees(np.arctan(10 / 70))) & (np.hypot(r - 200, c - 200) > 80)
    assert 0.7 < sh.mean() / wedge.mean() < 1.8
    assert abs(sh[:200].sum() / sh.sum() - 0.5) < 0.05
    assert sh[:100, :].sum() + sh[300:, :].sum() < 0.01 * sh.sum(), "far off-axis rows hold at most a lattice leak"


def test_shadow_is_empty_when_the_fire_misses_the_treatment():
    toa = radial_fire(); toa[toa > 60] = np.nan               # a 1-hour fire
    trt = np.zeros(toa.shape, bool); trt[10:20, 10:20] = True
    assert not treatment_shadow(toa, trt).any()


def test_ember_landings_do_not_inherit_the_shadow():
    toa = radial_fire()
    trt = np.zeros(toa.shape, bool); trt[195:205, 250:260] = True
    toa[150:160, 320:330] = 1.0                    # a spot fire: its own local minimum beyond the treatment
    sh = treatment_shadow(toa, trt)
    assert not sh[150:160, 320:330].any()


def test_structure_blocks_counts_and_threshold():
    import geopandas as gpd
    rng = np.random.default_rng(0)
    xs = np.r_[rng.uniform(0, 1000, 30), rng.uniform(5000, 6000, 5)]
    ys = np.r_[rng.uniform(0, 1000, 30), rng.uniform(5000, 6000, 5)]
    pts = gpd.GeoDataFrame({"at_risk": [True] * 10 + [False] * 25}, geometry=gpd.points_from_xy(xs, ys), crs="EPSG:5070")
    b = structure_blocks(pts, "EPSG:5070", (0, 0, 10_000, 10_000), block_m=1000, min_count=10, flag_col="at_risk")
    assert len(b) == 1 and int(b.n_structures.iloc[0]) == 30 and int(b.n_at_risk.iloc[0]) == 10
    assert b.geometry.iloc[0].bounds == (0.0, 0.0, 1000.0, 1000.0)


def test_min_by_zone_and_block_summary():
    z = np.array([[1, 1, 2], [0, 2, 2]]); v = np.array([[5.0, 3.0, np.nan], [1.0, 8.0, 7.0]])
    assert np.allclose(_min_by_zone(v, z, 3), [3.0, 7.0, np.nan], equal_nan=True)
    import geopandas as gpd
    from shapely.geometry import box
    blocks = gpd.GeoDataFrame({"block_id": [1, 2], "n_structures": [12, 30]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:5070")
    rows = pd.DataFrame({"block_id": [1, 1, 2], "arrival_h": [10.0, 20.0, 5.0], "shadow_frac": [1.0, 0.0, 0.2], "lead_h": [2.0, np.nan, np.nan]})
    s = summarize_blocks(rows, blocks, n_fires=4)
    assert s.set_index("block_id").loc[1, "n_intercepted"] == 1 and s.set_index("block_id").loc[1, "p_reached"] == 0.5
    assert s.set_index("block_id").loc[2, "p_intercepted"] == 0.0


def test_edge_contact_reads_first_edge_day(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from fb_tools.spread import edge_contact
    toa = np.full((50, 50), np.nan, "float32"); toa[20:30, 20:50] = np.tile(np.linspace(0, 2 * 1440, 30), (10, 1))
    with rasterio.open(tmp_path / "time_of_arrival.tif", "w", driver="GTiff", height=50, width=50, count=1, dtype="float32",
                       crs="EPSG:5070", transform=from_origin(0, 1500, 30, 30), nodata=np.nan) as dst:
        dst.write(toa, 1)
    e = edge_contact([tmp_path])
    assert 1.8 < e.edge_day.iloc[0] <= 2.0 and abs(e.burned_ha.iloc[0] - 300 * 0.09) < 1e-6


# ── near field ────────────────────────────────────────────────────────────────

from fb_tools.spread.transmission import (changed_cells, distance_bands, near_field_zones, lee_profile,      # noqa: E402
                                          lee_profile_batch, summarize_lee_profile, near_field_headlines)


def test_changed_cells_and_distance_bands():
    a = {"fuel_model": np.full((10, 10), 165.0), "canopy_cover": np.full((10, 10), 0.5)}
    b = {k: v.copy() for k, v in a.items()}
    b["canopy_cover"][4:6, 4:6] = 0.2                     # a 2 x 2 canopy-only change
    ch = changed_cells(a, b, layers=("fuel_model", "canopy_cover"))
    assert ch.sum() == 4 and ch[4:6, 4:6].all()
    band, labels = distance_bands(ch, 30.0, edges=(30, 100))
    assert labels == ["in", "0-30", "30-100"]
    assert (band[4:6, 4:6] == 0).all()
    assert band[3, 4] == 1 and band[6, 5] == 1 and band[3, 3] == 2      # edge-adjacent 30 m; diagonal 42 m
    assert band[0, 0] == -1                                            # beyond 100 m
    with pytest.raises(ValueError):
        distance_bands(np.zeros((3, 3), bool), 30.0)


def test_near_field_zones_do_not_overlap():
    ch = np.zeros((20, 20), bool); ch[8:12, 8:12] = True
    z = near_field_zones(ch, 30.0, collar_m=60.0)
    assert not (z["trt"] & z["collar"]).any() and z["trt"].sum() == 16
    assert z["collar"].sum() == 8 * 8 - 16 - 4 - 8    # two rings minus the block; the 4 corners (85 m) and 8 knight cells (67 m) lie beyond 60 m
    assert z["collar"][7, 9] and z["collar"][6, 9] and not z["collar"][5, 9]


def _write_run(run_dir, toa, fi, fire_type, manifest):
    import json
    import rasterio
    from rasterio.transform import from_origin
    run_dir.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=toa.shape[0], width=toa.shape[1], count=1, dtype="float32",
                crs="EPSG:5070", transform=from_origin(0, toa.shape[0] * 30, 30, 30), nodata=np.nan)
    for name, a in (("time_of_arrival", toa), ("fireline_intensity", fi), ("fire_type", fire_type)):
        with rasterio.open(run_dir / f"{name}.tif", "w", **prof) as dst:
            dst.write(np.asarray(a, "float32"), 1)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return prof


def _synthetic_pair(tmp_path):
    """A radial fire meeting a 4-cell strip; the treated arm burns the strip cooler and everything in its lee 90 min later."""
    import rasterio
    toa_b = radial_fire(n=200)
    trt = np.zeros(toa_b.shape, bool); trt[85:115, 120:124] = True
    sh = treatment_shadow(toa_b, trt)
    lee = sh >= 0.5
    toa_t = toa_b + np.where(trt, 30.0, 0.0) + np.where(lee & ~trt, 90.0, 0.0)   # 90, not 60: float32 rounding would straddle the 1 h threshold
    toa_t[95:100, 145:150] = np.nan                      # ~630 m into the lee: cells the treated fire never reached (burned in one arm only)
    fi_b = np.full(toa_b.shape, 2000.0); fi_t = np.where(trt, 200.0, 2000.0)
    ft_b = np.full(toa_b.shape, 2.0); ft_t = np.where(trt, 1.0, 2.0)
    key = dict(ignition_id="ign_000", episode_id="ep_000", condition_id="central")
    bdir, tdir = tmp_path / "background", tmp_path / "coswap"
    prof = _write_run(bdir, toa_b, fi_b, ft_b, {**key, "arm": "background"})
    _write_run(tdir, toa_t, fi_t, ft_t, {**key, "arm": "coswap"})
    with rasterio.open(bdir / "treatment_shadow.tif", "w", **{**prof, "nodata": None}) as dst:
        dst.write(sh, 1)
    return bdir, tdir, trt, key


def test_lee_profile_reads_the_carried_delay_and_the_recovery(tmp_path):
    bdir, tdir, trt, key = _synthetic_pair(tmp_path)
    bands, labels = distance_bands(trt, 30.0, edges=(100, 250, 500, 1000))
    p = lee_profile(bdir, tdir, bands=bands, labels=labels).set_index(["lee", "band"])
    assert p.loc[(True, "in"), "d_fi"] == pytest.approx(-1800.0) and p.loc[(True, "in"), "d_crown"] == pytest.approx(-1.0)
    assert p.loc[(True, "in"), "cls_hand_bg"] == 0.0 and p.loc[(True, "in"), "cls_hand_tr"] == 1.0
    assert p.loc[(True, "0-100"), "d_fi"] == 0.0, "the intensity effect stops at the treatment boundary"
    for band in ("0-100", "250-500", "500-1000"):
        assert p.loc[(True, band), "d_toa_med"] == pytest.approx(1.5, abs=1e-3), "the lee arrives 90 min later at every distance"
        assert p.loc[(True, band), "frac_delayed_30m"] == 1.0 and p.loc[(True, band), "frac_delayed_1h"] == 1.0
        assert p.loc[(False, band), "d_toa_med"] == 0.0 and p.loc[(False, band), "frac_delayed_30m"] == 0.0
    assert (p.n_both >= 20).all()
    assert p.loc[(True, "500-1000"), "frac_disagree"] > 0 and p.loc[(True, "250-500"), "frac_disagree"] == 0
    assert p.baseline.iloc[0] == "background" and p.treated.iloc[0] == "coswap" and p.ignition_id.iloc[0] == "ign_000"


def test_lee_profile_batch_summary_and_headlines(tmp_path):
    bdir, tdir, trt, key = _synthetic_pair(tmp_path)
    bands, labels = distance_bands(trt, 30.0, edges=(100, 250, 500, 1000, 2000))
    results = pd.DataFrame([{**key, "arm": "background", "out_dir": bdir, "status": "ok"},
                            {**key, "arm": "coswap", "out_dir": tdir, "status": "ok"},
                            {**key, "ignition_id": "ign_001", "arm": "coswap", "out_dir": tdir, "status": "ok"}])   # unpaired: skipped
    rows = lee_profile_batch(results, baseline="background", treated="coswap", bands=bands, labels=labels)
    assert rows.ignition_id.nunique() == 1
    s = summarize_lee_profile(rows, labels, min_pairs=1)
    assert list(s[s.lee].band.astype(str)) == [l for l in labels if l in set(rows[rows.lee].band)]   # ordered as the labels
    assert s.set_index(["lee", "band"]).loc[(True, "in"), "cls_hand_tr"] == 1.0
    h = near_field_headlines(s, carried_band="1000-2000")
    assert h["recovery_band"] == "0-100"
    assert h["carried_band"] == "1000-2000" and h["carried_delay_h"] == pytest.approx(1.5, abs=1e-3) and h["carried_frac_delayed_1h"] == 1.0
    assert h["in_cls_hand_bg"] == 0.0 and h["in_cls_hand_tr"] == 1.0 and h["in_cls_uncontrollable_bg"] == 1.0
    assert summarize_lee_profile(pd.DataFrame(), labels).empty


def test_zone_masks_pass_dicts_through():
    from fb_tools.spread.pyre_outputs import zone_masks
    m = {"trt": np.zeros((3, 3), int)}
    out = zone_masks(m, None, (3, 3))
    assert list(out) == ["trt"] and out["trt"].dtype == bool and out["trt"].shape == (3, 3)
