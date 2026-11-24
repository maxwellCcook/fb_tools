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


def _shift_origin(fp, dx):
    """Rewrite a raster's origin dx metres east, keeping its data — a same-shape, different-ground grid."""
    import rasterio
    from rasterio.transform import Affine
    with rasterio.open(fp) as src:
        a, prof = src.read(1), src.profile
    prof["transform"] = prof["transform"] * Affine.translation(dx / prof["transform"].a, 0)
    with rasterio.open(fp, "w", **prof) as dst:
        dst.write(a, 1)


def test_output_functions_reject_runs_on_different_grids(tmp_path):
    from fb_tools.spread.transmission import realization_transmission, shadow_probability
    bdir, tdir, trt, key = _synthetic_pair(tmp_path)
    bands, labels = distance_bands(trt, 30.0, edges=(100, 250, 500, 1000))
    blocks = np.ones(trt.shape, "int32")

    # same grid: all three run, and the per-run shadow feeds the ensemble
    summary, _, _ = realization_transmission(bdir, treated=trt, block_ids=blocks)
    realization_transmission(tdir, treated=trt, block_ids=blocks)
    assert summary["treated_ha_burned"] > 0
    with pytest.raises(ValueError, match="one arm"):                               # the shadow is a one-arm product
        shadow_probability([bdir, tdir], tmp_path / "ens", verbose=False)
    out = shadow_probability([bdir, tdir], tmp_path / "ens", arm="background", verbose=False)
    assert set(out) == {"ShadowProb", "ShadowGivenBurn"}
    import rasterio
    assert rasterio.open(bdir / "treatment_shadow.tif").tags()["arm"] == "background"
    from fb_tools.spread.transmission import transmission_batch
    res = pd.DataFrame({"arm": ["background", "coswap"], "out_dir": [bdir, tdir], "status": "ok"})
    with pytest.raises(ValueError, match="pass arm="):
        transmission_batch(res, treated=trt, block_ids=blocks, write_shadow=False)
    fires, _, _ = transmission_batch(res, arm="background", treated=trt, block_ids=blocks, write_shadow=False)
    assert fires.arm.tolist() == ["background"]

    with pytest.raises(ValueError, match="treated is not on the same grid"):       # masks are bare arrays: shape only
        realization_transmission(bdir, treated=trt[:-1], block_ids=blocks, write_shadow=False)
    with pytest.raises(ValueError, match="bands is not on the same grid"):
        lee_profile(bdir, tdir, bands=bands[:-1], labels=labels)

    # same shape, shifted half a cell: caught only by the transform
    for name in ("time_of_arrival", "fireline_intensity", "fire_type", "treatment_shadow"):
        _shift_origin(tdir / f"{name}.tif", 15.0)
    with pytest.raises(ValueError, match="not on the same grid: transform"):
        lee_profile(bdir, tdir, bands=bands, labels=labels)
    import json
    (tdir / "manifest.json").write_text(json.dumps({**key, "arm": "background"}))  # same arm, shifted grid
    with pytest.raises(ValueError, match="not on the same grid: transform"):
        shadow_probability([bdir, tdir], tmp_path / "ens2", verbose=False)

    # a shadow left over from an earlier grid is not read against new rasters
    _shift_origin(bdir / "treatment_shadow.tif", 15.0)
    with pytest.raises(ValueError, match="shadow"):
        lee_profile(bdir, bdir, bands=bands, labels=labels)


def test_shadow_bridges_skipped_cells_but_not_ember_landings():
    from fb_tools.spread.transmission import ember_descent
    n, dt = 120, 3.0
    r, c = np.mgrid[:n, :n]
    toa = (c * dt).astype("float64")                       # plane front moving east
    trt = np.zeros((n, n), bool); trt[:, 20:24] = True     # full-height strip: everything east is 1 by construction
    ign = np.zeros((n, n), bool); ign[:, 0] = True
    assert treatment_shadow(toa, trt)[:, 24:].min() == 1.0
    # a skipped cell: burns before all 8 neighbours while fire is 2 cells away (as the engine's raster does)
    skip = toa.copy(); skip[60, 50] = toa[60, 48] + 0.5
    old = treatment_shadow(skip, trt, bridge_cells=1)
    assert old[60, 50] == 0.0 and (old[:, 51:] < 0.5).sum() > 5            # unbridged: zeros trail downstream of it
    new = treatment_shadow(skip, trt)
    assert new[:, 24:].min() == pytest.approx(1.0)
    share, k = ember_descent(skip, ign)
    assert k == 0 and not share.any()
    # an ember landing: 8 cells ahead of the front, nothing earlier within 3 cells
    spot = toa.copy(); spot[60, 50] = toa[60, 42] + 0.5
    sh = treatment_shadow(spot, trt)
    assert sh[60, 50] == 0.0 and (sh[:, 51:] < 0.5).any()
    share, k = ember_descent(spot, ign)
    assert k == 1 and share[60, 50] == 1.0 and share[:, :42].max() == 0.0
    both = sh + share                                                       # every cell east of the strip is traced or flagged
    assert both[:, 24:].min() == pytest.approx(1.0, abs=1e-5)
    # no ignition mask: the earliest cells are the ignition
    assert ember_descent(spot)[1] == 1


def test_realization_transmission_reports_ember_descent(tmp_path):
    import rasterio
    from fb_tools.spread.transmission import realization_transmission
    n = 80
    r, c = np.mgrid[:n, :n]
    toa = np.hypot(r - 40, c - 10) * 3.0
    toa[40, 60] = toa[40, 52] + 0.5                                         # ember 8 cells ahead on the axis
    trt = np.zeros((n, n), bool); trt[:, 30:33] = True
    z = np.zeros((n, n))
    _write_run(tmp_path / "run", toa, z, z, {"arm": "a", "ignition_window": [39, 9, 3, 3]})
    s, _, _ = realization_transmission(tmp_path / "run", treated=trt, block_ids=np.ones((n, n), "int32"))
    assert s["n_ember_starts"] == 1 and 0 < s["ember_descent_ha"] < s["burned_ha"]
    with rasterio.open(tmp_path / "run" / "ember_descent.tif") as src:
        assert src.read(1)[40, 60] == 1.0


def test_shadow_agreement_scores_the_shadow_against_the_paired_change(tmp_path):
    from fb_tools.spread.transmission import shadow_agreement
    bdir, tdir, trt, key = _synthetic_pair(tmp_path)        # the treated arm is 90 min late exactly where the shadow >= 0.5
    a = shadow_agreement(bdir, tdir, trt, tolerances_min=(0.0, 60.0, 120.0)).set_index("tol_min")
    assert a.loc[0.0, "recall"] == 1.0 and a.loc[0.0, "precision"] == 1.0 and a.loc[0.0, "auc"] > 0.99
    assert a.loc[60.0, "n_changed"] == a.loc[0.0, "n_changed"]             # a 90 min delay clears 60 min
    assert a.loc[120.0, "n_changed"] == 25 and a.loc[120.0, "recall"] == 1.0   # only the 25 cells the treated fire never reached
    assert a.loc[120.0, "precision"] < 0.01
    assert a.baseline.iloc[0] == "background" and a.treated.iloc[0] == "coswap"
    # a shadow on the wrong ground carries no information
    off = np.zeros(trt.shape, bool); off[10:40, 20:24] = True
    b = shadow_agreement(bdir, tdir, off, tolerances_min=(0.0,))
    assert b.recall.iloc[0] < 0.2


def test_lead_time_is_clocked_from_the_upstream_treatment(tmp_path):
    from fb_tools.spread.transmission import realization_transmission, treatment_entry_time
    H, W, dt = 200, 300, 3.0
    toa = (np.mgrid[:H, :W][1] * dt).astype("float32")                      # plane front: 3 min per column
    trt = np.zeros((H, W), bool)
    trt[:100, 20:24] = True                                                 # north strip reached at 1 h
    trt[100:, 200:204] = True                                               # south strip reached at 10 h
    blocks = np.zeros((H, W), "int32"); blocks[30:60, 240:260] = 1; blocks[140:170, 240:260] = 2
    sh, entry = treatment_entry_time(toa, trt)
    assert entry[45, 240] == pytest.approx(60.0) and entry[155, 240] == pytest.approx(600.0)
    assert np.isnan(entry[199, 30]) and sh[199, 30] == 0 and sh[155, 100] < 1e-6   # south half is unshadowed before its strip
    z = np.zeros((H, W))
    _write_run(tmp_path / "run", toa, z, z, {"arm": "a"})
    s, b, _ = realization_transmission(tmp_path / "run", treated=trt, block_ids=blocks, write_shadow=False)
    b = b.set_index("block_id")
    assert b.arrival_h.tolist() == pytest.approx([12.0, 12.0])
    assert b.lead_h.tolist() == pytest.approx([11.0, 2.0])                  # was 11.0 for both
    assert s["trt_arrival_h"] == pytest.approx(1.0)
    # a wide unit is clocked from its first arrival, not from where the fire left it
    wide = np.zeros((H, W), bool); wide[:, 50:150] = True
    _, e2 = treatment_entry_time(toa, wide)
    assert e2[100, 200] == pytest.approx(50 * dt)
    # units given: two strips in one unit share that unit's first arrival
    ids = np.where(trt, 1, 0).astype("int32")
    _, e3 = treatment_entry_time(toa, trt, ids)
    assert e3[155, 240] == pytest.approx(60.0)
