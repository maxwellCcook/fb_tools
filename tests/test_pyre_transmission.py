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
