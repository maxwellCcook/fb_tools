"""
G3a: exact pairing on the realization runner. Requires the engine (fb_pyre env).

A realization is (ignition, episode, condition) with an arm-independent seed.
Pairing is exact in the null: a byte-identical sham arm must give Δ = 0 at
every pixel, the property FSPro's SPOTTING_SEED failed at 66% of burned cells.
"""

import shutil

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("pyretechnics", reason="run in the fb_pyre env")

from fb_tools.models import pyre                      # noqa: E402
from fb_tools.spread.pyre_outputs import read_realization  # noqa: E402

N, RES = 60, 30.0

TRANSFORM = from_origin(400000, 4400000, RES, RES)
CRS = "EPSG:32613"
HOURS = 8


@pytest.fixture(scope="module", autouse=True)
def restore_gate():
    """run_realization sets the process-global crown gate; put it back for later tests."""
    before = pyre.crown_cover_threshold()
    yield
    if pyre.is_patched_engine():
        pyre.set_crown_cover_threshold(before)


@pytest.fixture(scope="module")
def layers(tmp_path_factory):
    """TU5 timber at CC 0.25 with a 3-cell non-burnable strip the fire must spot across."""
    d = tmp_path_factory.mktemp("layers") / "untreated"
    d.mkdir()
    fuel = np.full((N, N), 165, dtype="int16")
    fuel[:, 30:33] = 91
    vals = {"elevation": 2500.0, "slope": 0.15, "aspect": 90.0, "canopy_cover": 0.25,
            "canopy_height": 18.0, "canopy_base_height": 1.0, "canopy_bulk_density": 0.10}
    prof = dict(driver="GTiff", height=N, width=N, count=1, transform=TRANSFORM, crs=CRS)
    for k, v in vals.items():
        with rasterio.open(d / f"{k}.tif", "w", dtype="float32", **prof) as dst:
            dst.write(np.full((N, N), v, dtype="float32"), 1)
    with rasterio.open(d / "fuel_model.tif", "w", dtype="int16", **prof) as dst:
        dst.write(fuel, 1)
    return d


@pytest.fixture(scope="module")
def weather():
    f = lambda v: np.full(HOURS, v, dtype="float32")  # noqa: E731
    return {"temperature": f(30.0), "wind_speed_10m": f(42.8), "upwind_direction": f(270.0),
            "fuel_moisture_dead_1hr": f(0.04), "fuel_moisture_dead_10hr": f(0.055),
            "fuel_moisture_dead_100hr": f(0.066), "fuel_moisture_live_herbaceous": f(0.593),
            "fuel_moisture_live_woody": f(0.787)}


@pytest.fixture(scope="module")
def ignition():
    import geopandas as gpd
    from shapely.geometry import box
    x0, y0 = TRANSFORM * (10, 30)
    fp = gpd.GeoDataFrame(geometry=[box(x0, y0 - 3 * RES, x0 + 3 * RES, y0)], crs=CRS)
    return pyre.ignition_matrix(fp, TRANSFORM, (N, N))


SPOT = {**pyre.SPOT_CENTRAL, "firebrands_per_unit_heat": 1e-6}


def _run(layers, weather, ignition, out, seed, **kw):
    kw.setdefault("crown_gate", 0.0 if pyre.is_patched_engine() else None)
    pyre.run_realization(layers, weather, ignition, out, seed=seed, duration_h=HOURS,
                         spot_config=SPOT, overwrite=True, **kw)
    return read_realization(out)


def _same(a, b):
    return all(np.array_equal(a[k], b[k], equal_nan=True) for k in pyre.OUTPUT_LAYERS)


def test_ignition_window_marks_footprint_negative(ignition):
    (r0, c0), win = ignition
    assert (r0, c0) == (30, 10) and win.shape == (3, 3) and (win == -1).all()


def test_repeatable_and_spotting_active(layers, weather, ignition, tmp_path):
    a = _run(layers, weather, ignition, tmp_path / "a", seed=11)
    b = _run(layers, weather, ignition, tmp_path / "b", seed=11)
    assert _same(a, b)
    burned = np.isfinite(a["time_of_arrival"])
    assert burned[:, 33:].any(), "fire never crossed the barrier: spotting not exercised"


def test_sham_arm_delta_is_exactly_zero(layers, weather, ignition, tmp_path):
    sham = tmp_path / "sham"
    shutil.copytree(layers, sham)
    a = _run(layers, weather, ignition, tmp_path / "a", seed=11)
    s = _run(sham, weather, ignition, tmp_path / "s", seed=11)
    assert _same(a, s)


def test_different_seed_differs(layers, weather, ignition, tmp_path):
    a = _run(layers, weather, ignition, tmp_path / "a", seed=11)
    b = _run(layers, weather, ignition, tmp_path / "b", seed=12)
    assert not _same(a, b)


def test_worker_count_is_irrelevant(layers, weather, ignition, tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    x0, y0 = TRANSFORM * (10, 30)
    fp = tmp_path / "ign.shp"
    gpd.GeoDataFrame(geometry=[box(x0, y0 - 3 * RES, x0 + 3 * RES, y0)], crs=CRS).to_file(fp)
    ep_csv = tmp_path / "ep.csv"
    pd.DataFrame({"hour": range(HOURS), "tmp_c": 30.0, "ws_kmh": 42.8, "wd_from_deg": 270.0,
                  "fm1": 0.04, "fm10": 0.055, "fm100": 0.066, "fm_herb": 0.593,
                  "fm_woody": 0.787}).to_csv(ep_csv, index=False)
    sham = tmp_path / "sham"
    shutil.copytree(layers, sham)
    cond = {"c": {"spot_config": SPOT, "crown_gate": 0.0 if pyre.is_patched_engine() else None}}
    outs = []
    for n_workers in (1, 2):
        runs = pyre.build_runs({"u": layers, "sham": sham}, {"i": fp}, {"e": ep_csv}, cond,
                               base_seed=5, out_root=tmp_path / f"w{n_workers}")
        res = pyre.run_batch(runs, duration_h=HOURS, n_workers=n_workers, verbose=False)
        assert (res.status == "ok").all(), res.error.tolist()
        outs.append([read_realization(d) for d in res.out_dir])
    assert _same(outs[0][0], outs[1][0]) and _same(outs[0][0], outs[0][1])


def test_burn_landscape_head_fire_vs_azimuth(layers, weather, tmp_path):
    gate = 0.0 if pyre.is_patched_engine() else None
    head = pyre.burn_landscape(layers, weather, tmp_path / "head", crown_gate=gate)
    az = pyre.burn_landscape(layers, weather, tmp_path / "az", azimuth=0.0, crown_gate=gate)
    assert head["product"] == "burn_landscape" and head["surface_lw_ratio_model"] == "behave"
    assert head["weather"]["wind_speed_10m"] == pytest.approx(42.8)
    with rasterio.open(tmp_path / "head" / "flame_length.tif") as a, \
            rasterio.open(tmp_path / "az" / "flame_length.tif") as b:
        fl_head, fl_az = a.read(1), b.read(1)
    assert np.isnan(fl_head[:, 30:33]).all()           # non-burnable strip
    assert np.isfinite(fl_head[:, :30]).all()
    assert (fl_az[:, :30] < fl_head[:, :30]).all()     # burning toward N with W wind is off-axis
    with pytest.raises(TypeError, match="unknown fire-behavior option"):
        pyre.burn_landscape(layers, weather, tmp_path / "x", bogus=1)


def test_spot_ignition_probability_is_monotone_and_bounded():
    p = pyre.spot_ignition_probability([20.0, 30.0, 40.0], 0.05)
    assert (np.diff(p) > 0).all() and ((p > 0) & (p < 1)).all()
    assert pyre.spot_ignition_probability(30.0, 0.05) > pyre.spot_ignition_probability(30.0, 0.15)
    far = pyre.spot_ignition_probability(30.0, 0.05, distance_m=500.0, decay_distance=500.0)
    assert far == pytest.approx(pyre.spot_ignition_probability(30.0, 0.05) / np.e)
