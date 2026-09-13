"""HRRR episode wind: index parsing, gap policy, block cubes, runner plumbing — no engine, no network."""

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.crs import CRS

from fb_tools.models import pyre
from fb_tools.spread import episodes as ep
from fb_tools.weather import hrrr

IDX = """1:0:d=2016070120:REFC:entire atmosphere:anl:
2:1000:d=2016070120:UGRD:10 m above ground:anl:
3:2500:d=2016070120:VGRD:10 m above ground:anl:
4:4000:d=2016070120:UGRD:80 m above ground:anl:
"""


def test_uv10_byte_ranges_pick_only_10m_messages():
    assert hrrr._uv10_byte_ranges(IDX) == [(1000, 2499), (2500, 3999)]
    tail = IDX.replace("4:4000:d=2016070120:UGRD:80 m above ground:anl:\n", "")
    assert hrrr._uv10_byte_ranges(tail)[-1] == (2500, None)
    with pytest.raises(ValueError):
        hrrr._uv10_byte_ranges(IDX.replace("VGRD:10 m", "VGRD:80 m"))


def _fake_grid(lon0=-106.2, lat0=38.4, n=12, step=0.03):
    lon, lat = np.meshgrid(lon0 + step * np.arange(n), lat0 + step * np.arange(n))
    return lat, (lon + 360.0) % 360.0        # HRRR stores 0..360


@pytest.fixture
def fake_uv10(monkeypatch):
    lat, lon = _fake_grid()
    holes = set()

    def _uv10(dt, cache_dir, **kw):
        if pd.Timestamp(dt) in holes:
            raise FileNotFoundError(dt)
        h = pd.Timestamp(dt).hour
        u = np.full(lat.shape, float(h), dtype="float32")            # u = hour -> easy to check interpolation
        v = np.zeros_like(u)
        return {"u": u, "v": v, "lat": lat, "lon": lon}

    monkeypatch.setattr(hrrr, "hrrr_uv10", _uv10)
    return holes


def test_fetch_grid_crops_to_box_and_interpolates_short_gaps(fake_uv10, tmp_path):
    times = pd.date_range("2020-08-18 00:00", periods=8, freq="h")
    fake_uv10.update({times[3], times[4]})
    g = hrrr.fetch_hrrr_wind_grid(times, (-106.15, 38.45, -106.0, 38.6), tmp_path, n_threads=1, verbose=False)
    assert g["u"].shape[0] == 8 and g["lat"].shape == g["u"].shape[1:]
    assert g["lon"].min() >= -106.15 - 0.03 and g["lon"].max() <= -106.0 + 0.03 and g["lon"].max() < 0
    assert g["missing"] == [times[3], times[4]]
    assert np.allclose(g["u"][3], 3.0) and np.allclose(g["u"][4], 4.0)   # linear between hour 2 and 5


def test_fetch_grid_refuses_long_gap_and_pre_archive(fake_uv10, tmp_path):
    times = pd.date_range("2020-08-18 00:00", periods=12, freq="h")
    fake_uv10.update(set(times[2:7]))
    with pytest.raises(ValueError, match="gap of 5"):
        hrrr.fetch_hrrr_wind_grid(times, (-106.15, 38.45, -106.0, 38.6), tmp_path, n_threads=1, verbose=False)
    with pytest.raises(ValueError, match="archive starts"):
        hrrr.fetch_hrrr_wind_grid(pd.date_range("2013-07-01", periods=3, freq="h"),
                                  (-106.15, 38.45, -106.0, 38.6), tmp_path, n_threads=1, verbose=False)


def test_block_factors():
    assert ep.block_factors((1738, 1730)) == (79, 10)
    assert ep.block_factors((1000, 900)) == (100, 100)
    assert ep.block_factors((7, 9), target=5) == (1, 3)


@pytest.fixture
def landscape():
    # EPSG:5070 grid near the North Cottonwood domain, 200 x 300 cells at 30 m
    return Affine(30.0, 0.0, -902145.0, 0.0, -30.0, 1829955.0), CRS.from_epsg(5070), (200, 300)


def test_episode_wind_cube_nearest_units_and_shape(landscape):
    transform, crs, shape = landscape
    lat, lon = _fake_grid(lon0=-106.6, lat0=38.3, n=30, step=0.02)
    n_t = 3
    u = np.broadcast_to((lon - 360.0)[None], (n_t, *lon.shape)).astype("float32")   # u = lon (negative) -> from east-ish
    u = -u                                                                             # positive u: wind toward east = FROM 270
    v = np.zeros_like(u)
    cube = ep.episode_wind_cube({"u": u, "v": v, "lat": lat, "lon": (lon + 180) % 360 - 180},
                                transform, crs, shape, block=(100, 100))
    assert cube["wind_speed_10m"].shape == (n_t, 2, 3) and cube["upwind_direction"].shape == (n_t, 2, 3)
    assert np.allclose(cube["upwind_direction"], 270.0)
    assert cube["block"] == (100, 100) and cube["block_m"] == (3000.0, 3000.0)
    # nearest-cell check for block (0, 0) by brute force
    from pyproj import Transformer
    cx, cy = transform * (50.0, 50.0)
    hx, hy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(((lon + 180) % 360 - 180).ravel(), lat.ravel())
    j = np.argmin(np.hypot(hx - cx, hy - cy))
    assert cube["wind_speed_10m"][0, 0, 0] == pytest.approx(abs(u[0].ravel()[j]) * 3.6, rel=1e-5)
    with pytest.raises(ValueError, match="does not divide"):
        ep.episode_wind_cube({"u": u, "v": v, "lat": lat, "lon": lon}, transform, crs, shape, block=(7, 100))


def test_wind_cube_roundtrip_override_and_summary(tmp_path):
    n_t, frame = 5, pd.DataFrame({
        "tmp_c": 25.0, "ws_kmh": 10.0, "wd_from_deg": 200.0, "fm1": 0.04, "fm10": 0.05,
        "fm100": 0.08, "fm_herb": 0.4, "fm_woody": 0.7}, index=range(5))
    cube = {"wind_speed_10m": np.full((n_t, 2, 3), 30.0, "float32"),
            "upwind_direction": np.full((n_t, 2, 3), 270.0, "float32"),
            "block": (100, 100), "block_m": (3000.0, 3000.0), "source": "test", "n_hrrr_cells": 6}
    fp = ep.write_wind_cube(tmp_path / "ep_x_wind.npz", cube)
    back = ep.load_wind_cube(fp)
    assert back["block"] == (100, 100) and back["source"] == "test" and back["n_hrrr_cells"] == 6
    w = ep.weather_arrays(frame, wind_cube=fp)
    assert w["wind_speed_10m"].shape == (n_t, 2, 3) and w["temperature"].shape == (n_t,)
    assert (w["upwind_direction"] == 270.0).all()
    with pytest.raises(ValueError, match="hours"):
        ep.weather_arrays(frame.iloc[:3], wind_cube=fp)
    s = ep.wind_cube_summary(cube)
    assert list(s.columns) == ["hour", "ws_mean", "ws_p90", "ws_max", "wd_vecmean"]
    assert np.allclose(s.ws_mean, 30.0) and np.allclose(s.wd_vecmean, 270.0)


def test_build_runs_carries_one_cube_per_episode(tmp_path):
    arms = {"a": tmp_path / "a", "b": tmp_path / "b"}
    runs = pyre.build_runs(arms, {"ign_0": tmp_path / "i.shp"}, {"ep_0": tmp_path / "ep_0.csv", "ep_1": tmp_path / "ep_1.csv"},
                           {"c": {"crown_gate": 0.0}}, base_seed=1, out_root=tmp_path,
                           wind_cubes={"ep_0": tmp_path / "ep_0_wind.npz"})
    assert set(runs.columns) >= {"wind_cube"}
    assert (runs[runs.episode_id == "ep_0"].wind_cube == str(tmp_path / "ep_0_wind.npz")).all()
    assert (runs[runs.episode_id == "ep_1"].wind_cube == "").all()
    broken = runs.copy()
    broken.loc[(broken.arm == "b") & (broken.episode_id == "ep_0"), "wind_cube"] = "other.npz"
    with pytest.raises(AssertionError, match="CRN"):
        pyre.assert_crn_contract(broken, n_arms=2)
