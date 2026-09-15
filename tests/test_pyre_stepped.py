"""
Stepped runner and rule S1 (ELMFIRE model 0) on the runner's toy landscape. Requires the engine (fb_pyre env).

Contract: stepping keeps exact pairing (a sham arm still gives Δ = 0), the legacy one-call
path is untouched (same request hash, cached manifests still answer), suppression holds
line and reduces the fire, and it never removes what already burned.
"""

import json
import shutil

import numpy as np
import pytest
import rasterio

pytest.importorskip("pyretechnics", reason="run in the fb_pyre env")

from fb_tools.models import pyre                                  # noqa: E402
from fb_tools.spread.pyre_outputs import read_realization         # noqa: E402
from fb_tools.suppression.containment import SectorContainment, sector_index, ELMFIRE_MODEL0  # noqa: E402
from tests.test_pyre_runner import layers, weather, ignition, restore_gate, HOURS, N, TRANSFORM, CRS, SPOT  # noqa: E402,F401

SUPP = {"rule": "elmfire0", "max_containment_per_day": 400.0}     # fast so a 8 h toy fire shows it


@pytest.fixture(scope="module")
def sdi_layers(layers, tmp_path_factory):
    """The toy layers plus an sdi.tif (nodata in one corner to exercise the fill)."""
    d = tmp_path_factory.mktemp("layers") / "with_sdi"
    shutil.copytree(layers, d)
    sdi = np.full((N, N), 40, dtype="int16")
    sdi[:10, :10] = -9999
    with rasterio.open(d / "sdi.tif", "w", driver="GTiff", height=N, width=N, count=1, dtype="int16",
                       transform=TRANSFORM, crs=CRS, nodata=-9999) as dst:
        dst.write(sdi, 1)
    return d


def _run(layers, weather, ignition, out, seed, **kw):
    kw.setdefault("crown_gate", 0.0 if pyre.is_patched_engine() else None)
    man = pyre.run_realization(layers, weather, ignition, out, seed=seed, duration_h=HOURS,
                               spot_config=SPOT, overwrite=True, **kw)
    return man, read_realization(out)


def _same(a, b):
    return all(np.array_equal(a[k], b[k], equal_nan=True) for k in pyre.OUTPUT_LAYERS)


def test_sector_index_quadrants():
    s = sector_index([0, 5, 10, 5], [5, 10, 5, 0], (5, 5), 4)      # north, east, south, west of the centre
    assert list(s) == [1, 0, 3, 2]


def test_rule_target_grows_faster_when_fire_is_slow_and_sdi_low():
    slow = SectorContainment(np.full((4, 4), 10.0), 30.0, (2, 2))
    fast = SectorContainment(np.full((4, 4), 250.0), 30.0, (2, 2))
    burned = np.zeros((4, 4), bool); burned[1:3, 1:3] = True
    fr = (np.array([0, 3]), np.array([1, 2]))
    slow.step(1, 1, burned, fr); fast.step(1, 1, burned, fr)
    assert slow.target > fast.target > 0
    assert slow.log[-1]["sdi_bar"] == pytest.approx(0.1) and fast.log[-1]["sdi_bar"] == pytest.approx(2.5)


def test_rule_holds_slowest_sectors_first_and_keeps_them():
    r = SectorContainment(np.zeros((20, 20)), 30.0, (10, 10), params={"max_containment_per_day": 1200.0})
    burned = np.zeros((20, 20), bool); burned[10:12, 10:16] = True                 # growth to the east only
    fr = (np.array([9, 9, 12, 12, 10, 10]), np.array([12, 14, 12, 14, 17, 3]))     # frontier north, south, east, west
    rr, cc = r.step(1, 1, burned, fr)
    assert r.target == pytest.approx(0.5, rel=0.01)
    assert 17 not in cc                                                              # the fast (east) sector is not held first
    frozen_before = r.frozen.copy()
    rr2, cc2 = r.step(2, 1, np.zeros((20, 20), bool), fr)
    assert (r.frozen | frozen_before).sum() == r.frozen.sum() and r.frozen.sum() >= frozen_before.sum()
    assert len(rr2) >= len(rr)


def test_legacy_path_hash_and_outputs_unchanged(layers, weather, ignition, tmp_path):
    """No step_h/suppression: the request hash has no new keys, so old manifests still answer."""
    man, _ = _run(layers, weather, ignition, tmp_path / "a", seed=5)
    again = pyre.run_realization(layers, weather, ignition, tmp_path / "a", seed=5, duration_h=HOURS,
                                 spot_config=SPOT, crown_gate=man["crown_gate"] if pyre.is_patched_engine() else None)
    assert again["request_hash"] == man["request_hash"] and again["wall_s"] == man["wall_s"]
    assert man["step_h"] is None and man["n_steps"] == 1 and man["suppression"] is None


def test_stepped_sham_arm_delta_is_exactly_zero(layers, weather, ignition, tmp_path):
    sham = tmp_path / "sham_layers"
    shutil.copytree(layers, sham)
    _, a = _run(layers, weather, ignition, tmp_path / "a", seed=11, step_h=1.0)
    _, b = _run(sham, weather, ignition, tmp_path / "b", seed=11, step_h=1.0)
    assert _same(a, b)


def test_stepped_run_is_a_fire_of_similar_size(layers, weather, ignition, tmp_path):
    m1, one = _run(layers, weather, ignition, tmp_path / "one", seed=11)
    m2, stp = _run(layers, weather, ignition, tmp_path / "stp", seed=11, step_h=1.0)
    assert m2["n_steps"] == HOURS and m2["step_h"] == 1.0
    assert m1["request_hash"] != m2["request_hash"]
    n1, n2 = np.isfinite(one["time_of_arrival"]).sum(), np.isfinite(stp["time_of_arrival"]).sum()
    assert n1 > 0 and abs(n2 - n1) / n1 < 0.35
    assert m2["stop_time_min"] == pytest.approx(HOURS * 60.0)


def test_stepped_different_seed_differs(layers, weather, ignition, tmp_path):
    _, a = _run(layers, weather, ignition, tmp_path / "a", seed=11, step_h=1.0)
    _, b = _run(layers, weather, ignition, tmp_path / "b", seed=12, step_h=1.0)
    assert not _same(a, b)


def test_suppression_holds_line_and_shrinks_the_fire(sdi_layers, weather, ignition, tmp_path):
    m0, free = _run(sdi_layers, weather, ignition, tmp_path / "free", seed=11, step_h=1.0)
    m1, supp = _run(sdi_layers, weather, ignition, tmp_path / "supp", seed=11, suppression=SUPP)
    assert m1["step_h"] == 1.0 and 1 < m1["n_steps"] <= HOURS and m1["suppression"] == SUPP   # a contained fire stops early
    assert m1["held_cells"] > 0 and 0 < m1["containment_target"] <= 1 and len(m1["suppression_log"]) == m1["n_steps"]
    assert m1["n_steps"] == HOURS or m1["pending_spot_cells"] == 0
    assert m1["sdi_fill"] == pytest.approx(0.4, abs=1e-6) and m1["sdi_known_share"] == pytest.approx(1 - 100 / N ** 2, abs=1e-4)
    with rasterio.open(tmp_path / "supp" / "held_line.tif") as src:
        held = src.read(1)
    assert np.isfinite(held).sum() == m1["held_cells"] and np.nanmin(held) >= 1
    burned = np.isfinite(supp["time_of_arrival"])
    assert (burned & np.isfinite(held)).mean() < 0.1 * np.isfinite(held).mean()   # held cells burn only by embers cast before the line went in
    assert m1["burned_ha"] < m0["burned_ha"]
    log = m1["suppression_log"]
    assert all(b["target"] >= a["target"] for a, b in zip(log, log[1:]))   # target never falls in this setup
    assert m0["request_hash"] != m1["request_hash"]


FIRELINES = {"rule": "pyre_firelines", "construction_rate_m_h": 2000.0}


def test_package_rule_builds_line_and_shrinks_the_fire(sdi_layers, weather, ignition, tmp_path):
    m0, free = _run(sdi_layers, weather, ignition, tmp_path / "free", seed=11, step_h=1.0)
    m1, supp = _run(sdi_layers, weather, ignition, tmp_path / "fl", seed=11, suppression=FIRELINES)
    assert m1["suppression"] == FIRELINES and m1["containment_target"] is None
    assert m1["held_cells"] > 0 and len(m1["suppression_log"]) == m1["n_steps"]
    with rasterio.open(tmp_path / "fl" / "held_line.tif") as src:
        held = src.read(1)
    assert np.isfinite(held).sum() == m1["held_cells"]
    assert (np.isfinite(supp["time_of_arrival"]) & np.isfinite(held)).sum() < 0.1 * m1["held_cells"]
    assert m1["burned_ha"] < m0["burned_ha"]
    assert m1["suppression_log"][-1]["line_m"] == pytest.approx(m1["held_cells"] * 30.0)


def test_package_rule_gate_and_priority_options(sdi_layers, weather, ignition, tmp_path):
    a, _ = _run(sdi_layers, weather, ignition, tmp_path / "a", seed=11, suppression={**FIRELINES, "max_flame_length_m": 0.01})
    b, _ = _run(sdi_layers, weather, ignition, tmp_path / "b", seed=11, suppression={**FIRELINES, "priority": "flame_length"})
    assert a["held_cells"] < b["held_cells"] and a["held_cells"] < 5           # a 1 cm gate makes almost every pair unbuildable
    with pytest.raises(TypeError):
        _run(sdi_layers, weather, ignition, tmp_path / "c", seed=1, suppression={**FIRELINES, "bogus": 1})


def test_package_rule_pairing_is_exact_across_arms(sdi_layers, weather, ignition, tmp_path):
    sham = tmp_path / "sham"
    shutil.copytree(sdi_layers, sham)
    _, a = _run(sdi_layers, weather, ignition, tmp_path / "a", seed=11, suppression=FIRELINES)
    _, b = _run(sham, weather, ignition, tmp_path / "b", seed=11, suppression=FIRELINES)
    assert _same(a, b)


def test_suppression_pairing_is_exact_across_arms(sdi_layers, weather, ignition, tmp_path):
    sham = tmp_path / "sham"
    shutil.copytree(sdi_layers, sham)
    _, a = _run(sdi_layers, weather, ignition, tmp_path / "a", seed=11, suppression=SUPP)
    _, b = _run(sham, weather, ignition, tmp_path / "b", seed=11, suppression=SUPP)
    assert _same(a, b)


def test_suppression_needs_sdi(layers, weather, ignition, tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(layers, weather, ignition, tmp_path / "x", seed=1, suppression=SUPP)


def test_unknown_rule_or_parameter_is_rejected(sdi_layers, weather, ignition, tmp_path):
    with pytest.raises(ValueError):
        _run(sdi_layers, weather, ignition, tmp_path / "x", seed=1, suppression={"rule": "nope"})
    with pytest.raises(TypeError):
        _run(sdi_layers, weather, ignition, tmp_path / "y", seed=1, suppression={"rule": "elmfire0", "bogus": 1})


def test_manifest_round_trips_through_json(sdi_layers, weather, ignition, tmp_path):
    m, _ = _run(sdi_layers, weather, ignition, tmp_path / "s", seed=3, suppression=SUPP)
    on_disk = json.loads((tmp_path / "s" / "manifest.json").read_text())
    assert on_disk["suppression_log"][-1]["target"] == pytest.approx(m["containment_target"], abs=1e-4)
    assert set(ELMFIRE_MODEL0) >= set(on_disk["suppression"]) - {"rule"}
