"""
P0.5 — FSPro output readers and the edge-effect / domain-adequacy diagnostic.

Reference run is the pyrome 47 output in ``data/fspro_test/build_test``:
100 fires, 7 days, 90 m, on a 783 x 740 domain (70 x 67 km).  It establishes
the passing reference the plan calls for, and pins the one signal that fails —
``Duration = 7`` truncates growth well before the domain does.
"""

import numpy as np
import pytest

from fb_tools.spread.fspro_outputs import check_domain_adequacy, read_daily_acres

from conftest import FSPRO_RUN_BASE


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily_acres_path(fspro_run_dir):
    path = fspro_run_dir / f"{FSPRO_RUN_BASE}_DailyAcres.txt"
    if not path.exists():
        pytest.skip(f"missing {path.name}")
    return path


@pytest.fixture(scope="module")
def bp_path(fspro_run_dir):
    path = fspro_run_dir / f"{FSPRO_RUN_BASE}_BurnProb.asc"
    if not path.exists():
        pytest.skip(f"missing {path.name}")
    return path


@pytest.fixture(scope="module")
def ignition_path(fspro_run_dir):
    path = fspro_run_dir / f"{FSPRO_RUN_BASE}_Ignitions.asc"
    if not path.exists():
        pytest.skip(f"missing {path.name}")
    return path


@pytest.fixture(scope="module")
def growth(daily_acres_path):
    return read_daily_acres(daily_acres_path, duration=7)


@pytest.fixture(scope="module")
def adequacy(bp_path, ignition_path, growth):
    return check_domain_adequacy(
        bp_path, ignition=ignition_path, daily_acres=growth, verbose=False
    )


# ── read_daily_acres ──────────────────────────────────────────────────────────

def test_daily_acres_shape(growth):
    """700 lines = 100 fires x 7 days, per the plan's verification target."""
    assert growth["fire_id"].nunique() == 100
    assert len(growth) == 700
    assert set(growth.groupby("fire_id").size()) == {7}
    assert set(growth["day"].unique()) == set(range(1, 8))


def test_daily_acres_are_increments_not_cumulative(growth):
    """
    Column 2 is the area burned *that day*.

    If it were cumulative it would be non-decreasing within every fire; it is
    not, so the cumulative column has to be derived.
    """
    within_fire = growth.groupby("fire_id")["acres_day"]
    assert not all((g.diff().dropna() >= 0).all() for _, g in within_fire)
    assert (growth["acres_day"] > 0).all()


def test_cumulative_is_the_running_sum(growth):
    for _, g in growth.groupby("fire_id"):
        np.testing.assert_allclose(
            g["acres_cum"].to_numpy(), g["acres_day"].cumsum().to_numpy(), rtol=1e-9
        )


def test_hectare_columns(growth):
    np.testing.assert_allclose(
        growth["hectares_day"], growth["acres_day"] * 0.404686, rtol=1e-9
    )


def test_fire_blocks_split_on_day_restart(tmp_path):
    """Fire identity comes from the day counter restarting, not a fixed length."""
    path = tmp_path / "ragged_DailyAcres.txt"
    path.write_text("1,10.0\n2,20.0\n3,30.0\n1,5.0\n2,6.0\n1,1.0\n")
    df = read_daily_acres(path)
    assert df["fire_id"].tolist() == [0, 0, 0, 1, 1, 2]
    assert df.groupby("fire_id")["acres_cum"].max().tolist() == [60.0, 11.0, 1.0]


def test_empty_daily_acres_raises(tmp_path):
    path = tmp_path / "empty_DailyAcres.txt"
    path.write_text("\n\n")
    with pytest.raises(ValueError, match="no parseable rows"):
        read_daily_acres(path)


def test_missing_daily_acres_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_daily_acres(tmp_path / "nope_DailyAcres.txt")


# ── check_domain_adequacy ─────────────────────────────────────────────────────

def test_domain_geometry(adequacy):
    assert adequacy["domain_shape"] == (783, 740)
    assert adequacy["cell_size_m"] == pytest.approx(90.0)
    assert adequacy["domain_area_ha"] == pytest.approx(469_330, rel=0.01)


def test_no_burn_probability_at_the_domain_edge(adequacy):
    """
    The reference domain is ~20:1 by area against the burned footprint, so no
    burn-probability mass should reach the boundary band.
    """
    assert adequacy["edge_ok"] is True
    assert adequacy["edge_bp_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert adequacy["edge_burned_cells"] == 0


def test_fire_had_room_to_grow(adequacy):
    """Spread headroom above 1 means the fire never reached the boundary."""
    assert adequacy["headroom_ok"] is True
    assert adequacy["spread_headroom"] > 1.0
    assert adequacy["max_spread_km"] < adequacy["ignition_boundary_dist_km"]


def test_duration_is_what_binds_this_reference_run(adequacy):
    """
    Documents the Phase 0 finding: at ``Duration = 7`` every fire is still
    growing when the run stops, so the seven-day sizes are an artefact of the
    run length rather than of weather and fuels.  Domain sizing therefore has
    to be redone at the production ``Duration``.
    """
    assert adequacy["growth_ok"] is False
    assert adequacy["final_day_growth_share"] > 0.20
    assert adequacy["fires_growing_on_final_day"] == pytest.approx(1.0)
    assert adequacy["passed"] is False


def test_expected_area_burned_matches_mean_fire_size(adequacy, growth):
    """
    Empirical check on the estimator behind the transmission formalism.

    ``Sum BP x cell_area`` over the whole domain is the expected area burned
    from this ignition, so it must reconcile with the mean simulated fire size
    from ``_DailyAcres.txt``.  This is the same identity that
    ``TF_ij = Sum_{p in j} BP_p x cell_area`` relies on, restricted to j = the
    whole domain.
    """
    mean_ha = growth.groupby("fire_id")["acres_day"].sum().mean() * 0.404686
    assert adequacy["expected_area_burned_ha"] == pytest.approx(mean_ha, rel=0.15)


def test_signals_are_skipped_when_inputs_are_absent(bp_path):
    res = check_domain_adequacy(bp_path, verbose=False)
    assert res["edge_ok"] is True
    assert res["headroom_ok"] is None
    assert res["growth_ok"] is None
    assert res["spread_headroom"] is None
    assert res["final_day_growth_share"] is None
    # A skipped signal must not count against the verdict.
    assert res["passed"] is True


def test_boolean_ignition_mask_is_accepted(bp_path, ignition_path, adequacy):
    import rioxarray as rxr

    mask = (
        rxr.open_rasterio(ignition_path, masked=True)
        .squeeze("band", drop=True)
        .fillna(0)
        .values
        > 0
    )
    res = check_domain_adequacy(bp_path, ignition=mask, verbose=False)
    assert res["ignition_cells"] == adequacy["ignition_cells"]
    assert res["max_spread_km"] == pytest.approx(adequacy["max_spread_km"])


def test_mismatched_ignition_mask_raises(bp_path):
    with pytest.raises(ValueError, match="shape"):
        check_domain_adequacy(
            bp_path, ignition=np.zeros((10, 10), dtype=bool), verbose=False
        )
