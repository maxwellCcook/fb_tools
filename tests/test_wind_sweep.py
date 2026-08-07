"""
Tests for build_wind_sweep_conditions — fixed fuel moisture, varying wind.

Depends on nothing in ``data/``; the scenario cache is built inline.
"""

import pandas as pd
import pytest

from fb_tools.models.scenarios import (
    build_scenarios,
    build_wind_sweep_conditions,
)


@pytest.fixture
def cache():
    """A minimal FlamMap scenario cache with two percentile bands."""
    return {
        "pyrome_id": "46",
        "percentiles": [0.5, 0.97],
        "wind_direction": -2,
        "scenarios": {
            "p50": {
                "FM_1hr": 6.6, "FM_10hr": 10.0, "FM_100hr": 12.5,
                "FM_herb": 89.9, "FM_woody": 98.1,
                "WIND_DIRECTION": -2, "WIND_SPEED": 12.3,
            },
            "p97": {
                "FM_1hr": 3.9, "FM_10hr": 6.1, "FM_100hr": 7.9,
                "FM_herb": 30.0, "FM_woody": 60.0,
                "WIND_DIRECTION": -2, "WIND_SPEED": 27.0,
            },
        },
    }


_FM_COLS = ["FM_1hr", "FM_10hr", "FM_100hr", "FM_herb", "FM_woody"]


def test_fuel_moisture_is_constant_across_the_sweep(cache):
    """The whole point: only wind varies, so FM must be identical everywhere."""
    df = build_wind_sweep_conditions(cache, [5, 15, 25, 35])
    for col in _FM_COLS:
        assert df[col].nunique() == 1, f"{col} varied across the sweep"
    assert df["FM_1hr"].iloc[0] == 3.9
    assert df["FM_woody"].iloc[0] == 60.0


def test_one_row_per_wind_speed_sorted_and_deduped(cache):
    df = build_wind_sweep_conditions(cache, [20, 5, 20, 10])
    assert df["WIND_SPEED"].tolist() == [5.0, 10.0, 20.0]


def test_scenario_names_are_unique_and_labelled(cache):
    df = build_wind_sweep_conditions(cache, [5, 10, 20])
    assert df["Scenario"].tolist() == ["Pct97_W5", "Pct97_W10", "Pct97_W20"]
    assert df["Scenario"].is_unique


def test_baseline_wind_speed_is_recorded(cache):
    """The cached wind speed is kept so the sweep can be placed against it."""
    df = build_wind_sweep_conditions(cache, [5, 40])
    assert (df["WIND_SPEED_baseline"] == 27.0).all()


def test_percentile_selects_the_right_fm_band(cache):
    df = build_wind_sweep_conditions(cache, [10], percentile=50)
    assert df["FM_1hr"].iloc[0] == 6.6
    assert df["Scenario"].iloc[0] == "Pct50_W10"


def test_percentile_accepts_string_key(cache):
    df = build_wind_sweep_conditions(cache, [10], percentile="p97")
    assert df["FM_1hr"].iloc[0] == 3.9


def test_wind_direction_defaults_to_cache_and_can_be_overridden(cache):
    assert (build_wind_sweep_conditions(cache, [10, 20])["WIND_DIRECTION"] == -2).all()
    upslope = build_wind_sweep_conditions(cache, [10, 20], wind_direction=-1)
    assert (upslope["WIND_DIRECTION"] == -1).all()
    fixed = build_wind_sweep_conditions(cache, [10], wind_direction=270)
    assert fixed["WIND_DIRECTION"].iloc[0] == 270


def test_zero_wind_is_allowed(cache):
    df = build_wind_sweep_conditions(cache, [0, 10])
    assert df["WIND_SPEED"].tolist() == [0.0, 10.0]


def test_empty_wind_speeds_raises(cache):
    with pytest.raises(ValueError, match="empty"):
        build_wind_sweep_conditions(cache, [])


def test_negative_wind_speed_raises(cache):
    with pytest.raises(ValueError, match="non-negative"):
        build_wind_sweep_conditions(cache, [10, -5])


def test_missing_percentile_raises(cache):
    with pytest.raises(KeyError, match="p99"):
        build_wind_sweep_conditions(cache, [10], percentile=99)


def test_output_feeds_build_scenarios(cache):
    """The sweep table must be directly consumable by build_scenarios."""
    speeds = [5, 10, 15, 20]
    conditions = build_wind_sweep_conditions(cache, speeds)
    df = build_scenarios(conditions, ["baseline.tif", "treated.tif"])

    assert len(df) == len(speeds) * 2
    assert {"Scenario", "LCP", "WIND_SPEED", "CROWN_FIRE_METHOD", "Outputs"} <= set(df.columns)
    # every (LCP, Scenario) pair is distinct, so run_batch output dirs never collide
    assert not df.duplicated(subset=["LCP", "Scenario"]).any()
    assert df["FM_1hr"].nunique() == 1


def test_no_cache_bookkeeping_columns_leak_through(cache):
    """erc_center / scenario_doy etc. must not reach the FlamMap input writer."""
    cache["scenarios"]["p97"].update(
        {"erc_center": 75.0, "scenario_doy": 236, "erc_quantile_band": [0.945, 0.995]}
    )
    df = build_wind_sweep_conditions(cache, [10])
    assert not {"erc_center", "scenario_doy", "erc_quantile_band"} & set(df.columns)
