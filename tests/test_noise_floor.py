"""
Tests for :mod:`fb_tools.spread.noise` — the P0.1 Monte Carlo null bands.

These pin the published P0.1 numbers.  If the calibration is ever re-measured
at production ``Duration``, these tests are the record of what changed.
"""

import numpy as np
import pandas as pd
import pytest

from fb_tools.spread.noise import (
    P01_REFERENCE,
    annotate_noise_floor,
    area_noise_floor,
    bp_noise_floor,
    describe_noise_floor,
    required_num_fires,
)


# ── Calibration ──────────────────────────────────────────────────────────────

def test_reference_reproduces_the_measurement_at_its_own_n():
    n = P01_REFERENCE["num_fires"]
    assert bp_noise_floor(n, "p50") == pytest.approx(0.01)
    assert bp_noise_floor(n, "p95") == pytest.approx(0.07)
    assert bp_noise_floor(n, "max") == pytest.approx(0.18)
    assert area_noise_floor(n) == pytest.approx(1.46)


@pytest.mark.parametrize("num_fires,expected", [(1000, 0.0221), (4000, 0.0111)])
def test_published_pixel_floors(num_fires, expected):
    """The figures quoted in the P0.1 record and CLAUDE.md."""
    assert bp_noise_floor(num_fires) == pytest.approx(expected, abs=5e-5)


def test_published_area_floor():
    assert area_noise_floor(4000) == pytest.approx(0.231, abs=5e-4)


def test_resolving_a_hundredth_of_bp_needs_about_4900_fires():
    assert required_num_fires(0.01) == 4900


def test_scaling_is_one_over_sqrt_n():
    """Quadrupling the fires must halve the floor."""
    assert bp_noise_floor(400) == pytest.approx(bp_noise_floor(100) / 2)
    assert area_noise_floor(1600) == pytest.approx(area_noise_floor(100) / 4)


def test_area_metric_is_far_tighter_than_the_pixel_metric():
    """
    The P0.1 conclusion that motivates leading with area-integrated metrics.

    The two floors are in different units — the pixel floor is absolute (BP
    units), the area floor is a coefficient of variation — so they are only
    comparable as *relative* precision.  Against a representative BP of 0.1 the
    pixel floor is ~11%, roughly 50x looser than the area estimator's ~0.23%.
    """
    typical_bp = 0.1
    pixel_relative_pct = 100.0 * bp_noise_floor(4000) / typical_bp
    area_relative_pct = area_noise_floor(4000)

    assert pixel_relative_pct == pytest.approx(11.1, abs=0.1)
    assert area_relative_pct == pytest.approx(0.231, abs=0.001)
    assert pixel_relative_pct > 40 * area_relative_pct


def test_difference_inflates_by_sqrt_two():
    single = area_noise_floor(4000, area=10_000)
    diff = area_noise_floor(4000, area=10_000, for_difference=True)
    assert diff == pytest.approx(single * np.sqrt(2))


def test_area_floor_scales_with_the_total():
    assert area_noise_floor(4000, area=10_000) == pytest.approx(
        10 * area_noise_floor(4000, area=1_000)
    )


def test_required_num_fires_round_trips():
    for target in (0.005, 0.01, 0.02, 0.05):
        n = required_num_fires(target)
        assert bp_noise_floor(n) <= target + 1e-9


# ── Argument validation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [0, -1, -100])
def test_non_positive_num_fires_raises(bad):
    with pytest.raises(ValueError, match="must be positive"):
        bp_noise_floor(bad)


def test_unknown_statistic_raises():
    with pytest.raises(ValueError, match="statistic must be"):
        bp_noise_floor(1000, statistic="p99")


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric must be"):
        required_num_fires(0.01, metric="volume")


def test_non_positive_target_raises():
    with pytest.raises(ValueError, match="target must be positive"):
        required_num_fires(0.0)


# ── annotate_noise_floor ─────────────────────────────────────────────────────

@pytest.fixture
def results_df():
    return pd.DataFrame({
        "TRT_ID":     [1, 2, 3],
        "dBP_mean":   [0.05, 0.004, -0.030],
        "TF_baseline": [1000.0, 1000.0, 1000.0],
        "dTF":        [50.0, 2.0, -40.0],
    })


def test_annotate_flags_effects_below_the_floor(results_df):
    out = annotate_noise_floor(results_df, num_fires=4000)
    floor = bp_noise_floor(4000)          # 0.0111

    assert (out["dBP_mean_noise_floor"] == floor).all()
    # 0.05 and -0.030 clear the floor; 0.004 does not.
    assert list(out["dBP_mean_resolvable"]) == [True, False, True]


def test_annotate_does_not_mutate_the_input(results_df):
    before = results_df.copy()
    annotate_noise_floor(results_df, num_fires=4000)
    pd.testing.assert_frame_equal(results_df, before)


def test_annotate_area_needs_a_total_column(results_df):
    with pytest.raises(ValueError, match="needs total_col"):
        annotate_noise_floor(results_df, 4000, delta_col="dTF", metric="area")


def test_annotate_area_uses_the_total_not_the_delta(results_df):
    out = annotate_noise_floor(
        results_df, 4000, delta_col="dTF", metric="area", total_col="TF_baseline"
    )
    expected = 1000.0 * area_noise_floor(4000, for_difference=True) / 100.0
    assert out["dTF_noise_floor"].iloc[0] == pytest.approx(expected)
    # dTF of 50 clears a ~3.3 floor; 2.0 does not.
    assert list(out["dTF_resolvable"]) == [True, False, True]


def test_annotate_rejects_missing_columns(results_df):
    with pytest.raises(KeyError, match="delta_col"):
        annotate_noise_floor(results_df, 4000, delta_col="nope")
    with pytest.raises(KeyError, match="total_col"):
        annotate_noise_floor(
            results_df, 4000, delta_col="dTF", metric="area", total_col="nope"
        )


def test_describe_mentions_the_calibration_and_the_caveat():
    text = describe_noise_floor(4000)
    assert "NumFires=4000" in text
    assert "P0.1" in text
    assert "confidence interval" in text
