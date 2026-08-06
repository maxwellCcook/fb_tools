"""
Tests for :func:`fb_tools.weather.gridmet.build_current_erc_values` (P1.2).

``CurrentERCValues`` is the season-to-date ERC stream leading up to ignition.
FSPro reads it **positionally** from fire-season day 1 (April 1, spec p.5), so
the pre-P1.2 signature — an arbitrary interior window given by ``start_doy`` and
``n_days`` — silently shifted the whole antecedent stream (defect #3).  Passing
``start_doy=91, n_days=79`` fed FSPro season days 91–169 while it read them as
days 1–79, conditioning the ERC generator on the monsoon decline.

These tests pin the alignment, the spec window, and the four stream modes.
"""

import json

import numpy as np
import pytest

from fb_tools.weather.gridmet import build_current_erc_values

from conftest import PYROME_ERC_DIR

PYROME = "47"
N_SEASON_DAYS = 214


@pytest.fixture(scope="module")
def historic():
    """``({pyrome: (n_years, 214)}, years)`` from the cached climatology."""
    cache = PYROME_ERC_DIR / f"pyrome_{PYROME}_gridmet.json"
    if not cache.exists():
        pytest.skip(f"no cached ERC climatology for pyrome {PYROME}")
    meta = json.loads(cache.read_text())
    arr = np.asarray(meta["HistoricERCValues"], dtype=float)
    return {PYROME: arr}, meta.get("years")


# ── Alignment: the defect #3 fix ──────────────────────────────────────────────

def test_stream_starts_at_season_day_one(historic):
    """
    Day 1 of the array must be fire-season day 1, not the start of an interior
    window.  Compared against the per-day median taken directly from column 0.
    """
    hist, years = historic
    values = build_current_erc_values(
        hist, ignition_season_day=80, mode="median", years=years
    )[PYROME]
    expected_day_1 = np.round(np.nanmedian(hist[PYROME][:, 0]))
    assert values[0] == expected_day_1


def test_length_is_ignition_day_minus_one(historic):
    hist, years = historic
    for day in (40, 80, 150):
        values = build_current_erc_values(
            hist, ignition_season_day=day, mode="median", years=years
        )[PYROME]
        assert len(values) == day - 1


def test_deprecated_start_doy_rejected(historic):
    """
    The old call has to fail loudly — silently reinterpreting it would change
    the scenario without the caller noticing.  The message names the
    replacement.
    """
    hist, _ = historic
    with pytest.raises(ValueError, match="start_doy"):
        build_current_erc_values(hist, start_doy=91, n_days=79)


def test_deprecated_n_days_still_maps_to_a_length(historic):
    """``n_days`` alone is unambiguous: it is the stream length."""
    hist, years = historic
    via_old = build_current_erc_values(hist, n_days=79, mode="median", years=years)[PYROME]
    via_new = build_current_erc_values(
        hist, ignition_season_day=80, mode="median", years=years
    )[PYROME]
    assert np.array_equal(via_old, via_new)


def test_contradictory_length_keywords_rejected(historic):
    hist, _ = historic
    with pytest.raises(ValueError, match="contradicts"):
        build_current_erc_values(hist, ignition_season_day=80, n_days=50)


# ── Spec window: MaxLag <= NumWxCurrYear < NumWxPerYear - Duration ────────────

def test_rejects_ignition_before_max_lag(historic):
    hist, years = historic
    with pytest.raises(ValueError, match="MaxLag"):
        build_current_erc_values(
            hist, ignition_season_day=20, years=years, max_lag=30, duration=7
        )


def test_rejects_ignition_too_late_for_duration(historic):
    hist, years = historic
    with pytest.raises(ValueError, match="NumWxPerYear"):
        build_current_erc_values(
            hist, ignition_season_day=200, years=years, max_lag=30, duration=21
        )


def test_duration_21_window_endpoints(historic):
    """
    At Duration=21 the plan states the admissible ignition days are 31–193.
    Check both edges hold and that just outside each edge fails.
    """
    hist, years = historic
    for day in (31, 193):
        build_current_erc_values(
            hist, ignition_season_day=day, years=years, max_lag=30, duration=21
        )
    for day in (30, 194):
        with pytest.raises(ValueError):
            build_current_erc_values(
                hist, ignition_season_day=day, years=years, max_lag=30, duration=21
            )


def test_validation_can_be_disabled(historic):
    hist, years = historic
    values = build_current_erc_values(
        hist, ignition_season_day=20, years=years, max_lag=30, validate=False
    )[PYROME]
    assert len(values) == 19


def test_rejects_length_beyond_the_record(historic):
    hist, years = historic
    with pytest.raises(ValueError, match="antecedent days"):
        build_current_erc_values(
            hist, ignition_season_day=N_SEASON_DAYS + 50, years=years, validate=False
        )


def test_rejects_ignition_on_day_one(historic):
    hist, _ = historic
    with pytest.raises(ValueError, match="at least 2"):
        build_current_erc_values(hist, ignition_season_day=1, validate=False)


# ── Modes ─────────────────────────────────────────────────────────────────────

def test_unknown_mode_rejected(historic):
    hist, _ = historic
    with pytest.raises(ValueError, match="mode"):
        build_current_erc_values(hist, ignition_season_day=80, mode="climatology")


def test_analog_year_is_a_real_observed_row(historic):
    """
    The whole point of the analog year is that it is an observed sequence, not
    a cross-year statistic.  It must match one row of the historic array
    exactly.
    """
    hist, years = historic
    meta = build_current_erc_values(
        hist, ignition_season_day=80, mode="analog_year", years=years, return_meta=True
    )[PYROME]
    window = np.round(hist[PYROME][:, :79]).astype(int)
    assert any(np.array_equal(meta["values"], row) for row in window)
    assert meta["analog_year"] in years


def test_analog_year_picks_the_driest_antecedent_season(historic):
    hist, years = historic
    meta = build_current_erc_values(
        hist, ignition_season_day=80, mode="analog_year", years=years, return_meta=True
    )[PYROME]
    totals = np.nansum(hist[PYROME][:, :79], axis=1)
    assert meta["analog_year"] == years[int(np.argmax(totals))]


def test_analog_year_preserves_more_variance_than_the_median(historic):
    """
    A cross-year median averages away day-to-day variability and
    autocorrelation.  That is exactly why it is no longer the default.
    """
    hist, years = historic
    analog = build_current_erc_values(
        hist, ignition_season_day=80, mode="analog_year", years=years
    )[PYROME]
    median = build_current_erc_values(
        hist, ignition_season_day=80, mode="median", years=years
    )[PYROME]
    assert analog.std() > median.std()


def test_explicit_analog_year_is_honored(historic):
    hist, years = historic
    target = years[len(years) // 2]
    meta = build_current_erc_values(
        hist, ignition_season_day=80, mode="analog_year", years=years,
        analog_year=target, return_meta=True,
    )[PYROME]
    assert meta["analog_year"] == target
    expected = np.round(hist[PYROME][years.index(target), :79]).astype(int)
    assert np.array_equal(meta["values"], expected)


def test_unknown_analog_year_rejected(historic):
    hist, years = historic
    with pytest.raises(ValueError, match="analog_year"):
        build_current_erc_values(
            hist, ignition_season_day=80, mode="analog_year",
            years=years, analog_year=1802,
        )


def test_analog_year_by_name_requires_years(historic):
    hist, _ = historic
    with pytest.raises(ValueError, match="years"):
        build_current_erc_values(
            hist, ignition_season_day=80, mode="analog_year", analog_year=2018
        )


def test_analog_year_without_years_still_selects_a_row(historic):
    """Selection is by accumulation, so the labels are optional."""
    hist, _ = historic
    meta = build_current_erc_values(
        hist, ignition_season_day=80, mode="analog_year", return_meta=True
    )[PYROME]
    assert meta["analog_year"] is None
    assert len(meta["values"]) == 79


def test_percentile_mode_is_more_severe_than_median(historic):
    hist, years = historic
    p80 = build_current_erc_values(
        hist, ignition_season_day=80, mode="percentile", percentile=80.0, years=years
    )[PYROME]
    median = build_current_erc_values(
        hist, ignition_season_day=80, mode="median", years=years
    )[PYROME]
    assert p80.mean() > median.mean()
    assert np.all(p80 >= median - 1)   # allow for independent rounding


def test_median_mode_matches_the_legacy_computation(historic):
    """``median`` reproduces the pre-P1.2 statistic — correctly aligned now."""
    hist, years = historic
    values = build_current_erc_values(
        hist, ignition_season_day=80, mode="median", years=years
    )[PYROME]
    expected = np.round(np.nanmedian(hist[PYROME][:, :79], axis=0)).astype(int)
    assert np.array_equal(values, expected)


def test_observed_mode_uses_the_supplied_stream(historic):
    hist, years = historic
    stream = np.arange(100, 200, dtype=float)
    values = build_current_erc_values(
        hist, ignition_season_day=80, mode="observed", observed=stream, years=years
    )[PYROME]
    assert np.array_equal(values, np.arange(100, 179))


def test_observed_mode_requires_a_stream(historic):
    hist, years = historic
    with pytest.raises(ValueError, match="observed"):
        build_current_erc_values(
            hist, ignition_season_day=80, mode="observed", years=years
        )


def test_observed_stream_must_be_long_enough(historic):
    hist, years = historic
    with pytest.raises(ValueError, match="need 79"):
        build_current_erc_values(
            hist, ignition_season_day=80, mode="observed",
            observed=np.arange(10, dtype=float), years=years,
        )


# ── Output shape and metadata ─────────────────────────────────────────────────

def test_values_are_integers(historic):
    """FSPro reads these as whole ERC values."""
    hist, years = historic
    for mode in ("analog_year", "percentile", "median"):
        values = build_current_erc_values(
            hist, ignition_season_day=80, mode=mode, years=years
        )[PYROME]
        assert np.issubdtype(values.dtype, np.integer)


def test_return_meta_carries_provenance(historic):
    hist, years = historic
    meta = build_current_erc_values(
        hist, ignition_season_day=80, mode="analog_year", years=years, return_meta=True
    )[PYROME]
    assert set(meta) == {
        "values", "mode", "analog_year", "ignition_season_day", "NumWxCurrYear"
    }
    assert meta["mode"] == "analog_year"
    assert meta["ignition_season_day"] == 80
    assert meta["NumWxCurrYear"] == 79


def test_mismatched_years_are_ignored_not_fatal(historic):
    """A stale ``years`` list must not silently mislabel the analog year."""
    hist, _ = historic
    meta = build_current_erc_values(
        hist, ignition_season_day=80, mode="analog_year",
        years=[1999, 2000], return_meta=True,
    )[PYROME]
    assert meta["analog_year"] is None
