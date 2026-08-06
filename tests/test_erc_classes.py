"""
Tests for :func:`fb_tools.weather.gridmet.build_erc_classes` (Phase 1).

Covers the three corrections landed in P1.1, P1.3, and P1.4:

- **P1.1** — dead FM from the RTMA hourly-EMC cache, live FM scaled from it.
  The pre-P1.1 path derived live FM from each bin's median day-of-year, which
  pooled spring green-up with cured autumn and handed the extreme-ERC class the
  greenest fuels (defect #2).
- **P1.3** — configurable, tail-weighted percentile edges.
- **P1.4** — column 8 is the daily burn period in minutes, not a spot distance.

Everything here needs the cached GridMET CSV and RTMA parquet, so the whole
module skips on a clean clone.
"""

import numpy as np
import pytest

from fb_tools.weather.gridmet import (
    _DEFAULT_CLASS_PERCENTILES,
    _DEFAULT_ERC_CLASS_BEHAVIOR,
    _DEFAULT_SPOTTING,
    build_erc_classes,
    load_rtma_daily_fm,
)

# Column indices into an ERC class row (spec p.4).
MIN_ERC, MAX_ERC, FM1, FM10, FM100, FM_HERB, FM_WOODY, BURN_PERIOD, SPOT_P, SPOT_D = range(10)
FM_COLS = (FM1, FM10, FM100, FM_HERB, FM_WOODY)

PYROME = "47"


@pytest.fixture(scope="module")
def p47(gridmet_df):
    """Single-pyrome slice — keeps the per-test rebuilds cheap."""
    sub = gridmet_df[gridmet_df["pyrome"].astype(str) == PYROME]
    if sub.empty:
        pytest.skip(f"pyrome {PYROME} not in the GridMET climatology")
    return sub


def _assert_monotonic(classes, cols=FM_COLS):
    """Rows run highest ERC first, so FM must be non-decreasing down them."""
    for col in cols:
        assert np.all(np.diff(classes[:, col]) >= -1e-9), (
            f"column {col} rises with ERC: {classes[:, col].tolist()}"
        )


# ── P1.1: fuel moisture sources ───────────────────────────────────────────────

def test_default_sources_are_rtma_and_dead_fm(p47, weather_dir):
    """The defaults are the combination that satisfies the validator."""
    classes = build_erc_classes(p47, weather_dir=weather_dir)[PYROME]
    assert classes.shape == (5, 10)
    _assert_monotonic(classes)


def test_all_pyromes_monotonic_with_defaults(gridmet_df, weather_dir):
    """
    The P1.1 acceptance criterion: every cached pyrome, not just the one whose
    inversion was originally recorded.  Seven of nine failed before the fix.
    """
    for pid, classes in build_erc_classes(gridmet_df, weather_dir=weather_dir).items():
        try:
            _assert_monotonic(classes)
        except AssertionError as exc:
            pytest.fail(f"pyrome {pid}: {exc}")


def test_extreme_class_live_fm_near_ager_range(p47, weather_dir):
    """
    Ager et al. (2014, Table 7) ran extreme conditions at live herbaceous 40–60%
    and live woody 60–90%.  The pre-P1.1 table wrote 142/193 for this pyrome.
    """
    classes = build_erc_classes(p47, weather_dir=weather_dir)[PYROME]
    assert 25.0 <= classes[0, FM_HERB] <= 90.0
    assert 55.0 <= classes[0, FM_WOODY] <= 110.0


def test_doy_source_reproduces_the_defect(p47, weather_dir):
    """
    The legacy path is retained for reproducibility, and it still exhibits the
    inversion — which is why it is not the default.  If this ever starts passing
    monotonicity, the fixture data changed and the P1.1 rationale needs review.
    """
    classes = build_erc_classes(
        p47, weather_dir=weather_dir, live_fm_source="doy"
    )[PYROME]
    live = np.concatenate([np.diff(classes[:, FM_HERB]), np.diff(classes[:, FM_WOODY])])
    assert np.any(live < -1e-9), (
        "the DOY path no longer inverts live FM — re-evaluate the P1.1 default"
    )


def test_gridmet_and_rtma_dead_fm_agree_at_the_extreme_class(p47, weather_dir):
    """
    The two dead-FM sources are independent derivations, so agreement at the
    extreme class is a real cross-check.  They diverge in the mild classes,
    where the GridMET lag reaches FM100 ~30%; that is expected and is why
    scaled live FM uses the RTMA series.
    """
    rtma = build_erc_classes(p47, weather_dir=weather_dir,
                             dead_fm_source="rtma")[PYROME]
    grid = build_erc_classes(p47, weather_dir=weather_dir,
                             dead_fm_source="gridmet")[PYROME]
    for col in (FM1, FM10, FM100):
        assert abs(rtma[0, col] - grid[0, col]) < 2.0, (
            f"extreme-class column {col}: rtma={rtma[0, col]}, gridmet={grid[0, col]}"
        )


def test_gridmet_dead_fm_stays_monotonic(p47, weather_dir):
    """The fallback source must satisfy the validator too."""
    classes = build_erc_classes(
        p47, weather_dir=weather_dir, dead_fm_source="gridmet"
    )[PYROME]
    _assert_monotonic(classes)


def test_live_fm_scales_follow_fm100(p47, weather_dir):
    """Raising the scale factors raises live FM wherever the floors are clear."""
    base = build_erc_classes(p47, weather_dir=weather_dir)[PYROME]
    hi = build_erc_classes(
        p47, weather_dir=weather_dir, herb_scale=9.0, woody_scale=12.0
    )[PYROME]
    off_floor = base[:, FM_HERB] > 30.0
    assert np.all(hi[off_floor, FM_HERB] >= base[off_floor, FM_HERB])
    _assert_monotonic(hi)


def test_gsi_source_requires_the_rtma_cache(p47):
    with pytest.raises(ValueError, match="gsi"):
        build_erc_classes(p47, live_fm_source="gsi", rtma_daily=None, weather_dir=None)


def test_unknown_sources_rejected(p47):
    with pytest.raises(ValueError, match="dead_fm_source"):
        build_erc_classes(p47, dead_fm_source="era5")
    with pytest.raises(ValueError, match="live_fm_source"):
        build_erc_classes(p47, live_fm_source="ndvi")


def test_load_rtma_daily_fm_returns_none_when_absent(tmp_path):
    assert load_rtma_daily_fm(None, weather_dir=tmp_path) is None
    assert load_rtma_daily_fm(tmp_path / "nope.parquet") is None


def test_load_rtma_daily_fm_normalizes(weather_dir):
    frame = load_rtma_daily_fm(None, weather_dir=weather_dir)
    if frame is None:
        pytest.skip("RTMA daily cache not present")
    # String-valued regardless of whether pandas backs it with object or StringDtype
    assert all(isinstance(v, str) for v in frame["pyrome_id"].head(20))
    assert (frame["date"] == frame["date"].dt.normalize()).all()


# ── P1.3: tail-weighted, configurable bands ───────────────────────────────────

def test_default_percentiles_weight_the_upper_tail(p47, weather_dir):
    """
    Quintiles gave pyrome 47 a top class spanning ERC 67–94.  The tail-weighted
    default must resolve the extreme more tightly than an equal-width split.
    """
    tail = build_erc_classes(p47, weather_dir=weather_dir)[PYROME]
    quint = build_erc_classes(
        p47, weather_dir=weather_dir,
        class_percentiles=list(np.linspace(0, 100, 6)),
    )[PYROME]
    tail_span = tail[0, MAX_ERC] - tail[0, MIN_ERC]
    quint_span = quint[0, MAX_ERC] - quint[0, MIN_ERC]
    assert tail_span < quint_span
    # A tighter extreme band means drier fuels in it.
    assert tail[0, FM100] <= quint[0, FM100]


def test_bands_are_gapless_and_descending_as_written(p47, weather_dir):
    """
    FSPro reads the ``:.0f``-rounded bounds, so contiguity has to survive
    rounding.  Adjacent rows share an edge exactly.
    """
    classes = build_erc_classes(p47, weather_dir=weather_dir)[PYROME]
    lo = np.rint(classes[:, MIN_ERC]).astype(int)
    hi = np.rint(classes[:, MAX_ERC]).astype(int)
    assert np.all(np.diff(lo) < 0)
    assert np.all(lo[:-1] == hi[1:])


def test_bands_cover_the_observed_range(p47, weather_dir):
    """Outer bounds are widened outward so no observation falls outside."""
    classes = build_erc_classes(p47, weather_dir=weather_dir)[PYROME]
    erc = p47["erc"].dropna()
    assert classes[-1, MIN_ERC] <= erc.min()
    assert classes[0, MAX_ERC] >= erc.max()


def test_custom_percentiles_and_class_count(p47, weather_dir):
    classes = build_erc_classes(
        p47, weather_dir=weather_dir, n_classes=3,
        class_percentiles=[0, 70, 90, 100],
        class_behavior=[[300, 0.1, 0], [200, 0.05, 0], [100, 0.0, 0]],
    )[PYROME]
    assert classes.shape == (3, 10)
    _assert_monotonic(classes)


def test_percentile_length_must_match_class_count(p47, weather_dir):
    with pytest.raises(ValueError, match="class_percentiles"):
        build_erc_classes(p47, weather_dir=weather_dir, class_percentiles=[0, 50, 100])


def test_percentiles_must_ascend(p47, weather_dir):
    with pytest.raises(ValueError, match="ascending"):
        build_erc_classes(
            p47, weather_dir=weather_dir, class_percentiles=[0, 80, 60, 90, 97, 100]
        )


def test_season_filter_changes_the_bands(gridmet_df, weather_dir):
    """
    The percentile edges are only meaningful over the fire season.  This CSV is
    already season-limited, so the two agree — the flag still has to be honored
    without raising, and the bands must stay valid either way.
    """
    off = build_erc_classes(gridmet_df, weather_dir=weather_dir, season_only=False)
    on = build_erc_classes(gridmet_df, weather_dir=weather_dir, season_only=True)
    assert set(off) == set(on)
    _assert_monotonic(on[PYROME])
    _assert_monotonic(off[PYROME])


# ── P1.4: column 8 is the burn period ─────────────────────────────────────────

def test_default_behavior_is_the_burn_period_ladder():
    """6 h → 2 h, matching the vendor 416 file and ``_DayTypes.txt`` burnPeriod."""
    assert [row[0] for row in _DEFAULT_ERC_CLASS_BEHAVIOR] == [360, 300, 240, 180, 120]
    assert _DEFAULT_SPOTTING is _DEFAULT_ERC_CLASS_BEHAVIOR


def test_burn_periods_override_only_column_eight(p47, weather_dir):
    custom = [480, 400, 320, 240, 160]
    classes = build_erc_classes(
        p47, weather_dir=weather_dir, burn_periods_min=custom
    )[PYROME]
    assert classes[:, BURN_PERIOD].tolist() == [float(v) for v in custom]
    # Spot probability and delay come from the default table, untouched.
    default = build_erc_classes(p47, weather_dir=weather_dir)[PYROME]
    assert np.array_equal(classes[:, SPOT_P], default[:, SPOT_P])
    assert np.array_equal(classes[:, SPOT_D], default[:, SPOT_D])


def test_burn_periods_length_validated(p47, weather_dir):
    with pytest.raises(ValueError, match="burn_periods_min"):
        build_erc_classes(p47, weather_dir=weather_dir, burn_periods_min=[360, 120])


def test_spotting_alias_still_accepted(p47, weather_dir):
    """The deprecated ``spotting=`` keyword maps onto ``class_behavior``."""
    rows = [[300, 0.2, 1], [280, 0.15, 1], [260, 0.1, 0], [240, 0.05, 0], [220, 0.0, 0]]
    via_alias = build_erc_classes(p47, weather_dir=weather_dir, spotting=rows)[PYROME]
    via_new = build_erc_classes(p47, weather_dir=weather_dir, class_behavior=rows)[PYROME]
    assert np.array_equal(via_alias, via_new)
    assert via_alias[:, BURN_PERIOD].tolist() == [300.0, 280.0, 260.0, 240.0, 220.0]


def test_behavior_length_validated(p47, weather_dir):
    with pytest.raises(ValueError, match="class_behavior"):
        build_erc_classes(p47, weather_dir=weather_dir, class_behavior=[[360, 0.1, 0]])


def test_default_percentiles_constant_is_tail_weighted():
    assert _DEFAULT_CLASS_PERCENTILES == [0.0, 60.0, 80.0, 90.0, 97.0, 100.0]
