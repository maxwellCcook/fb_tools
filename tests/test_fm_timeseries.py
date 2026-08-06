"""
Tests for the per-season lag restart in ``build_fm_timeseries`` (P1.7).

A fire-season record runs April 1 – October 31, so consecutive rows step from
October straight to the following April. The NFDRS78 time-lag integration used
to run straight through that five-month gap as if it were one day, carrying the
October end-state into spring.

Measured across the nine cached Colorado pyromes, that made April FM100 **too
moist** by 2.5–7.2 percentage points on season day 1 (late October is cool and
humid, so its EMC exceeds early April's), converging by season day 18–22.
"""

import numpy as np
import pandas as pd
import pytest

from fb_tools.weather.fm_scenario import build_fm_timeseries, _split_on_gaps


def _season_frame(years=(2020, 2021, 2022), tmax=85.0, rmin=15.0, october_rmin=None):
    """Synthetic April 1 – October 31 record, one row per day per year."""
    rows = []
    for y in years:
        dates = pd.date_range(f"{y}-04-01", f"{y}-10-31", freq="D")
        for d in dates:
            # Optionally make October cool and humid, as it is in reality.
            humid = october_rmin is not None and d.month == 10
            rows.append({
                "date": d,
                "year": y,
                "doy": d.dayofyear,
                "tmmx_f": 55.0 if humid else tmax,
                "rmin": october_rmin if humid else rmin,
                "pr": 0.0,
            })
    return pd.DataFrame(rows)


def _season_day(df):
    return (
        pd.to_datetime(df["date"])
        - pd.to_datetime(df["year"].astype(str) + "-04-01")
    ).dt.days + 1


# ── _split_on_gaps ────────────────────────────────────────────────────────────

def test_split_on_gaps_splits_at_the_season_boundary():
    dates = pd.to_datetime(
        ["2020-10-29", "2020-10-30", "2020-10-31", "2021-04-01", "2021-04-02"]
    ).values
    segs = list(_split_on_gaps(dates, np.arange(5), 45.0))
    assert len(segs) == 2
    assert segs[0].tolist() == [0, 1, 2]
    assert segs[1].tolist() == [3, 4]


def test_split_on_gaps_keeps_contiguous_runs_whole():
    dates = pd.date_range("2020-04-01", periods=10, freq="D").values
    segs = list(_split_on_gaps(dates, np.arange(10), 45.0))
    assert len(segs) == 1


def test_split_on_gaps_disabled_by_none():
    dates = pd.to_datetime(["2020-10-31", "2021-04-01"]).values
    segs = list(_split_on_gaps(dates, np.arange(2), None))
    assert len(segs) == 1
    assert segs[0].tolist() == [0, 1]


def test_split_on_gaps_handles_short_input():
    dates = pd.to_datetime(["2020-04-01"]).values
    assert len(list(_split_on_gaps(dates, np.arange(1), 45.0))) == 1


# ── The restart itself ────────────────────────────────────────────────────────

def test_each_season_cold_starts_from_its_own_emc():
    """
    With the restart on, every season's first FM100 is identical, because each
    one cold-starts from the same April 1 EMC. Integrating through the gap
    makes later seasons inherit the previous October instead.
    """
    df = _season_frame(october_rmin=70.0)
    out = build_fm_timeseries(df, precip_col="pr")
    out["dos"] = _season_day(out)
    first_days = out.loc[out["dos"] == 1, "FM100"].values
    assert np.allclose(first_days, first_days[0])


def test_continuous_integration_contaminates_later_seasons():
    """The pre-P1.7 behaviour, retained via ``max_gap_days=None``."""
    df = _season_frame(october_rmin=70.0)
    out = build_fm_timeseries(df, precip_col="pr", max_gap_days=None)
    out["dos"] = _season_day(out)
    first_days = out.loc[out["dos"] == 1, "FM100"].values
    # Season 1 cold-starts; seasons 2+ inherit the prior October.
    assert not np.allclose(first_days, first_days[0])


def test_humid_october_biases_the_next_april_moist():
    """
    Reproduces the sign measured on the real climatology: carrying a cool,
    humid October forward makes the following April look moister than it is.
    """
    df = _season_frame(october_rmin=70.0)
    restart = build_fm_timeseries(df, precip_col="pr")
    through = build_fm_timeseries(df, precip_col="pr", max_gap_days=None)
    for out in (restart, through):
        out["dos"] = _season_day(out)
    # Compare the second season, which is the first one able to inherit.
    later = restart["year"] == 2021
    r = restart.loc[later & (restart["dos"] == 1), "FM100"].iloc[0]
    t = through.loc[later & (through["dos"] == 1), "FM100"].iloc[0]
    assert t > r


def test_contamination_washes_out_within_the_season():
    """
    alpha_100 ~ 0.213/day, so the two paths must converge a few weeks in and
    stay converged — the fix is an early-season correction, not a level shift.
    """
    df = _season_frame(october_rmin=70.0)
    restart = build_fm_timeseries(df, precip_col="pr")
    through = build_fm_timeseries(df, precip_col="pr", max_gap_days=None)
    for out in (restart, through):
        out["dos"] = _season_day(out)
    late = restart["dos"] > 60
    assert np.allclose(
        restart.loc[late, "FM100"].values,
        through.loc[late, "FM100"].values,
        atol=1e-6,
    )


def test_first_season_is_unaffected():
    """There is nothing before it to carry over."""
    df = _season_frame(october_rmin=70.0)
    restart = build_fm_timeseries(df, precip_col="pr")
    through = build_fm_timeseries(df, precip_col="pr", max_gap_days=None)
    first = restart["year"] == 2020
    assert np.allclose(
        restart.loc[first, "FM100"].values, through.loc[first, "FM100"].values
    )


def test_fm10_also_restarts_but_recovers_faster():
    """
    alpha_10 ~ 0.909/day, so FM10 washes out in a day or two — the restart
    still applies, it just matters less.
    """
    df = _season_frame(october_rmin=70.0)
    restart = build_fm_timeseries(df, precip_col="pr")
    through = build_fm_timeseries(df, precip_col="pr", max_gap_days=None)
    for out in (restart, through):
        out["dos"] = _season_day(out)
    sel = (restart["dos"] > 5)
    assert np.allclose(
        restart.loc[sel, "FM10"].values, through.loc[sel, "FM10"].values, atol=1e-6
    )


def test_restart_is_per_group():
    """Grouped records restart per (group, season), not just per group."""
    a = _season_frame(years=(2020, 2021))
    a["pyrome"] = "A"
    b = _season_frame(years=(2020, 2021), tmax=70.0, rmin=40.0)
    b["pyrome"] = "B"
    df = pd.concat([a, b], ignore_index=True)

    out = build_fm_timeseries(df, precip_col="pr", group_col="pyrome")
    out["dos"] = _season_day(out)
    for pid in ("A", "B"):
        firsts = out.loc[(out["pyrome"] == pid) & (out["dos"] == 1), "FM100"].values
        assert np.allclose(firsts, firsts[0])
    # The two groups have different weather, so different levels.
    assert not np.isclose(
        out.loc[(out["pyrome"] == "A") & (out["dos"] == 1), "FM100"].iloc[0],
        out.loc[(out["pyrome"] == "B") & (out["dos"] == 1), "FM100"].iloc[0],
    )


def test_output_columns_and_length_unchanged():
    df = _season_frame(years=(2020,))
    out = build_fm_timeseries(df, precip_col="pr")
    assert len(out) == len(df)
    for col in ("EMC", "FM1", "FM10", "FM100"):
        assert col in out.columns
    assert np.all(np.isfinite(out["FM100"].values))


def test_real_climatology_early_season_shifts_drier(gridmet_df):
    """
    Against the cached GridMET record: the fix lowers early-season FM100 and
    leaves the rest of the season untouched.
    """
    g = gridmet_df[gridmet_df["pyrome"].astype(str) == "47"].copy()
    if g.empty:
        pytest.skip("pyrome 47 not in the climatology")
    restart = build_fm_timeseries(g, tmax_col="tmmx_f", rmin_col="rmin", precip_col="pr")
    through = build_fm_timeseries(
        g, tmax_col="tmmx_f", rmin_col="rmin", precip_col="pr", max_gap_days=None
    )
    for out in (restart, through):
        out["dos"] = _season_day(out)

    day1_r = restart.loc[restart["dos"] == 1, "FM100"].mean()
    day1_t = through.loc[through["dos"] == 1, "FM100"].mean()
    assert day1_t - day1_r > 1.0, "expected a multi-point moist bias on season day 1"

    late = restart["dos"] > 60
    assert np.allclose(
        restart.loc[late, "FM100"].values, through.loc[late, "FM100"].values, atol=1e-6
    )
