"""
fb_tools/weather/fm_scenario.py
===============================
Build NFDRS78 lag-derived dead fuel-moisture time series and pull FlamMap
scenario rows from them.

Substrate-agnostic: ``build_fm_timeseries`` accepts any daily atmospheric
DataFrame with ``tmmx_f`` (°F) and ``rmin`` (%) and optional ``pr_mm``.
The same function can be called against hourly RTMA-derived inputs (Phase 2)
by passing ``dt_hr=1`` through to :func:`~fb_tools.weather.nfdrs.calc_lagged_fm`.

Pipeline:

1. ``build_fm_timeseries(df)`` → adds ``EMC``, ``FM1``, ``FM10``, ``FM100``
   columns by running :func:`calc_lagged_fm` once over the full record per
   group (single group if no ``group_col`` is given; one group per pyrome /
   station / site otherwise).
2. ``extract_scenario_fm(df, scenario_date, fm10_observed_range)`` → returns
   the rows whose lagged FM10 falls within an observed/target range inside
   a ±15-day climatological window across all years, ready for FlamMap.

References for the lag model itself are in
:func:`~fb_tools.weather.nfdrs.calc_lagged_fm`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .nfdrs import calc_1hr_fm, calc_emc, calc_lagged_fm


# Default fuel-class parameters (NFDRS standard time constants)
_TIMELAG_10_HR = 10.0
_TIMELAG_100_HR = 100.0


def build_fm_timeseries(
    df: pd.DataFrame,
    tmax_col: str = "tmmx_f",
    rmin_col: str = "rmin",
    tmin_col: str | None = "tmmn_f",
    rmax_col: str | None = "rmax",
    date_col: str = "date",
    precip_col: str | None = "pr_mm",
    group_col: str | None = None,
    fm10_init: float | None = None,
    fm100_init: float | None = None,
    precip_boost_10: float = 5.0,
    precip_boost_100: float = 8.0,
    precip_threshold_mm: float = 1.0,
    fm_cap_10: float = 40.0,
    fm_cap_100: float = 40.0,
    dt_hr: float = 24.0,
) -> pd.DataFrame:
    """
    Append ``EMC``, ``FM1``, ``FM10``, ``FM100`` columns to a daily weather table.

    ``FM1`` is the Fosberg peak-burning-period fine-fuel estimator
    (``1.03 × EMC(tmax, rmin)``) — no memory; represents the instantaneous
    fine-fuel moisture at the hottest, driest part of the day. ``FM10`` and
    ``FM100`` are the NFDRS78 exponential time-lag integrations
    (Bradshaw et al. 1984) — see
    :func:`~fb_tools.weather.nfdrs.calc_lagged_fm` for equations, assumptions,
    and citations.

    **EMC input for the lag** — when both ``tmin_col`` and ``rmax_col`` are
    available the lag is driven by a 24-hr BNDRYT-equivalent EMC:

        EMC_24hr = 0.5 * (EMC(tmax, rmin) + EMC(tmin, rmax))

    matching the day/night equal-weight simplification of Bradshaw 1984
    BNDRYT. Without those columns the lag falls back to peak-hour EMC only,
    which systematically over-dries FM100 (nighttime saturation excluded);
    the function prints a warning in that case.

    If ``group_col`` is provided (e.g. ``"pyrome"``), the lag is run
    independently per group; each group must already be ordered or this
    function will sort by ``(group_col, date_col)`` ascending.

    Parameters
    ----------
    df : pd.DataFrame
        Daily atmospheric record. Must include ``tmax_col`` (°F), ``rmin_col``
        (%), and ``date_col`` (datetime-convertible). ``precip_col`` is
        optional — when absent, the precip rebound is silently skipped and a
        warning is printed.
    tmax_col, rmin_col, date_col, precip_col : str
        Column names.
    group_col : str, optional
        Spatial grouping column (e.g. ``"pyrome"``, ``"station_id"``). When
        provided, the lag integration restarts within each group.
    fm10_init, fm100_init : float, optional
        Cold-start FM (%) at the first record of each group. Defaults to
        ``EMC[0]`` of that group. Largely irrelevant after spin-up.
    precip_boost_10, precip_boost_100 : float
        FM added on rain days for 10-hr and 100-hr fuels.
    precip_threshold_mm : float
        Precipitation amount above which the boost is applied.
    fm_cap_10, fm_cap_100 : float
        Upper clip on the lag-integrated FM series.
    dt_hr : float
        Time step in hours. Default 24 (daily). Use 1 to drive the same
        machinery from an hourly RTMA-derived EMC series.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` (sorted) with ``EMC``, ``FM1``, ``FM10``, ``FM100``
        appended. Original columns preserved.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    for col in (tmax_col, rmin_col, date_col):
        if col not in df.columns:
            raise KeyError(f"Required column {col!r} not in DataFrame")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    sort_cols = [date_col] if group_col is None else [group_col, date_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    has_precip = precip_col is not None and precip_col in out.columns
    if precip_col and not has_precip:
        print(
            f"  [build_fm_timeseries] precip column {precip_col!r} missing — "
            "running lag without precip rebound."
        )

    # Peak-hour EMC for FM1 (Fosberg) and as fallback for the lag input
    emc_peak = calc_emc(out[tmax_col].values, out[rmin_col].values)
    out["EMC"] = emc_peak
    out["FM1"] = calc_1hr_fm(out[tmax_col].values, out[rmin_col].values)

    # Drive the lag with the 24-hr BNDRYT-equivalent EMC when day/night
    # inputs are present (Bradshaw 1984 §3.4, equal-weight simplification).
    # Falls back to peak-hour EMC if tmin/rmax are absent.
    has_diurnal = (
        tmin_col is not None and tmin_col in out.columns
        and rmax_col is not None and rmax_col in out.columns
    )
    if has_diurnal:
        emc_night = calc_emc(out[tmin_col].values, out[rmax_col].values)
        emc_lag_input = 0.5 * (emc_peak + emc_night)
        out["EMC_24hr"] = emc_lag_input
    else:
        print(
            "  [build_fm_timeseries] tmin/rmax columns missing — "
            "driving the lag with peak-hour EMC only. FM100 will be biased "
            "low (nighttime saturation excluded). See plan."
        )
        emc_lag_input = emc_peak

    fm10_arr = np.empty(len(out))
    fm100_arr = np.empty(len(out))

    lag_emc_col = "EMC_24hr" if has_diurnal else "EMC"

    if group_col is None:
        groups: Iterable = [(None, out.index.to_numpy())]
    else:
        groups = ((g, idx.to_numpy()) for g, idx in out.groupby(group_col).groups.items())

    for _, idx in groups:
        emc_g = out.loc[idx, lag_emc_col].values
        precip_g = out.loc[idx, precip_col].values if has_precip else None

        fm10_arr[idx] = calc_lagged_fm(
            emc_g,
            timelag_hr=_TIMELAG_10_HR,
            fm_init=fm10_init,
            precip_mm=precip_g,
            precip_fm_boost=precip_boost_10,
            precip_threshold_mm=precip_threshold_mm,
            fm_cap=fm_cap_10,
            dt_hr=dt_hr,
        )
        fm100_arr[idx] = calc_lagged_fm(
            emc_g,
            timelag_hr=_TIMELAG_100_HR,
            fm_init=fm100_init,
            precip_mm=precip_g,
            precip_fm_boost=precip_boost_100,
            precip_threshold_mm=precip_threshold_mm,
            fm_cap=fm_cap_100,
            dt_hr=dt_hr,
        )

    out["FM10"] = fm10_arr
    out["FM100"] = fm100_arr
    return out


def extract_scenario_fm(
    df: pd.DataFrame,
    scenario_date: str | pd.Timestamp,
    fm10_observed_range: tuple[float, float] | None = None,
    date_col: str = "date",
    doy_window: int = 15,
    burn_in_days: int = 60,
    group_col: str | None = None,
    keep_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Pull lag-derived FM rows that match an observed FM10 range within a
    climatological window around ``scenario_date``.

    Use after :func:`build_fm_timeseries`. The default behaviour treats the
    record as a single time series; pass ``group_col`` to restrict the
    burn-in clip to within each group (e.g. per-pyrome spin-up).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`build_fm_timeseries` — must include ``date_col``,
        ``EMC``, ``FM1``, ``FM10``, ``FM100``.
    scenario_date : str or pd.Timestamp
        Target date defining the day-of-year center. Year is ignored; the
        function pulls candidate days from a ±``doy_window`` window in every
        year of the record.
    fm10_observed_range : tuple of float, optional
        ``(min, max)`` % bounds on lag-derived FM10. When None, no FM10
        filter is applied and all DOY-window rows are returned.
    doy_window : int
        Half-width of the day-of-year window (days). Default 15 → 31-day
        seasonal slice.
    burn_in_days : int
        Drop the first ``burn_in_days`` rows of each group (or of the whole
        record when ``group_col`` is None) to discard spin-up. Default 60.
    group_col : str, optional
        If provided, the burn-in clip is applied per group.
    keep_cols : iterable of str, optional
        Subset of columns to return. Defaults to
        ``[date_col, "EMC", "FM1", "FM10", "FM100"]`` plus ``group_col`` when
        provided.

    Returns
    -------
    pd.DataFrame
        Matching rows sorted by ``FM10`` ascending. Empty DataFrame with the
        expected schema if no rows match.
    """
    required = {date_col, "EMC", "FM1", "FM10", "FM100"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns {missing} — run build_fm_timeseries() first."
        )

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    sort_cols = [date_col] if group_col is None else [group_col, date_col]
    work = work.sort_values(sort_cols).reset_index(drop=True)

    if burn_in_days > 0:
        if group_col is None:
            work = work.iloc[burn_in_days:].reset_index(drop=True)
        else:
            work = (
                work.groupby(group_col, group_keys=False)
                    .apply(lambda g: g.iloc[burn_in_days:])
                    .reset_index(drop=True)
            )

    target_doy = pd.Timestamp(scenario_date).dayofyear
    doy = work[date_col].dt.dayofyear

    lo = target_doy - doy_window
    hi = target_doy + doy_window
    if lo < 1:
        doy_mask = (doy >= lo + 366) | (doy <= hi)
    elif hi > 366:
        doy_mask = (doy >= lo) | (doy <= hi - 366)
    else:
        doy_mask = (doy >= lo) & (doy <= hi)

    work = work[doy_mask]

    if fm10_observed_range is not None:
        lo_fm, hi_fm = fm10_observed_range
        work = work[(work["FM10"] >= lo_fm) & (work["FM10"] <= hi_fm)]

    if keep_cols is None:
        cols = [date_col, "EMC", "FM1", "FM10", "FM100"]
        if group_col is not None:
            cols = [group_col] + cols
    else:
        cols = list(keep_cols)

    return work[cols].sort_values("FM10").reset_index(drop=True)
