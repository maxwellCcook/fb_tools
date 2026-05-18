"""
NWS RTMA-driven NFDRS78 / FireFamilyPlus fuel moisture pipeline.

Companion to :mod:`fb_tools.weather.gridmet` (daily) and
:mod:`fb_tools.weather.fm_scenario` (substrate-agnostic lag).  This module
consumes hourly NWS Real-Time Mesoscale Analysis (RTMA) data exported from
GEE (`NOAA/NWS/RTMA`, hourly 2.5 km, 2011–present) at pyrome-mean resolution
and produces:

  - hourly FM1 / FM10 / FM100 via :func:`build_rtma_dead_fm`, driven by the
    NFDRS78 exponential time-lag recursion at ``dt_hr=1`` with proper
    saturation-stall precipitation handling (Bradshaw 1984)
  - daily GSI-based FM_herb / FM_woody via :func:`build_rtma_live_fm`, with
    daytime-mean VPD and hourly-derived Tmin
  - daily peak-hour FM rows via :func:`collapse_to_peak_hour`, sampled at
    14:00 LST to match RAWS/FireFamilyPlus convention
  - per-pyrome, per-ERC-percentile-band FM medians via
    :func:`build_rtma_scenario_fm`, consumed by
    :func:`fb_tools.weather.gridmet.build_flammap_scenario_cache` when
    ``dead_fm_source="rtma"`` or ``live_fm_source="rtma"``.

The expected CSV schema (per pyrome-year export task; see notebook
``code/notebooks/00b_RTMA-EMC.ipynb``) is::

    pyrome_id, datetime_utc, tmp_f, rh_pct, emc_pct, vpd_pa, pcp_mm_hr

with EMC computed *per pixel* in GEE (before the spatial reduction) so the
nonlinear EMC(T, RH) is not biased by ``EMC(mean(T), mean(RH))``.

ERC remains GridMET-derived — the FSPro/FSim daily-climatology contract is
unchanged.  HRRR remains the wind source.  This module upgrades only the
dead-FM and (optionally) live-FM legs of the FlamMap percentile scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .nfdrs import (
    calc_daylength,
    calc_gsi,
    calc_herb_fm_gsi,
    calc_lagged_fm,
    calc_woody_fm_gsi,
)


_REQUIRED_RTMA_COLS = (
    "pyrome_id",
    "datetime_utc",
    "tmp_f",
    "rh_pct",
    "emc_pct",
    "vpd_pa",
    "pcp_mm_hr",
)

_TIMELAG_1_HR = 1.0
_TIMELAG_10_HR = 10.0
_TIMELAG_100_HR = 100.0


def load_rtma_csv(
    paths: str | Path | Iterable[str | Path],
) -> pd.DataFrame:
    """
    Load one or more RTMA pyrome-hourly CSVs and return a concatenated frame.

    Each input file is expected to follow the per-pyrome-year export schema
    documented at the module level.  Files are concatenated, sorted by
    ``(pyrome_id, datetime_utc)``, and duplicate (pyrome_id, datetime_utc)
    rows are dropped (keeping the first).

    Parameters
    ----------
    paths : str, Path, or iterable of those
        Single CSV path, a directory (all ``.csv`` files inside are loaded),
        or an iterable of CSV paths.

    Returns
    -------
    pd.DataFrame
        Columns ``pyrome_id, datetime_utc, tmp_f, rh_pct, emc_pct, vpd_pa,
        pcp_mm_hr``.  ``datetime_utc`` parsed to timezone-naive UTC
        ``pd.Timestamp``.

    Raises
    ------
    FileNotFoundError
        If a path doesn't exist or a directory contains no CSVs.
    KeyError
        If any required column is missing from any file.
    """
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            files = sorted(p.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No .csv files under {p}")
        elif p.is_file():
            files = [p]
        else:
            raise FileNotFoundError(f"Path not found: {p}")
    else:
        files = [Path(x) for x in paths]
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing CSV file(s): {missing}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing_cols = [c for c in _REQUIRED_RTMA_COLS if c not in df.columns]
        if missing_cols:
            raise KeyError(f"{f.name}: missing columns {missing_cols}")
        frames.append(df[list(_REQUIRED_RTMA_COLS)])

    out = pd.concat(frames, ignore_index=True)
    out["datetime_utc"] = pd.to_datetime(out["datetime_utc"], utc=True).dt.tz_convert(None)
    # RTMA ACPC01 is masked (not zero-filled) when there is no precipitation,
    # so GEE exports null for dry hours. Fill to 0 before the lag recursion sees it.
    out["pcp_mm_hr"] = out["pcp_mm_hr"].fillna(0.0)
    out = (
        out.sort_values(["pyrome_id", "datetime_utc"])
           .drop_duplicates(subset=["pyrome_id", "datetime_utc"], keep="first")
           .reset_index(drop=True)
    )
    return out


def build_rtma_dead_fm(
    df: pd.DataFrame,
    pyrome_col: str = "pyrome_id",
    datetime_col: str = "datetime_utc",
    emc_col: str = "emc_pct",
    precip_col: str | None = "pcp_mm_hr",
    precip_mode: str = "stall",
    precip_threshold_mm_hr: float = 0.25,
    fm1_init: float | None = None,
    fm10_init: float | None = None,
    fm100_init: float | None = None,
) -> pd.DataFrame:
    """
    Append hourly ``FM1``, ``FM10``, ``FM100`` columns to an RTMA frame.

    Runs the NFDRS78 time-lag recursion
    (:func:`~fb_tools.weather.nfdrs.calc_lagged_fm`) at ``dt_hr=1`` against
    the hourly per-pixel-averaged EMC series produced upstream in GEE.

    With hourly forcing the recursion is equivalent to FireFamilyPlus
    BNDRYT integration against RAWS observations: alpha_1 ≈ 0.632/hr,
    alpha_10 ≈ 0.0952/hr, alpha_100 ≈ 0.00995/hr.

    Precip handling defaults to ``"stall"`` — during hours with
    ``pcp_mm_hr > precip_threshold_mm_hr`` the lag EMC input is forced to
    the saturation ceiling appropriate for each fuel class (35 % for 10-hr,
    50 % for 100-hr; FM1 follows ambient EMC at its 1-hr lag).  See
    :func:`~fb_tools.weather.nfdrs.calc_lagged_fm` for the ``precip_mode``
    parameter contract.

    Per-pyrome lag is run independently; cold-start (``fm_*_init=None``) uses
    the first EMC value of each pyrome's series and is forgotten after the
    spin-up window of the slowest fuel (~22 days for FM100 at daily,
    ~525 hours ≈ 22 days at hourly).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_rtma_csv` (or equivalent).  Must include
        ``pyrome_col``, ``datetime_col``, ``emc_col``; ``precip_col`` is
        optional but recommended for ``precip_mode != "none"``.
    pyrome_col, datetime_col, emc_col, precip_col : str
        Column names.  ``precip_col`` may be None to disable precip
        handling entirely (equivalent to ``precip_mode="none"``).
    precip_mode : {"stall", "flat", "none"}
        See :func:`~fb_tools.weather.nfdrs.calc_lagged_fm`.  Default
        ``"stall"`` for the RTMA path.
    precip_threshold_mm_hr : float
        Hourly precip rate above which a step is treated as wet.  Default
        0.25 mm/hr — coarse threshold for fuel-surface wetting; tune per
        regional climatology if needed.
    fm1_init, fm10_init, fm100_init : float, optional
        Cold-start FM (%) at the first hourly record of each pyrome.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` sorted by ``(pyrome_col, datetime_col)`` with
        ``FM1``, ``FM10``, ``FM100`` appended.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    for col in (pyrome_col, datetime_col, emc_col):
        if col not in df.columns:
            raise KeyError(f"Required column {col!r} not in DataFrame")

    out = df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col])
    out = out.sort_values([pyrome_col, datetime_col]).reset_index(drop=True)

    has_precip = (
        precip_col is not None
        and precip_col in out.columns
        and precip_mode != "none"
    )

    n = len(out)
    fm1_arr = np.empty(n)
    fm10_arr = np.empty(n)
    fm100_arr = np.empty(n)

    for _, idx in out.groupby(pyrome_col).groups.items():
        idx_arr = idx.to_numpy()
        emc_g = out.loc[idx_arr, emc_col].values
        precip_g = out.loc[idx_arr, precip_col].values if has_precip else None

        fm1_arr[idx_arr] = calc_lagged_fm(
            emc_g,
            timelag_hr=_TIMELAG_1_HR,
            fm_init=fm1_init,
            precip_mm=precip_g,
            precip_mode=precip_mode,
            precip_threshold_mm=precip_threshold_mm_hr,
            dt_hr=1.0,
        )
        fm10_arr[idx_arr] = calc_lagged_fm(
            emc_g,
            timelag_hr=_TIMELAG_10_HR,
            fm_init=fm10_init,
            precip_mm=precip_g,
            precip_mode=precip_mode,
            precip_threshold_mm=precip_threshold_mm_hr,
            dt_hr=1.0,
        )
        fm100_arr[idx_arr] = calc_lagged_fm(
            emc_g,
            timelag_hr=_TIMELAG_100_HR,
            fm_init=fm100_init,
            precip_mm=precip_g,
            precip_mode=precip_mode,
            precip_threshold_mm=precip_threshold_mm_hr,
            dt_hr=1.0,
        )

    out["FM1"] = fm1_arr
    out["FM10"] = fm10_arr
    out["FM100"] = fm100_arr
    return out


def build_rtma_live_fm(
    df: pd.DataFrame,
    lat_deg: float,
    pyrome_col: str = "pyrome_id",
    datetime_col: str = "datetime_utc",
    tmp_col: str = "tmp_f",
    vpd_col: str = "vpd_pa",
    tz_offset_hours: float = -7.0,
    daytime_local_start: int = 6,
    daytime_local_end: int = 18,
) -> pd.DataFrame:
    """
    Compute daily GSI-based herb/woody fuel moisture from hourly RTMA data.

    Aggregates the hourly frame to daily inputs and applies the NFDRS 2016
    Growing Season Index (Jolly et al. 2005; NWCG PMS 437):

        Tmin       = min over hourly T in the local day
        VPD_day    = mean over daytime hours (``[daytime_local_start,
                     daytime_local_end)``) of hourly VPD — better stress
                     signal than the 24-hr mean which is depressed by
                     nighttime saturation
        photoperiod = ``calc_daylength(lat_deg, doy)``

    GSI ∈ [0, 1] maps to FM_herb ∈ [30 %, 250 %] and FM_woody ∈
    [60 %, 200 %] via :func:`~fb_tools.weather.nfdrs.calc_herb_fm_gsi` /
    :func:`~fb_tools.weather.nfdrs.calc_woody_fm_gsi`.

    Local-day calendar boundaries are derived by shifting ``datetime_utc``
    by ``tz_offset_hours`` (no DST handling — RTMA records are UTC; choose
    -7 for MST year-round, matching RAWS/WIMS reporting convention).

    **Calibration caveat.** Jolly thresholds were calibrated for
    temperate/boreal vegetation.  In semi-arid CO pyromes the VPD bound
    (4100 Pa "limiting") under-discriminates fire-season days — GSI tends
    to sit low across the entire summer.  Higher-resolution RTMA inputs
    cannot fix this; the threshold itself is the bottleneck.  See
    ``CLAUDE.md`` GridMET ERC climatology notes.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly RTMA frame (output of :func:`load_rtma_csv`).
    lat_deg : float
        Site latitude (decimal degrees) for the daylength calculation.
    pyrome_col, datetime_col, tmp_col, vpd_col : str
        Column names.
    tz_offset_hours : float
        UTC offset of the local time zone.  Default -7 (MST).
    daytime_local_start, daytime_local_end : int
        Half-open local-hour window for daytime VPD averaging.  Defaults
        ``[6, 18)``.

    Returns
    -------
    pd.DataFrame
        Per pyrome × local date: ``pyrome_id, date, doy, tmin_f, vpd_day_pa,
        gsi, FM_herb, FM_woody``.
    """
    for col in (pyrome_col, datetime_col, tmp_col, vpd_col):
        if col not in df.columns:
            raise KeyError(f"Required column {col!r} not in DataFrame")

    work = df[[pyrome_col, datetime_col, tmp_col, vpd_col]].copy()
    work[datetime_col] = pd.to_datetime(work[datetime_col])
    local = work[datetime_col] + pd.to_timedelta(tz_offset_hours, unit="h")
    work["date"] = local.dt.normalize()
    work["doy"] = local.dt.dayofyear
    work["hour_local"] = local.dt.hour

    daytime_mask = (
        (work["hour_local"] >= daytime_local_start)
        & (work["hour_local"] < daytime_local_end)
    )

    tmin = (
        work.groupby([pyrome_col, "date"])[tmp_col]
            .min()
            .rename("tmin_f")
            .reset_index()
    )
    vpd_day = (
        work.loc[daytime_mask]
            .groupby([pyrome_col, "date"])[vpd_col]
            .mean()
            .rename("vpd_day_pa")
            .reset_index()
    )
    doy = (
        work.groupby([pyrome_col, "date"])["doy"]
            .first()
            .reset_index()
    )

    daily = tmin.merge(vpd_day, on=[pyrome_col, "date"], how="left").merge(
        doy, on=[pyrome_col, "date"], how="left"
    )

    daylength_hr = calc_daylength(daily["doy"].values, lat_deg)
    daily["gsi"] = calc_gsi(
        tmin_f=daily["tmin_f"].values,
        vpd_pa=daily["vpd_day_pa"].values,
        daylength_hr=daylength_hr,
    )
    daily["FM_herb"] = calc_herb_fm_gsi(daily["gsi"].values)
    daily["FM_woody"] = calc_woody_fm_gsi(daily["gsi"].values)

    return daily.sort_values([pyrome_col, "date"]).reset_index(drop=True)


def collapse_to_peak_hour(
    df: pd.DataFrame,
    peak_hour_local: int = 14,
    tz_offset_hours: float = -7.0,
    pyrome_col: str = "pyrome_id",
    datetime_col: str = "datetime_utc",
    value_cols: Iterable[str] = ("FM1", "FM10", "FM100", "tmp_f", "rh_pct", "pcp_mm_hr"),
) -> pd.DataFrame:
    """
    Reduce an hourly RTMA frame to one daily row at peak fire-hour.

    Samples the row at ``peak_hour_local`` (default 14:00 LST = 13:00–14:00
    local burning period) in the local timezone.  Designed to align RTMA
    daily summaries with the RAWS/FireFamilyPlus 13:00 LST observation
    convention.  No DST handling — pass ``tz_offset_hours=-7`` for MST
    year-round (CONUS RAWS reporting standard).

    Parameters
    ----------
    df : pd.DataFrame
        Hourly RTMA frame with FM columns appended (typically the output
        of :func:`build_rtma_dead_fm`).
    peak_hour_local : int
        Local hour (0–23) to sample.  Default 14.
    tz_offset_hours : float
        UTC offset.  Default -7 (MST).
    pyrome_col, datetime_col : str
        Column names.
    value_cols : iterable of str
        Columns to carry through.  Missing columns are silently skipped.

    Returns
    -------
    pd.DataFrame
        Per pyrome × local date row at peak hour: ``pyrome_id, date, doy,
        <value_cols>``.
    """
    if pyrome_col not in df.columns or datetime_col not in df.columns:
        raise KeyError(f"Required columns {pyrome_col!r}/{datetime_col!r} missing")

    work = df.copy()
    work[datetime_col] = pd.to_datetime(work[datetime_col])
    local = work[datetime_col] + pd.to_timedelta(tz_offset_hours, unit="h")
    work["date"] = local.dt.normalize()
    work["doy"] = local.dt.dayofyear
    work["hour_local"] = local.dt.hour

    peak = work[work["hour_local"] == peak_hour_local].copy()

    keep = [pyrome_col, "date", "doy"] + [c for c in value_cols if c in peak.columns]
    out = peak[keep].sort_values([pyrome_col, "date"]).reset_index(drop=True)

    # Collapse duplicate (pyrome, date) rows that can arise at DST seams
    # — keep the first observation of the peak hour.
    return out.drop_duplicates(subset=[pyrome_col, "date"], keep="first").reset_index(drop=True)


def build_rtma_scenario_fm(
    df_daily: pd.DataFrame,
    erc_band_dates: dict[str, dict[str, Iterable[pd.Timestamp]]],
    pyrome_col: str = "pyrome_id",
    date_col: str = "date",
    fm_cols: Iterable[str] = ("FM1", "FM10", "FM100", "FM_herb", "FM_woody"),
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute per-pyrome × per-percentile median FM from RTMA daily peak-hour rows.

    The ERC percentile-band day-membership is owned by
    :func:`~fb_tools.weather.gridmet.build_flammap_scenario_cache` (driven
    by GridMET ERC).  This function just looks up the RTMA daily-peak-hour
    rows for each band's date set and returns medians of the requested FM
    columns.

    Missing columns in ``df_daily`` are silently skipped.  Date values not
    present in the RTMA frame are dropped from the median.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily peak-hour RTMA frame.  Dead-FM columns come from
        :func:`collapse_to_peak_hour` applied to :func:`build_rtma_dead_fm`
        output; live-FM columns can be merged in from
        :func:`build_rtma_live_fm`.
    erc_band_dates : dict[str, dict[str, iterable[pd.Timestamp]]]
        Mapping ``{pyrome_id_str: {"p25": [dates...], "p50": [...], ...}}``
        — the set of pyrome-days that fall in each ERC percentile band, as
        determined by the GridMET ERC distribution.
    pyrome_col, date_col : str
        Column names in ``df_daily``.
    fm_cols : iterable of str
        FM columns to take medians of.

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        ``{pyrome_id: {percentile_key: {col: median_value}}}``.  Columns
        absent from ``df_daily`` are omitted.  Missing percentile samples
        yield NaN medians.
    """
    work = df_daily.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.normalize()

    available_cols = [c for c in fm_cols if c in work.columns]
    out: dict[str, dict[str, dict[str, float]]] = {}

    indexed = {
        str(pid): grp.set_index(date_col)[available_cols]
        for pid, grp in work.groupby(pyrome_col)
    }

    for pid_str, bands in erc_band_dates.items():
        pid_frame = indexed.get(pid_str)
        if pid_frame is None:
            out[pid_str] = {key: {c: float("nan") for c in available_cols}
                            for key in bands}
            continue

        pid_out: dict[str, dict[str, float]] = {}
        for key, dates in bands.items():
            wanted = pd.to_datetime(list(dates)).normalize()
            matched = pid_frame.reindex(wanted).dropna(how="all")
            if matched.empty:
                pid_out[key] = {c: float("nan") for c in available_cols}
            else:
                pid_out[key] = {
                    c: float(np.nanmedian(matched[c].values))
                    for c in available_cols
                }
        out[pid_str] = pid_out

    return out
