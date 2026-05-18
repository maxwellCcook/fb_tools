"""
fb_tools/weather/nfdrs.py
=========================
NFDRS fuel moisture equations for deriving fire-weather FM inputs from
gridded climate data (GridMET tmmx + rmin).

Implements the three-regime equilibrium moisture content (EMC) equations
from Cohen & Deeming (1985) and the fine- and medium-fuel corrections from
Nelson (1984). These are the same equations used in the National Fire Danger
Rating System (NFDRS) and form the basis for ERC computation.

**Intended use in this pipeline:**
GridMET provides daily maximum temperature (``tmmx``, °K) and daily minimum
relative humidity (``rmin``, %). The afternoon minimum RH co-occurs with
maximum temperature, representing peak fire-danger conditions — the same
synoptic window captured by RAWS station observations used in traditional
FSPro parameterization. Applying the NFDRS equations at these peak
conditions yields 1-hr and 10-hr FM estimates that approximate what a RAWS
station would record during the fire-spread window.

References
----------
Cohen, J.D. and Deeming, J.E. (1985). The National Fire-Danger Rating System:
  Basic Equations. Gen. Tech. Rep. PSW-GTR-82. USDA Forest Service, Pacific
  Southwest Forest and Range Experiment Station. 16 p.

Nelson, R.M. (1984). A method for describing equilibrium moisture content of
  forest fuels. Canadian Journal of Forest Research, 14(4), 597-600.
"""

from __future__ import annotations

import warnings

import numpy as np


def kelvin_to_fahrenheit(temp_k: float | np.ndarray) -> np.ndarray:
    """
    Convert temperature from Kelvin to Fahrenheit.

    Used to convert GridMET ``tmmx`` (daily maximum temperature, °K)
    to °F for NFDRS fuel moisture calculations.

    Parameters
    ----------
    temp_k : float or array-like
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Temperature in Fahrenheit.
    """
    return np.asarray(temp_k, dtype=float) * 9.0 / 5.0 - 459.67


def calc_emc(
    temp_f: float | np.ndarray,
    rh_pct: float | np.ndarray,
) -> np.ndarray:
    """
    NFDRS equilibrium moisture content (EMC).

    Three-regime piecewise function of dry-bulb temperature (°F) and
    relative humidity (%) from Cohen & Deeming (1985), Table 1.

    Regimes:
      - RH < 10%  : adsorption at low humidity
      - 10 ≤ RH < 50% : intermediate range
      - RH ≥ 50%  : desorption / condensation regime

    Parameters
    ----------
    temp_f : float or array-like
        Dry-bulb temperature in Fahrenheit.
    rh_pct : float or array-like
        Relative humidity in percent (0–100).

    Returns
    -------
    np.ndarray
        Equilibrium moisture content (% dry weight).

    Examples
    --------
    >>> calc_emc(95, 10)   # hot dry afternoon → ~1.6%
    >>> calc_emc(70, 50)   # moderate conditions → ~8.7%
    """
    t = np.asarray(temp_f, dtype=float)
    h = np.asarray(rh_pct, dtype=float)

    emc = np.where(
        h < 10.0,
        0.03229 + 0.281073 * h - 0.000578 * t * h,
        np.where(
            h < 50.0,
            2.22749 + 0.160107 * h - 0.014784 * t,
            21.0606 + 0.005565 * h ** 2 - 0.00035 * t * h - 0.483199 * h,
        ),
    )
    return np.clip(emc, 0.0, None)


def calc_1hr_fm(
    temp_f: float | np.ndarray,
    rh_pct: float | np.ndarray,
) -> np.ndarray:
    """
    Estimate 1-hour timelag fuel moisture (% dry weight).

    Applies the Nelson (1984) fine-fuel correction factor (1.03) to the
    NFDRS EMC. Represents dead fine fuels (<0.64 cm diameter) that respond
    rapidly to atmospheric moisture conditions.

    In the GridMET pipeline, call with ``tmmx_f`` and ``rmin`` to obtain the
    peak-fire-hour 1-hr FM analogous to RAWS-station afternoon readings.

    Parameters
    ----------
    temp_f : float or array-like
        Dry-bulb temperature in Fahrenheit.
    rh_pct : float or array-like
        Relative humidity in percent (0–100).

    Returns
    -------
    np.ndarray
        1-hr timelag fuel moisture (%).
    """
    return 1.03 * calc_emc(temp_f, rh_pct)


def calc_10hr_fm_instantaneous(
    temp_f: float | np.ndarray,
    rh_pct: float | np.ndarray,
) -> np.ndarray:
    """
    Fosberg (1977) burning-period 10-hr fuel moisture (% dry weight).

    Applies the 1.28 × EMC scalar from Fosberg (1977) — a single-snapshot
    estimator valid only for *peak burning-period* conditions (afternoon
    max-temp / min-RH). It is **not** the NFDRS time-lag model: it carries
    no antecedent memory, so it cannot reproduce the integrated moisture
    history that produces ``FM_100hr > FM_10hr`` in humid or seasonally
    buffered systems.

    For time-series 10-hr FM use :func:`calc_lagged_fm` with
    ``timelag_hr=10`` instead. This function remains useful for one-off
    peak-hour estimates from a single (temp_f, rh_pct) pair.

    Parameters
    ----------
    temp_f : float or array-like
        Dry-bulb temperature in Fahrenheit (peak burning-period value).
    rh_pct : float or array-like
        Relative humidity in percent (peak burning-period value).

    Returns
    -------
    np.ndarray
        Instantaneous 10-hr fuel moisture (%).
    """
    return 1.28 * calc_emc(temp_f, rh_pct)


def calc_10hr_fm(
    temp_f: float | np.ndarray,
    rh_pct: float | np.ndarray,
) -> np.ndarray:
    """
    Deprecated alias for :func:`calc_10hr_fm_instantaneous`.

    Renamed to make the snapshot semantics explicit. Will be removed in a
    future release. For time-series use, switch to :func:`calc_lagged_fm`.
    """
    warnings.warn(
        "calc_10hr_fm is deprecated; use calc_10hr_fm_instantaneous for "
        "peak-hour snapshots or calc_lagged_fm(timelag_hr=10) for time series.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calc_10hr_fm_instantaneous(temp_f, rh_pct)


def calc_lagged_fm(
    emc_series: np.ndarray,
    timelag_hr: float,
    fm_init: float | None = None,
    precip_mm: np.ndarray | None = None,
    precip_mode: str = "flat",
    precip_fm_boost: float = 5.0,
    precip_threshold_mm: float = 1.0,
    saturation_emc_pct: float | None = None,
    fm_cap: float = 40.0,
    dt_hr: float = 24.0,
) -> np.ndarray:
    """
    Discrete exponential time-lag fuel moisture (NFDRS78).

    Implements the standard NFDRS78 dead-fuel-moisture response equation
    (Bradshaw et al. 1984, GTR INT-169 §3.4; Cohen & Deeming 1985, PSW-GTR-82),
    unchanged in NFDRS2016:

        FM_t = FM_{t-1} + alpha * (EMC_t − FM_{t-1})       (+ precip handling)
        alpha = 1 − exp(−dt_hr / timelag_hr)

    With the published stick time constants (Anderson 1990, INT-RP-429;
    Nelson 1969) — ``timelag_hr=10`` for 1/2-inch fuels and ``timelag_hr=100``
    for 1-inch fuels — and a daily timestep this gives:

        alpha_10  = 1 − exp(−24/10)  ≈ 0.909   (fast: ~91 %/day toward EMC)
        alpha_100 = 1 − exp(−24/100) ≈ 0.213   (slow: ~21 %/day toward EMC)

    so 100-hr FM carries weeks of antecedent moisture memory that 10-hr FM
    does not. With ``dt_hr=1`` against an hourly substrate (e.g. RTMA),
    alpha_10 ≈ 0.0952 and alpha_100 ≈ 0.00995, and the recursion tracks
    the diurnal EMC sine wave directly — equivalent to FireFamilyPlus
    BNDRYT integration against RAWS hourly observations.

    Precip handling modes (``precip_mode``):

    - ``"flat"`` (default, daily-GridMET compatible): adds ``precip_fm_boost``
      once on any step with ``precip_mm > precip_threshold_mm``. Empirical
      placeholder, acceptable as a coarse driver of post-storm rebound.
    - ``"stall"`` (NFDRS78 saturation-stall; Bradshaw 1984): during wet
      steps, replaces ``emc[i]`` with ``saturation_emc_pct`` so the lag
      recursion drives FM toward the saturated boundary-layer value. When
      ``saturation_emc_pct`` is None, the ceiling is inferred from
      ``timelag_hr``: 35 % for 10-hr fuels, 50 % for 100-hr fuels, no
      stall (ambient EMC) for 1-hr fuels which equilibrate within a
      single hourly step anyway. Recommended path for hourly RTMA
      (``dt_hr=1``).
    - ``"none"``: ignore ``precip_mm`` entirely; pure EMC-driven recursion.

    Other simplifications vs. full NFDRS78:

    - **Single daily EMC input.** Full NFDRS78 BNDRYT is a 24-hr weighted
      average of daytime (max-T/min-RH) and nighttime (min-T/max-RH) EMC.
      Driving the lag with peak-burning-period EMC alone over-dries the
      100-hr because nighttime EMC (typically near saturation) is excluded.
      This simplification is forced by daily-resolution forcing (e.g.
      GridMET) and is lifted by an hourly substrate (e.g. RTMA) where the
      same function can be called with ``dt_hr=1`` against an hourly EMC
      series.
    - ``fm_cap`` (default 40 %) and ``precip_threshold_mm`` (default 1 mm)
      are pragmatic safeguards, not NFDRS literature values. For hourly
      ``"stall"`` use, a lower threshold (e.g. 0.25 mm/hr) better matches
      sub-daily wetting events.

    Parameters
    ----------
    emc_series : np.ndarray
        Daily (or sub-daily) EMC values (%) computed by :func:`calc_emc`.
        Length = ``n_steps``. Order must be chronological.
    timelag_hr : float
        Fuel-moisture time-lag constant (hours). Use ``10`` for 10-hr fuels,
        ``100`` for 100-hr fuels.
    fm_init : float, optional
        Initial FM (%) at ``t = 0``. Defaults to ``emc_series[0]`` if None.
        Largely irrelevant after sufficient spin-up.
    precip_mm : np.ndarray, optional
        Precipitation series (mm), same length as ``emc_series``. When None,
        no precip rebound is applied regardless of ``precip_mode``.
    precip_mode : {"flat", "stall", "none"}
        Precipitation response model. See above. Default ``"flat"``
        preserves legacy daily-GridMET behavior.
    precip_fm_boost : float
        FM added on wet steps when ``precip_mode="flat"``.
    precip_threshold_mm : float
        Precip amount above which a step is treated as wet.
    saturation_emc_pct : float, optional
        Saturation EMC ceiling (%) used in ``precip_mode="stall"``. When
        None, inferred from ``timelag_hr`` (35 % for 10-hr, 50 % for
        100-hr, no stall for ≤ 1-hr).
    fm_cap : float
        Upper clip on output FM (%).
    dt_hr : float
        Time step in hours. Default 24 (daily). Use 1 with an hourly EMC
        series for the RTMA path.

    Returns
    -------
    np.ndarray
        Time-lagged FM series (%) with the same length as ``emc_series``.

    Raises
    ------
    ValueError
        If ``precip_mm`` is provided with a different length than
        ``emc_series``; if ``timelag_hr <= 0`` or ``dt_hr <= 0``; or if
        ``precip_mode`` is not one of ``{"flat", "stall", "none"}``.

    References
    ----------
    Bradshaw, L.S., Deeming, J.E., Burgan, R.E., Cohen, J.D. (1984).
      The 1978 National Fire-Danger Rating System: Technical Documentation.
      USDA FS GTR INT-169.
    Cohen, J.D., Deeming, J.E. (1985). The National Fire-Danger Rating
      System: Basic Equations. USDA FS PSW-GTR-82.
    Anderson, H.E. (1990). Moisture Diffusivity and Response Time in Fine
      Forest Fuels. USDA FS Res. Pap. INT-RP-429.
    Nelson, R.M. (1969). Some factors affecting the moisture timelags of
      woody materials. USDA FS Res. Pap. SE-44.
    """
    if timelag_hr <= 0:
        raise ValueError(f"timelag_hr must be > 0, got {timelag_hr}")
    if dt_hr <= 0:
        raise ValueError(f"dt_hr must be > 0, got {dt_hr}")
    if precip_mode not in ("flat", "stall", "none"):
        raise ValueError(
            f"precip_mode must be 'flat', 'stall', or 'none', got {precip_mode!r}"
        )

    emc = np.asarray(emc_series, dtype=float)
    n = emc.size

    alpha = 1.0 - np.exp(-dt_hr / timelag_hr)

    if fm_init is None:
        fm_init = float(emc[0]) if n > 0 else 0.0

    if precip_mm is not None and precip_mode != "none":
        precip = np.asarray(precip_mm, dtype=float)
        if precip.size != n:
            raise ValueError(
                f"precip_mm length ({precip.size}) must match emc_series ({n})"
            )
        wet = precip > precip_threshold_mm
    else:
        wet = np.zeros(n, dtype=bool)

    if precip_mode == "stall" and saturation_emc_pct is None:
        if timelag_hr <= 1.5:
            saturation_emc_pct = None
        elif timelag_hr <= 50.0:
            saturation_emc_pct = 35.0
        else:
            saturation_emc_pct = 50.0

    fm = np.empty(n)
    fm_prev = float(fm_init)
    for i in range(n):
        if wet[i] and precip_mode == "stall" and saturation_emc_pct is not None:
            emc_eff = max(emc[i], saturation_emc_pct)
            fm_t = fm_prev + alpha * (emc_eff - fm_prev)
        else:
            fm_t = fm_prev + alpha * (emc[i] - fm_prev)
            if wet[i] and precip_mode == "flat":
                fm_t = min(fm_t + precip_fm_boost, fm_cap)
        fm[i] = fm_t
        fm_prev = fm_t

    return fm


# ── Live fuel moisture — GSI-based (NFDRS 2016) ────────────────────────────────
#
# The 2016 NFDRS estimates live FM via the Growing Season Index (GSI), which
# combines three weather indicators: minimum temperature, vapor pressure deficit
# (VPD), and photoperiod (daylength).  Each indicator is linearly scaled 0→1
# between its "completely limiting" and "unconstrained" thresholds (NWCG PMS 437):
#
#   Tmin:       limiting ≤ −2°C (28°F),  unconstrained ≥ 5°C (41°F)
#   VPD:        limiting ≥ 4100 Pa,      unconstrained ≤ 900 Pa   (inverted)
#   Photoperiod: limiting ≤ 10 h,        unconstrained ≥ 11 h
#
# GSI = f(Tmin) × f(VPD) × f(photoperiod)   range [0, 1]
#
# FM derivation (linear in GSI):
#   HFM: 30% at GSI ≤ 0.5  →  250% at GSI = 1.0
#   WFM: dormant_min at GSI ≤ 0.5  →  200% at GSI = 1.0
#
# References
# ----------
# Jolly, W.M. et al. (2005). A generalized, bioclimatic index to predict
#   foliar phenology in response to climate. Global Change Biology 11, 619–632.
# NWCG PMS 437 – Live Fuel Moisture Content:
#   https://www.nwcg.gov/publications/pms437/fuel-moisture/live-fuel-moisture-content


def calc_vpd_pa(
    temp_f: float | np.ndarray,
    rh_pct: float | np.ndarray,
) -> np.ndarray:
    """
    Vapor pressure deficit (Pa) from temperature and relative humidity.

    Uses the Tetens (1930) saturation vapor pressure formula converted to Pa.
    Suitable for deriving VPD from GridMET ``tmmx_f`` (daily max temp) and
    ``rmin`` (daily min RH) to approximate the peak-fire-hour water stress
    indicator used in the Growing Season Index.

    Parameters
    ----------
    temp_f : float or array-like
        Temperature in Fahrenheit.
    rh_pct : float or array-like
        Relative humidity in percent (0–100).

    Returns
    -------
    np.ndarray
        Vapor pressure deficit in Pascals (≥ 0).
    """
    t = np.asarray(temp_f, dtype=float)
    rh = np.asarray(rh_pct, dtype=float)

    t_c = (t - 32.0) * 5.0 / 9.0  # °F → °C
    # Tetens saturation vapor pressure (Pa)
    es = 610.78 * np.exp(17.2694 * t_c / (t_c + 237.3))
    ea = es * (rh / 100.0)
    return np.clip(es - ea, 0.0, None)


def calc_daylength(
    doy: int | np.ndarray,
    lat_deg: float,
) -> np.ndarray:
    """
    Approximate astronomical daylength (hours) from day-of-year and latitude.

    Uses the Spencer (1971) solar declination formula and the standard
    sunrise/sunset hour angle equation.  Accurate to within ±5 minutes for
    latitudes 20°–60°N.

    Parameters
    ----------
    doy : int or array-like
        Day of year (1–366).
    lat_deg : float
        Latitude in decimal degrees (positive = North).

    Returns
    -------
    np.ndarray
        Daylength in decimal hours.
    """
    doy = np.asarray(doy, dtype=float)
    lat_rad = np.radians(lat_deg)

    # Spencer (1971) solar declination
    b = 2.0 * np.pi * (doy - 1) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(b)
        + 0.070257 * np.sin(b)
        - 0.006758 * np.cos(2 * b)
        + 0.000907 * np.sin(2 * b)
        - 0.002697 * np.cos(3 * b)
        + 0.00148  * np.sin(3 * b)
    )
    # Hour angle at sunrise/sunset (clamp for polar day/night)
    cos_ha = -np.tan(lat_rad) * np.tan(decl)
    cos_ha = np.clip(cos_ha, -1.0, 1.0)
    ha = np.arccos(cos_ha)
    return 2.0 * np.degrees(ha) / 15.0  # degrees → hours


def calc_gsi(
    tmin_f: float | np.ndarray,
    vpd_pa: float | np.ndarray,
    daylength_hr: float | np.ndarray,
) -> np.ndarray:
    """
    Growing Season Index (GSI) per Jolly et al. (2005) / NFDRS 2016.

    Combines three linearly-scaled weather indicators — minimum temperature,
    vapor pressure deficit, and photoperiod — into a single index (0–1)
    that tracks vegetation green-up and senescence.

    NWCG threshold values (PMS 437):

    =================== ================ =====================
    Indicator           Limiting (→ 0)   Unconstrained (→ 1)
    =================== ================ =====================
    Tmin                ≤ −2°C (28.4°F)  ≥ 5°C (41°F)
    VPD                 ≥ 4100 Pa        ≤ 900 Pa  (inverted)
    Photoperiod         ≤ 10 h           ≥ 11 h
    =================== ================ =====================

    Parameters
    ----------
    tmin_f : float or array-like
        Daily minimum temperature in Fahrenheit (GridMET ``tmmn_f``).
    vpd_pa : float or array-like
        Vapor pressure deficit in Pascals (from :func:`calc_vpd_pa`).
    daylength_hr : float or array-like
        Day length in hours (from :func:`calc_daylength`).

    Returns
    -------
    np.ndarray
        GSI values in [0, 1].
    """
    tmin = np.asarray(tmin_f, dtype=float)
    vpd  = np.asarray(vpd_pa, dtype=float)
    dl   = np.asarray(daylength_hr, dtype=float)

    # Tmin: limiting ≤ 28.4°F (−2°C), unconstrained ≥ 41°F (5°C)
    f_tmin = np.clip((tmin - 28.4) / (41.0 - 28.4), 0.0, 1.0)

    # VPD: unconstrained ≤ 900 Pa, limiting ≥ 4100 Pa (inverted scale)
    f_vpd = np.clip((4100.0 - vpd) / (4100.0 - 900.0), 0.0, 1.0)

    # Photoperiod: limiting ≤ 10 h, unconstrained ≥ 11 h
    f_photo = np.clip((dl - 10.0) / (11.0 - 10.0), 0.0, 1.0)

    return f_tmin * f_vpd * f_photo


def calc_herb_fm(
    doy: int | np.ndarray,
    dormant_fm: float = 30.0,
    green_fm: float = 250.0,
    green_start_doy: int = 91,
    dormant_start_doy: int = 250,
) -> np.ndarray:
    """
    Estimate live herbaceous fuel moisture (% dry weight) from DOY.

    Simplified DOY-based proxy for the NFDRS GSI model.  Linearly
    interpolates from *green_fm* (250%, peak green-up at *green_start_doy*
    = April 1) to *dormant_fm* (30%, fully cured at *dormant_start_doy*
    = early September), then holds at 30% thereafter.  Before
    *green_start_doy* also returns *dormant_fm* (dormant condition).

    NFDRS correct ranges per NWCG PMS 437:
    - **Dormant** (fully cured): 30% — treated as dead fine fuel
    - **Peak green**: 250%
    - **Dynamic transfer threshold**: 120% — below this, S&B 40 fuel models
      transfer herbaceous load from live to dead compartment

    For higher accuracy (especially when GridMET ``tmmn`` and site latitude
    are available), use :func:`calc_gsi` + :func:`calc_herb_fm_gsi` instead.

    Parameters
    ----------
    doy : int or array-like
        Day of year (1–366).
    dormant_fm : float
        Live herbaceous FM during dormancy (%). NFDRS default 30%.
    green_fm : float
        Live herbaceous FM at peak green-up (%). NFDRS default 250%.
    green_start_doy : int
        DOY of peak green-up (April 1 = 91). Before this, returns dormant_fm.
    dormant_start_doy : int
        DOY when fully dormant/cured (DOY 250 ≈ early September).

    Returns
    -------
    np.ndarray
        Live herbaceous fuel moisture (%), clipped [30, 250].

    Examples
    --------
    >>> calc_herb_fm(91)   # peak green-up → 250%
    >>> calc_herb_fm(250)  # fully cured → 30%
    >>> calc_herb_fm(170)  # mid-season → ~140%
    """
    doy = np.asarray(doy, dtype=float)

    span = float(dormant_start_doy - green_start_doy)
    frac = np.clip((doy - green_start_doy) / span, 0.0, 1.0)
    fm_herb = green_fm + frac * (dormant_fm - green_fm)
    return np.clip(fm_herb, 30.0, 250.0)


def calc_herb_fm_gsi(
    gsi: float | np.ndarray,
    dormant_fm: float = 30.0,
    green_fm: float = 250.0,
) -> np.ndarray:
    """
    Live herbaceous fuel moisture (%) from the Growing Season Index (GSI).

    Preferred NFDRS 2016 method.  At GSI ≤ 0.5 the fuel is dormant (30%),
    at GSI = 1.0 it is fully green (250%), with linear interpolation between.
    Use :func:`calc_gsi` to derive GSI from GridMET ``tmmn_f``, VPD, and
    daylength.

    Parameters
    ----------
    gsi : float or array-like
        Growing Season Index values in [0, 1].
    dormant_fm : float
        FM at dormancy (%). NFDRS default 30%.
    green_fm : float
        FM at peak green-up (%). NFDRS default 250%.

    Returns
    -------
    np.ndarray
        Live herbaceous FM (%), clipped [30, 250].
    """
    gsi = np.asarray(gsi, dtype=float)
    frac = np.clip((gsi - 0.5) / 0.5, 0.0, 1.0)
    return np.clip(dormant_fm + frac * (green_fm - dormant_fm), 30.0, 250.0)


def calc_woody_fm(
    doy: int | np.ndarray,
    peak_fm: float = 200.0,
    min_fm: float = 60.0,
    peak_doy: int = 152,
    trough_doy: int = 274,
) -> np.ndarray:
    """
    Estimate live woody fuel moisture (% dry weight) from DOY.

    Simplified DOY-based proxy for the NFDRS GSI model.  Models a seasonal
    sinusoidal cycle: peaks in late spring (*peak_doy* ≈ June 1, DOY 152)
    and reaches its annual minimum in early fall (*trough_doy* ≈ October 1,
    DOY 274).  Woody FM lags herbaceous FM and has a smaller seasonal range.

    NFDRS correct ranges per NWCG PMS 437:
    - **Dormant minimum**: 50–80% by NFDRS climate class (default 60%)
    - **Peak green**: 200%
    - No dynamic fuel load transfer is applied to live woody fuels.

    For higher accuracy use :func:`calc_woody_fm_gsi` with GSI from
    :func:`calc_gsi`.

    Parameters
    ----------
    doy : int or array-like
        Day of year (1–366).
    peak_fm : float
        Maximum live woody FM at peak green-up (%). NFDRS default 200%.
    min_fm : float
        Minimum live woody FM at dormancy (%).  NFDRS climate class range
        50–80%; default 60% (midpoint).
    peak_doy : int
        DOY of peak woody FM (≈ June 1 = DOY 152).
    trough_doy : int
        DOY of minimum woody FM (≈ October 1 = DOY 274).

    Returns
    -------
    np.ndarray
        Live woody fuel moisture (%), clipped [50, 200].

    Examples
    --------
    >>> calc_woody_fm(152)  # peak → 200%
    >>> calc_woody_fm(274)  # trough → 60%
    """
    doy = np.asarray(doy, dtype=float)

    half_period = float(trough_doy - peak_doy)
    amp = (peak_fm - min_fm) / 2.0
    mid = (peak_fm + min_fm) / 2.0

    # Cosine: +1 at peak_doy, −1 at trough_doy
    angle = np.pi * (doy - peak_doy) / half_period
    fm_woody = mid + amp * np.cos(angle)
    return np.clip(fm_woody, 50.0, 200.0)


def calc_woody_fm_gsi(
    gsi: float | np.ndarray,
    dormant_fm: float = 60.0,
    green_fm: float = 200.0,
) -> np.ndarray:
    """
    Live woody fuel moisture (%) from the Growing Season Index (GSI).

    Preferred NFDRS 2016 method.  At GSI ≤ 0.5 the fuel is dormant
    (*dormant_fm*, climate-class-dependent), at GSI = 1.0 it is fully green
    (*green_fm* = 200%).

    NFDRS dormant defaults by climate class: 50% (class 1) → 80% (class 4).

    Parameters
    ----------
    gsi : float or array-like
        Growing Season Index values in [0, 1].
    dormant_fm : float
        FM at dormancy (%). Default 60% (general midpoint).
    green_fm : float
        FM at peak green-up (%). NFDRS default 200%.

    Returns
    -------
    np.ndarray
        Live woody FM (%), clipped [50, 200].
    """
    gsi = np.asarray(gsi, dtype=float)
    frac = np.clip((gsi - 0.5) / 0.5, 0.0, 1.0)
    return np.clip(dormant_fm + frac * (green_fm - dormant_fm), 50.0, 200.0)


def calc_live_fm_from_dead(
    fm100: float | np.ndarray,
    herb_scale: float = 1.5,
    woody_scale: float = 2.0,
    herb_floor: float = 30.0,
    woody_floor: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive live herbaceous and woody fuel moisture from 100-hr dead FM.

    Proportional to dead FM with NFDRS floor clamping:

        fm_herb  = max(herb_floor,  fm100 * herb_scale)
        fm_woody = max(woody_floor, fm100 * woody_scale)

    Use this when dead FM conditions are known (custom FlamMap scenarios,
    ERC-class derivation) and phenology-based estimates are too high.
    Drought-stressed vegetation tracks dead FM more closely than DOY alone.

    Parameters
    ----------
    fm100 : float or array-like
        100-hr dead fuel moisture (%).
    herb_scale : float
        Multiplier applied to fm100 for herbaceous FM. Default 1.5.
    woody_scale : float
        Multiplier applied to fm100 for woody FM. Default 2.0.
    herb_floor : float
        Minimum FM_herb (%). NFDRS dormant minimum = 30%.
    woody_floor : float
        Minimum FM_woody (%). NFDRS climate-class midpoint = 60%.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (fm_herb, fm_woody), clipped to [herb_floor, 250] and [woody_floor, 200].

    Examples
    --------
    >>> calc_live_fm_from_dead(8)    # dry fire conditions → floors (30, 60)
    >>> calc_live_fm_from_dead(30)   # moderate → (45, 60)
    >>> calc_live_fm_from_dead(50)   # moist → (75, 100)
    """
    fm100 = np.asarray(fm100, dtype=float)
    fm_herb  = np.maximum(herb_floor,  fm100 * herb_scale)
    fm_woody = np.maximum(woody_floor, fm100 * woody_scale)
    return np.clip(fm_herb, herb_floor, 250.0), np.clip(fm_woody, woody_floor, 200.0)
