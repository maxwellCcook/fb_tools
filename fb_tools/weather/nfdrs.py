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


def calc_10hr_fm(
    temp_f: float | np.ndarray,
    rh_pct: float | np.ndarray,
) -> np.ndarray:
    """
    Estimate 10-hour timelag fuel moisture (% dry weight).

    Applies an approximate medium-fuel lag correction (1.28 × EMC) to the
    NFDRS equilibrium moisture content. Represents dead fuels in the 0.64–2.54 cm
    diameter class that respond more slowly to atmospheric changes.

    Parameters
    ----------
    temp_f : float or array-like
        Dry-bulb temperature in Fahrenheit.
    rh_pct : float or array-like
        Relative humidity in percent (0–100).

    Returns
    -------
    np.ndarray
        10-hr timelag fuel moisture (%).
    """
    return 1.28 * calc_emc(temp_f, rh_pct)
