"""
Monte Carlo noise floors for FSPro treatment-effect comparisons.

Why this module exists
----------------------
Experiment P0.1 established that ``SPOTTING_SEED`` does **not** deliver common
random numbers.  Two FSPro runs of the *same* landscape at the *same* seed
differ on 66% of burned cells, and same-seed noise is statistically
indistinguishable from different-seed noise — the seed reaches spotting only,
not the ERC stream or the wind draws.

The consequence is that a baseline-versus-treated difference is **never** an
exact pairing.  Every reported Δ carries run-to-run noise even where the
treatment did nothing, so every reported Δ needs a null band.  These helpers
supply it.

Calibration
-----------
Measured on the pyrome 47 reference run — 740 × 783 @ 90 m, ``Duration=7``,
``NumFires=100``, 416-sample ignition:

===========================  ========
per-pixel \\|ΔBP\\| p50        0.01
per-pixel \\|ΔBP\\| p95        0.07
per-pixel \\|ΔBP\\| max        0.18
expected-area-burned CV      1.46%
===========================  ========

Both quantities are Monte Carlo means over ``NumFires`` independent fires, so
they scale as ``1/√N``.  The area-integrated metric is roughly an order of
magnitude tighter than the pixel metric at the same *N*, because pixel errors
average out over the integration — a strong argument for leading with
area-integrated transmission metrics (``Σ BP × cell_area``, the Ager TF_ij
estimator) rather than pixel-level ΔBP maps.

Caveats to carry with any number from here
------------------------------------------
- Calibrated on **one landscape at Duration 7**.  Longer runs grow larger fires
  and may not scale identically; re-measure with
  ``code/dev/FSPro/p0_experiments.py --experiments p01`` at production settings
  if the margin matters.
- The ``1/√N`` extrapolation assumes fires are independent draws, which they are
  under FSPro's design, but it says nothing about *bias* — only about variance.
- These are **null bands, not confidence intervals** on the treatment effect.
  A Δ above the floor is resolvable above run-to-run noise; it is not thereby
  statistically significant in any formal sense.
"""

import numpy as np
import pandas as pd


#: The P0.1 reference measurement.  See module docstring for provenance.
P01_REFERENCE = {
    "num_fires":      100,
    "duration":       7,
    "pixel_dbp_p50":  0.01,
    "pixel_dbp_p95":  0.07,
    "pixel_dbp_max":  0.18,
    "area_cv_pct":    1.46,
    "grid":           "740x783 @ 90 m, pyrome 47, 416-sample ignition",
    "source":         "P0.1, 2026-08-05, data/fspro_test/p0_experiments/p0_results.json",
}

_PIXEL_STATS = {
    "p50": "pixel_dbp_p50",
    "p95": "pixel_dbp_p95",
    "max": "pixel_dbp_max",
}


def _scale_factor(num_fires, reference):
    """``1/√N`` scaling factor from the reference *N* to *num_fires*."""
    num_fires = int(num_fires)
    if num_fires <= 0:
        raise ValueError(f"num_fires must be positive, got {num_fires}.")
    return np.sqrt(reference["num_fires"] / num_fires)


def bp_noise_floor(num_fires, statistic="p95", reference=None):
    """
    Per-pixel |ΔBP| noise floor at *num_fires*.

    Parameters
    ----------
    num_fires : int
        ``NumFires`` for the run being reported.
    statistic : {"p50", "p95", "max"}
        Which measured quantile to scale.  ``"p95"`` (default) is the sensible
        reporting threshold: 95% of unaffected pixels fall below it.
    reference : dict, optional
        Override the P0.1 calibration.  Must carry ``num_fires`` and the
        relevant ``pixel_dbp_*`` key.

    Returns
    -------
    float
        Noise floor in burn-probability units.  A per-pixel ΔBP smaller than
        this is indistinguishable from run-to-run Monte Carlo noise.

    Raises
    ------
    ValueError
        If *num_fires* is not positive or *statistic* is unknown.

    Examples
    --------
    >>> round(bp_noise_floor(1000), 4)
    0.0221
    >>> round(bp_noise_floor(4000), 4)
    0.0111
    """
    reference = reference or P01_REFERENCE
    if statistic not in _PIXEL_STATS:
        raise ValueError(
            f"statistic must be one of {sorted(_PIXEL_STATS)}, got {statistic!r}."
        )
    base = reference[_PIXEL_STATS[statistic]]
    return float(base * _scale_factor(num_fires, reference))


def area_noise_floor(num_fires, area=None, for_difference=False, reference=None):
    """
    Run-to-run variability of an area-integrated metric at *num_fires*.

    Applies to ``Σ BP × cell_area`` — expected area burned, and hence the Ager
    TF_ij transmission estimator.

    Parameters
    ----------
    num_fires : int
        ``NumFires`` for the run being reported.
    area : float, optional
        A **total** area (any unit) — e.g. the baseline arm's ``Σ BP × area``
        for the zone.  When given, the floor is returned in the same unit
        instead of as a percentage.  Note this is the total, *not* the delta:
        the calibration is a coefficient of variation on the total.
    for_difference : bool
        When ``True``, inflate by ``√2`` because a baseline-minus-treated
        difference carries the noise of two independent runs.  Use this for any
        ΔTF; leave ``False`` for a single arm's TF.
    reference : dict, optional
        Override the P0.1 calibration.

    Returns
    -------
    float
        Percent variability when *area* is ``None``, otherwise the absolute
        noise floor in the units of *area*.

    Examples
    --------
    >>> round(area_noise_floor(4000), 3)
    0.231
    >>> round(area_noise_floor(4000, area=10_000), 1)
    23.1
    >>> round(area_noise_floor(4000, area=10_000, for_difference=True), 1)
    32.7
    """
    reference = reference or P01_REFERENCE
    pct = float(reference["area_cv_pct"] * _scale_factor(num_fires, reference))
    if for_difference:
        pct *= np.sqrt(2.0)
    if area is None:
        return pct
    return float(abs(area) * pct / 100.0)


def required_num_fires(target, statistic="p95", metric="pixel", reference=None):
    """
    Smallest ``NumFires`` whose noise floor sits at or below *target*.

    Parameters
    ----------
    target : float
        Effect size to resolve.  Burn-probability units when
        ``metric="pixel"``; percent when ``metric="area"``.
    statistic : {"p50", "p95", "max"}
        Quantile to resolve against (``metric="pixel"`` only).
    metric : {"pixel", "area"}
        ``"pixel"`` for per-pixel ΔBP, ``"area"`` for the area-integrated
        metric.
    reference : dict, optional
        Override the P0.1 calibration.

    Returns
    -------
    int
        Required number of fires, rounded up.

    Raises
    ------
    ValueError
        If *target* is not positive or *metric* is unknown.

    Examples
    --------
    Resolving a 0.01 ΔBP at the 95th percentile:

    >>> required_num_fires(0.01)
    4900

    The same effect is far cheaper area-integrated:

    >>> required_num_fires(1.0, metric="area")
    214
    """
    reference = reference or P01_REFERENCE
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}.")

    if metric == "pixel":
        if statistic not in _PIXEL_STATS:
            raise ValueError(
                f"statistic must be one of {sorted(_PIXEL_STATS)}, got {statistic!r}."
            )
        base = reference[_PIXEL_STATS[statistic]]
    elif metric == "area":
        base = reference["area_cv_pct"]
    else:
        raise ValueError(f"metric must be 'pixel' or 'area', got {metric!r}.")

    exact = reference["num_fires"] * (base / target) ** 2
    # Round before ceil: 0.07/0.01 is 7.000000000000001 in binary floating
    # point, which would otherwise push an exact 4900 up to 4901.
    return int(np.ceil(round(exact, 6)))


def annotate_noise_floor(
    df,
    num_fires,
    delta_col="dBP_mean",
    statistic="p95",
    metric="pixel",
    total_col=None,
    reference=None,
    prefix=None,
):
    """
    Add a noise floor and a resolvability flag to a results table.

    Convenience for reporting: attaches the floor alongside every Δ so a table
    can never be read without it.

    Parameters
    ----------
    df : pandas.DataFrame
        Results table, e.g. from
        :func:`~fb_tools.spread.bp.summarize_bp_treatments`.
    num_fires : int
        ``NumFires`` used for the runs behind *df*.
    delta_col : str
        Column holding the effect to test.  Default ``"dBP_mean"``.
    metric : {"pixel", "area"}
        ``"pixel"`` applies a single per-pixel ΔBP floor to every row.
        ``"area"`` computes a per-row floor from *total_col*.
    total_col : str, optional
        **Required when** ``metric="area"``.  Column holding the *total*
        area-integrated quantity for the zone — typically the baseline arm's
        ``Σ BP × cell_area``.  The calibration is a coefficient of variation on
        that total, so the floor cannot be derived from the delta alone.  The
        ``√2`` two-run inflation is applied automatically.
    statistic, reference
        Passed through to :func:`bp_noise_floor` / :func:`area_noise_floor`.
    prefix : str, optional
        Prefix for the added columns.  Defaults to *delta_col*.

    Returns
    -------
    pandas.DataFrame
        A **copy** with two added columns: ``{prefix}_noise_floor`` and
        ``{prefix}_resolvable`` (``|Δ| > floor``).

    Raises
    ------
    KeyError
        If *delta_col* or *total_col* is not in *df*.
    ValueError
        If *metric* is unknown, or ``metric="area"`` without *total_col*.
    """
    if delta_col not in df.columns:
        raise KeyError(
            f"delta_col '{delta_col}' not in df. Available: {list(df.columns)}"
        )

    out = df.copy()
    prefix = prefix or delta_col

    if metric == "pixel":
        floors = bp_noise_floor(num_fires, statistic=statistic, reference=reference)
    elif metric == "area":
        if total_col is None:
            raise ValueError(
                "metric='area' needs total_col — the zone's total "
                "Sum(BP x cell_area), since the P0.1 calibration is a "
                "coefficient of variation on the total, not on the delta."
            )
        if total_col not in df.columns:
            raise KeyError(
                f"total_col '{total_col}' not in df. Available: {list(df.columns)}"
            )
        pct = area_noise_floor(num_fires, for_difference=True, reference=reference)
        floors = out[total_col].abs() * pct / 100.0
    else:
        raise ValueError(f"metric must be 'pixel' or 'area', got {metric!r}.")

    out[f"{prefix}_noise_floor"] = floors
    out[f"{prefix}_resolvable"] = out[delta_col].abs() > floors
    return out


def describe_noise_floor(num_fires, reference=None):
    """
    Human-readable summary of the noise floors at *num_fires*.

    Returns
    -------
    str
        Multi-line text suitable for printing into a notebook or a methods
        section.
    """
    reference = reference or P01_REFERENCE
    return "\n".join([
        f"Monte Carlo noise floor at NumFires={num_fires}",
        f"  per-pixel |dBP|  p50 {bp_noise_floor(num_fires, 'p50', reference):.4f}"
        f"   p95 {bp_noise_floor(num_fires, 'p95', reference):.4f}"
        f"   max {bp_noise_floor(num_fires, 'max', reference):.4f}",
        f"  area-integrated  {area_noise_floor(num_fires, reference=reference):.3f}%",
        f"  calibration      P0.1, N={reference['num_fires']}, "
        f"Duration={reference['duration']} ({reference['grid']})",
        f"  caveat           1/sqrt(N) extrapolation from one landscape; "
        f"null band, not a confidence interval.",
    ])
