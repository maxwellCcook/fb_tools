"""
DEPRECATED — moved to :mod:`tealom.analyses.treatments`.

``summarize_treatments`` and ``run_treatment_pipeline`` used to live here.  They
were migrated to TEALOM because treatment *evaluation* is TEALOM's
responsibility, while ``fb_tools`` provides the fire-modeling primitives that
TEALOM calls (``apply_treatment``, ``build_scenarios``, ``run_batch``,
``calculate_sdi``).

This module remains only as a backward-compatibility shim: it lazily re-exports
the two functions from ``tealom`` and emits a ``DeprecationWarning``.  Import
them from ``tealom.analyses`` instead::

    from tealom.analyses import summarize_treatments, run_treatment_pipeline

The lazy import avoids an import-time cycle (``tealom`` imports ``fb_tools``).
"""

_MOVED_TO_TEALOM = {"summarize_treatments", "run_treatment_pipeline"}


def __getattr__(name):
    if name in _MOVED_TO_TEALOM:
        import warnings
        warnings.warn(
            f"fb_tools.analysis.treatments.{name} has moved to "
            f"tealom.analyses.treatments.{name}. Import it as "
            f"`from tealom.analyses import {name}`; this alias is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tealom.analyses import treatments
        return getattr(treatments, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
