"""
fb_tools.analysis — DEPRECATED treatment-evaluation shim.

Treatment-level fire behavior change analysis (``summarize_treatments``,
``run_treatment_pipeline``) has moved to ``tealom.analyses.treatments``.
Treatment *evaluation* belongs in TEALOM; ``fb_tools`` provides the
fire-modeling primitives that TEALOM calls.

These names are re-exported here lazily for backward compatibility and emit a
``DeprecationWarning``.  Import them from ``tealom.analyses`` instead.
"""

# Intentionally empty: the moved names resolve via __getattr__ (with a
# DeprecationWarning) for explicit imports, but are kept out of __all__ so
# `from fb_tools.analysis import *` does not trigger the warning.
__all__ = []

_MOVED_TO_TEALOM = {"summarize_treatments", "run_treatment_pipeline"}


def __getattr__(name):
    if name in _MOVED_TO_TEALOM:
        import warnings
        warnings.warn(
            f"fb_tools.analysis.{name} has moved to tealom.analyses.{name}. "
            f"Import it as `from tealom.analyses import {name}`; this alias is "
            f"deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tealom.analyses import treatments
        return getattr(treatments, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
