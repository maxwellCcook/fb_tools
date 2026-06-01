"""
fb_tools.spread — probabilistic fire spread analysis.

Provides delta burn probability computation, per-ignition ensemble
aggregation, treatment effect summaries, and FSPro per-fire perimeter
(early / extreme growth) analysis across MTT, FSPro, and Cell2Fire outputs.
"""

from .bp import (
    delta_burn_probability,
    aggregate_ignition_bp,
    summarize_bp_treatments,
    downwind_treatment_effect,
)
from .perimeters import (
    load_fspro_perimeters,
    summarize_early_growth,
    compare_growth,
)

__all__ = [
    "delta_burn_probability",
    "aggregate_ignition_bp",
    "summarize_bp_treatments",
    "downwind_treatment_effect",
    "load_fspro_perimeters",
    "summarize_early_growth",
    "compare_growth",
]
