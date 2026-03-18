"""
fb_tools.analysis — Treatment-level fire behavior change analysis.

Summarizes pre/post fire behavior metrics (flame length bins, crown state,
SDI) across treatment polygons using FlamMap outputs.
"""

from .treatments import summarize_treatments, run_treatment_pipeline

__all__ = [
    "summarize_treatments",
    "run_treatment_pipeline",
]
