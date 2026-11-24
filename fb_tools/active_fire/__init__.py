# Active-fire detections (AFD): access, QA and derived products.
#
# Modules:
#   firms.py      — FIRMS source catalog, cached per-fire fetch, shared schema, map_events
#   qa.py         — low-confidence, interscan-duplicate and static-source screens
#   timeutil.py   — local time, day bins, overpass grouping
#   events.py     — WFIGS / FIRED perimeters, detection-to-event association
#   footprints.py — scan x track footprint boxes
#   grids.py      — snapped grid, overlap-weighted FRP, first-detection grid, GeoTIFF export
#   treatments.py — per-polygon detection summaries for treatment tables (CFT / TWIG / atoms)
#   plot.py       — plot_grid, plot_detections (panels by satellite), plot_detection_series (per day / overpass), shared colour scales (linear / log / sqrt)
#
# Planned (docs/ACTIVE_FIRE_PLAN.md): pointpattern, more plots, animate; swath deferred.

from .firms import (
    STD_COLS,
    VIIRS_NRT,
    VIIRS_SP,
    default_firms_cache,
    fetch_firms,
    firms_sources,
    map_events,
    standardize_firms,
)
from .qa import (
    apply_qa,
    drop_low_confidence,
    drop_static_sources,
    remove_interscan_duplicates,
)
from .timeutil import assign_overpass, day_of, to_local
from .events import associate_detections, fetch_wfigs_perimeters, fired_events
from .footprints import footprint_boxes
from .grids import afd_grid, first_detection_grid, gridded_frp, to_raster
from .plot import color_norm, plot_detection_series, plot_detections, plot_footprint_workflow, plot_grid
from .treatments import treatment_detections

__all__ = [
    "STD_COLS", "VIIRS_NRT", "VIIRS_SP",
    "default_firms_cache", "fetch_firms", "firms_sources", "map_events", "standardize_firms",
    "apply_qa", "drop_low_confidence", "drop_static_sources", "remove_interscan_duplicates",
    "assign_overpass", "day_of", "to_local",
    "associate_detections", "fetch_wfigs_perimeters", "fired_events",
    "footprint_boxes",
    "afd_grid", "first_detection_grid", "gridded_frp", "to_raster",
    "color_norm", "plot_detection_series", "plot_detections", "plot_footprint_workflow", "plot_grid",
    "treatment_detections",
]
