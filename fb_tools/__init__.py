"""
fb_tools — Python wrapper for CLI fire behavior models.

Top-level imports expose the most commonly used functions so notebooks
can do a single import::

    from fb_tools import lfps_request, run_flammap_scenarios, list_files

Sub-packages
------------
fb_tools.fuelscape
    LANDFIRE data access (LFPS API), LCP/raster utilities, and fuel
    adjustments.

fb_tools.models
    CLI wrappers for FlamMap (and future models: FSPro, FSim, MTT).

fb_tools.utils
    Shared geospatial and file-I/O helpers.

fb_tools.suppression
    Suppression Difficulty Index (SDI) calculation and OSM road/trail
    extraction.

fb_tools.weather
    Weather input generation — GridMET, RAWS (coming soon).
"""

from .fuelscape import (
    lfps_request,
    stack_rasters,
    create_ignition_ascii,
    get_band_by_longname,
    adjust_lcp,
    apply_treatment,
    build_surface_lut,
)
from .models import run_flammap_scenarios, load_scenarios, build_scenarios, run_batch
from .utils import (
    is_valid_geom,
    mask_raster,
    geom_to_raster_crs,
    rasterize,
    list_files,
    plot_bands,
)
from .suppression import calculate_sdi, calculate_delta_sdi, fetch_osm_roads

__all__ = [
    # fuelscape
    "lfps_request",
    "stack_rasters",
    "create_ignition_ascii",
    "get_band_by_longname",
    "adjust_lcp",
    "apply_treatment",
    "build_surface_lut",
    # models
    "run_flammap_scenarios",
    "load_scenarios",
    "build_scenarios",
    "run_batch",
    # utils
    "is_valid_geom",
    "mask_raster",
    "geom_to_raster_crs",
    "rasterize",
    "list_files",
    "plot_bands",
    # suppression
    "calculate_sdi",
    "calculate_delta_sdi",
    "fetch_osm_roads",
]
