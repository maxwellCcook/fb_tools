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

fb_tools.analysis
    Treatment-level fire behavior change analysis (flame length bins,
    crown state, SDI delta summaries).

fb_tools.plotting
    Visualization for treatment fire behavior change outputs (FL stacked
    bars, delta bars, crown state panels, SDI boxplots).

fb_tools.weather
    Weather input generation — GridMET, RAWS (coming soon).
"""

from .fuelscape import (
    lfps_request,
    lfps_mosaic,
    stack_rasters,
    create_ignition_ascii,
    get_band_by_longname,
    adjust_lcp,
    apply_treatment,
    build_surface_lut,
)
from .models import run_flammap_scenarios, load_scenarios, build_scenarios, run_batch, stacked_output_path
from .utils import (
    is_valid_geom,
    mask_raster,
    geom_to_raster_crs,
    rasterize,
    clip_raster_inplace,
    list_files,
    plot_bands,
)
from .suppression import calculate_sdi, calculate_delta_sdi, fetch_osm_roads, fetch_counties
from .analysis import summarize_treatments, run_treatment_pipeline
from .plotting import (
    plot_fl_stackedbar,
    plot_fl_delta_bar,
    plot_cs_stackedbar_multipct,
    plot_sdi_boxplot,
)

__all__ = [
    # fuelscape
    "lfps_request",
    "lfps_mosaic",
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
    "stacked_output_path",
    # utils
    "is_valid_geom",
    "mask_raster",
    "geom_to_raster_crs",
    "rasterize",
    "clip_raster_inplace",
    "list_files",
    "plot_bands",
    # suppression
    "calculate_sdi",
    "calculate_delta_sdi",
    "fetch_osm_roads",
    "fetch_counties",
    # analysis
    "summarize_treatments",
    "run_treatment_pipeline",
    # plotting
    "plot_fl_stackedbar",
    "plot_fl_delta_bar",
    "plot_cs_stackedbar_multipct",
    "plot_sdi_boxplot",
]
