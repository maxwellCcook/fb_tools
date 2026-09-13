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
    DELTA_NODATA,
)
from .noise import (
    bp_noise_floor,
    area_noise_floor,
    required_num_fires,
    annotate_noise_floor,
    describe_noise_floor,
    P01_REFERENCE,
)
from .perimeters import (
    load_fspro_perimeters,
    summarize_early_growth,
    compare_growth,
)
from .fspro_outputs import (
    read_daily_acres,
    check_domain_adequacy,
)
from .convert import destack_lcp, PYRE_LAYERS
from .episodes import (select_episodes, dates_from_fod, dates_from_fast_fires, build_fast_fires_table,
                       FAST_FIRE_GROWTH_HA, FAST_FIRES_CSV, load_rtma_hourly, episode_hourly_frame,
                       layer_bounds_lonlat, block_factors, episode_wind_cube, write_wind_cube,
                       load_wind_cube, wind_cube_summary,
                       weather_arrays, write_episode_frames, weather_from_scenario, episode_weather_report)
from .pyre_outputs import (read_realization, summarize_realization, summarize_runs, paired_deltas,
                           aggregate_realizations, firebrand_production, REALIZATION_LAYERS, edge_contact)
from .transmission import (structure_blocks, zone_id_raster, treatment_shadow, realization_transmission,
                           transmission_batch, summarize_blocks, shadow_probability)

__all__ = [
    "destack_lcp",
    "PYRE_LAYERS",
    "select_episodes", "load_rtma_hourly", "episode_hourly_frame", "weather_arrays", "weather_from_scenario",
    "layer_bounds_lonlat", "block_factors", "episode_wind_cube", "write_wind_cube", "load_wind_cube",
    "wind_cube_summary",
    "write_episode_frames", "read_realization", "summarize_realization", "summarize_runs",
    "paired_deltas", "aggregate_realizations", "firebrand_production", "REALIZATION_LAYERS",
    "read_daily_acres",
    "check_domain_adequacy",
    "delta_burn_probability",
    "aggregate_ignition_bp",
    "summarize_bp_treatments",
    "downwind_treatment_effect",
    "DELTA_NODATA",
    "bp_noise_floor",
    "area_noise_floor",
    "required_num_fires",
    "annotate_noise_floor",
    "describe_noise_floor",
    "P01_REFERENCE",
    "load_fspro_perimeters",
    "summarize_early_growth",
    "compare_growth",
]
