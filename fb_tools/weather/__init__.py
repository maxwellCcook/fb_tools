# Weather input generation for fire behavior models.
#
# Modules:
#   hrrr.py     — HRRR fire-hour wind extraction; pyrome wind climatology; FlamMap WIND_DATA
#   gridmet.py  — GEE GridMET CSV processing; ERC arrays, classes, stats; FlamMap WEATHER_DATA
#   rtma.py     — NWS RTMA hourly NFDRS78/FireFamilyPlus FM (FlamMap percentile scenarios)
#   nfdrs.py        — NFDRS fuel moisture equations (EMC, 1-hr/10-hr/live FM)
#   fm_scenario.py  — NFDRS78 exponential lag → FM1/FM10/FM100 time series
#   raws.py         — RAWS station data access (planned)

from .hrrr import (
    fetch_hrrr_winds_at_fires,
    build_wind_cells,
    build_pyrome_wind_cells,
    load_pyrome_wind_cells,
    build_hrrr_wind_percentiles,
    wind_percentiles_from_cell_cache,
    build_flammap_wind_data,
)
from .gridmet import (
    load_gridmet_csv,
    build_historic_erc_arrays,
    build_erc_stats,
    build_erc_classes,
    build_current_erc_values,
    load_gridmet_pyrome_cache,
    save_erc_classes_to_cache,
    build_flammap_fuel_moistures,
    build_flammap_weather_data,
    build_flammap_scenario_cache,
    load_flammap_scenario_cache,
    build_gridmet_wind_percentiles,
)
from .nfdrs import (
    calc_emc,
    calc_1hr_fm,
    calc_10hr_fm,
    calc_10hr_fm_instantaneous,
    calc_lagged_fm,
    calc_vpd_pa,
    calc_daylength,
    calc_gsi,
    calc_herb_fm,
    calc_herb_fm_gsi,
    calc_woody_fm,
    calc_woody_fm_gsi,
    calc_live_fm_from_dead,
    kelvin_to_fahrenheit,
)
from .fm_scenario import (
    build_fm_timeseries,
    extract_scenario_fm,
)
from .rtma import (
    load_rtma_csv,
    build_rtma_dead_fm,
    build_rtma_live_fm,
    collapse_to_peak_hour,
    build_rtma_scenario_fm,
)

__all__ = [
    # hrrr
    "fetch_hrrr_winds_at_fires",
    "build_wind_cells",
    "build_pyrome_wind_cells",
    "load_pyrome_wind_cells",
    "build_hrrr_wind_percentiles",
    "wind_percentiles_from_cell_cache",
    "build_flammap_wind_data",
    # gridmet
    "load_gridmet_csv",
    "build_historic_erc_arrays",
    "build_erc_stats",
    "build_erc_classes",
    "build_current_erc_values",
    "load_gridmet_pyrome_cache",
    "save_erc_classes_to_cache",
    "build_flammap_fuel_moistures",
    "build_flammap_weather_data",
    "build_flammap_scenario_cache",
    "load_flammap_scenario_cache",
    "build_gridmet_wind_percentiles",
    # nfdrs
    "calc_emc",
    "calc_1hr_fm",
    "calc_10hr_fm",
    "calc_10hr_fm_instantaneous",
    "calc_lagged_fm",
    "calc_vpd_pa",
    "calc_daylength",
    "calc_gsi",
    "calc_herb_fm",
    "calc_herb_fm_gsi",
    "calc_woody_fm",
    "calc_woody_fm_gsi",
    "calc_live_fm_from_dead",
    "kelvin_to_fahrenheit",
    # fm_scenario
    "build_fm_timeseries",
    "extract_scenario_fm",
    # rtma
    "load_rtma_csv",
    "build_rtma_dead_fm",
    "build_rtma_live_fm",
    "collapse_to_peak_hour",
    "build_rtma_scenario_fm",
]
