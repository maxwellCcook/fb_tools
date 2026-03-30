# Weather input generation for fire behavior models.
#
# Modules:
#   hrrr.py     — HRRR fire-hour wind extraction; pyrome wind climatology
#   gridmet.py  — GEE GridMET CSV processing; ERC arrays, classes, stats
#   nfdrs.py    — NFDRS fuel moisture equations (EMC, 1-hr FM, 10-hr FM)
#   raws.py     — RAWS station data access (planned)

from .hrrr import (
    fetch_hrrr_winds_at_fires,
    build_wind_cells,
    build_pyrome_wind_cells,
    load_pyrome_wind_cells,
)
from .gridmet import (
    load_gridmet_csv,
    build_historic_erc_arrays,
    build_erc_stats,
    build_erc_classes,
    build_current_erc_values,
    load_gridmet_pyrome_cache,
)
from .nfdrs import (
    calc_emc,
    calc_1hr_fm,
    calc_10hr_fm,
    kelvin_to_fahrenheit,
)

__all__ = [
    # hrrr
    "fetch_hrrr_winds_at_fires",
    "build_wind_cells",
    "build_pyrome_wind_cells",
    "load_pyrome_wind_cells",
    # gridmet
    "load_gridmet_csv",
    "build_historic_erc_arrays",
    "build_erc_stats",
    "build_erc_classes",
    "build_current_erc_values",
    "load_gridmet_pyrome_cache",
    # nfdrs
    "calc_emc",
    "calc_1hr_fm",
    "calc_10hr_fm",
    "kelvin_to_fahrenheit",
]
