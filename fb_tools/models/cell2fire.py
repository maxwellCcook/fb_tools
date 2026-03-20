"""
Cell2Fire CLI wrapper (stub — not yet implemented).

Cell2Fire is an open-source, cross-platform cellular automata fire spread
simulator.  Unlike FlamMap MTT and FSPro, it runs natively on macOS and
Linux (no Windows VM required).

GitHub: https://github.com/cell2fire/Cell2Fire
Install: ``pip install cell2fire`` (requires C++ build tools)

Input format
------------
Cell2Fire uses ASCII raster (``.asc``) files for fuel, topography, and
canopy inputs, plus a weather CSV.  Use
:func:`~fb_tools.spread.convert.lcp_to_cell2fire` to convert a multi-band
LCP GeoTIFF to the required ASCII format.

Planned API
-----------
run_cell2fire(lcp_fp, output_directory, c2f_params, ...)
    Run a single Cell2Fire scenario.

run_cell2fire_batch(scenarios_df, output_root, ...)
    Batch runner mirroring :func:`~fb_tools.models.scenarios.run_batch`.
"""


def run_cell2fire(*args, **kwargs):
    raise NotImplementedError(
        "Cell2Fire integration is not yet implemented. "
        "See fb_tools/models/cell2fire.py for the planned API."
    )


def run_cell2fire_batch(*args, **kwargs):
    raise NotImplementedError(
        "Cell2Fire batch integration is not yet implemented. "
        "See fb_tools/models/cell2fire.py for the planned API."
    )
