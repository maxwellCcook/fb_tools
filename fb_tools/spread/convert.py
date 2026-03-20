"""
LCP → Cell2Fire input conversion (stub — not yet implemented).

Cell2Fire requires inputs as ASCII raster (``.asc``) files rather than
GeoTIFF.  The functions here will convert a multi-band LCP GeoTIFF (with
standard LANDFIRE long_name attributes) to the format expected by Cell2Fire.

Planned API
-----------
lcp_to_cell2fire(lcp_fp, output_dir, band_map=None, cell_size_m=None)
    Convert a multi-band LCP GeoTIFF to Cell2Fire ASCII inputs.
    Returns a ``{layer_name: Path}`` dict for all written ``.asc`` files.

build_cell2fire_weather(conditions_df, out_csv, scenario=None)
    Build a Cell2Fire weather CSV from a conditions DataFrame using the
    standard fb_tools scenario columns (WIND_SPEED, WIND_DIRECTION,
    FM_1hr … FM_woody).
"""


def lcp_to_cell2fire(*args, **kwargs):
    raise NotImplementedError(
        "LCP → Cell2Fire conversion is not yet implemented. "
        "See fb_tools/spread/convert.py for the planned API."
    )


def build_cell2fire_weather(*args, **kwargs):
    raise NotImplementedError(
        "Cell2Fire weather CSV builder is not yet implemented. "
        "See fb_tools/spread/convert.py for the planned API."
    )
