"""
Synthetic LCP (landscape) generation for parameterized fire behavior analysis.

Creates homogeneous single-value GeoTIFFs suitable for FlamMap short-term
runs where spatial spread is not of interest — only point fire behavior metrics
(flame length, rate of spread, crown state) for a given combination of fuel
type and weather conditions.

Because every pixel carries identical values, all FlamMap outputs are uniform
across the raster and the central pixel can be extracted as the representative
result.

LANDFIRE band encoding
----------------------
FlamMap reads LCP bands by position and expects LANDFIRE-encoded integers:

    Band  Name    Encoding
    ----  ------  -------------------------------------------------------
    1     ELEV    integer metres (stored as-is)
    2     SLP     integer percent (stored as-is)
    3     ASP     integer degrees 0–360; -1 for flat terrain
    4     FBFM40  surface fuel model code (stored as-is)
    5     CC      canopy cover, integer percent 0–100
    6     CH      canopy height, tenths of metres  (ch_m × 10)
    7     CBH     canopy base height, tenths of metres  (cbh_m × 10)
    8     CBD     canopy bulk density, hundredths of kg m⁻³  (cbd_kg_m3 × 100)
    9     EVT     existing vegetation type code (stored as-is)
"""

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — activates .rio accessor
import xarray as xr


# ---------------------------------------------------------------------------
# Named preset scenarios
# ---------------------------------------------------------------------------

# Band names in FlamMap/LANDFIRE order (used as long_name attribute)
_BAND_ORDER = ["ELEV", "SLP", "ASP", "FBFM40", "CC", "CH", "CBH", "CBD", "EVT"]

#: Ready-made fuel and topo parameters for the S. Fork wetland fire behavior analysis.
#: Derived from modal LANDFIRE values and field topo metrics (email, 2025).
#:
#: FBFM40 codes: GR1=101, GR2=102, GS2=122, SH1=141, TU1=161, TU5=165, TL3=183, TL5=185
#:
#: Lodgepole adjustment (_adj): CFRI standard rule applied to EVT 7050 lodgepole stands.
#:   CBH × 0.70 (reduces crown base height by 30 %) and FBFM40 → 185 (TL5).
SYNTHETIC_FUEL_PRESETS = {
    # --- Ponderosa pine
    "ponderosa_2019": {
        "fuel_params": {"fbfm40": 165, "cc_pct": 45, "cbd_kg_m3": 0.06, "cbh_m": 0.3,  "ch_m": 15},
        "topo_params": {"elev_m": 2468, "slope_pct": 17, "aspect_deg": -1},
    },
    "ponderosa_2024": {
        "fuel_params": {"fbfm40": 122, "cc_pct": 25, "cbd_kg_m3": 0.06, "cbh_m": 0.3,  "ch_m": 15},
        "topo_params": {"elev_m": 2468, "slope_pct": 17, "aspect_deg": -1},
    },
    # --- Lodgepole pine (unadjusted 2019 LANDFIRE)
    "lodgepole_2019": {
        "fuel_params": {"fbfm40": 183, "cc_pct": 55, "cbd_kg_m3": 0.11, "cbh_m": 0.8,  "ch_m": 15},
        "topo_params": {"elev_m": 2568, "slope_pct": 22, "aspect_deg": -1},
    },
    # --- Lodgepole pine (CFRI-adjusted: CBH × 0.70, FBFM40 → TL5)
    "lodgepole_2019_adj": {
        "fuel_params": {"fbfm40": 185, "cc_pct": 55, "cbd_kg_m3": 0.11, "cbh_m": 0.56, "ch_m": 15},
        "topo_params": {"elev_m": 2568, "slope_pct": 22, "aspect_deg": -1},
    },
    "lodgepole_2024": {
        "fuel_params": {"fbfm40": 161, "cc_pct": 15, "cbd_kg_m3": 0.03, "cbh_m": 3.0,  "ch_m": 15},
        "topo_params": {"elev_m": 2568, "slope_pct": 22, "aspect_deg": -1},
    },
    # --- Wetland (LANDFIRE-based)
    "wetland_2019": {
        "fuel_params": {"fbfm40": 161, "cc_pct": 35, "cbd_kg_m3": 0.01, "cbh_m": 10.0, "ch_m": 15},
        "topo_params": {"elev_m": 2391, "slope_pct": 2,  "aspect_deg": -1},
    },
    "wetland_2024": {
        "fuel_params": {"fbfm40": 183, "cc_pct": 45, "cbd_kg_m3": 0.01, "cbh_m": 10.0, "ch_m": 15},
        "topo_params": {"elev_m": 2391, "slope_pct": 2,  "aspect_deg": -1},
    },
    # --- Wetland custom / field-based (main analysis)
    "wetland_custom": {
        "fuel_params": {"fbfm40": 102, "cc_pct": 0,  "cbd_kg_m3": 0.0,  "cbh_m": 0.0,  "ch_m": 0},
        "topo_params": {"elev_m": 2391, "slope_pct": 2,  "aspect_deg": -1},
    },
    # --- Optional wetland add-ons
    "wetland_gr1": {
        "fuel_params": {"fbfm40": 101, "cc_pct": 0,  "cbd_kg_m3": 0.0,  "cbh_m": 0.0,  "ch_m": 0},
        "topo_params": {"elev_m": 2391, "slope_pct": 2,  "aspect_deg": -1},
    },
    "wetland_sh1": {
        "fuel_params": {"fbfm40": 141, "cc_pct": 0,  "cbd_kg_m3": 0.0,  "cbh_m": 0.0,  "ch_m": 0},
        "topo_params": {"elev_m": 2391, "slope_pct": 2,  "aspect_deg": -1},
    },
    "wetland_tu1_opt": {
        "fuel_params": {"fbfm40": 161, "cc_pct": 25, "cbd_kg_m3": 0.01, "cbh_m": 10.0, "ch_m": 15},
        "topo_params": {"elev_m": 2391, "slope_pct": 2,  "aspect_deg": -1},
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_synthetic_lcp(
    fuel_params,
    topo_params,
    out_path,
    n_pixels=9,
    resolution=30,
    crs="EPSG:5070",
    evt_code=9999,
):
    """
    Create a homogeneous synthetic LCP GeoTIFF suitable for FlamMap.

    Constructs an *n_pixels* × *n_pixels* multi-band raster in which every
    pixel carries identical values.  Because FlamMap is deterministic and the
    landscape is uniform, all output pixels will be identical — effectively
    giving a single "point" fire behavior result without requiring real
    LANDFIRE data.

    Parameters
    ----------
    fuel_params : dict
        Fuel attributes in physical units.  Required keys:

        - ``fbfm40``     : int   — surface fuel model code (e.g. 165 for TU5)
        - ``cc_pct``     : int   — canopy cover (0–100 %)
        - ``cbd_kg_m3``  : float — canopy bulk density (kg m⁻³)
        - ``cbh_m``      : float — canopy base height (m)
        - ``ch_m``       : float — canopy height (m)

    topo_params : dict
        Topographic attributes.  Required keys:

        - ``elev_m``     : int or float — elevation (m)
        - ``slope_pct``  : int or float — slope (%)
        - ``aspect_deg`` : int — aspect (degrees, 0–360); ``-1`` for flat

    out_path : str or Path
        Destination GeoTIFF path.  Parent directories are created if absent.
    n_pixels : int
        Side length of the square output grid (default ``5``).
    resolution : int or float
        Pixel size in CRS units — metres for EPSG:5070 (default ``30``).
    crs : str
        Coordinate reference system as an EPSG string (default ``"EPSG:5070"``).
    evt_code : int
        Existing vegetation type code stored in every pixel.  Use ``9999``
        (default) for synthetic / unknown vegetation.

    Returns
    -------
    pathlib.Path
        Absolute path to the written GeoTIFF.

    Raises
    ------
    ValueError
        If required keys are missing from *fuel_params* or *topo_params*, or
        if ``cc_pct`` is outside [0, 100].

    Examples
    --------
    >>> from fb_tools.fuelscape.synthetic import create_synthetic_lcp, SYNTHETIC_FUEL_PRESETS
    >>> preset = SYNTHETIC_FUEL_PRESETS["wetland_custom"]
    >>> lcp = create_synthetic_lcp(**preset, out_path="data/lcps/wetland_custom.tif")

    >>> from fb_tools import create_synthetic_lcp
    >>> lcp = create_synthetic_lcp(
    ...     fuel_params={"fbfm40": 165, "cc_pct": 45,
    ...                  "cbd_kg_m3": 0.06, "cbh_m": 0.3, "ch_m": 15},
    ...     topo_params={"elev_m": 2468, "slope_pct": 17, "aspect_deg": -1},
    ...     out_path="data/lcps/ponderosa_2019.tif",
    ... )
    """
    # --- Validate inputs
    _required_fuel = {"fbfm40", "cc_pct", "cbd_kg_m3", "cbh_m", "ch_m"}
    _required_topo = {"elev_m", "slope_pct", "aspect_deg"}
    missing_fuel = _required_fuel - set(fuel_params)
    missing_topo = _required_topo - set(topo_params)
    if missing_fuel:
        raise ValueError(f"fuel_params missing required keys: {missing_fuel}")
    if missing_topo:
        raise ValueError(f"topo_params missing required keys: {missing_topo}")
    if not (0 <= fuel_params["cc_pct"] <= 100):
        raise ValueError(f"cc_pct must be in [0, 100], got {fuel_params['cc_pct']}")

    # --- Encode physical values to LANDFIRE integer storage format
    # SLP: LANDFIRE stores slope in DEGREES (SLPD product); input is percent.
    from math import degrees as _deg, atan as _atan
    encoded = {
        "ELEV":   int(round(topo_params["elev_m"])),
        "SLP":    int(round(_deg(_atan(topo_params["slope_pct"] / 100)))),
        "ASP":    int(topo_params["aspect_deg"]),
        "FBFM40": int(fuel_params["fbfm40"]),
        "CC":     int(round(fuel_params["cc_pct"])),
        "CH":     int(round(fuel_params["ch_m"]    * 10)),
        "CBH":    int(round(fuel_params["cbh_m"]   * 10)),
        "CBD":    int(round(fuel_params["cbd_kg_m3"] * 100)),
        "EVT":    int(evt_code),
    }

    # --- Construct spatial metadata
    # Origin at (0, 0) in the CRS; FlamMap does not require a real-world location
    # for synthetic runs.
    import rasterio.transform  # deferred: rasterio is conda-managed

    origin_x = 0.0
    origin_y = float(n_pixels * resolution)

    transform = rasterio.transform.from_origin(
        west=origin_x,
        north=origin_y,
        xsize=resolution,
        ysize=resolution,
    )

    x_coords = [origin_x + (j + 0.5) * resolution for j in range(n_pixels)]
    y_coords = [origin_y - (i + 0.5) * resolution for i in range(n_pixels)]

    # --- Build one DataArray per band, then stack
    bands = []
    for name in _BAND_ORDER:
        arr = np.full((n_pixels, n_pixels), fill_value=encoded[name], dtype=np.int16)
        da = xr.DataArray(
            arr,
            dims=["y", "x"],
            coords={"y": y_coords, "x": x_coords},
        )
        bands.append(da)

    band_indices = list(range(1, len(_BAND_ORDER) + 1))
    stack = xr.concat(bands, dim=xr.Variable("band", band_indices))
    stack = stack.rio.write_crs(crs)
    stack = stack.rio.write_transform(transform)
    stack.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    stack.attrs["long_name"] = _BAND_ORDER

    # --- Write to disk
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stack.rio.to_raster(str(out_path), dtype="int16", compress="deflate", nodata=-9999)

    print(f"  [create_synthetic_lcp] Wrote {n_pixels}×{n_pixels} LCP → {out_path.name}")
    return out_path.resolve()
