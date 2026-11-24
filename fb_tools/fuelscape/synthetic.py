"""
Synthetic LCP (landscape) generation for FlamMap point fire behavior.

Two tools, both for runs where spatial spread is not of interest:

- :func:`create_synthetic_lcp` writes a homogeneous grid for one fuel /
  topography combination given in physical units.
- :func:`create_combination_lcp` writes each unique combination in a baseline
  landscape once, plus one block per treatment, so a single FlamMap run replaces
  a full-grid run on the baseline and on every treated version of it;
  :func:`expand_combinations` maps the result back and
  :func:`combination_spread` checks that the run settings allow it.

LANDFIRE band encoding
----------------------
FlamMap reads LCP bands by position and expects LANDFIRE-encoded integers:

    Band  Name    Encoding
    ----  ------  -------------------------------------------------------
    1     ELEV    integer metres (stored as-is)
    2     SLP     integer degrees (``create_synthetic_lcp`` converts from percent)
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
import pandas as pd
import rioxarray  # noqa: F401 — activates .rio accessor
import xarray as xr

from ..utils.io import raster_write_kwargs


# ---------------------------------------------------------------------------
# Named preset scenarios
# ---------------------------------------------------------------------------

# Band names in FlamMap/LANDFIRE order (used as long_name attribute)
_BAND_ORDER = ["ELEV", "SLP", "ASP", "FBFM40", "CC", "CH", "CBH", "CBD", "EVT"]

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
        Side length of the square output grid (default ``9``).
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
    stack.rio.to_raster(str(out_path), dtype="int16", nodata=-9999,
                        **raster_write_kwargs())

    print(f"  [create_synthetic_lcp] Wrote {n_pixels}×{n_pixels} LCP → {out_path.name}")
    return out_path.resolve()


# ---------------------------------------------------------------------------
# Combination LCP: one FlamMap cell per unique combination, not per pixel
# ---------------------------------------------------------------------------

#: LCP bands that determine FlamMap's per-cell output under uniform weather,
#: upslope wind (``WIND_DIRECTION: -1``), no gridded winds and no fuel-moisture
#: conditioning.  A fixed wind direction adds aspect; conditioning adds
#: elevation and aspect.  Confirm with :func:`combination_spread`.
COMBINATION_KEYS = ("FBFM40", "SlpD", "CC", "CH", "CBH", "CBD")

#: Tag on stacks written by :func:`expand_combinations`, so they are never
#: mistaken for FlamMap's own output.
EXPANDED_TAG = {"source": "fb_tools.expand_combinations"}


def _band_index(descriptions, name):
    """0-based band index by name; ``'CC'`` matches ``'LF2016_CC_CONUS'``."""
    short = [d.split("_")[1] if d and "_" in d else d for d in descriptions]
    if name not in short:
        raise ValueError(f"no band {name!r} in {descriptions}")
    return short.index(name)


def _keys(a, descriptions, keys, nodata):
    """``(n, len(keys))`` int64 keys over the valid cells of a band stack, and the valid mask."""
    k = a[[_band_index(descriptions, n) for n in keys]]
    valid = np.ones(k.shape[1:], bool) if nodata is None else (k != nodata).all(0)
    return k[:, valid].T.astype(np.int64), valid


def _read_keys(src, keys):
    """:func:`_keys` for an open raster, reading only the key bands."""
    idx = [_band_index(src.descriptions, n) for n in keys]
    return _keys(src.read([i + 1 for i in idx]), [src.descriptions[i] for i in idx],
                 keys, src.nodata)


def _pack(k, dims):
    """One int64 per key row."""
    if (k < 0).any() or (k >= np.asarray(dims)).any():
        raise ValueError("combination values outside the combination LCP's range")
    return np.ravel_multi_index(k.T, dims)


def combination_spread(lcp_path, output_path, keys=COMBINATION_KEYS):
    """
    Largest range of each FlamMap output band within any one combination of *keys*.

    All zeros means FlamMap's output on this landscape is a function of *keys*
    alone, so the run can be replaced by one on a combination LCP.  Check one
    full-grid run per project, and again whenever the run settings change.

    Parameters
    ----------
    lcp_path, output_path : str or Path
        A landscape and a FlamMap output stack run on it (same grid).
    keys : sequence of str
        LCP band names that define a combination.

    Returns
    -------
    dict
        ``{output band: max within-combination range}`` plus ``n_combinations``.
    """
    import rasterio

    with rasterio.open(lcp_path) as src:
        k, valid = _read_keys(src, keys)
    packed = _pack(k, k.max(0) + 1)
    order = np.argsort(packed, kind="stable")
    p = packed[order]
    starts = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])
    spread = {}
    with rasterio.open(output_path) as src:
        for i, d in enumerate(src.descriptions, start=1):
            v = src.read(i)[valid][order]
            spread[d] = float((np.maximum.reduceat(v, starts)
                               - np.minimum.reduceat(v, starts)).max())
    return {**spread, "n_combinations": len(starts)}


def create_combination_lcp(lcp_path, out_path, treatments=None, keys=COMBINATION_KEYS):
    """
    Write a small LCP: each unique combination of *keys* in a baseline once, per treatment.

    Block 0 holds the baseline's unique combinations, each as the complete band
    row of a real pixel (the first with that combination), so no band value is
    invented.  Every treatment is applied to a copy of that block, so cell *j*
    of block *t* is what treatment *t* makes of baseline combination *j*: the
    arithmetic is the treatment function's own, and no treated landscape has to
    be written.  Blocks are stacked top to bottom.

    FlamMap run on the result gives, cell for cell, what a full-grid run gives
    every pixel with that combination -- provided :func:`combination_spread`
    is zero under the run settings.  :func:`expand_combinations` maps a run
    back onto the baseline grid, per treatment.

    Parameters
    ----------
    lcp_path : str or Path
        Baseline landscape.
    out_path : str or Path
        Destination GeoTIFF.  Band descriptions, CRS, origin, resolution, dtype
        and nodata follow *lcp_path*; block layout and *keys* are stored as tags.
    treatments : dict, optional
        ``{name: fn}``.  ``fn`` takes and returns an LCP ``xarray.DataArray``
        (``band`` x ``y`` x ``x`` with a ``long_name`` attribute, as
        :func:`~fb_tools.fuelscape.adjust.apply_treatment` expects) and must act
        cell by cell -- e.g. a ``functools.partial`` of ``apply_treatment``.
    keys : sequence of str
        LCP band names that define a combination.

    Returns
    -------
    pd.DataFrame
        One row per block: its name and the unique combinations in it.
    """
    import json

    import rasterio

    treatments = dict(treatments or {})
    if "baseline" in treatments:
        raise ValueError("'baseline' is reserved for block 0")
    with rasterio.open(lcp_path) as src:
        desc, profile = src.descriptions, src.profile
        k, valid = _read_keys(src, keys)
        _, first = np.unique(_pack(k, k.max(0) + 1), return_index=True)
        cells = np.flatnonzero(valid)[first]
        rows = np.stack([src.read(b).ravel()[cells] for b in range(1, src.count + 1)])

    n = rows.shape[1]
    width = int(np.ceil(np.sqrt(n)))
    height = -(-n // width)
    nodata = -9999 if profile["nodata"] is None else profile["nodata"]
    pad = (np.arange(height * width) >= n).reshape(height, width)
    base = np.full((len(desc), height * width), nodata, dtype=rows.dtype)
    base[:, :n] = rows
    base = base.reshape(len(desc), height, width)
    da = xr.DataArray(base, dims=("band", "y", "x"),
                      coords={"band": np.arange(1, len(desc) + 1)},
                      attrs={"long_name": tuple(desc)})

    blocks = [base]
    for fn in treatments.values():
        t = np.asarray(fn(da.copy(deep=True))).astype(base.dtype)
        t[:, pad] = nodata
        blocks.append(t)
    names = ["baseline", *treatments]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.concatenate(blocks, axis=1)
    profile.update(height=grid.shape[1], width=width, nodata=nodata, **raster_write_kwargs())
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(grid)
        dst.descriptions = desc
        dst.update_tags(combination_blocks=json.dumps(names), combination_block_height=height,
                        combination_keys=json.dumps(list(keys)))
    print(f"  [create_combination_lcp] {n:,} combinations x {len(names)} blocks "
          f"-> {grid.shape[1]} x {width} {out_path.name}")
    return pd.DataFrame([dict(block=name, combinations=len(np.unique(
        _keys(b, desc, keys, nodata)[0], axis=0))) for name, b in zip(names, blocks)])


def expand_combinations(combo_lcp, combo_output, lcp_path, treatment=None,
                        outside=0.0, out_path=None):
    """
    Rebuild the full-grid FlamMap output of one treatment from a run on a combination LCP.

    Parameters
    ----------
    combo_lcp : str or Path
        Written by :func:`create_combination_lcp`.
    combo_output : str or Path
        FlamMap output stack of a run on *combo_lcp*.
    lcp_path : str or Path
        The baseline *combo_lcp* was built from (or any landscape whose
        combinations are all in its baseline block).
    treatment : str, optional
        Block to read; ``None`` for the baseline.
    outside : float
        Value for cells with no data in *lcp_path*.  FlamMap writes 0 there, not
        nodata.
    out_path : str or Path, optional
        Also write the result, laid out like *combo_output* (dtype, nodata, band
        descriptions) on *lcp_path*'s grid and tagged with ``EXPANDED_TAG``.

    Returns
    -------
    np.ndarray
        ``(bands, rows, cols)`` in *combo_output*'s dtype and band order.

    Raises
    ------
    ValueError
        If a valid cell of *lcp_path* has a combination absent from the baseline
        block, or *treatment* is not a block.
    """
    import json

    import rasterio

    with rasterio.open(combo_lcp) as src:
        tags, desc, nodata = src.tags(), src.descriptions, src.nodata
        names = json.loads(tags["combination_blocks"])
        h = int(tags["combination_block_height"])
        keys = json.loads(tags["combination_keys"])
        ck, cvalid = _keys(src.read(window=((0, h), (0, src.width))), desc, keys, nodata)
    name = "baseline" if treatment is None else treatment
    if name not in names:
        raise ValueError(f"no block {name!r} in {Path(combo_lcp).name}: {names}")
    b = names.index(name)
    with rasterio.open(combo_output) as src:
        profile, odesc = src.profile, src.descriptions
        cvals = src.read(window=((b * h, (b + 1) * h), (0, src.width)))[:, cvalid]

    dims = ck.max(0) + 1
    cpacked = _pack(ck, dims)
    order = np.argsort(cpacked)
    cpacked, cvals = cpacked[order], cvals[:, order]

    with rasterio.open(lcp_path) as src:
        k, valid = _read_keys(src, keys)
        transform, crs = src.transform, src.crs
    packed = _pack(k, dims)
    i = np.minimum(np.searchsorted(cpacked, packed), len(cpacked) - 1)
    missing = int((cpacked[i] != packed).sum())
    if missing:
        raise ValueError(f"{Path(lcp_path).name}: {missing:,} cells have combinations "
                         f"not in the baseline block of {Path(combo_lcp).name}")
    out = np.full((len(odesc),) + valid.shape, outside, dtype=cvals.dtype)
    out[:, valid] = cvals[:, i]

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(height=valid.shape[0], width=valid.shape[1], transform=transform,
                       crs=crs, **raster_write_kwargs())
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out)
            dst.descriptions = odesc
            dst.update_tags(**EXPANDED_TAG)
    return out
