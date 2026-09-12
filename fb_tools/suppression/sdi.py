"""
Suppression Difficulty Index (SDI) calculator.

Implements the Rodriguez y Silva et al. (2014) SDI framework as a pure-Python
(numpy / scipy / rioxarray) replacement for the original ArcPy workflow.

SDI combines five sub-indices:
  1. Accessibility        — Euclidean distance to nearest paved road (1–10)
  2. Penetrability        — slope, aspect, trail density, fuel control (formula)
  3. Energy Behavior      — flame length × heat unit area harmonic mean (1–10)
  4. Fireline Opening     — fuel control adjusted for slope (hand + machine work)
  5. KO Slope Hazard      — mobility hazard added for unroaded areas (0.01–0.8)

Final SDI = Energy / (Accessibility + 1 + Penetrability + Fireline)
            + KO slope hazard (unroaded areas only)

Delta SDI = Baseline SDI − Treatment SDI

Reference lookup table (RTC, western US 2021):
  ``08_RTC_lookup_SDIwt_westernUS_2021_update.txt``
"""

import gc
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
from scipy.ndimage import convolve, distance_transform_edt


# ── Classification helpers ───────────────────────────────────────────────────
# All reclassification functions accept and return float32 numpy arrays.
# Breakpoints transcribed from Delta_SDI_v3.ipynb (Rodriguez y Silva et al.).

def _classify_aspect(aspect):
    """Aspect (degrees 0–360) → difficulty class (1–10).

    North-facing slopes are hardest to access (class 10); south-facing
    slopes are easiest (class 1).  Flat terrain (aspect = -1) treated as
    class 5.

    Parameters
    ----------
    aspect : numpy.ndarray
        Aspect in degrees.

    Returns
    -------
    numpy.ndarray
        Float32 array, values in {1, 3, 4, 5, 6, 7, 8, 10}.
    """
    a = aspect.astype(np.float32)
    conditions = [
        (a >= 337.5) | (a < 22.5),   # North
        (a >= 22.5)  & (a < 67.5),   # Northeast
        (a >= 292.5) & (a < 337.5),  # Northwest
        (a >= 67.5)  & (a < 112.5),  # East
        (a >= 247.5) & (a < 292.5),  # West
        (a >= 112.5) & (a < 157.5),  # Southeast
        (a >= 202.5) & (a < 247.5),  # Southwest
        (a >= 157.5) & (a < 202.5),  # South
    ]
    choices = [10, 8, 7, 6, 5, 4, 3, 1]
    return np.select(conditions, choices, default=5).astype(np.float32)


def _classify_slope_pen(slope):
    """Slope (% rise) → penetrability difficulty class (1–10).

    Gentler slopes are harder to suppress (crews can move freely but fire
    also spreads freely); steeper slopes degrade machine mobility.

    Parameters
    ----------
    slope : numpy.ndarray
        Slope in percent rise.

    Returns
    -------
    numpy.ndarray
        Float32 array, values 1–10.
    """
    s = slope.astype(np.float32)
    conditions = [s < 6, s < 11, s < 16, s < 21, s < 26, s < 31, s < 36, s < 41, s < 46]
    choices    = [10,    9,      8,      7,      6,      5,      4,      3,      2]
    return np.select(conditions, choices, default=1).astype(np.float32)


def _classify_trail_length(trail_len_m):
    """Trail length within 1-ha kernel (m) → trail density class (1–10).

    More trail infrastructure = easier penetration = higher class.

    Parameters
    ----------
    trail_len_m : numpy.ndarray
        Trail length in metres within the ~1-ha moving window.

    Returns
    -------
    numpy.ndarray
        Float32 array, values 1–10.
    """
    t = trail_len_m.astype(np.float32)
    conditions = [t < 10, t < 20, t < 30, t < 40, t < 50, t < 60, t < 70, t < 80, t < 90]
    choices    = [1,      2,      3,      4,      5,      6,      7,      8,      9]
    return np.select(conditions, choices, default=10).astype(np.float32)


def _classify_accessibility(dist_m):
    """Road distance (m) → accessibility class (1–10, 10 = nearest road).

    Parameters
    ----------
    dist_m : numpy.ndarray
        Euclidean distance to nearest road in metres.

    Returns
    -------
    numpy.ndarray
        Float32 array, values 1–10.
    """
    d = dist_m.astype(np.float32)
    conditions = [d <= 100, d <= 200, d <= 300, d <= 400, d <= 500,
                  d <= 600, d <= 700, d <= 800, d <= 900]
    choices    = [10,       9,        8,        7,        6,
                  5,        4,        3,        2]
    return np.select(conditions, choices, default=1).astype(np.float32)


def _classify_fl(fl_m):
    """Flame length (m) → energy class (1–10).

    Parameters
    ----------
    fl_m : numpy.ndarray
        Flame length in metres (FlamMap FLAMELENGTH output).

    Returns
    -------
    numpy.ndarray
        Float32 array, values 1–10.
    """
    f = fl_m.astype(np.float32)
    conditions = [f <= 0.5, f <= 1.0, f <= 1.5, f <= 2.0, f <= 2.5,
                  f <= 3.0, f <= 3.5, f <= 4.0, f <= 4.5]
    choices    = [1,        2,        3,         4,         5,
                  6,        7,        8,         9]
    return np.select(conditions, choices, default=10).astype(np.float32)


def _classify_hua(hua_kj_m2):
    """Heat Unit Area (kJ/m²) → energy class (1–10).

    Converts to kcal/m² (× 0.2388459) before reclassification.

    Parameters
    ----------
    hua_kj_m2 : numpy.ndarray
        Heat unit area in kJ/m² (FlamMap HEATAREA output).

    Returns
    -------
    numpy.ndarray
        Float32 array, values 1–10.
    """
    h = (hua_kj_m2.astype(np.float32) * np.float32(0.2388459))
    conditions = [h <= 380,  h <= 1265, h <= 1415, h <= 1610, h <= 1905,
                  h <= 2190, h <= 4500, h <= 6630, h <= 8000]
    choices    = [1,         2,         3,         4,         5,
                  6,         7,         8,         9]
    return np.select(conditions, choices, default=10).astype(np.float32)


def _slope_adjust_hand(slope):
    """Slope adjustment factor for hand crews (0.5–1.0).

    Parameters
    ----------
    slope : numpy.ndarray
        Slope in percent rise.

    Returns
    -------
    numpy.ndarray
        Float32 array.
    """
    s = slope.astype(np.float32)
    conditions = [s < 20, s < 30, s < 40]
    choices    = [1.0,    0.75,   0.625]
    return np.select(conditions, choices, default=0.5).astype(np.float32)


def _slope_adjust_mach(slope):
    """Slope adjustment factor for machine crews (0.0–1.0).

    Parameters
    ----------
    slope : numpy.ndarray
        Slope in percent rise.

    Returns
    -------
    numpy.ndarray
        Float32 array.
    """
    s = slope.astype(np.float32)
    conditions = [s < 30, s < 40]
    choices    = [1.0,    0.5]
    return np.select(conditions, choices, default=0.0).astype(np.float32)


def _calc_slope_hazard(slope):
    """KO universal slope mobility hazard (0.01–0.8).

    Applied to unroaded areas only (road distance > 100 m).

    Parameters
    ----------
    slope : numpy.ndarray
        Slope in percent rise.

    Returns
    -------
    numpy.ndarray
        Float32 array.
    """
    s = slope.astype(np.float32)
    conditions = [s <= 10, s <= 20, s <= 30, s <= 40, s <= 50]
    choices    = [0.01,    0.1,     0.2,     0.4,     0.6]
    return np.select(conditions, choices, default=0.8).astype(np.float32)


# ── Raster computation helpers ───────────────────────────────────────────────

def _align_to_reference(da, reference):
    """Reproject and snap *da* to the grid of *reference*.

    Parameters
    ----------
    da : xarray.DataArray
    reference : xarray.DataArray
        Single-band (y, x) target grid.

    Returns
    -------
    xarray.DataArray
    """
    return da.rio.reproject_match(reference)


def _load_rtc_table(rtc_path):
    """Load an FBFM40 → RTC lookup text file into a numpy array.

    File format: one entry per line, ``FBFM40_code:RTC_value``, e.g.::

        101:10
        185:2
        ...

    RTC values (1–10) represent resistance to crew; higher = harder to
    control (more difficult fuel).

    Parameters
    ----------
    rtc_path : Path

    Returns
    -------
    numpy.ndarray
        1-D float32 array of length ``max(FBFM40_code) + 1``.  Index with
        an integer FBFM40 code to get its RTC value.

    Raises
    ------
    FileNotFoundError
        If *rtc_path* does not exist.
    ValueError
        If the file cannot be parsed.
    """
    if not rtc_path.is_file():
        raise FileNotFoundError(f"RTC lookup table not found: {rtc_path}")

    codes, values = [], []
    with rtc_path.open() as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                code_str, val_str = line.split(":")
                codes.append(int(code_str.strip()))
                values.append(float(val_str.strip()))
            except ValueError:
                raise ValueError(
                    f"Cannot parse line {line_num} of {rtc_path.name}: {line!r}"
                )

    if not codes:
        raise ValueError(f"No entries found in RTC lookup table: {rtc_path}")

    max_code = max(codes)
    lut = np.zeros(max_code + 1, dtype=np.float32)
    for code, val in zip(codes, values):
        lut[code] = val
    return lut


def _road_distance_m(roads_gdf, reference_da, max_road_code=7):
    """Compute per-pixel Euclidean distance (metres) to the nearest road.

    Only features with ``highway_code <= max_road_code`` are used (default
    7 = motorway through residential, excluding service and track).  This
    matches the original SpeedCat < 8 filter from the ArcPy workflow.

    Parameters
    ----------
    roads_gdf : GeoDataFrame
        Road features with a ``highway_code`` column.
    reference_da : xarray.DataArray
        Single-band (y, x) reference grid.
    max_road_code : int
        Highest highway_code to include (default 7).

    Returns
    -------
    numpy.ndarray
        Float32 2-D array of distances in metres.

    Raises
    ------
    RuntimeError
        If no roads remain after the code filter.
    """
    from ..utils.geo import rasterize

    roads_sub = roads_gdf[roads_gdf["highway_code"] <= max_road_code].copy()
    if roads_sub.empty:
        raise RuntimeError(
            f"No road features with highway_code <= {max_road_code}. "
            "Cannot compute accessibility sub-index."
        )

    roads_sub["burn"] = 1
    road_grid = rasterize(roads_sub, reference_da, attr="burn", fill_val=0)
    road_binary = (road_grid.values.squeeze() == 1)
    del road_grid
    gc.collect()

    cell_size = float(abs(reference_da.rio.transform().a))
    dist_pixels = distance_transform_edt(~road_binary)
    del road_binary
    return (dist_pixels * cell_size).astype(np.float32)


def _trail_density_grid(trails_gdf, reference_da, radius_m=56.42):
    """Compute per-pixel trail length (metres) within a ~1-ha circular kernel.

    Rasterizes trail features to a binary grid, then convolves with a
    disk-shaped kernel of radius *radius_m*.  The result is multiplied by
    cell size to express trail density as metres of trail per kernel window.

    A radius of 56.42 m corresponds to a 1-hectare circle (π r² ≈ 10 000 m²).

    Parameters
    ----------
    trails_gdf : GeoDataFrame
        Trail features.  An empty GeoDataFrame returns an all-zero grid.
    reference_da : xarray.DataArray
        Single-band (y, x) reference grid.
    radius_m : float
        Kernel radius in metres (default 56.42).

    Returns
    -------
    numpy.ndarray
        Float32 2-D array of trail length in metres within the kernel.
    """
    from ..utils.geo import rasterize

    ref_shape = reference_da.squeeze().shape

    if trails_gdf.empty:
        print("  Warning: no trail features provided; trail density set to zero.")
        return np.zeros(ref_shape, dtype=np.float32)

    trails_sub = trails_gdf.copy()
    trails_sub["burn"] = 1
    trail_grid = rasterize(trails_sub, reference_da, attr="burn", fill_val=0)
    trail_binary = (trail_grid.values.squeeze() == 1).astype(np.float32)
    del trail_grid
    gc.collect()

    cell_size = float(abs(reference_da.rio.transform().a))
    r_px = int(np.ceil(radius_m / cell_size))

    # Build a circular disk kernel
    y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    disk = (x ** 2 + y ** 2 <= r_px ** 2).astype(np.float32)

    trail_count = convolve(trail_binary, disk, mode="constant", cval=0.0)
    del trail_binary
    return (trail_count * cell_size).astype(np.float32)


# ── Public API ───────────────────────────────────────────────────────────────

SDI_NODATA = -9999


def _sdi_reference_grid(lcp):
    """Resolve the single-band (y, x) grid that SDI is computed on.

    SDI is written on the *slope* band of the LCP.  Both :func:`calculate_sdi`
    and :func:`sdi_static_layers` must resolve it identically, or a cached
    static layer would be silently misaligned against the scenario it is
    reused for.

    Parameters
    ----------
    lcp : str, Path, or xarray.DataArray

    Returns
    -------
    xarray.DataArray
        2-D reference grid.
    dict
        Normalised band name -> band index.
    """
    from ..fuelscape.adjust import _normalize_band_name

    if isinstance(lcp, (str, Path)):
        lcp = rxr.open_rasterio(Path(lcp), masked=True)

    long_names = lcp.attrs.get("long_name")
    if not long_names:
        raise ValueError(
            "lcp DataArray has no 'long_name' attribute. "
            "Load the LCP via stack_rasters or rxr.open_rasterio to preserve band names."
        )
    if isinstance(long_names, str):
        long_names = [long_names]

    band_map = {
        _normalize_band_name(n): int(b)
        for n, b in zip(long_names, lcp.band.values)
    }
    required = {"SlpD", "Asp", "FBFM40"}
    missing = required - band_map.keys()
    if missing:
        raise ValueError(
            f"Required band(s) not found in lcp: {sorted(missing)}. "
            f"Available (normalised): {sorted(band_map.keys())}"
        )
    return lcp.sel(band=band_map["SlpD"]).squeeze(), band_map


def sdi_static_layers(lcp, roads_gdf, trails_gdf, cache_path=None):
    """Compute (or load) the treatment-invariant SDI inputs.

    Road distance and trail density depend only on the OSM geometry and the
    grid — no treatment scenario touches either — so they are identical across
    every LCP variant and can be computed once and reused.  Pass the result
    into :func:`calculate_sdi` as ``road_dist=`` / ``trail_len=``.

    The saving is modest in wall-clock terms (measured ~13 s per scenario on a
    14783 x 13430 grid with 319k OSM roads); the real benefit is not rebuilding
    two 0.8 GB float32 arrays per scenario.

    Parameters
    ----------
    lcp : str, Path, or xarray.DataArray
        LCP defining the target grid.  Must already be aligned to the grid the
        scenarios will be computed on — see the note on `cache_path` below.
    roads_gdf, trails_gdf : GeoDataFrame
    cache_path : str or Path, optional
        ``.npz`` to read from if present, else write to.  The cache stores the
        grid shape and transform and **refuses to load** against a different
        grid, since a silently misaligned road-distance surface would corrupt
        every scenario that reused it.

    Returns
    -------
    dict
        ``{"road_dist": ndarray, "trail_len": ndarray}``, both float32 2-D.
    """
    ref, _ = _sdi_reference_grid(lcp)
    shape = tuple(ref.shape)
    transform = np.asarray(ref.rio.transform(), dtype="float64")[:6]

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            with np.load(cache_path) as z:
                if (tuple(z["shape"]) == shape
                        and np.allclose(z["transform"], transform)):
                    print(f"  SDI static layers: cache hit ({cache_path.name})")
                    return {"road_dist": z["road_dist"], "trail_len": z["trail_len"]}
                raise ValueError(
                    f"SDI static cache {cache_path.name} was built on grid "
                    f"{tuple(z['shape'])} / {z['transform'][:3]}, but this LCP is "
                    f"{shape} / {transform[:3]}. Delete the cache to rebuild it."
                )

    print("Computing road distance...")
    road_dist = _road_distance_m(roads_gdf, ref)
    print("Computing trail density...")
    trail_len = _trail_density_grid(trails_gdf, ref)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, road_dist=road_dist, trail_len=trail_len,
                            shape=np.array(shape), transform=transform)
        print(f"  SDI static layers cached -> {cache_path}")

    return {"road_dist": road_dist, "trail_len": trail_len}


def calculate_sdi(lcp, roads_gdf, trails_gdf, rtc_path,
                  flammap_stack=None,
                  flame_length=None, heat_area=None,
                  out_path=None, nodata=SDI_NODATA,
                  road_dist=None, trail_len=None):
    """
    Calculate the Suppression Difficulty Index (SDI) for a landscape.

    Assembles five sub-indices (accessibility, penetrability, energy
    behavior, fireline opening, and KO slope mobility hazard) into a final
    SDI raster scaled by 100 and cast to int16.

    Parameters
    ----------
    lcp : str, Path, or xarray.DataArray
        Landscape Characteristic Package raster.  Must contain bands named
        ``SlpD`` (slope in degrees, as downloaded by LFPS with ``LF2020_SLPD``),
        ``Asp`` (aspect, degrees 0–360), and ``FBFM40`` (40-class fuel model)
        in the ``long_name`` attribute.  Slope is converted internally to
        percent rise for SDI classification.  Opened with rioxarray if a file
        path is supplied.
    roads_gdf : GeoDataFrame
        Road features with a ``highway_code`` column (1–9).  Obtain via
        :func:`~fb_tools.suppression.roads.fetch_osm_roads`.
    trails_gdf : GeoDataFrame
        Trail features (path, footway, bridleway, track).  Obtain via
        :func:`~fb_tools.suppression.roads.fetch_osm_roads`.
    rtc_path : str or Path
        Path to the RTC (Resistance to Crew) lookup text file.  Format:
        ``FBFM40_code:RTC_value`` per line.  The western US 2021 version
        ships in ``code/dev/SDI/08_RTC_lookup_SDIwt_westernUS_2021_update.txt``.
    flammap_stack : str, Path, or xarray.DataArray, optional
        Stacked multi-band FlamMap output (produced by ``run_batch`` with
        ``stack_out=True``).  Must contain bands named ``FLAMELENGTH`` and
        ``HEATAREA`` in the ``long_name`` attribute.  Use
        :func:`~fb_tools.models.scenarios.stacked_output_path` to resolve
        the path from batch run parameters.  Mutually exclusive with
        *flame_length* / *heat_area*; provide one or the other.
    flame_length : str, Path, or xarray.DataArray, optional
        FlamMap FLAMELENGTH output raster, in metres.  Required when
        *flammap_stack* is not provided.  Reprojected to the *lcp* grid if
        extents differ.
    heat_area : str, Path, or xarray.DataArray, optional
        FlamMap HEATAREA output raster, in kJ/m².  Required when
        *flammap_stack* is not provided.  Reprojected to the *lcp* grid if
        extents differ.
    out_path : str or Path, optional
        If provided, the SDI raster is written to this GeoTIFF path.
    nodata : int, optional
        Value written wherever any required input is nodata (default -9999).
    road_dist, trail_len : numpy.ndarray, optional
        Precomputed static sub-index inputs from :func:`sdi_static_layers`.
        Neither depends on the treatment, so they can be built once and reused
        across every scenario sharing the grid.  Computed here if omitted.

    Returns
    -------
    xarray.DataArray
        SDI values = raw SDI × 100, dtype int16, with *nodata* outside the
        valid footprint.  CRS and spatial metadata are preserved from *lcp*.

    Raises
    ------
    FileNotFoundError
        If *rtc_path* does not exist.
    ValueError
        If required bands (SLP, ASP, FBFM40) are missing from *lcp*, or if
        *lcp* has no ``long_name`` attribute.
    RuntimeError
        If *roads_gdf* contains no usable features after filtering.

    Notes
    -----
    Assembly formula::

        fuel_cntrl  = RTC_lookup[FBFM40]
        energy      = 2·FL_recl·HUA_recl / (FL_recl + HUA_recl)   # harmonic mean
        access      = classify_accessibility(road_distance_m)
        penetr      = (slope_class + fuel_cntrl + aspect_class + 2·trail_class) / 5
        fireline    = fuel_cntrl · (slope_adj_hand + slope_adj_mach)
        sdi_base    = energy / (access + 1 + penetr + fireline)
        sdi_slope   = sdi_base + KO_slope_hazard
        sdi_final   = sdi_base  (where road_distance_m ≤ 100)
                    = sdi_slope (elsewhere)
        output      = round(sdi_final × 100) as int16, `nodata` where invalid

    Nodata is tracked explicitly because every ``_classify_*`` helper is built
    on ``np.select``, and NaN compares False against every condition — so a
    nodata pixel silently falls through to the ``default`` branch.  For
    ``_classify_fl`` and ``_classify_hua`` that default is **10**, the maximum
    class, so an unmasked FlamMap footprint used to yield ``energy = 10``
    (a fabricated maximum-energy field) rather than an obvious 0.  ``FBFM40``
    is worse still: ``np.nan.astype(np.int32)`` is undefined.  The validity
    mask below is built from the raw inputs, before any classification.
    """
    from ..fuelscape.lcp import get_band_by_longname

    # ── Accept stacked FlamMap output as single-file alternative ─────────────
    if flammap_stack is not None:
        if isinstance(flammap_stack, (str, Path)):
            flammap_stack = rxr.open_rasterio(Path(flammap_stack), masked=True)
        flame_length = get_band_by_longname(flammap_stack, "FLAMELENGTH")
        heat_area    = get_band_by_longname(flammap_stack, "HEATAREA")
        del flammap_stack
        gc.collect()
    elif flame_length is None or heat_area is None:
        raise ValueError(
            "Provide either 'flammap_stack' or both 'flame_length' and 'heat_area'."
        )

    # ── Load rasters if paths were provided ─────────────────────────────────
    if isinstance(lcp, (str, Path)):
        lcp = rxr.open_rasterio(Path(lcp), masked=True)
    if isinstance(flame_length, (str, Path)):
        flame_length = rxr.open_rasterio(Path(flame_length), masked=True)
    if isinstance(heat_area, (str, Path)):
        heat_area = rxr.open_rasterio(Path(heat_area), masked=True)

    rtc_path = Path(rtc_path)

    # ── Validate band names and resolve the reference grid ───────────────────
    ref, band_map = _sdi_reference_grid(lcp)

    # ── Align fire behavior outputs to LCP grid if needed ───────────────────
    flame_length = _align_to_reference(flame_length.squeeze(), ref)
    heat_area    = _align_to_reference(heat_area.squeeze(), ref)

    # ── Extract numpy arrays (float32 throughout to conserve memory) ─────────
    # SLPD band is in degrees (as required by FlamMap); convert to percent rise
    # for SDI classification functions (all breakpoints use percent rise).
    slope  = lcp.sel(band=band_map["SlpD"]).values.squeeze().astype(np.float32)
    slope  = (np.tan(np.radians(slope)) * 100.0).astype(np.float32)
    aspect = lcp.sel(band=band_map["Asp"]).values.squeeze().astype(np.float32)
    fbfm40 = lcp.sel(band=band_map["FBFM40"]).values.squeeze()
    fl_arr = flame_length.values.astype(np.float32)
    hu_arr = heat_area.values.astype(np.float32)

    # Release large xarray objects before the compute-heavy step
    del lcp, flame_length, heat_area
    gc.collect()

    # ── Validity mask ────────────────────────────────────────────────────────
    # Built from the RAW inputs, before any np.select classification silently
    # promotes NaN to a `default` class (see Notes).  The FlamMap footprint is
    # typically far smaller than the LCP extent, so this is what confines SDI
    # to the modeled area.
    valid = (np.isfinite(slope) & np.isfinite(aspect)
             & np.isfinite(fl_arr) & np.isfinite(hu_arr))
    if fbfm40.dtype.kind == "f":
        valid &= np.isfinite(fbfm40)
    if not valid.any():
        raise ValueError(
            "No valid pixels: the LCP and FlamMap inputs have no overlapping "
            "valid data. Check that they cover the same area."
        )

    # ── RTC lookup ───────────────────────────────────────────────────────────
    rtc_lut = _load_rtc_table(rtc_path)
    # np.nan.astype(int32) is undefined; zero the invalid cells first so the
    # clip below indexes a defined value. They are masked out at the end.
    fbfm40 = np.where(valid, fbfm40, 0)
    fbfm40_idx = np.clip(fbfm40.astype(np.int32), 0, len(rtc_lut) - 1)
    fuel_cntrl = rtc_lut[fbfm40_idx].astype(np.float32)
    del fbfm40, fbfm40_idx

    # ── Road distance and trail density ──────────────────────────────────────
    # Both are treatment-invariant; sdi_static_layers() builds them once for a
    # whole scenario set. Shape is asserted because a mismatched cache would
    # otherwise broadcast or silently misalign the accessibility sub-index.
    ref_shape = tuple(np.shape(slope))
    if road_dist is None:
        print("Computing road distance...")
        road_dist = _road_distance_m(roads_gdf, ref)
    elif tuple(road_dist.shape) != ref_shape:
        raise ValueError(
            f"road_dist shape {tuple(road_dist.shape)} != grid {ref_shape}"
        )
    road_dist = road_dist.astype(np.float32, copy=False)

    if trail_len is None:
        print("Computing trail density...")
        trail_len = _trail_density_grid(trails_gdf, ref)
    elif tuple(trail_len.shape) != ref_shape:
        raise ValueError(
            f"trail_len shape {tuple(trail_len.shape)} != grid {ref_shape}"
        )
    trail_len = trail_len.astype(np.float32, copy=False)

    # ── Sub-indices ───────────────────────────────────────────────────────────
    print("Computing sub-indices...")

    accessibility = _classify_accessibility(road_dist)
    aspect_class  = _classify_aspect(aspect)
    slope_class   = _classify_slope_pen(slope)
    trail_class   = _classify_trail_length(trail_len)
    del trail_len

    penetrability = (slope_class + fuel_cntrl + aspect_class + 2.0 * trail_class) / 5.0
    del aspect_class, slope_class, trail_class

    fl_recl  = _classify_fl(fl_arr)
    hua_recl = _classify_hua(hu_arr)
    del fl_arr, hu_arr
    denom_eb = fl_recl + hua_recl
    energy = np.where(denom_eb > 0, 2.0 * fl_recl * hua_recl / denom_eb, 0.0)
    energy = energy.astype(np.float32)
    del fl_recl, hua_recl, denom_eb

    adj_hand = _slope_adjust_hand(slope)
    adj_mach = _slope_adjust_mach(slope)
    fireline = (fuel_cntrl * (adj_hand + adj_mach)).astype(np.float32)
    del adj_hand, adj_mach, fuel_cntrl

    slope_hazard = _calc_slope_hazard(slope)
    del slope

    # ── Assemble final SDI ────────────────────────────────────────────────────
    print("Assembling final SDI...")
    denom = accessibility + 1.0 + penetrability + fireline
    sdi_base  = np.where(denom > 0, energy / denom, 0.0).astype(np.float32)
    sdi_slope = (sdi_base + slope_hazard).astype(np.float32)
    roaded    = road_dist <= 100.0
    sdi_final = np.where(roaded, sdi_base, sdi_slope)
    del energy, accessibility, penetrability, fireline, slope_hazard
    del sdi_base, sdi_slope, roaded, denom, road_dist
    gc.collect()

    sdi_out = np.round(sdi_final * 100.0).astype(np.int16)
    del sdi_final
    sdi_out[~valid] = nodata
    del valid
    gc.collect()

    # ── Wrap in DataArray, preserve CRS/transform ─────────────────────────────
    da_out = xr.DataArray(sdi_out, dims=ref.dims, coords=ref.coords)
    da_out = da_out.rio.write_crs(ref.rio.crs)
    da_out = da_out.rio.write_transform(ref.rio.transform())
    da_out = da_out.rio.write_nodata(nodata)
    da_out.attrs["long_name"] = "SDI"

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        da_out.rio.to_raster(
            str(out_path), dtype="int16", nodata=nodata,
            compress="deflate", predictor=2, zlevel=9,
            tiled=True, blockxsize=256, blockysize=256,
        )
        print(f"SDI saved → {out_path}")

    return da_out


def calculate_delta_sdi(baseline_sdi, treatment_sdi, out_path=None,
                        nodata=SDI_NODATA):
    """
    Compute the difference between a baseline and a treatment SDI raster.

    Delta SDI = Baseline SDI − Treatment SDI.  Positive values indicate
    that the treatment reduced suppression difficulty.

    Parameters
    ----------
    baseline_sdi : str, Path, or xarray.DataArray
        Baseline SDI raster (output of :func:`calculate_sdi`).
    treatment_sdi : str, Path, or xarray.DataArray
        Treatment SDI raster.  Reprojected to match *baseline_sdi* if
        grids differ.
    out_path : str or Path, optional
        If provided, write the delta raster as a GeoTIFF.
    nodata : int, optional
        Value written where either input is nodata (default -9999).

    Returns
    -------
    xarray.DataArray
        Delta SDI (int16, same ×100 scale as the inputs), with *nodata*
        wherever either input is nodata.

    Raises
    ------
    ValueError
        If *baseline_sdi* and *treatment_sdi* have different CRS.

    Notes
    -----
    Both inputs are opened with ``masked=True``, so nodata arrives as NaN and
    ``NaN.astype(np.int32)`` is undefined — the difference must be taken over a
    validity mask rather than over the raw arrays.  Without it, two nodata
    pixels would also difference to a plausible-looking 0 ("no change") outside
    the modeled footprint.
    """
    if isinstance(baseline_sdi, (str, Path)):
        baseline_sdi = rxr.open_rasterio(Path(baseline_sdi), masked=True).squeeze()
    if isinstance(treatment_sdi, (str, Path)):
        treatment_sdi = rxr.open_rasterio(Path(treatment_sdi), masked=True).squeeze()

    # Align treatment to baseline grid if needed
    if treatment_sdi.shape != baseline_sdi.shape:
        treatment_sdi = _align_to_reference(treatment_sdi, baseline_sdi)

    b_arr = baseline_sdi.values
    t_arr = treatment_sdi.values
    valid = np.isfinite(b_arr) & np.isfinite(t_arr)
    del treatment_sdi

    delta_arr = np.full(b_arr.shape, nodata, dtype=np.int16)
    np.subtract(b_arr, t_arr, out=delta_arr, where=valid, casting="unsafe")
    delta_arr[~valid] = nodata
    del b_arr, t_arr, valid

    da_out = xr.DataArray(delta_arr, dims=baseline_sdi.dims, coords=baseline_sdi.coords)
    da_out = da_out.rio.write_crs(baseline_sdi.rio.crs)
    da_out = da_out.rio.write_transform(baseline_sdi.rio.transform())
    da_out = da_out.rio.write_nodata(nodata)
    da_out.attrs["long_name"] = "Delta SDI"
    del baseline_sdi, delta_arr
    gc.collect()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        da_out.rio.to_raster(
            str(out_path), dtype="int16", nodata=nodata,
            compress="deflate", predictor=2, zlevel=9,
            tiled=True, blockxsize=256, blockysize=256,
        )
        print(f"Delta SDI saved → {out_path}")

    return da_out
