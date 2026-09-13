"""
LCP de-stacking — multi-band LANDFIRE landscape → per-variable engine inputs.

Two consumers, one de-stacker:

pyretechnics
    ``destack_lcp`` writes the eight rasters the engine requires, each in the
    units pyretechnics documents, so ``models.pyre.load_static_layers`` /
    ``build_cubes`` are a read-and-wrap with no arithmetic in them.

Cell2Fire
    ``lcp_to_cell2fire`` / ``build_cell2fire_weather`` remain stubs. They are
    kept, not deleted — see ``docs/pyretechnics_transmission_plan.md``.

Why not ``pyretechnics.load_landfire``
--------------------------------------
``read_landfire_rasters_as_space_time_cubes`` applies ``slope * 0.01`` because it
expects LANDFIRE **SLPP** (slope in percent). Our LCPs carry **SLPD** (slope in
degrees): band 2 is ``LF2020_SlpD_CONUS``, 0–70. Running a degrees band through
that loader turns a 20° slope into 0.20 rise/run — 11.3° — a silent 43%
under-slope on exactly the terrain that drives spread, with no error and a
plausible-looking answer.

The fix belongs here rather than upstream in the LCP. ``SLPD`` is what FlamMap,
FSPro and ``suppression/sdi.py::_classify_slope_pen`` all want, and adding a
tenth ``SLPP`` band would break ``synthetic.py::_BAND_ORDER`` and ``adjust.py``'s
band maps. So the conversion happens **once**, here, and ``slope_units="auto"``
**raises** rather than guessing when the band description says neither ``SlpD``
nor ``SlpP``. Every row of ``PYRE_LAYERS`` is unit-tested, because every one of
them produces a plausible wrong answer when it is wrong.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from ..fuelscape.lcp import _NB_CODES
from ..utils.io import raster_write_kwargs


# ---------------------------------------------------------------------------
# Layer table
# ---------------------------------------------------------------------------

#: pyretechnics layer name -> (canonical fb_tools band, multiplier, unit).
#:
#: ``multiplier=None`` marks a layer whose conversion is not a scalar multiply
#: and is handled explicitly in ``_convert``.
#:
#: Multipliers encode LANDFIRE's integer storage conventions:
#:   CC   percent           -> fraction
#:   CH   decimetres        -> metres
#:   CBH  decimetres        -> metres
#:   CBD  kg/m^3 * 100      -> kg/m^3
PYRE_LAYERS = {
    "elevation":           ("ELEV",   1.0,  "m"),
    "slope":               ("SLP",    None, "rise/run"),   # degrees -> tan(radians(x))
    "aspect":              ("ASP",    1.0,  "deg CW from N"),
    "fuel_model":          ("FBFM40", 1,    "Scott & Burgan 40 code"),
    "canopy_cover":        ("CC",     0.01, "fraction"),
    "canopy_height":       ("CH",     0.1,  "m"),
    "canopy_base_height":  ("CBH",    0.1,  "m"),
    "canopy_bulk_density": ("CBD",    0.01, "kg/m3"),
}

#: Layers pyretechnics reads as an integer fuel-model index. Everything else is
#: float32; these stay integral and must never be bilinearly resampled.
_CATEGORICAL = {"fuel_model"}

#: Physically admissible range per layer, checked after conversion. These are
#: deliberately generous — they catch a units error (a 43% under-slope, a canopy
#: cover of 8500%), not a questionable landscape.
_VALID_RANGE = {
    "elevation":           (-500.0, 9000.0),     # m
    "slope":               (0.0, 20.0),          # rise/run; tan(87 deg) ~ 19.1
    "aspect":              (0.0, 360.0),         # deg
    "canopy_cover":        (0.0, 1.0),           # fraction
    "canopy_height":       (0.0, 120.0),         # m
    "canopy_base_height":  (0.0, 100.0),         # m
    "canopy_bulk_density": (0.0, 5.0),           # kg/m3
}

#: Scott & Burgan 40 surface fuel models: 13 non-burnable/anchor codes plus
#: 101-204. 94-97 are unassigned in FBFM40 and are NOT valid.
_VALID_FBFM40 = (
    _NB_CODES
    | set(range(101, 110))   # GR1-GR9
    | set(range(121, 125))   # GS1-GS4
    | set(range(141, 150))   # SH1-SH9
    | set(range(161, 166))   # TU1-TU5
    | set(range(181, 190))   # TL1-TL9
    | set(range(201, 205))   # SB1-SB4
)

#: Aliases seen in LANDFIRE band descriptions, normalized to the canonical name.
#: Matching is case-insensitive on the middle token of ``LF<year>_<name>_CONUS``.
_BAND_ALIASES = {
    "elev": "ELEV", "elevation": "ELEV", "dem": "ELEV",
    "slp": "SLP", "slpd": "SLP", "slpp": "SLP", "slope": "SLP",
    "asp": "ASP", "aspect": "ASP",
    "fbfm40": "FBFM40", "f40": "FBFM40", "fbfm": "FBFM40",
    "cc": "CC", "cbd": "CBD", "cbh": "CBH", "ch": "CH",
    "evt": "EVT",
}

#: Band descriptions that declare slope units unambiguously.
_SLOPE_DEGREE_TOKENS = {"slpd"}
_SLOPE_PERCENT_TOKENS = {"slpp"}


def _canonical_band_name(description):
    """
    Normalize a LANDFIRE band description to a canonical fb_tools band name.

    ``"LF2020_SlpD_CONUS" -> "SLP"``, ``"LF2016_FBFM40_CONUS" -> "FBFM40"``.
    Returns ``None`` when the description matches no known band, so the caller
    can report which bands a file actually carries.
    """
    if not description:
        return None
    parts = str(description).split("_")
    # LF<year>_<name>_CONUS is the LFPS form; fall back to the whole string for
    # hand-built or synthetic stacks whose bands are named "SLP", "CC", ...
    candidates = ([parts[1]] if len(parts) >= 2 else []) + [parts[0], str(description)]
    for token in candidates:
        hit = _BAND_ALIASES.get(token.strip().lower())
        if hit:
            return hit
    return None


def _band_index(src, canonical):
    """1-based band index for *canonical* in an open rasterio dataset."""
    for idx, desc in enumerate(src.descriptions, start=1):
        if _canonical_band_name(desc) == canonical:
            return idx, desc
    return None, None


def _resolve_slope_units(description, slope_units):
    """
    Decide whether the slope band is degrees or percent.

    Parameters
    ----------
    description : str or None
        The slope band's description, e.g. ``"LF2020_SlpD_CONUS"``.
    slope_units : {"auto", "degrees", "percent"}
        ``"auto"`` reads the description and **raises** if it says neither
        ``SlpD`` nor ``SlpP``.

    Raises
    ------
    ValueError
        On an unrecognized *slope_units*, or on ``"auto"`` against a band
        description that does not declare its units.
    """
    if slope_units in ("degrees", "percent"):
        return slope_units
    if slope_units != "auto":
        raise ValueError(
            f"slope_units must be 'auto', 'degrees' or 'percent', got {slope_units!r}"
        )

    parts = str(description or "").split("_")
    token = (parts[1] if len(parts) >= 2 else str(description or "")).strip().lower()
    if token in _SLOPE_DEGREE_TOKENS:
        return "degrees"
    if token in _SLOPE_PERCENT_TOKENS:
        return "percent"

    raise ValueError(
        f"Cannot resolve slope units from band description {description!r}. "
        "LANDFIRE SLPD is degrees and SLPP is percent, and confusing them is "
        "silent: a 20 deg slope read as percent becomes 0.20 rise/run (11.3 deg), "
        "a 43% under-slope with no error. Pass slope_units='degrees' or "
        "'percent' explicitly once you have confirmed which this band is."
    )


def _convert(layer, arr, valid, slope_units):
    """
    Convert one raw LANDFIRE band to pyretechnics units.

    Parameters
    ----------
    layer : str
        pyretechnics layer name (a key of ``PYRE_LAYERS``).
    arr : numpy.ndarray
        Raw band values (float64 working copy).
    valid : numpy.ndarray of bool
        True where the source pixel is data, False at nodata.
    slope_units : {"degrees", "percent"}
        Resolved slope units; ignored for every layer but ``"slope"``.

    Returns
    -------
    numpy.ndarray
        Converted values, float64. Invalid pixels are left untouched for the
        caller to fill.
    """
    out = arr.astype("float64", copy=True)

    if layer == "slope":
        if slope_units == "degrees":
            # tan(radians(x)), NOT x * 0.01. This is the whole reason this
            # module exists; see the module docstring.
            out = np.where(valid, np.tan(np.radians(out)), out)
        else:
            out = np.where(valid, out / 100.0, out)
        return out

    if layer == "aspect":
        # LANDFIRE writes -1 where the surface is flat and aspect is undefined.
        # Slope is 0 there, so the slope-spread term is 0 regardless of the
        # value; the caller's flat_aspect just has to be a legal azimuth.
        return out

    multiplier = PYRE_LAYERS[layer][1]
    if multiplier != 1 and multiplier != 1.0:
        out = np.where(valid, out * multiplier, out)
    return out


def _check_range(layer, arr, valid):
    """Raise if a converted layer falls outside its physical range."""
    if layer not in _VALID_RANGE or not valid.any():
        return
    lo, hi = _VALID_RANGE[layer]
    vals = arr[valid]
    amin, amax = float(np.min(vals)), float(np.max(vals))
    if amin < lo or amax > hi:
        raise ValueError(
            f"Converted layer {layer!r} spans [{amin:.4g}, {amax:.4g}], outside the "
            f"physical range [{lo:g}, {hi:g}] for {PYRE_LAYERS[layer][2]}. This is "
            "almost always a units error in the source band, not a strange "
            "landscape — check the band description and the PYRE_LAYERS multiplier."
        )


def destack_lcp(lcp_fp, out_dir, *, layers=None, resolution=None, resample="nearest",
                slope_units="auto", nb_fill=91, flat_aspect=0.0,
                overwrite=False, verbose=True):
    """
    De-stack a multi-band LANDFIRE LCP into per-variable rasters for pyretechnics.

    Parameters
    ----------
    lcp_fp : str or Path
        Multi-band landscape GeoTIFF with per-band descriptions (the form
        ``stack_rasters`` and the LFPS client write).
    out_dir : str or Path
        Destination directory. Created if absent. One ``<layer>.tif`` per layer.
    layers : iterable of str, optional
        Subset of ``PYRE_LAYERS`` keys. Defaults to all eight.
    resolution : float, optional
        Target cell size in the LCP's CRS units. ``None`` (default) keeps the
        native grid. Resampling happens **here, once per arm**, so every arm
        lands on one congruent grid — resampling downstream, per realization,
        is how two arms end up a half-pixel apart and the paired difference
        becomes quietly meaningless.
    resample : str
        Resampling method for continuous layers when *resolution* is set
        (default ``"nearest"``). ``fuel_model`` always uses nearest/mode
        regardless — bilinear on fuel codes invents fuel models that do not
        exist.
    slope_units : {"auto", "degrees", "percent"}
        See ``_resolve_slope_units``. ``"auto"`` raises rather than guessing.
    nb_fill : int
        FBFM40 code written where the source is nodata (default ``91`` = NB1).
        pyretechnics has no nodata concept: an unfilled cell is read as a fuel
        model number and is undefined behaviour. Filling with a non-burnable
        code makes the domain edge inert, which is the intended meaning.
    flat_aspect : float
        Azimuth written where LANDFIRE's aspect is ``-1`` (flat). Default ``0.0``.
    overwrite : bool
        Overwrite existing outputs (default False — an existing layer is reused
        and reported, so re-running a campaign setup is cheap).
    verbose : bool
        Print per-layer progress.

    Returns
    -------
    dict
        ``{layer_name: Path}`` for every layer written or reused.

    Raises
    ------
    ValueError
        On a missing band, unresolvable slope units, a converted layer outside
        its physical range, or an FBFM40 code absent from Scott & Burgan 40.
    FileNotFoundError
        If *lcp_fp* does not exist.

    Notes
    -----
    Outputs are float32 except ``fuel_model``, which is int16. pyretechnics
    copies any non-float32 array into its ``SpaceTimeCube`` with a warning, so
    writing float32 here avoids a copy of every static layer in every worker.
    """
    lcp_fp = Path(lcp_fp)
    if not lcp_fp.exists():
        raise FileNotFoundError(f"LCP not found: {lcp_fp}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(PYRE_LAYERS) if layers is None else list(layers)
    unknown = [ly for ly in selected if ly not in PYRE_LAYERS]
    if unknown:
        raise ValueError(
            f"Unknown pyretechnics layer(s) {unknown}. Valid: {sorted(PYRE_LAYERS)}"
        )

    written = {}

    with rasterio.open(lcp_fp) as src:
        available = {_canonical_band_name(d): (i, d)
                     for i, d in enumerate(src.descriptions, start=1)}

        # Resolve every band up front so a missing one fails before any write.
        plan = {}
        for layer in selected:
            canonical = PYRE_LAYERS[layer][0]
            idx, desc = _band_index(src, canonical)
            if idx is None:
                raise ValueError(
                    f"Band {canonical!r} (for pyretechnics layer {layer!r}) not found in "
                    f"{lcp_fp.name}. Band descriptions are {src.descriptions}, which "
                    f"resolve to {sorted(k for k in available if k)}."
                )
            plan[layer] = (idx, desc)

        resolved_slope_units = None
        if "slope" in plan:
            resolved_slope_units = _resolve_slope_units(plan["slope"][1], slope_units)
            if verbose:
                print(f"  slope band {plan['slope'][1]!r} -> {resolved_slope_units}")

        # --- target grid -----------------------------------------------------
        # Anchored on the source's upper-left corner with an exact cell size, so
        # arms de-stacked separately at the same resolution are byte-congruent.
        # Not a decimated read: src.read(out_shape=...) stretches the full source
        # extent onto the requested shape, which yields an effective cell size
        # that is only approximately `resolution` and a transform that quietly
        # disagrees with the data.
        if resolution is None:
            dst_transform, dst_width, dst_height = src.transform, src.width, src.height
        else:
            from rasterio.transform import from_origin
            src_res_x, src_res_y = src.transform.a, -src.transform.e
            dst_width = max(1, int(np.ceil(src.width * src_res_x / resolution)))
            dst_height = max(1, int(np.ceil(src.height * src_res_y / resolution)))
            dst_transform = from_origin(
                src.transform.c, src.transform.f, resolution, resolution
            )

        for layer in selected:
            idx, desc = plan[layer]
            out_fp = out_dir / f"{layer}.tif"
            if out_fp.exists() and not overwrite:
                if verbose:
                    print(f"  {layer:20s} exists, reusing -> {out_fp.name}")
                written[layer] = out_fp
                continue

            categorical = layer in _CATEGORICAL
            method = Resampling.mode if categorical else getattr(Resampling, resample)

            if resolution is None:
                raw = src.read(idx).astype("float64")
            else:
                from rasterio.warp import reproject
                raw = np.empty((dst_height, dst_width), dtype="float64")
                reproject(
                    source=src.read(idx),
                    destination=raw,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=src.crs,
                    src_nodata=src.nodata, dst_nodata=src.nodata,
                    resampling=method,
                )

            # Nodata handling. int LCPs carry -9999; be tolerant of a file that
            # declares no nodata at all.
            nodata = src.nodata
            valid = np.ones(raw.shape, dtype=bool) if nodata is None else (raw != nodata)

            conv = _convert(layer, raw, valid, resolved_slope_units)

            if layer == "fuel_model":
                conv = np.where(valid, conv, float(nb_fill))
                codes = np.rint(conv).astype("int32")
                bad = sorted(set(np.unique(codes).tolist()) - _VALID_FBFM40)
                if bad:
                    raise ValueError(
                        f"fuel_model contains code(s) {bad[:12]} absent from Scott & "
                        "Burgan 40. Valid codes are 0/91-93/98/99 and 101-204 within "
                        "the GR/GS/SH/TU/TL/SB blocks; 94-97 are unassigned. A code "
                        "outside this set is read by pyretechnics as an index into "
                        "its fuel-model table and will not behave as you expect."
                    )
                data, dtype, out_nodata = codes.astype("int16"), "int16", None
            else:
                if layer == "aspect":
                    conv = np.where(conv < 0, float(flat_aspect), conv)
                    conv = np.mod(conv, 360.0)
                # Every non-fuel layer is filled at nodata too: pyretechnics
                # reads the whole grid, and a NaN there propagates silently.
                # 0.0 is inert for slope/canopy; elevation is unused by the
                # surface-fire equations (only slope and aspect are).
                conv = np.where(valid, conv, 0.0)
                _check_range(layer, conv, valid)
                data, dtype, out_nodata = conv.astype("float32"), "float32", None

            profile = src.profile.copy()
            profile.update(
                driver="GTiff", count=1, dtype=dtype, nodata=out_nodata,
                transform=dst_transform, width=dst_width, height=dst_height,
                **raster_write_kwargs(),
            )
            with rasterio.open(out_fp, "w", **profile) as dst:
                dst.write(data, 1)
                dst.set_band_description(1, layer)
                dst.update_tags(
                    pyre_layer=layer,
                    source_band=str(desc),
                    source_lcp=lcp_fp.name,
                    units=PYRE_LAYERS[layer][2],
                    slope_units=str(resolved_slope_units),
                )

            if verbose:
                finite = data[np.isfinite(data)]
                print(f"  {layer:20s} <- band {idx} {str(desc):24s} "
                      f"[{finite.min():.4g}, {finite.max():.4g}] {PYRE_LAYERS[layer][2]}")
            written[layer] = out_fp

    if verbose:
        print(f"De-stacked {len(written)} layer(s) -> {out_dir}")
    return written


# ---------------------------------------------------------------------------
# Cell2Fire — kept as stubs on purpose (see module docstring)
# ---------------------------------------------------------------------------

def lcp_to_cell2fire(*args, **kwargs):
    raise NotImplementedError(
        "LCP -> Cell2Fire conversion is not yet implemented. "
        "See fb_tools/spread/convert.py for the planned API."
    )


def build_cell2fire_weather(*args, **kwargs):
    raise NotImplementedError(
        "Cell2Fire weather CSV builder is not yet implemented. "
        "See fb_tools/spread/convert.py for the planned API."
    )
