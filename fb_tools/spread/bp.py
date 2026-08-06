"""
Burn probability analysis — delta burn probability and treatment effect summaries.

Four public entry points
------------------------
``delta_burn_probability``
    Pixel-wise difference between baseline and treated burn probability rasters.
    Positive values indicate the treatment reduced burn probability.

``aggregate_ignition_bp``
    Collapse per-ignition rasters into an ensemble surface and a treatment
    delta, optionally weighted by the Horvitz–Thompson design weights *w_i*
    written to ``runs.csv`` by
    :func:`~fb_tools.models.container.prepare_fspro_experiment`.

``summarize_bp_treatments``
    Zonal statistics of burn probability change per polygon, computed on the
    **already-aligned** delta raster.

``downwind_treatment_effect``
    Summarize delta burn probability in the downwind sector of a treatment
    polygon.

Two constraints run through all of them
---------------------------------------
**The ignition footprint must be masked out.**  The FSPro ``IgnitionFile`` is a
starting fire *perimeter*, so BP ≈ 1.0 everywhere inside it by construction in
every arm.  Including it drags every delta statistic toward zero by an amount
that depends only on footprint size.  Pass ``ignition=`` to any of these
functions.

**Grids must be congruent.**  ``xr.align(join="left")`` will happily paper over a
transform or shape mismatch and return a plausible, entirely wrong delta
surface.  All alignment here is guarded by :func:`_assert_same_grid` unless the
caller explicitly opts out with ``strict_grid=False``.

Every reported delta should be read against the P0.1 noise floor — see
:mod:`fb_tools.spread.noise`.  ``SPOTTING_SEED`` does **not** give common random
numbers, so baseline/treated pairing is statistical, never exact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr


#: int16 sentinel written for masked pixels of a scaled delta raster.  Chosen
#: so it cannot collide with a real value: delta ∈ [-1, 1] at ``scale=100``
#: spans [-100, 100].
DELTA_NODATA = -32768

#: FlamMap wind-direction sentinels — ``-1`` uphill, ``-2`` downhill.  These are
#: slope-driven flags, not azimuths, and must never be treated as bearings.
_WIND_SENTINELS = (-1, -2)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _open_bp(src):
    """Open *src* as a float32 DataArray if it is a path, else return as-is."""
    if isinstance(src, (str, Path)):
        da = rxr.open_rasterio(Path(src), masked=True).squeeze("band", drop=True)
        return da.astype("float32")
    return src.astype("float32")


def _find_bp_tif(directory, model="mtt"):
    """
    Locate a burn probability GeoTIFF in *directory*.

    Tries common output filenames by model type, then falls back to any
    ``.tif`` whose name contains ``burn`` or ``bp``.

    Parameters
    ----------
    directory : Path
    model : str
        ``"mtt"``, ``"fspro"``, or ``"cell2fire"``.

    Returns
    -------
    Path or None
    """
    candidates_by_model = {
        "mtt":        ["BurnProbability.tif", "burn_probability.tif"],
        "fspro":      ["BurnProb.tif", "BurnProbability.tif"],
        "cell2fire":  ["BurnProb.tif", "BurnProbability.tif"],
    }
    directory = Path(directory)
    for name in candidates_by_model.get(model, []):
        p = directory / name
        if p.exists():
            return p
    # broad fallback
    for p in sorted(directory.glob("*.tif")):
        if any(kw in p.stem.lower() for kw in ("burn", "bp")):
            return p
    return None


def _assert_same_grid(ref, da, label="raster", atol=1e-4):
    """
    Raise unless *da* sits on exactly the same grid as *ref*.

    Checks shape, affine transform, and CRS.  This guard exists because
    ``xr.align(join="left")`` silently fills non-overlapping coordinates with
    NaN rather than failing, which turns a grid mismatch into a plausible but
    wrong delta surface.

    Parameters
    ----------
    ref, da : xarray.DataArray
        Rasters to compare.
    label : str
        Name used in the error message.
    atol : float
        Absolute tolerance on the six affine coefficients, in CRS units.

    Raises
    ------
    ValueError
        If shape, transform, or CRS differ.
    """
    if tuple(ref.shape) != tuple(da.shape):
        raise ValueError(
            f"{label} shape {tuple(da.shape)} != reference shape "
            f"{tuple(ref.shape)}. Arm landscapes must share one grid; "
            f"regenerate them on a congruent grid or pass strict_grid=False "
            f"to fall back to lenient alignment."
        )

    t_ref = np.asarray(ref.rio.transform())[:6]
    t_da  = np.asarray(da.rio.transform())[:6]
    if not np.allclose(t_ref, t_da, rtol=0.0, atol=atol):
        raise ValueError(
            f"{label} transform {tuple(np.round(t_da, 6))} != reference "
            f"{tuple(np.round(t_ref, 6))}. A sub-pixel origin shift silently "
            f"mispairs every cell."
        )

    if ref.rio.crs is not None and da.rio.crs is not None:
        if ref.rio.crs != da.rio.crs:
            raise ValueError(
                f"{label} CRS {da.rio.crs.to_string()} != reference "
                f"{ref.rio.crs.to_string()}."
            )
    return True


def _footprint_mask(ignition, da):
    """
    Boolean DataArray aligned to *da*, ``True`` inside the ignition footprint.

    Parameters
    ----------
    ignition : str, Path, ndarray, GeoDataFrame, or shapely geometry
        Anything :func:`~fb_tools.spread.fspro_outputs._ignition_mask` accepts —
        typically the run's ``_Ignitions.asc`` or the ignition shapefile.
    da : xarray.DataArray
        Reference grid.

    Returns
    -------
    xarray.DataArray of bool
    """
    from .fspro_outputs import _ignition_mask

    arr = _ignition_mask(ignition, tuple(da.shape), da.rio.transform(), da.rio.crs)
    return xr.DataArray(arr, coords=da.coords, dims=da.dims)


def _apply_ignition_mask(da, ignition, verbose=True):
    """Return *da* with the ignition footprint set to NaN (no-op if ``None``)."""
    if ignition is None:
        return da
    mask = _footprint_mask(ignition, da)
    n = int(mask.values.sum())
    if verbose:
        print(f"  Masked {n:,} ignition-footprint pixel(s) from delta statistics.")
    return da.where(~mask)


def _resolve_weights(weights, n, keys=None):
    """
    Normalize *weights* to a length-*n* float array summing to 1.

    Parameters
    ----------
    weights : None, sequence, dict, or pandas.Series
        ``None`` gives equal weights.  A dict/Series is looked up by *keys*
        when given, otherwise by positional index.
    n : int
        Number of ignitions.
    keys : sequence, optional
        Ignition identifiers used to look up a dict/Series.

    Returns
    -------
    numpy.ndarray, shape (n,)

    Raises
    ------
    ValueError
        If the length is wrong, or any weight is negative, non-finite, or the
        weights sum to zero.
    """
    if weights is None:
        return np.full(n, 1.0 / n, dtype="float64")

    if isinstance(weights, (dict, pd.Series)):
        lookup = dict(weights)
        idx = list(keys) if keys is not None else list(range(n))
        missing = [k for k in idx if k not in lookup]
        if missing:
            raise ValueError(
                f"weights is missing entries for ignition(s) {missing[:5]}"
                f"{' …' if len(missing) > 5 else ''}."
            )
        w = np.asarray([float(lookup[k]) for k in idx], dtype="float64")
    else:
        w = np.asarray(weights, dtype="float64")

    if w.shape != (n,):
        raise ValueError(f"weights has length {w.shape} but {n} ignitions were given.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contains NaN or infinite values.")
    if np.any(w < 0):
        raise ValueError("weights contains negative values.")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights sum to zero.")
    return w / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def delta_burn_probability(
    baseline_bp,
    treatment_bp,
    out_path=None,
    scale=100,
    ignition=None,
    strict_grid=True,
    verbose=True,
):
    """
    Compute the delta burn probability raster (baseline minus treatment).

    Positive values indicate the treatment reduced burn probability at that
    pixel.

    Parameters
    ----------
    baseline_bp : str, Path, or xarray.DataArray
        Baseline burn probability raster (float, 0–1 range).  Accepts MTT,
        FSPro, or Cell2Fire GeoTIFF output.
    treatment_bp : str, Path, or xarray.DataArray
        Treated landscape burn probability raster (same units as *baseline_bp*).
    out_path : str or Path, optional
        If provided, write the delta raster as an int16 GeoTIFF.  Values are
        stored as ``delta × scale``; masked pixels are written as
        :data:`DELTA_NODATA` and tagged as the file's nodata value.
    scale : int
        Scale factor for the int16 output (default ``100``).  Set to ``1``
        if inputs are already in percent (0–100) units.
    ignition : str, Path, ndarray, GeoDataFrame, or shapely geometry, optional
        Ignition footprint to mask out.  BP ≈ 1.0 inside the starting
        perimeter in every arm, so leaving it in biases every statistic toward
        zero.  Strongly recommended.
    strict_grid : bool
        Require both rasters on an identical grid (default ``True``).  Set
        ``False`` to restore lenient ``xr.align`` behaviour for minor extent
        differences — at the risk of silently mispairing cells.
    verbose : bool
        Print the masked-pixel count and output path.

    Returns
    -------
    xarray.DataArray
        Delta burn probability as float32, NaN where masked.  Positive =
        treatment reduced BP.

    Raises
    ------
    ValueError
        If *strict_grid* and the grids differ, or if scaled values would
        overflow int16 on write.

    Notes
    -----
    Because ``SPOTTING_SEED`` does not deliver common random numbers (P0.1),
    this difference carries Monte Carlo noise even where the treatment did
    nothing.  Compare it against
    :func:`fb_tools.spread.noise.bp_noise_floor` before reporting.
    """
    bl = _open_bp(baseline_bp)
    tr = _open_bp(treatment_bp)

    if strict_grid:
        _assert_same_grid(bl, tr, label="treatment_bp")

    bl, tr = xr.align(bl, tr, join="left")

    delta = (bl - tr).astype("float32")
    delta.attrs = dict(bl.attrs)
    if bl.rio.crs is not None:
        delta = delta.rio.write_crs(bl.rio.crs)

    delta = _apply_ignition_mask(delta, ignition, verbose=verbose)
    delta = delta.rio.write_nodata(np.nan, encoded=False)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        scaled = delta * scale
        finite = scaled.values[np.isfinite(scaled.values)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if lo <= DELTA_NODATA or hi > 32767:
                raise ValueError(
                    f"Scaled delta range [{lo:.1f}, {hi:.1f}] does not fit in "
                    f"int16 alongside the nodata sentinel {DELTA_NODATA}. "
                    f"Lower `scale` (currently {scale})."
                )

        out_int = (scaled.round()
                        .fillna(DELTA_NODATA)
                        .astype("int16")
                        .rio.write_nodata(DELTA_NODATA, encoded=False))
        out_int.rio.to_raster(out_path, dtype="int16", nodata=DELTA_NODATA,
                              compress="deflate", predictor=2,
                              tiled=True, blockxsize=256, blockysize=256)
        if verbose:
            print(f"Delta burn probability written to {out_path} "
                  f"(scale={scale}, nodata={DELTA_NODATA})")

    return delta


def _ensemble_stack(bp_list, ref=None, strict_grid=True, label="raster"):
    """Align a list of rasters to a common grid and stack along ``ignition``.

    When *ref* is given, every raster is aligned to it — so that a baseline and
    a treated stack share one grid and can be differenced per ignition.
    Otherwise the first raster in the list is used as the reference.

    Returns a single DataArray with a leading ``ignition`` dimension.
    """
    das = [_open_bp(src) for src in bp_list]
    if ref is None:
        ref = das[0]
    aligned = []
    for i, da in enumerate(das):
        if strict_grid:
            _assert_same_grid(ref, da, label=f"{label}[{i}]")
        _, da_a = xr.align(ref, da, join="left")
        aligned.append(da_a)
    return xr.concat(aligned, dim=xr.Variable("ignition", range(len(aligned))))


def aggregate_ignition_bp(
    baseline_bps,
    treated_bps,
    out_dir=None,
    out_prefix="ensemble",
    paired=True,
    weights=None,
    ignition_ids=None,
    ignitions=None,
    burned_threshold=0.0,
    strict_grid=True,
    verbose=True,
):
    """
    Aggregate per-ignition rasters into an ensemble surface + treatment delta.

    Each ignition was run as a separate FSPro fire (see
    :func:`~fb_tools.models.container.prepare_fspro_experiment`).  This
    collapses the per-ignition rasters into a single ensemble surface.

    Weighting
    ---------
    ``prepare_fspro_experiment`` draws design fires by stratified sampling from
    an ignition-density surface and records a Horvitz–Thompson weight
    ``w_i = stratum_mass / draws_in_stratum`` per ignition in ``runs.csv``.
    Pass those weights here so the ensemble estimates ``Σ_i w_i · X_i`` — the
    Ager source-weighted quantity — rather than an unweighted mean over a
    deliberately non-uniform sample.  Omitting *weights* weights every design
    fire equally, which is only correct if the draw was uniform.

    The treatment delta (``delta_mean``) can be formed two ways:

    ``paired=True`` (default, recommended)
        Compute the per-ignition difference ``baseline_i - treated_i``, then
        take the weighted average across ignitions.  This is the robust
        estimator for grids defined only where the fire burned (flame length,
        arrival time): it never differences means taken over *different*
        subsets of ignitions.  For burn probability (defined at every pixel)
        it is identical to ``paired=False``.

    ``paired=False``
        Average each side across ignitions first, then difference the means.

    Parameters
    ----------
    baseline_bps : list of str, Path, or xarray.DataArray
        Per-ignition baseline rasters.
    treated_bps : list of str, Path, or xarray.DataArray
        Per-ignition treated rasters.  Must be the same length as
        *baseline_bps* and in matching ignition order.
    out_dir : str or Path, optional
        If provided, writes ``{out_prefix}_baseline.tif``,
        ``{out_prefix}_treated.tif``, ``{out_prefix}_delta.tif``,
        ``{out_prefix}_delta_std.tif``, ``{out_prefix}_n_ignitions.tif`` and
        ``{out_prefix}_n_burned.tif`` here.
    out_prefix : str
        Filename prefix for written GeoTIFFs.  Default ``"ensemble"``.
    paired : bool
        Use the per-ignition paired delta (default ``True``).
    weights : sequence, dict, or pandas.Series, optional
        Per-ignition weights *w_i*, normalized internally to sum to 1.  A dict
        or Series is keyed by *ignition_ids* when given, else by position.
    ignition_ids : sequence, optional
        Identifiers matching *baseline_bps* order, used to look up *weights*
        and to report which ignitions were dropped.
    ignitions : sequence, optional
        Per-ignition footprints (paths, arrays, or geometries), same order and
        length as *baseline_bps*.  Each is masked out of that ignition's delta
        before aggregation.  Every design fire has its own footprint, so this
        is a sequence, not a single mask.
    burned_threshold : float
        A pixel counts as burned for ``n_burned`` when either arm exceeds this
        (default ``0.0``).
    strict_grid : bool
        Require every raster on an identical grid (default ``True``).
    verbose : bool
        Print progress and masked-pixel counts.

    Returns
    -------
    dict
        ``"baseline_mean"`` / ``"treated_mean"`` : xarray.DataArray
            Weighted mean across ignitions for each landscape.
        ``"baseline_max"`` / ``"treated_max"`` : xarray.DataArray
            Per-pixel maximum across ignitions (worst-case ignition).
        ``"delta_mean"`` : xarray.DataArray
            Treatment delta (see *paired*).  Positive = ``baseline > treated``.
        ``"delta_std"`` : xarray.DataArray or None
            Weighted across-ignition standard deviation of the per-ignition
            delta — an uncertainty surface (``paired=True`` only).
        ``"n_ignitions"`` : xarray.DataArray of int16
            Per-pixel count of ignitions contributing **valid (non-NaN) data**.
            For flame-length and arrival-time grids — which are NaN outside the
            burned area — this is the co-burn count.  For **burn probability it
            is not**: unburned interior pixels are ``0.0``, not nodata, so this
            equals the ignition count almost everywhere.  Use ``n_burned`` to
            mask BP surfaces.
        ``"n_burned"`` : xarray.DataArray of int16
            Per-pixel count of ignitions where **either** arm exceeded
            *burned_threshold*.  This is the meaningful support count for burn
            probability.
        ``"weights"`` : numpy.ndarray
            The normalized weights actually applied, after dropping ignitions
            with missing output.

    Raises
    ------
    ValueError
        If the two lists differ in length, are empty, no valid pairs remain,
        the weights are invalid, or (when *strict_grid*) grids differ.
    """
    if len(baseline_bps) != len(treated_bps):
        raise ValueError(
            f"baseline_bps ({len(baseline_bps)}) and treated_bps "
            f"({len(treated_bps)}) must have the same length."
        )
    if not baseline_bps:
        raise ValueError("No rasters provided.")

    n_in = len(baseline_bps)
    if ignition_ids is None:
        ignition_ids = list(range(n_in))
    elif len(ignition_ids) != n_in:
        raise ValueError(
            f"ignition_ids has length {len(ignition_ids)} but {n_in} ignitions "
            f"were given."
        )
    if ignitions is not None and len(ignitions) != n_in:
        raise ValueError(
            f"ignitions has length {len(ignitions)} but {n_in} ignitions were "
            f"given. Pass one footprint per design fire."
        )

    # Resolve weights against the FULL input list, then subset alongside it, so
    # a dropped ignition cannot shift the weight-to-ignition correspondence.
    w_full = _resolve_weights(weights, n_in, keys=ignition_ids)

    keep = [
        i for i in range(n_in)
        if baseline_bps[i] is not None and treated_bps[i] is not None
    ]
    n_dropped = n_in - len(keep)
    if n_dropped:
        dropped = [ignition_ids[i] for i in range(n_in) if i not in set(keep)]
        print(f"Warning: {n_dropped} ignition(s) had missing outputs and were "
              f"skipped: {dropped}")
    if not keep:
        raise ValueError("No valid ignition pairs after filtering None outputs.")

    baseline_bps = [baseline_bps[i] for i in keep]
    treated_bps  = [treated_bps[i] for i in keep]
    kept_ids     = [ignition_ids[i] for i in keep]
    kept_ign     = [ignitions[i] for i in keep] if ignitions is not None else None
    # Re-normalize after dropping so the surviving weights still sum to 1.
    w = w_full[keep]
    w = w / w.sum()

    # Align BOTH sides to one reference grid so per-ignition pairs can be
    # differenced cell-for-cell.
    ref = _open_bp(baseline_bps[0])
    bl_stack = _ensemble_stack(baseline_bps, ref=ref, strict_grid=strict_grid,
                               label="baseline")
    tr_stack = _ensemble_stack(treated_bps, ref=ref, strict_grid=strict_grid,
                               label="treated")

    w_da = xr.DataArray(w, dims="ignition",
                        coords={"ignition": bl_stack.coords["ignition"]})

    def _wmean(stack):
        valid = stack.notnull()
        wsum = (w_da * valid).sum(dim="ignition")
        num  = (stack.fillna(0.0) * w_da).sum(dim="ignition")
        return (num / wsum).where(wsum > 0).astype("float32")

    bl_mean = _wmean(bl_stack)
    tr_mean = _wmean(tr_stack)
    bl_max  = bl_stack.max(dim="ignition").astype("float32")
    tr_max  = tr_stack.max(dim="ignition").astype("float32")

    # Per-ignition difference.  NaN wherever either run lacks data at the pixel.
    per_ign_delta = bl_stack - tr_stack

    # Each design fire's own footprint is masked from its own delta before the
    # ensemble is formed — BP ≈ 1 inside the starting perimeter in every arm.
    if kept_ign is not None:
        masks = []
        for i, ign in enumerate(kept_ign):
            m = _footprint_mask(ign, ref) if ign is not None \
                else xr.zeros_like(ref, dtype=bool)
            masks.append(m)
        ign_stack = xr.concat(
            masks, dim=xr.Variable("ignition", list(range(len(masks))))
        )
        ign_stack = ign_stack.assign_coords(ignition=per_ign_delta.coords["ignition"])
        n_masked = int(ign_stack.values.sum())
        per_ign_delta = per_ign_delta.where(~ign_stack)
        if verbose:
            print(f"  Masked {n_masked:,} ignition-footprint pixel-ignitions "
                  f"from the ensemble delta.")

    valid       = per_ign_delta.notnull()
    n_ignitions = valid.sum(dim="ignition").astype("int16")

    burned   = ((bl_stack > burned_threshold) | (tr_stack > burned_threshold))
    n_burned = (burned & valid).sum(dim="ignition").astype("int16")

    if paired:
        wsum = (w_da * valid).sum(dim="ignition")
        delta_mean = ((per_ign_delta.fillna(0.0) * w_da).sum(dim="ignition") / wsum)
        delta_mean = delta_mean.where(wsum > 0).astype("float32")

        # Weighted population variance about the weighted mean.
        dev = (per_ign_delta - delta_mean) ** 2
        var = ((dev.fillna(0.0) * w_da).sum(dim="ignition") / wsum)
        delta_std = np.sqrt(var.where(wsum > 0)).astype("float32")
    else:
        delta_mean = (bl_mean - tr_mean).astype("float32")
        delta_std  = None

    # Preserve CRS on derived surfaces (arithmetic can drop the grid mapping).
    crs = ref.rio.crs
    if crs is not None:
        bl_mean     = bl_mean.rio.write_crs(crs)
        tr_mean     = tr_mean.rio.write_crs(crs)
        bl_max      = bl_max.rio.write_crs(crs)
        tr_max      = tr_max.rio.write_crs(crs)
        delta_mean  = delta_mean.rio.write_crs(crs)
        n_ignitions = n_ignitions.rio.write_crs(crs)
        n_burned    = n_burned.rio.write_crs(crs)
        if delta_std is not None:
            delta_std = delta_std.rio.write_crs(crs)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Float means/deltas — plain DEFLATE (no predictor: the floating-point
        # predictor tends to enlarge this kind of output).
        _fkw = dict(compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        bl_mean.rio.to_raster(out_dir / f"{out_prefix}_baseline.tif", **_fkw)
        tr_mean.rio.to_raster(out_dir / f"{out_prefix}_treated.tif", **_fkw)
        delta_mean.rio.to_raster(out_dir / f"{out_prefix}_delta.tif", **_fkw)
        n_ignitions.rio.to_raster(out_dir / f"{out_prefix}_n_ignitions.tif", **_fkw)
        n_burned.rio.to_raster(out_dir / f"{out_prefix}_n_burned.tif", **_fkw)
        if delta_std is not None:
            delta_std.rio.to_raster(out_dir / f"{out_prefix}_delta_std.tif", **_fkw)
        if verbose:
            print(f"Ensemble ({len(baseline_bps)} ignitions, "
                  f"{'paired' if paired else 'mean-diff'} delta, "
                  f"{'weighted' if weights is not None else 'equal-weight'}) "
                  f"written to {out_dir}")

    return {
        "baseline_mean": bl_mean,
        "treated_mean":  tr_mean,
        "baseline_max":  bl_max,
        "treated_max":   tr_max,
        "delta_mean":    delta_mean,
        "delta_std":     delta_std,
        "n_ignitions":   n_ignitions,
        "n_burned":      n_burned,
        "weights":       w,
        "ignition_ids":  kept_ids,
    }


def summarize_bp_treatments(
    zones_gdf,
    delta_bp=None,
    id_col="TRT_ID",
    type_col=None,
    baseline_bp=None,
    treatment_bp=None,
    ignition=None,
    min_pixels=30,
    support=None,
    out_dir=None,
    csv_name="bp_change.csv",
    strict_grid=True,
    verbose=True,
):
    """
    Compute zonal burn probability change statistics per polygon.

    Statistics are taken on the **aligned** delta raster, so baseline and
    treated values are paired cell-for-cell within each zone.

    .. note::
       The pre-Phase-3 signature took ``baseline_bp_dir`` / ``treated_bp_dirs``
       and differenced two independently clipped flat arrays positionally.  It
       also called ``geom_to_raster_crs`` with three arguments against a
       two-argument signature, so it raised ``TypeError`` on the first polygon
       and never produced output.  This is a rewrite, not a patched version;
       calls using the old keywords will fail loudly.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Zones to summarize — treatment polygons, watersheds, PODs, firesheds.
        Reprojected internally to the raster CRS.
    delta_bp : str, Path, or xarray.DataArray, optional
        Pre-computed delta raster (positive = treatment reduced BP).  When
        omitted, both *baseline_bp* and *treatment_bp* must be given and the
        delta is computed here via :func:`delta_burn_probability`.
    id_col : str
        Zone identifier column.  Default ``"TRT_ID"``.
    type_col : str, optional
        Optional grouping column (e.g. ``"TRT_TYPE"``) carried into the output.
    baseline_bp, treatment_bp : str, Path, or xarray.DataArray, optional
        Source rasters.  Required when *delta_bp* is omitted; when supplied
        alongside it, their zonal means are reported too.
    ignition : str, Path, ndarray, GeoDataFrame, or shapely geometry, optional
        Ignition footprint to mask out before computing statistics.
    min_pixels : int
        Zones with fewer valid pixels than this are flagged ``reliable=False``
        (default ``30``).  Statistics are still reported.
    support : str, Path, or xarray.DataArray, optional
        A support-count raster (typically ``n_burned`` from
        :func:`aggregate_ignition_bp`).  When given, pixels with support below
        *min_pixels* are excluded and ``n_low_support`` counts them.
    out_dir : str or Path, optional
        If provided, write *csv_name* here.
    csv_name : str
        Output CSV filename.  Default ``"bp_change.csv"``.
    strict_grid : bool
        Require congruent grids when computing the delta (default ``True``).
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        One row per zone with columns: *id_col*, optionally *type_col*,
        ``n_pixels``, ``area_ha``, ``area_ac``, ``dBP_mean``, ``dBP_median``,
        ``dBP_p95``, ``dBP_max``, ``dBP_min``, ``dBP_sum_ha`` (the
        area-integrated ``Σ ΔBP × cell_area`` in hectares — the ΔTF_ij
        estimator), ``pct_improved``, ``reliable``, and ``BP_bl_mean`` /
        ``BP_tr_mean`` when source rasters were supplied.

    Raises
    ------
    ValueError
        If neither *delta_bp* nor both source rasters are given, or if
        *id_col* is missing from *zones_gdf*.

    Notes
    -----
    Read ``dBP_mean`` against :func:`fb_tools.spread.noise.bp_noise_floor` and
    ``dBP_sum_ha`` against :func:`fb_tools.spread.noise.area_noise_floor`.  The
    area-integrated column is far more robust to Monte Carlo noise (P0.1).
    """
    from ..utils.geo import geom_to_raster_crs

    if id_col not in zones_gdf.columns:
        raise ValueError(
            f"id_col '{id_col}' not found in zones_gdf. "
            f"Available: {list(zones_gdf.columns)}"
        )
    if type_col is not None and type_col not in zones_gdf.columns:
        raise ValueError(
            f"type_col '{type_col}' not found in zones_gdf. "
            f"Available: {list(zones_gdf.columns)}"
        )

    if delta_bp is None:
        if baseline_bp is None or treatment_bp is None:
            raise ValueError(
                "Provide either delta_bp, or both baseline_bp and treatment_bp."
            )
        delta = delta_burn_probability(
            baseline_bp, treatment_bp, ignition=ignition,
            strict_grid=strict_grid, verbose=verbose,
        )
    else:
        delta = _open_bp(delta_bp)
        delta = _apply_ignition_mask(delta, ignition, verbose=verbose)

    bl_da = _open_bp(baseline_bp) if baseline_bp is not None else None
    tr_da = _open_bp(treatment_bp) if treatment_bp is not None else None

    sup_da = None
    if support is not None:
        sup_da = _open_bp(support)
        if strict_grid:
            _assert_same_grid(delta, sup_da, label="support")

    cell_res = abs(float(delta.rio.resolution()[0]))
    cell_ha  = (cell_res ** 2) / 10_000.0
    cell_ac  = (cell_res ** 2) / 4046.86

    # Reproject all zones once, not per polygon.
    zones = geom_to_raster_crs(zones_gdf, delta)

    rows = []
    for _, zone in zones.iterrows():
        geom = zone.geometry
        rec = {id_col: zone[id_col]}
        if type_col is not None:
            rec[type_col] = zone[type_col]

        try:
            d_clip = delta.rio.clip([geom], all_touched=True, drop=True)
        except Exception:
            # No overlap between the zone and the raster.
            d_clip = None

        if d_clip is None:
            vals = np.array([], dtype="float32")
            keep = None
        else:
            arr = d_clip.values.ravel()
            keep = np.isfinite(arr)
            if sup_da is not None:
                s_clip = sup_da.rio.clip([geom], all_touched=True, drop=True)
                s_arr = s_clip.values.ravel()
                keep = keep & (np.nan_to_num(s_arr, nan=0.0) >= min_pixels)
            vals = arr[keep]

        n = int(vals.size)
        rec.update({
            "n_pixels":    n,
            "area_ha":     round(n * cell_ha, 3),
            "area_ac":     round(n * cell_ac, 2),
            "dBP_mean":    float(np.mean(vals))   if n else np.nan,
            "dBP_median":  float(np.median(vals)) if n else np.nan,
            "dBP_p95":     float(np.percentile(vals, 95)) if n else np.nan,
            "dBP_max":     float(np.max(vals))    if n else np.nan,
            "dBP_min":     float(np.min(vals))    if n else np.nan,
            "dBP_sum_ha":  float(np.sum(vals) * cell_ha) if n else np.nan,
            "pct_improved": float(np.mean(vals > 0) * 100.0) if n else np.nan,
            "reliable":    bool(n >= min_pixels),
        })

        for label, src in (("BP_bl_mean", bl_da), ("BP_tr_mean", tr_da)):
            if src is None:
                continue
            try:
                c = src.rio.clip([geom], all_touched=True, drop=True).values.ravel()
                # Restrict to exactly the pixels the delta used.
                c = c[keep] if keep is not None and c.shape == keep.shape else \
                    c[np.isfinite(c)]
                rec[label] = float(np.mean(c)) if c.size else np.nan
            except Exception:
                rec[label] = np.nan

        rows.append(rec)

    df = pd.DataFrame(rows)

    if verbose:
        n_unreliable = int((~df["reliable"]).sum()) if len(df) else 0
        if n_unreliable:
            print(f"  {n_unreliable} of {len(df)} zone(s) below min_pixels="
                  f"{min_pixels} — flagged reliable=False.")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / csv_name
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"BP change summary written to {csv_path}")

    return df


def downwind_treatment_effect(
    treatment_polygon,
    delta_bp,
    wind_direction=None,
    scenario_row=None,
    src_crs=None,
    buffer_km=10.0,
    sector_degrees=45.0,
    ignition=None,
    verbose=True,
):
    """
    Summarize the delta burn probability in the downwind sector of a treatment.

    Constructs a downwind sector from the treatment centroid and computes
    zonal statistics on *delta_bp* within it.

    Parameters
    ----------
    treatment_polygon : shapely geometry, GeoDataFrame, GeoSeries, or row
        Treatment boundary.  The CRS is resolved in this order: the object's
        own ``.crs`` (GeoDataFrame/GeoSeries), then *src_crs*, then the raster
        CRS is assumed.
    delta_bp : xarray.DataArray, str, or Path
        Delta burn probability raster (positive = treatment reduced BP).
    wind_direction : float, optional
        Prevailing wind azimuth in degrees from north (0–360), meteorological
        convention (the direction the wind blows *from*).  Takes precedence
        over *scenario_row*.
    scenario_row : pandas.Series, optional
        Scenario row containing a ``WIND_DIRECTION`` column used when
        *wind_direction* is ``None``.
    src_crs : pyproj.CRS, rasterio CRS, str, or int, optional
        CRS of *treatment_polygon* when it is a bare shapely geometry or a
        plain pandas row.  Without it a geometry in a different CRS lands
        outside the raster and every statistic returns NaN.
    buffer_km : float
        Downwind sector radius in kilometres (default ``10.0``).
    sector_degrees : float
        Full angular width of the sector in degrees (default ``45.0``,
        centred on the downwind bearing).
    ignition : str, Path, ndarray, GeoDataFrame, or shapely geometry, optional
        Ignition footprint to mask out before computing statistics.
    verbose : bool
        Print a warning when the CRS had to be assumed.

    Returns
    -------
    dict
        ``{"mean_delta_bp", "median_delta_bp", "max_delta_bp",
           "sum_delta_bp_ha", "n_pixels", "area_improved_ha",
           "pct_area_improved", "wind_direction", "downwind_azimuth"}``.
        ``area_improved_ha`` counts pixels where delta BP > 0.

    Raises
    ------
    ValueError
        If neither *wind_direction* nor *scenario_row* is provided, if the
        resolved direction is a FlamMap slope sentinel (``-1`` uphill, ``-2``
        downhill) rather than an azimuth, or if it falls outside 0–360.
    """
    import math
    from shapely.geometry import Polygon
    import geopandas as gpd

    # ── Resolve wind direction ───────────────────────────────────────────────
    if wind_direction is not None:
        wd = float(wind_direction)
    elif scenario_row is not None:
        wd = float(scenario_row["WIND_DIRECTION"])
    else:
        raise ValueError(
            "Provide either wind_direction (float) or scenario_row (Series "
            "with WIND_DIRECTION column)."
        )

    if float(wd).is_integer() and int(wd) in _WIND_SENTINELS:
        kind = "uphill" if int(wd) == -1 else "downhill"
        raise ValueError(
            f"wind_direction={int(wd)} is a FlamMap slope sentinel ({kind}), "
            f"not an azimuth. A downwind sector is undefined for slope-driven "
            f"winds — pass an explicit bearing, e.g. the pyrome's dominant "
            f"wind direction."
        )
    if not (0.0 <= wd <= 360.0):
        raise ValueError(
            f"wind_direction={wd} is outside 0–360 degrees."
        )

    delta = _open_bp(delta_bp)
    delta = _apply_ignition_mask(delta, ignition, verbose=verbose)

    # ── Resolve geometry and reproject it to the raster CRS ──────────────────
    if hasattr(treatment_polygon, "crs") and treatment_polygon.crs is not None:
        # GeoDataFrame or GeoSeries — it carries its own CRS.
        gdf = (treatment_polygon if isinstance(treatment_polygon, gpd.GeoDataFrame)
               else gpd.GeoDataFrame(geometry=gpd.GeoSeries(treatment_polygon),
                                     crs=treatment_polygon.crs))
        geom_crs = gdf.crs
        geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") \
            else gdf.geometry.unary_union
    else:
        geom = getattr(treatment_polygon, "geometry", treatment_polygon)
        geom_crs = src_crs
        if geom_crs is None and verbose:
            print("  Warning: no CRS available for treatment_polygon "
                  "(pass src_crs=). Assuming it is already in the raster CRS.")

    if geom_crs is not None:
        from ..utils.geo import geom_to_raster_crs
        one = gpd.GeoDataFrame(geometry=gpd.GeoSeries([geom], crs=geom_crs))
        geom = geom_to_raster_crs(one, delta).geometry.iloc[0]

    centroid = geom.centroid
    cx, cy = centroid.x, centroid.y
    buf_m = buffer_km * 1000.0
    half = sector_degrees / 2.0

    # Wind direction (met convention): wind FROM this direction blows TOWARD
    # the opposite bearing.  "Downwind" = the direction wind is blowing TO.
    downwind_azimuth = (wd + 180.0) % 360.0

    def _az_to_rad(az):
        """Azimuth (CW from north) to math angle (CCW from east)."""
        return math.radians(90.0 - az)

    left_rad  = _az_to_rad(downwind_azimuth - half)
    right_rad = _az_to_rad(downwind_azimuth + half)

    n_arc = 64
    arc_angles = np.linspace(left_rad, right_rad, n_arc)
    arc_pts = [(cx + buf_m * math.cos(a), cy + buf_m * math.sin(a))
               for a in arc_angles]
    sector_geom = Polygon([(cx, cy)] + arc_pts + [(cx, cy)])

    try:
        clipped = delta.rio.clip([sector_geom], all_touched=True, drop=True)
        vals = clipped.values.ravel()
        vals = vals[np.isfinite(vals)]
    except Exception:
        vals = np.array([], dtype="float32")

    cell_res = abs(float(delta.rio.resolution()[0]))
    cell_ha  = (cell_res ** 2) / 10_000.0

    if len(vals) == 0:
        return {
            "mean_delta_bp":     np.nan,
            "median_delta_bp":   np.nan,
            "max_delta_bp":      np.nan,
            "sum_delta_bp_ha":   np.nan,
            "n_pixels":          0,
            "area_improved_ha":  0.0,
            "pct_area_improved": np.nan,
            "wind_direction":    wd,
            "downwind_azimuth":  downwind_azimuth,
        }

    improved = vals > 0
    return {
        "mean_delta_bp":     float(np.mean(vals)),
        "median_delta_bp":   float(np.median(vals)),
        "max_delta_bp":      float(np.max(vals)),
        "sum_delta_bp_ha":   float(np.sum(vals) * cell_ha),
        "n_pixels":          int(vals.size),
        "area_improved_ha":  float(np.sum(improved) * cell_ha),
        "pct_area_improved": float(np.mean(improved) * 100.0),
        "wind_direction":    wd,
        "downwind_azimuth":  downwind_azimuth,
    }
