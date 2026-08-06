"""
FSPro output readers and run diagnostics.

TestFSPro writes far more than the three ASC grids the rest of this package
reads.  This module adds the products needed to judge whether a run is
trustworthy before its deltas are interpreted:

``read_daily_acres``
    ``_DailyAcres.txt`` — per-fire daily area increments.  This is the growth
    record; ``_Perimeters.shp`` cannot be loaded (FSPro writes malformed
    ``LinearRing`` geometries) and its DBF carries no time field.

``check_domain_adequacy``
    The edge-effect diagnostic.  A simulation domain that ends near the fire
    is a hard growth boundary: spread is truncated, burn probability is biased
    low near the edge, and transmission to destinations outside the source
    unit cannot be measured at all.  Run this on every FSPro run before its
    burn-probability surface is differenced or aggregated.

Both run on macOS — they read files produced by a completed Windows run.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Fraction of total burn-probability mass allowed within ``edge_cells`` of the
# domain boundary before the domain is judged to have truncated spread.
_DEFAULT_EDGE_BP_TOLERANCE = 1e-3

# Share of total growth occurring on the final simulated day above which
# ``Duration`` is judged to be binding rather than the weather.
_DEFAULT_FINAL_DAY_TOLERANCE = 0.20


# ── Private helpers ───────────────────────────────────────────────────────────

def _open_grid(src):
    """Open *src* as a masked float32 2-D array plus its cell size in metres.

    Returns
    -------
    tuple of (np.ndarray, float)
        The array with nodata as ``NaN``, and the (square) cell size.
    """
    import rioxarray as rxr

    if isinstance(src, (str, Path)):
        da = rxr.open_rasterio(Path(src), masked=True).squeeze("band", drop=True)
    else:
        da = src.squeeze() if src.ndim > 2 else src

    arr = np.asarray(da.values, dtype="float32")
    res = da.rio.resolution()
    cell = float(abs(res[0]))
    if not np.isclose(abs(res[0]), abs(res[1])):
        raise ValueError(
            f"non-square cells ({abs(res[0])} x {abs(res[1])}); "
            "the diagnostic assumes square pixels"
        )
    return arr, cell


def _distance_cells(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance in cells from every cell to the nearest ``True`` in *mask*."""
    from scipy import ndimage

    if not mask.any():
        raise ValueError("distance transform requires a non-empty mask")
    return ndimage.distance_transform_edt(~mask)


def _depth_from_boundary(valid: np.ndarray) -> np.ndarray:
    """Distance in cells from each valid cell to the nearest cell outside the domain.

    The array border counts as outside, so a domain that fills its grid is
    still measured against the grid edge.
    """
    from scipy import ndimage

    padded = np.pad(valid, 1, mode="constant", constant_values=False)
    depth = ndimage.distance_transform_edt(padded)
    return depth[1:-1, 1:-1]


def _ignition_mask(ignition, shape, transform, crs) -> np.ndarray:
    """Rasterize *ignition* onto the burn-probability grid.

    Parameters
    ----------
    ignition : str, Path, np.ndarray, GeoDataFrame, or shapely geometry
        A path to ``_Ignitions.asc`` (or any grid where positive values mark
        the footprint), a boolean array already on the grid, or vector
        geometry to burn in.
    shape : tuple of int
    transform : affine.Affine
    crs : rasterio CRS

    Returns
    -------
    np.ndarray
        Boolean mask, ``True`` inside the ignition footprint.
    """
    if isinstance(ignition, np.ndarray) and ignition.dtype == bool:
        if ignition.shape != shape:
            raise ValueError(
                f"ignition mask shape {ignition.shape} != BP grid shape {shape}"
            )
        return ignition

    if isinstance(ignition, (str, Path)):
        path = Path(ignition)
        if path.suffix.lower() in (".asc", ".tif", ".tiff", ".img"):
            arr, _ = _open_grid(path)
            if arr.shape != shape:
                raise ValueError(
                    f"ignition grid shape {arr.shape} != BP grid shape {shape}"
                )
            return np.nan_to_num(arr, nan=0.0) > 0
        import geopandas as gpd
        ignition = gpd.read_file(path)

    # Vector geometry — reproject to the grid CRS and burn it in.
    from rasterio import features

    if hasattr(ignition, "geometry"):          # GeoDataFrame / GeoSeries
        gdf = ignition.to_crs(crs) if ignition.crs is not None else ignition
        geoms = list(gdf.geometry)
    else:                                       # bare shapely geometry
        geoms = [ignition]

    return features.geometry_mask(
        geoms, out_shape=shape, transform=transform, invert=True, all_touched=True
    )


# ── Output readers ────────────────────────────────────────────────────────────

def read_daily_acres(path: "str | Path", duration: "int | None" = None) -> pd.DataFrame:
    """
    Read ``_DailyAcres.txt`` into a tidy per-fire, per-day growth table.

    The file has no header and no fire identifier.  Each line is
    ``day,acres_burned_that_day`` — a **daily increment**, not a cumulative
    total — and fires appear as consecutive blocks whose ``day`` column
    restarts at 1.  Fire identity is recovered from those restarts, so the
    file is parsed without assuming a fixed block length.

    Parameters
    ----------
    path : str or Path
        Path to ``{base}_DailyAcres.txt``.
    duration : int, optional
        Expected number of days per fire.  When given, a warning is printed
        for any fire whose block length differs — usually a sign that the file
        was truncated or that ``Duration`` changed mid-batch.

    Returns
    -------
    pd.DataFrame
        Columns ``fire_id`` (0-based), ``day`` (1-based), ``acres_day``
        (daily increment), ``acres_cum`` (cumulative within the fire),
        ``hectares_day``, ``hectares_cum``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is empty or has no parseable rows.

    Examples
    --------
    >>> df = read_daily_acres("out/fspro_p47_DailyAcres.txt")
    >>> df.groupby("fire_id")["acres_day"].sum().describe()  # final fire sizes
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DailyAcres file not found: {path}")

    days: list[int] = []
    acres: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            days.append(int(float(parts[0])))
            acres.append(float(parts[1]))
        except ValueError:
            continue          # header or trailer line

    if not days:
        raise ValueError(f"no parseable rows in {path}")

    day_arr = np.asarray(days, dtype=int)
    # A new fire starts wherever the day counter does not advance by one.
    starts = np.ones(len(day_arr), dtype=bool)
    starts[1:] = day_arr[1:] <= day_arr[:-1]
    fire_id = np.cumsum(starts) - 1

    df = pd.DataFrame({
        "fire_id":   fire_id,
        "day":       day_arr,
        "acres_day": np.asarray(acres, dtype=float),
    })
    df["acres_cum"] = df.groupby("fire_id")["acres_day"].cumsum()
    df["hectares_day"] = df["acres_day"] * 0.404686
    df["hectares_cum"] = df["acres_cum"] * 0.404686

    if duration is not None:
        lengths = df.groupby("fire_id").size()
        odd = lengths[lengths != duration]
        if len(odd):
            print(
                f"  [read_daily_acres] {len(odd)} of {len(lengths)} fires do not "
                f"have {duration} daily records (lengths "
                f"{sorted(odd.unique().tolist())})"
            )

    print(
        f"  [read_daily_acres] {df['fire_id'].nunique()} fires x "
        f"{df.groupby('fire_id').size().max()} days from {path.name}"
    )
    return df


# ── Diagnostics ───────────────────────────────────────────────────────────────

def check_domain_adequacy(
    bp_raster,
    ignition=None,
    daily_acres=None,
    edge_cells: int = 3,
    bp_min: float = 0.0,
    edge_tolerance: float = _DEFAULT_EDGE_BP_TOLERANCE,
    final_day_tolerance: float = _DEFAULT_FINAL_DAY_TOLERANCE,
    verbose: bool = True,
) -> dict:
    """
    Test whether the simulation domain was large enough to let fires run out.

    A domain edge is a hard, non-burnable boundary.  When burn probability
    piles up against it, three things follow: simulated fire sizes are bounded
    by the domain rather than by weather and fuels; burn probability is biased
    low near the edge by an amount that depends on how close the treatment
    sits to it; and transmission to destinations outside the source unit
    cannot be measured, because there is nowhere left to transmit to.

    Three independent signals are reported:

    1. **Edge mass** — the share of total burn-probability mass lying within
       *edge_cells* of the domain boundary.  Should be ≈ 0.
    2. **Spread headroom** — the farthest burned cell from the ignition,
       against the ignition's closest approach to the boundary.  A ratio above
       1 means the fire never had room to reach the edge.
    3. **Growth saturation** — from ``_DailyAcres.txt``, the share of total
       area accrued on the final simulated day.  A large share means growth
       was still rising when the run stopped, so ``Duration`` (not the domain)
       is what bound the fire.

    Parameters
    ----------
    bp_raster : str, Path, or xarray.DataArray
        Burn probability grid (``_BurnProb.asc`` or a GeoTIFF of it), values
        in 0–1.
    ignition : str, Path, np.ndarray, GeoDataFrame, or shapely geometry, optional
        The ignition footprint.  Easiest source is the run's ``_Ignitions.asc``,
        which is already on the burn-probability grid.  A boolean array on that
        grid, or any vector geometry, also works.  Without it the spread-headroom
        signal is skipped.
    daily_acres : str, Path, or pd.DataFrame, optional
        ``_DailyAcres.txt`` path or the frame from :func:`read_daily_acres`.
        Without it the growth-saturation signal is skipped.
    edge_cells : int
        Width in cells of the boundary band (default ``3``).
    bp_min : float
        Burn probability above which a cell counts as burned when measuring
        spread distance (default ``0.0``, i.e. any non-zero probability).
    edge_tolerance : float
        Maximum acceptable ``edge_bp_fraction`` (default ``0.001``).
    final_day_tolerance : float
        Maximum acceptable ``final_day_growth_share`` (default ``0.20``).
    verbose : bool
        Print a formatted report (default ``True``).

    Returns
    -------
    dict
        ``domain_shape``, ``cell_size_m``, ``domain_area_ha``,
        ``expected_area_burned_ha`` (``Σ BP × cell_area`` — the same quantity
        as Ager's TF over the whole domain), ``burned_cell_fraction``,
        ``edge_bp_fraction``, ``edge_max_bp``, ``edge_burned_cells``,
        ``max_spread_km``, ``ignition_boundary_dist_km``, ``spread_headroom``,
        ``final_day_growth_share``, ``fires_growing_on_final_day``,
        ``mean_final_size_ac``, and the booleans ``edge_ok``,
        ``headroom_ok``, ``growth_ok``, ``passed``.  Signals that were skipped
        are ``None`` and their boolean is ``None``.

    Notes
    -----
    Passing is necessary, not sufficient.  A domain can be large enough for
    the fire yet still too small for the *analysis*: TF_ij also needs the
    destination accounting units to lie inside it.

    Examples
    --------
    >>> res = check_domain_adequacy(
    ...     "out/fspro_p47_BurnProb.asc",
    ...     ignition="out/fspro_p47_Ignitions.asc",
    ...     daily_acres="out/fspro_p47_DailyAcres.txt",
    ... )
    >>> res["passed"]
    True
    """
    import rioxarray as rxr

    if isinstance(bp_raster, (str, Path)):
        da = rxr.open_rasterio(Path(bp_raster), masked=True).squeeze("band", drop=True)
    else:
        da = bp_raster.squeeze() if bp_raster.ndim > 2 else bp_raster

    bp, cell = _open_grid(da)
    transform = da.rio.transform()
    crs = da.rio.crs

    valid = np.isfinite(bp)
    if not valid.any():
        raise ValueError("burn probability grid has no valid cells")

    cell_ha = (cell * cell) / 10_000.0
    bp_valid = np.where(valid, np.nan_to_num(bp, nan=0.0), 0.0)
    total_bp = float(bp_valid.sum())

    res: dict = {
        "domain_shape": tuple(bp.shape),
        "cell_size_m": cell,
        "domain_area_ha": float(valid.sum()) * cell_ha,
        "expected_area_burned_ha": total_bp * cell_ha,
        "burned_cell_fraction": float((bp_valid > bp_min).sum()) / float(valid.sum()),
    }

    # ── 1. Edge mass ──────────────────────────────────────────────────────────
    depth = _depth_from_boundary(valid)
    edge_band = valid & (depth <= edge_cells)
    edge_bp = float(bp_valid[edge_band].sum())

    res["edge_bp_fraction"] = (edge_bp / total_bp) if total_bp > 0 else 0.0
    res["edge_max_bp"] = float(bp_valid[edge_band].max()) if edge_band.any() else 0.0
    res["edge_burned_cells"] = int((bp_valid[edge_band] > bp_min).sum())
    res["edge_ok"] = res["edge_bp_fraction"] <= edge_tolerance

    # ── 2. Spread headroom ────────────────────────────────────────────────────
    burned = valid & (bp_valid > bp_min)
    if ignition is None or not burned.any():
        res["max_spread_km"] = None
        res["ignition_boundary_dist_km"] = None
        res["spread_headroom"] = None
        res["headroom_ok"] = None
        res["ignition_cells"] = None
    else:
        ign = _ignition_mask(ignition, bp.shape, transform, crs)
        if not ign.any():
            raise ValueError("ignition footprint is empty on the burn-probability grid")

        dist_from_ign_km = _distance_cells(ign) * cell / 1000.0
        boundary_km = float(depth[ign].min()) * cell / 1000.0
        max_spread_km = float(dist_from_ign_km[burned].max())

        res["ignition_cells"] = int(ign.sum())
        res["max_spread_km"] = max_spread_km
        res["ignition_boundary_dist_km"] = boundary_km
        res["spread_headroom"] = (
            boundary_km / max_spread_km if max_spread_km > 0 else float("inf")
        )
        res["headroom_ok"] = res["spread_headroom"] > 1.0

    # ── 3. Growth saturation ──────────────────────────────────────────────────
    if daily_acres is None:
        res["final_day_growth_share"] = None
        res["fires_growing_on_final_day"] = None
        res["mean_final_size_ac"] = None
        res["growth_ok"] = None
    else:
        df = (daily_acres if isinstance(daily_acres, pd.DataFrame)
              else read_daily_acres(daily_acres))
        last_day = df.groupby("fire_id")["day"].transform("max")
        final = df[df["day"] == last_day]
        totals = df.groupby("fire_id")["acres_day"].sum()

        total_all = float(totals.sum())
        res["final_day_growth_share"] = (
            float(final["acres_day"].sum()) / total_all if total_all > 0 else 0.0
        )
        res["fires_growing_on_final_day"] = (
            float((final["acres_day"] > 0).mean()) if len(final) else 0.0
        )
        res["mean_final_size_ac"] = float(totals.mean())
        res["growth_ok"] = res["final_day_growth_share"] <= final_day_tolerance

    checks = [res["edge_ok"], res["headroom_ok"], res["growth_ok"]]
    res["passed"] = all(c for c in checks if c is not None)

    if verbose:
        _print_domain_report(res, edge_cells, edge_tolerance, final_day_tolerance)

    return res


def _print_domain_report(res, edge_cells, edge_tolerance, final_day_tolerance) -> None:
    """Print the :func:`check_domain_adequacy` result as a short report."""
    def flag(ok):
        return "  -  " if ok is None else ("PASS " if ok else "FAIL ")

    rows, cols = res["domain_shape"]
    cell = res["cell_size_m"]
    print("  [check_domain_adequacy]")
    print(f"    domain            {rows} x {cols} @ {cell:g} m "
          f"= {rows * cell / 1000:.0f} x {cols * cell / 1000:.0f} km "
          f"({res['domain_area_ha']:,.0f} ha)")
    print(f"    expected burned   {res['expected_area_burned_ha']:,.0f} ha "
          f"(sum BP x cell area); {res['burned_cell_fraction'] * 100:.2f}% of "
          "cells have BP > 0")

    print(f"    {flag(res['edge_ok'])}edge mass       "
          f"{res['edge_bp_fraction'] * 100:.4f}% of BP within {edge_cells} cells "
          f"of the boundary (limit {edge_tolerance * 100:g}%), "
          f"max edge BP {res['edge_max_bp']:.3f}")

    if res["spread_headroom"] is None:
        print(f"    {flag(None)}spread headroom skipped (no ignition supplied)")
    else:
        print(f"    {flag(res['headroom_ok'])}spread headroom "
              f"{res['spread_headroom']:.2f}x — fire reached "
              f"{res['max_spread_km']:.1f} km, boundary is "
              f"{res['ignition_boundary_dist_km']:.1f} km away")

    if res["final_day_growth_share"] is None:
        print(f"    {flag(None)}growth          skipped (no DailyAcres supplied)")
    else:
        print(f"    {flag(res['growth_ok'])}growth          "
              f"{res['final_day_growth_share'] * 100:.1f}% of area accrued on the "
              f"final day (limit {final_day_tolerance * 100:g}%); "
              f"{res['fires_growing_on_final_day'] * 100:.0f}% of fires still "
              f"growing; mean size {res['mean_final_size_ac']:,.0f} ac")

    print(f"    => {'PASS' if res['passed'] else 'FAIL'}")
