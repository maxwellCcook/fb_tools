"""
Burn probability analysis — delta burn probability and treatment effect summaries.

Three public entry points
-------------------------
``delta_burn_probability``
    Pixel-wise difference between baseline and treated burn probability rasters.
    Positive values indicate the treatment reduced burn probability.

``summarize_bp_treatments``
    Zonal statistics of burn probability change per treatment polygon.
    Mirrors the structure of :func:`~fb_tools.analysis.treatments.summarize_treatments`.

``downwind_treatment_effect``
    Summarize delta burn probability in the downwind sector of a treatment
    polygon.  Wind direction defaults to the scenario's ``WIND_DIRECTION``
    but can be overridden by the caller.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def delta_burn_probability(baseline_bp, treatment_bp, out_path=None, scale=100):
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
        stored as ``delta × scale`` to preserve sub-unit precision.
    scale : int
        Scale factor for the int16 output (default ``100``).  Set to ``1``
        if inputs are already in percent (0–100) units.

    Returns
    -------
    xarray.DataArray
        Delta burn probability as float32.  Positive = treatment reduced BP.

    Notes
    -----
    Both inputs are aligned to the baseline grid via ``xarray.align`` with
    ``join="left"`` before differencing, so minor extent differences between
    model outputs are handled gracefully.
    """
    bl = _open_bp(baseline_bp)
    tr = _open_bp(treatment_bp)

    bl, tr = xr.align(bl, tr, join="left")

    delta = (bl - tr).astype("float32")
    delta.attrs = bl.attrs

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_int = (delta * scale).round().astype("int16")
        out_int.rio.to_raster(out_path, dtype="int16")
        print(f"Delta burn probability written to {out_path}")

    return delta


def summarize_bp_treatments(
    treatments_gdf,
    id_col,
    type_col,
    baseline_bp_dir,
    treated_bp_dirs,
    model="mtt",
    out_dir=None,
):
    """
    Compute zonal burn probability change statistics per treatment polygon.

    Mirrors the structure of
    :func:`~fb_tools.analysis.treatments.summarize_treatments` for burn
    probability outputs.

    Parameters
    ----------
    treatments_gdf : GeoDataFrame
        Treatment polygons.  Must contain *id_col*, *type_col*, and geometry.
    id_col : str
        Numeric treatment identifier column (e.g. ``"TRT_ID"``).
    type_col : str
        Treatment type column (e.g. ``"TRT_TYPE"``).
    baseline_bp_dir : str or Path
        Directory containing the baseline burn probability GeoTIFF.
    treated_bp_dirs : dict
        Mapping ``{treatment_type: directory}`` pointing to treated burn
        probability outputs.  Keys must match values in *type_col*.
    model : str
        Source model — ``"mtt"``, ``"fspro"``, or ``"cell2fire"``.  Controls
        which filename pattern is used to locate burn probability rasters.
    out_dir : str or Path, optional
        If provided, write ``bp_change.csv`` here.

    Returns
    -------
    pd.DataFrame
        One row per treatment polygon with columns:
        *id_col*, *type_col*, ``ACRES_GIS``,
        ``BP_bl_mean``, ``BP_tr_mean``, ``BP_delta_mean``,
        ``BP_bl_max``, ``BP_tr_max``.

    Raises
    ------
    FileNotFoundError
        If no burn probability raster is found in *baseline_bp_dir*.
    ValueError
        If a treatment type in *treatments_gdf* has no entry in
        *treated_bp_dirs*.
    """
    from ..utils.geo import geom_to_raster_crs

    baseline_bp_dir = Path(baseline_bp_dir)
    bl_tif = _find_bp_tif(baseline_bp_dir, model=model)
    if bl_tif is None:
        raise FileNotFoundError(
            f"No burn probability raster found in {baseline_bp_dir}"
        )

    bl_da = _open_bp(bl_tif)

    rows = []
    for _, poly in treatments_gdf.iterrows():
        trt_type = poly[type_col]
        trt_id   = poly[id_col]

        if trt_type not in treated_bp_dirs:
            raise ValueError(
                f"Treatment type '{trt_type}' not found in treated_bp_dirs. "
                f"Available: {list(treated_bp_dirs.keys())}"
            )

        tr_dir = Path(treated_bp_dirs[trt_type])
        tr_tif = _find_bp_tif(tr_dir, model=model)
        if tr_tif is None:
            raise FileNotFoundError(
                f"No burn probability raster found in {tr_dir} "
                f"for treatment type '{trt_type}'"
            )
        tr_da = _open_bp(tr_tif)

        # Reproject geometry to raster CRS for clipping
        geom = geom_to_raster_crs(poly.geometry, treatments_gdf.crs, bl_da)

        bl_clip = bl_da.rio.clip([geom], all_touched=True).values.ravel()
        tr_clip = tr_da.rio.clip([geom], all_touched=True).values.ravel()

        bl_vals = bl_clip[np.isfinite(bl_clip)]
        tr_vals = tr_clip[np.isfinite(tr_clip)]

        # Acres: count valid baseline pixels × cell area in acres
        cell_res = abs(float(bl_da.rio.resolution()[0]))
        acres = len(bl_vals) * (cell_res ** 2) / 4046.86

        rows.append({
            id_col:          trt_id,
            type_col:        trt_type,
            "ACRES_GIS":     round(acres, 2),
            "BP_bl_mean":    round(float(np.mean(bl_vals)), 4) if len(bl_vals) else np.nan,
            "BP_tr_mean":    round(float(np.mean(tr_vals)), 4) if len(tr_vals) else np.nan,
            "BP_delta_mean": round(float(np.mean(bl_vals - tr_vals[:len(bl_vals)])), 4) if len(bl_vals) and len(tr_vals) else np.nan,
            "BP_bl_max":     round(float(np.max(bl_vals)), 4) if len(bl_vals) else np.nan,
            "BP_tr_max":     round(float(np.max(tr_vals)), 4) if len(tr_vals) else np.nan,
        })

    df = pd.DataFrame(rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "bp_change.csv"
        df.to_csv(csv_path, index=False)
        print(f"BP change summary written to {csv_path}")

    return df


def downwind_treatment_effect(
    treatment_polygon,
    delta_bp,
    wind_direction=None,
    scenario_row=None,
    buffer_km=10.0,
    sector_degrees=45.0,
):
    """
    Summarize the delta burn probability in the downwind sector of a treatment.

    Constructs a downwind cone/sector from the treatment centroid and computes
    zonal statistics on *delta_bp* within that sector.

    Parameters
    ----------
    treatment_polygon : shapely geometry or GeoDataFrame row
        Treatment boundary (any CRS; reprojected internally to the *delta_bp*
        raster CRS before constructing the sector).
    delta_bp : xarray.DataArray, str, or Path
        Delta burn probability raster (positive = treatment reduced BP).
    wind_direction : float, optional
        Prevailing wind azimuth in degrees from north (0–360).  When
        provided, takes precedence over *scenario_row*.
    scenario_row : pandas.Series, optional
        Scenario row containing a ``WIND_DIRECTION`` column used as the
        default when *wind_direction* is ``None``.  Pass either this or
        *wind_direction* — at least one must be non-``None``.
    buffer_km : float
        Downwind buffer distance in kilometres (default ``10.0``).
    sector_degrees : float
        Full angular width of the downwind sector in degrees (default
        ``45.0``; centred on *wind_direction*).

    Returns
    -------
    dict
        ``{"mean_delta_bp": float, "max_delta_bp": float,
           "area_improved_ha": float, "pct_area_improved": float,
           "wind_direction": float}``.
        ``area_improved_ha`` counts pixels where delta BP > 0.

    Raises
    ------
    ValueError
        If neither *wind_direction* nor *scenario_row* is provided.
    """
    import math
    from shapely.geometry import Point
    from shapely.affinity import affine_transform

    # Resolve wind direction
    if wind_direction is not None:
        wd = float(wind_direction)
    elif scenario_row is not None:
        wd = float(scenario_row["WIND_DIRECTION"])
    else:
        raise ValueError(
            "Provide either wind_direction (float) or scenario_row (Series "
            "with WIND_DIRECTION column)."
        )

    delta = _open_bp(delta_bp)

    # Get geometry from GeoDataFrame row or raw shapely geometry
    if hasattr(treatment_polygon, "geometry"):
        geom = treatment_polygon.geometry
        src_crs = None  # assume already in raster CRS or handle below
    else:
        geom = treatment_polygon
        src_crs = None

    # Reproject centroid to raster CRS (projected metres assumed)
    from ..utils.geo import geom_to_raster_crs
    import geopandas as gpd
    if src_crs is not None:
        geom = geom_to_raster_crs(geom, src_crs, delta)

    centroid = geom.centroid
    cx, cy = centroid.x, centroid.y
    buf_m = buffer_km * 1000.0
    half = sector_degrees / 2.0

    # Build downwind sector polygon
    # Wind direction (met convention): wind FROM this direction blows TOWARD
    # the opposite bearing.  "Downwind" = the direction wind is blowing TO.
    downwind_azimuth = (wd + 180.0) % 360.0

    # Convert azimuth to math angle (CCW from east)
    def _az_to_rad(az):
        return math.radians(90.0 - az)

    center_rad = _az_to_rad(downwind_azimuth)
    left_rad   = _az_to_rad(downwind_azimuth - half)
    right_rad  = _az_to_rad(downwind_azimuth + half)

    # Sector approximation: polygon with arc points
    n_arc = 32
    arc_angles = np.linspace(left_rad, right_rad, n_arc)
    arc_pts = [(cx + buf_m * math.cos(a), cy + buf_m * math.sin(a)) for a in arc_angles]
    sector_coords = [(cx, cy)] + arc_pts + [(cx, cy)]

    from shapely.geometry import Polygon
    sector_geom = Polygon(sector_coords)

    # Clip delta_bp to the sector
    clipped = delta.rio.clip([sector_geom], all_touched=True)
    vals = clipped.values.ravel()
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return {
            "mean_delta_bp":    np.nan,
            "max_delta_bp":     np.nan,
            "area_improved_ha": 0.0,
            "pct_area_improved": 0.0,
            "wind_direction":   wd,
        }

    cell_res = abs(float(delta.rio.resolution()[0]))
    cell_ha  = (cell_res ** 2) / 10000.0
    improved = vals > 0

    return {
        "mean_delta_bp":     float(np.mean(vals)),
        "max_delta_bp":      float(np.max(vals)),
        "area_improved_ha":  float(np.sum(improved) * cell_ha),
        "pct_area_improved": float(np.mean(improved) * 100.0),
        "wind_direction":    wd,
    }
