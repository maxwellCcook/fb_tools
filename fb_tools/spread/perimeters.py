"""
FSPro per-fire perimeter analysis — early and extreme fire-growth metrics.

When FSPro runs with ``SavePerimeters: 1`` it writes a perimeter shapefile
(``{basename}_Perimeters.shp``) holding, for every simulated fire, the burned
perimeter at successive points in the burn period.  This module turns those
perimeters into early-growth metrics so a counterfactual experiment can answer:

    "Did this group of treatments meaningfully alter potential fire growth
     during the first few days of ignition and moderate-to-extreme spread?"

Three entry points
------------------
``load_fspro_perimeters``
    Read a perimeter shapefile and normalize the fire-id / elapsed-time columns.

``summarize_early_growth``
    Per-fire burned area at chosen days, arrival at treatments / values, and a
    moderate-vs-extreme spread classification.

``compare_growth``
    Paired baseline-vs-treated comparison (fires pair 1:1 because both runs
    share ``SPOTTING_SEED``).

Schema note
-----------
The exact attribute names in an FSPro perimeter shapefile are not contractually
fixed and depend on the TestFSPro build.  ``load_fspro_perimeters`` auto-detects
the fire-id and elapsed-time columns by name and prints what it found.  **On the
first real Windows run, confirm the detected columns are correct** and pass
``fire_col`` / ``time_col`` explicitly if auto-detection picks the wrong field.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Candidate substrings for auto-detecting the relevant columns
_FIRE_HINTS = ("fire", "firenum", "fire_id", "fid", "sim", "run")
_TIME_HINTS = ("day", "time", "period", "elapsed", "hour", "burn")


def _detect_col(columns, hints, label, required=True):
    """Return the first column whose lowercased name contains any hint.

    When *required* is ``False`` returns ``None`` instead of raising if no
    match is found.
    """
    lowered = {c: c.lower() for c in columns}
    for hint in hints:
        for col, low in lowered.items():
            if hint in low:
                return col
    if required:
        raise ValueError(
            f"Could not auto-detect the {label} column among {list(columns)}. "
            f"Pass it explicitly."
        )
    return None


def load_fspro_perimeters(shp, fire_col=None, time_col=None):
    """
    Read an FSPro perimeter shapefile and normalize its key columns.

    Parameters
    ----------
    shp : str or Path
        Path to the FSPro ``{basename}_Perimeters.shp`` file.
    fire_col : str, optional
        Name of the per-fire identifier column.  Auto-detected when ``None``.
    time_col : str, optional
        Name of the elapsed-time / day column.  Auto-detected when ``None``.

    Returns
    -------
    geopandas.GeoDataFrame
        The perimeters with two added columns: ``fire_id`` (int) and
        ``elapsed`` (float — elapsed time as stored by FSPro, typically days).

    Raises
    ------
    FileNotFoundError
        If *shp* does not exist.
    ValueError
        If the fire-id or elapsed-time column cannot be resolved.
    """
    import geopandas as gpd

    shp = Path(shp)
    if not shp.exists():
        raise FileNotFoundError(f"Perimeter shapefile not found: {shp}")

    # on_invalid="warn" skips malformed ring geometries (unclosed LinearRings)
    # that FSPro occasionally writes, instead of raising a GEOSException.
    gdf = gpd.read_file(shp, on_invalid="warn")

    # Drop rows where geometry could not be parsed
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    n_dropped = n_before - len(gdf)
    if n_dropped:
        print(f"[load_fspro_perimeters] {shp.name}: dropped {n_dropped} invalid geometry row(s).")

    if fire_col is None:
        fire_col = _detect_col(gdf.columns.drop("geometry"), _FIRE_HINTS, "fire-id")
    if time_col is None:
        time_col = _detect_col(
            gdf.columns.drop("geometry"), _TIME_HINTS, "elapsed-time", required=False
        )

    gdf = gdf.copy()
    gdf["fire_id"] = gdf[fire_col].astype(int)

    if time_col is not None:
        gdf["elapsed"] = gdf[time_col].astype(float)
        elapsed_desc = f"time_col='{time_col}'"
    else:
        # FSPro SavePerimeters output contains only final-state perimeters
        # (no elapsed-time column).  Assign elapsed = 1.0 so downstream
        # logic treats each record as a "day 1+" snapshot — area and arrival
        # metrics will reflect final burned extent rather than daily growth.
        gdf["elapsed"] = 1.0
        elapsed_desc = "no time column found — elapsed set to 1.0 (final-state perimeters)"

    print(f"[load_fspro_perimeters] {shp.name}: {len(gdf)} perimeters; "
          f"fire_col='{fire_col}', {elapsed_desc}")
    return gdf


def summarize_early_growth(
    perims,
    treatments_gdf=None,
    values_gdf=None,
    days=(2, 3, 5),
    extreme_quantile=0.67,
):
    """
    Per-fire early-growth metrics from FSPro perimeters.

    For each simulated fire, computes the cumulative burned area at each
    requested day, the day the perimeter first reaches the treatments and the
    values layer, and a moderate-vs-extreme spread label based on the day-2
    growth rate.

    Parameters
    ----------
    perims : geopandas.GeoDataFrame
        Perimeters from :func:`load_fspro_perimeters` (must have ``fire_id``
        and ``elapsed`` columns).
    treatments_gdf : geopandas.GeoDataFrame, optional
        Treatment polygons.  When given, ``trt_arrival_day`` is computed.
    values_gdf : geopandas.GeoDataFrame, optional
        Values / assets polygons.  When given, ``val_arrival_day`` is computed.
    days : tuple of int
        Elapsed-time marks (matched against ``elapsed``) at which to record
        burned area.  Default ``(2, 3, 5)``.
    extreme_quantile : float
        Quantile of the day-2 growth-rate distribution above which a fire is
        labelled ``"extreme"``.  Default 0.67 (top third).

    Returns
    -------
    pandas.DataFrame
        One row per fire: ``fire_id``, ``area_d{D}_ha`` for each day,
        ``growth_d2_ha_per_day``, ``regime`` ("moderate"/"extreme"),
        and (when supplied) ``trt_arrival_day`` / ``val_arrival_day``.
    """
    crs = perims.crs
    trt_union = (
        treatments_gdf.to_crs(crs).geometry.union_all()
        if treatments_gdf is not None else None
    )
    val_union = (
        values_gdf.to_crs(crs).geometry.union_all()
        if values_gdf is not None else None
    )

    rows = []
    for fid, grp in perims.groupby("fire_id"):
        grp = grp.sort_values("elapsed")
        row = {"fire_id": int(fid)}

        for d in days:
            upto = grp[grp["elapsed"] <= d]
            if len(upto) == 0:
                row[f"area_d{d}_ha"] = np.nan
            else:
                # Largest cumulative perimeter at or before day d
                row[f"area_d{d}_ha"] = float(upto.geometry.area.max()) / 1e4

        a2 = row.get("area_d2_ha", np.nan)
        row["growth_d2_ha_per_day"] = a2 / 2.0 if np.isfinite(a2) else np.nan

        if trt_union is not None:
            hit = grp[grp.geometry.intersects(trt_union)]
            row["trt_arrival_day"] = (
                float(hit["elapsed"].min()) if len(hit) else np.nan
            )
        if val_union is not None:
            hit = grp[grp.geometry.intersects(val_union)]
            row["val_arrival_day"] = (
                float(hit["elapsed"].min()) if len(hit) else np.nan
            )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Classify spread regime by day-2 growth rate
    g2 = df["growth_d2_ha_per_day"]
    if g2.notna().any():
        thresh = g2.quantile(extreme_quantile)
        df["regime"] = np.where(g2 >= thresh, "extreme", "moderate")
    else:
        df["regime"] = "moderate"

    return df


def compare_growth(
    baseline_perims,
    treated_perims,
    treatments_gdf=None,
    values_gdf=None,
    days=(2, 3, 5),
    extreme_quantile=0.67,
):
    """
    Paired baseline-vs-treated comparison of early fire growth.

    Because the baseline and treated FSPro runs share ``SPOTTING_SEED``, their
    fires pair 1:1 by ``fire_id``.  This computes per-fire growth differences
    and aggregates them by spread regime, directly addressing whether the
    treatments altered early / extreme fire growth.

    Parameters
    ----------
    baseline_perims, treated_perims : geopandas.GeoDataFrame
        Perimeters from :func:`load_fspro_perimeters` for the two landscapes.
    treatments_gdf, values_gdf : geopandas.GeoDataFrame, optional
        Passed through to :func:`summarize_early_growth`.
    days : tuple of int
        Day marks for burned-area comparison.  Default ``(2, 3, 5)``.
    extreme_quantile : float
        Regime split quantile; applied to the **baseline** day-2 growth so
        each fire keeps one regime label across both landscapes.

    Returns
    -------
    dict
        ``"per_fire"`` : pandas.DataFrame
            Paired rows: ``area_d{D}_ha`` for baseline (``_bl``) and treated
            (``_tr``), ``delta_d{D}_ha``, ``pct_reduction_d{D}``,
            ``trt_arrival_delay_day``, ``val_arrival_delay_day``, ``regime``.
        ``"summary"`` : pandas.DataFrame
            Mean delta and % reduction per day, grouped by ``regime`` and
            with an ``all`` row.
    """
    bl = summarize_early_growth(
        baseline_perims, treatments_gdf, values_gdf, days, extreme_quantile
    )
    tr = summarize_early_growth(
        treated_perims, treatments_gdf, values_gdf, days, extreme_quantile
    )

    # Regime comes from the baseline so a fire has one label across both runs
    paired = bl.merge(tr, on="fire_id", suffixes=("_bl", "_tr"))
    paired["regime"] = paired["regime_bl"]

    for d in days:
        bl_col, tr_col = f"area_d{d}_ha_bl", f"area_d{d}_ha_tr"
        paired[f"delta_d{d}_ha"] = paired[bl_col] - paired[tr_col]
        with np.errstate(divide="ignore", invalid="ignore"):
            paired[f"pct_reduction_d{d}"] = (
                100.0 * (paired[bl_col] - paired[tr_col]) / paired[bl_col]
            )

    if treatments_gdf is not None:
        paired["trt_arrival_delay_day"] = (
            paired["trt_arrival_day_tr"] - paired["trt_arrival_day_bl"]
        )
    if values_gdf is not None:
        paired["val_arrival_delay_day"] = (
            paired["val_arrival_day_tr"] - paired["val_arrival_day_bl"]
        )

    metric_cols = (
        [f"delta_d{d}_ha" for d in days]
        + [f"pct_reduction_d{d}" for d in days]
    )
    for c in ("trt_arrival_delay_day", "val_arrival_delay_day"):
        if c in paired.columns:
            metric_cols.append(c)

    by_regime = paired.groupby("regime")[metric_cols].mean()
    overall = paired[metric_cols].mean().to_frame().T
    overall.index = ["all"]
    summary = pd.concat([by_regime, overall])

    return {"per_fire": paired, "summary": summary.reset_index(names="regime")}
