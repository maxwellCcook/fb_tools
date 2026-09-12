"""
Read, summarize, and aggregate pyretechnics realizations.

Per realization ``run_realization`` writes ``time_of_arrival`` (minutes),
``fire_type``, ``spread_rate`` (m/min), ``fireline_intensity`` (kW/m) and
``flame_length`` (m), NaN outside the burn. This module turns those into
per-realization metrics (growth, arrival at zones, firebrand production),
exactly paired deltas, and the three ensemble rasters ``bp.py`` consumes.
Engine-free.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY_MARKS = (1, 2, 3, 5, 7, 14, 21)


REALIZATION_LAYERS = ("time_of_arrival", "fire_type", "spread_rate", "spread_direction",
                      "fireline_intensity", "flame_length")


def read_realization(run_dir, layers=REALIZATION_LAYERS):
    """
    Load a realization's rasters and manifest.

    Returns
    -------
    dict
        ``{layer: 2-D float32 array}`` plus ``manifest`` (dict), ``transform``,
        ``crs`` and ``res_m``.
    """
    import rasterio
    run_dir = Path(run_dir)
    out = {"manifest": json.loads((run_dir / "manifest.json").read_text())}
    for name in layers:
        with rasterio.open(run_dir / f"{name}.tif") as src:
            out[name] = src.read(1)
            out["transform"], out["crs"] = src.transform, src.crs
    out["res_m"] = abs(out["transform"].a)
    return out


def firebrand_production(fireline_intensity, spread_rate, cell_area_m2, *, slope=None,
                         firebrands_per_unit_heat=5e-11):
    """
    Expected firebrands produced per cell — pyretechnics'
    ``spot_fire.expected_firebrand_production`` vectorized.

    ``heat/area = 60 * I / R`` (kJ/m2); cell area is slope-corrected by
    ``sqrt(1 + slope^2)`` when ``slope`` (rise/run) is given.
    """
    I = np.asarray(fireline_intensity, dtype="float64")
    R = np.asarray(spread_rate, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        heat = np.where(R > 0, 60.0 * I / R, 0.0)
    area = cell_area_m2 * (np.sqrt(1.0 + np.square(slope)) if slope is not None else 1.0)
    return np.nan_to_num(firebrands_per_unit_heat * heat * area)


def zone_masks(zones, transform, shape, id_col="zone"):
    """Rasterize zone polygons -> ``{zone_id: bool mask}`` (all_touched)."""
    from rasterio.features import rasterize
    masks = {}
    for zid, sub in zones.groupby(id_col):
        m = rasterize(((g, 1) for g in sub.geometry), out_shape=shape, transform=transform,
                      all_touched=True, dtype="uint8").astype(bool)
        masks[zid] = m
    return masks


def summarize_realization(run_dir, *, zones=None, id_col="zone", day_marks=DAY_MARKS,
                          firebrands_per_unit_heat=None, slope=None):
    """
    Per-realization metrics.

    Parameters
    ----------
    run_dir : Path
    zones : GeoDataFrame, optional
        Reporting polygons in the raster CRS (treatment units, SFAs, collars).
    id_col : str
    day_marks : tuple of int
        Days at which burned area is read off ``time_of_arrival``.
    firebrands_per_unit_heat : float, optional
        Flux for :func:`firebrand_production`; defaults to the run's
        ``spot_config`` value, else pyretechnics' 5e-11.
    slope : 2-D array, optional
        Rise/run for slope-corrected cell area.

    Returns
    -------
    dict
        ``burned_ha, crown_frac, fl_p95_m, fi_p95_kw_m, max_toa_h, firebrands,
        area_d{D}_ha`` and, per zone, ``{zone}_arrival_h`` (NaN if never
        reached), ``{zone}_frac_burned``, ``{zone}_fi_mean``, ``{zone}_firebrands``.
    """
    r = read_realization(run_dir)
    man, toa, ft = r["manifest"], r["time_of_arrival"], r["fire_type"]
    burned = np.isfinite(toa)
    cell_ha = r["res_m"] ** 2 / 1e4
    flux = firebrands_per_unit_heat or ((man.get("spot_config") or {}).get("firebrands_per_unit_heat", 5e-11))
    fb = firebrand_production(r["fireline_intensity"], r["spread_rate"], r["res_m"] ** 2,
                              slope=slope, firebrands_per_unit_heat=flux)

    out = {k: man.get(k) for k in ("arm", "ignition_id", "episode_id", "condition_id", "seed")}
    out.update({
        "burned_ha": round(float(burned.sum() * cell_ha), 3),
        "crown_frac": round(float((ft[burned] >= 2).mean()), 4) if burned.any() else np.nan,
        "fl_p95_m": round(float(np.nanpercentile(r["flame_length"], 95)), 2),
        "fi_p95_kw_m": round(float(np.nanpercentile(r["fireline_intensity"], 95)), 0),
        "max_toa_h": round(float(np.nanmax(toa) / 60.0), 1),
        "firebrands": float(fb.sum()),
    })
    for d in day_marks:
        out[f"area_d{d}_ha"] = round(float((toa <= d * 1440).sum() * cell_ha), 3)

    if zones is not None:
        for zid, m in zone_masks(zones, r["transform"], toa.shape, id_col).items():
            zb = burned & m
            out[f"{zid}_arrival_h"] = round(float(np.nanmin(toa[m]) / 60.0), 1) if zb.any() else np.nan
            out[f"{zid}_frac_burned"] = round(float(zb.sum() / max(m.sum(), 1)), 4)
            out[f"{zid}_fi_mean"] = round(float(r["fireline_intensity"][zb].mean()), 0) if zb.any() else np.nan
            out[f"{zid}_firebrands"] = float(fb[m].sum())
    return out


def summarize_runs(results, **kwargs):
    """:func:`summarize_realization` over a ``run_batch`` result; skips rows with ``status != "ok"``."""
    rows = [summarize_realization(r["out_dir"], **kwargs)
            for _, r in results.iterrows() if r["status"] == "ok"]
    return pd.DataFrame(rows)


def paired_deltas(summary, *, baseline, treated, cols=None):
    """
    Treated minus baseline on the shared ``(condition_id, ignition_id, episode_id)`` key.

    Returns one row per realization with ``d_<col>`` for every numeric column
    (or ``cols``). Positive arrival delta = treatment delayed arrival.
    """
    key = ["condition_id", "ignition_id", "episode_id"]
    b = summary[summary.arm == baseline].set_index(key)
    t = summary[summary.arm == treated].set_index(key)
    if not b.index.equals(t.index):
        t, b = t.align(b, join="inner", axis=0)
    cols = cols or [c for c in b.columns if c not in ("arm", "seed")
                    and pd.api.types.is_numeric_dtype(b[c])]
    d = (t[cols] - b[cols]).add_prefix("d_")
    d.insert(0, "seed", b["seed"])
    return d.reset_index()


def aggregate_realizations(run_dirs, out_dir, *, verbose=True):
    """
    Collapse realizations into the rasters ``bp.py`` consumes.

    Writes ``BurnProb.tif`` (fraction of realizations burned, 0 everywhere
    unburned), ``AvgFlameLength.tif`` (m, NaN outside), ``AvgTime.tif``
    (**days**, NaN outside) and ``n_burned.tif``.

    Returns
    -------
    dict
        ``{name: Path}``.
    """
    import rasterio
    run_dirs = [Path(p) for p in run_dirs]
    n_b = fl_sum = toa_sum = None
    profile = None
    for rd in run_dirs:
        r = read_realization(rd, layers=("time_of_arrival", "flame_length"))
        b = np.isfinite(r["time_of_arrival"])
        if n_b is None:
            n_b = np.zeros(b.shape, dtype="int32")
            fl_sum = np.zeros(b.shape, dtype="float64")
            toa_sum = np.zeros(b.shape, dtype="float64")
            profile = dict(driver="GTiff", height=b.shape[0], width=b.shape[1], count=1,
                           crs=r["crs"], transform=r["transform"], compress="deflate")
        n_b += b
        fl_sum[b] += r["flame_length"][b]
        toa_sum[b] += r["time_of_arrival"][b] / 1440.0

    with np.errstate(invalid="ignore", divide="ignore"):
        rasters = {
            "BurnProb": (n_b / len(run_dirs)).astype("float32"),
            "AvgFlameLength": np.where(n_b > 0, fl_sum / n_b, np.nan).astype("float32"),
            "AvgTime": np.where(n_b > 0, toa_sum / n_b, np.nan).astype("float32"),
            "n_burned": n_b,
        }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, a in rasters.items():
        fp = out_dir / f"{name}.tif"
        prof = {**profile, "dtype": str(a.dtype),
                "nodata": None if a.dtype.kind == "i" else np.nan}
        with rasterio.open(fp, "w", **prof) as dst:
            dst.write(a, 1)
        paths[name] = fp
    if verbose:
        print(f"[aggregate_realizations] {len(run_dirs)} realizations -> {out_dir}")
    return paths
