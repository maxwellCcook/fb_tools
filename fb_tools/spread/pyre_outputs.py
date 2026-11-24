"""
Read, summarize, and aggregate pyretechnics realizations.

Per realization ``run_realization`` writes ``time_of_arrival`` (minutes),
``fire_type``, ``spread_rate`` (m/min), ``fireline_intensity`` (kW/m) and
``flame_length`` (m), NaN outside the burn. This module turns those into
per-realization metrics (growth, arrival at zones, firebrand production),
exactly paired deltas, and the ensemble rasters: burn probability, arrival
probability by day, mean arrival, mean flame length, expected firebrands.
Engine-free.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY_MARKS = (1, 2, 3, 5, 7, 14, 21)

#: What one realization writes (``models.pyre`` imports this; the first entry is spread-only).
REALIZATION_LAYERS = ("time_of_arrival", "fire_type", "spread_rate", "spread_direction",
                      "fireline_intensity", "flame_length")

DEFAULT_FIREBRAND_FLUX = 5e-11        # pyretechnics' ``firebrands_per_unit_heat``


def rasterize_mask(geoms, transform, shape, *, all_touched=True):
    """Boolean mask of the cells touched by ``geoms`` (an iterable of shapely geometries)."""
    from rasterio.features import rasterize
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                     all_touched=all_touched, dtype="uint8").astype(bool)


def _flux(manifest, override=None):
    """Firebrand flux for a run: explicit override, else its ``spot_config``, else the engine default."""
    return override or (manifest.get("spot_config") or {}).get("firebrands_per_unit_heat", DEFAULT_FIREBRAND_FLUX)


def raster_grid(fp):
    """``{shape, transform, crs}`` of a raster, from its header."""
    import rasterio
    with rasterio.open(fp) as src:
        return {"shape": src.shape, "transform": src.transform, "crs": src.crs}


def assert_same_grid(ref, other, label="raster", atol=1e-4):
    """
    Raise unless two grids match in shape, transform and CRS.

    Parameters
    ----------
    ref, other : dict
        ``shape``, ``transform``, ``crs`` (:func:`raster_grid` or
        :func:`read_realization`). A grid with no ``transform`` (a bare array)
        is checked on shape only.
    label : str
        Names ``other`` in the error.
    atol : float
        Tolerance on the affine coefficients, CRS units.

    Raises
    ------
    ValueError
    """
    if tuple(ref["shape"]) != tuple(other["shape"]):
        raise ValueError(f"{label} is not on the same grid: shape {tuple(other['shape'])} != {tuple(ref['shape'])}")
    if ref.get("transform") is None or other.get("transform") is None:
        return
    a, b = np.asarray(ref["transform"])[:6], np.asarray(other["transform"])[:6]
    if not np.allclose(a, b, rtol=0.0, atol=atol):
        raise ValueError(f"{label} is not on the same grid: transform {tuple(np.round(b, 4))} != {tuple(np.round(a, 4))}")
    if ref.get("crs") is not None and other.get("crs") is not None and ref["crs"] != other["crs"]:
        raise ValueError(f"{label} is not on the same grid: CRS {other['crs']} != {ref['crs']}")


def read_realization(run_dir, layers=REALIZATION_LAYERS):
    """
    Load a realization's rasters and manifest.

    Returns
    -------
    dict
        ``{layer: 2-D float32 array}`` plus ``manifest`` (dict), ``shape``,
        ``transform``, ``crs`` and ``res_m``.
    """
    import rasterio
    run_dir = Path(run_dir)
    out = {"manifest": json.loads((run_dir / "manifest.json").read_text())}
    gone = [n for n in layers if n in out["manifest"].get("pruned_layers", [])]
    if gone:
        raise FileNotFoundError(f"{run_dir}: {gone} removed by prune_realizations; re-run with overwrite=True to restore")
    for name in layers:
        with rasterio.open(run_dir / f"{name}.tif") as src:
            out[name] = src.read(1)
            out["shape"], out["transform"], out["crs"] = src.shape, src.transform, src.crs
    out["res_m"] = abs(out["transform"].a)
    return out


def prune_realizations(run_dirs, keep=("time_of_arrival",), verbose=True):
    """
    Delete realization rasters that are no longer needed, once the summaries
    and ensemble rasters have been taken from them.

    Parameters
    ----------
    run_dirs : iterable of Path
    keep : tuple of str
        ``REALIZATION_LAYERS`` to keep. Files that are not realization layers
        (manifest, treatment shadow) are never touched.

    Returns
    -------
    int
        Bytes freed. Each manifest gains ``pruned_layers``; the run still
        counts as done for ``run_batch``, and :func:`read_realization` names
        the fix when a pruned layer is asked for.
    """
    freed = n = 0
    for rd in map(Path, run_dirs):
        man_fp = rd / "manifest.json"
        if not man_fp.exists():
            continue
        man = json.loads(man_fp.read_text())
        gone = set(man.get("pruned_layers", []))
        for name in REALIZATION_LAYERS:
            fp = rd / f"{name}.tif"
            if name not in keep and fp.exists():
                freed += fp.stat().st_size
                fp.unlink()
                gone.add(name)
        man["pruned_layers"] = sorted(gone)
        man_fp.write_text(json.dumps(man, indent=2, default=str))
        n += 1
    if verbose:
        print(f"[prune_realizations] {n} runs, kept {list(keep)}, freed {freed / 1e6:,.0f} MB")
    return freed


def firebrand_production(fireline_intensity, spread_rate, cell_area_m2, *, slope=None,
                         firebrands_per_unit_heat=DEFAULT_FIREBRAND_FLUX):
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
    """
    Rasterize zone polygons -> ``{zone_id: bool mask}`` (all_touched). A dict of
    boolean masks is passed through — use masks for features a few cells wide,
    where an all_touched polygon leaks into its own collar
    (:func:`~fb_tools.spread.transmission.near_field_zones`).
    """
    if isinstance(zones, dict):
        return {k: np.asarray(v, bool) for k, v in zones.items()}
    return {zid: rasterize_mask(sub.geometry, transform, shape) for zid, sub in zones.groupby(id_col)}


def summarize_realization(run_dir, *, zones=None, id_col="zone", day_marks=DAY_MARKS,
                          firebrands_per_unit_heat=None, slope=None):
    """
    Per-realization metrics.

    Parameters
    ----------
    run_dir : Path
    zones : GeoDataFrame or dict, optional
        Reporting polygons in the raster CRS (treatment units, SFAs, collars),
        or ``{zone_id: 2-D bool mask}`` on the run grid.
    id_col : str
    day_marks : tuple of int
        Days at which burned area is read off ``time_of_arrival``. Marks past
        the run's ``duration_h`` are dropped.
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
        reached), ``{zone}_frac_burned``, ``{zone}_fi_mean``, ``{zone}_crown_frac``,
        ``{zone}_firebrands``.
    """
    r = read_realization(run_dir)
    man, toa, ft = r["manifest"], r["time_of_arrival"], r["fire_type"]
    burned = np.isfinite(toa)
    cell_ha = r["res_m"] ** 2 / 1e4
    flux = _flux(man, firebrands_per_unit_heat)
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
    max_h = float(man.get("duration_h", np.inf))
    for d in day_marks:
        if d * 24.0 <= max_h:
            out[f"area_d{d}_ha"] = round(float((toa <= d * 1440).sum() * cell_ha), 3)

    if zones is not None:
        for zid, m in zone_masks(zones, r["transform"], toa.shape, id_col).items():
            zb = burned & m
            out[f"{zid}_arrival_h"] = round(float(np.nanmin(toa[m]) / 60.0), 1) if zb.any() else np.nan
            out[f"{zid}_frac_burned"] = round(float(zb.sum() / max(m.sum(), 1)), 4)
            out[f"{zid}_fi_mean"] = round(float(r["fireline_intensity"][zb].mean()), 0) if zb.any() else np.nan
            out[f"{zid}_crown_frac"] = round(float((ft[zb] >= 2).mean()), 4) if zb.any() else np.nan
            out[f"{zid}_firebrands"] = float(fb[m].sum())
    return out


def suppression_summary(run_dir, changed=None):
    """
    What rule S1 did in one stepped run (``run_realization(..., suppression=...)``).

    Parameters
    ----------
    run_dir : Path
    changed : 2-D bool array, optional
        Cells whose fuels differ between the arms (:func:`~fb_tools.spread.transmission.changed_cells`);
        gives the share of the held line that lies on treated ground — the anchor statistic.

    Returns
    -------
    dict
        ``held_ha``, ``held_in_changed_ha``, ``held_in_changed_share``, ``containment_target``,
        ``frozen_sectors``, ``first_hold_h`` (NaN if no line was held), ``target_d{1,3}`` (target at
        the end of days 1 and 3), ``sdi_known_share`` plus the run keys. Empty except the keys for a
        run without suppression.
    """
    import rasterio
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    out = {k: man.get(k) for k in ("arm", "ignition_id", "episode_id", "condition_id")}
    if not man.get("suppression"):
        return out
    with rasterio.open(run_dir / "held_line.tif") as src:
        held = src.read(1)
        cell_ha = abs(src.transform.a) ** 2 / 1e4
    h = np.isfinite(held)
    log = pd.DataFrame(man.get("suppression_log") or [])
    step_h = float(man.get("step_h") or 1.0)
    first = held[h].min() if h.any() else np.nan
    out.update({
        "held_ha": round(float(h.sum() * cell_ha), 3),
        "held_in_changed_ha": round(float((h & changed).sum() * cell_ha), 3) if changed is not None else np.nan,
        "held_in_changed_share": round(float((h & changed).sum() / h.sum()), 4) if changed is not None and h.any() else np.nan,
        "containment_target": man.get("containment_target"),
        "frozen_sectors": man.get("frozen_sectors"),
        "first_hold_h": round(float(first * step_h), 2) if h.any() else np.nan,
        "sdi_known_share": man.get("sdi_known_share"),
    })
    for d in (1, 3):
        at = log[log.hour <= d * 24] if len(log) else log
        out[f"target_d{d}"] = round(float(at.target.iloc[-1]), 4) if len(at) and "target" in at else np.nan
        out[f"held_ha_d{d}"] = round(float(np.isfinite(held[(held > 0) & (held * step_h <= d * 24)]).sum() * cell_ha), 3)
    return out


def suppression_batch(results, changed=None):
    """:func:`suppression_summary` over a ``run_batch`` result (rows with ``status == "ok"``)."""
    rows = [suppression_summary(r["out_dir"], changed=changed) for _, r in results.iterrows() if r["status"] == "ok"]
    return pd.DataFrame(rows)


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


def one_arm(run_dirs, arm=None):
    """
    Run directories of a single arm, with their manifests.

    Returns ``(run_dirs, manifests, arm)``. Raises ``ValueError`` if the runs
    span several arms and ``arm`` is not given, or if no run is in ``arm``.
    """
    run_dirs = [Path(p) for p in run_dirs]
    mans = [json.loads((rd / "manifest.json").read_text()) for rd in run_dirs]
    arms = sorted({str(m.get("arm")) for m in mans})
    if arm is not None:
        keep = [m.get("arm") == arm for m in mans]
        run_dirs = [rd for rd, k in zip(run_dirs, keep) if k]
        mans = [m for m, k in zip(mans, keep) if k]
        if not run_dirs:
            raise ValueError(f"no run in arm {arm!r}; arms present: {arms}")
    elif len(arms) > 1:
        raise ValueError(f"run_dirs hold arms {arms}; an ensemble is one arm — pass arm= or one arm's runs")
    return run_dirs, mans, arm if arm is not None else (arms[0] if arms else None)


def _run_weights(weights, ignition_ids):
    """Per-run weights from per-ignition ``weights``: each ignition's weight is split over its runs."""
    n = len(ignition_ids)
    if weights is None:
        return np.ones(n, dtype="float64"), float(n)
    from .bp import _resolve_weights
    if not isinstance(weights, (dict, pd.Series)):
        raise TypeError("weights must be a dict or Series keyed by ignition_id")
    if any(i is None for i in ignition_ids):
        raise ValueError("weights need an ignition_id in every run's manifest")
    counts = pd.Series(ignition_ids).value_counts()
    lost = [k for k, v in dict(weights).items() if v > 0 and k not in counts.index]
    if lost:
        raise ValueError(f"weighted ignition(s) with no run to aggregate: {lost[:5]}{' …' if len(lost) > 5 else ''}")
    w_ign = dict(zip(counts.index, _resolve_weights(weights, len(counts), keys=list(counts.index))))
    w = np.array([w_ign[i] / counts[i] for i in ignition_ids], dtype="float64")
    return w, float(w.sum())


def aggregate_realizations(run_dirs, out_dir, *, arm=None, weights=None, day_marks=DAY_MARKS, firebrands=True,
                           firebrands_per_unit_heat=None, verbose=True):
    """
    Collapse one arm's realizations into ensemble rasters, streaming one run at a time.

    Parameters
    ----------
    run_dirs : iterable of Path
        Realization directories (``run_batch`` rows with ``status == "ok"``).
    out_dir : Path
    arm : str, optional
        Arm to aggregate. Required when ``run_dirs`` hold more than one arm.
    weights : dict or pandas.Series, optional
        ``{ignition_id: w_i}``. Each ignition keeps its weight, split equally
        over the runs it has; default every run counts equally.
    day_marks : tuple of int
        Days ``d`` for which ``ArrivalProb_d{d}.tif`` = fraction of realizations
        that reached the cell by day ``d``. Marks past the longest run are dropped.
    firebrands : bool
        Also write ``Firebrands.tif``: expected firebrands produced per cell,
        averaged over realizations (:func:`firebrand_production`; 0 where unburned).
    firebrands_per_unit_heat : float, optional
        Flux override; default each run's ``spot_config`` value.

    Returns
    -------
    dict
        ``{name: Path}`` for ``BurnProb`` (fraction burned, 0 unburned),
        ``AvgFlameLength`` (m, NaN outside), ``AvgTime`` (**days**, NaN outside),
        ``n_burned`` (count of runs, unweighted), ``ArrivalProb_d{d}``, ``Firebrands``.

    Raises
    ------
    ValueError
        If the runs span several arms and ``arm`` is not given, if no run is in
        ``arm``, or if a weighted ignition has no run.
    """
    import rasterio
    run_dirs = [Path(p) for p in run_dirs]
    if not run_dirs:
        raise ValueError("aggregate_realizations needs at least one run directory")
    run_dirs, mans, arm = one_arm(run_dirs, arm)
    w_run, w_total = _run_weights(weights, [m.get("ignition_id") for m in mans])
    layers = ["time_of_arrival", "flame_length"] + (["fireline_intensity", "spread_rate"] if firebrands else [])
    sums = None
    max_h = 0.0
    for rd, w in zip(run_dirs, w_run):
        r = read_realization(rd, layers=layers)
        toa = r["time_of_arrival"]
        b = np.isfinite(toa)
        if sums is None:
            z = lambda dt: np.zeros(b.shape, dtype=dt)          # noqa: E731
            sums = {"n": z("int32"), "w": z("float64"), "fl": z("float64"), "toa": z("float64"), "fb": z("float64"),
                    "arr": {d: z("float64") for d in day_marks}}
            profile = dict(driver="GTiff", height=b.shape[0], width=b.shape[1], count=1,
                           crs=r["crs"], transform=r["transform"], compress="deflate")
            cell_m2 = r["res_m"] ** 2
            ref = r
        else:
            assert_same_grid(ref, r, label=str(rd))
        sums["n"] += b
        sums["w"] += w * b
        sums["fl"][b] += w * r["flame_length"][b]
        sums["toa"][b] += w * toa[b] / 1440.0
        for d, a in sums["arr"].items():
            a += w * (toa <= d * 1440.0)
        if firebrands:
            fb = firebrand_production(r["fireline_intensity"], r["spread_rate"], cell_m2,
                                      firebrands_per_unit_heat=_flux(r["manifest"], firebrands_per_unit_heat))
            sums["fb"][b] += w * fb[b]
        max_h = max(max_h, float(r["manifest"].get("duration_h", np.nanmax(toa) / 60.0 if b.any() else 0.0)))

    n_runs, n_b, w_b = len(run_dirs), sums["n"], sums["w"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rasters = {
            "BurnProb": (w_b / w_total).astype("float32"),
            "AvgFlameLength": np.where(w_b > 0, sums["fl"] / w_b, np.nan).astype("float32"),
            "AvgTime": np.where(w_b > 0, sums["toa"] / w_b, np.nan).astype("float32"),
            "n_burned": n_b,
        }
    for d, a in sums["arr"].items():
        if d * 24.0 <= max_h or d == min(day_marks):
            rasters[f"ArrivalProb_d{d}"] = (a / w_total).astype("float32")
    if firebrands:
        rasters["Firebrands"] = (sums["fb"] / w_total).astype("float32")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, a in rasters.items():
        fp = out_dir / f"{name}.tif"
        prof = {**profile, "dtype": str(a.dtype), "nodata": None if a.dtype.kind == "i" else np.nan}
        with rasterio.open(fp, "w", **prof) as dst:
            dst.write(a, 1)
        paths[name] = fp
    if verbose:
        print(f"[aggregate_realizations] {n_runs} realizations{'' if weights is None else ', weighted'} "
              f"(arm {arm}) -> {out_dir} ({', '.join(rasters)})")
    return paths


def edge_contact(run_dirs, *, edge_cells=3):
    """
    When each realization first touched the domain edge — the domain-adequacy
    diagnostic for spread runs (fires that reach the edge are truncated, not
    stopped, so their size and everything downwind of the edge is biased).

    Returns
    -------
    pandas.DataFrame
        One row per run: ``out_dir, burned_ha, edge_day`` (NaN if the fire
        never reached the outer ``edge_cells`` rows/columns).
    """
    import rasterio
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        with rasterio.open(rd / "time_of_arrival.tif") as src:
            toa = src.read(1)
            cell_ha = abs(src.transform.a) ** 2 / 1e4
        e = edge_cells
        border = np.concatenate([toa[:e].ravel(), toa[-e:].ravel(), toa[:, :e].ravel(), toa[:, -e:].ravel()])
        border = border[np.isfinite(border)]
        rows.append({"out_dir": str(rd), "burned_ha": float(np.isfinite(toa).sum() * cell_ha),
                     "edge_day": float(border.min() / 1440.0) if border.size else np.nan})
    return pd.DataFrame(rows)
