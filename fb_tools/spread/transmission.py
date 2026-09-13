"""
Treatment-to-value transmission from pyretechnics realizations.

Answers the *location* question — is a treatment in the line of fire between
where fires start and where structures are? — one realization at a time and
then over the ensemble.

The **treatment shadow** of a realization gives, for every burned cell, the
share of the fire that reached it having passed through a treatment. It is
traced on the arrival-time raster: each cell receives fire from its
earlier-arriving neighbours in proportion to the arrival-time gradient, so
the share propagates forward in time from the treated cells. A *value block*
(a 1 km cell holding structures) with a high share was reached by fire that
had already passed through a treatment; its *lead time* is the hours between
the fire entering the treatment and reaching the block. Cells lit by embers
are local minima of arrival time and root their own paths, so ember-driven
transmission past a treatment is counted as *not* intercepted.

Engine-free.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .pyre_outputs import read_realization

SHADOW_LAYER = "treatment_shadow"


# ── values: structure blocks ──────────────────────────────────────────────────

def structure_blocks(points, crs, bounds, *, block_m=1000.0, min_count=10, flag_col=None):
    """
    Grid the structure points into square blocks and keep the populated ones.

    Parameters
    ----------
    points : GeoDataFrame
        Structure locations (any CRS).
    crs : CRS
        Raster CRS the blocks are built in.
    bounds : tuple
        ``(left, bottom, right, top)`` of the landscape, in ``crs``.
    block_m : float
        Block edge length.
    min_count : int
        Blocks with fewer structures are dropped.
    flag_col : str, optional
        Boolean column counted separately (e.g. ``at_risk``).

    Returns
    -------
    GeoDataFrame
        ``block_id`` (1-based), ``n_structures``, ``n_<flag_col>``, square geometry.
    """
    import geopandas as gpd
    from shapely.geometry import box

    pts = points.to_crs(crs)
    left, bottom, right, top = bounds
    pts = pts.cx[left:right, bottom:top]
    bi = np.floor((top - pts.geometry.y.values) / block_m).astype(int)
    bj = np.floor((pts.geometry.x.values - left) / block_m).astype(int)
    df = pd.DataFrame({"bi": bi, "bj": bj})
    if flag_col is not None:
        df[flag_col] = pts[flag_col].fillna(False).astype(bool).values
    g = df.groupby(["bi", "bj"]).agg(n_structures=("bi", "size"),
                                     **({f"n_{flag_col}": (flag_col, "sum")} if flag_col else {}))
    g = g[g.n_structures >= min_count].reset_index()
    geoms = [box(left + j * block_m, top - (i + 1) * block_m, left + (j + 1) * block_m, top - i * block_m)
             for i, j in zip(g.bi, g.bj)]
    out = gpd.GeoDataFrame(g.drop(columns=["bi", "bj"]), geometry=geoms, crs=crs)
    out.insert(0, "block_id", np.arange(1, len(out) + 1))
    print(f"  [structure_blocks] {len(pts):,} structures -> {len(out)} blocks of {block_m / 1e3:.0f} km "
          f"with >= {min_count} structures")
    return out


def zone_id_raster(zones, transform, shape, id_col, *, all_touched=False):
    """Int32 raster of ``id_col`` values (0 where no zone). Later zones win where they overlap."""
    from rasterio.features import rasterize
    return rasterize(((g, int(i)) for g, i in zip(zones.geometry, zones[id_col])), out_shape=shape,
                     transform=transform, all_touched=all_touched, dtype="int32", fill=0)


# ── per-realization shadow ────────────────────────────────────────────────────

def treatment_shadow(toa, treated, *, power=1.0):
    """
    Share of the fire reaching each cell that had passed through a treatment.

    Every burned cell receives fire from its earlier-arriving neighbours in
    proportion to the arrival-time gradient toward each (a multiple-flow-
    direction rule, as in hydrology: the single steepest neighbour is
    lattice-biased). Treated cells carry a share of 1; the shares propagate
    forward in arrival time by one triangular solve. Cells with no earlier
    neighbour — the ignition, ember landings, level-set wrinkles — root their
    own paths and receive 0.

    Parameters
    ----------
    toa : 2-D array
        Arrival time (minutes), NaN outside the burn.
    treated : 2-D bool array
        Treatment cells.
    power : float
        Exponent on the gradient weights (1 = proportional).

    Returns
    -------
    2-D float32 array
        0–1 inside the burn, 0 outside.
    """
    from scipy.sparse import csr_matrix, identity
    from scipy.sparse.linalg import spsolve_triangular

    burned = np.isfinite(toa)
    H, W = toa.shape
    out = np.zeros((H, W), dtype="float32")
    seed = treated & burned
    if not seed.any():
        return out
    n = int(burned.sum())
    t = np.where(burned, toa, np.inf)
    rank = np.full((H, W), -1, dtype=np.int64)              # cells numbered in arrival order
    rank[burned] = np.argsort(np.argsort(toa[burned], kind="stable"), kind="stable")
    rows, cols, vals = [], [], []
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        a, a2 = slice(max(dr, 0), H + min(dr, 0)), slice(max(-dr, 0), H + min(-dr, 0))
        k, k2 = slice(max(dc, 0), W + min(dc, 0)), slice(max(-dc, 0), W + min(-dc, 0))
        ok = burned[a, k] & burned[a2, k2] & (t[a2, k2] < t[a, k])
        with np.errstate(invalid="ignore"):
            w = ((t[a, k] - t[a2, k2]) / np.hypot(dr, dc)) ** power
        rows.append(rank[a, k][ok]); cols.append(rank[a2, k2][ok]); vals.append(w[ok])
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    s = np.zeros(n)
    s[rank[seed]] = 1.0
    keep = s[rows] == 0                                       # treated cells take no inflow: their share is 1
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    inflow = csr_matrix((vals, (rows, cols)), shape=(n, n))   # strictly lower triangular in arrival order
    rs = np.asarray(inflow.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    inflow.data /= np.repeat(rs, np.diff(inflow.indptr))
    x = spsolve_triangular((identity(n, format="csr") - inflow).tocsr(), s, lower=True, unit_diagonal=True)
    out[burned] = np.clip(x[rank[burned]], 0.0, 1.0)
    return out


def _min_by_zone(values, zone_ids, n_zones):
    """Minimum of ``values`` per zone id (1..n_zones); NaN where the zone has no finite value."""
    out = np.full(n_zones + 1, np.inf)
    m = np.isfinite(values) & (zone_ids > 0)
    np.minimum.at(out, zone_ids[m], values[m])
    out[~np.isfinite(out)] = np.nan
    return out[1:]


def _count_by_zone(mask, zone_ids, n_zones):
    return np.bincount(zone_ids[mask & (zone_ids > 0)], minlength=n_zones + 1)[1:]


def realization_transmission(run_dir, *, treated, block_ids, atom_ids=None, write_shadow=True):
    """
    Treatment shadow and value-block arrivals for one realization.

    Parameters
    ----------
    run_dir : Path
    treated : 2-D bool array
        Treatment cells on the run grid.
    block_ids : 2-D int32 array
        :func:`zone_id_raster` of the value blocks (0 = none).
    atom_ids : 2-D int32 array, optional
        :func:`zone_id_raster` of the treatment atoms, for per-atom arrivals.
    write_shadow : bool
        Write ``treatment_shadow.tif`` (float32, 0–1) beside the run's rasters.

    Returns
    -------
    (dict, DataFrame, DataFrame)
        Fire-level summary; one row per block the fire reached
        (``block_id, arrival_h, shadow_frac, lead_h``); one row per atom
        reached (``atom_id, arrival_h``), empty if ``atom_ids`` is None.
    """
    import rasterio

    run_dir = Path(run_dir)
    r = read_realization(run_dir, layers=("time_of_arrival",))
    man, toa = r["manifest"], r["time_of_arrival"]
    burned = np.isfinite(toa)
    cell_ha = r["res_m"] ** 2 / 1e4
    keys = {k: man.get(k) for k in ("arm", "ignition_id", "episode_id", "condition_id")}

    shadow = treatment_shadow(toa, treated)
    hit = burned & treated
    t_trt = float(np.nanmin(toa[hit]) / 60.0) if hit.any() else np.nan

    n_blocks = int(block_ids.max())
    arr = _min_by_zone(toa, block_ids, n_blocks) / 60.0
    arr_sh = _min_by_zone(np.where(shadow >= 0.5, toa, np.nan), block_ids, n_blocks) / 60.0   # first mostly-shadowed arrival
    n_b = _count_by_zone(burned, block_ids, n_blocks)
    sum_s = np.bincount(block_ids[burned], weights=shadow[burned], minlength=n_blocks + 1)[1:]
    reached = np.where(n_b > 0)[0]
    blocks = pd.DataFrame({**keys, "block_id": reached + 1, "arrival_h": arr[reached],
                           "shadow_frac": sum_s[reached] / n_b[reached],   # share of the fire in the block that came through a treatment
                           "lead_h": arr_sh[reached] - t_trt})             # NaN where no cell of the block is mostly shadowed

    atoms = pd.DataFrame(columns=[*keys, "atom_id", "arrival_h"])
    if atom_ids is not None:
        n_a = int(atom_ids.max())
        a_arr = _min_by_zone(toa, atom_ids, n_a) / 60.0
        got = np.where(np.isfinite(a_arr))[0]
        atoms = pd.DataFrame({**keys, "atom_id": got + 1, "arrival_h": a_arr[got]})

    summary = {**keys, "burned_ha": round(float(burned.sum() * cell_ha), 2),
               "treated_ha_burned": round(float(hit.sum() * cell_ha), 2),
               "trt_arrival_h": t_trt,
               "shadow_ha": round(float(shadow.sum() * cell_ha), 2),       # expected area burned through a treatment
               "n_blocks_reached": int(len(reached)),
               "n_blocks_intercepted": int((blocks.shadow_frac >= 0.5).sum()),
               "n_atoms_reached": int(len(atoms))}
    if write_shadow:
        prof = dict(driver="GTiff", height=toa.shape[0], width=toa.shape[1], count=1, dtype="float32",
                    crs=r["crs"], transform=r["transform"], compress="deflate", nodata=None)
        with rasterio.open(run_dir / f"{SHADOW_LAYER}.tif", "w", **prof) as dst:
            dst.write(shadow, 1)
    return summary, blocks, atoms


def transmission_batch(results, **kwargs):
    """
    :func:`realization_transmission` over a ``run_batch`` result (rows with ``status == "ok"``).

    Returns
    -------
    (DataFrame, DataFrame, DataFrame)
        Fire-level summaries, block rows, atom rows.
    """
    fires, blocks, atoms = [], [], []
    for _, row in results[results.status == "ok"].iterrows():
        s, b, a = realization_transmission(row["out_dir"], **kwargs)
        fires.append(s)
        blocks.append(b)
        atoms.append(a)
    return (pd.DataFrame(fires), pd.concat(blocks, ignore_index=True),
            pd.concat(atoms, ignore_index=True))


def summarize_blocks(block_rows, blocks, *, n_fires, intercept_frac=0.5):
    """
    Per-block exposure and interception over the ensemble.

    Parameters
    ----------
    block_rows : DataFrame
        Concatenated block rows from :func:`realization_transmission`.
    blocks : GeoDataFrame
        From :func:`structure_blocks`.
    n_fires : int
        Realizations in the ensemble (for ``p_reached``).
    intercept_frac : float
        A block counts as intercepted when at least this fraction of its
        burned cells lie in the treatment shadow.

    Returns
    -------
    GeoDataFrame
        ``blocks`` plus ``n_reached, p_reached, n_intercepted, p_intercepted``
        (share of the fires that reached the block), ``lead_h_median``,
        ``arrival_h_median``.
    """
    g = block_rows.assign(intercepted=block_rows.shadow_frac >= intercept_frac).groupby("block_id")
    s = pd.DataFrame({"n_reached": g.size(), "n_intercepted": g.intercepted.sum(),
                      "arrival_h_median": g.arrival_h.median(),
                      "lead_h_median": g.lead_h.median()})
    s["p_reached"] = s.n_reached / n_fires
    s["p_intercepted"] = s.n_intercepted / s.n_reached
    out = blocks.merge(s, left_on="block_id", right_index=True, how="left")
    out[["n_reached", "n_intercepted"]] = out[["n_reached", "n_intercepted"]].fillna(0).astype(int)
    out["p_reached"] = out.p_reached.fillna(0.0)
    return out


def shadow_probability(run_dirs, out_dir, *, verbose=True):
    """
    Ensemble rasters from the per-run shadows: ``ShadowProb`` (expected share
    of realizations in which fire reached the cell through a treatment) and
    ``ShadowGivenBurn`` (same, conditional on the cell burning).
    """
    import rasterio

    run_dirs = [Path(p) for p in run_dirs]
    n_s = n_b = None
    for rd in run_dirs:
        with rasterio.open(rd / "time_of_arrival.tif") as s:
            b = np.isfinite(s.read(1))
            profile = {**s.profile, "dtype": "float32", "nodata": np.nan, "compress": "deflate"}
        with rasterio.open(rd / f"{SHADOW_LAYER}.tif") as s:
            sh = s.read(1)
        if n_s is None:
            n_s, n_b = np.zeros(b.shape, np.float64), np.zeros(b.shape, np.int32)
        n_s += sh
        n_b += b
    n = len(run_dirs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rasters = {"ShadowProb": (n_s / n).astype("float32"),
                   "ShadowGivenBurn": np.where(n_b > 0, n_s / n_b, np.nan).astype("float32")}
    paths = {}
    for name, a in rasters.items():
        paths[name] = out_dir / f"{name}.tif"
        with rasterio.open(paths[name], "w", **profile) as dst:
            dst.write(a, 1)
    if verbose:
        print(f"[shadow_probability] {n} realizations -> {out_dir} ({', '.join(rasters)})")
    return paths


# ── near field: zones and the lee profile ─────────────────────────────────────
#
# A paired delta can only be read on ground the two arms actually differ on and
# on the cells around it. Polygons rasterized ``all_touched`` leak into their own
# collar when the feature is a few cells wide (half of the Upper Arkansas COSWAP
# cells fell inside their 500 m collar mask), so zones here are built from the
# cells that differ between the arms and Euclidean distance from them.

CHANGE_LAYERS = ("fuel_model", "canopy_cover", "canopy_height", "canopy_base_height", "canopy_bulk_density")
DISTANCE_EDGES_M = (100, 250, 500, 1000, 2000, 4000, 8000)
#: Fireline-intensity classes of the Andrews & Rothermel (1982) hauling chart, kW/m: below 350 hand
#: crews can work the head; below 1,700 equipment; below 3,500 indirect attack only; above, no control.
INTENSITY_EDGES_KW_M = (350.0, 1700.0, 3500.0)
INTENSITY_LABELS = ("hand", "equipment", "indirect", "none")


def _static(s, layers):
    if isinstance(s, dict):
        return s
    import rasterio
    out = {}
    for k in layers:
        with rasterio.open(Path(s) / f"{k}.tif") as src:
            out[k] = src.read(1)
    return out


def changed_cells(static_a, static_b, layers=CHANGE_LAYERS):
    """
    Cells whose fuel or canopy differ between two arms — the treatment as the
    engine sees it, and the only footprint a paired delta can be read on.

    Parameters
    ----------
    static_a, static_b : dict or Path
        ``{layer: 2-D array}`` (:func:`fb_tools.models.pyre.load_static_layers`)
        or a ``destack_lcp`` output directory.
    layers : tuple of str
        Layers compared.

    Returns
    -------
    2-D bool array
    """
    a, b = _static(static_a, layers), _static(static_b, layers)
    out = np.zeros(np.shape(a[layers[0]]), bool)
    for k in layers:
        if np.shape(a[k]) != np.shape(b[k]):
            raise ValueError(f"{k}: arms differ in shape {np.shape(a[k])} vs {np.shape(b[k])}")
        out |= ~np.isclose(a[k], b[k], equal_nan=True)
    return out


def distance_bands(changed, res_m, edges=DISTANCE_EDGES_M):
    """
    Band every cell by Euclidean distance from the nearest changed cell.

    Parameters
    ----------
    changed : 2-D bool array
    res_m : float
        Cell size.
    edges : tuple of float
        Increasing outer edges of the bands, metres.

    Returns
    -------
    (2-D int16 array, list of str)
        ``0`` inside the changed cells, ``k`` where ``edges[k-1] < d <= edges[k]``
        (``edges[-1]`` taken as 0), ``-1`` beyond the last edge; and the band
        labels (``"in", "0-100", "100-250", ...``).
    """
    from scipy.ndimage import distance_transform_edt
    changed = np.asarray(changed, bool)
    if not changed.any():
        raise ValueError("distance_bands: no changed cells")
    edges = tuple(float(e) for e in edges)
    if any(b <= a for a, b in zip(edges, edges[1:])) or edges[0] <= 0:
        raise ValueError(f"edges must be positive and increasing: {edges}")
    d = distance_transform_edt(~changed) * float(res_m)
    band = np.digitize(d, [0.0, *edges], right=True).astype(np.int16)
    band[band == len(edges) + 1] = -1
    labels = ["in"] + [f"{lo:g}-{hi:g}" for lo, hi in zip((0.0, *edges[:-1]), edges)]
    return band, labels


def near_field_zones(changed, res_m, *, collar_m=500.0):
    """
    ``{"trt": the changed cells, "collar": cells within collar_m of one, excluding them}``
    — boolean zone masks for :func:`~fb_tools.spread.pyre_outputs.summarize_realization`.
    """
    band, _ = distance_bands(changed, res_m, edges=(collar_m,))
    return {"trt": np.asarray(changed, bool), "collar": band == 1}


def _read_shadow(run_dir):
    import rasterio
    with rasterio.open(Path(run_dir) / f"{SHADOW_LAYER}.tif") as src:
        return src.read(1)


def lee_profile(baseline_dir, treated_dir, *, bands, labels, shadow_dir=None, shadow_frac=0.5,
                min_cells=20, intensity_edges=INTENSITY_EDGES_KW_M, intensity_labels=INTENSITY_LABELS):
    """
    Paired near-field deltas of one realization by distance from the treatment,
    split by whether the fire reached the cell *through* the treatment.

    Cells burned in both arms are compared cell by cell. The baseline run's
    treatment shadow (``treatment_shadow.tif`` from
    :func:`realization_transmission`) puts a cell in the **lee** when at least
    ``shadow_frac`` of the fire reaching it had crossed the treatment; every
    other burned cell is the control the lee is read against. Arrival-time
    zone statistics read the *first* arrival, which is the windward edge and
    is unaffected by construction; this profile is where the carried delay
    and the recovery distance live.

    Parameters
    ----------
    baseline_dir, treated_dir : Path
        The two arms of one exactly paired realization.
    bands, labels : from :func:`distance_bands`.
    shadow_dir : Path, optional
        Run holding ``treatment_shadow.tif``; default ``baseline_dir``.
    shadow_frac : float
    min_cells : int
        Rows with fewer co-burned cells are dropped.
    intensity_edges, intensity_labels : hauling-chart classes.

    Returns
    -------
    DataFrame
        One row per (``lee``, ``band``): ``n_both`` co-burned cells, ``n_either``,
        ``frac_disagree`` (burned in one arm only), ``fi_bg`` mean baseline
        intensity (kW/m), ``d_fi`` mean treated − baseline, ``crown_bg``,
        ``d_crown`` (crown-fire share), ``d_toa_med`` / ``d_toa_p90`` (hours,
        positive = later on the treated landscape), ``frac_delayed_30m``,
        ``frac_delayed_1h``, ``frac_delayed_6h``, ``frac_earlier_1h``, and the
        intensity-class shares ``cls_<label>_bg`` / ``cls_<label>_tr``.
    """
    layers = ("time_of_arrival", "fireline_intensity", "fire_type")
    b, t = read_realization(baseline_dir, layers=layers), read_realization(treated_dir, layers=layers)
    toa_b, toa_t = b["time_of_arrival"], t["time_of_arrival"]
    if toa_b.shape != toa_t.shape or toa_b.shape != np.shape(bands):
        raise ValueError("lee_profile: arms and bands must share one grid")
    shadow = _read_shadow(shadow_dir or baseline_dir)
    keys = {k: b["manifest"].get(k) for k in ("ignition_id", "episode_id", "condition_id")}
    keys.update(baseline=b["manifest"].get("arm"), treated=t["manifest"].get("arm"))
    burned_b, burned_t = np.isfinite(toa_b), np.isfinite(toa_t)
    both, either = burned_b & burned_t, burned_b | burned_t
    lee = shadow >= shadow_frac
    fi_b, fi_t = b["fireline_intensity"], t["fireline_intensity"]
    cr_b, cr_t = b["fire_type"] >= 2, t["fire_type"] >= 2
    edges = list(intensity_edges)
    rows = []
    for in_lee, lmask in ((True, lee), (False, ~lee)):
        for k, lab in enumerate(labels):
            m = (bands == k) & lmask
            mb = m & both
            n = int(mb.sum())
            if n < min_cells:
                continue
            d_h = (toa_t[mb] - toa_b[mb]) / 60.0
            n_e = int((m & either).sum())
            row = {**keys, "lee": in_lee, "band": lab, "n_both": n, "n_either": n_e,
                   "frac_disagree": float(n_e - n) / n_e,
                   "fi_bg": float(fi_b[mb].mean()), "d_fi": float((fi_t[mb] - fi_b[mb]).mean()),
                   "crown_bg": float(cr_b[mb].mean()), "d_crown": float(cr_t[mb].mean() - cr_b[mb].mean()),
                   "d_toa_med": float(np.median(d_h)), "d_toa_p90": float(np.quantile(d_h, 0.9)),
                   "frac_delayed_30m": float((d_h > 0.5).mean()), "frac_delayed_1h": float((d_h > 1.0).mean()),
                   "frac_delayed_6h": float((d_h > 6.0).mean()), "frac_earlier_1h": float((d_h < -1.0).mean())}
            for arm, fi in (("bg", fi_b), ("tr", fi_t)):
                share = np.bincount(np.digitize(fi[mb], edges), minlength=len(edges) + 1) / n
                row.update({f"cls_{c}_{arm}": float(s) for c, s in zip(intensity_labels, share)})
            rows.append(row)
    return pd.DataFrame(rows)


def lee_profile_batch(results, *, baseline, treated, bands, labels, **kwargs):
    """
    :func:`lee_profile` over every exactly paired realization in a ``run_batch``
    result (rows with ``status == "ok"`` for both arms).

    Returns
    -------
    DataFrame
        Concatenated per-pair rows.
    """
    key = ["condition_id", "ignition_id", "episode_id"]
    ok = results[results.status == "ok"]
    b = ok[ok.arm == baseline].set_index(key)["out_dir"]
    t = ok[ok.arm == treated].set_index(key)["out_dir"]
    shared = b.index.intersection(t.index)
    out = [lee_profile(b[k], t[k], bands=bands, labels=labels, **kwargs) for k in shared]
    out = [o for o in out if len(o)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize_lee_profile(rows, labels, *, min_pairs=3):
    """
    Collapse per-pair profile rows into one table: per (``lee``, ``band``) the
    number of pairs, the co-burned cells, the **median across pairs** of every
    per-pair statistic, and the **cell-weighted** intensity-class shares.

    Bands are ordered as in ``labels``; groups seen in fewer than ``min_pairs``
    pairs are dropped.
    """
    if rows.empty:
        return pd.DataFrame()
    cls_cols = [c for c in rows.columns if c.startswith("cls_")]
    med_cols = [c for c in ("frac_disagree", "fi_bg", "d_fi", "crown_bg", "d_crown", "d_toa_med", "d_toa_p90",
                            "frac_delayed_30m", "frac_delayed_1h", "frac_delayed_6h", "frac_earlier_1h") if c in rows]
    g = rows.groupby(["lee", "band"], sort=False)
    out = pd.concat([g.size().rename("pairs"), g.n_both.sum().rename("n_cells"), g[med_cols].median(),
                     g.apply(lambda d: pd.Series({c: float((d[c] * d.n_both).sum() / d.n_both.sum()) for c in cls_cols}),
                             include_groups=False)], axis=1)
    out = out[out.pairs >= min_pairs].reset_index()
    out["band"] = pd.Categorical(out.band, categories=labels, ordered=True)
    return out.sort_values(["lee", "band"], ascending=[False, True]).reset_index(drop=True)


def near_field_headlines(summary, *, recovery_tol=0.10, carried_band="1000-2000"):
    """
    The two numbers the near field reduces to, plus the hauling-chart shift.

    Parameters
    ----------
    summary : DataFrame
        From :func:`summarize_lee_profile`.
    recovery_tol : float
        The intensity effect has *recovered* in the first lee band outside the
        treatment where ``|d_fi| <= recovery_tol * fi_bg``.
    carried_band : str
        Band whose lee arrival delay is reported as the *carried delay*
        (the farthest available lee band if absent).

    Returns
    -------
    dict
        ``recovery_band`` (label, or None if the effect never recovers within
        the bands), ``carried_delay_h``, ``carried_delay_p90_h``,
        ``carried_frac_delayed_1h``, ``carried_band``, ``in_d_fi``,
        ``in_d_crown``, ``in_cls_hand_bg`` / ``in_cls_hand_tr`` (share of treated
        cells hand crews could work), ``in_cls_uncontrollable_bg`` / ``_tr``
        (indirect + none).
    """
    lee = summary[summary.lee].set_index("band")
    out = {"recovery_band": None, "carried_band": None, "carried_delay_h": np.nan,
           "carried_delay_p90_h": np.nan, "carried_frac_delayed_1h": np.nan}
    for lab, r in lee.iterrows():
        if lab != "in" and abs(r.d_fi) <= recovery_tol * abs(r.fi_bg):
            out["recovery_band"] = str(lab)
            break
    outside = lee.drop(index="in", errors="ignore")
    if len(outside):
        lab = carried_band if carried_band in outside.index else outside.index[-1]
        r = outside.loc[lab]
        out.update(carried_band=str(lab), carried_delay_h=float(r.d_toa_med), carried_delay_p90_h=float(r.d_toa_p90),
                   carried_frac_delayed_1h=float(r.frac_delayed_1h))
    if "in" in lee.index:
        r = lee.loc["in"]
        out.update(in_d_fi=float(r.d_fi), in_d_crown=float(r.d_crown),
                   in_cls_hand_bg=float(r.cls_hand_bg), in_cls_hand_tr=float(r.cls_hand_tr),
                   in_cls_uncontrollable_bg=float(r.cls_indirect_bg + r.cls_none_bg),
                   in_cls_uncontrollable_tr=float(r.cls_indirect_tr + r.cls_none_tr))
    return out
