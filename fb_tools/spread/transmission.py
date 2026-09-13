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
