"""
pyretechnics engine wrapper: version pinning, cube assembly, and the paired
realization runner.

Runs in the ``fb_pyre`` env (see ``environment-pyre.yml``). ``import
pyretechnics`` is deferred to call time so ``import fb_tools`` works anywhere.
Design decisions are logged in ``docs/pyre_decisions.md``.
"""

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Version / physics-setting provenance ─────────────────────────────────────

PYRETECHNICS_VERSION = "2026.8.10"
PYRETECHNICS_PATCH_LABEL = "cfri1"          # CFRI build: crown gate is settable
STOCK_CROWN_COVER_THRESHOLD = 0.4

#: Layers ``spread_fire_with_phi_field`` reads from ``destack_lcp`` output.
STATIC_LAYERS = ("slope", "aspect", "fuel_model", "canopy_cover", "canopy_height",
                 "canopy_base_height", "canopy_bulk_density")

#: pyretechnics' own spotting defaults (``firebrands_per_unit_heat`` is the sweep axis).
SPOT_CENTRAL = {
    "firebrands_per_unit_heat": 5e-11,
    "downwind_distance_mean": 11.7,
    "fireline_intensity_exponent": 0.3,
    "wind_speed_exponent": 0.6,
    "downwind_variance_mean_ratio": 600.0,
    "crosswind_distance_stdev": 10.0,
    "decay_distance": 500.0,
}

OUTPUT_LAYERS = ("time_of_arrival", "fire_type", "spread_rate", "spread_direction",
                 "fireline_intensity", "flame_length")
BEHAVIOR_LAYERS = OUTPUT_LAYERS[1:]     # what burn_cells returns (no arrival time)

#: Engine fire-behavior options, stamped into every manifest.
BEHAVIOR_OPTS = {"use_wind_limit": True, "surface_lw_ratio_model": "behave", "crown_max_lw_ratio": 1e10}


def _installed_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("pyretechnics")
    except PackageNotFoundError:                                 # pragma: no cover
        return "unknown"


def _base_version(v):
    return str(v).split("+", 1)[0]


def require_pyretechnics(version=PYRETECHNICS_VERSION):
    """Import pyretechnics, asserting the pinned base version (``None`` skips)."""
    try:
        import pyretechnics
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "pyretechnics is not installed. Use the fb_pyre env:\n"
            "    conda env create -f environment-pyre.yml && conda activate fb_pyre\n"
            "    pip install -e . --no-deps\n"
            f'    pip install "pyretechnics=={PYRETECHNICS_VERSION}" --no-deps --no-build-isolation'
        ) from exc
    installed = _installed_version()
    if version is not None and _base_version(installed) != _base_version(version):
        raise RuntimeError(
            f"pyretechnics {installed!r} installed; analysis pinned to {version!r}. "
            "Crown-fire physics and the spotting API change across releases."
        )
    return pyretechnics


def is_patched_engine():
    """True on the CFRI build (``+cfri1``), where the crown gate is settable."""
    return _installed_version().endswith(f"+{PYRETECHNICS_PATCH_LABEL}")


def crown_cover_threshold():
    """Canopy-cover fraction above which the engine allows crown fire."""
    if not is_patched_engine():
        return STOCK_CROWN_COVER_THRESHOLD
    import pyretechnics.crown_fire as cf
    return float(cf.get_crown_cover_threshold())


def set_crown_cover_threshold(value):
    """
    Set the crown canopy-cover gate (0.4 = stock; 0.0 = Van Wagner as published).

    Process-global: workers set it per realization from the runs table.
    Raises ``RuntimeError`` on a stock build.
    """
    if not is_patched_engine():
        raise RuntimeError(
            f"pyretechnics {_installed_version()} hard-codes the crown gate at 0.4. "
            "Build the CFRI engine: python code/dev/pyre/build_patched_engine.py"
        )
    import pyretechnics.crown_fire as cf
    cf.set_crown_cover_threshold(float(value))


def pyre_version_info(strict=True):
    """Provenance stamp for a manifest: engine, patch, gate, python, numpy, scipy, env."""
    if strict:
        require_pyretechnics()
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:                                          # pragma: no cover
        scipy_version = None
    return {
        "pyretechnics_version": _installed_version(),
        "pyretechnics_pinned": PYRETECHNICS_VERSION,
        "pyretechnics_patched": is_patched_engine(),
        "crown_cover_threshold": crown_cover_threshold(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "platform": f"{platform.system()}-{platform.machine()}",
        "conda_env": sys.prefix.rsplit("/", 1)[-1],
    }


# ── Inputs: static layers, weather cubes, ignition ───────────────────────────

def load_static_layers(layers_dir):
    """
    Read ``destack_lcp`` output.

    Returns
    -------
    dict, affine.Affine, rasterio.crs.CRS
        ``{layer: 2-D array}`` (float32; ``fuel_model`` int64), transform, CRS.
    """
    import rasterio
    layers_dir = Path(layers_dir)
    arrays, transform, crs = {}, None, None
    for name in STATIC_LAYERS:
        with rasterio.open(layers_dir / f"{name}.tif") as src:
            a = src.read(1)
            arrays[name] = a.astype("int64") if name == "fuel_model" else a.astype("float32")
            transform, crs = src.transform, src.crs
    return arrays, transform, crs


def build_cubes(static, weather, *, foliar_moisture=0.9, adjustments=None):
    """
    Assemble the ``SpaceTimeCube`` dict for ``spread_fire_with_phi_field``.

    Parameters
    ----------
    static : dict
        From :func:`load_static_layers`.
    weather : dict
        ``{cube_name: 1-D float32 array}`` of ``n_hours`` bands, from
        :func:`~fb_tools.spread.episodes.weather_arrays`.
    foliar_moisture : float
        Fraction; constant over the run.
    adjustments : dict, optional
        ``{"fuel_spread_adjustment": 2-D array, "weather_spread_adjustment": float}``
        multipliers on spread rate (engine default 1.0).

    Returns
    -------
    dict, tuple
        Cubes and the cube shape ``(bands, rows, cols)``.
    """
    from pyretechnics.space_time_cube import SpaceTimeCube
    bands = {len(v) for v in weather.values()}
    if len(bands) != 1:
        raise ValueError(f"weather arrays differ in length: {bands}")
    rows, cols = static["slope"].shape
    shape = (bands.pop(), rows, cols)
    cubes = {k: SpaceTimeCube(shape, v) for k, v in static.items()}
    cubes.update({k: SpaceTimeCube(shape, np.asarray(v, dtype="float32")) for k, v in weather.items()})
    cubes["foliar_moisture"] = SpaceTimeCube(shape, np.float32(foliar_moisture))
    for k, v in (adjustments or {}).items():
        cubes[k] = SpaceTimeCube(shape, np.asarray(v, dtype="float32") if np.ndim(v) else np.float32(v))
    return cubes, shape


def ignition_matrix(footprint, transform, shape):
    """
    Rasterize an ignition footprint once for reuse across arms.

    Parameters
    ----------
    footprint : Path or GeoDataFrame
        Polygon(s); reprojected to the raster CRS if it carries one.
    transform : affine.Affine
    shape : tuple
        Raster ``(rows, cols)``.

    Returns
    -------
    tuple, numpy.ndarray
        ``(row0, col0)`` of the window and a float32 window with ``-1.0``
        inside the footprint, ``1.0`` outside — pyretechnics' ``ignite_cells`` convention.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    gdf = footprint if isinstance(footprint, gpd.GeoDataFrame) else gpd.read_file(footprint)
    mask = rasterize(((g, 1) for g in gdf.geometry), out_shape=shape, transform=transform,
                     all_touched=True, dtype="uint8").astype(bool)
    if not mask.any():
        raise ValueError("Ignition footprint rasterizes to zero cells on this grid")
    rr, cc = np.where(mask)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    win = np.where(mask[r0:r1, c0:c1], -1.0, 1.0).astype("float32")
    return (int(r0), int(c0)), win


def seed_for(base_seed, *keys):
    """Deterministic 31-bit seed from ``(base_seed, ignition, episode, condition)``; never the arm."""
    h = hashlib.sha256(repr((int(base_seed),) + tuple(str(k) for k in keys)).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


# ── One realization ───────────────────────────────────────────────────────────

def run_realization(layers_dir, weather, ignition, out_dir, *, seed, duration_h,
                    spot_config=None, crown_gate=None, max_cells_per_timestep=0.4,
                    buffer_width=3, foliar_moisture=0.9, adjustments=None, transform=None,
                    crs=None, meta=None, overwrite=False, **behavior_opts):
    """
    Run one pyretechnics spread and write its rasters and manifest.

    Parameters
    ----------
    layers_dir : Path
        ``destack_lcp`` output for one arm.
    weather : dict
        ``{cube_name: 1-D array}`` with at least ``duration_h`` bands.
    ignition : tuple or Path
        ``((row0, col0), window)`` from :func:`ignition_matrix`, or a footprint path.
    out_dir : Path
    seed : int
        Spotting RNG seed; identical across arms of a realization.
    duration_h : float
    spot_config : dict or None
        Spotting parameters (``random_seed`` is set here). ``None`` = off.
    crown_gate : float or None
        Crown canopy-cover threshold; ``None`` leaves the engine as is.
    max_cells_per_timestep, buffer_width : solver knobs, stamped into the manifest.
    adjustments : dict, optional
        See :func:`build_cubes`.
    meta : dict, optional
        Extra manifest fields (arm, ignition_id, ...).
    overwrite : bool
        Re-run even if ``manifest.json`` exists.
    **behavior_opts
        ``use_wind_limit``, ``surface_lw_ratio_model``, ``crown_max_lw_ratio`` (see ``BEHAVIOR_OPTS``).

    Returns
    -------
    dict
        The manifest (also written to ``out_dir/manifest.json``).
    """
    import rasterio
    import pyretechnics.eulerian_level_set as els

    opts = _behavior_opts(behavior_opts)

    out_dir = Path(out_dir)
    man_fp = out_dir / "manifest.json"
    if man_fp.exists() and not overwrite:
        return json.loads(man_fp.read_text())

    if crown_gate is not None:
        set_crown_cover_threshold(crown_gate)
    static, tf, src_crs = load_static_layers(layers_dir)
    transform, crs = transform or tf, crs or src_crs
    n_bands = int(np.ceil(duration_h))
    weather = {k: np.asarray(v, dtype="float32")[:n_bands] for k, v in weather.items()}
    cubes, shape = build_cubes(static, weather, foliar_moisture=foliar_moisture, adjustments=adjustments)
    if not isinstance(ignition, tuple):
        ignition = ignition_matrix(ignition, transform, shape[1:])
    (r0, c0), win = ignition
    res = abs(transform.a)

    state = els.SpreadState(shape).set_resolution((60.0, res, res)).set_start_time(0.0)
    state.ignite_cells((r0, c0), win)
    spot = None if spot_config is None else {**spot_config, "random_seed": int(seed)}

    t0 = time.perf_counter()
    out = els.spread_fire_with_phi_field(
        cubes, state, max_duration=duration_h * 60.0,
        max_cells_per_timestep=max_cells_per_timestep, buffer_width=buffer_width,
        spot_config=spot, **opts,
    )
    wall_s = time.perf_counter() - t0

    m = out["spread_state"].get_full_matrices()
    burned = np.asarray(m["fire_type"]) > 0
    n_burned = int(burned.sum())
    if n_burned <= int((win < 0).sum()):
        raise RuntimeError(f"Realization did not spread beyond the ignition ({n_burned} cells)")

    _write_layers(out_dir, {k: np.asarray(m[k]) for k in OUTPUT_LAYERS}, burned, transform, crs)

    pending = out.get("spot_ignitions") or {}
    manifest = {
        **(meta or {}),
        "seed": int(seed), "duration_h": float(duration_h), "resolution_m": res,
        "spot_config": spot, "crown_gate": crown_cover_threshold(),
        "max_cells_per_timestep": max_cells_per_timestep, "buffer_width": buffer_width, **opts,
        "foliar_moisture": foliar_moisture, "layers_dir": str(layers_dir),
        "adjustments": sorted(adjustments) if adjustments else [],
        "ignition_window": [r0, c0, *win.shape], "n_ignition_cells": int((win < 0).sum()),
        "n_burned": n_burned, "burned_ha": round(n_burned * res * res / 1e4, 2),
        "stop_condition": out["stop_condition"], "stop_time_min": float(out["stop_time"]),
        "pending_spot_cells": int(sum(len(v) for v in pending.values())),
        "wall_s": round(wall_s, 1),
        **pyre_version_info(),
    }
    man_fp.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _behavior_opts(kw):
    bad = set(kw) - set(BEHAVIOR_OPTS)
    if bad:
        raise TypeError(f"unknown fire-behavior option(s) {sorted(bad)}; allowed: {sorted(BEHAVIOR_OPTS)}")
    return {**BEHAVIOR_OPTS, **kw}


def _write_layers(out_dir, arrays, mask, transform, crs):
    """Write ``{name: 2-D array}`` as float32 GeoTIFFs, NaN where ``mask`` is False."""
    import rasterio
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = next(iter(arrays.values())).shape
    profile = dict(driver="GTiff", height=rows, width=cols, count=1, dtype="float32",
                   crs=crs, transform=transform, compress="deflate", nodata=np.nan)
    for name, a in arrays.items():
        a = a.astype("float32")
        a[~mask] = np.nan
        with rasterio.open(out_dir / f"{name}.tif", "w", **profile) as dst:
            dst.write(a, 1)


# ── Single-run product: potential fire behavior, no spread ───────────────────

def burn_landscape(layers_dir, weather, out_dir, *, t=0, azimuth=None, crown_gate=None,
                   foliar_moisture=0.9, adjustments=None, meta=None, overwrite=False,
                   **behavior_opts):
    """
    Potential fire behavior for every cell under one weather state — the
    FlamMap-style baseline map (``burn_cells.burn_all_cells_as_head_fire``).

    Parameters
    ----------
    layers_dir : Path
        ``destack_lcp`` output.
    weather : dict
        ``{cube_name: scalar or 1-D array}`` in engine units, e.g. from
        :func:`~fb_tools.spread.episodes.weather_from_scenario` or ``weather_arrays``.
    out_dir : Path
    t : int
        Band (hour) of ``weather`` to burn at.
    azimuth : float, optional
        Degrees clockwise from north. ``None`` burns every cell as a head fire
        in its own direction of maximum spread; a value burns all cells toward
        that azimuth (``burn_all_cells_toward_azimuth``).
    crown_gate, foliar_moisture, adjustments, meta, overwrite, **behavior_opts
        As in :func:`run_realization`.

    Returns
    -------
    dict
        Manifest; rasters ``fire_type, spread_rate, spread_direction,
        fireline_intensity, flame_length`` are written to ``out_dir`` (NaN on
        non-burnable cells).
    """
    import pyretechnics.burn_cells as bc

    out_dir = Path(out_dir)
    man_fp = out_dir / "manifest.json"
    if man_fp.exists() and not overwrite:
        return json.loads(man_fp.read_text())
    opts = _behavior_opts(behavior_opts)
    if crown_gate is not None:
        set_crown_cover_threshold(crown_gate)
    static, transform, crs = load_static_layers(layers_dir)
    weather = {k: np.atleast_1d(np.asarray(v, dtype="float32")) for k, v in weather.items()}
    cubes, shape = build_cubes(static, weather, foliar_moisture=foliar_moisture, adjustments=adjustments)

    t0 = time.perf_counter()
    if azimuth is None:
        out = bc.burn_all_cells_as_head_fire(cubes, int(t), **opts)
    else:
        out = bc.burn_all_cells_toward_azimuth(cubes, float(azimuth), int(t), **opts)
    wall_s = time.perf_counter() - t0

    burnable = np.asarray(out["fire_type"]) > 0
    _write_layers(out_dir, {k: np.asarray(out[k]) for k in BEHAVIOR_LAYERS}, burnable, transform, crs)
    ft = np.asarray(out["fire_type"])[burnable]
    manifest = {
        **(meta or {}), "product": "burn_landscape", "t": int(t), "azimuth": azimuth,
        "weather": {k: float(v[min(int(t), len(v) - 1)]) for k, v in weather.items()},
        "crown_gate": crown_cover_threshold(), **opts, "foliar_moisture": foliar_moisture,
        "adjustments": sorted(adjustments) if adjustments else [], "layers_dir": str(layers_dir),
        "resolution_m": abs(transform.a), "n_burnable": int(burnable.sum()),
        "crown_frac": round(float((ft >= 2).mean()), 4) if ft.size else 0.0,
        "wall_s": round(wall_s, 1), **pyre_version_info(),
    }
    man_fp.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def spot_ignition_probability(temperature_c, fm1, *, distance_m=0.0, decay_distance=500.0):
    """
    P(a landed firebrand ignites) per cell — the engine's Schroeder ignition
    probability times its flight-survival term, both exported by
    ``pyretechnics.spot_fire``. A receptivity map for risk products.

    Parameters
    ----------
    temperature_c, fm1 : array-like
        Air temperature (°C) and 1-hr dead fuel moisture (fraction); broadcast.
    distance_m : float or array-like
        Flight distance for the survival term; 0 gives pure receptivity.
    decay_distance : float
        ``SpotConfig.decay_distance`` (m).
    """
    import pyretechnics.spot_fire as spot
    schroeder = np.vectorize(spot.schroeder_ignition_probability, otypes=["float64"])
    survival = np.vectorize(spot.firebrand_flight_survival_probability, otypes=["float64"])
    return schroeder(temperature_c, fm1) * survival(distance_m, decay_distance)


# ── Campaign table and batch ─────────────────────────────────────────────────

def build_runs(arms, ignitions, episodes, conditions, *, base_seed, out_root):
    """
    One row per realization: every (ignition, episode, condition) crossed with every arm.

    Parameters
    ----------
    arms : dict
        ``{arm: layers_dir}``.
    ignitions : dict
        ``{ignition_id: footprint_path}``.
    episodes : dict
        ``{episode_id: hourly_frame_csv}`` from :func:`~fb_tools.spread.episodes.write_episode_frames`.
    conditions : dict
        ``{condition_id: {"spot_config": dict|None, "crown_gate": float, ...}}``;
        extra keys are passed to :func:`run_realization`.
    base_seed : int
    out_root : Path

    Returns
    -------
    pandas.DataFrame
        ``arm, ignition_id, episode_id, condition_id, seed, layers_dir,
        footprint, episode_csv, out_dir`` plus the condition fields as JSON.
    """
    rows = []
    for ign, fp in ignitions.items():
        for ep, csv in episodes.items():
            for cid, cfg in conditions.items():
                seed = seed_for(base_seed, ign, ep, cid)
                for arm, ldir in arms.items():
                    rows.append({
                        "arm": arm, "ignition_id": ign, "episode_id": ep, "condition_id": cid,
                        "seed": seed, "layers_dir": str(ldir), "footprint": str(fp),
                        "episode_csv": str(csv), "condition_json": json.dumps(cfg),
                        "out_dir": str(Path(out_root) / cid / ign / ep / arm),
                    })
    runs = pd.DataFrame(rows)
    assert_crn_contract(runs, n_arms=len(arms))
    return runs


def assert_crn_contract(runs, n_arms):
    """Every (condition, ignition, episode) group has one seed, one episode file, and ``n_arms`` rows."""
    g = runs.groupby(["condition_id", "ignition_id", "episode_id"])
    bad = g.agg(seeds=("seed", "nunique"), eps=("episode_csv", "nunique"), n=("arm", "size"))
    bad = bad[(bad.seeds != 1) | (bad.eps != 1) | (bad.n != n_arms)]
    if len(bad):
        raise AssertionError(f"CRN contract violated:\n{bad}")


def _run_row(row):
    from ..spread.episodes import weather_arrays
    cfg = json.loads(row["condition_json"])
    frame = pd.read_csv(row["episode_csv"])
    meta = {k: row[k] for k in ("arm", "ignition_id", "episode_id", "condition_id")}
    try:
        man = run_realization(row["layers_dir"], weather_arrays(frame), row["ignition"],
                              row["out_dir"], seed=row["seed"], duration_h=row["duration_h"],
                              meta=meta, **cfg)
        return {**meta, "out_dir": row["out_dir"], "status": "ok",
                "burned_ha": man["burned_ha"], "wall_s": man["wall_s"], "error": None}
    except Exception as exc:
        return {**meta, "out_dir": row["out_dir"], "status": "error",
                "burned_ha": np.nan, "wall_s": np.nan, "error": f"{type(exc).__name__}: {exc}"}


def run_batch(runs, *, duration_h, n_workers=1, verbose=True):
    """
    Run a runs table, serially or in a ``spawn`` pool.

    Ignitions are rasterized once per footprint on the first arm's grid and
    shared by every row, so pairing cannot be broken by per-arm rasterization.
    Rows whose ``manifest.json`` exists are skipped by :func:`run_realization`.

    Returns
    -------
    pandas.DataFrame
        One row per realization with ``status, burned_ha, wall_s, error``.
    """
    import rasterio
    first = runs.iloc[0]
    with rasterio.open(Path(first["layers_dir"]) / "slope.tif") as src:
        transform, shape = src.transform, src.shape
    igns = {fp: ignition_matrix(fp, transform, shape) for fp in runs["footprint"].unique()}
    payload = [{**r, "ignition": igns[r["footprint"]], "duration_h": duration_h}
               for r in runs.to_dict("records")]

    if n_workers == 1:
        results = [_run_row(p) for p in payload]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_workers) as pool:
            results = pool.map(_run_row, payload, chunksize=1)
    res = pd.DataFrame(results)
    if verbose:
        n_err = int((res["status"] == "error").sum())
        print(f"[run_batch] {len(res) - n_err}/{len(res)} ok, {n_err} errors, "
              f"{res['wall_s'].sum() / 60:.1f} min compute")
        for _, r in res[res["status"] == "error"].iterrows():
            print(f"  {r['arm']}/{r['ignition_id']}/{r['episode_id']}: {r['error']}")
    return res
