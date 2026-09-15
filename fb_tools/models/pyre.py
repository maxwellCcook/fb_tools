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

from ..spread.pyre_outputs import REALIZATION_LAYERS, rasterize_mask

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

OUTPUT_LAYERS = REALIZATION_LAYERS      # one realization's rasters (defined with their reader)
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
        ``{layer: 2-D float32 array}``, transform, CRS. ``fuel_model`` codes are
        exact in float32, and ``SpaceTimeCube`` copies anything else to float32 anyway.
    """
    import rasterio
    layers_dir = Path(layers_dir)
    arrays, transform, crs = {}, None, None
    for name in STATIC_LAYERS:
        with rasterio.open(layers_dir / f"{name}.tif") as src:
            arrays[name] = src.read(1).astype("float32")
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
        ``{cube_name: float32 array}`` from :func:`~fb_tools.spread.episodes.weather_arrays`:
        1-D ``(n_hours,)`` or 3-D ``(n_hours, rows/ky, cols/kx)`` — the engine repeats a
        coarse block grid by exact integer factors.
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
    bands = {int(np.shape(v)[0]) for v in weather.values()}
    if len(bands) != 1:
        raise ValueError(f"weather arrays differ in length: {bands}")
    rows, cols = static["slope"].shape
    shape = (bands.pop(), rows, cols)
    for k, v in weather.items():                     # the engine does not check this for 3-D bases
        if np.ndim(v) == 3 and (rows % np.shape(v)[1] or cols % np.shape(v)[2]):
            raise ValueError(f"{k} block grid {np.shape(v)[1:]} does not divide the landscape {(rows, cols)}")
    cubes = {k: SpaceTimeCube(shape, v) for k, v in static.items()}
    cubes.update({k: SpaceTimeCube(shape, np.asarray(v, dtype="float32")) for k, v in weather.items()})
    cubes["foliar_moisture"] = SpaceTimeCube(shape, np.float32(foliar_moisture))
    for k, v in (adjustments or {}).items():
        cubes[k] = SpaceTimeCube(shape, np.asarray(v, dtype="float32") if np.ndim(v) else np.float32(v))
    return cubes, shape


def pad_bands(weather, duration_h):
    """
    Trim or extend each weather array to ``ceil(duration_h) + 1`` bands along axis 0,
    repeating the last hour.

    The engine indexes cubes at ``stop_time // band_duration`` on its final step,
    which equals the band count when a run ends exactly at ``max_duration`` — one
    past the end of every cube (pyretechnics 2026.8.10 ``eulerian_level_set``
    ~L2690; ``check_start_and_stop_times`` only rejects ``>``). That read is
    undefined: garbage on most heaps, ``SIGSEGV`` on some (4 of 90 workers died
    on 2026-09-13). One spare band makes the last lookup a real hour.
    """
    n = int(np.ceil(duration_h)) + 1
    out = {}
    for k, v in weather.items():
        a = np.asarray(v, dtype="float32")[:n]
        if a.shape[0] < n:
            a = np.concatenate([a, np.repeat(a[-1:], n - a.shape[0], axis=0)])
        out[k] = a
    return out


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
    gdf = footprint if isinstance(footprint, gpd.GeoDataFrame) else gpd.read_file(footprint)
    mask = rasterize_mask(gdf.geometry, transform, shape)
    if not mask.any():
        raise ValueError("Ignition footprint rasterizes to zero cells on this grid")
    rr, cc = np.where(mask)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    win = np.where(mask[r0:r1, c0:c1], -1.0, 1.0).astype("float32")
    return (int(r0), int(c0)), win


def request_hash(**parts):
    """
    sha256 over everything that determines a run's output, so a cached
    ``manifest.json`` is reused only for the identical request.

    Arrays hash by bytes (float32), everything else by ``repr`` after JSON
    normalization; ``None`` and dict key order are stable.
    """
    h = hashlib.sha256()
    for k in sorted(parts):
        v = parts[k]
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v, dtype="float32").tobytes())
        elif isinstance(v, dict) and v and all(isinstance(x, np.ndarray) for x in v.values()):
            for kk in sorted(v):
                h.update(kk.encode()); h.update(np.ascontiguousarray(v[kk], dtype="float32").tobytes())
        else:
            h.update(json.dumps(v, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _cached_manifest(man_fp, want_hash, overwrite):
    """The existing manifest if it answers this exact request; ``None`` when a (re)run is needed."""
    if overwrite or not man_fp.exists():
        return None
    man = json.loads(man_fp.read_text())
    return man if man.get("request_hash") == want_hash else None


def seed_for(base_seed, *keys):
    """Deterministic 31-bit seed from ``(base_seed, ignition, episode, condition)``; never the arm."""
    h = hashlib.sha256(repr((int(base_seed),) + tuple(str(k) for k in keys)).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


# ── One realization ───────────────────────────────────────────────────────────

def run_realization(layers_dir, weather, ignition, out_dir, *, seed, duration_h,
                    spot_config=None, crown_gate=None, max_cells_per_timestep=0.4,
                    buffer_width=3, foliar_moisture=0.9, adjustments=None, meta=None,
                    overwrite=False, step_h=None, suppression=None, **behavior_opts):
    """
    Run one pyretechnics spread and write its rasters and manifest.

    Parameters
    ----------
    layers_dir : Path
        ``destack_lcp`` output for one arm. With ``suppression`` it must also hold
        ``sdi.tif`` (suppression difficulty on the same grid).
    weather : dict
        ``{cube_name: 1-D or 3-D array}`` with at least ``duration_h`` hourly bands;
        padded to one spare band by :func:`pad_bands`.
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
    step_h : float, optional
        Step the engine in windows of this many hours instead of one call. The
        embers pending at a step's end are carried into the next; the spotting RNG is
        re-seeded per step from ``seed`` (``seed_for(seed, step)``), so pairing across
        arms holds exactly. A stepped run is *not* bitwise the unstepped run — every
        arm of a stepped experiment, including its no-suppression arm, must be
        stepped the same way. Required by ``suppression`` (default 1 h there).
    suppression : dict, optional
        ``{"rule": "pyre_firelines" | "elmfire0", **params}`` — a containment rule from
        :mod:`fb_tools.suppression.containment` (the engine's own ``build_firelines``, or
        rule S1), applied after every step: the frontier cells it holds are set to fuel
        model 91 for the rest of the run. Writes
        ``held_line.tif`` (step index at which each cell was first held) and
        ``suppression_log`` rows into the manifest.
    **behavior_opts
        ``use_wind_limit``, ``surface_lw_ratio_model``, ``crown_max_lw_ratio`` (see ``BEHAVIOR_OPTS``).

    Returns
    -------
    dict
        The manifest (also written to ``out_dir/manifest.json``). An existing
        manifest is returned without running only if its ``request_hash``
        matches this call — a changed footprint, weather, seed or setting
        under the same ``out_dir`` re-runs rather than returning stale output.
        ``step_h`` and ``suppression`` enter the hash only when set, so runs made
        before they existed keep their hash.
    """
    import rasterio
    import pyretechnics.eulerian_level_set as els

    opts = _behavior_opts(behavior_opts)
    out_dir = Path(out_dir)
    man_fp = out_dir / "manifest.json"
    if suppression is not None and step_h is None:
        step_h = 1.0

    static, transform, crs = load_static_layers(layers_dir)
    weather = pad_bands(weather, duration_h)
    if not isinstance(ignition, tuple):
        ignition = ignition_matrix(ignition, transform, static["slope"].shape)
    (r0, c0), win = ignition
    sdi = None
    if suppression is not None:
        sdi_fp = Path(layers_dir) / "sdi.tif"
        if not sdi_fp.exists():
            raise FileNotFoundError(f"suppression needs {sdi_fp} (SDI on the run grid; NaN/nodata where unknown)")
        with rasterio.open(sdi_fp) as src:
            sdi = src.read(1).astype("float32")
            if src.nodata is not None:
                sdi[sdi == src.nodata] = np.nan
        if sdi.shape != static["slope"].shape:
            raise ValueError(f"sdi.tif {sdi.shape} is not on the run grid {static['slope'].shape}")
    req = request_hash(seed=int(seed), duration_h=float(duration_h), layers_dir=str(layers_dir),
                       ignition_origin=[int(r0), int(c0)], ignition_window=win, weather=weather,
                       spot_config=spot_config, crown_gate=crown_gate, max_cells_per_timestep=max_cells_per_timestep,
                       buffer_width=buffer_width, foliar_moisture=foliar_moisture,
                       adjustments={k: np.asarray(v, dtype="float32") for k, v in (adjustments or {}).items()},
                       **({"step_h": float(step_h)} if step_h is not None else {}),
                       **({"suppression": suppression, "sdi": sdi} if suppression is not None else {}),
                       **opts)
    cached = _cached_manifest(man_fp, req, overwrite)
    if cached is not None:
        return cached

    if crown_gate is not None:
        set_crown_cover_threshold(crown_gate)
    cubes, shape = build_cubes(static, weather, foliar_moisture=foliar_moisture, adjustments=adjustments)
    res = abs(transform.a)

    state = els.SpreadState(shape).set_resolution((60.0, res, res)).set_start_time(0.0)
    state.ignite_cells((r0, c0), win)
    spot = None if spot_config is None else {**spot_config, "random_seed": int(seed)}

    t0 = time.perf_counter()
    if step_h is None:
        out = els.spread_fire_with_phi_field(
            cubes, state, max_duration=duration_h * 60.0,
            max_cells_per_timestep=max_cells_per_timestep, buffer_width=buffer_width,
            spot_config=spot, **opts,
        )
        n_steps, held, rule = 1, None, None
    else:
        rule = None
        if suppression is not None:
            from ..suppression.containment import make_rule
            rr, cc = np.nonzero(win < 0)
            rule = make_rule(suppression, sdi, res, (r0 + rr.mean(), c0 + cc.mean()))
        out, n_steps, held = _spread_stepped(
            els, cubes, state, duration_min=duration_h * 60.0, step_min=float(step_h) * 60.0,
            seed=int(seed), spot_config=spot_config, rule=rule,
            max_cells_per_timestep=max_cells_per_timestep, buffer_width=buffer_width, **opts)
    wall_s = time.perf_counter() - t0

    m = out["spread_state"].get_full_matrices()
    burned = np.asarray(m["fire_type"]) > 0
    n_burned = int(burned.sum())
    # A fire that never leaves its footprint is an outcome (an ignition in a non-spreading
    # fuel), not a failure: it is written like any other run and flagged in the manifest.
    stop_condition = "no_spread" if n_burned <= int((win < 0).sum()) else out["stop_condition"]

    _write_layers(out_dir, {k: np.asarray(m[k]) for k in OUTPUT_LAYERS}, burned, transform, crs)
    if held is not None:
        _write_layers(out_dir, {"held_line": held}, held > 0, transform, crs)

    pending = out.get("spot_ignitions") or {}
    manifest = {
        **(meta or {}), "request_hash": req,
        "seed": int(seed), "duration_h": float(duration_h), "resolution_m": res,
        "spot_config": spot, "crown_gate": crown_cover_threshold(),
        "max_cells_per_timestep": max_cells_per_timestep, "buffer_width": buffer_width, **opts,
        "foliar_moisture": foliar_moisture, "layers_dir": str(layers_dir),
        "adjustments": sorted(adjustments) if adjustments else [],
        "ignition_window": [r0, c0, *win.shape], "n_ignition_cells": int((win < 0).sum()),
        "wind_cube_shape": list(np.shape(weather["wind_speed_10m"])),
        "n_burned": n_burned, "burned_ha": round(n_burned * res * res / 1e4, 2),
        "stop_condition": stop_condition, "stop_time_min": float(out["stop_time"]),
        "pending_spot_cells": int(sum(len(v) for v in pending.values())),
        "step_h": None if step_h is None else float(step_h), "n_steps": int(n_steps),
        "suppression": suppression,
        "wall_s": round(wall_s, 1),
        **pyre_version_info(),
    }
    if rule is not None:
        manifest.update({
            "held_cells": int((held > 0).sum()), "held_ha": round(float((held > 0).sum()) * res * res / 1e4, 2),
            "containment_target": None if np.isnan(rule.target) else rule.target, "frozen_sectors": int(rule.frozen.sum()),
            "sdi_fill": rule.sdi_fill, "sdi_known_share": round(rule.sdi_known_share, 4),
            "suppression_log": rule.log,
        })
    man_fp.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _spread_stepped(els, cubes, state, *, duration_min, step_min, seed, spot_config, rule, **engine_kw):
    """
    Drive ``spread_fire_with_phi_field`` in windows of ``step_min``.

    Pending embers are carried between windows and the spotting RNG is re-seeded
    per window with ``seed_for(seed, k)``. After each window ``rule.step`` (if any)
    names the frontier cells to hold; they are set to fuel model 91 in the live
    fuel-model cube, which the engine re-reads when it rebuilds its tracked cells
    at the next window. Stops early when the fire has no tracked cells and no
    pending embers.

    Returns
    -------
    (engine output of the last window, number of windows, held-line raster or None)
    """
    fuel = np.asarray(cubes["fuel_model"].data)        # (1, rows, cols) float32 view of the array the engine reads
    rows, cols = state.get_cube_shape()[1:]
    burned_prev = np.zeros((rows, cols), dtype=bool)
    held = np.zeros((rows, cols), dtype="int16") if rule is not None else None
    pending, t, k, out = {}, 0.0, 0, None
    while t < duration_min - 0.5:                     # the engine may stop a fraction of a minute short
        dt = min(step_min, duration_min - t)
        spot = None if spot_config is None else {**spot_config, "random_seed": seed_for(seed, k)}
        out = els.spread_fire_with_phi_field(cubes, state, max_duration=dt, spot_ignitions=pending,
                                             spot_config=spot, **engine_kw)
        state, k = out["spread_state"], k + 1
        pending = dict(out.get("spot_ignitions") or {})   # the engine hands back a SortedDict and wants a dict
        t = float(out["stop_time"])
        if rule is not None:
            mats = state.get_full_matrices(["fire_type", "flame_length"])
            burned = np.asarray(mats["fire_type"]) > 0
            new_burned = burned & ~burned_prev
            burned_prev = burned
            rr, cc = rule.step(hour=t / 60.0, dt_h=dt / 60.0, new_burned=new_burned, frontier=out["frontier_cells"],
                               fuel=fuel[0], flame_length=np.asarray(mats["flame_length"]))
            fuel[0, rr, cc] = 91.0
            first = held[rr, cc] == 0
            held[rr[first], cc[first]] = k
        if out["num_tracked_cells"] == 0 and not pending:
            break
    return out, k, held


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
    opts = _behavior_opts(behavior_opts)
    weather = {k: np.atleast_1d(np.asarray(v, dtype="float32")) for k, v in weather.items()}
    req = request_hash(product="burn_landscape", layers_dir=str(layers_dir), weather=weather, t=int(t),
                       azimuth=azimuth, crown_gate=crown_gate, foliar_moisture=foliar_moisture,
                       adjustments={k: np.asarray(v, dtype="float32") for k, v in (adjustments or {}).items()}, **opts)
    cached = _cached_manifest(man_fp, req, overwrite)
    if cached is not None:
        return cached
    if crown_gate is not None:
        set_crown_cover_threshold(crown_gate)
    static, transform, crs = load_static_layers(layers_dir)
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
        **(meta or {}), "product": "burn_landscape", "request_hash": req, "t": int(t), "azimuth": azimuth,
        "weather": {k: float(np.mean(v[min(int(t), len(v) - 1)])) for k, v in weather.items()},
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

def build_runs(arms, ignitions, episodes, conditions, *, base_seed, out_root, wind_cubes=None):
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
    wind_cubes : dict, optional
        ``{episode_id: wind_cube_npz}`` from :func:`~fb_tools.spread.episodes.write_wind_cube`;
        episodes without an entry use the frame's 1-D wind.

    Returns
    -------
    pandas.DataFrame
        ``arm, ignition_id, episode_id, condition_id, seed, layers_dir,
        footprint, episode_csv, wind_cube, out_dir`` plus the condition fields as JSON.
    """
    wind_cubes = wind_cubes or {}
    rows = []
    for ign, fp in ignitions.items():
        for ep, csv in episodes.items():
            for cid, cfg in conditions.items():
                seed = seed_for(base_seed, ign, ep, cid)
                for arm, ldir in arms.items():
                    rows.append({
                        "arm": arm, "ignition_id": ign, "episode_id": ep, "condition_id": cid,
                        "seed": seed, "layers_dir": str(ldir), "footprint": str(fp),
                        "episode_csv": str(csv), "wind_cube": str(wind_cubes.get(ep, "")),
                        "condition_json": json.dumps(cfg),
                        "out_dir": str(Path(out_root) / cid / ign / ep / arm),
                    })
    runs = pd.DataFrame(rows)
    assert_crn_contract(runs, n_arms=len(arms))
    return runs


def assert_crn_contract(runs, n_arms):
    """Every (condition, ignition, episode) group has one seed, one episode file, and ``n_arms`` rows."""
    g = runs.groupby(["condition_id", "ignition_id", "episode_id"])
    agg = dict(seeds=("seed", "nunique"), eps=("episode_csv", "nunique"), n=("arm", "size"))
    if "wind_cube" in runs:
        agg["cubes"] = ("wind_cube", "nunique")
    bad = g.agg(**agg)
    bad = bad[(bad.seeds != 1) | (bad.eps != 1) | (bad.n != n_arms) | (bad.get("cubes", 1) != 1)]
    if len(bad):
        raise AssertionError(f"CRN contract violated:\n{bad}")


def _run_row(row):
    from ..spread.episodes import weather_arrays
    cfg = json.loads(row["condition_json"])
    frame = pd.read_csv(row["episode_csv"])
    cube = row.get("wind_cube") or None
    meta = {k: row[k] for k in ("arm", "ignition_id", "episode_id", "condition_id")}
    meta["wind_source"] = f"cube:{Path(cube).name}" if cube else "frame"
    try:
        man = run_realization(row["layers_dir"], weather_arrays(frame, wind_cube=cube), row["ignition"],
                              row["out_dir"], seed=row["seed"], duration_h=row["duration_h"],
                              meta=meta, **cfg)
        return {**meta, "out_dir": row["out_dir"], "status": "ok",
                "burned_ha": man["burned_ha"], "wall_s": man["wall_s"], "error": None}
    except Exception as exc:
        return {**meta, "out_dir": row["out_dir"], "status": "error",
                "burned_ha": np.nan, "wall_s": np.nan, "error": f"{type(exc).__name__}: {exc}"}


def _run_parallel(fn, payload, n_workers, *, progress=None):
    """
    ``[fn(p) for p in payload]`` on a ``spawn`` pool that survives worker deaths.

    ``multiprocessing.Pool.map`` hangs forever when a worker is killed (a native
    segfault, OOM): the task is lost and the pool waits for it. Here rows are fed
    to a ``ProcessPoolExecutor`` at most ``n_workers`` at a time, each in a fresh
    process (``max_tasks_per_child=1``, so nothing the engine leaks accumulates).
    When a worker dies the executor breaks; the rows in flight at that moment
    (at most ``n_workers``) are suspects and are re-run one per isolated process
    at the end, so a crash names its row and that row alone comes back ``None``.
    Everything else continues in a new executor. Results are in ``payload`` order.
    """
    import multiprocessing as mp
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, ProcessPoolExecutor, wait

    ctx = mp.get_context("spawn")
    results = [None] * len(payload)
    queue, suspects, done = deque(range(len(payload))), [], 0

    def tick():
        nonlocal done
        done += 1
        if progress:
            progress(done)

    while queue:
        in_flight = {}
        with ProcessPoolExecutor(n_workers, mp_context=ctx, max_tasks_per_child=1) as ex:
            try:
                while queue or in_flight:
                    while queue and len(in_flight) < n_workers:
                        i = queue.popleft()
                        in_flight[ex.submit(fn, payload[i])] = i
                    ready, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in ready:
                        results[in_flight[fut]] = fut.result()      # raises BrokenExecutor if a worker died
                        del in_flight[fut]
                        tick()
            except BrokenExecutor:
                suspects.extend(in_flight.values())
    for i in suspects:                                    # one fresh process each: a crash names its row
        with ProcessPoolExecutor(1, mp_context=ctx) as ex:
            try:
                results[i] = ex.submit(fn, payload[i]).result()
            except BrokenExecutor:
                results[i] = None
        tick()
    return results


def run_batch(runs, *, duration_h, n_workers=1, verbose=True):
    """
    Run a runs table, serially or in a ``spawn`` pool.

    Ignitions are rasterized once per footprint on the first arm's grid and
    shared by every row, so pairing cannot be broken by per-arm rasterization.
    Rows whose ``manifest.json`` already answers the identical request are
    skipped by :func:`run_realization`; a changed footprint, weather or setting
    under the same ``out_dir`` is re-run. A row whose worker process dies is
    retried once and then reported with ``status == "error"`` — the batch never
    hangs on a lost task (see :func:`_run_parallel`).

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
    n = len(payload)
    step = max(1, n // 10)
    progress = (lambda k: print(f"[run_batch] {k}/{n}", flush=True) if k % step == 0 or k == n else None) if verbose else None

    if n_workers == 1:
        results = []
        for k, p in enumerate(payload, 1):
            results.append(_run_row(p))
            if progress:
                progress(k)
    else:
        results = _run_parallel(_run_row, payload, n_workers, progress=progress)
        for i, r in enumerate(results):
            if r is None:                       # worker died twice on this row
                meta = {k: payload[i][k] for k in ("arm", "ignition_id", "episode_id", "condition_id")}
                results[i] = {**meta, "out_dir": payload[i]["out_dir"], "status": "error", "burned_ha": np.nan,
                              "wall_s": np.nan, "error": "WorkerDied: process exited without returning (retried once)"}
    res = pd.DataFrame(results)
    if verbose:
        n_err = int((res["status"] == "error").sum())
        print(f"[run_batch] {len(res) - n_err}/{len(res)} ok, {n_err} errors, "
              f"{res['wall_s'].sum() / 60:.1f} min compute")
        for _, r in res[res["status"] == "error"].iterrows():
            print(f"  {r['arm']}/{r['ignition_id']}/{r['episode_id']}: {r['error']}")
    return res
