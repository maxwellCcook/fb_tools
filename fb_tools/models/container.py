"""
Spatial-container FSPro orchestration.

High-level entry point that takes any spatial container (HUC12, fireshed,
POD, county) as a GeoDataFrame and assembles a complete, ready-to-run FSPro
simulation directory on macOS, ready for execution on Windows.

Public API
----------
prepare_fspro_experiment
    **The Phase 2 entry point.**  N treatment arms with declared contrasts,
    density-weighted stratified design fires, one FSPro run per ignition per
    arm, and a pre-defined simulation domain used for validation only.

postprocess_fspro_outputs
    Converts FSPro ASC output grids to GeoTIFFs, optionally clips to the
    container boundary (display only), and stacks into a multi-band output.

prepare_container_fspro, prepare_counterfactual_fspro
    Frozen pre-Phase-2 entry points: one and two arms respectively, each
    building a single run from a single ``IgnitionFile``.  Retained for
    reproducing existing runs and observed fires.  New work should use
    ``prepare_fspro_experiment``.

Platform note
-------------
``prepare_container_fspro`` and ``prepare_counterfactual_fspro`` run on
macOS (data preparation only).  ``postprocess_fspro_outputs`` also runs on
macOS (post-processing after Windows execution).  Model execution
(``run_fspro`` / ``run_fspro_batch``) is Windows-only.

Windows path note
-----------------
The ``IgnitionFile`` path embedded in the FSPro input file is written as a
Mac absolute path.  Before executing on Windows, update that line to the
Windows-equivalent path (e.g. map the Box-synced directory to its Windows
drive letter).
"""

import datetime
import json
from pathlib import Path

import numpy as np

from ..utils.io import raster_write_kwargs

# ── Module-level constants ─────────────────────────────────────────────────────

# Primary FSPro ASC output suffixes (from TestFSPro.exe)
_ASC_OUTPUTS: dict[str, str] = {
    "burn_prob":    "_BurnProb.asc",
    "flame_length": "_AvgFlameLength.asc",
    "arrival_time": "_AvgTime.asc",
}

# Band names for the multi-band stacked output (order = band index)
_STACK_BAND_NAMES: list[str] = ["BurnProb", "AvgFlameLength", "AvgTime"]


# ── Private helpers ────────────────────────────────────────────────────────────

def _write_manifest(manifest: dict, out_path: Path) -> None:
    """Serialize manifest dict to JSON, converting Path objects to strings."""
    serializable = {}
    for k, v in manifest.items():
        if isinstance(v, Path):
            serializable[k] = str(v)
        elif isinstance(v, dict):
            serializable[k] = {
                kk: str(vv) if isinstance(vv, Path) else vv
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v
    serializable["created_at"] = datetime.datetime.now().isoformat()
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)


def _resolve_ignition_season_day(
    ignition_season_day: int,
    current_erc_start_doy: "int | None",
    current_erc_n_days: "int | None",
) -> int:
    """
    Resolve the ignition day, rejecting the pre-P1.2 window keywords.

    ``current_erc_start_doy``/``current_erc_n_days`` described an arbitrary
    interior slice of the fire season.  FSPro reads ``CurrentERCValues``
    positionally from season day 1 (spec p.5), so any start other than day 1
    silently shifted the antecedent stream (defect #3).
    """
    if current_erc_start_doy is None and current_erc_n_days is None:
        return int(ignition_season_day)

    start = 1 if current_erc_start_doy is None else int(current_erc_start_doy)
    n = 79 if current_erc_n_days is None else int(current_erc_n_days)
    equivalent = start + n
    if start != 1:
        raise ValueError(
            f"current_erc_start_doy={current_erc_start_doy} is no longer supported. "
            "CurrentERCValues must begin at fire-season day 1 (April 1) — FSPro "
            "reads it positionally, so starting elsewhere shifts the whole "
            "antecedent stream (spec p.5). Replace it with "
            f"ignition_season_day={equivalent} to ignite on the same calendar day, "
            f"or ignition_season_day={n + 1} to keep the same stream length."
        )
    print(
        "  [prepare_*_fspro] current_erc_n_days is deprecated — "
        f"using ignition_season_day={equivalent}"
    )
    return equivalent


def _resolve_arms(
    lcps: "dict | None",
    baseline_lcp_path: "str | Path | None",
    treated_lcp_path: "str | Path | None",
    contrasts: "list | None",
) -> "tuple[dict, list]":
    """
    Normalize the two-arm and N-arm landscape arguments into one mapping.

    Accepts either the N-arm form (``lcps={"untreated": ..., "background":
    ..., "coswap": ...}``) or the legacy two-arm shim
    (``baseline_lcp_path`` / ``treated_lcp_path``), never both.

    Returns
    -------
    tuple
        ``(arms, contrasts)`` — an insertion-ordered ``{arm_name: Path}``
        mapping and a list of ``(arm_a, arm_b)`` pairs.  When *contrasts* is
        omitted it defaults to every ordered pair in arm order, so three arms
        give ``untreated−background``, ``untreated−coswap``,
        ``background−coswap``.

    Raises
    ------
    ValueError
        If both forms are supplied, if neither is, if fewer than two arms are
        given, or if a contrast names an arm that does not exist.
    FileNotFoundError
        If any landscape path does not exist.
    """
    legacy = baseline_lcp_path is not None or treated_lcp_path is not None
    if lcps is not None and legacy:
        raise ValueError(
            "Pass either lcps={arm: path, ...} or the legacy "
            "baseline_lcp_path/treated_lcp_path pair, not both."
        )

    if lcps is None:
        if not legacy:
            raise ValueError(
                "No landscapes supplied. Pass lcps={'untreated': ..., "
                "'background': ..., 'coswap': ...} (arm name → LCP path)."
            )
        if baseline_lcp_path is None or treated_lcp_path is None:
            raise ValueError(
                "The two-arm shim needs both baseline_lcp_path and "
                "treated_lcp_path."
            )
        print("  [_resolve_arms] baseline_lcp_path/treated_lcp_path is the "
              "two-arm shim; prefer lcps={'baseline': ..., 'treated': ...}.")
        arms = {"baseline": Path(baseline_lcp_path),
                "treated":  Path(treated_lcp_path)}
    else:
        arms = {str(k): Path(v) for k, v in lcps.items()}

    if len(arms) < 2:
        raise ValueError(
            f"A counterfactual needs at least two arms, got {list(arms)}."
        )

    for name, path in arms.items():
        if not path.exists():
            raise FileNotFoundError(f"LCP for arm '{name}' not found: {path}")

    names = list(arms)
    if contrasts is None:
        contrasts = [(names[i], names[j])
                     for i in range(len(names)) for j in range(i + 1, len(names))]
    else:
        contrasts = [(str(a), str(b)) for a, b in contrasts]
        for a, b in contrasts:
            for n in (a, b):
                if n not in arms:
                    raise ValueError(
                        f"Contrast ({a}, {b}) names arm '{n}', which is not in "
                        f"lcps: {names}."
                    )
    return arms, contrasts


def _assert_grid_congruence(arms: dict) -> dict:
    """
    Assert every arm's landscape shares one grid, and return that grid.

    A paired difference between arms is only meaningful cell-for-cell.
    ``xr.align(join="left")`` inside
    :func:`~fb_tools.spread.bp.delta_burn_probability` would silently paper
    over a mismatch, producing a Δ surface that looks plausible and is wrong,
    so the check belongs here — before any run is launched.

    Parameters
    ----------
    arms : dict
        ``{arm_name: Path}`` landscape mapping.

    Returns
    -------
    dict
        ``crs``, ``transform``, ``width``, ``height``, ``res``, ``bounds`` of
        the shared grid.

    Raises
    ------
    ValueError
        If any arm differs in CRS, transform, or shape.
    """
    import rasterio

    grids = {}
    for name, path in arms.items():
        with rasterio.open(path) as src:
            grids[name] = {
                "crs":       src.crs,
                "transform": src.transform,
                "width":     src.width,
                "height":    src.height,
                "res":       src.res,
                "bounds":    src.bounds,
            }

    ref_name = next(iter(grids))
    ref = grids[ref_name]
    problems = []
    for name, g in grids.items():
        if name == ref_name:
            continue
        if g["crs"] != ref["crs"]:
            problems.append(
                f"  {name}: CRS {g['crs'].to_string()} != "
                f"{ref_name} {ref['crs'].to_string()}"
            )
        if (g["width"], g["height"]) != (ref["width"], ref["height"]):
            problems.append(
                f"  {name}: shape {g['width']}x{g['height']} != "
                f"{ref_name} {ref['width']}x{ref['height']}"
            )
        if not np.allclose(np.asarray(g["transform"]).astype(float),
                           np.asarray(ref["transform"]).astype(float)):
            problems.append(
                f"  {name}: transform {tuple(round(v, 4) for v in g['transform'])} "
                f"!= {ref_name} {tuple(round(v, 4) for v in ref['transform'])}"
            )

    if problems:
        raise ValueError(
            "Arm landscapes are not on a congruent grid, so a paired "
            "difference between them would compare different ground:\n"
            + "\n".join(problems)
            + "\nRebuild every arm LCP over the same pre-defined domain."
        )

    print(f"  [_assert_grid_congruence] {len(arms)} arm(s) congruent: "
          f"{ref['width']}x{ref['height']} @ {ref['res'][0]:g} m, "
          f"EPSG:{ref['crs'].to_epsg()}")
    return ref


def _check_domain(domain_gdf, grid: dict, label: str = "domain") -> dict:
    """
    Validate a pre-defined simulation domain against the landscape grid.

    The domain is **never** used to clip.  Clipping the LCP to an analysis
    unit makes the container edge a hard boundary fire cannot cross, which
    truncates growth, biases burn probability low near the edge, and leaves
    nowhere for fire to be transmitted *to* — TF_ij becomes unmeasurable.
    This function only records provenance and warns when the landscape does
    not actually cover the domain it claims to.

    Parameters
    ----------
    domain_gdf : geopandas.GeoDataFrame or None
        Pre-defined simulation domain, delineated outside the package.
    grid : dict
        Output of :func:`_assert_grid_congruence`.
    label : str
        Name used in messages.

    Returns
    -------
    dict
        Provenance for the manifest: ``crs``, ``bounds``, ``area_ha``,
        ``n_features``, ``covered_by_lcp``.  Empty dict when *domain_gdf*
        is None.
    """
    if domain_gdf is None:
        return {}

    from shapely.geometry import box

    dom = domain_gdf.to_crs(grid["crs"])
    try:
        dom_union = dom.geometry.union_all()
    except AttributeError:
        dom_union = dom.geometry.unary_union

    lcp_box = box(*grid["bounds"])
    covered = lcp_box.contains(dom_union)
    if not covered:
        outside_ha = (dom_union.difference(lcp_box)).area / 10_000.0
        print(f"  [_check_domain] Warning: {outside_ha:,.0f} ha of the {label} "
              f"falls outside the landscape extent. Fire cannot grow there, so "
              f"burn probability is biased low along that edge. Rebuild the "
              f"LCPs over the full domain.")

    return {
        "crs":            grid["crs"].to_string(),
        "bounds":         [round(v, 2) for v in dom_union.bounds],
        "area_ha":        round(dom_union.area / 10_000.0, 1),
        "n_features":     int(len(dom)),
        "covered_by_lcp": bool(covered),
    }


def _resolve_single_ignition(
    ignition_mode: "str | None",
    container_proj,
    lcp_path,
    ign_dir: Path,
    n_ignitions: int,
    fod_gdf,
    ignition_seed,
    caller: str,
):
    """
    Build the one ``IgnitionFile`` the frozen single-run entry points need.

    ``ignition_mode`` has no default (P2.1).  It used to default to
    ``"container"``, which dissolves the whole analysis unit and hands it to
    FSPro as a starting fire perimeter — every simulated fire began with the
    container already burned, at BP ≈ 1.0 throughout.  Because that silently
    invalidates any treatment effect measured inside the container, the mode
    must now be stated explicitly.

    Raises
    ------
    ValueError
        If *ignition_mode* is None or unrecognized, if ``"fod"`` is requested
        without *fod_gdf*, or if a mode yields more than one ignition — which
        needs one FSPro run each and therefore
        :func:`prepare_fspro_experiment`.
    """
    from fb_tools.fuelscape.lcp import (
        create_container_ignition,
        create_random_ignitions,
        create_fod_ignitions,
    )

    valid = ("container", "random", "fod")
    if ignition_mode is None:
        raise ValueError(
            f"{caller} requires an explicit ignition_mode — one of {valid}.\n"
            "It used to default to 'container', which passes the dissolved "
            "analysis unit to FSPro as a *starting fire perimeter*: every "
            "simulated fire begins with the whole container burned (72.7% of "
            "pixels at BP >= 0.999 on the vendor 416 sample), so no treatment "
            "effect inside it is measurable. Use 'container' only to reproduce "
            "an observed fire from its real perimeter.\n"
            "For design-fire transmission work use prepare_fspro_experiment()."
        )
    if ignition_mode not in valid:
        raise ValueError(
            f"Unknown ignition_mode={ignition_mode!r}; expected one of {valid}."
        )

    if ignition_mode == "container":
        return create_container_ignition(
            container_proj, ign_dir / "container_ignition.shp"
        )

    if ignition_mode == "random":
        paths = create_random_ignitions(
            container_proj, n_ignitions, lcp_path, ign_dir,
            seed=ignition_seed, prefix="random_ign",
        )
    else:
        if fod_gdf is None:
            raise ValueError("ignition_mode='fod' requires fod_gdf to be provided.")
        paths = create_fod_ignitions(container_proj, fod_gdf, lcp_path, ign_dir)

    if len(paths) != 1:
        raise ValueError(
            f"ignition_mode={ignition_mode!r} produced {len(paths)} ignitions, but "
            f"{caller} builds a single FSPro run from a single IgnitionFile.\n"
            "FSPro treats every feature in one IgnitionFile as part of *one* "
            "fire's starting perimeter, so N ignitions in one file give N "
            "simultaneous disjoint starts in every simulation — not N design "
            "fires. Each ignition needs its own input file and its own run: use "
            "prepare_fspro_experiment(), or set n_ignitions=1."
        )
    return paths[0]


def _load_weather_for_pyrome(
    pyrome_id: "str | int",
    weather_dir: "str | Path",
    ignition_season_day: int,
    current_erc_mode: str = "analog_year",
    analog_year: "int | None" = None,
    current_erc_percentile: float = 80.0,
    max_lag: int = 30,
    duration: int = 7,
    erc_classes: "np.ndarray | None" = None,
    gridmet_csv: "str | Path | None" = None,
) -> dict:
    """Load all weather arrays needed for a single FSPro input file.

    Parameters
    ----------
    pyrome_id : str or int
        Pyrome identifier matching cache filenames.
    weather_dir : str or Path
        Root weather cache directory.  Expected sub-directories::

            weather_dir/
              pyrome_erc/   ← pyrome_{id}_gridmet.json files
              pyrome_wind/  ← pyrome_{id}_wind.json files

    ignition_season_day : int
        1-based fire-season day of ignition (1 = April 1).  ``CurrentERCValues``
        always starts at season day 1 and runs to the day before ignition, so
        it carries ``ignition_season_day - 1`` values.
    current_erc_mode : {"analog_year", "percentile", "median", "observed"}
        How the antecedent ERC stream is built.  See
        :func:`~fb_tools.weather.gridmet.build_current_erc_values`.
    analog_year : int, optional
        Force a specific year for ``current_erc_mode="analog_year"``.
    current_erc_percentile : float
        Quantile for ``current_erc_mode="percentile"``.
    max_lag, duration : int
        The run's ``MaxLag`` and ``Duration``, used to enforce the spec-p.5
        window on the length of ``CurrentERCValues``.
    erc_classes : np.ndarray, optional
        Pre-computed ERC class table, shape ``(5, 10)``.  If provided,
        ``gridmet_csv`` is ignored for class building.
    gridmet_csv : str or Path, optional
        Path to the GEE-exported GridMET CSV.  Required when ``erc_classes``
        is ``None``; used to build the ERC class table on the fly.

    Returns
    -------
    dict
        Keys: ``wind_cells``, ``calm_value``, ``speed_breaks``, ``dir_breaks``,
        ``erc_historic``, ``erc_avg``, ``erc_std``, ``erc_classes``,
        ``current_erc``, ``current_erc_meta``.  ``speed_breaks``/``dir_breaks``
        are the bin edges the wind matrix was built on, or None when the cache
        predates them.

    Raises
    ------
    ValueError
        If neither ``erc_classes`` nor ``gridmet_csv`` is provided.
    FileNotFoundError
        If a required cache file does not exist.
    """
    from fb_tools.weather.gridmet import (
        load_gridmet_pyrome_cache,
        build_erc_stats,
        build_erc_classes,
        build_current_erc_values,
        load_gridmet_csv,
    )
    from fb_tools.weather.hrrr import load_pyrome_wind_cells

    pyrome_id = str(pyrome_id)
    weather_dir = Path(weather_dir)

    # ── Validate weather_dir ──────────────────────────────────────────────────
    if weather_dir.is_file():
        raise ValueError(
            f"weather_dir={weather_dir!r} is a file, not a directory.\n"
            "Pass the root weather cache directory (e.g. 'data/weather/'), not "
            "the GridMET CSV path.  The CSV goes in the gridmet_csv= parameter."
        )
    wind_dir = weather_dir / "pyrome_wind"
    erc_dir  = weather_dir / "pyrome_erc"
    if not wind_dir.exists():
        raise FileNotFoundError(
            f"Wind cache directory not found: {wind_dir}\n"
            "Expected layout:\n"
            "  weather_dir/\n"
            "    pyrome_wind/   ← pyrome_{id}_wind.json  (build_pyrome_wind_cells)\n"
            "    pyrome_erc/    ← pyrome_{id}_gridmet.json  (build_historic_erc_arrays)\n"
            "Run build_pyrome_wind_cells(fod_gdf, out_dir=weather_dir/'pyrome_wind') "
            "to generate the wind cache for this pyrome."
        )
    if not erc_dir.exists():
        raise FileNotFoundError(
            f"ERC cache directory not found: {erc_dir}\n"
            "Run build_historic_erc_arrays(df, out_dir=weather_dir/'pyrome_erc') "
            "to generate the ERC cache."
        )

    # Wind
    wind_meta = load_pyrome_wind_cells(
        pyrome_id, wind_dir, return_meta=True
    )
    wind_cells = wind_meta["WindCellValues"]   # already np.ndarray
    calm_value = float(wind_meta["CalmValue"])
    # P1.5 — the cache records the bin edges the matrix was built on.  Dropping
    # them let build_fspro_inputs fall back to its defaults, so a cache built
    # with custom breaks silently contradicted its own frequency table.
    speed_breaks = wind_meta.get("WindSpeedBreaks_mph")
    dir_breaks = wind_meta.get("WindDirBreaks_deg")
    if speed_breaks is not None and len(speed_breaks) != np.shape(wind_cells)[0]:
        raise ValueError(
            f"Pyrome {pyrome_id} wind cache is inconsistent: "
            f"{len(speed_breaks)} speed breaks but the matrix has "
            f"{np.shape(wind_cells)[0]} rows"
        )
    if dir_breaks is not None and len(dir_breaks) != np.shape(wind_cells)[1]:
        raise ValueError(
            f"Pyrome {pyrome_id} wind cache is inconsistent: "
            f"{len(dir_breaks)} direction breaks but the matrix has "
            f"{np.shape(wind_cells)[1]} columns"
        )

    # Historic ERC
    erc_meta = load_gridmet_pyrome_cache(
        pyrome_id, erc_dir, return_meta=True
    )
    erc_historic = np.array(erc_meta["HistoricERCValues"], dtype=float)

    # Per-DOY stats
    stats = build_erc_stats({pyrome_id: erc_historic})
    erc_avg = stats[pyrome_id]["avg"]
    erc_std = stats[pyrome_id]["std"]

    # Current-season ERC — always season day 1 up to the day before ignition.
    current_erc_meta = build_current_erc_values(
        {pyrome_id: erc_historic},
        ignition_season_day=ignition_season_day,
        mode=current_erc_mode,
        years=erc_meta.get("years"),
        analog_year=analog_year,
        percentile=current_erc_percentile,
        max_lag=max_lag,
        duration=duration,
        return_meta=True,
    )[pyrome_id]
    current_erc = current_erc_meta["values"]

    # ERC classes — priority: explicit array > cache JSON > on-the-fly from CSV
    if erc_classes is not None:
        # 1. Caller supplied a precomputed array — use directly
        erc_classes = np.asarray(erc_classes, dtype=float)
    elif "ERCClasses" in erc_meta:
        # 2. Cache JSON was enriched by save_erc_classes_to_cache() — use it
        erc_classes = np.array(erc_meta["ERCClasses"], dtype=float)
    elif gridmet_csv is not None:
        # 3. Raw CSV provided — build on the fly
        df = load_gridmet_csv(gridmet_csv)
        df_p = df[df["pyrome"].astype(str) == pyrome_id]
        if df_p.empty:
            raise ValueError(
                f"Pyrome '{pyrome_id}' not found in {gridmet_csv}. "
                "Check that the CSV contains this pyrome ID."
            )
        erc_classes_dict = build_erc_classes(df_p, weather_dir=weather_dir)
        erc_classes = erc_classes_dict[pyrome_id]
    else:
        raise ValueError(
            f"ERC class table not found for pyrome '{pyrome_id}'.\n"
            "Fix with one of:\n"
            "  (a) Run once to enrich all cache files:\n"
            "        from fb_tools import save_erc_classes_to_cache\n"
            f"        save_erc_classes_to_cache(gridmet_csv, '{erc_dir}')\n"
            "  (b) Pass gridmet_csv= to this call each time.\n"
            "  (c) Pass erc_classes= as a precomputed (5,10) array."
        )

    return {
        "wind_cells":       wind_cells,
        "calm_value":       calm_value,
        "speed_breaks":     speed_breaks,
        "dir_breaks":       dir_breaks,
        "erc_historic":     erc_historic,
        "erc_avg":          erc_avg,
        "erc_std":          erc_std,
        "erc_classes":      erc_classes,
        "current_erc":      current_erc,
        "current_erc_meta": current_erc_meta,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def prepare_container_fspro(
    container_gdf,
    out_dir: "str | Path",
    weather_dir: "str | Path",
    pyromes_gdf,
    lf_year: "str | int",
    lcp_path: "str | Path | None" = None,
    lcp_name: "str | None" = None,
    num_fires: int = 1000,
    duration: int = 7,
    resolution: float = 90.0,
    erc_classes: "np.ndarray | None" = None,
    gridmet_csv: "str | Path | None" = None,
    ignition_season_day: int = 80,
    current_erc_mode: str = "analog_year",
    analog_year: "int | None" = None,
    current_erc_percentile: float = 80.0,
    current_erc_start_doy: "int | None" = None,
    current_erc_n_days: "int | None" = None,
    ignition_mode: "str | None" = None,
    n_ignitions: int = 200,
    fod_gdf=None,
    ignition_seed=None,
    **fspro_kwargs,
) -> dict:
    """Prepare a complete FSPro simulation directory for a spatial container.

    .. deprecated:: Phase 2

       Frozen in favour of :func:`prepare_fspro_experiment`, which handles N
       treatment arms, density-weighted design fires, and one FSPro run per
       ignition.  This function builds exactly **one** run from **one**
       ``IgnitionFile``, which is only correct when that file is a single
       starting fire perimeter.  Retained for reproducing an observed fire and
       for pre-Phase-2 runs; it receives no new Phase 2+ features.

    Orchestrates landscape download, ignition file creation, pyrome weather
    extraction, and FSPro input file assembly.  All steps run on macOS; model
    execution is performed separately on Windows using ``run_fspro()``.

    Parameters
    ----------
    container_gdf : geopandas.GeoDataFrame
        Spatial container defining the simulation domain.  Any CRS; reprojected
        internally as needed.  Accepted container types: HUC12, fireshed, POD,
        county, or any arbitrary polygon GeoDataFrame.
    out_dir : str or Path
        Root output directory.  Sub-directories are created automatically::

            out_dir/
              lcp/             ← LANDFIRE landscape download (or symlink)
              ignitions/       ← container ignition shapefile
              fspro_inputs/    ← FSPro input file
              outputs/         ← empty; populated by run_fspro on Windows

    weather_dir : str or Path
        Root weather cache directory containing pre-built pyrome JSON files::

            weather_dir/
              pyrome_erc/   ← pyrome_{id}_gridmet.json
              pyrome_wind/  ← pyrome_{id}_wind.json

    pyromes_gdf : geopandas.GeoDataFrame or str or Path
        NIFC pyrome polygons.  Accepts either an already-loaded GeoDataFrame
        or a file path (shapefile, GeoPackage, etc.) that will be read with
        ``geopandas.read_file()``.  Must contain a ``Pyrome_ID`` column (or
        pass ``pyrome_col`` via ``fspro_kwargs`` — note: ``pyrome_col`` is
        consumed here and not forwarded to ``build_fspro_inputs``).
    lf_year : str or int
        LANDFIRE year for ``lfps_request`` (e.g. ``"2023"``).  Ignored when
        ``lcp_path`` is provided.
    lcp_path : str or Path, optional
        Pre-existing LCP GeoTIFF.  When provided, the LFPS download and the
        existence check are both skipped entirely.  The LCP CRS is used to
        reproject the container for ignition creation.
    lcp_name : str, optional
        Base filename (no extension) for the downloaded LCP GeoTIFF, passed
        as ``rename`` to ``lfps_request``.  When provided, the function
        checks for ``out_dir/lcp/{lcp_name}.tif`` before submitting an LFPS
        job — if the file already exists it is reused and the download is
        skipped.  When omitted (default), any existing ``*.tif`` in
        ``out_dir/lcp/`` is reused if exactly one is found; otherwise LFPS
        is called and the file keeps its default name.
    erc_classes : np.ndarray, optional
        Pre-computed ERC class table, shape ``(5, 10)``.  Provide this *or*
        ``gridmet_csv`` — the pyrome cache JSON does not store ERC classes.
    gridmet_csv : str or Path, optional
        Path to GEE-exported GridMET CSV.  Used to build ERC classes on the
        fly when ``erc_classes`` is not provided.
    num_fires : int
        Number of fire simulations (``NumFires``).  More fires produce
        smoother burn-probability surfaces.  Default 1000; production runs
        typically use 1000–3000.
    duration : int
        Maximum burn period per fire in days (``Duration``).  Default 7.
    resolution : float
        Output grid cell size in metres (``Resolution``).  Default 90.0.
        Must be a multiple of the LCP cell size.
    ignition_season_day : int
        1-based fire-season day of ignition (1 = April 1).  Default 80
        (≈ June 19).  ``CurrentERCValues`` runs from season day 1 to the day
        before ignition, so it carries ``ignition_season_day - 1`` values, and
        the spec-p.5 window ``MaxLag <= NumWxCurrYear < NumWxPerYear - Duration``
        is enforced.
    current_erc_mode : {"analog_year", "percentile", "median", "observed"}
        How the antecedent ERC stream is built.  Default ``"analog_year"`` —
        a real year's sequence, preserving variance and autocorrelation that a
        cross-year median strips out.  See
        :func:`~fb_tools.weather.gridmet.build_current_erc_values`.
    analog_year : int, optional
        Force a specific calendar year for ``current_erc_mode="analog_year"``.
        When None, the year with the highest season-to-date ERC accumulation
        is chosen.
    current_erc_percentile : float
        Quantile for ``current_erc_mode="percentile"``.  Default 80.
    current_erc_start_doy, current_erc_n_days : int, optional
        Deprecated pre-P1.2 window keywords.  ``current_erc_start_doy`` other
        than 1 raises — FSPro reads ``CurrentERCValues`` positionally from
        season day 1, so an interior slice shifted the whole stream (defect #3).
    ignition_mode : {"container", "random", "fod"}
        **Required — there is no default.**  Controls how the single
        ``IgnitionFile`` is built.

        ``"container"``
            Dissolves the full container polygon and hands it to FSPro as a
            **starting fire perimeter**, so every simulated fire begins with
            the whole analysis unit burned (72.7% of interior pixels at
            BP ≥ 0.999 on the vendor 416 sample, against 0.55% grid-wide).
            Correct only for reproducing an observed fire from its real
            perimeter — never for a treatment-effect study.
        ``"random"``
            Samples burnable pixels at random from within the container.
            Because each ignition is a separate fire needing its own run,
            only ``n_ignitions=1`` is accepted here; use
            :func:`prepare_fspro_experiment` for a design-fire set.
        ``"fod"``
            Uses historical FPA-FOD (or equivalent) point locations clipped
            to the container.  Requires *fod_gdf*, and likewise accepts only
            a single resulting ignition.

        This parameter used to default to ``"container"`` (defect #1), which
        silently made every treatment effect measured inside the container
        unmeasurable.  Omitting it now raises.
    n_ignitions : int
        Number of random ignition points for ``ignition_mode="random"``.
        Default 200.
    fod_gdf : GeoDataFrame, optional
        FPA-FOD (or equivalent) point GeoDataFrame, pre-filtered to the
        desired years/classes.  Required when ``ignition_mode="fod"``.
    ignition_seed : int, optional
        Random seed for ``ignition_mode="random"``.
    **fspro_kwargs
        Additional overrides for any key in ``_FSPRO_DEFAULTS``, e.g.
        ``CROWN_FIRE_METHOD="Scott/Reinhardt"``, ``SavePerimeters=0``.
        ``pyrome_col`` is also intercepted here if present.

    Returns
    -------
    dict
        Manifest with keys:

        ``"lcp_path"`` : Path
            Landscape GeoTIFF used for this run.
        ``"ignition_path"`` : Path
            Ignition shapefile (.shp) — container polygon, random circles,
            or FOD circles depending on *ignition_mode*.
        ``"fspro_input_path"`` : Path
            Written FSPro input file (FSPRO-Inputs-File-Version-4).
        ``"pyrome_id"`` : str
            Dominant pyrome ID used for weather.
        ``"out_dir"`` : Path
            Absolute root output directory.
        ``"outputs_dir"`` : Path
            Empty directory where FSPro will write outputs on Windows.
        ``"manifest_path"`` : Path
            ``out_dir/run_manifest.json`` capturing all paths and metadata.

    Raises
    ------
    FileNotFoundError
        If ``lcp_path`` is provided but does not exist.
    ValueError
        If neither ``erc_classes`` nor ``gridmet_csv`` is supplied.
        If the container does not intersect any pyrome in ``pyromes_gdf``.

    Notes
    -----
    **Windows path**: The ``IgnitionFile`` line in the FSPro input file is
    written as a Mac absolute path.  Before running on Windows, update that
    line to the Windows-equivalent path (e.g. replace the Mac mount point
    with the corresponding Windows drive letter).
    """
    import rasterio
    import geopandas as gpd
    from fb_tools.fuelscape.lfps import lfps_request
    from fb_tools.fuelscape.lcp import (
        create_container_ignition,
        create_random_ignitions,
        create_fod_ignitions,
    )
    from fb_tools.models.fspro import build_fspro_inputs
    from fb_tools.utils.geo import lookup_pyrome

    # Accept a file path for pyromes_gdf
    if isinstance(pyromes_gdf, (str, Path)):
        pyromes_gdf = gpd.read_file(pyromes_gdf)

    out_dir = Path(out_dir).resolve()
    lcp_dir     = out_dir / "lcp"
    ign_dir     = out_dir / "ignitions"
    inputs_dir  = out_dir / "fspro_inputs"
    outputs_dir = out_dir / "outputs"
    for d in (lcp_dir, ign_dir, inputs_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. LCP ────────────────────────────────────────────────────────────────
    if lcp_path is not None:
        # Caller supplied an explicit path — use it directly, no checks.
        lcp_path = Path(lcp_path)
        if not lcp_path.exists():
            raise FileNotFoundError(f"lcp_path not found: {lcp_path}")
    else:
        # Check whether the LCP was already downloaded to lcp_dir.
        if lcp_name is not None:
            candidate = lcp_dir / f"{lcp_name}.tif"
        else:
            existing = sorted(lcp_dir.glob("*.tif"))
            candidate = existing[0] if len(existing) == 1 else None

        if candidate is not None and candidate.exists():
            print(f"[prepare_container_fspro] Reusing existing LCP: {candidate.name}")
            lcp_path = candidate
        else:
            print("[prepare_container_fspro] Downloading LANDFIRE landscape …")
            lcp_path = lfps_request(
                container_gdf, lcp_dir, str(lf_year),
                rename=lcp_name, clip=True,
            )

    # Report domain pixel count
    with rasterio.open(lcp_path) as src:
        lcp_crs     = src.crs
        pixel_count = src.width * src.height
    print(f"[prepare_container_fspro] LCP: {lcp_path.name} "
          f"({pixel_count:,} pixels, CRS: {lcp_crs.to_epsg() or lcp_crs.to_string()})")

    # ── 2. Dominant pyrome ────────────────────────────────────────────────────
    pyrome_col = fspro_kwargs.pop("pyrome_col", "Pyrome_ID")
    # Reproject container to pyromes CRS for the overlap query.
    # lookup_pyrome assigns pyromes_gdf.crs to the geometry it receives, so we
    # must NOT pass coordinates already in LCP CRS (often UTM) — they would be
    # mis-labelled as WGS84 and the overlay would find nothing.
    container_pyromes_crs = container_gdf.to_crs(pyromes_gdf.crs)
    try:
        container_union = container_pyromes_crs.geometry.union_all()
    except AttributeError:
        container_union = container_pyromes_crs.geometry.unary_union

    pyrome_id = str(lookup_pyrome(container_union, pyromes_gdf, pyrome_col=pyrome_col))
    print(f"[prepare_container_fspro] Dominant pyrome: {pyrome_id}")

    # Separate reproject to LCP CRS — used for ignition file creation below.
    container_proj = container_gdf.to_crs(lcp_crs)

    # ── 3. Weather ────────────────────────────────────────────────────────────
    print("[prepare_container_fspro] Loading pyrome weather …")
    wx = _load_weather_for_pyrome(
        pyrome_id,
        weather_dir,
        ignition_season_day=_resolve_ignition_season_day(
            ignition_season_day, current_erc_start_doy, current_erc_n_days
        ),
        current_erc_mode=current_erc_mode,
        analog_year=analog_year,
        current_erc_percentile=current_erc_percentile,
        max_lag=int(fspro_kwargs.get("MaxLag", 30)),
        duration=duration,
        erc_classes=erc_classes,
        gridmet_csv=gridmet_csv,
    )

    # ── 4. Ignition file ──────────────────────────────────────────────────────
    ign_path = _resolve_single_ignition(
        ignition_mode, container_proj, lcp_path, ign_dir,
        n_ignitions, fod_gdf, ignition_seed,
        caller="prepare_container_fspro",
    )
    print(f"[prepare_container_fspro] Ignition ({ignition_mode}): {ign_path.name}")

    # ── 5. FSPro input file ───────────────────────────────────────────────────
    # Merge explicit sim params — caller kwargs take precedence over the
    # named parameters so advanced users can still override via **fspro_kwargs.
    sim_params = {"NumFires": num_fires, "Duration": duration, "Resolution": resolution}
    sim_params.update(fspro_kwargs)

    fspro_input_path = build_fspro_inputs(
        output_path   = inputs_dir / "fspro.input",
        wind_cells    = wx["wind_cells"],
        calm_value    = wx["calm_value"],
        erc_historic  = wx["erc_historic"],
        erc_avg       = wx["erc_avg"],
        erc_std       = wx["erc_std"],
        erc_classes   = wx["erc_classes"],
        current_erc   = wx["current_erc"],
        speed_breaks  = wx["speed_breaks"],
        dir_breaks    = wx["dir_breaks"],
        ignition_file = ign_path,
        **sim_params,
    )
    print(f"[prepare_container_fspro] FSPro input: {fspro_input_path.name} "
          f"(NumFires={num_fires}, Duration={duration}, Resolution={resolution}m)")

    # ── 6. Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "lcp_path":        lcp_path,
        "ignition_path":   ign_path,
        "fspro_input_path": fspro_input_path,
        "pyrome_id":       pyrome_id,
        "out_dir":         out_dir,
        "outputs_dir":     outputs_dir,
        "lf_year":         str(lf_year),
        "fspro_kwargs":    fspro_kwargs,
    }
    manifest_path = out_dir / "run_manifest.json"
    _write_manifest(manifest, manifest_path)
    manifest["manifest_path"] = manifest_path

    print(f"[prepare_container_fspro] Done. Manifest → {manifest_path}")
    return manifest


def postprocess_fspro_outputs(
    output_dir: "str | Path",
    output_basename: str = "fspro_out",
    container_gdf=None,
    ref_lcp: "str | Path | None" = None,
    stack: bool = True,
    out_crs: "int | str | None" = None,
) -> dict:
    """Convert FSPro ASCII output grids to GeoTIFFs and optionally stack them.

    Reads the primary FSPro raster outputs (BurnProb, AvgFlameLength,
    AvgTime) from ArcInfo ASCII Grid (.asc) format, injects the CRS from a
    reference LCP, writes float32 GeoTIFFs, and optionally clips to the
    container boundary and stacks into a multi-band output.

    Parameters
    ----------
    output_dir : str or Path
        Directory where FSPro wrote its output files.  Expected to contain
        ``{output_basename}_BurnProb.asc``, ``{output_basename}_AvgFlameLength.asc``,
        and ``{output_basename}_AvgTime.asc``.
    output_basename : str
        Prefix used by FSPro for output filenames.  Default ``"fspro_out"``.
    container_gdf : geopandas.GeoDataFrame, optional
        If provided, each output GeoTIFF is clipped to this boundary after
        conversion.  **Display only.**  Every Δ statistic — ΔBP, ΔTF_ij,
        Δ flame length, Δ arrival time — must be computed on the *unclipped*
        grid: destinations for transmitted fire lie outside any one container
        by definition, and clipping first discards exactly the pixels TF_ij is
        about.  Clip afterwards, for maps.
    ref_lcp : str or Path, optional
        Reference LCP GeoTIFF.  Its CRS is injected into each output raster
        (FSPro ASC outputs carry no embedded CRS).  When omitted, the output
        GeoTIFFs will have an undefined CRS — always pass ``ref_lcp`` in
        practice.
    stack : bool
        Stack individual GeoTIFFs into one multi-band file (default True).
        Band order: BurnProb (1), AvgFlameLength (2), AvgTime (3).
    out_crs : int or str, optional
        Override the CRS for output GeoTIFFs.  If None and ``ref_lcp`` is
        given, the LCP CRS is used.

    Returns
    -------
    dict
        Output file paths (``Path`` or ``None`` for missing outputs):

        ``"burn_prob_tif"``    : Path or None
        ``"flame_length_tif"`` : Path or None
        ``"arrival_time_tif"`` : Path or None
        ``"stacked_tif"``      : Path or None  (None when ``stack=False``)
        ``"perimeters_shp"``   : Path or None

    Raises
    ------
    FileNotFoundError
        If ``output_dir`` does not exist.

    Notes
    -----
    FSPro ASC outputs use the same cell size and origin as the input LCP.
    The CRS injection step (from ``ref_lcp``) is therefore sufficient for
    correct georeferencing — full reprojection is not needed.
    """
    import rasterio
    from rasterio.transform import from_origin

    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"output_dir not found: {output_dir}")

    # ── Resolve reference CRS ─────────────────────────────────────────────────
    ref_crs = None
    if ref_lcp is not None:
        with rasterio.open(ref_lcp) as src:
            ref_crs = src.crs
    if out_crs is not None:
        import rasterio.crs as rcrs
        ref_crs = rcrs.CRS.from_user_input(out_crs)

    if ref_crs is None:
        print("[postprocess_fspro_outputs] Warning: no ref_lcp or out_crs — "
              "output GeoTIFFs will have undefined CRS.")

    # ── ASC → GeoTIFF loop ────────────────────────────────────────────────────
    result_keys = {
        "burn_prob":    "burn_prob_tif",
        "flame_length": "flame_length_tif",
        "arrival_time": "arrival_time_tif",
    }
    tif_paths: dict[str, "Path | None"] = {
        v: None for v in result_keys.values()
    }
    written_tifs: list[Path] = []

    for key, suffix in _ASC_OUTPUTS.items():
        asc_path = output_dir / f"{output_basename}{suffix}"
        if not asc_path.exists():
            print(f"[postprocess_fspro_outputs] Warning: not found: {asc_path.name}")
            continue

        tif_path = output_dir / f"{asc_path.stem}.tif"

        with rasterio.open(asc_path) as src:
            arr = src.read(1).astype("float32")
            nodata_val = src.nodata if src.nodata is not None else -9999.0

            out_profile = src.profile.copy()
            out_profile.update(
                driver="GTiff",
                dtype="float32",
                nodata=np.nan,
                compress="deflate",
            )
            if ref_crs is not None:
                out_profile["crs"] = ref_crs

        # Mask nodata
        arr[arr == nodata_val] = np.nan

        with rasterio.open(tif_path, "w", **out_profile) as dst:
            dst.write(arr, 1)

        if container_gdf is not None:
            from fb_tools.utils.geo import clip_raster_inplace
            clip_raster_inplace(tif_path, container_gdf)

        tif_paths[result_keys[key]] = tif_path
        written_tifs.append(tif_path)
        print(f"[postprocess_fspro_outputs] {asc_path.name} → {tif_path.name}")

    # ── Stack ─────────────────────────────────────────────────────────────────
    stacked_path = None
    if stack and written_tifs:
        import rioxarray as rxr
        import xarray as xr

        bands = []
        band_names_written = []
        # Collect in fixed order matching _STACK_BAND_NAMES
        key_order = list(result_keys.values())
        name_order = _STACK_BAND_NAMES[:]
        for tif_key, band_name in zip(key_order, name_order):
            p = tif_paths[tif_key]
            if p is not None:
                da = rxr.open_rasterio(p, masked=True).squeeze(drop=True)
                bands.append(da)
                band_names_written.append(band_name)

        if bands:
            stacked = xr.concat(bands, dim=xr.Variable("band", band_names_written))
            stacked.attrs["long_name"] = band_names_written
            stacked_path = output_dir / f"{output_basename}_FSProGrids.tif"
            stacked.rio.to_raster(stacked_path, **raster_write_kwargs())
            print(f"[postprocess_fspro_outputs] Stacked ({len(bands)} bands) → "
                  f"{stacked_path.name}")

    # ── Perimeters ────────────────────────────────────────────────────────────
    perims_path = output_dir / f"{output_basename}_Perimeters.shp"
    perimeters = perims_path if perims_path.exists() else None

    return {
        "burn_prob_tif":    tif_paths["burn_prob_tif"],
        "flame_length_tif": tif_paths["flame_length_tif"],
        "arrival_time_tif": tif_paths["arrival_time_tif"],
        "stacked_tif":      stacked_path,
        "perimeters_shp":   perimeters,
    }


def prepare_counterfactual_fspro(
    container_gdf,
    out_dir: "str | Path",
    weather_dir: "str | Path",
    pyromes_gdf,
    baseline_lcp_path: "str | Path",
    treated_lcp_path: "str | Path",
    num_fires: int = 1000,
    duration: int = 7,
    resolution: float = 90.0,
    erc_classes: "np.ndarray | None" = None,
    gridmet_csv: "str | Path | None" = None,
    ignition_season_day: int = 80,
    current_erc_mode: str = "analog_year",
    analog_year: "int | None" = None,
    current_erc_percentile: float = 80.0,
    current_erc_start_doy: "int | None" = None,
    current_erc_n_days: "int | None" = None,
    seed: int = 617327,
    ignition_mode: "str | None" = None,
    n_ignitions: int = 200,
    fod_gdf=None,
    ignition_seed=None,
    **fspro_kwargs,
) -> dict:
    """Prepare paired baseline and treated FSPro runs for counterfactual analysis.

    .. deprecated:: Phase 2

       Frozen in favour of :func:`prepare_fspro_experiment`, which generalizes
       this to N arms (``untreated`` / ``background`` / ``coswap``) with
       declared contrasts, asserts grid congruence across arms, accepts a
       pre-defined simulation domain, and builds density-weighted design fires
       with one FSPro run each.  This function is hard-wired to two arms and
       one ignition.  Retained for pre-Phase-2 runs; no new features.

    Assembles the shared experimental design — ignition file, pyrome weather,
    and a single FSPro input file with a fixed ``SPOTTING_SEED`` — for two
    pre-built landscape files.  Because TestFSPro.exe takes the LCP as a
    runtime positional argument, one input file serves both the baseline and
    treated runs, guaranteeing identical weather draws for a clean
    counterfactual comparison via ``delta_burn_probability()``.

    LCP preparation (LFPS download, ``apply_treatment``) is intentionally kept
    separate from this function.  Pass finished GeoTIFFs for both landscapes.

    Parameters
    ----------
    container_gdf : geopandas.GeoDataFrame
        Spatial container defining the simulation domain.  Any CRS.
    out_dir : str or Path
        Root output directory.  Sub-directories are created automatically::

            out_dir/
              ignitions/           ← shared container ignition shapefile
              fspro_inputs/        ← single shared FSPro input file
              baseline/outputs/    ← empty; populated by run_fspro on Windows
              treated/outputs/     ← empty; populated by run_fspro on Windows
              run_manifest.json

    weather_dir : str or Path
        Root weather cache directory::

            weather_dir/
              pyrome_erc/   ← pyrome_{id}_gridmet.json
              pyrome_wind/  ← pyrome_{id}_wind.json

    pyromes_gdf : geopandas.GeoDataFrame or str or Path
        NIFC pyrome polygons.  Accepts either an already-loaded GeoDataFrame
        or a file path (shapefile, GeoPackage, etc.) read with
        ``geopandas.read_file()``.
    baseline_lcp_path : str or Path
        Baseline landscape GeoTIFF.  Must already exist — use
        ``prepare_container_fspro`` or ``lfps_request`` to download it first.
    treated_lcp_path : str or Path
        Treated landscape GeoTIFF.  Must already exist — use
        ``apply_treatment`` to build it from the baseline first.
    num_fires : int
        Number of fire simulations (``NumFires``).  Default 1000.
    duration : int
        Maximum burn period per fire in days (``Duration``).  Default 7.
    resolution : float
        Output grid cell size in metres (``Resolution``).  Default 90.0.
    erc_classes : np.ndarray, optional
        Pre-computed ERC class table, shape ``(5, 10)``.
    gridmet_csv : str or Path, optional
        GEE-exported GridMET CSV, used when ``erc_classes`` is not provided.
    ignition_season_day : int
        1-based fire-season day of ignition (1 = April 1).  Default 80
        (≈ June 19).  ``CurrentERCValues`` runs from season day 1 to the day
        before ignition, so it carries ``ignition_season_day - 1`` values, and
        the spec-p.5 window ``MaxLag <= NumWxCurrYear < NumWxPerYear - Duration``
        is enforced.
    current_erc_mode : {"analog_year", "percentile", "median", "observed"}
        How the antecedent ERC stream is built.  Default ``"analog_year"`` —
        a real year's sequence, preserving variance and autocorrelation that a
        cross-year median strips out.  See
        :func:`~fb_tools.weather.gridmet.build_current_erc_values`.
    analog_year : int, optional
        Force a specific calendar year for ``current_erc_mode="analog_year"``.
        When None, the year with the highest season-to-date ERC accumulation
        is chosen.
    current_erc_percentile : float
        Quantile for ``current_erc_mode="percentile"``.  Default 80.
    current_erc_start_doy, current_erc_n_days : int, optional
        Deprecated pre-P1.2 window keywords.  ``current_erc_start_doy`` other
        than 1 raises — FSPro reads ``CurrentERCValues`` positionally from
        season day 1, so an interior slice shifted the whole stream (defect #3).
    seed : int
        ``SPOTTING_SEED`` shared by both runs.  **Do not vary between
        baseline and treated.**  Default 617327.
    ignition_mode : {"container", "random", "fod"}
        **Required — there is no default.**  Controls how the shared
        ``IgnitionFile`` is built (same for both baseline and treated runs).

        ``"container"``
            Dissolves the full container polygon and hands it to FSPro as a
            **starting fire perimeter**, so every fire begins with the whole
            analysis unit burned.  Correct only for reproducing an observed
            fire — never for a treatment-effect study.
        ``"random"``
            Samples burnable pixels at random from the baseline LCP.  Only
            ``n_ignitions=1`` is accepted; see
            :func:`prepare_fspro_experiment` for a design-fire set.
        ``"fod"``
            Uses historical FPA-FOD point locations clipped to the
            container.  Requires *fod_gdf*; likewise single-ignition only.

        This parameter used to default to ``"container"`` (defect #1).
        Omitting it now raises.
    n_ignitions : int
        Number of random ignition points for ``ignition_mode="random"``.
        Default 200.
    fod_gdf : GeoDataFrame, optional
        FPA-FOD (or equivalent) points, pre-filtered.  Required when
        ``ignition_mode="fod"``.
    ignition_seed : int, optional
        Random seed for ``ignition_mode="random"``.
    **fspro_kwargs
        Additional overrides for any key in ``_FSPRO_DEFAULTS``, e.g.
        ``CROWN_FIRE_METHOD="Scott/Reinhardt"``, ``SavePerimeters=0``.
        ``pyrome_col`` is also intercepted here if present.

    Returns
    -------
    dict
        Manifest with keys:

        ``"baseline_lcp_path"``    : Path
        ``"treated_lcp_path"``     : Path
        ``"ignition_path"``        : Path
        ``"fspro_input_path"``     : Path  (shared by both runs)
        ``"baseline_outputs_dir"`` : Path  (empty; for Windows execution)
        ``"treated_outputs_dir"``  : Path  (empty; for Windows execution)
        ``"pyrome_id"``            : str
        ``"seed"``                 : int
        ``"out_dir"``              : Path
        ``"manifest_path"``        : Path

    Raises
    ------
    FileNotFoundError
        If either LCP path does not exist.

    Examples
    --------
    On Mac (data prep):

    >>> manifest = prepare_counterfactual_fspro(
    ...     container_gdf=huc12,
    ...     out_dir=OUT_DIR,
    ...     weather_dir=WEATHER_DIR,
    ...     pyromes_gdf=PYROMES_GDF,
    ...     pyrome_col="PYROME",
    ...     baseline_lcp_path=OUT_DIR / "lcp" / "baseline_lcp.tif",
    ...     treated_lcp_path=OUT_DIR / "lcp" / "treated_lcp.tif",
    ...     num_fires=1000,
    ...     duration=7,
    ...     resolution=90,
    ... )
    >>> patch_fspro_input_paths(
    ...     manifest["fspro_input_path"],
    ...     mac_prefix="/Users/mcc/Library/CloudStorage/Box-Box",
    ...     win_prefix="Z:\\\\",
    ... )

    On Windows (run both scenarios using the shared input file):

    >>> run_fspro(exe, manifest["baseline_lcp_path"],
    ...           manifest["fspro_input_path"], manifest["baseline_outputs_dir"])
    >>> run_fspro(exe, manifest["treated_lcp_path"],
    ...           manifest["fspro_input_path"], manifest["treated_outputs_dir"])

    Back on Mac (post-process and compare):

    >>> bl = postprocess_fspro_outputs(manifest["baseline_outputs_dir"],
    ...                                ref_lcp=manifest["baseline_lcp_path"])
    >>> tr = postprocess_fspro_outputs(manifest["treated_outputs_dir"],
    ...                                ref_lcp=manifest["treated_lcp_path"])
    >>> delta_bp = delta_burn_probability(bl["burn_prob_tif"], tr["burn_prob_tif"])
    """
    import geopandas as gpd
    import rasterio
    from fb_tools.fuelscape.lcp import (
        create_container_ignition,
        create_random_ignitions,
        create_fod_ignitions,
    )
    from fb_tools.models.fspro import build_treatment_pair
    from fb_tools.utils.geo import lookup_pyrome

    # Accept a file path for pyromes_gdf
    if isinstance(pyromes_gdf, (str, Path)):
        pyromes_gdf = gpd.read_file(pyromes_gdf)

    # ── Validate LCP paths ────────────────────────────────────────────────────
    baseline_lcp_path = Path(baseline_lcp_path)
    treated_lcp_path  = Path(treated_lcp_path)
    if not baseline_lcp_path.exists():
        raise FileNotFoundError(f"baseline_lcp_path not found: {baseline_lcp_path}")
    if not treated_lcp_path.exists():
        raise FileNotFoundError(f"treated_lcp_path not found: {treated_lcp_path}")

    # ── Output directories ────────────────────────────────────────────────────
    out_dir    = Path(out_dir).resolve()
    ign_dir    = out_dir / "ignitions"
    inputs_dir = out_dir / "fspro_inputs"
    bl_out_dir = out_dir / "baseline" / "outputs"
    tr_out_dir = out_dir / "treated"  / "outputs"
    for d in (ign_dir, inputs_dir, bl_out_dir, tr_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. LCP metadata (CRS from baseline) ──────────────────────────────────
    with rasterio.open(baseline_lcp_path) as src:
        lcp_crs     = src.crs
        pixel_count = src.width * src.height
    print(f"[prepare_counterfactual_fspro] Baseline LCP : {baseline_lcp_path.name} "
          f"({pixel_count:,} pixels, CRS: {lcp_crs.to_epsg() or lcp_crs.to_string()})")
    print(f"[prepare_counterfactual_fspro] Treated LCP  : {treated_lcp_path.name}")

    # ── 2. Dominant pyrome ────────────────────────────────────────────────────
    pyrome_col = fspro_kwargs.pop("pyrome_col", "Pyrome_ID")
    # Reproject container to pyromes CRS for the overlap query — NOT LCP CRS.
    container_pyromes_crs = container_gdf.to_crs(pyromes_gdf.crs)
    try:
        container_union = container_pyromes_crs.geometry.union_all()
    except AttributeError:
        container_union = container_pyromes_crs.geometry.unary_union

    pyrome_id = str(lookup_pyrome(container_union, pyromes_gdf, pyrome_col=pyrome_col))
    print(f"[prepare_counterfactual_fspro] Dominant pyrome: {pyrome_id}")

    # ── 3. Weather (once, shared by both runs) ────────────────────────────────
    print("[prepare_counterfactual_fspro] Loading pyrome weather …")
    wx = _load_weather_for_pyrome(
        pyrome_id,
        weather_dir,
        ignition_season_day=_resolve_ignition_season_day(
            ignition_season_day, current_erc_start_doy, current_erc_n_days
        ),
        current_erc_mode=current_erc_mode,
        analog_year=analog_year,
        current_erc_percentile=current_erc_percentile,
        max_lag=int(fspro_kwargs.get("MaxLag", 30)),
        duration=duration,
        erc_classes=erc_classes,
        gridmet_csv=gridmet_csv,
    )

    # ── 4. Ignition file (shared by baseline and treated runs) ───────────────
    container_proj = container_gdf.to_crs(lcp_crs)
    ign_path = _resolve_single_ignition(
        ignition_mode, container_proj, baseline_lcp_path, ign_dir,
        n_ignitions, fod_gdf, ignition_seed,
        caller="prepare_counterfactual_fspro",
    )
    print(f"[prepare_counterfactual_fspro] Ignition ({ignition_mode}): {ign_path.name}")

    # ── 5. Shared FSPro input file with fixed SPOTTING_SEED ───────────────────
    sim_params = {"NumFires": num_fires, "Duration": duration, "Resolution": resolution}
    sim_params.update(fspro_kwargs)

    fspro_input_path = build_treatment_pair(
        out_path      = inputs_dir / "fspro.input",
        ignition_file = ign_path,
        wind_cells    = wx["wind_cells"],
        calm_value    = wx["calm_value"],
        erc_historic  = wx["erc_historic"],
        erc_avg       = wx["erc_avg"],
        erc_std       = wx["erc_std"],
        erc_classes   = wx["erc_classes"],
        current_erc   = wx["current_erc"],
        speed_breaks  = wx["speed_breaks"],
        dir_breaks    = wx["dir_breaks"],
        seed          = seed,
        **sim_params,
    )
    print(f"[prepare_counterfactual_fspro] FSPro input (shared): {fspro_input_path.name} "
          f"(NumFires={num_fires}, Duration={duration}, Resolution={resolution}m, "
          f"seed={seed})")

    # ── 6. Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "baseline_lcp_path":    baseline_lcp_path,
        "treated_lcp_path":     treated_lcp_path,
        "ignition_path":        ign_path,
        "fspro_input_path":     fspro_input_path,
        "baseline_outputs_dir": bl_out_dir,
        "treated_outputs_dir":  tr_out_dir,
        "pyrome_id":            pyrome_id,
        "seed":                 seed,
        "out_dir":              out_dir,
    }
    manifest_path = out_dir / "run_manifest.json"
    _write_manifest(manifest, manifest_path)
    manifest["manifest_path"] = manifest_path

    print(f"[prepare_counterfactual_fspro] Done. Manifest → {manifest_path}")
    return manifest


def prepare_fspro_experiment(
    treatments_gdf,
    values_gdf,
    out_dir: "str | Path",
    weather_dir: "str | Path",
    pyromes_gdf,
    lcps: "dict | None" = None,
    contrasts: "list | None" = None,
    domain_gdf=None,
    baseline_lcp_path: "str | Path | None" = None,
    treated_lcp_path: "str | Path | None" = None,
    fod_gdf=None,
    density: "dict | None" = None,
    density_bandwidth_m: "float | None" = None,
    wind_from_deg: "float | None" = None,
    cone_half_angle_deg: "float | None" = None,
    cone_coverage: float = 0.5,
    n_ignitions: int = 10,
    dist_band_km: tuple = (5.0, 25.0),
    n_bearing_strata: int = 3,
    n_distance_strata: int = 2,
    footprint_acres: "float | None" = 10.0,
    require_ordering: bool = True,
    num_fires: int = 4000,
    duration: int = 21,
    resolution: float = 90.0,
    erc_classes: "np.ndarray | None" = None,
    gridmet_csv: "str | Path | None" = None,
    ignition_season_day: int = 80,
    current_erc_mode: str = "analog_year",
    analog_year: "int | None" = None,
    current_erc_percentile: float = 80.0,
    current_erc_start_doy: "int | None" = None,
    current_erc_n_days: "int | None" = None,
    seed: int = 617327,
    ignition_seed=None,
    **fspro_kwargs,
) -> dict:
    """Prepare an N-arm FSPro transmission experiment from design fires.

    The Phase 2 entry point, superseding
    :func:`prepare_counterfactual_fspro` (two arms, one ignition) and
    :func:`prepare_container_fspro` (one arm, one ignition).  It assembles:

    - **N treatment arms** — ``lcps={"untreated": ..., "background": ...,
      "coswap": ...}`` — with declared *contrasts*.  Because ``TestFSPro.exe``
      takes the LCP as a runtime argument, **one input file serves every arm**,
      so all pairwise contrasts share an ignition, a weather stream, and a
      ``SPOTTING_SEED``, and cost scales as arms × design fires.
    - **Design fires** selected by
      :func:`~fb_tools.fuelscape.ignitions.select_design_ignitions`: candidates
      drawn from an FPA-FOD ignition-likelihood surface, filtered so the
      treatment lies inside the ignition's downwind spread cone and nearer than
      the values, then stratified over approach bearing and distance.  Each
      carries a weight *w_i* for ``Σ_i w_i · TF_ij``.
    - **One FSPro input file and one run per ignition per arm**, tabulated in
      ``runs.csv`` for :func:`~fb_tools.models.fspro.run_fspro_batch`.

    The simulation domain is **not** derived here and is never used to clip.
    Domain delineation is data prep, decided per study area; pass it as
    *domain_gdf* for validation and manifest provenance only.  Clipping the
    landscape to an analysis unit makes its edge a boundary fire cannot cross,
    which truncates growth, biases burn probability low near the edge, and
    leaves nowhere for fire to be transmitted *to* — TF_ij becomes
    unmeasurable.

    Parameters
    ----------
    treatments_gdf : geopandas.GeoDataFrame
        Treatment polygons under test.  Any CRS.
    values_gdf : geopandas.GeoDataFrame or None
        Values to protect (WUI, structures, POD), orienting the
        ignition → treatment → values geometry.  ``None`` skips the ordering
        and downwind checks.
    out_dir : str or Path
        Root output directory.  Sub-directories created automatically::

            out_dir/
              ignitions/      ← ign_XXX.shp  (one feature each)
              fspro_inputs/   ← ign_XXX.input  (one per ignition, all arms)
              runs.csv        ← scenario table for run_fspro_batch
              ignition_density.tif   (when a density surface is built here)
              run_manifest.json

    weather_dir : str or Path
        Root weather cache directory (``pyrome_erc/``, ``pyrome_wind/``).
    pyromes_gdf : geopandas.GeoDataFrame or str or Path
        NIFC pyrome polygons, or a path read with ``geopandas.read_file``.
    lcps : dict, optional
        Ordered ``{arm_name: lcp_path}``.  Arm names become the FSPro output
        basenames and the ``Arm`` column of ``runs.csv``.  Every arm must sit
        on a congruent grid — asserted before anything is written.
    contrasts : list of tuple, optional
        Declared ``(arm_a, arm_b)`` comparisons.  Defaults to every ordered
        pair in arm order, so three arms give ``untreated−background``,
        ``untreated−coswap``, and ``background−coswap``.
    domain_gdf : geopandas.GeoDataFrame, optional
        Pre-defined simulation domain.  **Validation and provenance only** —
        never used to clip.  A warning is printed if the landscapes do not
        cover it.
    baseline_lcp_path, treated_lcp_path : str or Path, optional
        Two-arm convenience shim, equivalent to
        ``lcps={"baseline": ..., "treated": ...}``.  Cannot be combined with
        *lcps*.
    fod_gdf : geopandas.GeoDataFrame, optional
        Historical ignition points (FPA-FOD, pre-filtered to large-fire-capable
        records).  Used to build the density surface when *density* is not
        supplied.  Omitting both gives uniform candidate weights — Ager's
        assumption, which
        :func:`~fb_tools.fuelscape.ignitions.check_ignition_clustering` rejects
        for Colorado.
    density : dict, optional
        A pre-built surface from
        :func:`~fb_tools.fuelscape.ignitions.ignition_density_surface`.
        Takes precedence over *fod_gdf*.
    density_bandwidth_m : float, optional
        Kernel bandwidth when building the surface from *fod_gdf*.
    wind_from_deg : float, optional
        Dominant wind direction, degrees FROM.  Derived from the pyrome wind
        climatology when omitted.
    cone_half_angle_deg : float, optional
        Half-width of the downwind spread cone.  When omitted it is derived
        from the pyrome wind-direction spread at *cone_coverage* via
        :func:`~fb_tools.fuelscape.ignitions.wind_cone_half_angle`.
    cone_coverage : float
        Fraction of non-calm fire-hour wind frequency the derived cone spans.
        Default 0.5.
    n_ignitions : int
        Number of design fires.  Default 10 — the Part A target of 8–12
        weighted source locations rather than a dense ensemble.
    dist_band_km : tuple of float
        ``(min, max)`` upwind distance from the treatment centroid, km.
        Default ``(5, 25)``.
    n_bearing_strata, n_distance_strata : int
        Stratification grid.  Default 3 × 2.
    footprint_acres : float or None
        Ignition footprint area.  Default 10 ac; ``None`` gives a half-pixel
        circle.
    require_ordering : bool
        Require the treatment nearer the ignition than the values are.
    num_fires : int
        ``NumFires`` per run.  Default 4000 — P0.1 measured per-pixel ΔBP
        noise at p95 ≈ 0.011 there, against 0.07 at 100 fires.
    duration : int
        ``Duration`` in days.  Default 21.
    resolution : float
        Output cell size, metres.  Default 90.
    erc_classes : np.ndarray, optional
        Pre-computed ERC class table ``(5, 10)``.
    gridmet_csv : str or Path, optional
        GridMET CSV, used when *erc_classes* is not supplied.
    ignition_season_day : int
        1-based fire-season day of ignition (1 = April 1).  Default 80.
    current_erc_mode : {"analog_year", "percentile", "median", "observed"}
        How the antecedent ERC stream is built.  Default ``"analog_year"``.
    analog_year : int, optional
        Force a specific year for ``current_erc_mode="analog_year"``.
    current_erc_percentile : float
        Quantile for ``current_erc_mode="percentile"``.  Default 80.
    current_erc_start_doy, current_erc_n_days : int, optional
        Deprecated pre-P1.2 window keywords; a start other than day 1 raises.
    seed : int
        Shared ``SPOTTING_SEED``.  Default 617327.  Note P0.1: this seeds only
        spotting, **not** the ERC stream or wind draws, so it buys no variance
        reduction and pairing between arms is statistical, never exact.  Every
        reported Δ needs a null band.
    ignition_seed : int, optional
        Random seed for design-fire selection.
    **fspro_kwargs
        Overrides for ``_FSPRO_DEFAULTS``.  ``pyrome_col`` is intercepted.
        ``SavePerimeters`` is forced to 1.

    Returns
    -------
    dict
        ``"arms"``                : dict, arm name → LCP Path
        ``"contrasts"``           : list of (arm_a, arm_b)
        ``"grid"``                : shared grid metadata
        ``"domain"``              : domain provenance (empty without *domain_gdf*)
        ``"pyrome_id"``           : str
        ``"wind_from_deg"``, ``"downwind_az"``, ``"cone_half_angle_deg"``
        ``"seed"``                : int
        ``"ignition_shapefiles"`` : list of Path
        ``"fspro_input_paths"``   : list of Path
        ``"ignitions_gdf"``       : GeoDataFrame with ``ign_id`` and ``w_i``
        ``"ignition_weights"``    : dict, ``ign_id`` → *w_i*
        ``"density_raster"``      : Path or None
        ``"runs_df"``, ``"runs_csv"``, ``"out_dir"``, ``"manifest_path"``

    Raises
    ------
    ValueError
        If fewer than two arms are given, if arm landscapes are not congruent,
        or if no design fire survives the transmission-geometry filters.
    FileNotFoundError
        If any arm's landscape does not exist.

    Notes
    -----
    ``runs_df`` has one row per (ignition × arm) with columns ``Scenario``,
    ``Arm``, ``LCP``, ``FSPro_input``, ``output_basename``, ``w_i``.  Run it on
    Windows after patching ``IgnitionFile`` paths with
    :func:`patch_fspro_input_paths`.
    """
    import geopandas as gpd
    import pandas as pd
    from fb_tools.fuelscape.ignitions import (
        ignition_density_surface,
        select_design_ignitions,
        wind_cone_half_angle,
        write_density_raster,
    )
    from fb_tools.models.fspro import build_treatment_pair
    from fb_tools.utils.geo import lookup_pyrome
    from fb_tools.weather.hrrr import dominant_wind_direction

    if isinstance(pyromes_gdf, (str, Path)):
        pyromes_gdf = gpd.read_file(pyromes_gdf)

    # ── Arms, grid congruence, domain provenance (P2.0 / P2.0b) ──────────────
    arms, contrasts = _resolve_arms(lcps, baseline_lcp_path, treated_lcp_path,
                                    contrasts)
    grid = _assert_grid_congruence(arms)
    domain_info = _check_domain(domain_gdf, grid, label="simulation domain")
    ref_lcp = next(iter(arms.values()))

    out_dir    = Path(out_dir).resolve()
    ign_dir    = out_dir / "ignitions"
    inputs_dir = out_dir / "fspro_inputs"
    for d in (ign_dir, inputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_fspro_experiment] {len(arms)} arm(s): {list(arms)}")
    print(f"[prepare_fspro_experiment] {len(contrasts)} contrast(s): "
          + ", ".join(f"{a}−{b}" for a, b in contrasts))

    # ── Dominant pyrome from the treatments + values footprint ───────────────
    pyrome_col = fspro_kwargs.pop("pyrome_col", "Pyrome_ID")
    geoms = list(treatments_gdf.to_crs(pyromes_gdf.crs).geometry)
    if values_gdf is not None:
        geoms += list(values_gdf.to_crs(pyromes_gdf.crs).geometry)
    analysis_area = gpd.GeoDataFrame(geometry=geoms, crs=pyromes_gdf.crs)
    try:
        analysis_union = analysis_area.geometry.union_all()
    except AttributeError:
        analysis_union = analysis_area.geometry.unary_union
    pyrome_id = str(lookup_pyrome(analysis_union, pyromes_gdf,
                                  pyrome_col=pyrome_col))
    print(f"[prepare_fspro_experiment] Dominant pyrome: {pyrome_id}")

    # ── Weather (shared by every ignition and every arm) ─────────────────────
    print("[prepare_fspro_experiment] Loading pyrome weather …")
    wx = _load_weather_for_pyrome(
        pyrome_id, weather_dir,
        ignition_season_day=_resolve_ignition_season_day(
            ignition_season_day, current_erc_start_doy, current_erc_n_days
        ),
        current_erc_mode=current_erc_mode,
        analog_year=analog_year,
        current_erc_percentile=current_erc_percentile,
        max_lag=int(fspro_kwargs.get("MaxLag", 30)),
        duration=duration,
        erc_classes=erc_classes,
        gridmet_csv=gridmet_csv,
    )

    # ── Wind direction and spread-cone half-angle ────────────────────────────
    wind_dir = Path(weather_dir) / "pyrome_wind"
    if wind_from_deg is None:
        wind_from_deg = dominant_wind_direction(pyrome_id, wind_dir)
        print(f"[prepare_fspro_experiment] Dominant wind FROM "
              f"(pyrome climatology): {wind_from_deg:.0f}°")
    if cone_half_angle_deg is None:
        cone = wind_cone_half_angle(pyrome_id, wind_dir, coverage=cone_coverage)
        cone_half_angle_deg = cone["half_angle_deg"]
        print(f"[prepare_fspro_experiment] Spread cone ±"
              f"{cone_half_angle_deg:.0f}° — the narrowest arc holding "
              f"{cone_coverage:.0%} of non-calm fire-hour wind frequency "
              f"(centred {cone['center_az']:.0f}°).")

    # ── Ignition-likelihood surface (P2.2) ───────────────────────────────────
    density_raster = None
    if density is None and fod_gdf is not None:
        print("[prepare_fspro_experiment] Building ignition density surface …")
        density = ignition_density_surface(
            fod_gdf, ref_lcp, bandwidth_m=density_bandwidth_m
        )
        density_raster = write_density_raster(
            density, out_dir / "ignition_density.tif"
        )
    elif density is None:
        print("[prepare_fspro_experiment] No fod_gdf or density surface — "
              "candidates will be weighted uniformly (Ager's assumption). "
              "check_ignition_clustering() rejects it for Colorado.")

    # ── Design fires (P2.3 / P2.4 / P2.5) ────────────────────────────────────
    ign = select_design_ignitions(
        treatments_gdf=treatments_gdf,
        values_gdf=values_gdf,
        lcp_fp=ref_lcp,
        out_dir=ign_dir,
        wind_from_deg=wind_from_deg,
        density=density,
        n_ignitions=n_ignitions,
        dist_band_km=dist_band_km,
        cone_half_angle_deg=cone_half_angle_deg,
        n_bearing_strata=n_bearing_strata,
        n_distance_strata=n_distance_strata,
        footprint_acres=footprint_acres,
        require_ordering=require_ordering,
        seed=ignition_seed,
    )
    ign_shps    = ign["ignition_shapefiles"]
    ignitions   = ign["ignitions_gdf"]
    weights     = {int(r.ign_id): float(r.w_i) for r in ignitions.itertuples()}

    # ── One FSPro input per ignition; one run per (ignition × arm) ───────────
    sim_params = {
        "NumFires": num_fires, "Duration": duration, "Resolution": resolution,
    }
    sim_params.update(fspro_kwargs)
    sim_params["SavePerimeters"] = 1  # always on — growth analysis needs it

    input_paths = []
    run_rows = []
    for i, ign_shp in enumerate(ign_shps):
        scenario = f"ign_{i:03d}"
        inp = build_treatment_pair(
            out_path      = inputs_dir / f"{scenario}.input",
            ignition_file = ign_shp,
            wind_cells    = wx["wind_cells"],
            calm_value    = wx["calm_value"],
            erc_historic  = wx["erc_historic"],
            erc_avg       = wx["erc_avg"],
            erc_std       = wx["erc_std"],
            erc_classes   = wx["erc_classes"],
            current_erc   = wx["current_erc"],
            speed_breaks  = wx["speed_breaks"],
            dir_breaks    = wx["dir_breaks"],
            seed          = seed,
            **sim_params,
        )
        input_paths.append(inp)
        for arm_name, arm_lcp in arms.items():
            run_rows.append({
                "Scenario":        scenario,
                "Arm":             arm_name,
                "LCP":             str(arm_lcp),
                "FSPro_input":     str(inp),
                "output_basename": arm_name,
                "w_i":             weights.get(i, float("nan")),
            })

    runs_df = pd.DataFrame(run_rows)
    runs_csv = out_dir / "runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"[prepare_fspro_experiment] {len(ign_shps)} design fire(s) × "
          f"{len(arms)} arm(s) = {len(runs_df)} FSPro runs "
          f"(NumFires={num_fires}, Duration={duration}, "
          f"Resolution={resolution}m)")

    manifest = {
        "arms":               {k: str(v) for k, v in arms.items()},
        "contrasts":          [list(c) for c in contrasts],
        "grid":               {
            "crs":    grid["crs"].to_string(),
            "width":  grid["width"],
            "height": grid["height"],
            "res":    list(grid["res"]),
            "bounds": [round(v, 2) for v in grid["bounds"]],
        },
        "domain":             domain_info,
        "pyrome_id":          pyrome_id,
        "wind_from_deg":      float(wind_from_deg),
        "downwind_az":        ign["downwind_az"],
        "cone_half_angle_deg": float(cone_half_angle_deg),
        "seed":               seed,
        "ignition_shapefiles": [str(p) for p in ign_shps],
        "fspro_input_paths":  [str(p) for p in input_paths],
        "ignition_weights":   weights,
        "uniform_weights":    ign["uniform_weights"],
        "density_raster":     str(density_raster) if density_raster else None,
        "num_fires":          num_fires,
        "duration":           duration,
        "resolution":         resolution,
        "runs_csv":           runs_csv,
        "out_dir":            out_dir,
    }
    manifest_path = out_dir / "run_manifest.json"
    _write_manifest(manifest, manifest_path)

    manifest["manifest_path"]  = manifest_path
    manifest["runs_df"]        = runs_df
    manifest["ignitions_gdf"]  = ignitions
    manifest["arms"]           = arms
    manifest["contrasts"]      = contrasts
    manifest["density_raster"] = density_raster
    print(f"[prepare_fspro_experiment] Done. Manifest → {manifest_path}")
    return manifest


def prepare_counterfactual_ignition_set(*args, **kwargs) -> dict:
    """Deprecated alias for :func:`prepare_fspro_experiment`.

    .. deprecated:: Phase 2

       Renamed and generalized.  The replacement takes N arms via
       ``lcps={arm: path}`` (the ``baseline_lcp_path`` / ``treated_lcp_path``
       shim still works), accepts a pre-defined ``domain_gdf``, asserts grid
       congruence across arms, draws design fires from an FPA-FOD
       ignition-likelihood surface with weights *w_i*, and replaces the
       zero-width ray test with a wind-derived downwind spread cone.

       Two argument changes to note when migrating: ``sector_deg`` and
       ``require_treatment_intersect`` are gone — the cone geometry is set by
       ``cone_half_angle_deg`` / ``cone_coverage`` and is always applied.
    """
    for gone in ("sector_deg", "require_treatment_intersect"):
        if gone in kwargs:
            raise TypeError(
                f"{gone!r} is no longer accepted. The zero-width downwind ray "
                "test it configured has been replaced by a spread cone — see "
                "cone_half_angle_deg and cone_coverage on "
                "prepare_fspro_experiment()."
            )
    print("  [prepare_counterfactual_ignition_set] Deprecated — "
          "call prepare_fspro_experiment() instead.")
    return prepare_fspro_experiment(*args, **kwargs)

def patch_fspro_input_paths(
    input_file: "str | Path",
    mac_prefix: str,
    win_prefix: str,
    inplace: bool = True,
) -> Path:
    """
    Translate Mac absolute paths to Windows paths in an FSPro input file.

    ``prepare_container_fspro`` writes ``IgnitionFile`` (and any other path
    fields) as Mac absolute paths.  Before executing on Windows, call this
    function once to replace the Mac filesystem prefix with its Windows
    equivalent so that TestFSPro.exe can locate the ignition shapefile.

    Parameters
    ----------
    input_file : str or Path
        Path to the FSPro ``.input`` file to patch (on Mac, via Box).
    mac_prefix : str
        The Mac path prefix to replace (e.g.
        ``"/Users/mcc/Library/CloudStorage/Box-Box"``).
        All occurrences in the file are replaced, so partial prefixes work too.
    win_prefix : str
        The Windows replacement prefix (e.g. ``"Z:\\\\"``, or the UNC path
        ``"\\\\\\\\Mac\\\\Home\\\\Library\\\\CloudStorage\\\\Box-Box"``).
        Forward slashes are converted to backslashes automatically after
        substitution.
    inplace : bool
        If ``True`` (default), overwrite the existing file.  If ``False``,
        write a sibling file with ``_win`` appended before the extension and
        leave the original unchanged.

    Returns
    -------
    Path
        Absolute path to the patched file (same as *input_file* when
        ``inplace=True``).

    Examples
    --------
    On Mac, before copying / syncing to Windows for execution:

    >>> from fb_tools import patch_fspro_input_paths
    >>> patch_fspro_input_paths(
    ...     fb_inputs["fspro_input_path"],
    ...     mac_prefix="/Users/mcc/Library/CloudStorage/Box-Box",
    ...     win_prefix="Z:\\\\",
    ... )

    Parallels shared-folder UNC variant (no Box native install on Windows):

    >>> patch_fspro_input_paths(
    ...     fb_inputs["fspro_input_path"],
    ...     mac_prefix="/Users/mcc",
    ...     win_prefix="\\\\\\\\Mac\\\\Home",
    ... )
    """
    input_file = Path(input_file).resolve()
    text = input_file.read_text()

    # Replace prefix and normalise to Windows backslashes
    patched = text.replace(mac_prefix, win_prefix)
    # Convert any remaining forward slashes in path lines to backslashes.
    # Only touch lines that look like path fields (contain the prefix or :\).
    lines = []
    for line in patched.splitlines(keepends=True):
        stripped = line.lstrip()
        is_path_line = (
            ":" in line
            and any(
                stripped.lower().startswith(k)
                for k in ("ignitionfile", "barrierfile", "outputfile")
            )
        )
        if is_path_line:
            # Replace forward slashes only in the value portion (after the colon)
            key, _, val = line.partition(":")
            val_win = val.replace("/", "\\")
            lines.append(f"{key}:{val_win}")
        else:
            lines.append(line)
    patched = "".join(lines)

    if inplace:
        out_path = input_file
    else:
        out_path = input_file.with_stem(input_file.stem + "_win")

    out_path.write_text(patched)
    print(f"[patch_fspro_input_paths] Patched → {out_path.name}")
    return out_path
