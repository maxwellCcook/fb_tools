"""
Spatial-container FSPro orchestration.

High-level entry point that takes any spatial container (HUC12, fireshed,
POD, county) as a GeoDataFrame and assembles a complete, ready-to-run FSPro
simulation directory on macOS, ready for execution on Windows.

Public API
----------
prepare_container_fspro
    Orchestrates LCP download, ignition creation, weather extraction, and
    FSPro input file assembly.  Returns a manifest dict with all file paths.

postprocess_fspro_outputs
    Converts FSPro ASC output grids to GeoTIFFs, optionally clips to the
    container boundary, and stacks into a multi-band output.

prepare_counterfactual_fspro
    Builds a paired baseline/treated run using a shared SPOTTING_SEED for
    clean counterfactual comparison.

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


def _load_weather_for_pyrome(
    pyrome_id: "str | int",
    weather_dir: "str | Path",
    current_erc_start_doy: int,
    current_erc_n_days: int,
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

    current_erc_start_doy : int
        1-based fire-season DOY (1 = April 1) at which ``CurrentERCValues``
        begins.
    current_erc_n_days : int
        Length of the ``CurrentERCValues`` sequence.
    erc_classes : np.ndarray, optional
        Pre-computed ERC class table, shape ``(5, 10)``.  If provided,
        ``gridmet_csv`` is ignored for class building.
    gridmet_csv : str or Path, optional
        Path to the GEE-exported GridMET CSV.  Required when ``erc_classes``
        is ``None``; used to build the ERC class table on the fly.

    Returns
    -------
    dict
        Keys: ``wind_cells``, ``calm_value``, ``erc_historic``,
        ``erc_avg``, ``erc_std``, ``erc_classes``, ``current_erc``.

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

    # Historic ERC
    erc_meta = load_gridmet_pyrome_cache(
        pyrome_id, erc_dir, return_meta=True
    )
    erc_historic = np.array(erc_meta["HistoricERCValues"], dtype=float)

    # Per-DOY stats
    stats = build_erc_stats({pyrome_id: erc_historic})
    erc_avg = stats[pyrome_id]["avg"]
    erc_std = stats[pyrome_id]["std"]

    # Current ERC (climatological median window)
    current_erc = build_current_erc_values(
        {pyrome_id: erc_historic},
        start_doy=current_erc_start_doy,
        n_days=current_erc_n_days,
    )[pyrome_id]

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
        erc_classes_dict = build_erc_classes(df_p)
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
        "wind_cells":   wind_cells,
        "calm_value":   calm_value,
        "erc_historic": erc_historic,
        "erc_avg":      erc_avg,
        "erc_std":      erc_std,
        "erc_classes":  erc_classes,
        "current_erc":  current_erc,
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
    current_erc_start_doy: int = 91,
    current_erc_n_days: int = 79,
    ignition_mode: str = "container",
    n_ignitions: int = 200,
    fod_gdf=None,
    ignition_seed=None,
    **fspro_kwargs,
) -> dict:
    """Prepare a complete FSPro simulation directory for a spatial container.

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
    current_erc_start_doy : int
        1-based fire-season DOY (1 = April 1) at which ``CurrentERCValues``
        begins.  Default 91 (≈ July 1).
    current_erc_n_days : int
        Length of the current-season ERC sequence.  Default 79.
    ignition_mode : {"container", "random", "fod"}
        Controls how the ``IgnitionFile`` is built.

        ``"container"`` (default)
            Dissolves the full container polygon.  Uniform sampling across
            the entire analysis area.
        ``"random"``
            Samples *n_ignitions* burnable pixels at random from within the
            container and buffers each to a small circle.  Requires
            ``lcp_path`` to be known (i.e. the LCP is available before
            calling this function, or will be downloaded by it).
        ``"fod"``
            Uses historical FPA-FOD (or equivalent) point locations clipped
            to the container.  Requires *fod_gdf*.
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
        current_erc_start_doy=current_erc_start_doy,
        current_erc_n_days=current_erc_n_days,
        erc_classes=erc_classes,
        gridmet_csv=gridmet_csv,
    )

    # ── 4. Ignition file ──────────────────────────────────────────────────────
    if ignition_mode == "random":
        ign_path = create_random_ignitions(
            container_proj, n_ignitions, lcp_path,
            ign_dir / "random_ignitions.shp", seed=ignition_seed,
        )
    elif ignition_mode == "fod":
        if fod_gdf is None:
            raise ValueError("ignition_mode='fod' requires fod_gdf to be provided.")
        ign_path = create_fod_ignitions(
            container_proj, fod_gdf, lcp_path,
            ign_dir / "fod_ignitions.shp",
        )
    else:
        ign_path = create_container_ignition(
            container_proj, ign_dir / "container_ignition.shp"
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
        conversion.
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
            stacked.rio.to_raster(stacked_path, compress="deflate")
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
    current_erc_start_doy: int = 91,
    current_erc_n_days: int = 79,
    seed: int = 617327,
    ignition_mode: str = "container",
    n_ignitions: int = 200,
    fod_gdf=None,
    ignition_seed=None,
    **fspro_kwargs,
) -> dict:
    """Prepare paired baseline and treated FSPro runs for counterfactual analysis.

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
    current_erc_start_doy : int
        1-based fire-season DOY (1 = April 1) at which ``CurrentERCValues``
        begins.  Default 91 (≈ July 1).
    current_erc_n_days : int
        Length of the current-season ERC sequence.  Default 79.
    seed : int
        ``SPOTTING_SEED`` shared by both runs.  **Do not vary between
        baseline and treated.**  Default 617327.
    ignition_mode : {"container", "random", "fod"}
        Controls how the shared ``IgnitionFile`` is built (same for both
        baseline and treated runs).

        ``"container"`` (default)
            Dissolves the full container polygon.
        ``"random"``
            Samples *n_ignitions* burnable pixels at random from the
            baseline LCP and buffers each to a small circle.
        ``"fod"``
            Uses historical FPA-FOD point locations clipped to the
            container.  Requires *fod_gdf*.
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
        current_erc_start_doy=current_erc_start_doy,
        current_erc_n_days=current_erc_n_days,
        erc_classes=erc_classes,
        gridmet_csv=gridmet_csv,
    )

    # ── 4. Ignition file (shared by baseline and treated runs) ───────────────
    container_proj = container_gdf.to_crs(lcp_crs)
    if ignition_mode == "random":
        ign_path = create_random_ignitions(
            container_proj, n_ignitions, baseline_lcp_path,
            ign_dir / "random_ignitions.shp", seed=ignition_seed,
        )
    elif ignition_mode == "fod":
        if fod_gdf is None:
            raise ValueError("ignition_mode='fod' requires fod_gdf to be provided.")
        ign_path = create_fod_ignitions(
            container_proj, fod_gdf, baseline_lcp_path,
            ign_dir / "fod_ignitions.shp",
        )
    else:
        ign_path = create_container_ignition(
            container_proj, ign_dir / "container_ignition.shp"
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


def prepare_counterfactual_ignition_set(
    treatments_gdf,
    values_gdf,
    out_dir: "str | Path",
    weather_dir: "str | Path",
    pyromes_gdf,
    baseline_lcp_path: "str | Path",
    treated_lcp_path: "str | Path",
    wind_from_deg: "float | None" = None,
    n_ignitions: int = 15,
    dist_band_km: tuple = (2.0, 10.0),
    sector_deg: float = 45.0,
    require_treatment_intersect: bool = True,
    num_fires: int = 1000,
    duration: int = 5,
    resolution: float = 90.0,
    erc_classes: "np.ndarray | None" = None,
    gridmet_csv: "str | Path | None" = None,
    current_erc_start_doy: int = 91,
    current_erc_n_days: int = 79,
    seed: int = 617327,
    ignition_seed=None,
    **fspro_kwargs,
) -> dict:
    """Prepare a per-ignition counterfactual FSPro experiment ("upwind toward values").

    Unlike :func:`prepare_counterfactual_fspro` — which builds a single
    ``IgnitionFile`` that FSPro treats as one fire — this function generates
    *N independent ignitions* placed upwind of a treatment cluster, and writes
    one FSPro input file per ignition.  Each ignition becomes its own FSPro run
    (``NumFires`` simulations), so per-ignition burn-probability and fire-growth
    results can be aggregated across the ensemble.

    All ignitions share the same pyrome weather and the same ``SPOTTING_SEED``,
    and each input file is run against both the baseline and treated LCP — a
    clean paired counterfactual.  ``SavePerimeters`` is forced on so per-fire
    daily perimeters are available for early/extreme growth analysis.

    Parameters
    ----------
    treatments_gdf : geopandas.GeoDataFrame
        Treatment polygons (the treatment group under test).  Any CRS.
    values_gdf : geopandas.GeoDataFrame
        Values / assets to protect (WUI, communities, structures, POD).
        Used to orient the ignition→treatments→values geometry.
    out_dir : str or Path
        Root output directory.  Sub-directories created automatically::

            out_dir/
              ignitions/      ← ign_XXX.shp  (one feature each)
              fspro_inputs/   ← ign_XXX.input  (one per ignition)
              runs.csv        ← scenario table for run_fspro_batch
              run_manifest.json

    weather_dir : str or Path
        Root weather cache directory (``pyrome_erc/``, ``pyrome_wind/``).
    pyromes_gdf : geopandas.GeoDataFrame or str or Path
        NIFC pyrome polygons, or a path read with ``geopandas.read_file``.
    baseline_lcp_path, treated_lcp_path : str or Path
        Pre-built landscape GeoTIFFs.  Must already exist.
    wind_from_deg : float, optional
        Dominant wind direction (degrees FROM).  When ``None`` (default), it
        is derived from the pyrome wind climatology via
        :func:`~fb_tools.weather.dominant_wind_direction`.
    n_ignitions : int
        Number of ignitions to generate.  Default 15.
    dist_band_km : tuple of float
        ``(min, max)`` upwind distance band from the treatment centroid, km.
        Default ``(2, 10)``.
    sector_deg : float
        Angular width of the upwind placement wedge.  Default 45.
    require_treatment_intersect : bool
        Keep only ignitions whose downwind ray crosses the treatments.
    num_fires : int
        ``NumFires`` per ignition.  Default 1000.
    duration : int
        ``Duration`` (max burn period, days).  Default 5 — the early window.
    resolution : float
        Output grid cell size, metres.  Default 90.
    erc_classes : np.ndarray, optional
        Pre-computed ERC class table ``(5, 10)``.
    gridmet_csv : str or Path, optional
        GridMET CSV, used when ``erc_classes`` is not supplied.
    current_erc_start_doy, current_erc_n_days : int
        Current-season ERC window (1 = April 1).  Defaults 91, 79.
    seed : int
        Shared ``SPOTTING_SEED`` for all runs.  Default 617327.
    ignition_seed : int, optional
        Random seed for ignition placement.
    **fspro_kwargs
        Overrides for ``_FSPRO_DEFAULTS``.  ``pyrome_col`` is intercepted.
        ``SavePerimeters`` is forced to 1.

    Returns
    -------
    dict
        Manifest with keys: ``baseline_lcp_path``, ``treated_lcp_path``,
        ``pyrome_id``, ``wind_from_deg``, ``downwind_az``, ``seed``,
        ``ignition_shapefiles`` (list), ``fspro_input_paths`` (list),
        ``ignitions_gdf`` (GeoDataFrame), ``runs_df`` (DataFrame for
        :func:`~fb_tools.models.fspro.run_fspro_batch`), ``runs_csv``,
        ``out_dir``, ``manifest_path``.

    Notes
    -----
    ``runs_df`` is laid out for ``run_fspro_batch`` — it organises outputs as
    ``output_root/<lcp_stem>/<ign_XXX>/``.  Run it on Windows after patching
    the ``IgnitionFile`` paths in each input file with
    :func:`patch_fspro_input_paths`.
    """
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from fb_tools.fuelscape.lcp import create_directional_ignitions
    from fb_tools.models.fspro import build_treatment_pair
    from fb_tools.utils.geo import lookup_pyrome
    from fb_tools.weather.hrrr import dominant_wind_direction

    if isinstance(pyromes_gdf, (str, Path)):
        pyromes_gdf = gpd.read_file(pyromes_gdf)

    baseline_lcp_path = Path(baseline_lcp_path)
    treated_lcp_path  = Path(treated_lcp_path)
    if not baseline_lcp_path.exists():
        raise FileNotFoundError(f"baseline_lcp_path not found: {baseline_lcp_path}")
    if not treated_lcp_path.exists():
        raise FileNotFoundError(f"treated_lcp_path not found: {treated_lcp_path}")

    out_dir    = Path(out_dir).resolve()
    ign_dir    = out_dir / "ignitions"
    inputs_dir = out_dir / "fspro_inputs"
    for d in (ign_dir, inputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    with rasterio.open(baseline_lcp_path) as src:
        lcp_crs = src.crs

    # ── Dominant pyrome from the treatments + values footprint ───────────────
    pyrome_col = fspro_kwargs.pop("pyrome_col", "Pyrome_ID")
    analysis_area = gpd.GeoDataFrame(
        geometry=list(treatments_gdf.geometry) + list(values_gdf.geometry),
        crs=treatments_gdf.crs,
    ).to_crs(pyromes_gdf.crs)
    try:
        analysis_union = analysis_area.geometry.union_all()
    except AttributeError:
        analysis_union = analysis_area.geometry.unary_union
    pyrome_id = str(lookup_pyrome(analysis_union, pyromes_gdf, pyrome_col=pyrome_col))
    print(f"[prepare_counterfactual_ignition_set] Dominant pyrome: {pyrome_id}")

    # ── Weather (shared by every ignition and both landscapes) ───────────────
    print("[prepare_counterfactual_ignition_set] Loading pyrome weather …")
    wx = _load_weather_for_pyrome(
        pyrome_id, weather_dir,
        current_erc_start_doy=current_erc_start_doy,
        current_erc_n_days=current_erc_n_days,
        erc_classes=erc_classes,
        gridmet_csv=gridmet_csv,
    )

    # ── Wind direction for upwind placement ──────────────────────────────────
    if wind_from_deg is None:
        wind_from_deg = dominant_wind_direction(
            pyrome_id, Path(weather_dir) / "pyrome_wind"
        )
        print(f"[prepare_counterfactual_ignition_set] Dominant wind FROM "
              f"(pyrome climatology): {wind_from_deg:.0f}°")

    # ── Directional ignition set ──────────────────────────────────────────────
    ign = create_directional_ignitions(
        treatments_gdf=treatments_gdf,
        values_gdf=values_gdf,
        wind_from_deg=wind_from_deg,
        lcp_fp=baseline_lcp_path,
        out_dir=ign_dir,
        n_ignitions=n_ignitions,
        dist_band_km=dist_band_km,
        sector_deg=sector_deg,
        require_treatment_intersect=require_treatment_intersect,
        seed=ignition_seed,
    )
    ign_shps = ign["ignition_shapefiles"]

    # ── One FSPro input file per ignition (shared seed, SavePerimeters on) ────
    sim_params = {
        "NumFires": num_fires, "Duration": duration, "Resolution": resolution,
        "SavePerimeters": 1,
    }
    sim_params.update(fspro_kwargs)
    sim_params["SavePerimeters"] = 1  # always on — perimeter analysis needs it

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
            seed          = seed,
            **sim_params,
        )
        input_paths.append(inp)
        run_rows.append({
            "Scenario": scenario, "LCP": str(baseline_lcp_path),
            "FSPro_input": str(inp), "output_basename": "baseline",
        })
        run_rows.append({
            "Scenario": scenario, "LCP": str(treated_lcp_path),
            "FSPro_input": str(inp), "output_basename": "treated",
        })

    runs_df = pd.DataFrame(run_rows)
    runs_csv = out_dir / "runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"[prepare_counterfactual_ignition_set] {len(ign_shps)} ignition(s) → "
          f"{len(runs_df)} FSPro runs (NumFires={num_fires}, Duration={duration})")

    manifest = {
        "baseline_lcp_path":    baseline_lcp_path,
        "treated_lcp_path":     treated_lcp_path,
        "pyrome_id":            pyrome_id,
        "wind_from_deg":        float(wind_from_deg),
        "downwind_az":          ign["downwind_az"],
        "seed":                 seed,
        "ignition_shapefiles":  [str(p) for p in ign_shps],
        "fspro_input_paths":    [str(p) for p in input_paths],
        "runs_csv":             runs_csv,
        "out_dir":              out_dir,
    }
    manifest_path = out_dir / "run_manifest.json"
    _write_manifest(manifest, manifest_path)

    manifest["manifest_path"] = manifest_path
    manifest["runs_df"]       = runs_df
    manifest["ignitions_gdf"] = ign["ignitions_gdf"]
    print(f"[prepare_counterfactual_ignition_set] Done. Manifest → {manifest_path}")
    return manifest


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
