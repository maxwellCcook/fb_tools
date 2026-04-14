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

    # Wind
    wind_meta = load_pyrome_wind_cells(
        pyrome_id, weather_dir / "pyrome_wind", return_meta=True
    )
    wind_cells = wind_meta["WindCellValues"]   # already np.ndarray
    calm_value = float(wind_meta["CalmValue"])

    # Historic ERC
    erc_meta = load_gridmet_pyrome_cache(
        pyrome_id, weather_dir / "pyrome_erc", return_meta=True
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

    # ERC classes — require CSV or precomputed array
    if erc_classes is not None:
        erc_classes = np.asarray(erc_classes, dtype=float)
    elif gridmet_csv is not None:
        df = load_gridmet_csv(gridmet_csv)
        # Filter to this pyrome only before passing to build_erc_classes
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
            "Provide either 'erc_classes' (precomputed ndarray, shape (5,10)) "
            "or 'gridmet_csv' (path to GEE-exported GridMET CSV). "
            "The pyrome cache JSON does not store the ERC class table."
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
    erc_classes: "np.ndarray | None" = None,
    gridmet_csv: "str | Path | None" = None,
    current_erc_start_doy: int = 91,
    current_erc_n_days: int = 79,
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

    pyromes_gdf : geopandas.GeoDataFrame
        NIFC pyrome polygons.  Must contain a ``Pyrome_ID`` column (or pass
        ``pyrome_col`` via ``fspro_kwargs`` — note: ``pyrome_col`` is consumed
        here and not forwarded to ``build_fspro_inputs``).
    lf_year : str or int
        LANDFIRE year for ``lfps_request`` (e.g. ``"2023"``).  Ignored when
        ``lcp_path`` is provided.
    lcp_path : str or Path, optional
        Pre-existing LCP GeoTIFF.  When provided, the LFPS download is skipped.
        The LCP CRS is used to reproject the container for ignition creation.
    erc_classes : np.ndarray, optional
        Pre-computed ERC class table, shape ``(5, 10)``.  Provide this *or*
        ``gridmet_csv`` — the pyrome cache JSON does not store ERC classes.
    gridmet_csv : str or Path, optional
        Path to GEE-exported GridMET CSV.  Used to build ERC classes on the
        fly when ``erc_classes`` is not provided.
    current_erc_start_doy : int
        1-based fire-season DOY (1 = April 1) at which ``CurrentERCValues``
        begins.  Default 91 (≈ July 1).
    current_erc_n_days : int
        Length of the current-season ERC sequence.  Default 79.
    **fspro_kwargs
        Passed to ``build_fspro_inputs`` (e.g. ``NumFires=2000``,
        ``Duration=14``).  ``pyrome_col`` is intercepted here if present.

    Returns
    -------
    dict
        Manifest with keys:

        ``"lcp_path"`` : Path
            Landscape GeoTIFF used for this run.
        ``"ignition_path"`` : Path
            Dissolved container ignition shapefile (.shp).
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
    from fb_tools.fuelscape.lfps import lfps_request
    from fb_tools.fuelscape.lcp import create_container_ignition
    from fb_tools.models.fspro import build_fspro_inputs
    from fb_tools.utils.geo import lookup_pyrome

    out_dir = Path(out_dir).resolve()
    lcp_dir     = out_dir / "lcp"
    ign_dir     = out_dir / "ignitions"
    inputs_dir  = out_dir / "fspro_inputs"
    outputs_dir = out_dir / "outputs"
    for d in (lcp_dir, ign_dir, inputs_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. LCP ────────────────────────────────────────────────────────────────
    if lcp_path is None:
        print("[prepare_container_fspro] Downloading LANDFIRE landscape …")
        lcp_path = lfps_request(
            container_gdf, lcp_dir, str(lf_year), clip=True
        )
    else:
        lcp_path = Path(lcp_path)
        if not lcp_path.exists():
            raise FileNotFoundError(f"lcp_path not found: {lcp_path}")

    # Report domain pixel count
    with rasterio.open(lcp_path) as src:
        lcp_crs     = src.crs
        pixel_count = src.width * src.height
    print(f"[prepare_container_fspro] LCP: {lcp_path.name} "
          f"({pixel_count:,} pixels, CRS: {lcp_crs.to_epsg() or lcp_crs.to_string()})")

    # ── 2. Dominant pyrome ────────────────────────────────────────────────────
    pyrome_col = fspro_kwargs.pop("pyrome_col", "Pyrome_ID")
    container_proj = container_gdf.to_crs(lcp_crs)
    # unary_union is deprecated in newer geopandas; prefer union_all if available
    try:
        container_union = container_proj.geometry.union_all()
    except AttributeError:
        container_union = container_proj.geometry.unary_union

    pyrome_id = str(lookup_pyrome(container_union, pyromes_gdf, pyrome_col=pyrome_col))
    print(f"[prepare_container_fspro] Dominant pyrome: {pyrome_id}")

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

    # ── 4. Ignition file (container boundary → polygon shapefile) ─────────────
    ign_path = create_container_ignition(
        container_proj, ign_dir / "container_ignition.shp"
    )
    print(f"[prepare_container_fspro] Ignition: {ign_path.name}")

    # ── 5. FSPro input file ───────────────────────────────────────────────────
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
        **fspro_kwargs,
    )
    print(f"[prepare_container_fspro] FSPro input: {fspro_input_path.name}")

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
    lf_year: "str | int",
    baseline_lcp_path: "str | Path | None" = None,
    treated_lcp_path: "str | Path | None" = None,
    treatment_gdf=None,
    canopy_df=None,
    surface_df=None,
    treatment_scenario: "dict | None" = None,
    erc_classes: "np.ndarray | None" = None,
    gridmet_csv: "str | Path | None" = None,
    seed: int = 617327,
    **fspro_kwargs,
) -> dict:
    """Prepare paired baseline and treated FSPro runs for counterfactual analysis.

    Builds a single FSPro input file with a fixed ``SPOTTING_SEED`` shared by
    both the baseline and treated landscape runs.  Because TestFSPro.exe takes
    the LCP as a runtime positional argument, one input file can serve both
    runs — ensuring identical weather draws and a clean delta comparison via
    ``delta_burn_probability()``.

    Parameters
    ----------
    container_gdf : geopandas.GeoDataFrame
        Spatial container defining the simulation domain.  Any CRS.
    out_dir : str or Path
        Root output directory.  Structure::

            out_dir/
              baseline/lcp/        ← baseline landscape
              baseline/outputs/    ← empty; populated on Windows
              treated/lcp/         ← treated landscape
              treated/outputs/     ← empty; populated on Windows
              ignitions/           ← shared container ignition shapefile
              fspro_inputs/        ← single shared FSPro input file
              run_manifest.json

    weather_dir : str or Path
        Root weather cache directory (same layout as ``prepare_container_fspro``).
    pyromes_gdf : geopandas.GeoDataFrame
        NIFC pyrome polygons with a ``Pyrome_ID`` column.
    lf_year : str or int
        LANDFIRE year for LFPS download if ``baseline_lcp_path`` is None.
    baseline_lcp_path : str or Path, optional
        Pre-existing baseline LCP GeoTIFF.  If None, downloaded via LFPS.
    treated_lcp_path : str or Path, optional
        Pre-existing treated LCP GeoTIFF.  If None and all treatment
        parameters are provided, the treated LCP is derived from the baseline
        by applying ``apply_treatment()``.
    treatment_gdf : geopandas.GeoDataFrame, optional
        Treatment polygons.  Required when ``treated_lcp_path`` is None.
    canopy_df, surface_df : pd.DataFrame, optional
        Canopy and surface effects tables for ``apply_treatment()``.  Required
        when ``treated_lcp_path`` is None.
    treatment_scenario : dict, optional
        Scenario descriptor passed to ``apply_treatment()`` as the
        ``scenario`` argument.  Example: ``{"canopy": "PCT25", "surface": "PCT25"}``.
        Required when ``treated_lcp_path`` is None.
    erc_classes : np.ndarray, optional
        Pre-computed ERC class table, shape ``(5, 10)``.
    gridmet_csv : str or Path, optional
        GEE-exported GridMET CSV, used when ``erc_classes`` is not provided.
    seed : int
        ``SPOTTING_SEED`` shared by both runs.  **Do not vary between
        baseline and treated.**  Default 617327.
    **fspro_kwargs
        Passed to ``build_treatment_pair`` (e.g. ``NumFires=2000``).

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
    ValueError
        If ``treated_lcp_path`` is None and any required treatment parameters
        are missing (``treatment_gdf``, ``canopy_df``, ``surface_df``,
        ``treatment_scenario``).
    FileNotFoundError
        If a provided LCP path does not exist.

    Examples
    --------
    On Mac (data prep):

    >>> manifest = prepare_counterfactual_fspro(
    ...     container_gdf=huc12_gdf,
    ...     out_dir="runs/huc12_042",
    ...     weather_dir="data/weather",
    ...     pyromes_gdf=pyromes,
    ...     lf_year="2023",
    ...     baseline_lcp_path="data/lcp/baseline.tif",
    ...     treatment_gdf=treatment_polygons,
    ...     canopy_df=canopy_effects,
    ...     surface_df=surface_effects,
    ...     treatment_scenario={"canopy": "PCT25", "surface": "PCT25"},
    ...     gridmet_csv="data/weather/gridmet_clima_co_pyromes.csv",
    ...     NumFires=1000,
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
    import rasterio
    from fb_tools.fuelscape.lfps import lfps_request
    from fb_tools.fuelscape.lcp import create_container_ignition
    from fb_tools.models.fspro import build_treatment_pair
    from fb_tools.utils.geo import lookup_pyrome

    out_dir = Path(out_dir).resolve()
    bl_dir  = out_dir / "baseline"
    tr_dir  = out_dir / "treated"
    ign_dir     = out_dir / "ignitions"
    inputs_dir  = out_dir / "fspro_inputs"
    bl_lcp_dir  = bl_dir / "lcp"
    tr_lcp_dir  = tr_dir / "lcp"
    bl_out_dir  = bl_dir / "outputs"
    tr_out_dir  = tr_dir / "outputs"
    for d in (bl_lcp_dir, tr_lcp_dir, ign_dir, inputs_dir, bl_out_dir, tr_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Baseline LCP ───────────────────────────────────────────────────────
    if baseline_lcp_path is None:
        print("[prepare_counterfactual_fspro] Downloading baseline LANDFIRE landscape …")
        baseline_lcp_path = lfps_request(
            container_gdf, bl_lcp_dir, str(lf_year), clip=True
        )
    else:
        baseline_lcp_path = Path(baseline_lcp_path)
        if not baseline_lcp_path.exists():
            raise FileNotFoundError(f"baseline_lcp_path not found: {baseline_lcp_path}")

    with rasterio.open(baseline_lcp_path) as src:
        lcp_crs     = src.crs
        pixel_count = src.width * src.height
    print(f"[prepare_counterfactual_fspro] Baseline LCP: {baseline_lcp_path.name} "
          f"({pixel_count:,} pixels)")

    # ── 2. Dominant pyrome ────────────────────────────────────────────────────
    pyrome_col = fspro_kwargs.pop("pyrome_col", "Pyrome_ID")
    container_proj = container_gdf.to_crs(lcp_crs)
    try:
        container_union = container_proj.geometry.union_all()
    except AttributeError:
        container_union = container_proj.geometry.unary_union

    pyrome_id = str(lookup_pyrome(container_union, pyromes_gdf, pyrome_col=pyrome_col))
    print(f"[prepare_counterfactual_fspro] Dominant pyrome: {pyrome_id}")

    # ── 3. Treated LCP ────────────────────────────────────────────────────────
    if treated_lcp_path is None:
        missing = [
            n for n, v in [
                ("treatment_gdf",      treatment_gdf),
                ("canopy_df",          canopy_df),
                ("surface_df",         surface_df),
                ("treatment_scenario", treatment_scenario),
            ]
            if v is None
        ]
        if missing:
            raise ValueError(
                f"treated_lcp_path is None — must also provide: {missing}"
            )
        print("[prepare_counterfactual_fspro] Applying treatment to baseline LCP …")

        import rioxarray as rxr
        from fb_tools.fuelscape.adjust import apply_treatment
        from fb_tools.utils.geo import rasterize

        lcp_da = rxr.open_rasterio(baseline_lcp_path, masked=True)
        ref_da = lcp_da.isel(band=0)
        treatment_reproj = treatment_gdf.to_crs(lcp_crs)
        mask = rasterize(treatment_reproj, ref_da) > 0

        treated_da = apply_treatment(
            lcp_da, canopy_df, surface_df, treatment_scenario, mask=mask
        )
        treated_lcp_out = tr_lcp_dir / "treated.tif"
        treated_da.rio.to_raster(treated_lcp_out, compress="deflate")
        treated_lcp_path = treated_lcp_out
        print(f"[prepare_counterfactual_fspro] Treated LCP: {treated_lcp_path.name}")
    else:
        treated_lcp_path = Path(treated_lcp_path)
        if not treated_lcp_path.exists():
            raise FileNotFoundError(f"treated_lcp_path not found: {treated_lcp_path}")

    # ── 4. Weather (once, shared) ─────────────────────────────────────────────
    print("[prepare_counterfactual_fspro] Loading pyrome weather …")
    wx = _load_weather_for_pyrome(
        pyrome_id,
        weather_dir,
        current_erc_start_doy=fspro_kwargs.pop("current_erc_start_doy", 91),
        current_erc_n_days=fspro_kwargs.pop("current_erc_n_days", 79),
        erc_classes=erc_classes,
        gridmet_csv=gridmet_csv,
    )

    # ── 5. Ignition file (shared, container boundary) ─────────────────────────
    ign_path = create_container_ignition(
        container_proj, ign_dir / "container_ignition.shp"
    )
    print(f"[prepare_counterfactual_fspro] Ignition: {ign_path.name}")

    # ── 6. Shared FSPro input file with fixed SPOTTING_SEED ───────────────────
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
        **fspro_kwargs,
    )
    print(f"[prepare_counterfactual_fspro] FSPro input (shared): {fspro_input_path.name}")

    # ── 7. Manifest ───────────────────────────────────────────────────────────
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
        "lf_year":              str(lf_year),
        "fspro_kwargs":         fspro_kwargs,
    }
    manifest_path = out_dir / "run_manifest.json"
    _write_manifest(manifest, manifest_path)
    manifest["manifest_path"] = manifest_path

    print(f"[prepare_counterfactual_fspro] Done. Manifest → {manifest_path}")
    return manifest
