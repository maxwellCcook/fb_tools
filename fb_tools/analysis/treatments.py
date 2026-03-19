"""
Treatment-level fire behavior change analysis.

Two public entry points
-----------------------
``summarize_treatments``
    Case 2: FlamMap outputs and optional SDI rasters already exist.
    Computes zonal statistics per treatment polygon and returns delta
    summaries for flame length bins, crown state, and SDI.

``run_treatment_pipeline``
    Case 1: No FlamMap outputs exist.  Generates treated LCPs via
    apply_treatment, runs FlamMap for baseline and each treatment type,
    optionally computes SDI, then delegates to summarize_treatments.

Both functions return a dict of three DataFrames keyed ``'fl'``, ``'cs'``,
and ``'sdi'``, following the output format of the TEALOM reference workflow.
"""

import gc
from pathlib import Path

import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd

from ..models.scenarios import build_scenarios, run_batch, stacked_output_path
from ..fuelscape.adjust import apply_treatment
from ..suppression.sdi import calculate_sdi
from ..utils.geo import geom_to_raster_crs
from .zonal import zonal_categorical, zonal_continuous, _make_zone_arr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Flame-length bin edges in METRES (2, 4, 8, 12 ft converted).
# np.digitize(fl_metres, _FL_BIN_EDGES_M) returns indices 0–4.
_FL_BIN_EDGES_M = [0.6096, 1.2192, 2.4384, 3.6576]
_FL_BIN_LABELS  = ["0-2ft", "2-4ft", "4-8ft", "8-12ft", ">12ft"]

# FlamMap CROWNSTATE integer codes → display labels.
_CS_CLASSES = {0: "NonBurnable", 1: "Surface", 2: "Passive Crown", 3: "Active Crown"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_fm_tif(directory, percentile, band_name):
    """
    Locate a FlamMap output GeoTIFF for a given percentile and band name.

    Tries the stacked-output naming convention first
    (``{scenario}_{LCP_STEM}.tif``), then falls back to a glob for any
    ``.tif`` file whose stem contains both *percentile* and *band_name*.

    Parameters
    ----------
    directory : Path
        Root output directory from run_batch for one LCP/treatment type.
        Should contain sub-directories named after each scenario/percentile.
    percentile : str
        Scenario name (e.g., ``'Pct97'``).
    band_name : str
        FlamMap output band name to search for (e.g., ``'FLAMELENGTH'``,
        ``'CROWNSTATE'``).

    Returns
    -------
    Path
        Path to the located raster file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    directory = Path(directory)

    # Try stacked multi-band TIF inside percentile sub-directory.
    pct_dir = directory / percentile
    if pct_dir.is_dir():
        candidates = sorted(pct_dir.glob("*.tif"))
        if candidates:
            # Stacked file is typically the only (or largest) .tif here.
            return candidates[0]

    # Fallback: search recursively for a file matching both labels.
    bn_lower = band_name.lower()
    pct_lower = percentile.lower()
    for p in sorted(directory.rglob("*.tif")):
        stem = p.stem.lower()
        if bn_lower in stem and pct_lower in stem:
            return p

    raise FileNotFoundError(
        f"Could not find FlamMap raster for percentile={percentile!r}, "
        f"band={band_name!r} in directory: {directory}"
    )


def _open_band(raster, band_name):
    """
    Open a raster file (or accept a DataArray) and return the named band as
    a squeezed DataArray.

    For stacked multi-band TIFs, selects the band whose ``long_name``
    attribute contains *band_name* (case-insensitive).  For single-band
    files, returns band 1 regardless of name.

    Parameters
    ----------
    raster : str, Path, or xarray.DataArray
        Source raster.
    band_name : str
        Band to extract from a stacked file (e.g., ``'FLAMELENGTH'``).

    Returns
    -------
    xarray.DataArray
        Squeezed single-band DataArray.
    """
    if isinstance(raster, xr.DataArray):
        da = raster
    else:
        da = rxr.open_rasterio(raster, masked=True)

    # Multi-band: select by long_name attribute.
    if "band" in da.dims and da.sizes["band"] > 1:
        bn_lower = band_name.lower()
        for i, name in enumerate(da.attrs.get("long_name", []), start=1):
            if bn_lower in str(name).lower():
                return da.sel(band=i).squeeze(drop=True)
        raise ValueError(
            f"Band {band_name!r} not found in stacked raster. "
            f"Available long_names: {da.attrs.get('long_name')}"
        )

    return da.squeeze(drop=True)


def _bin_fl(raster, band_name="FLAMELENGTH"):
    """
    Open a flame-length raster, bin pixel values into 5 FL classes, and
    return the integer bin array with a reference DataArray.

    Parameters
    ----------
    raster : str, Path, or xarray.DataArray
        FlamMap FLAMELENGTH output (values in metres).
    band_name : str
        Band name to extract from stacked files.

    Returns
    -------
    binned_arr : numpy.ndarray  (int, shape = H × W)
        Bin indices 0–4 matching ``_FL_BIN_LABELS``.
        Pixels with FL ≤ 0 (non-burnable / nodata) are set to -9999.
    reference_da : xarray.DataArray
        Single-band DataArray for use as the rasterize() reference grid.
    """
    da = _open_band(raster, band_name)
    arr = da.values.squeeze().astype(np.float32)
    binned = np.digitize(arr, bins=_FL_BIN_EDGES_M).astype(np.int8)
    binned = np.where(arr <= 0, -1, binned) # set NoData explicitly
    return binned, da


def _load_cs_raster(raster, band_name="CROWNSTATE"):
    """
    Open a crown-state raster and return the integer code array with a
    reference DataArray.

    Parameters
    ----------
    raster : str, Path, or xarray.DataArray
        FlamMap CROWNSTATE output (integer codes 0–3).
    band_name : str
        Band name to extract from stacked files.

    Returns
    -------
    cs_arr : numpy.ndarray  (int16, shape = H × W)
        Crown state codes. Pixels outside valid range are set to -9999.
    reference_da : xarray.DataArray
        Single-band DataArray for use as the rasterize() reference grid.
    """
    da = _open_band(raster, band_name)
    arr = da.values.squeeze()
    arr = np.where(np.isnan(arr) | (arr < 0) | (arr > 3), -1, arr)
    arr = arr.astype(np.int8)
    return arr, da


def _zonal_fl(zones_gdf, id_col, raster, band_name="FLAMELENGTH", zone_arr=None):
    """
    Zonal categorical statistics on a flame-length raster.

    Parameters
    ----------
    zones_gdf : GeoDataFrame or None
        Treatment polygons. Ignored when *zone_arr* is provided.
    id_col : str
        Zone identifier column.
    raster : str, Path, or xarray.DataArray
        FlamMap FLAMELENGTH output.
    band_name : str
        Band name for stacked files.
    zone_arr : numpy.ndarray, optional
        Pre-burned zone grid from :func:`~.zonal._make_zone_arr`. When
        provided, rasterization is skipped.
    """
    binned, ref_da = _bin_fl(raster, band_name)
    if zone_arr is None:
        zones = geom_to_raster_crs(zones_gdf[[id_col, "geometry"]], ref_da)
        df = zonal_categorical(zones, binned, ref_da, id_col)
    else:
        df = zonal_categorical(None, binned, ref_da, id_col, zone_arr=zone_arr)
    del binned, ref_da
    df["FL_class"] = df["class_val"].map(dict(enumerate(_FL_BIN_LABELS)))
    return df.drop(columns="class_val")


def _zonal_cs(zones_gdf, id_col, raster, band_name="CROWNSTATE", zone_arr=None):
    """
    Zonal categorical statistics on a crown-state raster.

    Parameters
    ----------
    zones_gdf : GeoDataFrame or None
        Treatment polygons. Ignored when *zone_arr* is provided.
    id_col : str
        Zone identifier column.
    raster : str, Path, or xarray.DataArray
        FlamMap CROWNSTATE output.
    band_name : str
        Band name for stacked files.
    zone_arr : numpy.ndarray, optional
        Pre-burned zone grid from :func:`~.zonal._make_zone_arr`. When
        provided, rasterization is skipped.
    """
    cs_arr, ref_da = _load_cs_raster(raster, band_name)
    if zone_arr is None:
        zones = geom_to_raster_crs(zones_gdf[[id_col, "geometry"]], ref_da)
        df = zonal_categorical(zones, cs_arr, ref_da, id_col)
    else:
        df = zonal_categorical(None, cs_arr, ref_da, id_col, zone_arr=zone_arr)
    del cs_arr, ref_da
    df["CS_class"] = df["class_val"].map(_CS_CLASSES)
    return df.drop(columns="class_val")


def _zonal_sdi(zones_gdf, id_col, sdi_raster):
    """Run zonal_continuous on an SDI raster (int16, SDI × 100)."""
    if isinstance(sdi_raster, xr.DataArray):
        da = sdi_raster.squeeze(drop=True)
    else:
        da = rxr.open_rasterio(sdi_raster, masked=True).squeeze(drop=True)

    arr = da.values.astype(np.float32)
    zones_proj = geom_to_raster_crs(zones_gdf[[id_col, "geometry"]], da)
    df = zonal_continuous(zones_proj, arr, da, id_col,
                          stat="mean", scale_factor=100.0)
    del arr, da
    gc.collect()
    return df.rename(columns={"mean": "SDI_mean"})


def _safe_name(s):
    """Convert a treatment type label to a filesystem-safe string."""
    return s.replace(" ", "_").replace("/", "-").lower()


def _acres(gdf, crs="EPSG:5070"):
    """Return per-row area in acres, reprojected to an equal-area CRS."""
    projected = gdf.geometry.to_crs(crs)
    return projected.area * 0.000247105


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_treatments(
    treatments_gdf,
    id_col,
    type_col,
    baseline_dir,
    treated_dirs,
    percentiles=("Pct25", "Pct50", "Pct90", "Pct97"),
    baseline_sdi=None,
    treated_sdi=None,
    out_dir=None,
):
    """
    Compute treatment-level fire behavior change summaries from existing rasters.

    Parameters
    ----------
    treatments_gdf : GeoDataFrame
        Treatment polygons. Must contain *id_col*, *type_col*, and geometry.
        The *id_col* values must be numeric so they can be burned as raster
        pixel values.
    id_col : str
        Column with a unique numeric treatment identifier (e.g., ``'TRT_ID'``).
    type_col : str
        Column with treatment type / scenario label (e.g., ``'TRT_TYPE'``).
        Unique values are matched against keys in *treated_dirs*.
    baseline_dir : str or Path
        Directory containing baseline FlamMap output GeoTIFFs.  Must contain
        one sub-directory per percentile (e.g., ``Pct25/``) produced by
        :func:`~fb_tools.models.scenarios.run_batch` with ``stack_out=True``.
    treated_dirs : dict
        Mapping ``{treatment_type: directory}`` for treated FlamMap outputs.
        Keys must match the unique values in *treatments_gdf[type_col]*.
        Example::

            {'Mech Thin': Path('/outputs/treated_mech_thin')}
    percentiles : sequence of str
        Percentile scenario names to process (default ``Pct25, Pct50, Pct90,
        Pct97``).  Must match the scenario names used in ``run_batch``.
    baseline_sdi : str, Path, or DataArray, optional
        Baseline SDI raster (output of :func:`~fb_tools.suppression.sdi.calculate_sdi`).
        Stored as int16 at SDI × 100 scale.  If ``None``, SDI columns are
        filled with ``NaN``.
    treated_sdi : dict, optional
        Mapping ``{treatment_type: sdi_path_or_da}`` for treated SDI rasters.
        Required if *baseline_sdi* is provided.
    out_dir : str or Path, optional
        If provided, write ``fl_change.csv``, ``cs_change.csv``, and
        ``sdi_summary.csv`` to this directory.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Keys are ``'fl'``, ``'cs'``, ``'sdi'``:

        **result['fl']** — one row per (TRT_ID × percentile × FL_class):
            id_col, type_col, ACRES_GIS, percentile, scenario,
            FL_class, FL_bl_count, FL_bl_pct, FL_tr_count, FL_tr_pct, delta_pct

        **result['cs']** — one row per (TRT_ID × percentile × CS_class):
            id_col, type_col, ACRES_GIS, percentile, scenario,
            CS_class, CS_bl_count, CS_bl_pct, CS_tr_count, CS_tr_pct, delta_pct

        **result['sdi']** — one row per treatment (SDI at Pct90 only):
            id_col, type_col, ACRES_GIS, scenario,
            SDI_bl_mean, SDI_tr_mean, SDI_delta_mean

        SDI columns contain ``NaN`` when *baseline_sdi* is not provided.

    Notes
    -----
    - Baseline metrics are computed once across **all** treatment polygons per
      percentile, then joined to each treatment-type subset.
    - Treated metrics are computed per treatment-type subset (filtered by
      *type_col* == treatment_type) using the corresponding *treated_dirs* entry.
    - ``rasterize()`` is called once for all treatment polygons combined and
      once per treatment-type subset. The resulting zone grids are reused
      across all percentiles and metrics, reducing rasterization from
      ``N_percentiles × 4 × N_types`` calls to ``N_types + 1`` calls.
    - The per-zone numpy loops iterate only over pixels inside treatment
      polygons (valid-pixel masking), skipping the large nodata regions
      common in full-landscape FlamMap outputs.
    - SDI is not computed per-percentile; the SDI columns are identical across
      all percentile rows for each treatment.
    """
    baseline_dir = Path(baseline_dir)
    treatments_gdf = treatments_gdf.copy()

    # Compute treatment areas once.
    treatments_gdf["ACRES_GIS"] = _acres(treatments_gdf)

    acres_map = treatments_gdf.set_index(id_col)["ACRES_GIS"].to_dict()
    type_map  = treatments_gdf.set_index(id_col)[type_col].to_dict()

    fl_rows = []
    cs_rows = []

    # ------------------------------------------------------------------
    # Pre-rasterize zone grids — done ONCE per polygon scope, then reused
    # across all percentiles and metrics to avoid repeated make_geocube calls.
    # Zone arrays are built from the first available raster's reference grid
    # (all FlamMap outputs for a given run share the same spatial grid).
    # ------------------------------------------------------------------
    all_zone_arr  = None   # zone grid for all treatment polygons (baseline)
    trt_zone_arrs = {}     # {trt_type: zone_arr} for each treatment subset

    for pct in percentiles:
        print(f"  Processing percentile: {pct}")

        # --- Baseline FL and CS zonal stats (all polygons) ---
        bl_fl_raster = _find_fm_tif(baseline_dir, pct, "FLAMELENGTH")
        bl_cs_raster = _find_fm_tif(baseline_dir, pct, "CROWNSTATE")

        # Build the all-polygon zone array exactly once (first percentile).
        if all_zone_arr is None:
            print("    Building zone grid for all treatment polygons...")
            _ref = _open_band(bl_fl_raster, "FLAMELENGTH")
            _zones_proj = geom_to_raster_crs(
                treatments_gdf[[id_col, "geometry"]], _ref
            )
            all_zone_arr = _make_zone_arr(_zones_proj, _ref, id_col)
            del _ref, _zones_proj
            gc.collect()

        bl_fl = _zonal_fl(None, id_col, bl_fl_raster, zone_arr=all_zone_arr)
        bl_cs = _zonal_cs(None, id_col, bl_cs_raster, zone_arr=all_zone_arr)

        # --- Per treatment type: treated FL and CS ---
        for trt_type, trt_dir in treated_dirs.items():
            trt_dir = Path(trt_dir)
            scenario_label = _safe_name(trt_type)
            trt_subset = treatments_gdf[treatments_gdf[type_col] == trt_type]

            if trt_subset.empty:
                continue

            tr_fl_raster = _find_fm_tif(trt_dir, pct, "FLAMELENGTH")
            tr_cs_raster = _find_fm_tif(trt_dir, pct, "CROWNSTATE")

            # Build the per-type zone array once (first percentile for this type).
            if trt_type not in trt_zone_arrs:
                print(f"    Building zone grid for treatment type: {trt_type!r}...")
                _ref = _open_band(tr_fl_raster, "FLAMELENGTH")
                _zones_proj = geom_to_raster_crs(
                    trt_subset[[id_col, "geometry"]], _ref
                )
                trt_zone_arrs[trt_type] = _make_zone_arr(_zones_proj, _ref, id_col)
                del _ref, _zones_proj
                gc.collect()

            tr_fl = _zonal_fl(None, id_col, tr_fl_raster,
                              zone_arr=trt_zone_arrs[trt_type])
            tr_cs = _zonal_cs(None, id_col, tr_cs_raster,
                              zone_arr=trt_zone_arrs[trt_type])

            # --- Merge and compute deltas: FL ---
            merged_fl = bl_fl[bl_fl[id_col].isin(trt_subset[id_col])].merge(
                tr_fl, on=[id_col, "FL_class"], how="outer", suffixes=("_bl", "_tr")
            ).fillna(0)
            merged_fl["delta_pct"] = merged_fl["pct_cover_tr"] - merged_fl["pct_cover_bl"]
            merged_fl = merged_fl.rename(columns={
                "pixel_count_bl": "FL_bl_count",
                "pct_cover_bl":   "FL_bl_pct",
                "pixel_count_tr": "FL_tr_count",
                "pct_cover_tr":   "FL_tr_pct",
            })
            merged_fl["percentile"] = pct
            merged_fl["scenario"]   = scenario_label
            merged_fl[type_col]     = trt_type
            merged_fl["ACRES_GIS"]  = merged_fl[id_col].map(acres_map)
            fl_rows.append(merged_fl)

            # --- Merge and compute deltas: CS ---
            merged_cs = bl_cs[bl_cs[id_col].isin(trt_subset[id_col])].merge(
                tr_cs, on=[id_col, "CS_class"], how="outer", suffixes=("_bl", "_tr")
            ).fillna(0)
            merged_cs["delta_pct"] = merged_cs["pct_cover_tr"] - merged_cs["pct_cover_bl"]
            merged_cs = merged_cs.rename(columns={
                "pixel_count_bl": "CS_bl_count",
                "pct_cover_bl":   "CS_bl_pct",
                "pixel_count_tr": "CS_tr_count",
                "pct_cover_tr":   "CS_tr_pct",
            })
            merged_cs["percentile"] = pct
            merged_cs["scenario"]   = scenario_label
            merged_cs[type_col]     = trt_type
            merged_cs["ACRES_GIS"]  = merged_cs[id_col].map(acres_map)
            cs_rows.append(merged_cs)

    # Assemble FL and CS DataFrames.
    fl_cols = [id_col, type_col, "ACRES_GIS", "percentile", "scenario",
               "FL_class", "FL_bl_count", "FL_bl_pct", "FL_tr_count", "FL_tr_pct", "delta_pct"]
    cs_cols = [id_col, type_col, "ACRES_GIS", "percentile", "scenario",
               "CS_class", "CS_bl_count", "CS_bl_pct", "CS_tr_count", "CS_tr_pct", "delta_pct"]

    fl_df = pd.concat(fl_rows, ignore_index=True)[fl_cols] if fl_rows else pd.DataFrame(columns=fl_cols)
    cs_df = pd.concat(cs_rows, ignore_index=True)[cs_cols] if cs_rows else pd.DataFrame(columns=cs_cols)

    # --- SDI summary ---
    sdi_cols = [id_col, type_col, "ACRES_GIS", "scenario",
                "SDI_bl_mean", "SDI_tr_mean", "SDI_delta_mean"]

    if baseline_sdi is not None and treated_sdi is not None:
        print("  Computing SDI zonal statistics...")
        bl_sdi_df = _zonal_sdi(treatments_gdf, id_col, baseline_sdi)
        bl_sdi_df = bl_sdi_df.rename(columns={"SDI_mean": "SDI_bl_mean"})

        sdi_rows = []
        for trt_type, tr_sdi in treated_sdi.items():
            scenario_label = _safe_name(trt_type)
            trt_subset = treatments_gdf[treatments_gdf[type_col] == trt_type]
            if trt_subset.empty:
                continue
            tr_sdi_df = _zonal_sdi(trt_subset, id_col, tr_sdi)
            tr_sdi_df = tr_sdi_df.rename(columns={"SDI_mean": "SDI_tr_mean"})

            merged_sdi = bl_sdi_df[bl_sdi_df[id_col].isin(trt_subset[id_col])].merge(
                tr_sdi_df, on=id_col, how="left"
            )
            merged_sdi["SDI_delta_mean"] = (
                merged_sdi["SDI_bl_mean"] - merged_sdi["SDI_tr_mean"]
            )
            merged_sdi["scenario"]  = scenario_label
            merged_sdi[type_col]    = trt_type
            merged_sdi["ACRES_GIS"] = merged_sdi[id_col].map(acres_map)
            sdi_rows.append(merged_sdi)

        sdi_df = pd.concat(sdi_rows, ignore_index=True)[sdi_cols] if sdi_rows else pd.DataFrame(columns=sdi_cols)
    else:
        # Return empty SDI frame with NaN columns.
        ids = treatments_gdf[[id_col, type_col]].copy()
        ids["ACRES_GIS"]    = ids[id_col].map(acres_map)
        ids["scenario"]     = ids[type_col].map(_safe_name)
        ids["SDI_bl_mean"]  = np.nan
        ids["SDI_tr_mean"]  = np.nan
        ids["SDI_delta_mean"] = np.nan
        sdi_df = ids[sdi_cols]

    # Optionally save to CSV.
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fl_df.to_csv(out_dir / "fl_change.csv", index=False)
        cs_df.to_csv(out_dir / "cs_change.csv", index=False)
        sdi_df.to_csv(out_dir / "sdi_summary.csv", index=False)
        print(f"Results saved to {out_dir}")

    return {"fl": fl_df, "cs": cs_df, "sdi": sdi_df}


def run_treatment_pipeline(
    treatments_gdf,
    id_col,
    type_col,
    baseline_lcp,
    treatment_params,
    conditions,
    fm_exe,
    output_root,
    rtc_path,
    roads_gdf,
    trails_gdf,
    percentiles=("Pct25", "Pct50", "Pct90", "Pct97"),
    n_process=1,
    compute_sdi=True,
    out_dir=None,
):
    """
    Full pipeline: apply treatments, run FlamMap, compute SDI, then summarize.

    Orchestrates ``apply_treatment`` → ``run_batch`` → ``calculate_sdi`` →
    ``summarize_treatments`` for each unique treatment type.  Generates one
    landscape-wide treated LCP per treatment type (not per polygon).

    Parameters
    ----------
    treatments_gdf : GeoDataFrame
        Treatment polygons. Must contain *id_col*, *type_col*, and geometry.
        *id_col* values must be numeric.
    id_col : str
        Unique numeric treatment identifier column.
    type_col : str
        Treatment type column. Each unique value must have a matching entry in
        *treatment_params* and will become a directory under *output_root*.
    baseline_lcp : str or Path
        Path to the baseline multi-band LCP raster.
    treatment_params : dict
        Mapping ``{treatment_type: params_dict}`` where each *params_dict*
        contains:

        ``'canopy_df'`` : pandas.DataFrame
            Canopy adjustment table for :func:`~fb_tools.fuelscape.adjust.apply_treatment`.
        ``'surface_df'`` : pandas.DataFrame
            Surface fuel adjustment table.
        ``'scenario'`` : dict
            Scenario name dict for ``apply_treatment``, e.g.
            ``{'canopy': 'MechThin', 'surface': 'MechThin'}``.

        Example::

            {
                'Mech Thin': {
                    'canopy_df':  pd.DataFrame(...),
                    'surface_df': pd.DataFrame(...),
                    'scenario':   {'canopy': 'MechThin', 'surface': 'MechThin'},
                }
            }
    conditions : pandas.DataFrame
        Fire-weather conditions table consumed by
        :func:`~fb_tools.models.scenarios.build_scenarios`.
        Must include at minimum ``Scenario``, ``WIND_SPEED``, ``WIND_DIRECTION``,
        ``FM_1hr``, ``FM_10hr``, ``FM_100hr``, ``FM_herb``, ``FM_woody``.
    fm_exe : str or Path
        Path to the FlamMap executable (``TestFlamMap.exe``).
    output_root : str or Path
        Root directory for all generated LCPs and FlamMap output.
        Sub-directories created automatically:

        * ``output_root/baseline/`` — baseline FlamMap outputs
        * ``output_root/lcp_{safe_name}.tif`` — treated LCP per type
        * ``output_root/treated_{safe_name}/`` — treated FlamMap outputs
        * ``output_root/sdi_baseline.tif`` — baseline SDI (if compute_sdi)
        * ``output_root/sdi_treated_{safe_name}.tif`` — treated SDI per type
    rtc_path : str or Path
        Path to the RTC lookup table
        (``08_RTC_lookup_SDIwt_westernUS_2021_update.txt``).
    roads_gdf : GeoDataFrame
        Road features from :func:`~fb_tools.suppression.roads.fetch_osm_roads`.
    trails_gdf : GeoDataFrame
        Trail features from ``fetch_osm_roads``.
    percentiles : sequence of str
        Percentile scenario names.  Values must match the ``Scenario`` column
        in *conditions*.
    n_process : int
        Number of parallel FlamMap threads (passed to ``run_batch``).
    compute_sdi : bool
        Whether to compute SDI rasters.  SDI is computed at ``Pct90`` only.
        Set ``False`` to skip SDI (faster, no SDI columns in output).
    out_dir : str or Path, optional
        Passed to :func:`summarize_treatments` for optional CSV output.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Same structure as :func:`summarize_treatments`:
        ``{'fl': DataFrame, 'cs': DataFrame, 'sdi': DataFrame}``.
    """
    baseline_lcp = Path(baseline_lcp)
    output_root  = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Validate treatment_params keys cover all types in the GeoDataFrame.
    unique_types = treatments_gdf[type_col].unique()
    missing = [t for t in unique_types if t not in treatment_params]
    if missing:
        raise ValueError(
            f"treatment_params is missing entries for: {missing}. "
            f"Keys provided: {list(treatment_params)}"
        )

    # ------------------------------------------------------------------
    # Step 1: Baseline FlamMap
    # ------------------------------------------------------------------
    print("Running baseline FlamMap scenarios...")
    bl_scenarios = build_scenarios(conditions, [baseline_lcp])
    bl_dir = output_root / "baseline"
    run_batch(fm_exe, bl_scenarios, bl_dir, n_process=n_process, stack_out=True)

    # ------------------------------------------------------------------
    # Step 2: Baseline SDI (Pct90 only)
    # ------------------------------------------------------------------
    baseline_sdi_path = None
    if compute_sdi:
        print("Computing baseline SDI...")
        bl_fm_stack = stacked_output_path(bl_dir, baseline_lcp, "Pct90")
        baseline_sdi_path = output_root / "sdi_baseline.tif"
        calculate_sdi(
            lcp=baseline_lcp,
            roads_gdf=roads_gdf,
            trails_gdf=trails_gdf,
            rtc_path=rtc_path,
            flammap_stack=bl_fm_stack,
            out_path=baseline_sdi_path,
        )

    # ------------------------------------------------------------------
    # Step 3: Loop over treatment types
    # ------------------------------------------------------------------
    treated_dirs  = {}
    treated_sdi   = {}

    for trt_type in unique_types:
        safe = _safe_name(trt_type)
        params = treatment_params[trt_type]
        print(f"Applying treatment: {trt_type!r}...")

        # 3a: Generate treated LCP (landscape-wide).
        treated_lcp_path = output_root / f"lcp_{safe}.tif"
        treated_lcp = apply_treatment(
            canopy_df=params["canopy_df"],
            surface_df=params["surface_df"],
            scenario=params["scenario"],
            lcp=baseline_lcp,
            mask=None,
        )
        treated_lcp.rio.to_raster(str(treated_lcp_path), dtype="int16",
                                  compress="deflate")
        del treated_lcp
        gc.collect()

        # 3b: Run FlamMap for this treatment type.
        print(f"  Running FlamMap ({trt_type!r})...")
        tr_scenarios = build_scenarios(conditions, [treated_lcp_path])
        tr_dir = output_root / f"treated_{safe}"
        run_batch(fm_exe, tr_scenarios, tr_dir, n_process=n_process, stack_out=True)
        treated_dirs[trt_type] = tr_dir

        # 3c: Compute treated SDI.
        if compute_sdi:
            print(f"  Computing treated SDI ({trt_type!r})...")
            tr_fm_stack = stacked_output_path(tr_dir, treated_lcp_path, "Pct90")
            tr_sdi_path = output_root / f"sdi_treated_{safe}.tif"
            calculate_sdi(
                lcp=treated_lcp_path,
                roads_gdf=roads_gdf,
                trails_gdf=trails_gdf,
                rtc_path=rtc_path,
                flammap_stack=tr_fm_stack,
                out_path=tr_sdi_path,
            )
            treated_sdi[trt_type] = tr_sdi_path

    # ------------------------------------------------------------------
    # Step 4: Summarize
    # ------------------------------------------------------------------
    print("Building treatment summaries...")
    return summarize_treatments(
        treatments_gdf=treatments_gdf,
        id_col=id_col,
        type_col=type_col,
        baseline_dir=bl_dir,
        treated_dirs=treated_dirs,
        percentiles=percentiles,
        baseline_sdi=baseline_sdi_path if compute_sdi else None,
        treated_sdi=treated_sdi if compute_sdi else None,
        out_dir=out_dir,
    )
