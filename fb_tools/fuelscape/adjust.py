"""
Fuelscape (LCP) adjustment utilities.

Apply per-pixel fuel modifications to a multi-band landscape DataArray,
driven by EVT codes, treatment tables, or custom masks.
"""

import re
from pathlib import Path

import numpy as np
import rioxarray as rxr  # noqa: F401 — required for .rio accessor
import xarray as xr


def _normalize_band_name(name):
    """Normalize an LFPS product code to a short canonical band name.
    """
    name = name.split("_")[1]
    return name


def adjust_lcp(lcp, evt_code=7050, cbh_adjust=0.70, fbfm_to=185):
    """
    Adjust fuel attributes for pixels matching a given EVT code.

    Applies a canopy-base-height scalar and/or a surface fuel model
    reassignment wherever the EVT band equals *evt_code*.

    Parameters
    ----------
    lcp : xarray.Dataset
        Multi-band landscape dataset with variables named ``"EVT"``,
        ``"CBH"``, and ``"FBFM40"``.
    evt_code : int
        EVT value to target (default ``7050``, lodgepole pine).
    cbh_adjust : float, optional
        Multiplicative factor applied to CBH for matching pixels
        (e.g. ``0.70`` → reduce by 30 %).  Pass ``None`` to skip.
    fbfm_to : int, optional
        Surface fuel model code to assign to matching pixels
        (e.g. ``185``).  Pass ``None`` to skip.

    Returns
    -------
    xarray.Dataset
        A modified copy of *lcp*.

    Notes
    -----
    The input dataset is **not** modified in place; a deep copy is returned.
    """
    fs = lcp.copy()
    evt_mask = fs["EVT"] == evt_code

    if not evt_mask.any():
        print(f"No pixels found for EVT code {evt_code} — no adjustments applied.")
        return fs

    if cbh_adjust is not None:
        if "CBH" in fs:
            fs["CBH"] = xr.where(evt_mask, fs["CBH"] * cbh_adjust, fs["CBH"])
        else:
            print("CBH band not found in dataset — skipping CBH adjustment.")

    if fbfm_to is not None:
        if "FBFM40" in fs:
            fs["FBFM40"] = xr.where(evt_mask, fbfm_to, fs["FBFM40"])
        else:
            print("FBFM40 band not found in dataset — skipping fuel model adjustment.")

    return fs


# ---------------------------------------------------------------------------
# Treatment-table adjustments
# ---------------------------------------------------------------------------

def build_surface_lut(surface_df, scenario_col, fm_col="FBFM40"):
    """
    Build a numpy lookup table for fast surface fuel model remapping.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Surface effects table.  Must contain *fm_col* (original fuel models)
        and *scenario_col* (post-treatment fuel models).
    scenario_col : str
        Column name for the desired treatment scenario.
    fm_col : str
        Column containing the original FBFM40 codes (default ``"FBFM40"``).

    Returns
    -------
    np.ndarray
        1-D integer array of length ``max(FBFM40) + 1``.  Index with the
        original code to get the post-treatment code.
    """
    base = surface_df[fm_col].to_numpy()
    new  = surface_df[scenario_col].to_numpy()
    vmax = int(np.nanmax(base))
    lut  = np.arange(vmax + 1, dtype=np.int16)   # identity by default
    lut[base.astype(int)] = new.astype(np.int16)
    return lut


def apply_treatment(lcp, canopy_df, surface_df, scenario, band_map=None, mask=None, fill_val=-999):
    """
    Apply a fuel treatment scenario to an LCP DataArray.

    Canopy bands (CC, CH, CBH, CBD) are scaled by per-band adjustment
    factors.  Pixels where post-treatment canopy cover drops below 10 %
    have all canopy bands zeroed out.  The FBFM40 band is remapped using
    a pre-built lookup table.

    Parameters
    ----------
    lcp : xarray.DataArray
        Multi-band landscape raster opened with rioxarray.
    canopy_df : pd.DataFrame
        Canopy effects table.  Must have a ``Treatment`` column plus
        ``cc_AF``, ``ch_AF``, ``cbh_AF``, ``cbd_AF`` columns.
    surface_df : pd.DataFrame
        Surface effects table.  Must have a ``FBFM40`` column and one
        column per treatment type containing the post-treatment fuel model.
    scenario : dict
        Mapping of ``{'canopy': <treatment_name>, 'surface': <treatment_name>}``.
        Use the same names as appear in ``canopy_df["Treatment"]`` and the
        column names of ``surface_df``.
    band_map : dict, optional
        Maps band names to 1-based integer indices, e.g.
        ``{"FBFM40": 4, "CC": 5, "CH": 6, "CBH": 7, "CBD": 8}``.
        Auto-detected from the ``long_name`` raster attribute if ``None``.
    mask : xarray.DataArray, optional
        Boolean DataArray aligned to *lcp*.  Where ``True`` the treatment
        is applied; elsewhere the original values are kept.  Pass ``None``
        (default) to apply the treatment to every pixel (landscape-scale).
    fill_val : int, optional

    Returns
    -------
    xarray.DataArray
        A modified copy of *lcp*.  Canopy bands are ``int16``; FBFM40 is
        remapped in place.

    Notes
    -----
    To generate *mask* from treatment polygons use
    :func:`fb_tools.utils.rasterize` with ``fill_val=0``, then cast::

        mask = rasterize(treatments, lcp.isel(band=0), attr="treated") > 0
    """
    # --- accept file path or DataArray
    if isinstance(lcp, (str, Path)):
        lcp = rxr.open_rasterio(Path(lcp), masked=True)

    # --- Create a band map from the input LCP
    long_names = lcp.attrs.get("long_name", [])
    band_map = {
        _normalize_band_name(name): idx for idx, name in enumerate(long_names, start=1)
    }

    out = lcp.copy(deep=True)

    if mask is not None:
        mask = mask.reindex_like(lcp.isel(band=0), fill_value=0)

    canopy_nm  = scenario["canopy"]
    surface_nm = scenario["surface"]

    # --- 1. Canopy adjustments
    canopy_idx = canopy_df.set_index("Treatment")
    if canopy_nm not in canopy_idx.index:
        raise ValueError(f"Canopy scenario '{canopy_nm}' not found in canopy_df.")
    r = canopy_idx.loc[canopy_nm]

    band_af_pairs = [
        ("CC",  "cc_AF"),
        ("CH",  "ch_AF"),
        ("CBH", "cbh_AF"),
        ("CBD", "cbd_AF"),
    ]

    # post-treatment forest mask (CC >= 10 after treatment)
    cc_band   = lcp.sel(band=band_map["CC"]).astype(np.float32)
    cc_post   = np.floor(cc_band * float(r["cc_AF"]))
    is_forest = cc_post >= 10

    for band_name, af_col in band_af_pairs:
        if band_name not in band_map:
            continue
        idx = band_map[band_name]
        arr = np.floor(
            lcp.sel(band=idx).astype(np.float32) * float(r[af_col])
        )
        arr = arr.fillna(fill_val)
        arr = xr.where(is_forest, arr, 0)              # zero out non-forest post-treatment
        if mask is not None:
            arr = xr.where(mask, arr, lcp.sel(band=idx))  # in-situ: only treated pixels
        out.loc[dict(band=idx)] = arr

    # --- 2. Surface fuel remapping
    if surface_nm not in surface_df.columns:
        raise ValueError(f"Surface scenario '{surface_nm}' not a column in surface_df.")
    lut    = build_surface_lut(surface_df, surface_nm)
    fidx   = band_map["FBFM40"]
    fm_raw = out.sel(band=fidx).values           # float; may contain NaN (masked nodata)
    nodata = np.isnan(fm_raw)
    fm     = np.where(nodata, 0, fm_raw).astype(np.int32)  # safe cast; 0 is placeholder

    fm_clipped = np.clip(fm, 0, len(lut) - 1)
    fm_new     = lut[fm_clipped].astype(np.int16)
    fm_new     = np.where(nodata, fm_raw, fm_new)           # restore nodata pixels

    if mask is not None:
        fm_new = np.where(mask.values, fm_new, out.sel(band=fidx).values)
    out.loc[dict(band=fidx)] = fm_new

    return out


def mask_lcp(lcp, mask, nodata=None, out_path=None, compress="lzw", tiled=True):
    """
    Restrict an LCP to an analysis area by masking every band outside it.

    All bands — topography, fuel, and canopy — are set to *nodata* wherever
    *mask* is false.  FlamMap evaluates fire behaviour per pixel, so it does
    not need continuous terrain across the extent and simply returns NoData
    for masked cells.  (WindNinja-generated gridded winds are the exception:
    those solve over the terrain surface and do need it continuous.)

    Parameters
    ----------
    lcp : xarray.DataArray or str or Path
        Multi-band landscape raster with a ``long_name`` attribute naming the
        bands (as written by :func:`~fb_tools.fuelscape.lfps.lfps_request`).
        Opened unmasked when a path is given, to keep the integer dtype.
    mask : xarray.DataArray
        Boolean (or 0/1) DataArray aligned to a single band of *lcp*.  Pixels
        that are true are **kept**; everything else becomes *nodata*.
    nodata : int or float, optional
        Fill value written outside the mask.  Defaults to the raster's own
        NoData value, falling back to ``-9999``.  An integer fill keeps the
        LCP in its native integer dtype; passing ``np.nan`` forces the array
        to ``float32``, which doubles it in memory.
    out_path : str or Path, optional
        If given, write the result to GeoTIFF at this path, preserving band
        names and dtype.
    compress : str or None
        GeoTIFF compression passed through to ``rio.to_raster`` (default
        ``"lzw"``, matching what LFPS ships).  Masking sets most of the grid
        to a single constant, which compresses hard — leaving this unset
        writes a file several times *larger* than the source.  ``None``
        writes uncompressed.
    tiled : bool
        Write a tiled GeoTIFF (default ``True``).  Ignored when *compress*
        is ``None``.

    Returns
    -------
    xarray.DataArray
        A masked copy of *lcp*.  Same shape as the input; same dtype unless
        *nodata* is NaN.

    Raises
    ------
    ValueError
        If *lcp* has no ``long_name`` band metadata.

    Notes
    -----
    This does not shrink the grid — FlamMap still walks the full rectangle,
    it just skips masked cells.  Crop beforehand if the analysis area sits in
    one corner of the extent, but check the mask's bounding box first: a mask
    scattered across the landscape crops to nothing.

    Examples
    --------
    >>> keep = bps_band.isin(analysis_values)
    >>> masked = mask_lcp(baseline_lcp, keep, out_path="LF2016_LCP_bpsmask.tif")
    """
    if isinstance(lcp, (str, Path)):
        # masked=False keeps the native integer dtype; masked=True would give
        # float + NaN before we have decided what the fill should be.
        lcp = rxr.open_rasterio(Path(lcp), masked=False)

    long_names = lcp.attrs.get("long_name", [])
    if isinstance(long_names, str):
        long_names = [long_names]
    if not long_names:
        raise ValueError(
            "lcp has no 'long_name' band metadata — refusing to write a "
            "landscape whose bands cannot be identified."
        )

    if nodata is None:
        nodata = lcp.rio.nodata
        if nodata is None:
            nodata = -9999

    keep = mask.astype(bool)
    if "band" in keep.dims:
        keep = keep.squeeze("band", drop=True)

    # NaN cannot live in an integer array; everything else keeps the dtype.
    dtype = np.float32 if np.isnan(np.array(nodata, dtype="float64")) else lcp.dtype

    out = xr.where(keep, lcp, nodata).astype(dtype)
    out = out.transpose(*lcp.dims)
    out.attrs = dict(lcp.attrs)
    out.attrs["long_name"] = tuple(long_names)
    out.rio.write_crs(lcp.rio.crs, inplace=True)
    out.rio.write_transform(lcp.rio.transform(), inplace=True)
    out.rio.write_nodata(nodata, inplace=True)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_kwargs = {}
        if compress is not None:
            write_kwargs["compress"] = compress
            write_kwargs["tiled"] = tiled
        out.rio.to_raster(out_path, dtype=str(np.dtype(dtype)), **write_kwargs)
        # rioxarray drops per-band descriptions; write them back so the file
        # round-trips through get_band_by_longname and plot_bands.
        import rasterio
        with rasterio.open(out_path, "r+") as dst:
            dst.descriptions = tuple(long_names)
        size_gb = out_path.stat().st_size / 1e9
        print(
            f"[mask_lcp] wrote {out_path} ({np.dtype(dtype)}, {out.shape}, "
            f"nodata={nodata}, compress={compress}, {size_gb:.2f} GB)"
        )

    return out
