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


def apply_treatment(lcp, canopy_df, surface_df, scenario, band_map=None, mask=None):
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
        ).astype(np.int16)
        arr = xr.where(is_forest, arr, 0)              # zero out non-forest post-treatment
        if mask is not None:
            arr = xr.where(mask, arr, lcp.sel(band=idx))  # in-situ: only treated pixels
        out.loc[dict(band=idx)] = arr

    # --- 2. Surface fuel remapping
    if surface_nm not in surface_df.columns:
        raise ValueError(f"Surface scenario '{surface_nm}' not a column in surface_df.")
    lut  = build_surface_lut(surface_df, surface_nm)
    fidx = band_map["FBFM40"]
    fm   = out.sel(band=fidx).values.astype(np.int32)

    # clip to LUT bounds before indexing
    fm_clipped = np.clip(fm, 0, len(lut) - 1)
    fm_new = lut[fm_clipped].astype(np.int16)

    if mask is not None:
        fm_new = np.where(mask.values, fm_new, out.sel(band=fidx).values)
    out.loc[dict(band=fidx)] = fm_new

    return out
