"""
LCP / fuelscape raster utilities.

Covers:
  - Stacking individual FlamMap output bands into a single multi-band GeoTIFF.
  - Creating an ASCII ignition grid from point features.
  - Selecting a band from a multi-band DataArray by its long_name attribute.
"""

import gc
from pathlib import Path

import pandas as pd
import rioxarray as rxr
import xarray as xr


def get_band_by_longname(da, long_name_value):
    """
    Select a band from a multi-band DataArray by its ``long_name`` attribute.

    Parameters
    ----------
    da : xarray.DataArray
        Multi-band rioxarray DataArray with a ``long_name`` attribute list.
    long_name_value : str
        The name to search for (e.g. ``"FBFM40"``).

    Returns
    -------
    xarray.DataArray
        Single-band DataArray.

    Raises
    ------
    ValueError
        If no ``long_name`` attribute is present or the value is not found.
    """
    longnames = da.attrs.get("long_name", [])
    if not longnames:
        raise ValueError("No long_name attribute found in DataArray.")
    if long_name_value not in longnames:
        raise ValueError(f"{long_name_value!r} not found in long_name list: {longnames}")
    idx = list(longnames).index(long_name_value) + 1  # bands are 1-based in xarray
    return da.sel(band=idx)


def stack_rasters(in_dir, tag=None, out_dir=None, cleanup=True):
    """
    Stack individual single-band GeoTIFFs in *in_dir* into one multi-band file.

    Expects files named ``<scenario>_<bandname>.tif``
    (e.g. ``PCT25_FLAMELENGTH.tif``).  The stacked output is written to
    ``<out_dir>/<SCENARIO>_<TAG>.tif``.

    Parameters
    ----------
    in_dir : str or Path
        Directory containing single-band TIFFs.
    tag : str, optional
        Label for the fuelscape (derived from ``in_dir.name`` if omitted).
    out_dir : str or Path, optional
        Destination directory (defaults to *in_dir*).
    cleanup : bool
        Delete the input single-band TIFFs after stacking (default True).
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir) if out_dir else in_dir

    parts = in_dir.name.split("_", 1)
    if tag is None:
        tag = parts[1].upper() if len(parts) > 1 else in_dir.name.upper()

    file_prefix = parts[0].upper()

    tifs = sorted(in_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No TIFFs found in {in_dir}")

    bands, band_names = [], []
    for tif in tifs:
        band_name = tif.stem.split("_")[1]
        band_names.append(band_name)
        with rxr.open_rasterio(tif) as da:
            da = da.squeeze()
            da.attrs["long_name"] = f"{file_prefix}_{band_name}"
            bands.append(da)

    stack = xr.concat(bands, dim=pd.Index(band_names, name="band"))
    out_fp = out_dir / f"{file_prefix}_{tag.upper()}.tif"
    stack.rio.to_raster(out_fp, compress="deflate")
    print(f"Stacked {len(tifs)} rasters → {out_fp}")

    del stack, bands
    gc.collect()

    if cleanup:
        for tif in tifs:
            tif.unlink()
            aux = tif.with_suffix(".tif.aux")
            if aux.exists():
                aux.unlink()


def create_ignition_ascii(ign_gdf, ref_img_fp, out_ascii_fp):
    """
    Rasterize ignition points to an ASCII grid (.asc) snapped to a reference raster.

    Parameters
    ----------
    ign_gdf : GeoDataFrame
        Ignition point features (must be in the same CRS as the reference raster).
    ref_img_fp : str or Path
        Path to the reference raster (e.g. your landscape .tif).
    out_ascii_fp : str or Path
        Output path for the ASCII grid file.
    """
    from ..utils.geo import rasterize  # avoid circular at module level

    ign_gdf = ign_gdf.copy()
    ign_gdf["burn"] = 1

    ref_img = rxr.open_rasterio(ref_img_fp)[0]
    ign_grid = rasterize(ign_gdf, to_img=ref_img, attr="burn", fill_val=0)
    ign_grid.rio.to_raster(str(out_ascii_fp), driver="AAIGrid", nodata=0)

    del ref_img
    print(f"Saved ignition grid → {out_ascii_fp}")
