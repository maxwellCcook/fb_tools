"""
File I/O utilities for fb_tools.
"""

import glob
import os
from pathlib import Path


def list_files(path, ext, recursive=True):
    """
    Find files under *path* matching a glob extension pattern.

    Parameters
    ----------
    path : str or Path
        Directory to search.
    ext : str
        Extension or glob pattern to match (e.g. "*.tif", ".csv").
    recursive : bool
        Search recursively (default True).

    Returns
    -------
    list[str]
        Matching file paths.
    """
    path = str(path)
    # normalise ext: accept both "*.tif" and ".tif"
    if not ext.startswith("*"):
        pattern = f"*{ext}"
    else:
        pattern = ext

    if recursive:
        return glob.glob(os.path.join(path, "**", pattern), recursive=True)
    else:
        return glob.glob(os.path.join(path, pattern), recursive=False)


_ZSTD_OK = None


def _zstd_available():
    """True if the linked GDAL can write ZSTD-compressed GeoTIFFs."""
    global _ZSTD_OK
    if _ZSTD_OK is None:
        import warnings
        import numpy as np
        from rasterio.errors import NotGeoreferencedWarning
        from rasterio.io import MemoryFile
        try:
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with MemoryFile() as mem:
                with mem.open(driver="GTiff", height=8, width=8, count=1,
                              dtype="int16", compress="zstd") as dst:
                    dst.write(np.zeros((8, 8), "int16"), 1)
            _ZSTD_OK = True
        except Exception:
            _ZSTD_OK = False
    return _ZSTD_OK


def raster_write_kwargs(compress="deflate", blocksize=512, zstd_level=9):
    """
    GeoTIFF creation options for fb_tools raster writes.

    Tiled DEFLATE with **no predictor**.  Tiling is the portable win — measured
    1.38x smaller than the striped default on a Pyrome 46 FlamMap stack, and it
    makes the windowed reads downstream pixel extraction depends on far cheaper.

    ``compress="zstd"`` compresses considerably harder (2.36x on the same stack,
    1.48x on an int16 LCP) but is **not the default, deliberately**: QGIS ships
    a GDAL built without the ZSTD codec, so a ZSTD GeoTIFF fails to open there
    with "Cannot open TIFF file due to missing codec ZSTD".  Anything a human
    will inspect should stay DEFLATE.  Pass ``compress="zstd"`` only for
    intermediates that are read solely by this package.

    Horizontal differencing (``predictor=2``) is deliberately omitted for
    integer as well as float dtypes — these grids are dominated by long
    ``-9999`` runs that RLE well, and differencing destroys them (~35% larger
    on the same LCP).

    Parameters
    ----------
    compress : str
        Compressor (default ``"deflate"`` — see above).  ``"zstd"`` falls back
        to ``"deflate"`` when the linked GDAL lacks the codec.
    blocksize : int
        Internal tile size (default 512).
    zstd_level : int
        ZSTD level, ignored for other compressors.

    Returns
    -------
    dict
        Keyword arguments for ``DataArray.rio.to_raster`` / ``rasterio.open``.
    """
    if compress == "zstd" and not _zstd_available():
        compress = "deflate"
    kw = dict(compress=compress, tiled=True,
              blockxsize=blocksize, blockysize=blocksize)
    if compress == "zstd":
        kw["zstd_level"] = zstd_level
    return kw
