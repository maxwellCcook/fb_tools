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
