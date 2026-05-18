"""
Plotting utilities for fb_tools raster outputs.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr


def _band_label(name):
    """Normalise an LFPS product code (or any long_name) to a short label.

    Examples: ``200CC_20`` → ``CC``, ``ELEV2020`` → ``ELEV``,
    ``200F40_20`` → ``FBFM40``, ``US_200CBH_20`` → ``CBH``.
    """
    name = re.sub(r'^[A-Z]{2}_', '', name)   # strip region prefix (US_, AK_)
    name = re.sub(r'^\d{3}', '', name)        # strip LF version (200, 220, …)
    name = re.sub(r'_\d{2}$', '', name)       # strip year suffix (_19, _20)
    name = re.sub(r'\d{4}$', '', name)        # strip 4-digit year (ELEV2020)
    if name == 'F40':
        name = 'FBFM40'
    if name == 'SLPD':
        name = 'SLP'
    return name


# Per-band colormaps keyed on canonical (normalised) band names.
_BAND_CMAPS = {
    "ELEV":   "terrain",
    "SLP":    "YlOrRd",
    "ASP":    "twilight",
    "FBFM40": "tab20",
    "CC":     "Greens",
    "CH":     "YlGn",
    "CBH":    "YlGn",
    "CBD":    "YlGn",
    "EVT":    "tab20b",
}


def plot_bands(raster, titles=None, cols=3, figsize=None, cmap=None):
    """
    Plot every band of a multiband raster in a grid of panels.

    Parameters
    ----------
    raster : str, Path, or xarray.DataArray
        Multiband raster.  If a file path, it is opened with rioxarray.
    titles : list of str, optional
        Panel titles.  Defaults to the ``long_name`` attribute if present,
        otherwise ``Band 1``, ``Band 2``, …
    cols : int
        Number of columns in the grid (default 3).
    figsize : tuple, optional
        Figure size ``(width, height)``.  Auto-sized if omitted.
    cmap : str, optional
        Colormap to use for all bands.  If omitted, a per-band default is
        used for known LANDFIRE layers, falling back to ``'terrain'``.

    Returns
    -------
    fig, axes : matplotlib Figure and ndarray of Axes
    """
    if not hasattr(raster, "values"):
        raster = rxr.open_rasterio(Path(raster), masked=True)

    n_bands = raster.shape[0]

    # --- resolve titles
    if titles is None:
        long_names = raster.attrs.get("long_name", [])
        if isinstance(long_names, str):
            long_names = [long_names]
        if long_names and len(long_names) == n_bands:
            titles = [_band_label(ln) for ln in long_names]
        else:
            titles = [f"Band {i + 1}" for i in range(n_bands)]

    # --- layout
    rows = int(np.ceil(n_bands / cols))
    if figsize is None:
        figsize = (cols * 4, rows * 3.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_bands):
        ax = axes[i]
        band_data = raster.isel(band=i).values.astype(float)

        # resolve colormap
        if cmap:
            band_cmap = cmap
        else:
            band_cmap = _BAND_CMAPS.get(titles[i], "terrain")

        im = ax.imshow(band_data, cmap=band_cmap, interpolation="none")
        ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide any unused axes
    for ax in axes[n_bands:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_fm_timeseries(
    df,
    date_col: str = "date",
    fm_cols=("FM1", "FM10", "FM100"),
    gridmet_fm100_col: str | None = "fm100",
    title: str | None = None,
    fire_season: tuple[int, int] = (91, 304),
):
    """Diagnostic plot for `build_fm_timeseries` output.

    Three panels stacked: (1) FM1/FM10/FM100 time series with optional native
    GridMET fm100 overlay; (2) monthly mean ordering check FM100 >= FM10 >= FM1;
    (3) histogram of monotonicity violations during the fire-season window.
    Returns (fig, axes). No file I/O.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values(date_col).reset_index(drop=True)
    doy = work[date_col].dt.dayofyear
    fs = (doy >= fire_season[0]) & (doy <= fire_season[1])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    palette = {"FM1": "#cc4125", "FM10": "#e69138", "FM100": "#3d85c6"}

    ax = axes[0]
    for c in fm_cols:
        ax.plot(work[date_col], work[c], lw=0.7, label=c,
                color=palette.get(c, None))
    if gridmet_fm100_col and gridmet_fm100_col in work.columns:
        ax.plot(work[date_col], work[gridmet_fm100_col], lw=0.7,
                ls="--", color="black", alpha=0.6,
                label=f"{gridmet_fm100_col} (GridMET native)")
    ax.set_ylabel("Fuel moisture (%)")
    ax.set_title(title or "NFDRS78 lag-derived dead FM time series")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    monthly = (work.assign(month=work[date_col].dt.month)
                    .groupby("month")[list(fm_cols)].mean())
    monthly.plot(ax=ax, marker="o", color=[palette.get(c) for c in monthly.columns])
    ax.set_xlabel("Month")
    ax.set_ylabel("FM (%) monthly mean")
    ax.set_title("Monthly means — heavier fuels should sit above lighter ones")
    ax.grid(alpha=0.3)

    ax = axes[2]
    fs_df = work[fs]
    v_10_vs_1   = (fs_df["FM10"]  < fs_df["FM1"]).sum()
    v_100_vs_10 = (fs_df["FM100"] < fs_df["FM10"]).sum()
    n_fs = max(len(fs_df), 1)
    pcts = [100.0 * v_10_vs_1 / n_fs, 100.0 * v_100_vs_10 / n_fs]
    ax.bar(["FM10 < FM1", "FM100 < FM10"], pcts,
           color=["#e69138", "#3d85c6"])
    ax.set_ylabel("% of fire-season days violated")
    ax.set_title(f"Monotonicity violations (fire season, n={n_fs:,})")
    for i, v in enumerate(pcts):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return fig, axes
