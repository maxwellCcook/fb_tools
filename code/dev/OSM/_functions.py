import pyproj
import numpy as np
from matplotlib.ticker import FuncFormatter

def _lonlat_ticks(ax, crs_proj, n_ticks=5, precision=2):
    """
    Format x/y ticks as lon/lat labels while keeping projected data.

    Parameters
    ----------
    ax : matplotlib Axes
    crs_proj : CRS or EPSG code
        The CRS of the plotted data (e.g., EPSG:5070).
    n_ticks : int
        Approximate number of ticks per axis.
    precision : int
        Decimal places for labels.
    """

    transformer = pyproj.Transformer.from_crs(
        crs_proj, "EPSG:4326", always_xy=True
    )

    # current plot extent
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xticks = np.linspace(xmin, xmax, n_ticks)
    yticks = np.linspace(ymin, ymax, n_ticks)

    def format_lon(x, _):
        lon, _ = transformer.transform(x, ymin)
        return f"{lon:.{precision}f}°"

    def format_lat(y, _):
        _, lat = transformer.transform(xmin, y)
        return f"{lat:.{precision}f}°"

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))