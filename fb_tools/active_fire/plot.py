"""Minimal, manuscript-style plots for active-fire products. No titles; every call returns ``(fig, ax)``."""
import numpy as np
import pandas as pd

SCALES = ("linear", "log", "sqrt")
SATELLITE_COLORS = {"VIIRS S-NPP": "#1b9e77", "VIIRS NOAA-20": "#d95f02", "VIIRS NOAA-21": "#7570b3",
                    "MODIS": "#e7298a", "GOES": "#66a61e", "Landsat": "#a6761d"}


def color_norm(values, scale: str = "linear", vmin=None, vmax=None):
    """
    Matplotlib colour normalization for a value scale.

    Parameters
    ----------
    values : array-like
    scale : {"linear", "log", "sqrt"}
        ``"log"`` needs positive values: the lower limit defaults to the smallest
        positive value, and zeros or negatives fall below the scale.
    vmin, vmax : float, optional
        Default to the data range.

    Returns
    -------
    matplotlib.colors.Normalize
    """
    from matplotlib import colors

    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}; got {scale!r}.")
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)] if scale == "log" else v[np.isfinite(v)]
    if not len(v):
        raise ValueError(f"No values to plot on a {scale} scale.")
    vmin, vmax = (v.min() if vmin is None else vmin), (v.max() if vmax is None else vmax)
    if scale == "log":
        return colors.LogNorm(vmin, vmax)
    if scale == "sqrt":
        return colors.PowerNorm(0.5, vmin, vmax)
    return colors.Normalize(vmin, vmax)


def plot_grid(grid_gdf, column: str, *, scale: str = "linear", cmap="inferno", vmin=None, vmax=None,
              events=None, label=None, ax=None, figsize=(5, 5)):
    """
    Map one column of a grid GeoDataFrame.

    Parameters
    ----------
    grid_gdf : GeoDataFrame
        From ``gridded_frp`` or ``first_detection_grid``.
    column : str
        A datetime column is drawn as days since its first UTC midnight.
    scale : {"linear", "log", "sqrt"}
        Colour scale. On ``"log"``, cells with a value ≤ 0 are left uncoloured.
    cmap, vmin, vmax
    events : GeoDataFrame, optional
        Perimeters outlined on top.
    label : str, optional
        Colour-bar label; the column name by default.
    ax : matplotlib axes, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    vals = grid_gdf[column]
    if pd.api.types.is_datetime64_any_dtype(vals):
        t0 = vals.min().normalize()
        vals, label = (vals - t0) / pd.Timedelta(days=1), label or f"days since {t0:%Y-%m-%d}"
    vals = pd.to_numeric(vals, errors="coerce")
    norm = color_norm(vals, scale, vmin, vmax)
    show = vals.notna() & ((vals > 0) if scale == "log" else True)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    grid_gdf[show].assign(_v=vals[show]).plot(column="_v", cmap=cmap, norm=norm, ax=ax, linewidth=0)
    if events is not None:
        events.to_crs(grid_gdf.crs).boundary.plot(ax=ax, color="k", linewidth=0.8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.figure.colorbar(sm, ax=ax, shrink=0.7, label=label or column)
    ax.set_axis_off()
    return ax.figure, ax


def plot_detections(gdf, color: str = "frp", *, by=None, scale: str = "linear", cmap=None, vmin=None,
                    vmax=None, events=None, tz=None, crs="EPSG:5070", markersize: float = 2, label=None,
                    ax=None, figsize=None):
    """
    Map detections, optionally one panel per satellite.

    Parameters
    ----------
    gdf : GeoDataFrame
        Detections (points) or ``footprint_boxes`` output (drawn as boxes).
    color : str
        ``"frp"``, ``"time"`` (days since the first day), ``"satellite"`` (one colour
        each), or any numeric column.
    by : str, optional
        Column to split into panels (e.g. ``"satellite"``). Panels share one extent
        and one colour scale and carry a small corner label.
    scale : {"linear", "log", "sqrt"}
        Colour scale for numeric ``color``. On ``"log"``, values ≤ 0 are not drawn.
    cmap, vmin, vmax
    events : GeoDataFrame, optional
        Perimeters outlined on every panel.
    tz : str, optional
        Time zone for ``color="time"``; UTC days if omitted.
    crs : any
        Map CRS.
    markersize : float
    label : str, optional
        Colour-bar label.
    ax : matplotlib axes, optional
        Single panel only (``by=None``).
    figsize : tuple, optional

    Returns
    -------
    fig, ax
        ``ax`` is an array of axes when ``by`` is given.
    """
    import matplotlib.pyplot as plt

    g = gdf.to_crs(crs)
    ev = events.to_crs(crs) if events is not None else None
    categorical = color == "satellite"
    if categorical:
        cats = sorted(g["satellite"].dropna().unique())
        spare = iter(plt.cm.tab10.colors)
        palette = {c: SATELLITE_COLORS.get(c) or next(spare) for c in cats}
    else:
        if color == "time":
            t = g["acq_datetime"].dt.tz_convert(tz or "UTC")
            t0 = t.min().normalize()
            vals, label = (t - t0) / pd.Timedelta(days=1), label or f"days since {t0:%Y-%m-%d}"
        else:
            vals = pd.to_numeric(g[color], errors="coerce")
        norm = color_norm(vals, scale, vmin, vmax)
        cmap = cmap or ("viridis" if color == "time" else "inferno")
        ok = vals.notna() & ((vals > 0) if scale == "log" else True)
        g = g[ok].assign(_v=vals[ok]).sort_values("_v")         # highest values drawn last, on top

    groups = [(None, g)] if by is None else list(g.groupby(by, sort=True))
    if ax is None or by is not None:
        n = len(groups)
        fig, axes = plt.subplots(1, n, figsize=figsize or (4.2 * n + (0 if categorical else 0.8), 4.6),
                                 sharex=True, sharey=True, constrained_layout=True, squeeze=False)
        axes = axes[0]
    else:
        fig, axes = ax.figure, [ax]
    bounds = np.vstack([g.total_bounds] + ([ev.total_bounds] if ev is not None else []))
    x0, y0, x1, y1 = *bounds[:, :2].min(0), *bounds[:, 2:].max(0)
    pad = 0.04 * max(x1 - x0, y1 - y0) or 1.0                # room for the corner label
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad

    for a, (name, part) in zip(axes, groups):
        if categorical:
            for c, sub in part.groupby("satellite"):
                sub.plot(ax=a, color=palette[c], markersize=markersize, linewidth=0)
        else:
            part.plot(ax=a, column="_v", cmap=cmap, norm=norm, markersize=markersize, linewidth=0)
        if ev is not None:
            ev.boundary.plot(ax=a, color="k", linewidth=0.8)
        if name is not None:
            a.text(0.02, 0.98, f"{name}  (n = {len(part):,})", transform=a.transAxes, va="top", fontsize=8)
        a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal"); a.set_axis_off()

    if categorical:
        handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=palette[c], label=c) for c in cats]
        axes[-1].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    else:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), shrink=0.7, label=label or color)
    return fig, (axes[0] if by is None else axes)


def plot_detection_series(gdf, value: str = "count", *, step: str = "day", by="satellite", tz=None, t0=None,
                          cumulative: bool = False, scale: str = "linear", ax=None, figsize=(6, 3)):
    """
    Detections or summed FRP per day or per overpass, as bars stacked by satellite.

    A detection count is fire *activity*, not growth: a cell that keeps burning is
    counted on every overpass. Area first detected per day (``first_detection_grid``)
    is the growth measure.

    Parameters
    ----------
    gdf : GeoDataFrame
    value : {"count", "frp_sum"}
        ``"frp_sum"`` sums FRP within each satellite; the stack is for comparison of
        timing, not a pooled total.
    step : {"day", "overpass"}
        Day bins (UTC, or local with ``tz``) or overpasses (``assign_overpass``;
        drawn at their time, in ``tz``).
    by : str or None
        Stacking column; None for a single series.
    tz : str, optional
    t0 : date-like, optional
        Day 0 for ``step="day"``; the first detection day by default.
    cumulative : bool
        Add the cumulative share of the total (%) as a line on a right-hand axis.
    scale : {"linear", "log"}
        Y-axis scale.
    ax : matplotlib axes, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    from .timeutil import assign_overpass, day_of

    if value not in ("count", "frp_sum") or step not in ("day", "overpass"):
        raise ValueError("value must be 'count' or 'frp_sum'; step must be 'day' or 'overpass'.")
    g = gdf
    if step == "day":
        day = day_of(g["acq_datetime"], tz)
        t0 = day.min() if t0 is None else pd.Timestamp(t0).normalize()
        x, xlabel, width = (day - t0).dt.days, f"days since {t0:%Y-%m-%d}", 0.85
    else:
        g = g if "overpass_time" in g else assign_overpass(g)
        x = g["overpass_time"].dt.tz_convert(tz or "UTC").dt.tz_localize(None)
        xlabel, width = "", 0.04
    key = g[by] if by else pd.Series("all", index=g.index)
    vals = pd.to_numeric(g["frp"], errors="coerce") if value == "frp_sum" else pd.Series(1, index=g.index)
    table = vals.groupby([x.to_numpy(), key.to_numpy()]).sum().unstack(fill_value=0)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bottom = np.zeros(len(table))
    spare = iter(plt.cm.tab10.colors)
    for name in table.columns:
        colour = SATELLITE_COLORS.get(name) or ("0.3" if by is None else next(spare))
        ax.bar(table.index, table[name], width, bottom=bottom, color=colour, label=name, linewidth=0)
        bottom += table[name].to_numpy()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("detections" if value == "count" else "FRP (MW)")
    ax.set_yscale(scale)
    ax.spines[["top", "right"]].set_visible(False)
    if by:
        ax.legend(frameon=False, fontsize=8, loc="center right" if cumulative else "upper right")
    if cumulative:
        ax2 = ax.twinx()
        ax2.plot(table.index, 100 * np.cumsum(bottom) / bottom.sum(), color="k", lw=1, marker="o", ms=2.5)
        ax2.set_ylim(0, 105); ax2.set_ylabel("cumulative share (%)")
        ax2.spines["top"].set_visible(False)
    return ax.figure, ax


def _scale_bar(ax, frac: float = 0.2):
    """Scale bar (1-2-5 km steps, about ``frac`` of the panel width) and north mark, lower left."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    target = frac * (x1 - x0) / 1000
    km = max(s * 10 ** np.floor(np.log10(target)) for s in (1, 2, 5) if s * 10 ** np.floor(np.log10(target)) <= target)
    bx, by = x0 + 0.04 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    ax.plot([bx, bx + km * 1000], [by, by], color="0.2", lw=2, solid_capstyle="butt")
    ax.text(bx + km * 500, by - 0.012 * (y1 - y0), f"{km:g} km", ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.85))
    ax.annotate("N", xy=(bx, by + 0.12 * (y1 - y0)), xytext=(bx, by + 0.03 * (y1 - y0)), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.2))


def plot_footprint_workflow(detections, boxes=None, grid_gdf=None, column: str = "frp_csum", *,
                            scale: str = "linear", cmap="inferno", vmin=None, vmax=None, events=None,
                            extent=None, label=None, markersize: float = 4, figsize=None):
    """
    Three-panel diagram of the gridding workflow: (a) detection centres by satellite,
    (b) their scan x track footprints, overlapping, (c) the gridded product.

    Parameters
    ----------
    detections : GeoDataFrame
        Points in the shared schema.
    boxes : GeoDataFrame, optional
        ``footprint_boxes`` output; built from ``detections`` in the grid's CRS
        (EPSG:5070 without a grid) when omitted.
    grid_gdf : GeoDataFrame, optional
        ``gridded_frp`` or ``first_detection_grid`` output. Without it only (a) and
        (b) are drawn.
    column : str
        Grid column for (c).
    scale, cmap, vmin, vmax
        Colour scale of (c), as in ``plot_grid``.
    events : GeoDataFrame, optional
        Perimeters outlined on every panel.
    extent : tuple or GeoDataFrame, optional
        ``(xmin, ymin, xmax, ymax)`` in the map CRS, or a geometry frame, to zoom
        in. On a large fire the overlapping footprints only read at a zoom.
    label : str, optional
        Colour-bar label for (c).
    markersize : float
    figsize : tuple, optional

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    from .footprints import footprint_boxes

    crs = grid_gdf.crs if grid_gdf is not None else (boxes.crs if boxes is not None else "EPSG:5070")
    pts = detections.to_crs(crs)
    boxes = footprint_boxes(detections, crs=crs, instruments=None) if boxes is None else boxes.to_crs(crs)
    if extent is None:
        b = np.vstack([boxes.total_bounds] + ([grid_gdf.total_bounds] if grid_gdf is not None else []))
        x0, y0, x1, y1 = *b[:, :2].min(0), *b[:, 2:].max(0)
        pad = 0.04 * max(x1 - x0, y1 - y0)
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    else:
        x0, y0, x1, y1 = extent.to_crs(crs).total_bounds if hasattr(extent, "total_bounds") else extent
        pts, boxes = pts.cx[x0:x1, y0:y1], boxes.cx[x0:x1, y0:y1]
        grid_gdf = grid_gdf.cx[x0:x1, y0:y1] if grid_gdf is not None else None

    n = 3 if grid_gdf is not None else 2
    fig, axes = plt.subplots(1, n, figsize=figsize or (4.2 * n + (0.8 if n == 3 else 0), 4.6),
                             constrained_layout=True)
    plot_detections(pts, "satellite", crs=crs, markersize=markersize, ax=axes[0])
    spare = iter(plt.cm.tab10.colors)
    palette = {c: SATELLITE_COLORS.get(c) or next(spare) for c in sorted(boxes["satellite"].dropna().unique())}
    b = boxes.dropna(subset=["satellite"]).sort_values("acq_datetime")      # time order, so no satellite sits on top
    colours = b["satellite"].map(palette).to_list()
    b.plot(ax=axes[1], color=colours, alpha=0.25, linewidth=0)
    b.boundary.plot(ax=axes[1], color=colours, linewidth=0.4, alpha=0.8)
    if grid_gdf is not None:
        plot_grid(grid_gdf, column, scale=scale, cmap=cmap, vmin=vmin, vmax=vmax, label=label, ax=axes[2])
    for a, tag in zip(axes, "abc"):
        if events is not None:
            events.to_crs(crs).boundary.plot(ax=a, color="k", linewidth=0.8)
        a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal"); a.set_axis_off()
        a.text(0.02, 0.98, f"({tag})", transform=a.transAxes, va="top", fontsize=11, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=0.85), zorder=10)
    leg = axes[0].get_legend()
    leg.set_loc("upper right"); leg.set_frame_on(True); leg.get_frame().set(facecolor="w", edgecolor="none", alpha=0.85)
    _scale_bar(axes[0])
    return fig, axes
