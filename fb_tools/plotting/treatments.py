"""
Fire behavior treatment change visualization.

Plotting functions for summarize_treatments() output DataFrames.

Functions
---------
plot_fl_stackedbar
    Paired horizontal stacked bars (baseline faded / treated solid)
    for a single fire weather percentile, one pair per scenario.

plot_fl_delta_bar
    Faceted grouped horizontal delta bar chart, one panel per percentile,
    FL bins on y-axis, scenarios grouped within each bin.

plot_cs_stackedbar_multipct
    Multi-panel crown state stacked bars, one panel per percentile.
    Scenario order fixed by Surface class improvement at the first percentile.

plot_sdi_boxplot
    Grouped boxplots of delta SDI by treatment type. Optional secondary
    grouping by POD position (Interior / Line).

All functions return ``(fig, axes)`` for downstream customization and accept
an optional *save_path* to write PNG/PDF at *dpi* resolution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Module-level constants — override after import if needed
# ---------------------------------------------------------------------------

FL_CLASSES = ["0-2ft", "2-4ft", "4-8ft", "8-12ft", ">12ft"]

# Crown state labels matching fb_tools _CS_CLASSES values exactly
CS_CLASSES = ["NonBurnable", "Surface", "Passive Crown", "Active Crown"]

FL_CLASS_COLORS = {
    "0-2ft":  "#1B5E20",
    "2-4ft":  "#66BB6A",
    "4-8ft":  "#FFF176",
    "8-12ft": "#EF6C00",
    ">12ft":  "#B71C1C",
}

CS_CLASS_COLORS = {
    "NonBurnable":   "#d4d4d4",
    "Surface":       "#91b563",
    "Passive Crown": "#e0883a",
    "Active Crown":  "#bf3b2f",
}

PCT_LABELS = {
    "Pct25": "25th percentile",
    "Pct50": "50th percentile",
    "Pct90": "90th percentile",
    "Pct97": "97th percentile",
}

PCT_COLORS = {
    "Pct25": "#90CAF9",
    "Pct50": "#1565C0",
    "Pct90": "#E53935",
    "Pct97": "#6A1B9A",
}

_FONT_TITLE = {"fontsize": 10, "fontweight": "bold", "color": "#1A1A1A"}
_FONT_LABEL = {"fontsize": 8.5, "color": "#333333"}
_FONT_TICK  = {"labelsize": 7.5}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save(fig, save_path, dpi):
    """Save figure to *save_path* if provided."""
    if save_path is not None:
        fig.savefig(str(Path(save_path)), dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved → {save_path}")


def _agg_fl(fl_df, scenario_col, id_col):
    """
    Aggregate FL pixel counts to scenario-level proportions.

    Uses pixel counts (FL_bl_count / FL_tr_count) for area-weighted
    landscape-level percentages rather than averaging per-polygon pct values.
    Baseline counts are deduplicated per (id_col, percentile, FL_class) before
    summing to avoid double-counting if a polygon appears across multiple rows.

    Parameters
    ----------
    fl_df : pandas.DataFrame
        Output of summarize_treatments()['fl'].
    scenario_col : str
        Treatment type / scenario column.
    id_col : str
        Treatment polygon ID column.

    Returns
    -------
    pandas.DataFrame
        Columns: [scenario_col, 'percentile', 'FL_class', 'bl_pct', 'tr_pct'].
    """
    bl_counts = (
        fl_df
        .drop_duplicates(subset=[scenario_col, id_col, "percentile", "FL_class"])
        .groupby([scenario_col, "percentile", "FL_class"])["FL_bl_count"]
        .sum().reset_index(name="bl_px")
    )
    bl_totals = (
        bl_counts.groupby([scenario_col, "percentile"])["bl_px"]
        .sum().reset_index(name="bl_total")
    )
    bl_agg = (
        bl_counts.merge(bl_totals, on=[scenario_col, "percentile"])
        .assign(bl_pct=lambda d: d["bl_px"] / d["bl_total"] * 100)
        [[scenario_col, "percentile", "FL_class", "bl_pct"]]
    )

    tr_counts = (
        fl_df
        .groupby([scenario_col, "percentile", "FL_class"])["FL_tr_count"]
        .sum().reset_index(name="tr_px")
    )
    tr_totals = (
        tr_counts.groupby([scenario_col, "percentile"])["tr_px"]
        .sum().reset_index(name="tr_total")
    )
    tr_agg = (
        tr_counts.merge(tr_totals, on=[scenario_col, "percentile"])
        .assign(tr_pct=lambda d: d["tr_px"] / d["tr_total"] * 100)
        [[scenario_col, "percentile", "FL_class", "tr_pct"]]
    )

    return bl_agg.merge(tr_agg, on=[scenario_col, "percentile", "FL_class"], how="outer").fillna(0)


def _agg_cs(cs_df, scenario_col, id_col):
    """
    Aggregate CS pixel counts to scenario-level proportions.

    Parameters
    ----------
    cs_df : pandas.DataFrame
        Output of summarize_treatments()['cs'].
    scenario_col : str
        Treatment type / scenario column.
    id_col : str
        Treatment polygon ID column.

    Returns
    -------
    pandas.DataFrame
        Columns: [scenario_col, 'percentile', 'CS_class', 'bl_pct', 'tr_pct'].
    """
    bl_counts = (
        cs_df
        .drop_duplicates(subset=[scenario_col, id_col, "percentile", "CS_class"])
        .groupby([scenario_col, "percentile", "CS_class"])["CS_bl_count"]
        .sum().reset_index(name="bl_px")
    )
    bl_totals = (
        bl_counts.groupby([scenario_col, "percentile"])["bl_px"]
        .sum().reset_index(name="bl_total")
    )
    bl_agg = (
        bl_counts.merge(bl_totals, on=[scenario_col, "percentile"])
        .assign(bl_pct=lambda d: d["bl_px"] / d["bl_total"] * 100)
        [[scenario_col, "percentile", "CS_class", "bl_pct"]]
    )

    tr_counts = (
        cs_df
        .groupby([scenario_col, "percentile", "CS_class"])["CS_tr_count"]
        .sum().reset_index(name="tr_px")
    )
    tr_totals = (
        tr_counts.groupby([scenario_col, "percentile"])["tr_px"]
        .sum().reset_index(name="tr_total")
    )
    tr_agg = (
        tr_counts.merge(tr_totals, on=[scenario_col, "percentile"])
        .assign(tr_pct=lambda d: d["tr_px"] / d["tr_total"] * 100)
        [[scenario_col, "percentile", "CS_class", "tr_pct"]]
    )

    return bl_agg.merge(tr_agg, on=[scenario_col, "percentile", "CS_class"], how="outer").fillna(0)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_fl_stackedbar(
    fl_df,
    percentile="Pct97",
    scenarios=None,
    scenario_col="scenario",
    id_col="TRT_ID",
    fl_classes=None,
    fl_colors=None,
    scenario_labels=None,
    figsize=None,
    save_path=None,
    dpi=300,
):
    """
    Paired horizontal stacked bars: baseline (faded) vs. treated (solid)
    for a single fire weather percentile.

    One pair of bars per scenario (treatment type), colored by FL class.
    Proportions are pixel-count weighted across all treatment polygons.

    Parameters
    ----------
    fl_df : pandas.DataFrame
        Output of summarize_treatments()['fl'].
    percentile : str
        Percentile to display (default ``'Pct90'``).
    scenarios : list of str, optional
        Ordered scenario labels to include. ``None`` plots all, sorted
        by descending mean treated proportion in the ``'>12ft'`` class.
    scenario_col : str
        Treatment type column (default ``'scenario'``).
    id_col : str
        Treatment polygon ID column (default ``'TRT_ID'``).
    fl_classes : list of str, optional
        Ordered FL class labels. Defaults to module ``FL_CLASSES``.
    fl_colors : dict, optional
        Color overrides keyed by FL class label.
    scenario_labels : dict, optional
        Display name overrides keyed by scenario value.
    figsize : tuple, optional
    save_path : str or Path, optional
    dpi : int

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    _fl_classes = fl_classes or FL_CLASSES
    _fl_colors  = {**FL_CLASS_COLORS, **(fl_colors or {})}
    _scen_labels = scenario_labels or {}

    agg = _agg_fl(fl_df, scenario_col, id_col)
    pct_data = agg[agg["percentile"] == percentile]

    if scenarios is None:
        high = (
            pct_data[pct_data["FL_class"] == ">12ft"]
            .sort_values("tr_pct", ascending=True)
        )
        scenarios = high[scenario_col].tolist()
        # add any scenarios missing from the ">12ft" slice
        for s in pct_data[scenario_col].unique():
            if s not in scenarios:
                scenarios.append(s)

    n     = len(scenarios)
    bar_h = 0.32
    gap   = 1.1
    y_bl  = np.arange(n) * gap + bar_h / 2
    y_tr  = np.arange(n) * gap - bar_h / 2

    fig, ax = plt.subplots(
        figsize=figsize or (9, max(3.5, n * 0.9)),
        facecolor="white",
        constrained_layout=True,
    )

    bl_idx = pct_data.set_index([scenario_col, "FL_class"])["bl_pct"]
    tr_idx = pct_data.set_index([scenario_col, "FL_class"])["tr_pct"]

    for j, fl_cls in enumerate(_fl_classes):
        col = _fl_colors.get(fl_cls, "#aaa")

        bl_lefts = np.zeros(n)
        tr_lefts = np.zeros(n)
        for prev_cls in _fl_classes[:j]:
            for k, s in enumerate(scenarios):
                bl_lefts[k] += bl_idx.get((s, prev_cls), 0.0)
                tr_lefts[k] += tr_idx.get((s, prev_cls), 0.0)

        bl_vals = np.array([bl_idx.get((s, fl_cls), 0.0) for s in scenarios])
        tr_vals = np.array([tr_idx.get((s, fl_cls), 0.0) for s in scenarios])

        ax.barh(y_bl, bl_vals, left=bl_lefts, height=bar_h,
                color=col, alpha=0.45, edgecolor="white", linewidth=0.5,
                label=fl_cls)
        ax.barh(y_tr, tr_vals, left=tr_lefts, height=bar_h,
                color=col, alpha=1.0, edgecolor="white", linewidth=0.5)

    ax.set_yticks(np.arange(n) * gap)
    ax.set_yticklabels(
        [_scen_labels.get(s, s) for s in scenarios], fontsize=8.5
    )
    ax.set_xlim(0, 108)
    ax.set_xlabel("% Area by Flame Length Class", **_FONT_LABEL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(**_FONT_TICK)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.xaxis.grid(True, linestyle="--", lw=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.text(101, y_bl[-1], "Baseline", fontsize=7.5, color="#888888",
            va="center", ha="left")
    ax.text(101, y_tr[-1], "Treated",  fontsize=7.5, color="#1A1A1A",
            va="center", ha="left")

    patches = [
        mpatches.Patch(facecolor=_fl_colors.get(c, "#aaa"), label=c,
                       edgecolor="white")
        for c in _fl_classes
    ]
    ax.legend(handles=patches, title="FL Class", title_fontsize=7,
              fontsize=7.5, frameon=False, loc="lower right",
              ncol=len(_fl_classes))

    pct_label = PCT_LABELS.get(percentile, percentile)
    fig.suptitle(
        f"Flame Length Class Distribution — {pct_label} Fire Weather",
        fontsize=11, fontweight="bold", color="#1A1A1A",
    )

    _save(fig, save_path, dpi)
    return fig, ax


def plot_fl_delta_bar(
    fl_df,
    percentiles=("Pct25", "Pct97"),
    scenarios=None,
    scenario_col="scenario",
    id_col="TRT_ID",
    fl_class_order=None,
    scenario_colors=None,
    scenario_labels=None,
    figsize=None,
    save_path=None,
    dpi=300,
):
    """
    Faceted grouped horizontal delta bar chart.

    Shows change in % area (treated − baseline) per FL bin and scenario.
    One panel per percentile, FL bins on y-axis, scenarios grouped within
    each bin, colored by treatment type.

    Parameters
    ----------
    fl_df : pandas.DataFrame
        Output of summarize_treatments()['fl'].
    percentiles : sequence of str
        Percentiles to render as panels.
    scenarios : list of str, optional
        Ordered scenario labels to include. ``None`` orders by descending
        delta in the ``'>12ft'`` bin at the last percentile.
    scenario_col : str
        Treatment type column (default ``'scenario'``).
    id_col : str
        Treatment polygon ID column (default ``'TRT_ID'``).
    fl_class_order : list of str, optional
        Ordered FL class labels. Defaults to module ``FL_CLASSES``.
    scenario_colors : dict, optional
        ``{scenario: color}`` mapping. Unmapped scenarios get tab10 colors.
    scenario_labels : dict, optional
        Display name overrides keyed by scenario value.
    figsize : tuple, optional
    save_path : str or Path, optional
    dpi : int

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray of Axes)
    """
    _fl_order    = fl_class_order or FL_CLASSES
    _scen_labels = scenario_labels or {}

    agg = _agg_fl(fl_df, scenario_col, id_col)
    agg = agg.assign(delta=lambda d: d["tr_pct"] - d["bl_pct"])

    if scenarios is None:
        ref_pct = list(percentiles)[-1]
        scenarios = (
            agg
            .query(f'percentile == @ref_pct and FL_class == ">12ft"')
            .groupby(scenario_col, as_index=False)
            .agg(delta_mean=("delta", "mean"))
            .sort_values("delta_mean")
            [scenario_col].tolist()
        )
        for s in agg[scenario_col].unique():
            if s not in scenarios:
                scenarios.append(s)

    # Assign colors — use supplied map then fall back to tab10
    _tab10 = plt.get_cmap("tab10").colors
    _scen_colors = {s: _tab10[i % 10] for i, s in enumerate(scenarios)}
    if scenario_colors:
        _scen_colors.update(scenario_colors)

    n_scen = len(scenarios)
    n_bins = len(_fl_order)

    band_h = 1.0
    bar_h  = (band_h * 0.72) / n_scen
    gap    = bar_h * 0.25

    bin_centers = {cls: i * band_h for i, cls in enumerate(_fl_order)}

    total_bar_span = n_scen * bar_h + (n_scen - 1) * gap
    start_offset   = -total_bar_span / 2 + bar_h / 2
    scen_offsets   = {
        s: start_offset + i * (bar_h + gap)
        for i, s in enumerate(scenarios)
    }

    n_panels = len(percentiles)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=figsize or (5.2 * n_panels, max(4.0, n_bins * 0.9)),
        sharey=True,
        facecolor="white",
    )
    axes = np.atleast_1d(axes)

    for ax, pct in zip(axes, percentiles):
        pct_data = agg[agg["percentile"] == pct]

        for scen in scenarios:
            color     = _scen_colors.get(scen, "#888888")
            scen_data = pct_data[pct_data[scenario_col] == scen]
            idx       = scen_data.set_index("FL_class")["delta"]

            for cls in _fl_order:
                if cls not in idx.index:
                    continue
                y     = bin_centers[cls] + scen_offsets[scen]
                delta = idx[cls]
                ax.barh(y, delta, height=bar_h, color=color, alpha=0.9,
                        edgecolor="white", linewidth=0.4, zorder=2)

        ax.axvline(0, color="0.3", lw=0.8, zorder=3)

        for i, cls in enumerate(_fl_order):
            if i % 2 == 0:
                ax.axhspan(
                    bin_centers[cls] - band_h / 2,
                    bin_centers[cls] + band_h / 2,
                    color="0.96", zorder=0,
                )

        ax.set_title(PCT_LABELS.get(pct, pct), fontsize=9.5, loc="center",
                     pad=6, color="0.2")
        ax.set_xlabel("Change in % area (treated − baseline)", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%")
        )
        ax.grid(axis="x", lw=0.35, color="0.90", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(list(bin_centers.values()))
    axes[0].set_yticklabels(_fl_order, fontsize=9)
    axes[0].set_ylabel("Flame length class", fontsize=9)
    axes[0].spines["left"].set_visible(False)
    axes[0].tick_params(axis="y", length=0)

    leg_patches = [
        mpatches.Patch(facecolor=_scen_colors.get(s, "#888"),
                       label=_scen_labels.get(s, s), alpha=0.9)
        for s in scenarios
    ]
    fig.legend(handles=leg_patches, title="Treatment", title_fontsize=8.5,
               fontsize=8.5, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=n_scen)

    fig.subplots_adjust(wspace=0.08, top=0.88, bottom=0.12,
                        left=0.14, right=0.97)

    _save(fig, save_path, dpi)
    return fig, axes


def plot_cs_stackedbar_multipct(
    cs_df,
    percentiles=("Pct25", "Pct97"),
    scenarios=None,
    scenario_col="scenario",
    id_col="TRT_ID",
    cs_labels=None,
    cs_colors=None,
    scenario_labels=None,
    percentile_labels=None,
    figsize=None,
    save_path=None,
    dpi=300,
):
    """
    Multi-panel paired horizontal stacked bars, one panel per percentile.

    Baseline (faded) vs. treated (solid) proportions by crown state class.
    Scenario order is fixed across all panels by Surface class increase at
    the first percentile, so rows are directly comparable across panels.
    In-bar delta annotations show Surface class change on treated bars.

    Parameters
    ----------
    cs_df : pandas.DataFrame
        Output of summarize_treatments()['cs'].
    percentiles : sequence of str
        Percentiles to render as stacked panels.
    scenarios : list of str, optional
        Ordered scenario labels to include. ``None`` orders by descending
        Surface class improvement at the first percentile.
    scenario_col : str
        Treatment type column (default ``'scenario'``).
    id_col : str
        Treatment polygon ID column (default ``'TRT_ID'``).
    cs_labels : list of str, optional
        Ordered crown state class labels. Defaults to module ``CS_CLASSES``
        (``["NonBurnable", "Surface", "Passive Crown", "Active Crown"]``).
    cs_colors : dict, optional
        Color overrides keyed by class label.
    scenario_labels : dict, optional
        Display name overrides keyed by scenario value.
    percentile_labels : dict, optional
        Display label overrides for percentile names.
    figsize : tuple, optional
    save_path : str or Path, optional
    dpi : int

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray of Axes)
    """
    _cs_labels   = cs_labels or CS_CLASSES
    _cs_colors   = {**CS_CLASS_COLORS, **(cs_colors or {})}
    _pct_labels  = percentile_labels or PCT_LABELS
    _scen_labels = scenario_labels or {}

    agg = _agg_cs(cs_df, scenario_col, id_col)

    if scenarios is None:
        first_pct = list(percentiles)[0]
        surface_delta = (
            agg
            .query("percentile == @first_pct and CS_class == 'Surface'")
            .assign(delta=lambda d: d["tr_pct"] - d["bl_pct"])
            .sort_values("delta", ascending=True)
        )
        scenarios = surface_delta[scenario_col].tolist()
        for s in agg[scenario_col].unique():
            if s not in scenarios:
                scenarios.append(s)

    n     = len(scenarios)
    bar_h = 0.35
    gap   = 0.82
    y_bl  = np.arange(n) * gap + bar_h / 2
    y_tr  = np.arange(n) * gap - bar_h / 2

    n_panels = len(percentiles)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize or (4.5, max(3.0, n * 0.82) * n_panels),
        sharex=True,
        facecolor="white",
    )
    if n_panels == 1:
        axes = [axes]

    for ax, pct in zip(axes, percentiles):
        pct_data = agg[agg["percentile"] == pct]
        bl_idx   = pct_data.set_index([scenario_col, "CS_class"])["bl_pct"]
        tr_idx   = pct_data.set_index([scenario_col, "CS_class"])["tr_pct"]

        for j, cls in enumerate(_cs_labels):
            color = _cs_colors.get(cls, "#aaa")

            bl_lefts = np.zeros(n)
            tr_lefts = np.zeros(n)
            for prev_cls in _cs_labels[:j]:
                for k, s in enumerate(scenarios):
                    bl_lefts[k] += bl_idx.get((s, prev_cls), 0.0)
                    tr_lefts[k] += tr_idx.get((s, prev_cls), 0.0)

            bl_vals = np.array([bl_idx.get((s, cls), 0.0) for s in scenarios])
            tr_vals = np.array([tr_idx.get((s, cls), 0.0) for s in scenarios])

            ax.barh(y_bl, bl_vals, left=bl_lefts, height=bar_h,
                    color=color, alpha=0.55, edgecolor="white", linewidth=0.5)
            ax.barh(y_tr, tr_vals, left=tr_lefts, height=bar_h,
                    color=color, alpha=1.0, edgecolor="white", linewidth=0.5)

        # In-bar delta annotations for the Surface class on treated bars
        for k, s in enumerate(scenarios):
            bl_surface = bl_idx.get((s, "Surface"), 0.0)
            tr_surface = tr_idx.get((s, "Surface"), 0.0)
            delta      = tr_surface - bl_surface

            # x offset: NonBurnable width + Surface width
            nb_offset = tr_idx.get((s, "NonBurnable"), 0.0)
            bar_right = nb_offset + tr_surface

            if tr_surface > 8:
                ax.text(bar_right - 1.0, y_tr[k], f"{delta:+.0f}%",
                        ha="right", va="center", fontsize=8,
                        fontweight="normal", color="0.1")

        ax.set_title(_pct_labels.get(pct, pct), fontsize=9.5, loc="center",
                     pad=8, color="0.2", fontweight="normal")
        ax.set_xlim(0, 108)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
        ax.xaxis.grid(True, linestyle="--", lw=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        if ax is axes[0]:
            ax.text(104, y_bl[-1], "Baseline", fontsize=7.5,
                    color="0.6", va="center", ha="left")
            ax.text(104, y_tr[-1], "Treated", fontsize=7.5,
                    color="0.15", va="center", ha="left")

        if ax is axes[-1]:
            ax.set_xlabel("% Area by fire type", fontsize=9)
        else:
            ax.set_xlabel("")

        ax.set_yticks(np.arange(n) * gap)
        ax.set_yticklabels(
            [_scen_labels.get(s, s) for s in scenarios], fontsize=8.5
        )

    fig.subplots_adjust(hspace=0.22, top=0.88, bottom=0.10,
                        left=0.18, right=0.94)

    patches = [
        mpatches.Patch(facecolor=_cs_colors.get(c, "#aaa"),
                       edgecolor="white", label=c)
        for c in _cs_labels
    ]
    fig.legend(handles=patches, title="Fire type", title_fontsize=8.5,
               fontsize=8.5, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=len(_cs_labels))

    _save(fig, save_path, dpi)
    return fig, axes


def plot_sdi_boxplot(
    sdi_df,
    scenario_col="scenario",
    y_col="SDI_delta_mean",
    pod_col=None,
    scenarios=None,
    scenario_colors=None,
    pod_colors=None,
    scenario_labels=None,
    figsize=None,
    save_path=None,
    dpi=300,
):
    """
    Grouped boxplots of delta SDI by treatment type.

    Parameters
    ----------
    sdi_df : pandas.DataFrame
        Output of summarize_treatments()['sdi'].
    scenario_col : str
        Treatment type column (default ``'scenario'``).
    y_col : str
        Delta SDI column (default ``'SDI_delta_mean'``).
    pod_col : str, optional
        Secondary grouping column (e.g., ``'PODLine'`` with values
        ``'Interior'`` / ``'Line'``). When ``None`` or not present in
        *sdi_df*, single-group boxplots are rendered.
    scenarios : list of str, optional
        Ordered scenario labels. ``None`` orders by ascending median delta
        SDI (most negative = leftmost).
    scenario_colors : dict, optional
        ``{scenario: color}`` mapping.
    pod_colors : dict, optional
        ``{pod_value: color}`` mapping. Defaults to blue/orange.
    scenario_labels : dict, optional
        Display name overrides.
    figsize : tuple, optional
    save_path : str or Path, optional
    dpi : int

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    _scen_labels = scenario_labels or {}
    _pod_colors  = pod_colors or {
        "Interior": "#4A7FB5",
        "Line":     "#E8955A",
    }

    use_pod = pod_col is not None and pod_col in sdi_df.columns

    if scenarios is None:
        scenarios = (
            sdi_df.groupby(scenario_col)[y_col]
            .median().sort_values().index.tolist()
        )

    # Assign colors — fall back to tab10
    _tab10 = plt.get_cmap("tab10").colors
    _scen_colors = {s: _tab10[i % 10] for i, s in enumerate(scenarios)}
    if scenario_colors:
        _scen_colors.update(scenario_colors)

    fig, ax = plt.subplots(
        figsize=figsize or (max(4.0, len(scenarios) * 1.5), 4.0),
        facecolor="white",
    )

    if use_pod:
        pod_order = list(sdi_df[pod_col].dropna().unique())
        pod_order = sorted(pod_order)
        n_pod   = len(pod_order)
        width   = 0.32
        gap     = 0.06
        total_w = n_pod * width + (n_pod - 1) * gap
        pod_offsets = np.linspace(
            -total_w / 2 + width / 2,
             total_w / 2 - width / 2,
            n_pod,
        )

        for xi, scen in enumerate(scenarios):
            scen_data = sdi_df[sdi_df[scenario_col] == scen]
            for pi, pod in enumerate(pod_order):
                pod_data = scen_data[scen_data[pod_col] == pod][y_col].dropna()
                if pod_data.empty:
                    continue
                x     = xi + pod_offsets[pi]
                color = _pod_colors.get(pod, "#888888")
                ax.boxplot(
                    pod_data,
                    positions=[x], widths=width,
                    patch_artist=True, notch=False, vert=True,
                    showfliers=False,
                    boxprops=dict(facecolor=color, alpha=0.75,
                                  edgecolor=color, linewidth=0.8),
                    medianprops=dict(color="white", linewidth=1.8),
                    whiskerprops=dict(color=color, linewidth=0.9, linestyle="-"),
                    capprops=dict(color=color, linewidth=0.9),
                )

        pod_handles = [
            mpatches.Patch(facecolor=_pod_colors.get(p, "#888"), alpha=0.75,
                           edgecolor=_pod_colors.get(p, "#888"),
                           linewidth=0.8, label=p)
            for p in pod_order
        ]
        ax.legend(handles=pod_handles, title="POD position",
                  title_fontsize=8, fontsize=8, frameon=True,
                  loc="lower right")

    else:
        width = 0.5
        for xi, scen in enumerate(scenarios):
            scen_data = sdi_df[sdi_df[scenario_col] == scen][y_col].dropna()
            if scen_data.empty:
                continue
            color = _scen_colors.get(scen, "#888888")
            ax.boxplot(
                scen_data,
                positions=[xi], widths=width,
                patch_artist=True, notch=False, vert=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.75,
                              edgecolor=color, linewidth=0.8),
                medianprops=dict(color="white", linewidth=1.8),
                whiskerprops=dict(color=color, linewidth=0.9, linestyle="-"),
                capprops=dict(color=color, linewidth=0.9),
            )

    ax.axhline(0, color="0.3", lw=0.8, ls="--", zorder=2)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(
        [_scen_labels.get(s, s) for s in scenarios], fontsize=9
    )
    ax.set_xlim(-0.6, len(scenarios) - 0.4)
    ax.set_ylabel("Change in SDI (absolute)", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:+.0f}")
    )
    ax.grid(axis="y", lw=0.35, color="0.92", zorder=0)

    for spine in ax.spines.values():
        spine.set_edgecolor("0.88")
        spine.set_linewidth(0.6)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.12)

    _save(fig, save_path, dpi)
    return fig, ax
