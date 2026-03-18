"""
Plotting functions to accompany TEALOM class outputs
----------
plot_fl_kde(fl_cat, scenarios, percentiles, shade_pct, figsize, save_path, dpi)
    KDE of FL distribution: baseline (dashed) vs. treated (solid),
    one subplot per scenario, lines colored by fire weather percentile.

plot_fl_heatmap(fl_cat, scenarios, percentiles, figsize, save_path, dpi)
    Diverging heatmap of mean delta_pct_cover (FL class x percentile),
    one subplot per scenario + shared colorbar.

plot_fl_stackedbar(fl_cat, scenarios, percentile, figsize, save_path, dpi)
    Paired horizontal stacked bars (baseline faded / treated solid)
    per scenario for a single fire weather percentile.

All functions
-------------
- Accept optional `scenarios` and `percentiles` lists to subset data.
- Return (fig, axes) for downstream customization.
- Accept `save_path` to write PNG/PDF at `dpi` resolution.

Example
-------
    from fl_treatment_plots import plot_fl_kde, plot_fl_heatmap, plot_fl_stackedbar

    fig, axes = plot_fl_kde(fl_cat, scenarios=["hand", "mech", "mechRxFire"])
    fig, axes = plot_fl_heatmap(fl_cat)
    fig, ax   = plot_fl_stackedbar(fl_cat, percentile="Pct97")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Module-level constants  (override after import if needed)
# ---------------------------------------------------------------------------

FL_CLASSES    = ["0-2ft", "2-4ft", "4-8ft", "8-12ft", ">12ft"]
BIN_MIDPOINTS = {"0-2ft": 1, "2-4ft": 3, "4-8ft": 6, "8-12ft": 10, ">12ft": 15}
PERCENTILES   = ["Pct25", "Pct50", "Pct90", "Pct97"]
PCT_LABELS    = {"Pct25": "25th", "Pct50": "50th", "Pct90": "90th", "Pct97": "97th"}

PCT_COLORS = {
    "Pct25": "#90CAF9",
    "Pct50": "#1565C0",
    "Pct90": "#E53935",
    "Pct97": "#6A1B9A",
}

FL_CLASS_COLORS = {
    "0-2ft":  "#1B5E20",
    "2-4ft":  "#66BB6A",
    "4-8ft":  "#FFF176",
    "8-12ft": "#EF6C00",
    ">12ft":  "#B71C1C",
}

# Scenario display names — edit to match your project
SCENARIO_LABELS = {
    "hand":        "Hand Thin",
    "handRxFire":  "Hand Thin +\nRx Fire",
    "mech":        "Mech Thin",
    "mechRxFire":  "Mech Thin +\nRx Fire",
    "rxfire":      "Rx Fire Only",
    "mast":        "Mastication",
}

# Diverging colormap: dark red → white → dark green
DIVERG_CMAP = LinearSegmentedColormap.from_list(
    "fl_diverge",
    ["#8B0000", "#D32F2F", "#FFFFFF", "#388E3C", "#1B5E20"],
    N=256,
)

_FONT_TITLE = {"fontsize": 10, "fontweight": "bold", "color": "#1A1A1A"}
_FONT_LABEL = {"fontsize": 8.5, "color": "#333333"}
_FONT_TICK  = {"labelsize": 7.5}
_RNG        = np.random.default_rng(0)  # fixed seed for reproducible KDE jitter


# ===========================================================================
# Public plotting functions
# ===========================================================================

# --- Module-level constants (add alongside your FL equivalents)
CS_CLASSES = ['NB', 'Surface', 'Passive', 'Active']
CS_CLASS_COLORS = {
    'NB':      '#d4d4d4',
    'Surface': '#91b563',
    'Passive': '#e0883a',
    'Active':  '#bf3b2f',
}


def plot_cs_stackedbar(
    cs_df: pd.DataFrame,
    scenarios: list[str] | None = None,
    percentile: str = 'Pct90',
    type_col: str = 'scenario',
    cs_labels: list[str] | None = None,
    cs_colors: dict | None = None,
    figsize: tuple | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Paired horizontal stacked bars: baseline (faded) vs. treated (solid)
    for a single fire weather percentile, one pair per scenario.
    Baseline is scenario-specific (i.e., aggregated over polygons belonging
    to that scenario), so each comparison reflects where that treatment occurred.

    Parameters
    ----------
    cs_df      : output of TEALOM.crownstate_change()
    scenarios  : scenarios to plot/order; None -> all, sorted by Surface increase
    percentile : single percentile string to display
    type_col   : treatment scenario column name
    cs_labels  : ordered crown state class labels
    cs_colors  : optional color overrides
    figsize    : override default dimensions
    save_path  : output path (.png / .pdf)
    dpi        : resolution for saved figure
    """
    _cs_labels = cs_labels or CS_CLASSES
    _cs_colors = {**CS_CLASS_COLORS, **(cs_colors or {})}

    # --- Pixel-count aggregation, scenario-specific baseline
    # drop_duplicates ensures each polygon counted once per (scenario, percentile, class)
    bl_counts = (cs_df
        .drop_duplicates(subset=[type_col, 'TRT_ID', 'percentile', 'CS_class'])
        .groupby([type_col, 'percentile', 'CS_class'])['CS_bl_count']
        .sum().reset_index(name='bl_px'))
    bl_totals = (bl_counts
        .groupby([type_col, 'percentile'])['bl_px']
        .sum().reset_index(name='bl_total'))
    bl_agg = (bl_counts
        .merge(bl_totals, on=[type_col, 'percentile'])
        .assign(bl=lambda d: d['bl_px'] / d['bl_total'] * 100)
        [[type_col, 'percentile', 'CS_class', 'bl']])

    tr_counts = (cs_df
        .groupby([type_col, 'percentile', 'CS_class'])['CS_tr_count']
        .sum().reset_index(name='tr_px'))
    tr_totals = (tr_counts
        .groupby([type_col, 'percentile'])['tr_px']
        .sum().reset_index(name='tr_total'))
    tr_agg = (tr_counts
        .merge(tr_totals, on=[type_col, 'percentile'])
        .assign(tr=lambda d: d['tr_px'] / d['tr_total'] * 100)
        [[type_col, 'percentile', 'CS_class', 'tr']])

    # --- Resolve and sort scenarios
    pct_data = (bl_agg.merge(tr_agg, on=[type_col, 'percentile', 'CS_class'])
                .query('percentile == @percentile'))

    if scenarios is None:
        surface = pct_data[pct_data['CS_class'] == 'Surface'].copy()
        surface['delta'] = surface['tr'] - surface['bl']
        scenarios = surface.sort_values('delta', ascending=True)[type_col].tolist()
        # ascending=True so most-improved (highest Surface increase) plots at top

    n     = len(scenarios)
    bar_h = 0.32
    gap   = 0.8
    y_bl  = np.arange(n) * gap + bar_h / 2
    y_tr  = np.arange(n) * gap - bar_h / 2

    fig, ax = plt.subplots(
        figsize=figsize or (9, max(3.5, n * 0.9)),
        facecolor='white',
    )

    # index for fast lookup
    bl_idx = pct_data.set_index([type_col, 'CS_class'])['bl']
    tr_idx = pct_data.set_index([type_col, 'CS_class'])['tr']

    for j, cls in enumerate(_cs_labels):
        color = _cs_colors.get(cls, '#aaa')

        # cumulative left offsets from previously drawn classes
        bl_lefts = np.zeros(n)
        tr_lefts = np.zeros(n)
        for prev_cls in _cs_labels[:j]:
            for k, s in enumerate(scenarios):
                bl_lefts[k] += bl_idx.get((s, prev_cls), 0.0)
                tr_lefts[k] += tr_idx.get((s, prev_cls), 0.0)

        bl_vals = np.array([bl_idx.get((s, cls), 0.0) for s in scenarios])
        tr_vals = np.array([tr_idx.get((s, cls), 0.0) for s in scenarios])

        ax.barh(y_bl, bl_vals, left=bl_lefts, height=bar_h,
                color=color, alpha=0.60, edgecolor='white', linewidth=0.5)
        ax.barh(y_tr, tr_vals, left=tr_lefts, height=bar_h,
                color=color, alpha=1.0, edgecolor='white', linewidth=0.5)

    # --- y-ticks centered between each bl/tr pair
    ax.set_yticks(np.arange(n) * gap)
    ax.set_yticklabels(
        [SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=8.5
    )

    ax.set_xlim(0, 105)
    ax.set_xlabel('% Area by Fire Type', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.xaxis.grid(True, linestyle='--', lw=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # --- Baseline/Treated margin annotations (top pair only, like FL plot)
    ax.text(101, y_bl[-1], 'Baseline', fontsize=7.5, color='#888888', va='center', ha='left')
    ax.text(101, y_tr[-1], 'Treated',  fontsize=7.5, color='#1A1A1A', va='center', ha='left')

    # --- Legend
    patches = [
        mpatches.Patch(facecolor=_cs_colors[c], edgecolor='white', label=c)
        for c in _cs_labels
    ]
    ax.legend(handles=patches, title='Fire Type', title_fontsize=9,
              fontsize=8.5, frameon=False,
              loc='upper center', bbox_to_anchor=(0.48, 1.12),
              ncol=len(_cs_labels))
    plt.subplots_adjust(bottom=0.1)
    plt.show()

    # pct_label = PCT_LABELS.get(percentile, percentile)
    # fig.suptitle(
    #     f'Crown State Distribution — {pct_label} Percentile Fire Weather',
    #     fontsize=11, fontweight='bold', color='#1A1A1A',
    # )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax

def plot_cs_stackedbar_multipct(
    cs_df: pd.DataFrame,
    percentiles: list[str] = ['Pct25', 'Pct97'],
    percentile_labels: dict | None = None,
    scenarios: list[str] | None = None,
    type_col: str = 'scenario',
    cs_labels: list[str] | None = None,
    cs_colors: dict | None = None,
    figsize: tuple | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side paired stacked bars for multiple fire weather percentiles.
    Scenario order is fixed across panels (sorted by Surface increase at
    the first percentile) so rows are directly comparable.
    """
    _cs_labels  = cs_labels or CS_CLASSES
    _cs_colors  = {**CS_CLASS_COLORS, **(cs_colors or {})}
    _pct_labels = percentile_labels or {
        'Pct25': '25th percentile',
        'Pct50': '50th percentile',
        'Pct90': '90th percentile',
        'Pct97': '97th percentile',
    }

    # ── Pixel-count aggregation (all percentiles at once) ─────────────────────

    bl_counts = (
        cs_df
        .drop_duplicates(subset=[type_col, 'TRT_ID', 'percentile', 'CS_class'])
        .groupby([type_col, 'percentile', 'CS_class'])['CS_bl_count']
        .sum().reset_index(name='bl_px')
    )
    bl_totals = (
        bl_counts
        .groupby([type_col, 'percentile'])['bl_px']
        .sum().reset_index(name='bl_total')
    )
    bl_agg = (
        bl_counts
        .merge(bl_totals, on=[type_col, 'percentile'])
        .assign(bl=lambda d: d['bl_px'] / d['bl_total'] * 100)
        [[type_col, 'percentile', 'CS_class', 'bl']]
    )

    tr_counts = (
        cs_df
        .groupby([type_col, 'percentile', 'CS_class'])['CS_tr_count']
        .sum().reset_index(name='tr_px')
    )
    tr_totals = (
        tr_counts
        .groupby([type_col, 'percentile'])['tr_px']
        .sum().reset_index(name='tr_total')
    )
    tr_agg = (
        tr_counts
        .merge(tr_totals, on=[type_col, 'percentile'])
        .assign(tr=lambda d: d['tr_px'] / d['tr_total'] * 100)
        [[type_col, 'percentile', 'CS_class', 'tr']]
    )

    combined = bl_agg.merge(tr_agg, on=[type_col, 'percentile', 'CS_class'])

    # ── Fix scenario order using first percentile's Surface delta ─────────────

    if scenarios is None:
        first_pct = percentiles[0]
        surface_delta = (
            combined
            .query('percentile == @first_pct and CS_class == "Surface"')
            .assign(delta=lambda d: d['tr'] - d['bl'])
            .sort_values('delta', ascending=True)  # most improved at top
        )
        scenarios = surface_delta[type_col].tolist()

    n     = len(scenarios)
    bar_h = 0.35
    gap = 0.82
    y_bl  = np.arange(n) * gap + bar_h / 2   # baseline: upper slot
    y_tr  = np.arange(n) * gap - bar_h / 2   # treated:  lower slot

    # ── Figure ────────────────────────────────────────────────────────────────

    n_panels = len(percentiles)
    fig, axes = plt.subplots(
        n_panels, 1,  # rows, cols flipped
        figsize=figsize or (4.5, max(3.0, n * 0.82) * n_panels),
        sharex=True,  # share x instead of y
        facecolor='white',
    )
    if n_panels == 1:
        axes = [axes]

    for ax, pct in zip(axes, percentiles):
        pct_data = combined.query('percentile == @pct')
        bl_idx   = pct_data.set_index([type_col, 'CS_class'])['bl']
        tr_idx   = pct_data.set_index([type_col, 'CS_class'])['tr']

        for j, cls in enumerate(_cs_labels):
            color = _cs_colors.get(cls, '#aaa')

            bl_lefts = np.zeros(n)
            tr_lefts = np.zeros(n)
            for prev_cls in _cs_labels[:j]:
                for k, s in enumerate(scenarios):
                    bl_lefts[k] += bl_idx.get((s, prev_cls), 0.0)
                    tr_lefts[k] += tr_idx.get((s, prev_cls), 0.0)

            bl_vals = np.array([bl_idx.get((s, cls), 0.0) for s in scenarios])
            tr_vals = np.array([tr_idx.get((s, cls), 0.0) for s in scenarios])

            ax.barh(y_bl, bl_vals, left=bl_lefts, height=bar_h,
                    color=color, alpha=0.55, edgecolor='white', linewidth=0.5)
            ax.barh(y_tr, tr_vals, left=tr_lefts, height=bar_h,
                    color=color, alpha=1.0, edgecolor='white', linewidth=0.5)

            for k, s in enumerate(scenarios):
                bl_surface = bl_idx.get((s, 'Surface'), 0.0)
                tr_surface = tr_idx.get((s, 'Surface'), 0.0)
                delta = tr_surface - bl_surface

                # x position: NB left offset + half of surface segment width
                nb_offset = tr_idx.get((s, 'NB'), 0.0)
                bar_center = nb_offset + tr_surface / 2

                # Only annotate if surface segment is wide enough to hold text (~8% minimum)
                if tr_surface > 8:
                    bar_right = nb_offset + tr_surface
                    pad = 1.0  # units inside the right edge — nudge as needed

                    ax.text(
                        bar_right - pad, y_tr[k],
                        f"{delta:+.0f}%",
                        ha='right', va='center',
                        fontsize=8, fontweight='normal',
                        color='0.1',
                    )

        # Panel title
        ax.set_title(
            _pct_labels.get(pct, pct),
            fontsize=9.5, loc='center', pad=8,
            color='0.2', fontweight='normal'
        )

        ax.set_xlim(0, 108)
        ax.set_xlabel('% Area by fire type', fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
        ax.xaxis.grid(True, linestyle='--', lw=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        # Baseline / Treated labels on top panel only
        if ax is axes[0]:
            ax.text(104, y_bl[-1], 'Baseline', fontsize=7.5,
                    color='0.6', va='center', ha='left')
            ax.text(104, y_tr[-1], 'Treated', fontsize=7.5,
                    color='0.15', va='center', ha='left')

        # ── y-tick labels on left panel only (sharey=True handles the rest) ───────

        # x-axis label on bottom panel only
        if ax is axes[-1]:
            ax.set_xlabel('% Area by fire type', fontsize=9)
        else:
            ax.set_xlabel('')

        # y-tick labels on BOTH panels now (no longer sharing y-axis)
        ax.set_yticks(np.arange(n) * gap)
        ax.set_yticklabels(
            [SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=8.5
        )

    # ── Shared legend above both panels ───────────────────────────────────────

    fig.subplots_adjust(
        hspace=0.22,
        top=0.88,
        bottom=0.10,
        left=0.18,
        right=0.94,
    )

    patches = [
        mpatches.Patch(facecolor=_cs_colors[c], edgecolor='white', label=c)
        for c in _cs_labels
    ]

    fig.legend(
        handles=patches,
        title='Fire type', title_fontsize=8.5,
        fontsize=8.5, frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),  # was 1.04 — pull it closer
        ncol=len(_cs_labels),
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return fig, axes



# --- Suppression Difficulty Index (SDI)

def plot_sdi_boxplot(
    sdi_df: pd.DataFrame,
    type_col: str = 'TRT_LABEL',
    pod_col: str = 'PODLine',
    y_col: str = 'delta_SDI',
    scenarios: list[str] | None = None,
    scenario_colors: dict | None = None,
    pod_colors: dict | None = None,
    figsize: tuple | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grouped boxplot of absolute delta SDI by treatment type and POD position.
    Treatment type on x-axis, POD position (Interior/Line) as hue.
    """
    _scen_colors = scenario_colors or cmap_trts
    _pod_colors  = pod_colors or {
        'Interior': '#4A7FB5',   # muted blue
        'Line':     '#E8955A',   # muted orange
    }

    # ── Resolve and order scenarios by median delta SDI ───────────────────────

    if scenarios is None:
        scenarios = (
            sdi_df.groupby(type_col)[y_col]
            .median()
            .sort_values()          # most negative first (left on x)
            .index.tolist()
        )

    pod_order = ['Interior', 'Line']

    # ── Figure ────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(
        figsize=figsize or (6.0, 4.0),
        facecolor='white'
    )

    n_scen = len(scenarios)
    n_pod  = len(pod_order)
    width  = 0.32
    gap    = 0.06

    # Center group, offset each POD box within
    total_w   = n_pod * width + (n_pod - 1) * gap
    pod_offsets = np.linspace(-total_w / 2 + width / 2,
                               total_w / 2 - width / 2,
                               n_pod)

    for xi, scen in enumerate(scenarios):
        scen_data = sdi_df[sdi_df[type_col] == scen]

        for pi, pod in enumerate(pod_order):
            pod_data = scen_data[scen_data[pod_col] == pod][y_col].dropna()
            if pod_data.empty:
                continue

            x      = xi + pod_offsets[pi]
            color  = _pod_colors[pod]
            tcolor = _scen_colors.get(scen, '#888888')

            bp = ax.boxplot(
                pod_data,
                positions=[x],
                widths=width,
                patch_artist=True,
                notch=False,
                vert=True,
                showfliers=False,   # suppress outliers — noted in caption
                boxprops=dict(
                    facecolor=color, alpha=0.75,
                    edgecolor=color, linewidth=0.8
                ),
                medianprops=dict(color='white', linewidth=1.8),
                whiskerprops=dict(color=color, linewidth=0.9, linestyle='-'),
                capprops=dict(color=color, linewidth=0.9),
            )


    # ── Reference line ────────────────────────────────────────────────────────

    ax.axhline(0, color='0.3', lw=0.8, ls='--', zorder=2)

    # ── Axes ──────────────────────────────────────────────────────────────────

    ax.set_xticks(range(n_scen))
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_xlim(-0.6, n_scen - 0.4)
    ax.set_ylabel('Change in SDI (absolute)', fontsize=9)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:+.0f}'
    ))
    ax.grid(axis='y', lw=0.35, color='0.92', zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('0.88')
        spine.set_linewidth(0.6)

    # ── Legend ────────────────────────────────────────────────────────────────

    pod_handles = [
        mpatches.Patch(
            facecolor=_pod_colors[p], alpha=0.75,
            edgecolor=_pod_colors[p], linewidth=0.8,
            label=p
        )
        for p in pod_order
    ]
    ax.legend(
        handles=pod_handles,
        title='POD position', title_fontsize=8,
        fontsize=8, frameon=True,
        loc='lower right',
    )

    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.12)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return fig, ax


# --- Flame length

def plot_fl_delta_bar(
    fl_df: pd.DataFrame,
    percentiles: list[str] = ['Pct25', 'Pct97'],
    percentile_labels: dict | None = None,
    scenarios: list[str] | None = None,
    type_col: str = 'TRT_LABEL',
    fl_class_col: str = 'FL_class',
    fl_class_order: list[str] | None = None,
    scenario_colors: dict | None = None,
    figsize: tuple | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Faceted grouped horizontal delta bar chart.
    Shows change in % area (treated - baseline) per FL bin and scenario.
    One panel per percentile, FL bins on y-axis, scenarios grouped within
    each bin, colored by treatment type.
    """
    _pct_labels = percentile_labels or {
        'Pct25': '25th percentile',
        'Pct50': '50th percentile',
        'Pct90': '90th percentile',
        'Pct97': '97th percentile',
    }
    _fl_order      = fl_class_order or ['0-2ft', '2-4ft', '4-8ft', '8-12ft', '>12ft']
    _scen_colors   = scenario_colors or cmap_trts

    # ── Pixel-count weighted aggregation ─────────────────────────────────────

    bl_counts = (
        fl_df
        .drop_duplicates(subset=[type_col, 'TRT_ID', 'percentile', fl_class_col])
        .groupby([type_col, 'percentile', fl_class_col])['FL_bl_count']
        .sum().reset_index(name='bl_px')
    )
    bl_totals = (
        bl_counts.groupby([type_col, 'percentile'])['bl_px']
        .sum().reset_index(name='bl_total')
    )
    bl_agg = (
        bl_counts.merge(bl_totals, on=[type_col, 'percentile'])
        .assign(bl_pct=lambda d: d['bl_px'] / d['bl_total'] * 100)
    )

    tr_counts = (
        fl_df
        .groupby([type_col, 'percentile', fl_class_col])['FL_tr_count']
        .sum().reset_index(name='tr_px')
    )
    tr_totals = (
        tr_counts.groupby([type_col, 'percentile'])['tr_px']
        .sum().reset_index(name='tr_total')
    )
    tr_agg = (
        tr_counts.merge(tr_totals, on=[type_col, 'percentile'])
        .assign(tr_pct=lambda d: d['tr_px'] / d['tr_total'] * 100)
    )

    combined = (
        bl_agg.merge(tr_agg, on=[type_col, 'percentile', fl_class_col])
        .assign(delta=lambda d: d['tr_pct'] - d['bl_pct'])
    )

    # ── Resolve scenario order by delta in >12ft bin at Pct97 ────────────────

    if scenarios is None:
        scenarios = (
            combined
            .query(f'percentile == "Pct97" and {fl_class_col} == ">12ft"')
            .groupby(type_col, as_index=False)
            .agg(delta=('delta', 'mean'))
            .sort_values('delta')           # most negative = greatest reduction first
            [type_col]
            .tolist()
        )

    n_scen = len(scenarios)
    n_bins = len(_fl_order)

    # ── Y-position layout ─────────────────────────────────────────────────────

    band_h  = 1.0
    bar_h   = (band_h * 0.72) / n_scen
    gap     = bar_h * 0.25

    # Bin centers — 0-2ft at bottom
    bin_centers = {cls: i * band_h for i, cls in enumerate(_fl_order)}

    # Scenario offsets within each bin band — centered
    total_bar_span = n_scen * bar_h + (n_scen - 1) * gap
    start_offset   = -total_bar_span / 2 + bar_h / 2
    scen_offsets   = {
        s: start_offset + i * (bar_h + gap)
        for i, s in enumerate(scenarios)
    }

    # ── Figure ────────────────────────────────────────────────────────────────

    n_panels = len(percentiles)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=figsize or (5.2 * n_panels, max(4.0, n_bins * 0.9)),
        sharey=True,
        facecolor='white',
    )
    axes = np.atleast_1d(axes)

    for ax, pct in zip(axes, percentiles):
        pct_data = combined[combined['percentile'] == pct]

        for si, scen in enumerate(scenarios):
            color     = _scen_colors.get(scen, '#888888')
            scen_data = pct_data[pct_data[type_col] == scen]
            idx       = scen_data.set_index(fl_class_col)['delta']

            for cls in _fl_order:
                if cls not in idx.index:
                    continue
                y     = bin_centers[cls] + scen_offsets[scen]
                delta = idx[cls]
                color_alpha = 0.9

                ax.barh(
                    y, delta,
                    height=bar_h,
                    color=color,
                    alpha=color_alpha,
                    edgecolor='white',
                    linewidth=0.4,
                    zorder=2,
                )

        # Zero reference line
        ax.axvline(0, color='0.3', lw=0.8, zorder=3)

        # Alternating bin band shading
        for i, cls in enumerate(_fl_order):
            if i % 2 == 0:
                ax.axhspan(
                    bin_centers[cls] - band_h / 2,
                    bin_centers[cls] + band_h / 2,
                    color='0.96', zorder=0
                )

        ax.set_title(
            _pct_labels.get(pct, pct),
            fontsize=9.5, loc='center', pad=6, color='0.2'
        )
        ax.set_xlabel('Change in % area (treated − baseline)', fontsize=8.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:+.0f}%')
        )
        ax.grid(axis='x', lw=0.35, color='0.90', zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

    # ── y-axis labels on left panel only ─────────────────────────────────────

    axes[0].set_yticks(list(bin_centers.values()))
    axes[0].set_yticklabels(_fl_order, fontsize=9)
    axes[0].set_ylabel('Flame length class', fontsize=9)
    axes[0].spines['left'].set_visible(False)
    axes[0].tick_params(axis='y', length=0)

    # ── Shared legend ─────────────────────────────────────────────────────────

    leg_patches = [
        mpatches.Patch(
            facecolor=_scen_colors.get(s, '#888'),
            label=s, alpha=0.9
        )
        for s in scenarios
    ]
    fig.legend(
        handles=leg_patches,
        title='Treatment', title_fontsize=8.5,
        fontsize=8.5, frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=n_scen,
    )

    fig.subplots_adjust(
        wspace=0.08, top=0.88, bottom=0.12,
        left=0.14, right=0.97
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return fig, axes


def plot_fl_kde(
    fl_cat: pd.DataFrame,
    scenarios: list[str] | None = None,
    percentiles: list[str] | None = None,
    shade_pct: str | None = "Pct90",
    figsize: tuple | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, np.ndarray]:
    """
    KDE plot of FL distributions: baseline (dashed) vs. treated (solid).
    One subplot per scenario; lines colored by fire weather percentile.

    Parameters
    ----------
    fl_cat      : fl_change categorical DataFrame from TEALOM.fl_change()
    scenarios   : scenarios to plot; None -> all present (ordered by SCENARIO_LABELS)
    percentiles : percentiles to plot; None -> all four
    shade_pct   : percentile to shade bl->tr gap for visual emphasis; None to skip
    figsize     : override default figure dimensions
    save_path   : output path (.png / .pdf)
    dpi         : resolution for saved figure

    Returns
    -------
    fig, axes   : Figure and 1-D ndarray of Axes (one per scenario)
    """
    agg       = _aggregate_cat(fl_cat)
    scenarios = _resolve_scenarios(agg, scenarios)
    pcts      = percentiles or PERCENTILES
    n         = len(scenarios)

    fig, axes = plt.subplots(
        1, n,
        figsize=figsize or (max(10, n * 2.6), 3.8),
        sharey=False,
        facecolor="white",
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for i, (ax, scen) in enumerate(zip(axes, scenarios)):
        scen_data = agg[agg["scenario"] == scen]

        for pct in pcts:
            pct_data = scen_data[scen_data["percentile"] == pct]
            if pct_data.empty:
                continue
            col  = PCT_COLORS.get(pct, "#888888")
            bl_s = pct_data.set_index("FL_class")["FL_bl_cls_pct_cover"]
            tr_s = pct_data.set_index("FL_class")["FL_tr_cls_pct_cover"]

            x_bl, y_bl = _kde_from_counts(bl_s)
            x_tr, y_tr = _kde_from_counts(tr_s)

            if x_bl is not None:
                ax.plot(x_bl, y_bl, color=col, lw=1.3, ls="--", alpha=0.70)
            if x_tr is not None:
                ax.plot(x_tr, y_tr, color=col, lw=1.8, ls="-", alpha=0.95,
                        label=PCT_LABELS[pct])

            if pct == shade_pct and x_bl is not None and x_tr is not None:
                ax.fill_between(x_bl, y_bl, y_tr, alpha=0.09, color=col)

        for edge in [2, 4, 8, 12]:
            ax.axvline(edge, color="#CCCCCC", lw=0.6, ls=":")

        ax.set_title(SCENARIO_LABELS.get(scen, scen), **_FONT_TITLE)
        ax.set_xlim(0, 17)
        ax.set_ylim(bottom=0)
        ax.set_xticks([1, 3, 6, 10, 15])
        ax.set_xticklabels(["0-2", "2-4", "4-8", "8-12", ">12"], fontsize=7)
        ax.tick_params(**_FONT_TICK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Flame Length Class (ft)", **_FONT_LABEL)
        if i == 0:
            ax.set_ylabel("Density", **_FONT_LABEL)

    # Shared legend on first axis
    legend_lines = [
        plt.Line2D([0], [0], color=PCT_COLORS.get(p, "#888"), lw=1.6,
                   ls="-", label=f"{PCT_LABELS[p]}th — Treated")
        for p in pcts
    ] + [
        plt.Line2D([0], [0], color="#AAAAAA", lw=1.3, ls="--", label="Baseline")
    ]
    axes[0].legend(handles=legend_lines, fontsize=6.5, frameon=False,
                   loc="upper right")

    fig.suptitle("Flame Length Distribution: Baseline vs. Treated",
                 fontsize=11, fontweight="bold", color="#1A1A1A")

    _save(fig, save_path, dpi)
    return fig, axes


# ---------------------------------------------------------------------------

def plot_fl_stackedbar(
    fl_cat: pd.DataFrame,
    scenarios: list[str] | None = None,
    percentile: str = "Pct90",
    figsize: tuple | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Paired horizontal stacked bars: baseline (faded) vs. treated (solid)
    for a single fire weather percentile, one pair of bars per scenario.

    Parameters
    ----------
    fl_cat      : fl_change categorical DataFrame
    scenarios   : scenarios to plot; None -> all present
    percentile  : single percentile to display (default "Pct90")
    figsize     : override default figure dimensions
    save_path   : output path (.png / .pdf)
    dpi         : resolution for saved figure

    Returns
    -------
    fig, ax     : Figure and single Axes
    """
    agg       = _aggregate_cat(fl_cat)
    scenarios = _resolve_scenarios(agg, scenarios)
    pct_data  = agg[agg["percentile"] == percentile]
    n         = len(scenarios)

    bar_h = 0.32
    gap   = 1.1
    y_bl  = np.arange(n) * gap + bar_h / 2
    y_tr  = np.arange(n) * gap - bar_h / 2

    fig, ax = plt.subplots(
        figsize=figsize or (9, max(3.5, n * 0.9)),
        facecolor="white",
        constrained_layout=True,
    )

    for j, fl_cls in enumerate(FL_CLASSES):
        col      = FL_CLASS_COLORS[fl_cls]
        cls_data = pct_data[pct_data["FL_class"] == fl_cls].set_index("scenario")

        # Cumulative left offsets from all previously drawn classes
        bl_lefts = np.zeros(n)
        tr_lefts = np.zeros(n)
        for prev_cls in FL_CLASSES[:j]:
            prev = pct_data[pct_data["FL_class"] == prev_cls].set_index("scenario")
            for k, s in enumerate(scenarios):
                bl_lefts[k] += prev.loc[s, "FL_bl_cls_pct_cover"] if s in prev.index else 0.0
                tr_lefts[k] += prev.loc[s, "FL_tr_cls_pct_cover"] if s in prev.index else 0.0

        bl_vals = np.array([
            cls_data.loc[s, "FL_bl_cls_pct_cover"] if s in cls_data.index else 0.0
            for s in scenarios
        ])
        tr_vals = np.array([
            cls_data.loc[s, "FL_tr_cls_pct_cover"] if s in cls_data.index else 0.0
            for s in scenarios
        ])

        ax.barh(y_bl, bl_vals, left=bl_lefts, height=bar_h,
                color=col, alpha=0.45, edgecolor="white", linewidth=0.5,
                label=fl_cls)
        ax.barh(y_tr, tr_vals, left=tr_lefts, height=bar_h,
                color=col, alpha=1.0, edgecolor="white", linewidth=0.5)

    ax.set_yticks(np.arange(n) * gap)
    ax.set_yticklabels(
        [SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=8.5
    )
    ax.set_xlim(0, 108)
    ax.set_xlabel("Mean % Cover by Flame Length Class", **_FONT_LABEL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(**_FONT_TICK)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    # Baseline / Treated margin annotations
    ax.text(101, y_bl[-1], "Baseline", fontsize=7, color="#888888",
            va="center", ha="left")
    ax.text(101, y_tr[-1], "Treated",  fontsize=7, color="#1A1A1A",
            va="center", ha="left")

    patches = [
        mpatches.Patch(facecolor=FL_CLASS_COLORS[c], label=c, edgecolor="white")
        for c in FL_CLASSES
    ]
    ax.legend(handles=patches, title="FL Class", title_fontsize=7,
              fontsize=7.5, frameon=False, loc="lower right",
              ncol=len(FL_CLASSES))

    pct_label = PCT_LABELS.get(percentile, percentile)
    fig.suptitle(
        f"Flame Length Class Distribution — {pct_label} Percentile Fire Weather",
        fontsize=11, fontweight="bold", color="#1A1A1A",
    )

    _save(fig, save_path, dpi)
    return fig, ax


# ===========================================================================
# Private helpers
# ===========================================================================

def _aggregate_cat(fl_cat: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bl/tr pct_cover and delta across TRT_IDs.

    Baseline is scenario-agnostic (same value replicated per scenario in fl_cat),
    so it must be averaged independently to avoid inflated means caused by
    unequal TRT_ID counts across scenarios. drop_duplicates ensures each
    [TRT_ID, percentile, FL_class] contributes exactly one baseline observation.
    """
    # Treated: mean per [scenario, percentile, FL_class]
    tr_agg = (
        fl_cat
        .groupby(["scenario", "percentile", "FL_class"], observed=True)
        [["FL_tr_cls_pct_cover"]]
        .mean()
        .reset_index()
    )

    # Baseline: mean per [percentile, FL_class] — drop scenario replication first
    bl_agg = (
        fl_cat
        .drop_duplicates(subset=["TRT_ID", "percentile", "FL_class"])
        .groupby(["percentile", "FL_class"], observed=True)
        [["FL_bl_cls_pct_cover"]]
        .mean()
        .reset_index()
    )

    agg = tr_agg.merge(bl_agg, on=["percentile", "FL_class"], how="left")
    agg["delta_pct_cover"] = agg["FL_tr_cls_pct_cover"] - agg["FL_bl_cls_pct_cover"]
    agg["FL_class"] = pd.Categorical(
        agg["FL_class"], categories=FL_CLASSES, ordered=True
    )
    return agg.sort_values(["scenario", "percentile", "FL_class"])


def _resolve_scenarios(
    agg: pd.DataFrame,
    scenarios: list[str] | None,
) -> list[str]:
    """Return display-ordered scenario list, optionally subsetted."""
    preferred = list(SCENARIO_LABELS.keys())
    present   = agg["scenario"].unique().tolist()
    ordered   = [s for s in preferred if s in present]
    ordered  += sorted(s for s in present if s not in ordered)
    if scenarios is not None:
        ordered = [s for s in ordered if s in scenarios]
    return ordered


def _kde_from_counts(
    pct_cover_series: pd.Series,
    fl_classes: list[str] = FL_CLASSES,
    bw: float = 1.5,
    n_expand: int = 1000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Reconstruct a KDE from categorical pct_cover values using bin midpoints.
    Adds small Gaussian jitter to prevent singular covariance when coverage
    is concentrated in a single bin (common post-treatment at low percentiles).

    Returns (x_grid, density) or (None, None) if data are insufficient.
    """
    midpoints = np.array([BIN_MIDPOINTS[c] for c in fl_classes])
    weights   = np.array(
        [pct_cover_series.get(c, 0.0) for c in fl_classes], dtype=float
    )
    weights = np.clip(weights, 0, None)
    if weights.sum() == 0:
        return None, None
    weights /= weights.sum()

    samples = np.repeat(midpoints, np.round(weights * n_expand).astype(int))
    if len(samples) < 3:
        return None, None

    # Jitter prevents LinAlgError from singular covariance (all-mass-in-one-bin)
    samples = samples + _RNG.normal(0, 0.2, size=len(samples))

    try:
        kde    = gaussian_kde(samples, bw_method=bw / samples.std(ddof=1))
        x_grid = np.linspace(0, 18, 300)
        return x_grid, kde(x_grid)
    except np.linalg.LinAlgError:
        return None, None


def _save(fig: plt.Figure, save_path: str | None, dpi: int) -> None:
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved → {save_path}")