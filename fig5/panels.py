"""Figure 5 — panel plotting functions.

Reusable plotting functions for every panel in Figure 5. Each function
accepts pre-created Axes and pre-loaded data.

Panel groups:
    A     Raster + instantaneous firing rate heatmap (one per condition)
    B–C   FR correlation and STTC matrices (D0, D50, difference)
    D     FR vs STTC scatter with marginal histograms
    E     Pairwise FR correlation violin distributions
    F     Normalized pairwise FR correlation percentile bands
    G     MEA spatial network visualization
    H–M   Graph metrics (strength, clustering, path length, betweenness,
          modularity, rich-club)

SpikeLab API used:
    - ``plot_recording()`` for raster + FR heatmap
    - ``plot_heatmap()`` for correlation matrices
    - ``plot_scatter_with_marginals()`` for FR vs STTC scatter
    - ``plot_distribution()`` for violin panels
    - ``plot_percentile_bands()`` for normalized band panels
    - ``plot_spatial_network()`` for MEA network
    - ``PairwiseCompMatrix.to_networkx()`` for graph construction
"""

import os
import sys

import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENT_DIR, "..")))

from plot_config import COLORS, FONT_SIZES, LINE_WIDTHS, CMAPS, CONDITIONS

from spikelab.spikedata.plot_utils import (
    plot_recording,
    plot_heatmap,
    plot_distribution,
    plot_percentile_bands,
    plot_scatter_with_marginals,
    plot_spatial_network,
)


# ═══════════════════════════════════════════════════════════════════════════
# Panel A — Raster + FR heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_raster_fr_on_axes(
    ax_raster,
    ax_fr,
    sd,
    fr_rates,
    sort_indices,
    start_ms=0,
    time_window_ms=20000,
    fr_vlim=None,
):
    """Plot spike raster above FR heatmap onto pre-existing axes.

    Parameters
    ----------
    ax_raster, ax_fr : Axes
        Target axes for raster and firing rate heatmap.
    sd : SpikeData
    fr_rates : np.ndarray
        Instantaneous firing rates, shape (U, T).
    sort_indices : np.ndarray
        Unit display order.
    start_ms, time_window_ms : float
        Time window to display.
    fr_vlim : tuple[float, float] or None
        Color limits for the heatmap.
    """
    fig = ax_raster.figure

    # Hidden colorbar axes
    cax_r = fig.add_axes([1.1, 1.1, 0.001, 0.001])
    cax_f = fig.add_axes([1.1, 1.1, 0.001, 0.001])

    plot_recording(
        sd,
        show_raster=True,
        show_fr_rates=True,
        fr_rates=fr_rates,
        time_range=(start_ms, start_ms + time_window_ms),
        sort_indices=sort_indices,
        raster_style="eventplot",
        vmin_heatmap=fr_vlim[0] if fr_vlim else None,
        vmax_heatmap=fr_vlim[1] if fr_vlim else None,
        height_ratios=[2, 2],
        absolute_xticks=False,
        font_size=FONT_SIZES["axes_label"],
        axes=[(ax_raster, cax_r), (ax_fr, cax_f)],
        show=False,
    )

    for cax in [cax_r, cax_f]:
        cax.clear()
        cax.axis("off")
        cax.set_visible(False)

    # Apply hot colormap
    if ax_fr.images:
        ax_fr.images[0].set_cmap(CMAPS["heatmap"])

    # Raster line widths
    for coll in ax_raster.collections:
        coll.set_linewidth(LINE_WIDTHS["raster_marker"])

    # Y-axis: first and last unit
    n_units = sd.N
    for ax in [ax_raster, ax_fr]:
        ax.set_yticks([0, n_units - 1])
        ax.set_yticklabels(["1", str(n_units)])
        ax.spines["left"].set_bounds(0, n_units - 1)

    # Scale bar (2 s)
    for ax in [ax_raster, ax_fr]:
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.spines["bottom"].set_visible(False)

    xlim = ax_fr.get_xlim()
    ylim = ax_fr.get_ylim()
    bar_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
    bar_start = bar_end - 2000
    bar_y = ylim[0] - (ylim[1] - ylim[0]) * 0.05
    ax_fr.plot(
        [bar_start, bar_end], [bar_y, bar_y],
        color="black", linewidth=LINE_WIDTHS["mean_trace"],
        clip_on=False, solid_capstyle="butt",
    )
    ax_fr.text(
        (bar_start + bar_end) / 2, bar_y - (ylim[1] - ylim[0]) * 0.03,
        "2s", ha="center", va="top", fontsize=FONT_SIZES["axes_label"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Panels B–C — Correlation matrices
# ═══════════════════════════════════════════════════════════════════════════
def get_fr_corr_sort_indices(fr_corr_matrix):
    """Return unit indices sorted by mean pairwise FR correlation (high to low).

    Parameters
    ----------
    fr_corr_matrix : PairwiseCompMatrix

    Returns
    -------
    np.ndarray
    """
    mat = fr_corr_matrix.matrix.copy()
    np.fill_diagonal(mat, np.nan)
    return np.argsort(np.nanmean(mat, axis=1))[::-1]


def plot_corr_matrix(ax, matrix, sort_indices, vmin, vmax, cmap):
    """Plot a single reordered correlation matrix.

    Parameters
    ----------
    ax : Axes
    matrix : np.ndarray
        (N, N) correlation matrix.
    sort_indices : np.ndarray
    vmin, vmax : float
    cmap : str

    Returns
    -------
    AxesImage
    """
    ordered = matrix[np.ix_(sort_indices, sort_indices)]
    n = len(sort_indices)
    mid = (n - 1) / 2

    plot_heatmap(
        ordered, ax=ax,
        vmin=vmin, vmax=vmax, cmap=cmap,
        aspect="equal", origin="upper",
        xlabel="", ylabel="",
        xticks=([0, mid, n - 1], ["1", "Unit", str(n)]),
        yticks=([0, mid, n - 1], ["1", "Unit", str(n)]),
        show_colorbar=False,
    )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # Hide tick mark on middle (label) tick; rotate y label
    for tick in ax.xaxis.get_major_ticks():
        if tick.get_loc() == mid:
            tick.tick1line.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        if tick.get_loc() == mid:
            tick.tick1line.set_visible(False)
            tick.label1.set_rotation(90)
            tick.label1.set_va("center")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    return ax.images[0]


# ═══════════════════════════════════════════════════════════════════════════
# Panel D — FR vs STTC scatter with marginals
# ═══════════════════════════════════════════════════════════════════════════
def _extract_pairs(mat_a, mat_b):
    """Extract upper-triangle values from two matrices, removing NaNs."""
    mask = np.triu(np.ones(mat_a.shape, dtype=bool), k=1)
    x = mat_a[mask]
    y = mat_b[mask]
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def make_scatter_with_marginals(
    gs_slot, fig, mat_a, mat_b, xlabel, ylabel, show_zero_lines=False,
):
    """Create FR-vs-STTC scatter with marginal histograms in a GridSpec slot.

    Parameters
    ----------
    gs_slot : SubplotSpec
        GridSpec slot to place the scatter + marginals.
    fig : Figure
    mat_a, mat_b : np.ndarray
        (N, N) matrices whose upper triangles are scattered.
    xlabel, ylabel : str
    show_zero_lines : bool
    """
    x, y = _extract_pairs(mat_a, mat_b)

    ax_scatter, ax_histx, ax_histy, sc = plot_scatter_with_marginals(
        gs_slot, fig, x, y,
        xlabel=xlabel, ylabel=ylabel,
        color_vals="density", cmap="viridis",
        marker_size=0.5, alpha=1.0,
        show_identity=True, show_colorbar=False,
        show_zero_lines=show_zero_lines,
    )

    # Red dotted identity line — draw above scatter points
    for line in ax_scatter.get_lines():
        if line.get_linestyle() == "--":
            line.set_linestyle(":")
            line.set_color("red")
            line.set_linewidth(1.5)
            line.set_zorder(10)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    margin = (hi - lo) * 0.03
    ax_scatter.set_xlim(lo - margin, hi + margin)
    ax_scatter.set_ylim(lo - margin, hi + margin)


# ═══════════════════════════════════════════════════════════════════════════
# Panels E–F — FR correlation violins + normalized bands
# ═══════════════════════════════════════════════════════════════════════════
def plot_pairwise_corr_violins(ax, corr_data, recordings=None, ylabel="FR correlation"):
    """Violin distributions of pairwise correlations (upper triangle).

    Parameters
    ----------
    ax : Axes
    corr_data : dict[str, np.ndarray]
        Condition -> (N, N) correlation matrix.
    recordings : list[str] or None
    ylabel : str
    """
    if recordings is None:
        recordings = CONDITIONS

    triu_data = {}
    for rec in recordings:
        mat = corr_data[rec]
        triu = mat[np.triu_indices_from(mat, k=1)]
        triu_data[rec] = triu[np.isfinite(triu)]

    colors = [COLORS[rec] for rec in recordings]
    labels = [r.replace("D", "") for r in recordings]
    return plot_distribution(
        ax, triu_data, labels=labels, colors=colors,
        ylabel=ylabel, xlabel="Diazepam (µM)",
        show_median=True, show_quartiles=True,
    )


def plot_pairwise_corr_normalized(
    ax, corr_data, recordings=None, ylabel="Norm. FR correlation", show_legend=False,
):
    """Percentile bands of per-pair normalized correlations.

    Parameters
    ----------
    ax : Axes
    corr_data : dict[str, np.ndarray]
        Condition -> (N, N) correlation matrix.
    recordings : list[str] or None
    ylabel : str
    show_legend : bool
    """
    if recordings is None:
        recordings = CONDITIONS

    n = corr_data[recordings[0]].shape[0]
    triu_idx = np.triu_indices(n, k=1)
    pairs = {rec: corr_data[rec][triu_idx] for rec in recordings}
    labels = [r.replace("D", "") for r in recordings]

    plot_percentile_bands(
        ax, pairs, labels=labels,
        normalize=True, style="bands",
        ylabel=ylabel, xlabel="Diazepam (µM)",
        ylim_range=0.6, show_legend=show_legend,
    )
    if show_legend:
        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, labels=leg_labels,
            loc="upper left", fontsize=FONT_SIZES["legend"],
            ncol=2, columnspacing=1.0, handlelength=1.5,
            bbox_to_anchor=(0, 1.02),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Panels H–M — Graph metrics
# ═══════════════════════════════════════════════════════════════════════════
def weighted_rich_club(G, n_bins=20):
    """Compute weighted rich-club coefficient vs strength percentile.

    Parameters
    ----------
    G : networkx.Graph
        Weighted graph.
    n_bins : int
        Number of strength-percentile thresholds.

    Returns
    -------
    percentiles : np.ndarray
    phi_w : np.ndarray
    """
    import networkx as nx

    strengths = dict(G.degree(weight="weight"))
    all_weights = sorted(
        [d["weight"] for _, _, d in G.edges(data=True)], reverse=True,
    )
    all_weights = np.array(all_weights)
    s_values = np.array(list(strengths.values()))
    percentiles = np.linspace(10, 90, n_bins)
    thresholds = np.percentile(s_values, percentiles)

    phi_w = []
    for s_thr in thresholds:
        rich_nodes = [n for n, s in strengths.items() if s > s_thr]
        if len(rich_nodes) < 2:
            phi_w.append(np.nan)
            continue
        rich_set = set(rich_nodes)
        w_rich = sum(
            d["weight"] for u, v, d in G.edges(data=True)
            if u in rich_set and v in rich_set
        )
        n_rich_edges = sum(
            1 for u, v in G.edges() if u in rich_set and v in rich_set
        )
        if n_rich_edges == 0:
            phi_w.append(np.nan)
            continue
        denom = all_weights[:n_rich_edges].sum()
        phi_w.append(w_rich / denom if denom > 0 else np.nan)

    return percentiles, np.array(phi_w)
