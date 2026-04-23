"""Figure 6 — panel plotting functions.

Reusable plotting functions for every panel in Figure 6. Each function
accepts pre-created Axes and pre-loaded data.

Panel groups:
    A     Raster + FR heatmap + population rate with burst windows (one per condition)
    B     Single-unit burst-aligned raster + average rate heatmap
    C     5x5 grid of average burst-aligned firing rate heatmaps (group 0 units)
    D     PCA of burst u2u correlation matrices
    E     Burst-aligned average population rate with closest burst edge markers
    F     Slice-to-slice time correlation clipped to burst edges
    G     Burst-to-burst correlation heatmap
    H     Within-condition burst correlation violins + p-value inset
    I     Rank order correlation heatmap
    J     Within-condition rank order violins

SpikeLab API used:
    - ``plot_recording()`` for raster + FR heatmap + population rate
    - ``plot_heatmap()`` for burst-aligned heatmaps and correlation matrices
    - ``plot_scatter()`` for PCA scatter
    - ``plot_distribution()`` for violin panels
    - ``plot_pvalue_matrix()`` for p-value insets
    - ``SpikeSliceStack.plot_aligned_slice_single_unit()`` for unit burst rasters
    - ``SpikeData.plot_aligned_pop_rate()`` for burst-aligned population rate
"""

import os
import sys

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENT_DIR, "..")))

from plot_config import COLORS, FONT_SIZES, LINE_WIDTHS, CONDITIONS

from spikelab.spikedata.plot_utils import (
    plot_recording,
    plot_heatmap,
    plot_scatter,
    plot_distribution,
    plot_pvalue_matrix,
)
from spikelab.spikedata.spikeslicestack import SpikeSliceStack

PRE_MS = 250
POST_MS = 500


# ═══════════════════════════════════════════════════════════════════════════
# Panel A — Raster + FR heatmap + population rate with burst windows
# ═══════════════════════════════════════════════════════════════════════════
def plot_raster_burst_windows(
    ax_raster,
    ax_heatmap,
    ax_poprate,
    ax_cbar,
    sd,
    pop_rate,
    fr_rates,
    burst_peaks,
    sort_indices,
    color,
    pre_ms=250,
    post_ms=500,
    start_ms=0,
    time_window_ms=30000,
    poprate_ylim=None,
    heatmap_vmax=None,
):
    """Plot raster + FR heatmap + pop rate with burst windows onto pre-existing axes.

    Parameters
    ----------
    ax_raster, ax_heatmap, ax_poprate : matplotlib.axes.Axes
        Target axes for the three panels.
    ax_cbar : matplotlib.axes.Axes
        Target axes for the FR heatmap colorbar.
    sd : SpikeData
        Full recording.
    pop_rate : np.ndarray
        Pre-computed population rate (1 ms bins, full recording).
    fr_rates : np.ndarray
        Per-unit instantaneous firing rates, shape (U, T).
    burst_peaks : np.ndarray
        Burst peak times in ms, shape (B,).
    sort_indices : np.ndarray
        Unit ordering indices.
    color : str
        Color for pop rate trace and burst window shading.
    pre_ms, post_ms : float
        Window before/after burst peak in ms.
    start_ms : float
        Start time in ms.
    time_window_ms : float
        Duration to plot in ms.
    poprate_ylim : tuple or None
        Fixed (ymin, ymax) for pop rate subplot.
    heatmap_vmax : float or None
        Max value for FR heatmap colorscale.
    """
    from matplotlib.patches import Rectangle

    burst_edges = np.column_stack([burst_peaks - pre_ms, burst_peaks + post_ms])

    # Create hidden colorbar axes for raster and poprate panels
    fig = ax_raster.figure
    cax_r = fig.add_axes([0, 0, 0.001, 0.001])
    cax_r.axis("off")
    cax_p = fig.add_axes([0, 0, 0.001, 0.001])
    cax_p.axis("off")

    plot_recording(
        sd,
        show_raster=True,
        show_fr_rates=True,
        show_pop_rate=True,
        pop_rate=pop_rate,
        fr_rates=fr_rates,
        time_range=(start_ms, start_ms + time_window_ms),
        sort_indices=sort_indices,
        raster_style="eventplot",
        burst_times=burst_peaks,
        burst_edges=burst_edges,
        vmax_heatmap=heatmap_vmax,
        axes=[
            (ax_raster, cax_r),
            (ax_poprate, cax_p),
            (ax_heatmap, ax_cbar),
        ],
        absolute_xticks=False,
        font_size=FONT_SIZES["axes_label"],
        show=False,
    )

    # --- Restyle pop rate line ---
    for line in ax_poprate.get_lines():
        line.set_color(color)
        line.set_linewidth(LINE_WIDTHS["data_trace"])

    if poprate_ylim is not None:
        ax_poprate.set_ylim(poprate_ylim)

    # --- Restyle burst edge shading: clip to y=0, use condition color ---
    for patch in list(ax_poprate.patches):
        patch.remove()

    ylim = ax_poprate.get_ylim()
    ymax = ylim[1]
    start_val = start_ms
    end_val = start_ms + time_window_ms
    for t0, t1 in burst_edges:
        if t1 < start_val or t0 > end_val:
            continue
        x0 = max(t0, start_val) - start_val
        x1 = min(t1, end_val) - start_val
        rect = Rectangle((x0, 0), x1 - x0, ymax, facecolor=color, alpha=0.3)
        ax_poprate.add_patch(rect)

    # --- Vertical lines at burst window edges across all panels ---
    for t0, t1 in burst_edges:
        if t1 < start_val or t0 > end_val:
            continue
        x0 = max(t0, start_val) - start_val
        x1 = min(t1, end_val) - start_val
        for ax in [ax_raster, ax_heatmap, ax_poprate]:
            ax.axvline(x0, color="green", linewidth=1.0, linestyle=":", zorder=0)
            ax.axvline(x1, color="green", linewidth=1.0, linestyle=":", zorder=0)

    # --- Restyle burst peak markers ---
    for coll in ax_poprate.collections:
        coll.set_facecolor("black")
        coll.set_edgecolor("black")
        coll.set_sizes([15])

    # --- Restyle raster ---
    for coll in ax_raster.collections:
        coll.set_linewidth(LINE_WIDTHS["raster_marker"])

    n_units = sd.N
    ax_raster.set_yticks([0, n_units - 1])
    ax_raster.set_yticklabels(["1", str(n_units)])

    # --- Hide x-axis on raster and heatmap (shared with poprate) ---
    for ax in [ax_raster, ax_heatmap]:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")

    # --- Scale bar on poprate ---
    ax_poprate.set_xticks([])
    ax_poprate.set_xticklabels([])
    ax_poprate.set_xlabel("")
    ax_poprate.spines["bottom"].set_visible(False)
    ax_raster.spines["bottom"].set_visible(False)

    scale_bar_ms = 1000
    xlim = ax_poprate.get_xlim()
    ylim = ax_poprate.get_ylim()
    bar_x_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
    bar_x_start = bar_x_end - scale_bar_ms
    bar_y = ylim[0] - (ylim[1] - ylim[0]) * 0.05
    ax_poprate.plot(
        [bar_x_start, bar_x_end],
        [bar_y, bar_y],
        color="black",
        linewidth=LINE_WIDTHS["mean_trace"],
        clip_on=False,
        solid_capstyle="butt",
    )
    ax_poprate.text(
        (bar_x_start + bar_x_end) / 2,
        bar_y - (ylim[1] - ylim[0]) * 0.08,
        "1 s",
        ha="center",
        va="top",
        fontsize=FONT_SIZES["axes_label"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Panel B — Single-unit burst-aligned raster + average rate heatmap
# ═══════════════════════════════════════════════════════════════════════════
def build_combined_spike_slice_stack(ws_path, conditions, pre_ms, post_ms):
    """Build a combined SpikeSliceStack across all conditions.

    Parameters
    ----------
    ws_path : str
        Path to the workspace directory.
    conditions : list[str]
        Condition labels.
    pre_ms, post_ms : float
        Window bounds relative to burst peak.

    Returns
    -------
    combined : SpikeSliceStack
    condition_boundaries : list[int]
        Slice indices where each condition starts, plus final count.
    """
    from spikelab.workspace.hdf5_io import load_workspace_item

    all_slices = []
    all_times = []
    condition_boundaries = []

    for rec in conditions:
        sd = load_workspace_item(ws_path, rec, "spikedata")
        tburst = load_workspace_item(ws_path, rec, "tburst")

        sss = sd.align_to_events(tburst, pre_ms=pre_ms, post_ms=post_ms, kind="spike")
        condition_boundaries.append(len(all_slices))
        all_slices.extend(sss.spike_stack)
        all_times.extend(sss.times)

    condition_boundaries.append(len(all_slices))

    combined = SpikeSliceStack(spike_stack=all_slices, times_start_to_end=all_times)
    return combined, condition_boundaries


def compute_avg_rate_per_condition(ws_path, unit_idx, conditions, pre_ms, post_ms):
    """Compute average instantaneous firing rate per condition for one unit.

    Parameters
    ----------
    ws_path : str
        Path to the workspace directory.
    unit_idx : int
        Index of the unit.
    conditions : list[str]
        Condition labels.
    pre_ms, post_ms : float
        Window bounds relative to burst peak.

    Returns
    -------
    avg_rates : np.ndarray (n_conditions, T)
        Mean firing rate across bursts for the given unit.
    """
    from spikelab.workspace.hdf5_io import load_workspace_item

    avg_rates = []
    for rec in conditions:
        rss = load_workspace_item(ws_path, rec, "burst_rss")
        # rss.event_stack shape: (U, T, S)
        unit_rates = rss.event_stack[unit_idx, :, :]  # (T, S)
        avg = np.nanmean(unit_rates, axis=1)  # (T,)
        avg_rates.append(avg)
    return np.array(avg_rates)  # (n_conditions, T)


def plot_unit_burst_raster(
    ax, sss, unit_idx, condition_boundaries, conditions=None, pre_ms=250, post_ms=500
):
    """Plot a single unit's burst-aligned raster with condition dividers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    sss : SpikeSliceStack
        Combined stack across all conditions.
    unit_idx : int
        Index of the unit to plot.
    condition_boundaries : list[int]
        Slice indices where each condition starts, plus final count.
    conditions : list[str] or None
        Condition labels. Defaults to CONDITIONS.
    pre_ms, post_ms : float
        Window bounds relative to burst peak.
    """
    if conditions is None:
        conditions = CONDITIONS

    sss.plot_aligned_slice_single_unit(
        unit_idx,
        ax=ax,
        time_offset=0,
        x_range=(-pre_ms, post_ms),
        vlines=None,
        show_colorbar=False,
        font_size=FONT_SIZES["axes_label"],
        style="eventplot",
        invert_y=True,
        linewidths=0.5,
    )

    # Add condition boundary lines
    for i in range(1, len(condition_boundaries) - 1):
        ax.axhline(
            condition_boundaries[i] - 0.5,
            color="red",
            linewidth=0.7,
            linestyle=":",
            zorder=5,
        )

    # Y-ticks at condition midpoints with concentration labels
    cond_labels = {"D0": "0D", "D3": "3D", "D10": "10D", "D30": "30D", "D50": "50D"}
    ytick_positions = []
    ytick_labels = []
    for i in range(len(condition_boundaries) - 1):
        mid = (condition_boundaries[i] + condition_boundaries[i + 1]) / 2
        ytick_positions.append(mid)
        ytick_labels.append(cond_labels[conditions[i]])
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)

    ax.set_ylabel("Burst", fontsize=FONT_SIZES["axes_label"])


def plot_unit_avg_rate_heatmap(
    ax, ax_cbar, avg_rates, conditions=None, pre_ms=250, post_ms=500
):
    """Plot average firing rate heatmap per condition for one unit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the heatmap.
    ax_cbar : matplotlib.axes.Axes
        Target axes for the colorbar.
    avg_rates : np.ndarray (n_conditions, T)
        Mean firing rate per condition.
    conditions : list[str] or None
        Condition labels. Defaults to CONDITIONS.
    pre_ms, post_ms : float
        Window bounds relative to burst peak.
    """
    if conditions is None:
        conditions = CONDITIONS

    cond_labels = {"D0": "0D", "D3": "3D", "D10": "10D", "D30": "30D", "D50": "50D"}

    plot_heatmap(
        avg_rates,
        ax=ax,
        norm="row",
        cmap="hot",
        vmin=0,
        vmax=1,
        aspect="auto",
        origin="upper",
        extent=[-pre_ms, post_ms, -0.5, len(conditions) - 0.5],
        xlabel="Time from burst peak (ms)",
        ylabel="Av. rate",
        yticks=(list(range(len(conditions))), [cond_labels[c] for c in conditions]),
        show_colorbar=False,
    )

    # Colorbar on separate axes
    im = ax.images[0]
    cbar = ax.figure.colorbar(im, cax=ax_cbar)
    cbar.set_label("Norm. rate", fontsize=FONT_SIZES["colorbar_label"])
    cbar.ax.tick_params(labelsize=FONT_SIZES["tick_label"])
    cbar.outline.set_linewidth(0.5)

    # Re-enable all spines for heatmap style
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Panel C — 5x5 heatmap grid of average burst-aligned firing rates
# ═══════════════════════════════════════════════════════════════════════════
def compute_group0_avg_rates_and_orders(ws_path, conditions):
    """Compute average burst-aligned rates for group 0 units and per-condition orderings.

    Parameters
    ----------
    ws_path : str
        Path to the workspace directory.
    conditions : list[str]
        Condition labels.

    Returns
    -------
    avg_rates : dict[str, np.ndarray]
        condition -> (n_group0, T) average firing rate across bursts.
    orders : dict[str, np.ndarray]
        condition -> indices into group0 units sorted by median peak time.
    group0_mask : np.ndarray
        Boolean mask for group 0 units.
    """
    from spikelab.workspace.hdf5_io import load_workspace_item

    group_id = load_workspace_item(ws_path, "all", "unit_burst_group_id")
    group0_mask = group_id == 0
    group0_indices = np.where(group0_mask)[0]
    n_group0 = len(group0_indices)

    avg_rates = {}
    orders = {}

    for rec in conditions:
        rss = load_workspace_item(ws_path, rec, "burst_rss")
        # rss.event_stack: (U, T, S)
        # Subset to group 0 units, average across slices
        unit_avg = np.nanmean(
            rss.event_stack[group0_indices, :, :], axis=2
        )  # (n_group0, T)
        avg_rates[rec] = unit_avg

        # Order by median peak time across slices
        peak_times = np.zeros(n_group0)
        for i, u in enumerate(group0_indices):
            unit_rates = rss.event_stack[u, :, :]  # (T, S)
            slice_peaks = np.argmax(unit_rates, axis=0)  # (S,)
            peak_times[i] = np.median(slice_peaks) - PRE_MS
        orders[rec] = np.argsort(peak_times)

    return avg_rates, orders, group0_mask


def plot_avg_burst_heatmap(ax, avg_rates, order, vmax=None):
    """Plot average burst-aligned firing rate heatmap for group 0 units.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    avg_rates : np.ndarray (n_group0, T)
    order : np.ndarray
        Unit ordering indices.
    vmax : float or None

    Returns
    -------
    AxesImage
    """
    data = avg_rates[order, :]

    plot_heatmap(
        data,
        ax=ax,
        norm="row",
        cmap="hot",
        vmin=0,
        vmax=1,
        aspect="auto",
        origin="upper",
        extent=[-PRE_MS, data.shape[1] - PRE_MS, -0.5, data.shape[0] - 0.5],
        xlabel="",
        ylabel="",
        show_colorbar=False,
    )

    # Re-enable all spines for heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    return ax.images[0]


# ═══════════════════════════════════════════════════════════════════════════
# Panel D — PCA of burst u2u correlation matrices
# ═══════════════════════════════════════════════════════════════════════════
def plot_burst_pca(
    ax, pca_coords, cond_idx, var_explained, pc_x=0, pc_y=1, recordings=None
):
    """Plot PCA scatter of burst u2u correlations, colored by condition.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    pca_coords : np.ndarray (S, n_components)
    cond_idx : np.ndarray (S,)
        Integer condition index per slice (0=D0, ..., 4=D50).
    var_explained : np.ndarray (n_components,)
    pc_x, pc_y : int
    recordings : list[str] or None
    """
    if recordings is None:
        recordings = CONDITIONS

    group_colors = [COLORS[rec] for rec in recordings]

    plot_scatter(
        ax,
        pca_coords[:, pc_x],
        pca_coords[:, pc_y],
        xlabel=f"PC{pc_x + 1} ({var_explained[pc_x] * 100:.1f}%)",
        ylabel=f"PC{pc_y + 1} ({var_explained[pc_y] * 100:.1f}%)",
        groups=cond_idx,
        group_labels=recordings,
        group_colors=group_colors,
        marker_size=15,
        alpha=0.7,
        show_legend=True,
    )

    # Adjust legend to match manuscript style
    ax.legend(
        fontsize=FONT_SIZES["legend"],
        handlelength=0.8,
        handletextpad=0.3,
        labelspacing=0.2,
        markerscale=0.8,
    )

    ax.invert_xaxis()


# ═══════════════════════════════════════════════════════════════════════════
# Panels E–F — Burst-aligned pop rate + slice-to-slice time correlation
# ═══════════════════════════════════════════════════════════════════════════
def compute_closest_burst_edges(tburst, burst_edges):
    """Compute the closest (most conservative) burst start/end relative to peak.

    Parameters
    ----------
    tburst : np.ndarray
        Burst peak times (ms).
    burst_edges : np.ndarray
        Burst start/end times (ms), shape (B, 2).

    Returns
    -------
    start_rel : float
        Latest burst start relative to peak (closest to 0, negative).
    end_rel : float
        Earliest burst end relative to peak (closest to 0, positive).
    """
    starts_rel = burst_edges[:, 0] - tburst
    ends_rel = burst_edges[:, 1] - tburst
    return np.max(starts_rel), np.min(ends_rel)


def plot_avg_poprate_with_edges(ax, sd_data, recordings=None):
    """Plot burst-aligned average pop rate with closest burst edge markers.

    Uses SpikeData.plot_aligned_pop_rate() per condition on the same axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    sd_data : dict[str, dict]
        Condition -> dict with keys 'sd' (SpikeData), 'pop_rate' (1-D array),
        'tburst' (1-D array), 'burst_edges' ((B, 2) array).
    recordings : list[str] or None
    """
    if recordings is None:
        recordings = CONDITIONS

    for rec in recordings:
        d = sd_data[rec]
        d["sd"].plot_aligned_pop_rate(
            events=d["tburst"],
            pre_ms=PRE_MS,
            post_ms=POST_MS,
            ax=ax,
            pop_rate=d["pop_rate"],
            color=COLORS[rec],
            linewidth=LINE_WIDTHS["data_trace"],
            burst_edges=d["burst_edges"],
            edge_percentile=100,
            xlabel="",
            ylabel="Av. pop. rate (kHz)",
        )

    ax.axvline(0, color="0.4", linewidth=0.5, linestyle="--", zorder=0)


def plot_s2s_time_corr(ax, s2s_data, burst_edge_ranges, recordings=None):
    """Plot s2s time correlation clipped to closest burst edges.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    s2s_data : dict[str, np.ndarray]
        Condition -> 1-D s2s correlation (T,).
    burst_edge_ranges : dict[str, tuple]
        Condition -> (start_rel, end_rel) in ms.
    recordings : list[str] or None
    """
    if recordings is None:
        recordings = CONDITIONS

    T = len(s2s_data[recordings[0]])
    t_axis = np.arange(T) - PRE_MS

    for rec in recordings:
        label = rec.replace("D", "")
        vals = s2s_data[rec].copy()
        start_rel, end_rel = burst_edge_ranges[rec]
        vals[t_axis < start_rel] = np.nan
        vals[t_axis > end_rel] = np.nan
        ax.plot(
            t_axis,
            vals,
            color=COLORS[rec],
            linewidth=LINE_WIDTHS["data_trace"],
            label=label,
        )

    ax.axvline(0, color="0.4", linewidth=0.5, linestyle="--", zorder=0)
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.set_xlabel("Time from burst peak (ms)", fontsize=FONT_SIZES["axes_label"])
    ax.set_ylabel("Burst similarity", fontsize=FONT_SIZES["axes_label"])
    ax.legend(
        fontsize=FONT_SIZES["legend"],
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.2,
        columnspacing=0.5,
        loc="upper left",
        ncol=2,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Panel G — Burst-to-burst correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_burst_corr_heatmap(
    ax, corr_stack, burst_counts, recordings=None, vmin=0.65, vmax=0.85, cbar_ticks=None
):
    """Plot burst-to-burst correlation heatmap (nanmean over units) onto the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    corr_stack : np.ndarray
        Shape (S, S, U) — pairwise burst correlation per unit.
    burst_counts : list[int]
        Number of bursts per recording, in order.
    recordings : list[str] or None
        Recording labels for tick marks. Defaults to CONDITIONS.
    vmin, vmax : float
        Color axis limits.
    cbar_ticks : list[float] or None
        Custom colorbar ticks.

    Returns
    -------
    AxesImage
    """
    if recordings is None:
        recordings = CONDITIONS

    corr_matrix = np.nanmean(corr_stack, axis=2)

    # Compute group centers and boundaries for ticks and dividers
    centers = []
    cum = 0
    for count in burst_counts:
        centers.append(cum + count / 2)
        cum += count
    tick_labels = [r.replace("D", "") for r in recordings]

    boundaries = np.cumsum(burst_counts[:-1])
    hlines = [
        {"y": b - 0.5, "color": "red", "linestyle": ":", "linewidth": 0.8}
        for b in boundaries
    ]
    vlines = [
        {"x": b - 0.5, "color": "red", "linestyle": ":", "linewidth": 0.8}
        for b in boundaries
    ]

    plot_heatmap(
        corr_matrix,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        aspect="equal",
        origin="upper",
        xlabel="Burst",
        ylabel="Burst",
        xticks=(centers, tick_labels),
        yticks=(centers, tick_labels),
        hlines=hlines,
        vlines=vlines,
        show_colorbar=False,
    )

    # Re-enable all spines for heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # Colorbar with custom ticks
    im = ax.images[0]
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0.5)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    else:
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.ax.tick_params(labelsize=FONT_SIZES["tick_label"], width=0.5, length=2)
    cbar.set_label("Av. burst to burst corr.", fontsize=FONT_SIZES["colorbar_label"], rotation=270, labelpad=10)

    return im


# ═══════════════════════════════════════════════════════════════════════════
# Panel H — Within-condition burst correlation violins + p-value inset
# ═══════════════════════════════════════════════════════════════════════════
def extract_within_condition_corrs(corr_stack, burst_counts, recordings=None):
    """Extract lower-triangle within-condition correlations from the full stack.

    Parameters
    ----------
    corr_stack : np.ndarray
        Shape (S, S, U) — pairwise burst correlation per unit.
    burst_counts : list[int]
        Number of bursts per recording, in order.
    recordings : list[str] or None
        Condition labels.

    Returns
    -------
    dict[str, np.ndarray]
        Within-condition lower-triangle correlation values (nanmean over units).
    """
    if recordings is None:
        recordings = CONDITIONS

    corr_matrix = np.nanmean(corr_stack, axis=2)  # (S, S)

    within_corrs = {}
    offset = 0
    for rec, count in zip(recordings, burst_counts):
        block = corr_matrix[offset : offset + count, offset : offset + count]
        # Extract lower triangle (excluding diagonal)
        tri_idx = np.tril_indices(count, k=-1)
        within_corrs[rec] = block[tri_idx]
        offset += count

    return within_corrs


def compute_pairwise_ttests(within_corrs, recordings=None):
    """Compute pairwise t-tests with Bonferroni correction.

    Parameters
    ----------
    within_corrs : dict[str, np.ndarray]
        Within-condition values per recording.
    recordings : list[str] or None

    Returns
    -------
    pval_matrix : np.ndarray
        (K, K) matrix of corrected p-values. Diagonal is NaN.
    sig_matrix : np.ndarray
        (K, K) boolean matrix — True where corrected p < 0.05.
    n_comparisons : int
        Number of pairwise comparisons (for reporting).
    """
    if recordings is None:
        recordings = CONDITIONS

    K = len(recordings)
    pval_matrix = np.full((K, K), np.nan)
    n_comparisons = K * (K - 1) // 2

    for i in range(K):
        for j in range(i + 1, K):
            _, p = stats.ttest_ind(
                within_corrs[recordings[i]],
                within_corrs[recordings[j]],
                equal_var=False,
            )
            p_corrected = min(p * n_comparisons, 1.0)  # Bonferroni
            pval_matrix[i, j] = p_corrected
            pval_matrix[j, i] = p_corrected

    sig_matrix = pval_matrix < 0.05
    return pval_matrix, sig_matrix, n_comparisons


def plot_within_condition_violins(ax, within_corrs, recordings=None):
    """Plot violin plots of within-condition burst-to-burst correlations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    within_corrs : dict[str, np.ndarray]
        Mapping from recording name to 1-D array of lower-triangle
        pairwise correlation values (nanmean across units) for burst
        pairs within that condition.
    recordings : list[str] or None
        Ordered list of condition labels. Defaults to CONDITIONS.

    Returns
    -------
    parts : ViolinPlotReturn
    """
    if recordings is None:
        recordings = CONDITIONS

    colors = [COLORS[rec] for rec in recordings]
    labels = [r.replace("D", "") for r in recordings]

    parts = plot_distribution(
        ax,
        within_corrs,
        labels=labels,
        colors=colors,
        ylabel="Av. burst to burst corr.",
        xlabel="Diazepam (\u00b5M)",
        show_median=True,
        show_quartiles=True,
    )
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])

    # Extra headroom for inset
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.35)

    return parts


def add_pvalue_inset(ax, pval_matrix, sig_matrix, recordings=None):
    """Add a -log10(p) heatmap inset with significance markers.

    Delegates to plot_pvalue_matrix from spikelab.spikedata.plot_utils.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Parent axes (the violin plot).
    pval_matrix : np.ndarray
        (K, K) Bonferroni-corrected p-value matrix.
    sig_matrix : np.ndarray
        (K, K) boolean — True where p < 0.05.
    recordings : list[str] or None
        Condition labels.

    Returns
    -------
    ax_inset : matplotlib.axes.Axes
    """
    if recordings is None:
        recordings = CONDITIONS

    labels = [r.replace("D", "") for r in recordings]

    return plot_pvalue_matrix(
        pval_matrix,
        sig_matrix=sig_matrix,
        labels=labels,
        parent_ax=ax,
        inset_loc="upper left",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Panel I — Rank order correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_rank_order_heatmap(
    ax,
    matrix,
    burst_counts,
    cbar_label,
    recordings=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """Plot a slice-to-slice rank order heatmap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    matrix : np.ndarray (S, S)
        Pairwise rank order correlation matrix (z-scored).
    burst_counts : list[int]
        Number of bursts per recording, in order.
    cbar_label : str
        Colorbar label.
    recordings : list[str] or None
        Recording labels. Defaults to CONDITIONS.
    vmin, vmax : float or None
        Color axis limits. If None, auto-scaled.
    cmap : str
        Colormap name.

    Returns
    -------
    AxesImage
    """
    if recordings is None:
        recordings = CONDITIONS

    # Set diagonal to vmax for display
    matrix = matrix.copy()
    if vmax is not None:
        np.fill_diagonal(matrix, vmax)

    # Compute group centers and boundaries
    centers = []
    cum = 0
    for count in burst_counts:
        centers.append(cum + count / 2)
        cum += count
    tick_labels = [r.replace("D", "") for r in recordings]

    boundaries = np.cumsum(burst_counts[:-1])
    hlines = [
        {"y": b - 0.5, "color": "red", "linestyle": ":", "linewidth": 0.8}
        for b in boundaries
    ]
    vlines = [
        {"x": b - 0.5, "color": "red", "linestyle": ":", "linewidth": 0.8}
        for b in boundaries
    ]

    plot_heatmap(
        matrix,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
        origin="upper",
        xlabel="Burst",
        ylabel="Burst",
        xticks=(centers, tick_labels),
        yticks=(centers, tick_labels),
        hlines=hlines,
        vlines=vlines,
        show_colorbar=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # Colorbar
    im = ax.images[0]
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=FONT_SIZES["tick_label"], width=0.5, length=2)
    cbar.set_label(cbar_label, fontsize=FONT_SIZES["colorbar_label"], rotation=270, labelpad=10)

    return im


# ═══════════════════════════════════════════════════════════════════════════
# Panel J — Within-condition rank order violins
# ═══════════════════════════════════════════════════════════════════════════
def extract_within_condition_values(matrix, burst_counts, recordings=None):
    """Extract lower-triangle within-condition values from a (S, S) matrix.

    Parameters
    ----------
    matrix : np.ndarray (S, S)
        Pairwise slice-to-slice matrix.
    burst_counts : list[int]
        Number of bursts per recording.
    recordings : list[str] or None

    Returns
    -------
    dict[str, np.ndarray]
        Within-condition lower-triangle values per recording.
    """
    if recordings is None:
        recordings = CONDITIONS

    within = {}
    offset = 0
    for rec, count in zip(recordings, burst_counts):
        block = matrix[offset : offset + count, offset : offset + count]
        tri_idx = np.tril_indices(count, k=-1)
        vals = block[tri_idx]
        within[rec] = vals[~np.isnan(vals)]
        offset += count
    return within


def plot_rank_order_violins(ax, within_vals, recordings=None):
    """Plot violin plots of within-condition rank order correlations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    within_vals : dict[str, np.ndarray]
        Within-condition rank order z-scores per recording.
    recordings : list[str] or None

    Returns
    -------
    parts : ViolinPlotReturn
    """
    if recordings is None:
        recordings = CONDITIONS

    colors = [COLORS[rec] for rec in recordings]
    labels = [r.replace("D", "") for r in recordings]

    parts = plot_distribution(
        ax,
        within_vals,
        labels=labels,
        colors=colors,
        ylabel="Rank order corr. (z-score)",
        xlabel="Diazepam (\u00b5M)",
        show_median=True,
        show_quartiles=True,
    )

    # Extra room at the bottom for inset
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - (ymax - ymin) * 0.5, ymax)
    ax.set_yticks([0, 2, 4, 6, 8])

    return parts
