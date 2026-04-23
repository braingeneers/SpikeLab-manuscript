"""Figure 7 — panel plotting functions.

Reusable plotting functions for every panel in Figure 7. Each function
accepts pre-created matplotlib Axes and pre-loaded data — it never accesses
the workspace directly.

Panel groups:
    A     Burst raster + pop rate + GPLVM model states (stitched bursts)
    B     State transition probability matrices
    C     Average pop rate + average P(continuous) around burst peaks
    D     Average GPLVM state probability per recording
    E     Entropy distributions of GPLVM state posterior
    F     Cumulative PCA variance per recording
    G-H   PCA manifold colored by GPLVM state (D0, D50)
    I     Combined PCA manifold colored by condition

SpikeLab API used:
    - ``plot_recording()`` for raster + pop rate + model states
    - ``plot_heatmap()`` for transition matrices
    - ``plot_lines()`` for state probability and variance curves
    - ``plot_distribution()`` for entropy violins
    - ``plot_manifold()`` for PCA scatter
    - ``gplvm_state_entropy()`` for Shannon entropy computation

Helpers:
    - ``stitch_burst_spikedata()`` — append burst SpikeData segments with gaps
    - ``stitch_pop_rate()`` — extract and stitch pop rate segments
    - ``stitch_gplvm_states()`` — extract and stitch GPLVM model states
    - ``compute_transition_matrix()`` — state-to-state transition probabilities
    - ``map_gplvm_states_to_ms()`` — map GPLVM states to 1ms time axis
    - ``compute_avg_poprate_around_bursts()`` — burst-aligned average pop rate
    - ``compute_closest_burst_edges()`` — closest burst start/end relative to peak
    - ``compute_avg_cont_prob_per_condition()`` — burst-aligned average P(continuous)
    - ``compute_avg_state_prob_per_condition()`` — average state occupancy
    - ``compute_entropy_per_condition()`` — per-bin Shannon entropy
    - ``compute_avg_poprate_and_states()`` — average pop rate + state per bin
"""

import os
import sys

import numpy as np

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
    plot_lines,
    plot_distribution,
    plot_manifold,
)
from spikelab.spikedata.utils import gplvm_state_entropy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRE_MS = 250
POST_MS = 500
BIN_SIZE_MS = 50
BINS_PER_BURST = int((PRE_MS + POST_MS) / BIN_SIZE_MS)  # 15


# ===========================================================================
# Stitching helpers
# ===========================================================================
def stitch_burst_spikedata(burst_sds, gap_ms=150):
    """Append burst SpikeData segments with silence gaps.

    Parameters
    ----------
    burst_sds : list[SpikeData]
        Individual burst SpikeData objects (e.g. from a SpikeSliceStack).
    gap_ms : float
        Silence gap between bursts in ms.

    Returns
    -------
    stitched : SpikeData
        Combined SpikeData.
    boundaries_ms : list[tuple]
        (start_ms, end_ms) for each burst in the stitched data.
    """
    stitched = burst_sds[0]
    window_ms = burst_sds[0].length
    boundaries = [(0.0, window_ms)]
    for sd in burst_sds[1:]:
        stitched = stitched.append(sd, offset=gap_ms)
        prev_end = boundaries[-1][1] + gap_ms
        boundaries.append((prev_end, prev_end + window_ms))
    return stitched, boundaries


def stitch_pop_rate(pop_rate_full, tburst, pre_ms, post_ms, burst_indices, gap_ms=150):
    """Extract and stitch pop rate segments around burst peaks.

    Parameters
    ----------
    pop_rate_full : np.ndarray
        Full recording pop rate at 1ms resolution.
    tburst : np.ndarray
        Burst peak times in ms.
    pre_ms, post_ms : float
        Window before/after burst peak.
    burst_indices : list[int]
        Which bursts to include.
    gap_ms : float
        Gap between segments (filled with NaN).

    Returns
    -------
    stitched : np.ndarray
        Stitched pop rate (1ms resolution).
    """
    segments = []
    for i, idx in enumerate(burst_indices):
        t_peak = tburst[idx]
        start = int(t_peak - pre_ms)
        end = int(t_peak + post_ms)
        start = max(0, start)
        end = min(len(pop_rate_full), end)
        seg = pop_rate_full[start:end]
        # Pad if segment is short (edge of recording)
        expected_len = int(pre_ms + post_ms)
        if len(seg) < expected_len:
            seg = np.pad(seg, (0, expected_len - len(seg)), constant_values=np.nan)
        segments.append(seg)
        if i < len(burst_indices) - 1:
            segments.append(np.full(int(gap_ms), np.nan))
    return np.concatenate(segments)


def stitch_gplvm_states(gplvm_result, boundaries_bins, burst_indices, gap_bins=3):
    """Extract and stitch GPLVM model states for selected bursts.

    Parameters
    ----------
    gplvm_result : dict
        Full burst GPLVM result (from 'all' namespace).
    boundaries_bins : np.ndarray
        Per-burst bin boundaries, shape (n_total_bursts, 2).
    burst_indices : list[int]
        Global burst indices to include.
    gap_bins : int
        NaN gap bins between bursts.

    Returns
    -------
    model_states : np.ndarray
        Shape (n_latent, total_bins) -- posterior latent marginal.
    cont_prob : np.ndarray
        Shape (total_bins,) -- continuous dynamics probability.
    """
    decode = gplvm_result["decode_res"]
    post_marg = np.array(decode["posterior_latent_marg"])  # (T_total, n_latent)
    dyn_marg = np.array(decode["posterior_dynamics_marg"])  # (T_total, 2)

    n_latent = post_marg.shape[1]
    segments_states = []
    segments_cont = []

    for i, idx in enumerate(burst_indices):
        b_start, b_end = boundaries_bins[idx]
        segments_states.append(post_marg[b_start:b_end])  # (bins_per_burst, n_latent)
        segments_cont.append(dyn_marg[b_start:b_end, 0])  # (bins_per_burst,)
        if i < len(burst_indices) - 1:
            segments_states.append(np.full((gap_bins, n_latent), np.nan))
            segments_cont.append(np.full(gap_bins, np.nan))

    model_states = np.concatenate(segments_states, axis=0).T  # (n_latent, total_bins)
    cont_prob = np.concatenate(segments_cont)
    return model_states, cont_prob


# ===========================================================================
# Panel A — Burst raster + pop rate + GPLVM (standalone figure version)
# ===========================================================================
def plot_burst_raster_gplvm(
    stitched_sd,
    pop_rate,
    model_states,
    cont_prob,
    sort_indices,
    burst_boundaries_ms,
    figsize=None,
    pop_rate_ylim=None,
):
    """Plot raster + pop rate + GPLVM for stitched burst windows.

    Parameters
    ----------
    stitched_sd : SpikeData
        Stitched burst SpikeData.
    pop_rate : np.ndarray
        Stitched pop rate (1ms resolution), same length as stitched_sd.
    model_states : np.ndarray
        Shape (n_latent, total_bins) -- stitched model states.
    cont_prob : np.ndarray
        Shape (total_bins,) -- stitched continuity probability.
    sort_indices : np.ndarray
        Unit reorder indices.
    burst_boundaries_ms : list[tuple]
        (start_ms, end_ms) for each burst in the stitched data.
    figsize : tuple or None
        Figure size.
    pop_rate_ylim : tuple or None
        Fixed (ymin, ymax) for the pop rate panel. None = auto.

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig = plot_recording(
        stitched_sd,
        show_raster=True,
        show_pop_rate=True,
        pop_rate=pop_rate,
        cont_prob=cont_prob,
        show_model_states=True,
        model_states=model_states,
        sort_indices=sort_indices,
        raster_style="eventplot",
        figsize=figsize,
        height_ratios=[3, 1, 2],
        absolute_xticks=False,
        font_size=FONT_SIZES["axes_label"],
        show=False,
    )

    # --- Identify axes ---
    all_axes = fig.get_axes()
    ax_raster = all_axes[0]

    ax_poprate = None
    ax_model = None
    for ax in all_axes:
        if ax is ax_raster:
            continue
        if ax.images and ax is not ax_raster:
            ax_model = ax
        elif ax.get_lines() and ax is not ax_raster:
            ax_poprate = ax

    # --- Restyle raster ---
    for coll in ax_raster.collections:
        coll.set_linewidth(LINE_WIDTHS["raster_marker"])

    n_units = stitched_sd.N
    ax_raster.set_yticks([0, n_units - 1])
    ax_raster.set_yticklabels(["1", str(n_units)])
    ax_raster.spines["left"].set_bounds(0, n_units - 1)

    # --- Model states y-axis ---
    if ax_model is not None:
        img = ax_model.images[0]
        extent = img.get_extent()
        if extent is not None:
            ax_model.spines["left"].set_bounds(extent[2], extent[3])

    # --- Style colorbar ---
    for ax in all_axes:
        if ax not in (ax_raster, ax_poprate, ax_model) and ax.get_visible():
            ax.tick_params(width=0.5, length=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    # --- Style pop rate lines ---
    if ax_poprate is not None:
        for line in ax_poprate.get_lines():
            line.set_linewidth(LINE_WIDTHS["data_trace"])
        for child_ax in fig.get_axes():
            if child_ax is not ax_poprate and hasattr(child_ax, "_shared_axes"):
                for line in child_ax.get_lines():
                    line.set_linewidth(LINE_WIDTHS["data_trace"])
        if pop_rate_ylim is not None:
            ax_poprate.set_ylim(pop_rate_ylim)
        # Pin cont_prob (right y-axis) to 0-1
        for child_ax in fig.get_axes():
            if child_ax.get_ylabel() == "P(continuous)":
                child_ax.set_ylim(0, 1)

    # --- Draw burst boundary separators ---
    data_axes = [a for a in [ax_raster, ax_poprate, ax_model] if a is not None]
    for b_start, b_end in burst_boundaries_ms:
        for ax in data_axes:
            ax.axvline(b_start, color="0.7", linewidth=0.4, linestyle=":", zorder=0)
            ax.axvline(b_end, color="0.7", linewidth=0.4, linestyle=":", zorder=0)

    # --- Replace x-axis with scale bar ---
    bottom_ax = ax_model if ax_model is not None else (ax_poprate or ax_raster)
    for ax in all_axes:
        if ax is not bottom_ax:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel("")
    bottom_ax.set_xticks([])
    bottom_ax.set_xticklabels([])
    bottom_ax.set_xlabel("")
    bottom_ax.spines["bottom"].set_visible(False)
    ax_raster.spines["bottom"].set_visible(False)
    if ax_poprate is not None:
        ax_poprate.spines["bottom"].set_visible(False)

    # Scale bar: 200 ms
    scale_bar_ms = 200
    xlim = bottom_ax.get_xlim()
    ylim = bottom_ax.get_ylim()
    bar_x_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
    bar_x_start = bar_x_end - scale_bar_ms
    bar_y = ylim[0] - (ylim[1] - ylim[0]) * 0.05
    bottom_ax.plot(
        [bar_x_start, bar_x_end],
        [bar_y, bar_y],
        color="black",
        linewidth=LINE_WIDTHS["mean_trace"],
        clip_on=False,
        solid_capstyle="butt",
    )
    bottom_ax.text(
        (bar_x_start + bar_x_end) / 2,
        bar_y - (ylim[1] - ylim[0]) * 0.08,
        "200 ms",
        ha="center",
        va="top",
        fontsize=FONT_SIZES["axes_label"],
    )

    # --- Move colorbar closer to heatmap ---
    if ax_model is not None:
        model_pos = ax_model.get_position()
        ax_cont = None
        for ax in fig.get_axes():
            if ax.get_ylabel() == "P(continuous)":
                ax_cont = ax
        known = {id(ax_raster), id(ax_poprate), id(ax_model)}
        if ax_cont is not None:
            known.add(id(ax_cont))
        for ax in all_axes:
            if id(ax) in known or not ax.get_visible():
                continue
            pos = ax.get_position()
            if abs(pos.y0 - model_pos.y0) < 0.05:
                gap = pos.x0 - model_pos.x1
                if gap > 0.005:
                    ax.set_position(
                        [
                            model_pos.x1 + 0.005,
                            pos.y0,
                            pos.width,
                            pos.height,
                        ]
                    )

    return fig


# ===========================================================================
# Panel A helper — Full-recording raster + GPLVM (standalone figure version)
# ===========================================================================
def plot_raster_gplvm(
    sd,
    gplvm_result,
    sort_indices,
    start_ms=0,
    time_window_ms=30000,
    figsize=None,
):
    """Plot raster + GPLVM latent states using plot_recording.

    Parameters
    ----------
    sd : SpikeData
        Full recording.
    gplvm_result : dict
        GPLVM result dict from workspace (contains decode_res, reorder_indices).
    sort_indices : np.ndarray
        Unit ordering (indices into sd.train).
    start_ms : float
        Start time in ms.
    time_window_ms : float
        Duration to plot in ms.
    figsize : tuple or None
        Figure size (width, height) in inches.

    Returns
    -------
    fig : matplotlib.Figure
    """
    # Convert time range from ms to bin indices for GPLVM
    bin_size_ms = gplvm_result["bin_size_ms"]

    # Extract model states and cont_prob
    decode = gplvm_result["decode_res"]
    model_states = np.array(decode["posterior_latent_marg"]).T  # (S, T_bins)
    cont_prob_full = np.array(decode["posterior_dynamics_marg"])
    cont_prob = cont_prob_full[:, 0] if cont_prob_full.ndim == 2 else cont_prob_full

    fig = plot_recording(
        sd,
        show_raster=True,
        show_model_states=True,
        model_states=model_states,
        cont_prob=cont_prob,
        time_range=(start_ms, start_ms + time_window_ms),
        sort_indices=sort_indices,
        raster_style="eventplot",
        figsize=figsize,
        height_ratios=[2, 1, 2],
        absolute_xticks=False,
        font_size=FONT_SIZES["axes_label"],
        show=False,
    )

    # --- Identify axes ---
    all_axes = fig.get_axes()
    ax_raster = all_axes[0]

    ax_poprate = None
    ax_model = None
    for ax in all_axes:
        if ax is ax_raster:
            continue
        if ax.images and ax is not ax_raster:
            ax_model = ax
        elif ax.get_lines() and ax is not ax_raster:
            ax_poprate = ax

    # --- Restyle raster eventplot lines ---
    for coll in ax_raster.collections:
        coll.set_linewidth(LINE_WIDTHS["raster_marker"])

    # --- Raster y-axis: only first and last unit ---
    n_units = sd.N
    ax_raster.set_yticks([0, n_units - 1])
    ax_raster.set_yticklabels(["1", str(n_units)])
    ax_raster.spines["left"].set_bounds(0, n_units - 1)

    # --- Model states y-axis: bound spine to heatmap extent ---
    if ax_model is not None:
        img = ax_model.images[0]
        extent = img.get_extent()
        if extent is not None:
            ax_model.spines["left"].set_bounds(extent[2], extent[3])
        else:
            ylim_model = ax_model.get_ylim()
            ax_model.spines["left"].set_bounds(min(ylim_model), max(ylim_model))

    # --- Style model states colorbar ---
    for ax in all_axes:
        if ax not in (ax_raster, ax_poprate, ax_model) and ax.get_visible():
            ax.tick_params(width=0.5, length=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    # --- Style cont_prob overlay if present ---
    if ax_poprate is not None:
        for line in ax_poprate.get_lines():
            line.set_linewidth(LINE_WIDTHS["data_trace"])
        for child_ax in fig.get_axes():
            if child_ax is not ax_poprate and hasattr(child_ax, "_shared_axes"):
                for line in child_ax.get_lines():
                    line.set_linewidth(LINE_WIDTHS["data_trace"])

    # --- Replace x-axis with 2 s scale bar on bottom axes ---
    bottom_ax = ax_model if ax_model is not None else (ax_poprate or ax_raster)
    for ax in all_axes:
        if ax is not bottom_ax:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel("")
    bottom_ax.set_xticks([])
    bottom_ax.set_xticklabels([])
    bottom_ax.set_xlabel("")
    bottom_ax.spines["bottom"].set_visible(False)
    ax_raster.spines["bottom"].set_visible(False)
    if ax_poprate is not None:
        ax_poprate.spines["bottom"].set_visible(False)

    # Draw scale bar
    scale_bar_ms = 2000
    xlim = bottom_ax.get_xlim()
    ylim = bottom_ax.get_ylim()
    bar_x_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
    bar_x_start = bar_x_end - scale_bar_ms
    bar_y = ylim[0] - (ylim[1] - ylim[0]) * 0.05
    bottom_ax.plot(
        [bar_x_start, bar_x_end],
        [bar_y, bar_y],
        color="black",
        linewidth=LINE_WIDTHS["mean_trace"],
        clip_on=False,
        solid_capstyle="butt",
    )
    bottom_ax.text(
        (bar_x_start + bar_x_end) / 2,
        bar_y - (ylim[1] - ylim[0]) * 0.08,
        "2 s",
        ha="center",
        va="top",
        fontsize=FONT_SIZES["axes_label"],
    )

    return fig


# ===========================================================================
# Panel B — Transition matrices
# ===========================================================================
def compute_transition_matrix(
    gplvm_result,
    boundaries_bins,
    condition_idx,
    condition,
    n_conditions,
    n_states,
):
    """Compute state transition probability matrix for one condition.

    Transitions across burst boundaries (including silence gaps) are excluded --
    only within-burst consecutive bin transitions are counted.

    Parameters
    ----------
    gplvm_result : dict
        Full burst GPLVM result.
    boundaries_bins : np.ndarray
        (n_bursts, 2) bin boundaries per burst.
    condition_idx : np.ndarray
        Condition index per burst.
    condition : int
        Condition index to compute for.
    n_conditions : int
    n_states : int
        Number of latent states.

    Returns
    -------
    trans_prob : np.ndarray
        (n_states, n_states) transition probability matrix.
        Row i = current state, column j = next state.
        Rows sum to 1 (or 0 if state never visited).
    trans_counts : np.ndarray
        (n_states, n_states) raw transition counts.
    """
    decode = gplvm_result["decode_res"]
    post_marg = np.array(decode["posterior_latent_marg"])  # (T_total, n_latent)
    most_likely = np.argmax(post_marg, axis=1)

    counts = np.zeros((n_states, n_states), dtype=int)
    burst_indices = np.where(condition_idx == condition)[0]

    for idx in burst_indices:
        b_start, b_end = boundaries_bins[idx]
        states = most_likely[b_start:b_end]
        for t in range(len(states) - 1):
            counts[states[t], states[t + 1]] += 1

    # Normalize rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    trans_prob = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0,
    )
    return trans_prob, counts


def plot_transition_matrix(
    ax, trans_prob, cax=None, vmax=None, show_ylabel=True, show_xlabel=True
):
    """Plot a single transition probability matrix as a heatmap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    trans_prob : np.ndarray
        (n_states, n_states) transition probability matrix.
    cax : matplotlib.axes.Axes or None
        Axes for colorbar. If None, no colorbar.
    vmax : float or None
        Colormap maximum. None = auto.
    show_ylabel : bool
    show_xlabel : bool
    """
    import matplotlib.pyplot as plt

    n = trans_prob.shape[0]

    plot_heatmap(
        trans_prob,
        ax=ax,
        vmin=0,
        vmax=vmax,
        cmap="hot",
        aspect="equal",
        origin="lower",
        xlabel="Next state" if show_xlabel else "",
        ylabel="Current state" if show_ylabel else "",
        xticks=([0, n - 1], ["1", str(n)]),
        yticks=([0, n - 1], ["1", str(n)]),
        show_colorbar=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    im = ax.images[0]
    if cax is not None:
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("P(transition)", fontsize=FONT_SIZES["colorbar_label"])
        cb.outline.set_linewidth(0.5)
        cax.tick_params(width=0.5, length=2)

    return im


# ===========================================================================
# Panel C — Avg pop rate + avg P(continuous)
# ===========================================================================
def compute_avg_poprate_around_bursts(pop_rate, tburst, pre_ms, post_ms):
    """Cut pop_rate around each burst peak and average across bursts."""
    slices = []
    for t in tburst:
        t0 = int(t) - int(pre_ms)
        t1 = int(t) + int(post_ms)
        if t0 >= 0 and t1 <= len(pop_rate):
            slices.append(pop_rate[t0:t1])
    return np.mean(slices, axis=0)


def compute_closest_burst_edges(tburst, burst_edges):
    """Compute the closest burst start/end relative to peak."""
    starts_rel = burst_edges[:, 0] - tburst
    ends_rel = burst_edges[:, 1] - tburst
    return np.max(starts_rel), np.min(ends_rel)


def compute_avg_cont_prob_per_condition(
    gplvm_result,
    boundaries_bins,
    condition_idx,
    n_conditions,
):
    """Compute burst-aligned average P(continuous) per condition.

    Parameters
    ----------
    gplvm_result : dict
        Full burst GPLVM result.
    boundaries_bins : np.ndarray
        (n_bursts, 2) bin boundaries per burst.
    condition_idx : np.ndarray
        Condition index per burst.
    n_conditions : int

    Returns
    -------
    avg_cont : list[np.ndarray]
        One array of shape (bins_per_burst,) per condition.
    """
    decode = gplvm_result["decode_res"]
    dyn_marg = np.array(decode["posterior_dynamics_marg"])  # (T_total, 2)
    cont_prob_full = dyn_marg[:, 0]

    avg_cont = []
    for c in range(n_conditions):
        mask = condition_idx == c
        burst_indices = np.where(mask)[0]
        slices = []
        for idx in burst_indices:
            b_start, b_end = boundaries_bins[idx]
            seg = cont_prob_full[b_start:b_end]
            if len(seg) == BINS_PER_BURST:
                slices.append(seg)
        avg_cont.append(np.mean(slices, axis=0))
    return avg_cont


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


def plot_avg_cont_prob(ax, avg_cont_data, burst_edge_ranges, recordings=None):
    """Plot burst-aligned average P(continuous) per condition.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    avg_cont_data : dict[str, np.ndarray]
        Condition -> 1-D average P(continuous) (bins_per_burst,).
    burst_edge_ranges : dict[str, tuple]
        Condition -> (start_rel, end_rel) in ms.
    recordings : list[str] or None
    """
    if recordings is None:
        recordings = CONDITIONS

    # Time axis in ms, centered on burst peak (bin centers)
    t_axis = (np.arange(BINS_PER_BURST) + 0.5) * BIN_SIZE_MS - PRE_MS

    for rec in recordings:
        label = rec.replace("D", "")
        vals = avg_cont_data[rec].copy()
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
    ax.set_xlim(-PRE_MS, POST_MS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time from burst peak (ms)", fontsize=FONT_SIZES["axes_label"])
    ax.set_ylabel("Av. P(continuous)", fontsize=FONT_SIZES["axes_label"])


# ===========================================================================
# Panel D — Average state probability
# ===========================================================================
def compute_avg_state_prob_per_condition(
    gplvm_result,
    boundaries_bins,
    condition_idx,
    n_conditions,
):
    """Average posterior_latent_marg over burst bins per condition.

    Parameters
    ----------
    gplvm_result : dict
        Full burst GPLVM result.
    boundaries_bins : np.ndarray
        (n_bursts, 2) bin boundaries per burst.
    condition_idx : np.ndarray
        Condition index per burst.
    n_conditions : int

    Returns
    -------
    avg_probs : list[np.ndarray]
        One array of shape (n_latent_bin,) per condition.
    """
    decode = gplvm_result["decode_res"]
    post_marg = np.array(decode["posterior_latent_marg"])  # (T_total, n_latent)

    avg_probs = []
    for c in range(n_conditions):
        burst_indices = np.where(condition_idx == c)[0]
        # Collect only burst bins (not silence gaps)
        all_bins = []
        for idx in burst_indices:
            b_start, b_end = boundaries_bins[idx]
            all_bins.append(post_marg[b_start:b_end])
        stacked = np.concatenate(all_bins, axis=0)  # (total_burst_bins, n_latent)
        avg_probs.append(np.mean(stacked, axis=0))
    return avg_probs


def plot_avg_state_prob(ax, avg_probs, recordings=None):
    """Plot average state probability per recording on the same axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    avg_probs : dict[str, np.ndarray]
        Condition -> 1-D average state probability (n_latent_bin,).
    recordings : list[str] or None
    """
    if recordings is None:
        recordings = CONDITIONS

    n_states = len(avg_probs[recordings[0]])
    states = np.arange(n_states)
    colors = [COLORS[rec] for rec in recordings]

    plot_lines(
        ax,
        avg_probs,
        x=states,
        colors=colors,
        xlabel="Latent state",
        ylabel="Av. state probability",
        linewidth=LINE_WIDTHS["data_trace"],
    )

    # Cap y-axis to show detail in lower conditions; mark clipped peaks
    ymax = 0.25
    ax.set_ylim(0, ymax)
    for rec in recordings:
        peak_idx = np.argmax(avg_probs[rec])
        peak_val = avg_probs[rec][peak_idx]
        if peak_val > ymax * 0.95:
            ax.annotate(
                f"{peak_val:.2f}",
                xy=(peak_idx, ymax * 0.97),
                xytext=(peak_idx + 3, ymax * 0.82),
                fontsize=FONT_SIZES["annotation"],
                color=COLORS[rec],
                arrowprops=dict(arrowstyle="->", color=COLORS[rec], lw=0.7),
                clip_on=False,
                zorder=10,
            )

    ax.legend(
        fontsize=FONT_SIZES["legend"],
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.5,
        loc="best",
        ncol=1,
    )


# ===========================================================================
# Panel E — Entropy distributions
# ===========================================================================
def compute_entropy_per_condition(
    gplvm_result,
    boundaries_bins,
    condition_idx,
    n_conditions,
):
    """Compute per-time-bin entropy for burst bins per condition.

    Parameters
    ----------
    gplvm_result : dict
        Full burst GPLVM result.
    boundaries_bins : np.ndarray
        (n_bursts, 2) bin boundaries per burst.
    condition_idx : np.ndarray
        Condition index per burst.
    n_conditions : int

    Returns
    -------
    entropies : list[np.ndarray]
        One 1-D array of entropies per condition.
    """
    decode = gplvm_result["decode_res"]
    post_marg = np.array(decode["posterior_latent_marg"])  # (T_total, n_latent)
    all_entropy = gplvm_state_entropy(post_marg)

    entropies = []
    for c in range(n_conditions):
        burst_indices = np.where(condition_idx == c)[0]
        bins = []
        for idx in burst_indices:
            b_start, b_end = boundaries_bins[idx]
            bins.append(all_entropy[b_start:b_end])
        entropies.append(np.concatenate(bins))
    return entropies


def plot_entropy_violins(ax, entropies, recordings=None):
    """Plot entropy distributions as violins.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    entropies : dict[str, np.ndarray]
        Condition -> 1-D entropy values.
    recordings : list[str] or None
    """
    if recordings is None:
        recordings = CONDITIONS

    metric_data = [entropies[rec] for rec in recordings]
    colors = [COLORS[rec] for rec in recordings]

    plot_distribution(
        ax,
        metric_data,
        labels=recordings,
        colors=colors,
        ylabel="State entropy (nats)",
        style="violin",
        show_median=True,
        show_quartiles=True,
        show_data=False,
        font_size=FONT_SIZES["axes_label"],
    )


# ===========================================================================
# Panel F — PCA cumulative variance
# ===========================================================================
def plot_cumulative_variance(ax, var_data, recordings=None):
    """Plot cumulative explained variance per PC for each recording.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    var_data : dict[str, np.ndarray]
        Condition -> explained variance ratio per component.
    recordings : list[str] or None
    """
    if recordings is None:
        recordings = CONDITIONS

    # Build cumulative variance traces
    cumvar_data = {rec: np.cumsum(var_data[rec]) for rec in recordings}
    n_pcs = len(next(iter(cumvar_data.values())))
    pcs = np.arange(1, n_pcs + 1)
    colors = [COLORS[rec] for rec in recordings]

    plot_lines(
        ax,
        cumvar_data,
        x=pcs,
        colors=colors,
        xlabel="PC",
        ylabel="Cum. explained var.",
        linewidth=LINE_WIDTHS["data_trace"],
    )

    # Add markers and adjust styling
    for line in ax.get_lines():
        line.set_marker("o")
        line.set_markersize(3)

    ax.set_xticks(pcs)
    ax.set_ylim(0.55, 0.85)
    ax.legend(
        fontsize=FONT_SIZES["legend"],
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.5,
        loc="best",
        ncol=1,
    )


# ===========================================================================
# Panels G-H — PCA with GPLVM states
# ===========================================================================
def map_gplvm_states_to_ms(
    tburst,
    gplvm_result,
    boundaries_bins,
    global_burst_indices,
    recording_length,
):
    """Map GPLVM most-likely state back to 1ms time axis of the recording.

    Parameters
    ----------
    tburst : np.ndarray
        Burst peak times in ms for this recording.
    gplvm_result : dict
        Full burst GPLVM result.
    boundaries_bins : np.ndarray
        (n_total_bursts, 2) bin boundaries per burst.
    global_burst_indices : list[int]
        Global burst indices for this recording's bursts.
    recording_length : int
        Number of 1ms time points in the recording.

    Returns
    -------
    state_at_ms : np.ndarray
        (recording_length,) array. -1 for non-burst time points,
        0..n_states-1 for burst time points.
    """
    decode = gplvm_result["decode_res"]
    post_marg = np.array(decode["posterior_latent_marg"])  # (T_total, n_latent)
    most_likely = np.argmax(post_marg, axis=1)

    state_at_ms = np.full(recording_length, -1, dtype=int)

    for local_i, global_i in enumerate(global_burst_indices):
        t_peak = tburst[local_i]
        burst_start_ms = int(t_peak - PRE_MS)

        b_start, b_end = boundaries_bins[global_i]
        states = most_likely[b_start:b_end]

        for j, state in enumerate(states):
            ms_start = burst_start_ms + j * BIN_SIZE_MS
            ms_end = ms_start + BIN_SIZE_MS
            ms_start = max(0, ms_start)
            ms_end = min(recording_length, ms_end)
            state_at_ms[ms_start:ms_end] = state

    return state_at_ms


def plot_pca_with_states(
    ax, embedding, state_at_ms, n_states, cmap="viridis", point_size=0.1, bg_alpha=0.05
):
    """Plot PC1 vs PC2 with points colored by GPLVM state.

    Uses plot_manifold from spikelab.spikedata.plot_utils.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    embedding : np.ndarray
        (T, >=2) PCA embedding.
    state_at_ms : np.ndarray
        (T,) state index per ms. -1 = non-burst.
    n_states : int
    cmap : str
    point_size : float
    bg_alpha : float
        Alpha for non-burst (background) points.
    """
    bg_mask = state_at_ms == -1

    sc = plot_manifold(
        ax,
        embedding,
        bg_mask=bg_mask,
        bg_size=point_size * 3,
        bg_alpha=bg_alpha,
        color_vals=state_at_ms.astype(float),
        cmap=cmap,
        vmin=0,
        vmax=n_states - 1,
        marker_size=point_size * 8,
        alpha=0.5,
        show_colorbar=False,
    )
    return sc


def compute_avg_poprate_and_states(
    pop_rate,
    tburst,
    gplvm_result,
    boundaries_bins,
    global_indices,
):
    """Compute burst-peak-centered average pop rate and average most-likely state.

    Parameters
    ----------
    pop_rate : np.ndarray
        Full recording pop rate at 1ms resolution.
    tburst : np.ndarray
        Burst peak times in ms.
    gplvm_result : dict
        Full burst GPLVM result.
    boundaries_bins : np.ndarray
        (n_total_bursts, 2) per-burst bin boundaries.
    global_indices : list[int]
        Global burst indices for this recording.

    Returns
    -------
    avg_pr : np.ndarray
        (PRE_MS + POST_MS,) average pop rate.
    avg_state : np.ndarray
        (BINS_PER_BURST,) most likely state per bin from averaged posterior.
    """
    decode = gplvm_result["decode_res"]
    post_marg = np.array(decode["posterior_latent_marg"])  # (T_total, n_latent)

    pr_slices = []
    post_slices = []
    for local_i, global_i in enumerate(global_indices):
        t_peak = tburst[local_i]
        t0 = int(t_peak - PRE_MS)
        t1 = int(t_peak + POST_MS)
        if t0 >= 0 and t1 <= len(pop_rate):
            pr_slices.append(pop_rate[t0:t1])
        b_start, b_end = boundaries_bins[global_i]
        seg = post_marg[b_start:b_end]
        if len(seg) == BINS_PER_BURST:
            post_slices.append(seg)

    avg_pr = np.mean(pr_slices, axis=0)
    avg_post = np.mean(post_slices, axis=0)  # (BINS_PER_BURST, n_latent)
    avg_state = np.argmax(avg_post, axis=1)  # (BINS_PER_BURST,)
    return avg_pr, avg_state


def add_poprate_inset(ax, avg_pr, avg_state, n_states, cmap="viridis"):
    """Add a pop rate inset colored by GPLVM state to the top-right of ax.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Parent axes.
    avg_pr : np.ndarray
        (PRE_MS + POST_MS,) average pop rate.
    avg_state : np.ndarray
        (BINS_PER_BURST,) most likely state per GPLVM bin.
    n_states : int
    cmap : str
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    ax_in = inset_axes(ax, width="40%", height="30%", loc="lower left", borderpad=0.5)

    t_axis = np.arange(len(avg_pr)) - PRE_MS
    norm = Normalize(vmin=0, vmax=n_states - 1)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    # Plot each 50ms segment colored by its state
    for j in range(BINS_PER_BURST):
        idx_start = max(0, j * BIN_SIZE_MS)
        idx_end = min(len(avg_pr), (j + 1) * BIN_SIZE_MS)
        color = sm.to_rgba(avg_state[j])
        ax_in.plot(
            t_axis[idx_start : idx_end + 1],
            avg_pr[idx_start : idx_end + 1],
            color=color,
            linewidth=2.0,
            solid_capstyle="butt",
        )

    # Marker at burst peak (t=0)
    peak_val = avg_pr[PRE_MS]
    ax_in.plot(
        0, peak_val, marker="v", markersize=3, color="0.3", zorder=5, clip_on=False
    )

    ax_in.set_xlim(-PRE_MS, POST_MS)
    # Remove all axes, spines, ticks -- only the line and marker
    for spine in ax_in.spines.values():
        spine.set_visible(False)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.patch.set_alpha(0)

    return ax_in
