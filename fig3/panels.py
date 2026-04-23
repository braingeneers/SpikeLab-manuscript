"""Figure 3 — panel plotting functions.

Reusable plotting functions for every panel in Figure 3. Each function
accepts pre-created Axes and pre-loaded data.

Panel groups:
    A-C   Raster + FR heatmap + population rate (one per recording)
    D-E   Single-unit event-aligned raster + PSTH
    F-H   Population coupling over time (heatmap, norm bands, burst rate)
    I     Cue vs no-cue coupling scatter
    J     Raw vs z-scored coupling scatter
    K     Burst detection sensitivity curves

SpikeLab API used:
    - ``plot_recording()`` for raster + FR heatmap + pop rate
    - ``plot_scatter_with_marginals()`` for scatter panels
    - ``SpikeSliceStack.plot_aligned_slice_single_unit()`` for unit rasters
"""

import os
import sys

import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))
sys.path.insert(0, CODE_DIR)

from plot_config import FONT_SIZES, LINE_WIDTHS
from spikelab.spikedata.plot_utils import plot_recording, plot_scatter_with_marginals

FS = FONT_SIZES["axes_label"]


# ── Axes registry ─────────────────────────────────────────────────────────
axes_registry = {}


def reg(ax, role, desc):
    axes_registry[id(ax)] = {"role": role, "desc": desc, "ax": ax}
    return ax


# ══════════════════════════════════════════════════════════════════════════
# PANEL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════


def plot_raster_panel(gs_slot, fig, sd, pop_rate, fr_rates, tburst, burst_edges,
                      color, start_ms, time_window_ms, ns_label="",
                      heatmap_vmax=100, show_colorbar=True, show_ylabels=True):
    """Raster + FR heatmap + pop rate for a recording.

    Parameters
    ----------
    sd : SpikeData
    pop_rate : ndarray
    fr_rates : RateData
    tburst : ndarray or None
    burst_edges : ndarray or None
    color : str
    start_ms, time_window_ms : float
    ns_label : str
        Label prefix for axes registry.
    """
    rates = sd.rates(unit="Hz")
    sort_indices = np.argsort(rates)[::-1]

    end_ms = start_ms + time_window_ms
    if tburst is not None and len(tburst) > 0:
        mask = (tburst >= start_ms) & (tburst <= end_ms)
        tburst_win = tburst[mask]
        edges_win = burst_edges[mask] if burst_edges is not None else None
    else:
        tburst_win = None
        edges_win = None

    inner = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=gs_slot,
        height_ratios=[3, 2, 1], width_ratios=[1, 0.02],
        hspace=0.05, wspace=0.05,
    )
    ax_raster = reg(fig.add_subplot(inner[0, 0]), "panel", f"{ns_label}_raster")
    ax_heatmap = reg(fig.add_subplot(inner[1, 0], sharex=ax_raster), "panel", f"{ns_label}_heatmap")
    ax_poprate = reg(fig.add_subplot(inner[2, 0], sharex=ax_raster), "panel", f"{ns_label}_poprate")
    ax_cbar = reg(fig.add_subplot(inner[1, 1]), "colorbar", f"{ns_label}_cbar")
    for row in [0, 2]:
        ax_e = fig.add_subplot(inner[row, 1]); ax_e.axis("off")
        reg(ax_e, "hidden", f"{ns_label}_empty_{row}")

    cax_r = fig.add_axes([0, 0, 0.001, 0.001]); cax_r.axis("off")
    cax_p = fig.add_axes([0, 0, 0.001, 0.001]); cax_p.axis("off")

    plot_recording(
        sd,
        show_raster=True, show_fr_rates=True, show_pop_rate=True,
        pop_rate=pop_rate, fr_rates=fr_rates,
        time_range=(start_ms, end_ms),
        sort_indices=sort_indices, raster_style="eventplot",
        burst_times=tburst_win, burst_edges=edges_win,
        vmax_heatmap=heatmap_vmax,
        axes=[(ax_raster, cax_r), (ax_poprate, cax_p), (ax_heatmap, ax_cbar)],
        absolute_xticks=False, font_size=FS, show=False,
    )

    # Restyle
    for line in ax_poprate.get_lines():
        line.set_color(color); line.set_linewidth(LINE_WIDTHS["data_trace"])
    for coll in ax_poprate.collections:
        coll.set_facecolor("black"); coll.set_edgecolor("black"); coll.set_sizes([10])
    for patch in list(ax_poprate.patches):
        patch.remove()
    if edges_win is not None and len(edges_win) > 0:
        ylim = ax_poprate.get_ylim()
        for t0, t1 in edges_win:
            if t1 < start_ms or t0 > end_ms:
                continue
            x0 = max(t0, start_ms) - start_ms
            x1 = min(t1, end_ms) - start_ms
            ax_poprate.add_patch(Rectangle((x0, 0), x1 - x0, ylim[1], facecolor=color, alpha=0.2))
    for coll in ax_raster.collections:
        coll.set_linewidth(LINE_WIDTHS["raster_marker"])
    ax_raster.set_yticks([0, sd.N - 1])
    ax_raster.set_yticklabels(["1", str(sd.N)])
    for ax in [ax_raster, ax_heatmap]:
        ax.set_xticks([]); ax.set_xticklabels([]); ax.set_xlabel("")
    ax_poprate.set_xlabel("Time (s)", fontsize=FS)
    xticks = ax_poprate.get_xticks()
    ax_poprate.set_xticklabels([f"{t/1000:.0f}" for t in xticks])

    if not show_colorbar:
        ax_cbar.clear()
        ax_cbar.axis("off")
    else:
        ax_cbar.yaxis.label.set_size(FONT_SIZES["colorbar_label"])
        ax_cbar.yaxis.label.set_rotation(270)
        ax_cbar.yaxis.label.set_va("bottom")
        ax_cbar.yaxis.set_label_coords(7.0, 0.5)
    if not show_ylabels:
        for ax in [ax_raster, ax_heatmap, ax_poprate]:
            ax.set_ylabel("")

    # Time scale bar on poprate
    ax_poprate.set_xticks([])
    ax_poprate.set_xticklabels([])
    ax_poprate.set_xlabel("")
    ax_poprate.spines["bottom"].set_visible(False)
    ax_raster.spines["bottom"].set_visible(False)

    scale_bar_ms = 5000  # 5 s
    xlim = ax_poprate.get_xlim()
    ylim = ax_poprate.get_ylim()
    bar_x_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
    bar_x_start = bar_x_end - scale_bar_ms
    bar_y = ylim[0] - (ylim[1] - ylim[0]) * 0.05
    ax_poprate.plot(
        [bar_x_start, bar_x_end], [bar_y, bar_y],
        color="black", linewidth=LINE_WIDTHS["mean_trace"],
        clip_on=False, solid_capstyle="butt",
    )
    ax_poprate.text(
        (bar_x_start + bar_x_end) / 2,
        bar_y - (ylim[1] - ylim[0]) * 0.08,
        "5 s", ha="center", va="top", fontsize=FS,
    )

    return ax_raster, ax_poprate


def plot_unit_raster_psth(gs_slot, fig, sd, unit_idx, events, event_label,
                          pre_ms, post_ms, ns_label="", color=None,
                          feedback_split=False, feedback_type=None):
    """Single-unit event-aligned raster + PSTH.

    Parameters
    ----------
    sd : SpikeData
    unit_idx : int
    events : ndarray
    event_label : str
    pre_ms, post_ms : float
    """
    bin_ms = 10

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_slot,
        height_ratios=[1.5, 1], hspace=0.08,
    )
    ax_raster = reg(fig.add_subplot(inner[0]), "panel", f"{ns_label}_unit{unit_idx}_raster")
    ax_rate = reg(fig.add_subplot(inner[1], sharex=ax_raster), "panel", f"{ns_label}_unit{unit_idx}_rate")

    if feedback_split and feedback_type is not None:
        events = np.asarray(events).ravel()
        feedback_type = np.asarray(feedback_type).ravel()
        valid = ~np.isnan(events)
        events = events[valid]
        feedback_type = feedback_type[valid]

        correct_mask = feedback_type == 1.0
        events_correct = events[correct_mask]
        events_error = events[~correct_mask]

        sss_c = sd.align_to_events(events_correct, pre_ms=pre_ms, post_ms=post_ms, kind="spike")
        sss_e = sd.align_to_events(events_error, pre_ms=pre_ms, post_ms=post_ms, kind="spike")
        n_c = len(sss_c.spike_stack)
        n_e = len(sss_e.spike_stack)

        spikes_c = [s.train[unit_idx] for s in sss_c.spike_stack]
        spikes_e = [s.train[unit_idx] for s in sss_e.spike_stack]
        all_spikes = spikes_c + spikes_e
        colors_per = ["#1a9641"] * n_c + ["#d7191c"] * n_e

        ax_raster.eventplot(all_spikes, colors=colors_per, linewidths=0.4)
        ax_raster.axhline(n_c - 0.5, color="black", linewidth=0.8)
        ax_raster.axvline(0, color="red", linestyle=":", linewidth=0.8)
        ax_raster.set_ylim(-0.5, n_c + n_e - 0.5)
        ax_raster.invert_yaxis()
        ax_raster.set_ylabel("Trial", fontsize=FS)

        # PSTH
        n_bins = int((pre_ms + post_ms) / bin_ms)
        bin_edges = np.linspace(-pre_ms, post_ms, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for sss, col, lab in [(sss_c, "#1a9641", "Correct"), (sss_e, "#d7191c", "Error")]:
            counts = np.zeros(n_bins)
            for s in sss.spike_stack:
                h, _ = np.histogram(s.train[unit_idx], bins=bin_edges)
                counts += h
            rate = counts / (len(sss.spike_stack) * bin_ms / 1000)
            rate = gaussian_filter1d(rate, sigma=2)
            ax_rate.plot(bin_centers, rate, color=col, linewidth=0.8, label=lab)
        ax_rate.legend(fontsize=FONT_SIZES["legend"], loc="upper right", frameon=False,
                       bbox_to_anchor=(1.05, 1.0))
    else:
        sss = sd.align_to_events(events, pre_ms=pre_ms, post_ms=post_ms, kind="spike")
        sss.plot_aligned_slice_single_unit(
            unit_idx, ax=ax_raster, time_offset=0, xlabel="", ylabel=event_label,
            vlines=[{"x": 0, "color": "red", "linestyle": ":", "linewidth": 0.8}],
            style="eventplot", invert_y=True, show_colorbar=False,
            marker_size=8, font_size=FS, linewidths=0.4,
        )
        # PSTH
        n_bins = int((pre_ms + post_ms) / bin_ms)
        bin_edges = np.linspace(-pre_ms, post_ms, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts = np.zeros(n_bins)
        for s in sss.spike_stack:
            h, _ = np.histogram(s.train[unit_idx], bins=bin_edges)
            counts += h
        rate = counts / (len(sss.spike_stack) * bin_ms / 1000)
        rate = gaussian_filter1d(rate, sigma=2)
        ax_rate.plot(bin_centers, rate, color="black", linewidth=0.8)

    ax_raster.tick_params(labelsize=FONT_SIZES["tick_label"], labelbottom=False, bottom=False)
    ax_rate.axvline(0, color="red", linestyle=":", linewidth=0.8)
    ax_rate.set_xlabel("Relative time (ms)", fontsize=FS)
    ax_rate.set_ylabel("Rate (Hz)", fontsize=FS)
    ax_rate.tick_params(labelsize=FONT_SIZES["tick_label"], labelbottom=False, bottom=False)
    ax_rate.set_xlim(-pre_ms, post_ms)

    # Scale bar
    scale_ms = 200 if post_ms <= 500 else 500
    xlim = ax_raster.get_xlim()
    total_range = xlim[1] - xlim[0]
    bar_x_start = xlim[0] + total_range * 0.02
    bar_x_end = bar_x_start + scale_ms
    ylim_rast = ax_raster.get_ylim()
    y_bottom = ylim_rast[0]
    bar_y = y_bottom + abs(ylim_rast[0] - ylim_rast[1]) * 0.03
    ax_raster.plot(
        [bar_x_start, bar_x_end], [bar_y, bar_y],
        color="black", linewidth=LINE_WIDTHS["mean_trace"],
        clip_on=False, solid_capstyle="butt", transform=ax_raster.transData,
    )
    ax_raster.text(
        bar_x_end + total_range * 0.01, bar_y,
        f"{scale_ms} ms", ha="left", va="center",
        fontsize=FONT_SIZES["tick_label"], clip_on=False,
    )

    return ax_raster


def plot_coupling_panels(gs_slot, fig, coupling_data, tburst_cue, tburst_nocue,
                         first_cue_s, cue_boundary):
    """Pop coupling heatmap + norm bands + burst rate (3 vertically stacked).

    Parameters
    ----------
    coupling_data : dict
        From pop_coupling_over_time_full.npz: coupling_matrix, time_centers_s,
        cue_boundary_s, spike_counts.
    tburst_cue, tburst_nocue : ndarray
        Burst times for human cue and no-cue conditions.
    first_cue_s : float
        Time of first cue onset in seconds.
    cue_boundary : float
        Boundary between cue and no-cue periods in seconds.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    coupling = coupling_data["coupling_matrix"]
    time_centers = coupling_data["time_centers_s"]
    spike_counts = coupling_data["spike_counts"]
    n_slices, n_units_raw = coupling.shape

    # 80% activity filter
    n_valid = np.sum(~np.isnan(coupling), axis=0)
    active = n_valid >= 0.8 * n_slices
    coupling = coupling[:, active]
    n_units = active.sum()

    # Symmetric normalization
    x0 = coupling[0, :]
    x_pos = np.clip(coupling, 0, None)
    x0_pos = np.clip(x0, 0, None)[np.newaxis, :]
    denom = x_pos + x0_pos
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_coupling = np.where(denom > 0, (x_pos - x0_pos) / denom, 0.0)

    # Heatmap bins
    N_BINS = 40
    vmin = np.nanpercentile(coupling, 1)
    vmax = np.nanpercentile(coupling, 99)
    bin_edges = np.linspace(vmin, vmax, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_matrix = np.zeros((N_BINS, n_slices))
    for i in range(n_slices):
        counts, _ = np.histogram(coupling[i, :], bins=bin_edges)
        hist_matrix[:, i] = counts / n_units

    # Burst counts
    FRAME_LENGTH_MS = 120000
    tburst_all = np.concatenate([tburst_cue, tburst_nocue + cue_boundary * 1000])
    tburst_all.sort()
    burst_per_frame = np.array([
        np.sum((tburst_all >= (tc * 1000 - FRAME_LENGTH_MS / 2)) &
               (tburst_all < (tc * 1000 + FRAME_LENGTH_MS / 2)))
        for tc in time_centers
    ])
    burst_per_min = burst_per_frame / (FRAME_LENGTH_MS / 1000 / 60)

    t0, t1 = time_centers[0], time_centers[-1]
    COLOR = "#2166AC"

    inner = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=gs_slot,
        height_ratios=[0.7, 0.8, 0.7], width_ratios=[1, 0.02],
        hspace=0.25, wspace=0.03,
    )

    # F: Heatmap
    ax_heat = reg(fig.add_subplot(inner[0, 0]), "panel", "F_heatmap")
    ax_cbar_placeholder = fig.add_subplot(inner[0, 1]); ax_cbar_placeholder.axis("off")
    reg(ax_cbar_placeholder, "hidden", "F_cbar_placeholder")
    im = ax_heat.imshow(hist_matrix, aspect="auto", origin="lower",
                        extent=[t0, t1, bin_centers[0], bin_centers[-1]],
                        cmap="hot", interpolation="nearest")
    ax_heat.axvline(cue_boundary, color="white", linestyle="--", linewidth=0.8)
    ax_heat.set_ylabel("Coupling", fontsize=FS)
    ax_heat.tick_params(labelsize=FONT_SIZES["tick_label"], labelbottom=False, bottom=False)

    ax_cbar = inset_axes(
        ax_heat, width="45%", height="6%",
        loc="lower right", borderpad=0,
        bbox_to_anchor=(0, -0.12, 1, 1), bbox_transform=ax_heat.transAxes,
    )
    reg(ax_cbar, "colorbar", "F_cbar")
    ax_cbar.set_zorder(10)
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Frac. neurons", fontsize=FONT_SIZES["colorbar_label"], labelpad=1)
    cbar.ax.tick_params(labelsize=FONT_SIZES["tick_label"])
    cbar.ax.set_zorder(10)

    # G: Norm bands
    ax_band = reg(fig.add_subplot(inner[1, 0], sharex=ax_heat), "panel", "G_bands")
    ax_e1 = fig.add_subplot(inner[1, 1]); ax_e1.axis("off"); reg(ax_e1, "hidden", "G_empty")
    median_line = np.nanmedian(norm_coupling, axis=1)
    ax_band.plot(time_centers, median_line, color=COLOR, linewidth=0.8)
    for (lo, hi), alpha in [((5, 95), 0.15), ((25, 75), 0.35)]:
        lo_v = np.nanpercentile(norm_coupling, lo, axis=1)
        hi_v = np.nanpercentile(norm_coupling, hi, axis=1)
        ax_band.fill_between(time_centers, lo_v, hi_v, color=COLOR, alpha=alpha)
    ax_band.axvline(cue_boundary, color="red", linestyle="--", linewidth=0.8)
    ax_band.axhline(0, color="black", linestyle=":", linewidth=0.5)
    ax_band.set_ylim(-0.75, 0.75)
    ax_band.set_yticks([-0.5, 0, 0.5])
    ax_band.set_ylabel("Norm. change", fontsize=FS, labelpad=1)
    ax_band.tick_params(labelsize=FONT_SIZES["tick_label"], labelbottom=False, bottom=False)

    # Scale bar
    scale_bar_s = 500
    xlim = ax_band.get_xlim()
    ylim_g = ax_band.get_ylim()
    bar_y = ylim_g[1] + (ylim_g[1] - ylim_g[0]) * 0.15
    bar_x_start = xlim[0] + (xlim[1] - xlim[0]) * 0.08
    bar_x_end = bar_x_start + scale_bar_s
    ax_band.plot(
        [bar_x_start, bar_x_end], [bar_y, bar_y],
        color="black", linewidth=LINE_WIDTHS["mean_trace"],
        clip_on=False, solid_capstyle="butt",
    )
    ax_band.text(
        bar_x_end + (xlim[1] - xlim[0]) * 0.01, bar_y,
        "500 s", ha="left", va="center", fontsize=FONT_SIZES["tick_label"],
    )

    # H: Burst rate + spike count
    ax_burst = reg(fig.add_subplot(inner[2, 0], sharex=ax_heat), "panel", "H_burst")
    ax_e2 = fig.add_subplot(inner[2, 1]); ax_e2.axis("off"); reg(ax_e2, "hidden", "H_empty")
    ax_burst.plot(time_centers, burst_per_min, color="#B2182B", linewidth=0.8)
    ax_burst.axvline(cue_boundary, color="red", linestyle="--", linewidth=0.8)
    ax_burst.set_xlabel("Recording time (s)", fontsize=FS)
    ax_burst.set_ylabel("Bursts/min", fontsize=FS, color="#B2182B")
    ax_burst.tick_params(labelsize=FONT_SIZES["tick_label"], axis="y", colors="#B2182B")
    ax_burst.tick_params(axis="x", labelbottom=False, length=0)
    ax_spk = ax_burst.twinx()
    ax_spk.plot(time_centers, spike_counts / 1000, color="black", linewidth=0.5, alpha=0.6)
    ax_spk.set_ylabel("Spikes (k)", fontsize=FS, color="black", rotation=270)
    ax_spk.yaxis.set_label_coords(1.09, 0.55)
    ax_spk.set_yticks([15, 25])
    ax_spk.set_yticklabels(["15k", "25k"])
    ax_spk.tick_params(labelsize=FONT_SIZES["tick_label"], colors="black")
    ax_spk.spines["right"].set_visible(True)
    ax_burst.set_xlim(t0, t1)

    return ax_heat, ax_band, ax_burst


def plot_cue_vs_nocue(gs_slot, fig, coupling_data):
    """Cue vs no-cue mean coupling scatter.

    Parameters
    ----------
    coupling_data : dict
        From pop_coupling_over_time_full.npz.
    """
    coupling = coupling_data["coupling_matrix"]
    time_centers = coupling_data["time_centers_s"]
    cue_boundary = float(coupling_data["cue_boundary_s"])
    n_slices = coupling.shape[0]

    n_valid = np.sum(~np.isnan(coupling), axis=0)
    active = n_valid >= 0.8 * n_slices
    mean_coupling = np.nanmean(coupling, axis=0)
    std_coupling = np.nanstd(coupling, axis=0)
    cv = np.where(mean_coupling > 0, std_coupling / mean_coupling, np.nan)

    cue_mask = time_centers < cue_boundary
    mean_cue = np.nanmean(coupling[cue_mask], axis=0)
    mean_nocue = np.nanmean(coupling[~cue_mask], axis=0)

    v = active & ~np.isnan(mean_cue) & ~np.isnan(mean_nocue) & ~np.isnan(cv)

    ax_sc, ax_hx, ax_hy, sc = plot_scatter_with_marginals(
        gs_slot, fig, mean_cue[v], mean_nocue[v],
        xlabel="Coupling (cue)", ylabel="Coupling (no cue)",
        show_identity=True, font_size=FS,
        marker_size=8, marginal_bins=15,
    )
    ax_sc.set_aspect("auto")
    return ax_sc


def plot_raw_vs_zscore(gs_slot, fig, shuffle_data, sd_nocue):
    """Raw vs z-scored coupling scatter.

    Parameters
    ----------
    shuffle_data : dict
        From coupling_shuffle.npz: coupling_original, coupling_zscore.
    sd_nocue : SpikeData
        Human no-cue SpikeData (for firing rates).
    """
    raw = shuffle_data["coupling_original"]
    z = shuffle_data["coupling_zscore"]

    valid = ~np.isnan(raw) & ~np.isnan(z)
    ax_sc, ax_hx, ax_hy, sc = plot_scatter_with_marginals(
        gs_slot, fig, raw[valid], z[valid],
        xlabel="Coupling (raw)", ylabel="Coupling (z)",
        font_size=FS,
        marker_size=8, marginal_bins=15,
    )
    ax_sc.set_aspect("auto")
    return ax_sc


def plot_burst_sensitivity(ax, sensitivity_data):
    """Burst sensitivity curves.

    Parameters
    ----------
    sensitivity_data : dict
        From burst_sensitivity.npz.
    """
    thresholds = sensitivity_data["thresholds"]

    NAMESPACES = ["human_nocue", "mouse_nocue", "D0"]
    LABELS = {"human_nocue": "Human", "mouse_nocue": "Mouse", "D0": "Organoid"}
    COLORS_BS = {"human_nocue": "#2166AC", "mouse_nocue": "#B2182B", "D0": "#1B7837"}
    THRESHOLDS_USED = {"human_nocue": 1.5, "mouse_nocue": 1.25, "D0": 2.5}

    for ns in NAMESPACES:
        counts = sensitivity_data[f"counts_{ns}"]
        dur = float(sensitivity_data[f"duration_min_{ns}"])
        rate = counts / dur
        ax.plot(thresholds, rate, color=COLORS_BS[ns], linewidth=0.8, label=LABELS[ns])
        rate_at_thr = np.interp(THRESHOLDS_USED[ns], thresholds, rate)
        ax.plot(THRESHOLDS_USED[ns], rate_at_thr, "o", color=COLORS_BS[ns], markersize=4, zorder=5)

    ax.set_xlabel("Threshold (x RMS)", fontsize=FS)
    ax.set_ylabel("Bursts / min", fontsize=FS, labelpad=-8)
    ax.set_yticks([0, 30])
    ax.tick_params(labelsize=FONT_SIZES["tick_label"])
    ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)
