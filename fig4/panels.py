"""Figure 4 — panel plotting functions.

Reusable plotting functions for every panel in Figure 4. Each function
accepts pre-created matplotlib Axes and pre-loaded data — it never accesses
the workspace directly. This makes the functions composable: they work
both in standalone previews and when called from ``assemble.py``.

Panel groups:
    A   Raster + population rate (one per condition)
    B–F Violin distributions of per-unit metrics
    G–K Normalized per-unit metric percentile bands across conditions
    L   Burst detection threshold sensitivity
    M   Burst width distributions + burst count overlay
    N–P D0 vs D50 scatter plots colored by normalized FR change

SpikeLab API used:
    - ``plot_recording()`` for raster + population rate
    - ``plot_distribution()`` for violin panels
    - ``plot_percentile_bands()`` for normalized band panels
    - ``plot_burst_sensitivity()`` for threshold sweep
    - ``plot_scatter()`` for D0 vs D50 panels
"""

import os
import sys

import numpy as np
from matplotlib.patches import Rectangle

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
    plot_distribution,
    plot_percentile_bands,
    plot_burst_sensitivity as _lib_burst_sensitivity,
    plot_scatter,
)


# ═══════════════════════════════════════════════════════════════════════════
# Panel A — Raster + population rate
# ═══════════════════════════════════════════════════════════════════════════
def get_unit_order_by_firing_rate(sd):
    """Return unit indices sorted from highest to lowest firing rate.

    With the raster y-axis inverted (unit 0 at top), this places the
    most active units at the top of the raster.

    Parameters
    ----------
    sd : SpikeData

    Returns
    -------
    np.ndarray
        Integer indices into ``sd.train``.
    """
    return np.argsort(sd.rates(unit="Hz"))[::-1]


def plot_raster_poprate(
    ax_raster,
    ax_poprate,
    sd,
    pop_rate,
    burst_edges,
    sort_indices,
    color,
    start_ms=0,
    time_window_ms=20000,
    poprate_ylim=None,
):
    """Plot spike raster and population rate onto pre-existing axes.

    Delegates core plotting to ``plot_recording`` via its ``axes`` parameter,
    then applies manuscript styling (burst shading, scale bar, line styles).

    Parameters
    ----------
    ax_raster, ax_poprate : Axes
        Target axes for the raster and population rate, respectively.
    sd : SpikeData
        Full recording.
    pop_rate : np.ndarray
        Pre-computed population rate (1 ms bins, full recording length).
    burst_edges : np.ndarray
        Burst start/end times (ms), shape ``(B, 2)``.
    sort_indices : np.ndarray
        Unit display order (indices into ``sd.train``).
    color : str
        Color for the population rate trace and burst markers.
    start_ms : float
        Start of the time window to display (ms).
    time_window_ms : float
        Duration of the time window (ms).
    poprate_ylim : tuple[float, float] or None
        Fixed y-axis limits for the population rate panel.
    """
    fig = ax_raster.figure

    # Hidden colorbar axes required by plot_recording
    cax_r = fig.add_axes([0, 0, 0.001, 0.001])
    cax_r.axis("off")
    cax_p = fig.add_axes([0, 0, 0.001, 0.001])
    cax_p.axis("off")

    plot_recording(
        sd,
        show_raster=True,
        show_pop_rate=True,
        pop_rate=pop_rate,
        time_range=(start_ms, start_ms + time_window_ms),
        sort_indices=sort_indices,
        raster_style="eventplot",
        burst_edges=burst_edges,
        axes=[(ax_raster, cax_r), (ax_poprate, cax_p)],
        absolute_xticks=False,
        font_size=FONT_SIZES["axes_label"],
        show=False,
    )

    # --- Pop rate line color ---
    for line in ax_poprate.get_lines():
        line.set_color(color)
        line.set_linewidth(LINE_WIDTHS["data_trace"])

    # --- Fixed y-axis ---
    if poprate_ylim is not None:
        ax_poprate.set_ylim(poprate_ylim)

    # --- Burst shading: clip to y >= 0 ---
    for patch in list(ax_poprate.patches):
        patch.remove()
    ylim = ax_poprate.get_ylim()
    ymax = ylim[1]
    if burst_edges is not None:
        end_val = start_ms + time_window_ms
        for t0, t1 in burst_edges:
            if t1 < start_ms or t0 > end_val:
                continue
            x0 = max(t0, start_ms) - start_ms
            x1 = min(t1, end_val) - start_ms
            rect = Rectangle((x0, 0), x1 - x0, ymax, facecolor=color, alpha=0.4)
            ax_poprate.add_patch(rect)

    # --- Raster line widths ---
    for coll in ax_raster.collections:
        coll.set_linewidth(LINE_WIDTHS["raster_marker"])

    # --- Raster y-axis: first and last unit only ---
    n_units = sd.N
    ax_raster.set_yticks([0, n_units - 1])
    ax_raster.set_yticklabels(["1", str(n_units)])

    # --- Replace x-axis with a 5 s scale bar ---
    for ax in [ax_raster, ax_poprate]:
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.spines["bottom"].set_visible(False)

    scale_bar_ms = 5000
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
        "5 s",
        ha="center",
        va="top",
        fontsize=FONT_SIZES["axes_label"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Panels B–F — Metric violin distributions
# ═══════════════════════════════════════════════════════════════════════════
def _plot_metric_violins(ax, metric_data, ylabel="", recordings=None):
    """Plot violin distributions of a per-unit metric across conditions.

    Parameters
    ----------
    ax : Axes
    metric_data : dict[str, np.ndarray]
        Condition name -> 1-D array of per-unit values.
    ylabel : str
    recordings : list[str] or None
        Condition order. Defaults to CONDITIONS.
    """
    if recordings is None:
        recordings = CONDITIONS
    colors = [COLORS[rec] for rec in recordings]
    labels = [r.replace("D", "") for r in recordings]
    return plot_distribution(
        ax,
        metric_data,
        labels=labels,
        colors=colors,
        ylabel=ylabel,
        xlabel="Diazepam (µM)",
        show_median=True,
        show_quartiles=True,
    )


def plot_firing_rate(ax, data, recordings=None):
    """Violin: firing rate (Hz)."""
    return _plot_metric_violins(ax, data, "Firing rate (Hz)", recordings)


def plot_isi_cv(ax, data, recordings=None):
    """Violin: ISI coefficient of variation."""
    return _plot_metric_violins(ax, data, "ISI CV", recordings)


def plot_pop_coupling(ax, data, recordings=None):
    """Violin: population coupling (zero-lag)."""
    return _plot_metric_violins(ax, data, "Population coupling", recordings)


def plot_frac_spikes_in_burst(ax, data, recordings=None):
    """Violin: fraction of spikes inside burst windows."""
    return _plot_metric_violins(ax, data, "Frac. spikes in bursts", recordings)


def plot_frac_bursts_active(ax, data, recordings=None):
    """Violin: fraction of bursts with >=2 spikes."""
    return _plot_metric_violins(ax, data, "Frac. bursts ≥2 spikes", recordings)


# ═══════════════════════════════════════════════════════════════════════════
# Panels G–K — Normalized metric percentile bands
# ═══════════════════════════════════════════════════════════════════════════
def _plot_normalized_bands(ax, metric_dict, ylabel, recordings=None,
                           ylim_range=0.6, show_legend=False):
    """Plot percentile bands of a normalized per-unit metric across conditions.

    Normalization: ``N = (D' - D0') / (D' + D0')`` where ``D' = max(D, 0)``.
    Shows median line with shaded IQR band per condition.

    Parameters
    ----------
    ax : Axes
    metric_dict : dict[str, np.ndarray]
    ylabel : str
    recordings : list[str] or None
    ylim_range : float
        Symmetric y-axis limit (±ylim_range).
    show_legend : bool
    """
    if recordings is None:
        recordings = CONDITIONS
    labels = [r.replace("D", "") for r in recordings]
    plot_percentile_bands(
        ax,
        metric_dict,
        labels=labels,
        normalize=True,
        style="bands",
        ylabel=ylabel,
        xlabel="Diazepam (µM)",
        ylim_range=ylim_range,
        show_legend=show_legend,
    )
    if show_legend:
        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=leg_labels,
            loc="upper left",
            fontsize=5.5,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.2,
        )


def plot_fr_normalized(ax, data, recordings=None, show_legend=False):
    """Percentile bands: normalized firing rate."""
    _plot_normalized_bands(ax, data, "Norm. firing rate", recordings,
                           ylim_range=0.75, show_legend=show_legend)


def plot_isi_cv_normalized(ax, data, recordings=None):
    """Percentile bands: normalized ISI CV."""
    _plot_normalized_bands(ax, data, "Norm. ISI CV", recordings)


def plot_pop_coupling_normalized(ax, data, recordings=None):
    """Percentile bands: normalized population coupling."""
    _plot_normalized_bands(ax, data, "Norm. pop. coupling", recordings)


def plot_frac_spikes_in_burst_normalized(ax, data, recordings=None):
    """Percentile bands: normalized fraction of spikes in bursts."""
    _plot_normalized_bands(ax, data, "Norm. frac. spikes in bursts", recordings)


def plot_frac_bursts_active_normalized(ax, data, recordings=None):
    """Percentile bands: normalized fraction of bursts >=2 spikes."""
    _plot_normalized_bands(ax, data, "Norm. frac. bursts ≥2 spikes", recordings,
                           ylim_range=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Panel L — Burst threshold sensitivity
# ═══════════════════════════════════════════════════════════════════════════
def plot_burst_sensitivity(ax, thresholds, burst_counts, recordings=None):
    """Plot burst count vs RMS threshold for each condition.

    Parameters
    ----------
    ax : Axes
    thresholds : np.ndarray
        1-D array of RMS multiplier thresholds.
    burst_counts : dict[str, np.ndarray]
        Condition -> burst count array (same length as *thresholds*).
    recordings : list[str] or None
    """
    if recordings is None:
        recordings = CONDITIONS
    colors = [COLORS[rec] for rec in recordings]

    _lib_burst_sensitivity(
        ax,
        thresholds,
        burst_counts,
        labels=recordings,
        colors=colors,
    )

    # Strip "D" prefix from legend entries
    legend = ax.get_legend()
    for text in legend.get_texts():
        text.set_text(text.get_text().replace("D", ""))
    ax.legend(
        fontsize=FONT_SIZES["legend"],
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.2,
    )

    # Reference line at the threshold used in this study (2.5x RMS)
    ax.axvline(2.5, color="0.4", linewidth=0.7, linestyle=":", zorder=0)


# ═══════════════════════════════════════════════════════════════════════════
# Panel M — Burst width distributions
# ═══════════════════════════════════════════════════════════════════════════
def plot_burst_widths(ax, burst_widths, burst_counts, recordings=None):
    """Plot burst width violins with burst count overlay on secondary y-axis.

    Parameters
    ----------
    ax : Axes
    burst_widths : dict[str, np.ndarray]
        Condition -> array of burst widths (ms).
    burst_counts : dict[str, int]
        Condition -> number of bursts.
    recordings : list[str] or None

    Returns
    -------
    parts : ViolinPlotReturn
    ax2 : Axes
        Secondary y-axis with burst count line.
    """
    if recordings is None:
        recordings = CONDITIONS
    colors = [COLORS[rec] for rec in recordings]
    labels = [r.replace("D", "") for r in recordings]

    parts = plot_distribution(
        ax,
        burst_widths,
        labels=labels,
        colors=colors,
        ylabel="Burst width (ms)",
        xlabel="Diazepam (µM)",
        show_median=True,
        show_quartiles=True,
    )

    # Secondary y-axis: burst counts
    positions = list(range(len(recordings)))
    ax2 = ax.twinx()
    counts = [burst_counts[rec] for rec in recordings]
    ax2.plot(
        positions,
        counts,
        "o-",
        color="black",
        markersize=4,
        linewidth=LINE_WIDTHS["data_trace"],
        zorder=5,
    )
    ax2.set_yticks([40, 80])
    ax2.set_ylabel("")
    ax2.text(
        1.06,
        0.5,
        "N bursts",
        transform=ax2.transAxes,
        fontsize=FONT_SIZES["axes_label"],
        rotation=270,
        va="center",
        ha="left",
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)

    return parts, ax2


# ═══════════════════════════════════════════════════════════════════════════
# Panels N–P — D0 vs D50 scatter plots
# ═══════════════════════════════════════════════════════════════════════════
def compute_norm_change(d0_arr, d50_arr):
    """Compute normalized change between two conditions.

    Formula: ``(max(D50, 0) - max(D0, 0)) / (max(D50, 0) + max(D0, 0))``

    Returns values in [-1, 1]; NaN where denominator is zero.

    Parameters
    ----------
    d0_arr, d50_arr : np.ndarray

    Returns
    -------
    np.ndarray
    """
    a = np.maximum(d0_arr, 0)
    b = np.maximum(d50_arr, 0)
    denom = a + b
    with np.errstate(invalid="ignore"):
        return np.where(denom > 0, (b - a) / denom, np.nan)


def _plot_d0_vs_d50(
    ax, d0_vals, d50_vals, name, color_vals, color_label, unit=None,
    show_colorbar=True,
):
    """D0 vs D50 scatter for a single metric, colored by a change metric.

    Parameters
    ----------
    ax : Axes
    d0_vals, d50_vals : np.ndarray
        Per-unit values at D0 and D50.
    name : str
        Metric display name.
    color_vals : np.ndarray
        Per-unit color values in [-1, 1].
    color_label : str
    unit : str or None
        Unit string for axis labels (e.g. "Hz").
    show_colorbar : bool

    Returns
    -------
    sc : PathCollection
    """
    valid = ~np.isnan(d0_vals) & ~np.isnan(d50_vals) & ~np.isnan(color_vals)
    x = d0_vals[valid]
    y = d50_vals[valid]
    c = color_vals[valid]
    unit_str = f" ({unit})" if unit else ""

    sc = plot_scatter(
        ax,
        x,
        y,
        xlabel=f"D0 {name}{unit_str}",
        ylabel=f"D50 {name}{unit_str}",
        color_vals=c,
        color_label=color_label,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        show_identity=True,
        show_colorbar=show_colorbar,
        marker_size=8,
        alpha=0.7,
    )

    # Dotted identity line style
    for line in ax.get_lines():
        if line.get_linestyle() == "--":
            line.set_linestyle(":")
            line.set_color("0.3")
            line.set_linewidth(0.7)

    # Equal aspect with symmetric limits
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    margin = (hi - lo) * 0.05
    lims = [lo - margin, hi + margin]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    return sc


def plot_pop_coupling_d0_vs_d50(
    ax, d0, d50, color_vals, color_label, show_colorbar=True,
):
    """Scatter: population coupling D0 vs D50."""
    return _plot_d0_vs_d50(
        ax, d0, d50, "pop. coupling", color_vals, color_label,
        show_colorbar=show_colorbar,
    )


def plot_frac_bursts_d0_vs_d50(
    ax, d0, d50, color_vals, color_label, show_colorbar=True,
):
    """Scatter: fraction of bursts active D0 vs D50."""
    return _plot_d0_vs_d50(
        ax, d0, d50, "frac. bursts ≥2 spikes", color_vals, color_label,
        show_colorbar=show_colorbar,
    )


def plot_frac_spikes_d0_vs_d50(
    ax, d0, d50, color_vals, color_label, show_colorbar=True,
):
    """Scatter: fraction of spikes in bursts D0 vs D50."""
    return _plot_d0_vs_d50(
        ax, d0, d50, "frac. spikes in bursts", color_vals, color_label,
        show_colorbar=show_colorbar,
    )
