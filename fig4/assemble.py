"""Assemble Figure 4 — Neuronal firing and burst properties.

Layout (5-column, 4-row composite):
    Row A (panels A):   Spike raster + population rate, one per condition
    Row B-F:            Violin distributions — firing rate, ISI CV,
                        population coupling, frac. bursts active,
                        frac. spikes in bursts
    Row G-K:            Normalized percentile-band plots of the same 5 metrics
    Row L-P:            Burst sensitivity, burst widths, 3 D0-vs-D50 scatters

Prerequisites:
    - compute_shared.py  (pop rates, bursts)
    - fig4/compute.py    (unit metrics, burst metrics, sensitivity)

Usage:
    python -m fig4.assemble     (from the 200123_2953 directory)
    python fig4/assemble.py     (same)

Outputs:
    fig4/Figure4_preview.png    PNG preview
    fig4/Figure4.tif            Submission-ready 8-bit RGB TIFF at 900 dpi
"""

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib

matplotlib.use("Agg")

import sys

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style,
    save_fig,
    add_panel_label,
    FULL_PAGE_WIDTH,
    COLORS,
    FONT_SIZES,
    CONDITIONS,
)

apply_style()

UNIFIED_FONTSIZE = FONT_SIZES["tick_label"]  # 7 pt for all labels
mpl.rcParams["axes.labelsize"] = UNIFIED_FONTSIZE
mpl.rcParams["xtick.labelsize"] = UNIFIED_FONTSIZE
mpl.rcParams["ytick.labelsize"] = UNIFIED_FONTSIZE

# ---------------------------------------------------------------------------
# Panel plotting functions (from sibling module)
# ---------------------------------------------------------------------------
from fig4.panels import (
    get_unit_order_by_firing_rate,
    plot_raster_poprate,
    plot_firing_rate,
    plot_isi_cv,
    plot_pop_coupling,
    plot_frac_bursts_active,
    plot_frac_spikes_in_burst,
    plot_fr_normalized,
    plot_isi_cv_normalized,
    plot_pop_coupling_normalized,
    plot_frac_spikes_in_burst_normalized,
    plot_frac_bursts_active_normalized,
    plot_burst_sensitivity,
    plot_burst_widths,
    plot_pop_coupling_d0_vs_d50,
    plot_frac_bursts_d0_vs_d50,
    plot_frac_spikes_d0_vs_d50,
    compute_norm_change,
)

# ---------------------------------------------------------------------------
# Axes registry — tracks every axes for consistent font enforcement
# ---------------------------------------------------------------------------
axes_registry = {}


def reg(ax, role, desc):
    """Register an axes. role: 'panel', 'colorbar', 'hidden'."""
    axes_registry[id(ax)] = {"role": role, "desc": desc, "ax": ax}
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
from spikelab.workspace.hdf5_io import load_workspace_item

results_dir = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
ws_path = os.path.join(results_dir, "workspace")


def ws_get(ns, key):
    return load_workspace_item(ws_path, ns, key)


# Row A: raster data
raster_data = {}
for cond in CONDITIONS:
    raster_data[cond] = {
        "sd": ws_get(cond, "spikedata"),
        "pop_rate": ws_get(cond, "pop_rate_acc"),
        "burst_edges": ws_get(cond, "burst_edges"),
    }

sort_indices = get_unit_order_by_firing_rate(raster_data["D0"]["sd"])

# Global pop rate ylim (plot_recording displays in Hz/unit)
n_units = raster_data["D0"]["sd"].N
bin_s = 1.0 / 1000.0  # 1 ms bins
raw_max = max(np.max(raster_data[c]["pop_rate"]) for c in CONDITIONS)
global_max_hz = raw_max / (bin_s * n_units)
poprate_ylim = (0, global_max_hz * 1.05)

# Per-condition raster start times (ms) — chosen to show representative activity
start_times = {"D0": 5000, "D3": 1500, "D10": 0, "D30": 2700, "D50": 500}

# Rows B-K: per-unit metrics
fr_data, cv_data, coupling_data = {}, {}, {}
frac_spike_data, frac_burst_data = {}, {}
for rec in CONDITIONS:
    fr_data[rec] = ws_get(rec, "firing_rates_hz")
    cv_data[rec] = ws_get(rec, "isi_cv")
    coupling_data[rec] = ws_get(rec, "pop_coupling_zero")
    frac_spike_data[rec] = ws_get(rec, "frac_spikes_in_burst")
    frac_burst_data[rec] = ws_get(rec, "frac_bursts_active")

# Row L-P: burst sensitivity, widths, scatters
thresholds = ws_get("all", "burst_sensitivity_thresholds")
burst_sensitivity_counts = ws_get("all", "burst_sensitivity_counts")

burst_widths, burst_counts = {}, {}
for rec in CONDITIONS:
    edges = ws_get(rec, "burst_edges")
    widths = edges[:, 1] - edges[:, 0]
    burst_widths[rec] = widths
    burst_counts[rec] = len(widths)

fr_change = compute_norm_change(
    ws_get("D0", "firing_rates_hz"), ws_get("D50", "firing_rates_hz")
)

# ═══════════════════════════════════════════════════════════════════════════
# Figure layout
# ═══════════════════════════════════════════════════════════════════════════
col_width = FULL_PAGE_WIDTH / 5
raster_height = 2.2
poprate_height = 0.6
metric_height = col_width * 0.85

fig_height = raster_height + poprate_height + metric_height * 3 + 0.8
fig = plt.figure(figsize=(FULL_PAGE_WIDTH, fig_height))

# Two major blocks: raster on top, metrics below
gs_top = gridspec.GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=[raster_height + poprate_height, metric_height * 3],
    left=0.07,
    right=0.97,
    top=0.98,
    bottom=0.05,
    hspace=0.1,
)

gs_raster = gridspec.GridSpecFromSubplotSpec(
    2,
    5,
    subplot_spec=gs_top[0],
    height_ratios=[raster_height, poprate_height],
    hspace=0.08,
    wspace=0.3,
)

gs_metrics = gridspec.GridSpecFromSubplotSpec(
    3,
    5,
    subplot_spec=gs_top[1],
    height_ratios=[metric_height, metric_height, metric_height],
    hspace=0.55,
    wspace=0.5,
)

# ═══════════════════════════════════════════════════════════════════════════
# Row A — Raster + population rate
# ═══════════════════════════════════════════════════════════════════════════
cond_titles = {
    "D0": "0 µM",
    "D3": "3 µM",
    "D10": "10 µM",
    "D30": "30 µM",
    "D50": "50 µM",
}
for j, cond in enumerate(CONDITIONS):
    ax_raster = reg(fig.add_subplot(gs_raster[0, j]), "panel", f"A_{cond}_raster")
    ax_poprate = reg(fig.add_subplot(gs_raster[1, j]), "panel", f"A_{cond}_poprate")

    plot_raster_poprate(
        ax_raster,
        ax_poprate,
        raster_data[cond]["sd"],
        raster_data[cond]["pop_rate"],
        raster_data[cond]["burst_edges"],
        sort_indices,
        COLORS[cond],
        start_ms=start_times[cond],
        poprate_ylim=poprate_ylim,
    )

    ax_raster.set_title(
        cond_titles[cond], fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=4
    )

    if j > 0:
        ax_raster.set_ylabel("")
        ax_raster.set_yticklabels([])
        ax_poprate.set_ylabel("")
        ax_poprate.set_yticklabels([])

    if j == 0:
        add_panel_label(ax_raster, "A")

# ═══════════════════════════════════════════════════════════════════════════
# Row B–F — Violin distributions
# ═══════════════════════════════════════════════════════════════════════════
row2_specs = [
    (plot_firing_rate, fr_data),
    (plot_isi_cv, cv_data),
    (plot_pop_coupling, coupling_data),
    (plot_frac_bursts_active, frac_burst_data),
    (plot_frac_spikes_in_burst, frac_spike_data),
]
row2_labels = ["B", "C", "D", "E", "F"]

for j, (plot_fn, data) in enumerate(row2_specs):
    ax = reg(fig.add_subplot(gs_metrics[0, j]), "panel", f"{row2_labels[j]}_violin")
    plot_fn(ax, data)
    add_panel_label(ax, row2_labels[j])

# ═══════════════════════════════════════════════════════════════════════════
# Row G–K — Normalized percentile bands
# ═══════════════════════════════════════════════════════════════════════════
row3_specs = [
    (plot_fr_normalized, fr_data, {"show_legend": True}),
    (plot_isi_cv_normalized, cv_data, {}),
    (plot_pop_coupling_normalized, coupling_data, {}),
    (plot_frac_bursts_active_normalized, frac_burst_data, {}),
    (plot_frac_spikes_in_burst_normalized, frac_spike_data, {}),
]
row3_labels = ["G", "H", "I", "J", "K"]

for j, (plot_fn, data, kwargs) in enumerate(row3_specs):
    ax = reg(fig.add_subplot(gs_metrics[1, j]), "panel", f"{row3_labels[j]}_normalized")
    plot_fn(ax, data, **kwargs)
    add_panel_label(ax, row3_labels[j])

# ═══════════════════════════════════════════════════════════════════════════
# Row L–P — Burst sensitivity, widths, D0-vs-D50 scatters
# ═══════════════════════════════════════════════════════════════════════════
row4_labels = ["L", "M", "N", "O", "P"]

# Panel L: burst sensitivity
ax_bs = reg(fig.add_subplot(gs_metrics[2, 0]), "panel", "L_burst_sensitivity")
plot_burst_sensitivity(ax_bs, thresholds, burst_sensitivity_counts)
add_panel_label(ax_bs, row4_labels[0])

# Panel M: burst widths
ax_bw = reg(fig.add_subplot(gs_metrics[2, 1]), "panel", "M_burst_widths")
plot_burst_widths(ax_bw, burst_widths, burst_counts)
add_panel_label(ax_bw, row4_labels[1])

# Panels N–P: D0 vs D50 scatters
scatter_specs = [
    (plot_pop_coupling_d0_vs_d50, "pop_coupling_zero"),
    (plot_frac_bursts_d0_vs_d50, "frac_bursts_active"),
    (plot_frac_spikes_d0_vs_d50, "frac_spikes_in_burst"),
]

scatter_axes = []
sc_artist = None
for j, (plot_fn, key) in enumerate(scatter_specs):
    ax = reg(
        fig.add_subplot(gs_metrics[2, 2 + j]), "panel", f"{row4_labels[2 + j]}_scatter"
    )
    d0 = ws_get("D0", key)
    d50 = ws_get("D50", key)
    sc_artist = plot_fn(ax, d0, d50, fr_change, "Norm. FR change", show_colorbar=False)
    ax.set_aspect("auto")

    if j >= 1:
        add_panel_label(ax, row4_labels[2 + j], x=-0.08, y=1.05)
    else:
        add_panel_label(ax, row4_labels[2 + j])

    # Panel N: nudge right to clear panel M's secondary y-axis
    if j == 0:
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.01, pos.y0, pos.width - 0.01, pos.height])

    scatter_axes.append(ax)

# Panels O, P: integer ticks (0, 1)
for ax_op in scatter_axes[1:]:
    ax_op.set_xticks([0, 1])
    ax_op.set_xticklabels(["0", "1"])
    ax_op.set_yticks([0, 1])
    ax_op.set_yticklabels(["0", "1"])

# Shared colorbar for N–P
cbar = fig.colorbar(
    sc_artist, ax=scatter_axes, fraction=0.02, pad=0.04, location="right"
)
cbar.set_label("Norm. FR change", fontsize=FONT_SIZES["colorbar_label"])
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(["-1", "0", "1"])
cbar.ax.tick_params(labelsize=FONT_SIZES["tick_label"])
cbar.outline.set_linewidth(0.5)
reg(cbar.ax, "colorbar", "NOP_scatter_colorbar")

# ═══════════════════════════════════════════════════════════════════════════
# Enforce consistent font sizes
# ═══════════════════════════════════════════════════════════════════════════
for entry in axes_registry.values():
    if entry["role"] == "hidden":
        continue
    ax = entry["ax"]
    ax.tick_params(labelsize=UNIFIED_FONTSIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(UNIFIED_FONTSIZE)
    ax.xaxis.label.set_fontsize(UNIFIED_FONTSIZE)
    ax.yaxis.label.set_fontsize(UNIFIED_FONTSIZE)
    if ax.get_title():
        ax.title.set_fontsize(UNIFIED_FONTSIZE)
    for txt in ax.texts:
        if txt.get_fontweight() != "bold":
            txt.set_fontsize(UNIFIED_FONTSIZE)

# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════
save_fig(fig, os.path.join(SCRIPT_DIR, "Figure4_preview.png"))
print("Preview saved as Figure4_preview.png")

from PIL import Image as PILImage

png_path = os.path.join(SCRIPT_DIR, "Figure4_preview.png")
tif_path = os.path.join(SCRIPT_DIR, "Figure4.tif")
img = PILImage.open(png_path).convert("RGB")
img.save(tif_path, format="TIFF", dpi=(900, 900))
print(f"TIFF saved as Figure4.tif ({img.size[0]}x{img.size[1]} px)")
