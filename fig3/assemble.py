"""Assemble Figure 3 — Different data sources: in vivo and organoid recording analysis.

Layout:
  Row 1 (3 cols):  A: Mouse raster    B: Human raster    C: Organoid raster
  Row 2 (4 cols):  D: Mouse unit avg  E: Org unit avg    F: Pop coupling     I: Cue vs no-cue
                                                          G: Norm bands       J: Raw vs z-score
                                                          H: Burst rate       K: Burst sensitivity

Prerequisites:
    - fig3/compute.py  (workspace + .npz files)

Usage:
    python -m fig3.assemble     (from the figure_code directory)

Output:
    fig3/Figure3_preview.png    PNG preview
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style, save_fig, save_fig_submission, add_panel_label,
    FULL_PAGE_WIDTH, FONT_SIZES,
)
apply_style()

from spikelab.workspace.hdf5_io import load_workspace_item

from fig3.panels import (
    plot_raster_panel,
    plot_unit_raster_psth,
    plot_coupling_panels,
    plot_cue_vs_nocue,
    plot_raw_vs_zscore,
    plot_burst_sensitivity,
    reg,
)

# ── Data paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(REPO_ROOT, "manuscript", "data", "different_samples")
WS_PATH = os.path.join(DATA_DIR, "workspace")

# ── Load all data ─────────────────────────────────────────────────────────
NAMESPACES = ["mouse_nocue", "human_nocue", "D0"]
raster_data = {}
for ns in NAMESPACES:
    sd = load_workspace_item(WS_PATH, ns, "spikedata")
    # Compute fr_rates on the fly (removed from workspace to save 11 GB)
    times = np.arange(0, sd.length, 1.0)
    fr_rates = sd.resampled_isi(times)
    print(f"  {ns}: computed fr_rates {fr_rates.inst_Frate_data.shape}")
    raster_data[ns] = {
        "sd": sd,
        "pop_rate": load_workspace_item(WS_PATH, ns, "pop_rate_acc"),
        "fr_rates": fr_rates,
        "tburst": load_workspace_item(WS_PATH, ns, "tburst"),
        "burst_edges": load_workspace_item(WS_PATH, ns, "burst_edges"),
    }

sd_mouse_cue = load_workspace_item(WS_PATH, "mouse_cue", "spikedata")
sd_human_cue = load_workspace_item(WS_PATH, "human_cue", "spikedata")
sd_human_nocue = load_workspace_item(WS_PATH, "human_nocue", "spikedata")
tburst_d0 = load_workspace_item(WS_PATH, "D0", "tburst")
tburst_human_cue = load_workspace_item(WS_PATH, "human_cue", "tburst")
tburst_human_nocue = load_workspace_item(WS_PATH, "human_nocue", "tburst")

coupling_data = load_workspace_item(WS_PATH, "human_nocue", "coupling_over_time_full")
shuffle_data = load_workspace_item(WS_PATH, "human_nocue", "coupling_shuffle")

# Burst sensitivity: reassemble from per-recording keys
sensitivity_data = {
    "thresholds": load_workspace_item(WS_PATH, "human_nocue", "burst_sensitivity")["thresholds"],
}
for ns in NAMESPACES + ["human_cue", "mouse_cue"]:
    bs = load_workspace_item(WS_PATH, ns, "burst_sensitivity")
    sensitivity_data[f"counts_{ns}"] = bs["counts"]
    sensitivity_data[f"duration_min_{ns}"] = bs["duration_min"]

first_cue_s = sd_human_cue.metadata["all_cue_onset_times"][0] / 1000
cue_boundary = float(coupling_data["cue_boundary_s"])

# ══════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig = plt.figure(figsize=(FULL_PAGE_WIDTH, 10))

    gs_top = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.9, 1],
        left=0.05, right=0.97, top=0.98, bottom=0.04,
        hspace=0.12,
    )

    # ── Row 1: 3 raster panels ──
    gs_row1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs_top[0],
        width_ratios=[1, 0.08, 1, 1], wspace=0.12,
    )

    COLORS = {"mouse_nocue": "#B2182B", "human_nocue": "#2166AC", "D0": "#1B7837"}

    print("Row 1: Recording overviews...")
    d = raster_data["mouse_nocue"]
    ax_A, ax_A_pop = plot_raster_panel(
        gs_row1[0], fig, d["sd"], d["pop_rate"], d["fr_rates"],
        d["tburst"], d["burst_edges"],
        COLORS["mouse_nocue"], 30000, 30000, ns_label="mouse_nocue",
    )
    ax_spacer = fig.add_subplot(gs_row1[1]); ax_spacer.axis("off")

    d = raster_data["human_nocue"]
    ax_B, ax_B_pop = plot_raster_panel(
        gs_row1[2], fig, d["sd"], d["pop_rate"], d["fr_rates"],
        d["tburst"], d["burst_edges"],
        COLORS["human_nocue"], 30000, 30000, ns_label="human_nocue",
        show_colorbar=False, show_ylabels=False,
    )

    d = raster_data["D0"]
    ax_C, ax_C_pop = plot_raster_panel(
        gs_row1[3], fig, d["sd"], d["pop_rate"], d["fr_rates"],
        d["tburst"], d["burst_edges"],
        COLORS["D0"], 30000, 30000, ns_label="D0",
        show_colorbar=False, show_ylabels=False,
    )

    ax_A_pop.yaxis.set_label_coords(-0.16, 0.5)

    add_panel_label(ax_A, "A")
    add_panel_label(ax_B, "B")
    add_panel_label(ax_C, "C")

    # ── Row 2: 4 columns ──
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs_top[1],
        width_ratios=[1, 1, 1.4, 1],
        wspace=0.3,
    )

    # D: Mouse unit event average
    print("Row 2: Mouse unit 036...")
    stim_on = np.asarray(sd_mouse_cue.metadata["stim_on_times"]).ravel()
    feedback = np.asarray(sd_mouse_cue.metadata["feedback_type"]).ravel()
    ax_D = plot_unit_raster_psth(
        gs_row2[0], fig, sd_mouse_cue, 36, stim_on, "Stimulus",
        pre_ms=200, post_ms=1500, ns_label="mouse_cue",
        feedback_split=True, feedback_type=feedback,
    )
    add_panel_label(ax_D, "D")

    # E: Organoid unit event average
    print("Row 2: Organoid unit 085...")
    ax_E = plot_unit_raster_psth(
        gs_row2[1], fig, raster_data["D0"]["sd"], 85, tburst_d0, "Burst",
        pre_ms=200, post_ms=500, ns_label="D0",
    )
    add_panel_label(ax_E, "E")

    # F/G/H: Pop coupling panels
    print("Row 2: Pop coupling...")
    ax_F, ax_G, ax_H = plot_coupling_panels(
        gs_row2[2], fig, coupling_data,
        tburst_human_cue, tburst_human_nocue,
        first_cue_s, cue_boundary,
    )
    add_panel_label(ax_F, "F")
    add_panel_label(ax_G, "G")
    add_panel_label(ax_H, "H")

    # I/J/K: Right column
    pos = gs_row2[3].get_position(fig)
    gs_right = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1, 1, 1], hspace=0.3,
        left=pos.x0, right=pos.x1,
        top=pos.y1 + 0.03, bottom=pos.y0,
    )

    print("Row 2: Cue vs no-cue...")
    ax_I = plot_cue_vs_nocue(gs_right[0], fig, coupling_data)
    add_panel_label(ax_I, "I")

    print("Row 2: Raw vs z-score...")
    ax_J = plot_raw_vs_zscore(gs_right[1], fig, shuffle_data, sd_human_nocue)
    add_panel_label(ax_J, "J")

    print("Row 2: Burst sensitivity...")
    ax_K = reg(fig.add_subplot(gs_right[2]), "panel", "K_sensitivity")
    plot_burst_sensitivity(ax_K, sensitivity_data)
    add_panel_label(ax_K, "K")

    # ── Save ──
    save_fig(fig, os.path.join(SCRIPT_DIR, "Figure3_preview.png"))
    print("Saved preview PNG")

    save_fig_submission(fig, os.path.join(SCRIPT_DIR, "Figure3.tif"))
    print("Saved: Figure3.tif")
