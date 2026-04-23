"""Assemble Figure 6 — Burst structure and reproducibility.

Layout (3-block composite):
    Block 0 (rows 1-2):  D0 raster + D50 raster + single-unit burst raster
    Block 1 (rows 3-4):  5x5 heatmap grid (left 2/3) + PCA + s2s poprate + s2s corr (right 1/3)
    Block 2 (row 5):     Burst corr heatmap, burst corr violins, rank order heatmap, rank order violins

Prerequisites:
    - compute_shared.py  (spikedata, pop rates, bursts)
    - fig6/compute.py    (burst rss, PCA, s2s corr, rank order)

Usage:
    python -m fig6.assemble     (from the 200123_2953 directory)
    python fig6/assemble.py     (same)

Outputs:
    fig6/Figure6_preview.png    PNG preview
    fig6/Figure6.tif            Submission-ready TIFF
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    save_fig_submission,
    add_panel_label,
    FULL_PAGE_WIDTH,
    COLORS,
    FONT_SIZES,
    CONDITIONS,
)

apply_style()

UNIFIED_FONTSIZE = FONT_SIZES["tick_label"]
INSET_FONTSIZE = UNIFIED_FONTSIZE - 3  # heatmap inset tick labels
INSET_CBAR_FONTSIZE = FONT_SIZES["tick_label"] - 1  # colorbar inset tick labels
mpl.rcParams["axes.labelsize"] = UNIFIED_FONTSIZE
mpl.rcParams["xtick.labelsize"] = UNIFIED_FONTSIZE
mpl.rcParams["ytick.labelsize"] = UNIFIED_FONTSIZE

# ---------------------------------------------------------------------------
# Panel plotting functions (from sibling module)
# ---------------------------------------------------------------------------
from fig6.panels import (
    plot_raster_burst_windows,
    build_combined_spike_slice_stack,
    plot_unit_burst_raster,
    compute_avg_rate_per_condition,
    plot_unit_avg_rate_heatmap,
    PRE_MS,
    POST_MS,
    compute_group0_avg_rates_and_orders,
    plot_avg_burst_heatmap,
    plot_burst_pca,
    compute_closest_burst_edges,
    plot_avg_poprate_with_edges,
    plot_s2s_time_corr,
    plot_burst_corr_heatmap,
    plot_within_condition_violins,
    extract_within_condition_corrs,
    compute_pairwise_ttests,
    add_pvalue_inset,
    plot_rank_order_heatmap,
    plot_rank_order_violins,
    extract_within_condition_values,
)

from spikelab.workspace.hdf5_io import load_workspace_item

# ---------------------------------------------------------------------------
# Axes registry — tracks every axes for consistent font enforcement
# ---------------------------------------------------------------------------
axes_registry = {}  # ax -> {"role": str, "desc": str}


def reg(ax, role, desc):
    """Register an axes with a role and description."""
    axes_registry[id(ax)] = {"role": role, "desc": desc, "ax": ax}
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
results_dir = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
ws_path = os.path.join(results_dir, "workspace")

# Raster data
sd_d0 = load_workspace_item(ws_path, "D0", "spikedata")
sd_d50 = load_workspace_item(ws_path, "D50", "spikedata")
pop_rate_d0 = load_workspace_item(ws_path, "D0", "pop_rate_acc")
pop_rate_d50 = load_workspace_item(ws_path, "D50", "pop_rate_acc")
fr_rates_d0 = load_workspace_item(ws_path, "D0", "fr_rates")
fr_rates_d50 = load_workspace_item(ws_path, "D50", "fr_rates")
tburst_d0 = load_workspace_item(ws_path, "D0", "tburst")
tburst_d50 = load_workspace_item(ws_path, "D50", "tburst")
sort_indices = load_workspace_item(ws_path, "D0", "burst_unit_order")
poprate_ylim = (0, 28)

# Unit burst raster data
sss, boundaries = build_combined_spike_slice_stack(ws_path, CONDITIONS, PRE_MS, POST_MS)
unit_idx = 0
avg_rates = compute_avg_rate_per_condition(
    ws_path, unit_idx, CONDITIONS, PRE_MS, POST_MS
)

# Heatmap grid data
avg_rates_grid, orders_grid, _ = compute_group0_avg_rates_and_orders(
    ws_path, CONDITIONS
)

# PCA data
pca_coords = load_workspace_item(ws_path, "all", "burst_u2u_pca")
cond_idx = load_workspace_item(ws_path, "all", "burst_slice_condition_idx")
var_explained = load_workspace_item(ws_path, "all", "burst_u2u_pca_variance")

# S2S data
s2s_data = {}
sd_burst_data = {}
burst_edge_ranges = {}
for rec in CONDITIONS:
    s2s_data[rec] = load_workspace_item(ws_path, rec, "burst_s2s_time_corr_avg")
    pr = load_workspace_item(ws_path, rec, "pop_rate_acc")
    tb = load_workspace_item(ws_path, rec, "tburst")
    edges = load_workspace_item(ws_path, rec, "burst_edges")
    sd = load_workspace_item(ws_path, rec, "spikedata")
    sd_burst_data[rec] = {
        "sd": sd,
        "pop_rate": pr,
        "tburst": tb,
        "burst_edges": edges,
    }
    burst_edge_ranges[rec] = compute_closest_burst_edges(tb, edges)

# Bottom row data
burst_counts = [len(load_workspace_item(ws_path, rec, "tburst")) for rec in CONDITIONS]
corr_stack = load_workspace_item(ws_path, "all", "burst_corr_per_unit").stack
within_corrs = extract_within_condition_corrs(corr_stack, burst_counts)
pval_corr, sig_corr, _ = compute_pairwise_ttests(within_corrs)

rank_item = load_workspace_item(ws_path, "all", "burst_rank_rate_corr")
rank_matrix = (
    rank_item.matrix if hasattr(rank_item, "matrix") else np.asarray(rank_item)
)
within_rank = extract_within_condition_values(rank_matrix, burst_counts)
pval_rank, sig_rank, _ = compute_pairwise_ttests(within_rank)

# ═══════════════════════════════════════════════════════════════════════════
# Figure layout
# ═══════════════════════════════════════════════════════════════════════════
row_heights = [2.5, 2.5, 2.0, 2.0, 1.5]
fig_height = sum(row_heights) + 1.5
fig = plt.figure(figsize=(FULL_PAGE_WIDTH, fig_height))

gs_main = gridspec.GridSpec(
    3,
    1,
    figure=fig,
    height_ratios=[sum(row_heights[:2]), sum(row_heights[2:4]), row_heights[4]],
    left=0.06,
    right=0.96,
    top=0.98,
    bottom=0.04,
    hspace=0.16,
)

# ═══════════════════════════════════════════════════════════════════════════
# Block 0: rows 1-2 — raster panels
# ═══════════════════════════════════════════════════════════════════════════
gs_block0 = gridspec.GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gs_main[0],
    wspace=0.3,
)


def make_raster_axes(gs_slot, fig, prefix):
    """Create raster+heatmap+poprate axes in a slot and register them."""
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs_slot,
        height_ratios=[3, 2, 1],
        width_ratios=[1, 0.03],
        hspace=0.05,
        wspace=0.05,
    )
    ax_r = reg(fig.add_subplot(gs_inner[0, 0]), "panel", f"{prefix}_raster")
    ax_h = reg(
        fig.add_subplot(gs_inner[1, 0], sharex=ax_r), "panel", f"{prefix}_heatmap"
    )
    ax_p = reg(
        fig.add_subplot(gs_inner[2, 0], sharex=ax_r), "panel", f"{prefix}_poprate"
    )
    ax_cb = reg(fig.add_subplot(gs_inner[1, 1]), "colorbar", f"{prefix}_heatmap_cbar")
    for row in [0, 2]:
        ax_e = fig.add_subplot(gs_inner[row, 1])
        ax_e.axis("off")
        reg(ax_e, "hidden", f"{prefix}_empty_{row}")
    return ax_r, ax_h, ax_p, ax_cb


# D0 raster
ax_r0, ax_h0, ax_p0, ax_cb0 = make_raster_axes(gs_block0[0], fig, "A_d0")
plot_raster_burst_windows(
    ax_r0,
    ax_h0,
    ax_p0,
    ax_cb0,
    sd_d0,
    pop_rate_d0,
    fr_rates_d0,
    tburst_d0,
    sort_indices,
    COLORS["D0"],
    start_ms=32500,
    time_window_ms=9000,
    poprate_ylim=poprate_ylim,
    heatmap_vmax=100,
)
ax_r0.set_title("0 \u00b5M", fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=4)
add_panel_label(ax_r0, "A")

# D50 raster (hide its colorbar, share D0's)
ax_r50, ax_h50, ax_p50, ax_cb50 = make_raster_axes(gs_block0[1], fig, "A_d50")
plot_raster_burst_windows(
    ax_r50,
    ax_h50,
    ax_p50,
    ax_cb50,
    sd_d50,
    pop_rate_d50,
    fr_rates_d50,
    tburst_d50,
    sort_indices,
    COLORS["D50"],
    start_ms=20500,
    time_window_ms=9000,
    poprate_ylim=poprate_ylim,
    heatmap_vmax=100,
)
ax_r50.set_title("50 \u00b5M", fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=4)
ax_cb50.set_visible(False)
for ax in [ax_r50, ax_h50, ax_p50]:
    ax.set_ylabel("")
    ax.set_yticklabels([])

# Unit burst raster
gs_ubr = gridspec.GridSpecFromSubplotSpec(
    2,
    2,
    subplot_spec=gs_block0[2],
    height_ratios=[5, 1],
    width_ratios=[1, 0.03],
    hspace=0.08,
    wspace=0.08,
)
ax_ubr = reg(fig.add_subplot(gs_ubr[0, 0]), "panel", "B_raster")
ax_ubr_heat = reg(fig.add_subplot(gs_ubr[1, 0], sharex=ax_ubr), "panel", "B_heatmap")
ax_ubr_cbar = reg(fig.add_subplot(gs_ubr[1, 1]), "colorbar", "B_heatmap_cbar")
ax_ubr_empty = fig.add_subplot(gs_ubr[0, 1])
ax_ubr_empty.axis("off")
reg(ax_ubr_empty, "hidden", "B_empty")

plot_unit_burst_raster(ax_ubr, sss, unit_idx, boundaries)
ax_ubr.set_xlabel("")
ax_ubr.tick_params(labelbottom=False)
plot_unit_avg_rate_heatmap(ax_ubr_heat, ax_ubr_cbar, avg_rates)

# Scale bar below B, outside axes
ax_ubr_heat.set_xlabel("Time from burst peak (ms)", fontsize=UNIFIED_FONTSIZE)
ax_ubr_heat.set_xticks([])
ax_ubr_heat.annotate(
    "",
    xy=(0.98, -0.08),
    xytext=(0.98 - 100 / (POST_MS + PRE_MS), -0.08),
    xycoords="axes fraction",
    arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
)
ax_ubr_heat.text(
    0.98 - 50 / (POST_MS + PRE_MS),
    -0.18,
    "100 ms",
    transform=ax_ubr_heat.transAxes,
    ha="center",
    va="top",
    fontsize=UNIFIED_FONTSIZE,
)

add_panel_label(ax_ubr, "B", y=1.02)

# ═══════════════════════════════════════════════════════════════════════════
# Block 1: rows 3-4 — heatmap grid + PCA + s2s
# ═══════════════════════════════════════════════════════════════════════════
gs_block1 = gridspec.GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gs_main[1],
    width_ratios=[1, 1, 1],
    wspace=0.3,
)

# Heatmap grid (spans 2 columns)
gs_heatgrid = gridspec.GridSpecFromSubplotSpec(
    5,
    6,
    subplot_spec=gs_block1[0:2],
    width_ratios=[1, 1, 1, 1, 1, 0.05],
    hspace=0.15,
    wspace=0.15,
)

global_vmax = max(np.nanmax(avg_rates_grid[rec]) for rec in CONDITIONS)
n_group0 = avg_rates_grid[CONDITIONS[0]].shape[0]
im_grid = None
grid_axes = {}
for row_idx, order_rec in enumerate(CONDITIONS):
    for col_idx, data_rec in enumerate(CONDITIONS):
        ax = reg(
            fig.add_subplot(gs_heatgrid[row_idx, col_idx]),
            "panel",
            f"C_grid_{order_rec}_{data_rec}",
        )
        grid_axes[(row_idx, col_idx)] = ax
        im_grid = plot_avg_burst_heatmap(
            ax, avg_rates_grid[data_rec], orders_grid[order_rec]
        )

        if col_idx == 0:
            ax.set_ylabel(f"{order_rec} order", fontsize=UNIFIED_FONTSIZE)
            ax.set_yticks([0, n_group0 - 1])
            ax.set_yticklabels([str(n_group0), "1"])
        else:
            ax.set_yticks([])

        if row_idx == 4:
            ax.set_xlabel(f"{data_rec} activity", fontsize=UNIFIED_FONTSIZE)
            ax.set_xticks([])
        if row_idx == 0 and col_idx == 0:
            bar_frac = 200 / (POST_MS + PRE_MS)
            ax.annotate(
                "",
                xy=(0.98, 1.08),
                xytext=(0.98 - bar_frac, 1.08),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
            )
            ax.text(
                0.98 - bar_frac / 2,
                1.15,
                "200 ms",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=UNIFIED_FONTSIZE,
            )
        else:
            ax.set_xticks([])

# Hide unused colorbar column cells
for row in range(5):
    ax_e = fig.add_subplot(gs_heatgrid[row, 5])
    ax_e.axis("off")
    reg(ax_e, "hidden", f"C_grid_empty_{row}")

# Horizontal colorbar above the last heatmap column
ax_hg_cbar_pos = grid_axes[(0, 4)]
cbar_hg_ax = ax_hg_cbar_pos.inset_axes([0.0, 1.15, 1.0, 0.08])
reg(cbar_hg_ax, "colorbar", "C_grid_cbar")
cbar_hg = fig.colorbar(im_grid, cax=cbar_hg_ax, orientation="horizontal")
cbar_hg.set_ticks([])
cbar_hg.outline.set_linewidth(0.5)
cbar_hg_ax.set_title("Norm. rate", fontsize=FONT_SIZES["colorbar_label"], pad=2)
cbar_hg_ax.text(
    -0.05,
    0.5,
    "0",
    transform=cbar_hg_ax.transAxes,
    ha="right",
    va="center",
    fontsize=UNIFIED_FONTSIZE,
)
cbar_hg_ax.text(
    1.05,
    0.5,
    "1",
    transform=cbar_hg_ax.transAxes,
    ha="left",
    va="center",
    fontsize=UNIFIED_FONTSIZE,
)

# Label C
hg_first_ax = grid_axes[(0, 0)]
add_panel_label(hg_first_ax, "C", x=-0.3, y=1.1)

# PCA + s2s in right column
gs_right = gridspec.GridSpecFromSubplotSpec(
    3,
    1,
    subplot_spec=gs_block1[2],
    height_ratios=[1, 1, 1],
    hspace=0.25,
)

ax_pca = reg(fig.add_subplot(gs_right[0]), "panel", "D_pca")
plot_burst_pca(ax_pca, pca_coords, cond_idx, var_explained)
ax_pca.xaxis.set_label_position("top")
ax_pca.xaxis.tick_top()
ax_pca.spines["bottom"].set_visible(False)
ax_pca.spines["top"].set_visible(True)
ax_pca.legend(
    fontsize=FONT_SIZES["legend"],
    handlelength=0.8,
    handletextpad=0.3,
    columnspacing=0.5,
    markerscale=0.8,
    loc="upper left",
    ncol=5,
    borderpad=0.2,
    bbox_to_anchor=(0.0, -0.02),
)
add_panel_label(ax_pca, "D")

ax_s2s_pop = reg(fig.add_subplot(gs_right[1]), "panel", "E_poprate")
ax_s2s_corr = reg(
    fig.add_subplot(gs_right[2], sharex=ax_s2s_pop), "panel", "F_similarity"
)
plot_avg_poprate_with_edges(ax_s2s_pop, sd_burst_data)
ax_s2s_pop.set_ylim(bottom=0)
plot_s2s_time_corr(ax_s2s_corr, s2s_data, burst_edge_ranges)

ax_s2s_pop.set_xticks([])
ax_s2s_pop.set_xlabel("")
ax_s2s_corr.set_xticks([])

ax_s2s_corr.get_legend().remove()
xlim = ax_s2s_pop.get_xlim()
ylim_e = ax_s2s_pop.get_ylim()
ax_s2s_pop.legend(
    *ax_s2s_corr.get_legend_handles_labels(),
    fontsize=FONT_SIZES["legend"],
    handlelength=1.0,
    handletextpad=0.3,
    columnspacing=0.5,
    loc="upper left",
    ncol=5,
    borderpad=0.2,
    bbox_to_anchor=(0.0, -0.02),
)

# Scale bar below E
bar_x_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
bar_x_start = bar_x_end - 100
bar_y = ylim_e[0] - (ylim_e[1] - ylim_e[0]) * 0.08
ax_s2s_pop.plot(
    [bar_x_start, bar_x_end],
    [bar_y, bar_y],
    color="black",
    linewidth=1.5,
    clip_on=False,
    solid_capstyle="butt",
)
ax_s2s_pop.text(
    (bar_x_start + bar_x_end) / 2,
    bar_y - (ylim_e[1] - ylim_e[0]) * 0.06,
    "100 ms",
    ha="center",
    va="top",
    fontsize=UNIFIED_FONTSIZE,
)

add_panel_label(ax_s2s_pop, "E")

# ═══════════════════════════════════════════════════════════════════════════
# Block 2: row 5 — correlation panels
# ═══════════════════════════════════════════════════════════════════════════
gs_block2 = gridspec.GridSpecFromSubplotSpec(
    1,
    4,
    subplot_spec=gs_main[2],
    wspace=0.45,
)

ax_bch = reg(fig.add_subplot(gs_block2[0]), "panel", "G_corr_heatmap")
plot_burst_corr_heatmap(ax_bch, corr_stack, burst_counts, cbar_ticks=[0.7, 0.8])
add_panel_label(ax_bch, "F", y=1.15)

ax_bcv = reg(fig.add_subplot(gs_block2[1]), "panel", "H_corr_violins")
plot_within_condition_violins(ax_bcv, within_corrs)
ax_h_inset = add_pvalue_inset(ax_bcv, pval_corr, sig_corr)
reg(ax_h_inset, "inset", "H_pvalue_heatmap")
# Register inset colorbar (last child of ax_h_inset)
for child in ax_h_inset.child_axes:
    reg(child, "inset", "H_pvalue_cbar")
pos_h = ax_bcv.get_position()
ax_bcv.set_position([pos_h.x0 + 0.0235, pos_h.y0, pos_h.width, pos_h.height])
add_panel_label(ax_bcv, "G")

ax_roh = reg(fig.add_subplot(gs_block2[2]), "panel", "I_rank_heatmap")
rank_matrix_diag = rank_matrix.copy()
np.fill_diagonal(rank_matrix_diag, 7)
plot_rank_order_heatmap(
    ax_roh, rank_matrix_diag, burst_counts, "Rank order corr. (zscore)", vmin=0, vmax=7
)
# Nudge rank heatmap and its colorbar left
pos_roh = ax_roh.get_position()
ax_roh.set_position([pos_roh.x0 - 0.0065, pos_roh.y0, pos_roh.width, pos_roh.height])
# The colorbar was auto-created by plot_rank_order_heatmap — find and nudge it too
for child_ax in fig.axes:
    if child_ax not in [v["ax"] for v in axes_registry.values()] and child_ax is not ax_roh:
        pos_c = child_ax.get_position()
        if abs(pos_c.y0 - pos_roh.y0) < 0.05 and pos_c.x0 > pos_roh.x0:
            child_ax.set_position([pos_c.x0 - 0.0065, pos_c.y0, pos_c.width, pos_c.height])
            break
add_panel_label(ax_roh, "H", y=1.15)

ax_rov = reg(fig.add_subplot(gs_block2[3]), "panel", "J_rank_violins")
plot_rank_order_violins(ax_rov, within_rank)

# J p-value inset
ax_j_inset = inset_axes(
    ax_rov,
    width="30%",
    height="30%",
    loc="lower left",
    bbox_to_anchor=(0.08, 0.05, 1, 1),
    bbox_transform=ax_rov.transAxes,
    borderpad=1.0,
)
reg(ax_j_inset, "inset", "J_pvalue_heatmap")

K = len(CONDITIONS)
neg_log_p = -np.log10(pval_rank)
finite_vals = neg_log_p[np.isfinite(neg_log_p) & ~np.isnan(neg_log_p)]
vmax_p = np.max(finite_vals) if len(finite_vals) > 0 else 1
neg_log_p = np.where(np.isfinite(neg_log_p), neg_log_p, vmax_p)
np.fill_diagonal(neg_log_p, np.nan)
cmap_p = plt.cm.viridis.copy()
cmap_p.set_bad(color="black")
im_p = ax_j_inset.imshow(
    neg_log_p, cmap=cmap_p, aspect="equal", interpolation="none", vmin=0, vmax=vmax_p
)
for i in range(K):
    for j in range(K):
        if i != j and sig_rank[i, j]:
            ax_j_inset.plot(j, i, "o", color="red", markersize=2.5, markeredgewidth=0)
tick_labels = [r.replace("D", "") for r in CONDITIONS]
ax_j_inset.set_xticks(range(K))
ax_j_inset.set_xticklabels(tick_labels)
ax_j_inset.set_yticks(range(K))
ax_j_inset.set_yticklabels(tick_labels)
for spine in ax_j_inset.spines.values():
    spine.set_linewidth(0.5)

cbar_j_inset = inset_axes(
    ax_j_inset,
    width="8%",
    height="100%",
    loc="center right",
    bbox_to_anchor=(0.18, 0, 1, 1),
    bbox_transform=ax_j_inset.transAxes,
    borderpad=0,
)
reg(cbar_j_inset, "inset", "J_pvalue_cbar")
cbar_p = fig.colorbar(im_p, cax=cbar_j_inset)
cbar_p.outline.set_linewidth(0.5)
cbar_p.ax.tick_params(labelsize=FONT_SIZES["tick_label"] - 1, width=0.5, length=1.5)
cbar_p.set_label("$-\\log_{10}(\\mathrm{P})$", fontsize=FONT_SIZES["colorbar_label"])

add_panel_label(ax_rov, "I")

# ═══════════════════════════════════════════════════════════════════════════
# Enforce consistent font sizes using axes registry
# ═══════════════════════════════════════════════════════════════════════════
for entry in axes_registry.values():
    ax = entry["ax"]
    role = entry["role"]

    if role == "hidden":
        continue

    if role == "inset" and entry["desc"].endswith("_cbar"):
        fs = INSET_CBAR_FONTSIZE
    elif role == "inset":
        fs = INSET_FONTSIZE
    else:
        fs = UNIFIED_FONTSIZE

    if role == "inset" and entry["desc"].endswith("_cbar"):
        ax.xaxis.label.set_fontsize(FONT_SIZES["colorbar_label"])
        ax.yaxis.label.set_fontsize(FONT_SIZES["colorbar_label"])
    else:
        ax.xaxis.label.set_fontsize(fs)
        ax.yaxis.label.set_fontsize(fs)
    if role == "inset":
        ax.tick_params(labelsize=fs, width=0.5, length=1.5)
    else:
        ax.tick_params(labelsize=fs)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fs)
    if ax.get_title():
        ax.title.set_fontsize(fs)
    for txt in ax.texts:
        if txt.get_fontweight() != "bold":
            txt.set_fontsize(fs)

# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════
save_fig(fig, os.path.join(SCRIPT_DIR, "Figure6_preview.png"))
save_fig_submission(fig, os.path.join(SCRIPT_DIR, "Figure6.tif"))
print("Preview saved as Figure6_preview.png")
print("Submission TIFF saved as Figure6.tif")
