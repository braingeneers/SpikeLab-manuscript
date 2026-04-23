"""Assemble Figure 5 — Pairwise correlations and network structure.

Layout:
    Row A:          5 raster + FR heatmap panels (full width)
    Left 2/3:       FR correlation matrices, STTC matrices, FR vs STTC scatter
    Right 1/3:      FR correlation violins + normalized bands
    Bottom:         MEA network (left, tall) + 2x3 graph metrics (right)

Prerequisites:
    - compute_shared.py  (spikedata, fr_rates)
    - fig5/compute.py    (fr_corr_matrix, sttc_matrix)

Usage:
    python -m fig5.assemble     (from the 200123_2953 directory)
    python fig5/assemble.py     (same)

Outputs:
    fig5/Figure5_preview.png    PNG preview
    fig5/Figure5.tif            Submission-ready TIFF
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
from matplotlib.lines import Line2D

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
    CMAPS,
)

apply_style()

UNIFIED_FONTSIZE = FONT_SIZES["tick_label"]
mpl.rcParams["axes.labelsize"] = UNIFIED_FONTSIZE
mpl.rcParams["xtick.labelsize"] = UNIFIED_FONTSIZE
mpl.rcParams["ytick.labelsize"] = UNIFIED_FONTSIZE

from fig5.panels import (
    plot_raster_fr_on_axes,
    get_fr_corr_sort_indices,
    plot_corr_matrix,
    make_scatter_with_marginals,
    plot_pairwise_corr_violins,
    plot_pairwise_corr_normalized,
    weighted_rich_club,
)

from spikelab.workspace.hdf5_io import load_workspace_item
from spikelab.spikedata.plot_utils import (
    plot_distribution,
    plot_percentile_bands,
    plot_spatial_network,
)

import networkx as nx
from community import community_louvain

# ---------------------------------------------------------------------------
# Axes registry
# ---------------------------------------------------------------------------
axes_registry = {}


def reg(ax, role, desc):
    axes_registry[id(ax)] = {"role": role, "desc": desc, "ax": ax}
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
results_dir = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
ws_path = os.path.join(results_dir, "workspace")


def ws_get(ns, key):
    return load_workspace_item(ws_path, ns, key)


# Row A
raster_data = {}
for cond in CONDITIONS:
    raster_data[cond] = {
        "sd": ws_get(cond, "spikedata"),
        "fr_rates": ws_get(cond, "fr_rates"),
    }

fr_corr_d0 = ws_get("D0", "fr_corr_matrix")
sort_indices = get_fr_corr_sort_indices(fr_corr_d0)

# Raster sort: high-to-low mean FR corr (y-axis inverted, unit 0 at top)
mat_tmp = fr_corr_d0.matrix.copy()
np.fill_diagonal(mat_tmp, np.nan)
raster_sort = np.argsort(np.nanmean(mat_tmp, axis=1))[::-1]

fr_corr_d50 = ws_get("D50", "fr_corr_matrix")
sttc_d0 = ws_get("D0", "sttc_matrix")
sttc_d50 = ws_get("D50", "sttc_matrix")

mat_fr_d0 = fr_corr_d0.matrix
mat_fr_d50 = fr_corr_d50.matrix
mat_sttc_d0 = sttc_d0.matrix
mat_sttc_d50 = sttc_d50.matrix

corr_data = {rec: ws_get(rec, "fr_corr_matrix").matrix for rec in CONDITIONS}

start_times = {"D0": 5000, "D3": 1500, "D10": 0, "D30": 2700, "D50": 500}
fr_vlim = (0, 100)

# ═══════════════════════════════════════════════════════════════════════════
# Figure layout
# ═══════════════════════════════════════════════════════════════════════════
raster_height = 2.2
fr_height = 2.2
matrix_row_height = 1.6
scatter_row_height = 1.4
lower_height = matrix_row_height * 2 + scatter_row_height
graph_block_height = 3.8

fig_height = raster_height + fr_height + lower_height + graph_block_height + 0.7
fig = plt.figure(figsize=(FULL_PAGE_WIDTH, fig_height))

block1_top = 0.98
block1_bot = block1_top - (raster_height + fr_height) / fig_height
gap_ab = 0.005
block2_top = block1_bot - gap_ab
block2_bot = block2_top - lower_height / fig_height
gap_gm = 0.04
block3_top = block2_bot - gap_gm
block3_bot = 0.03

gs_top = [
    gridspec.GridSpec(1, 1, figure=fig, left=0.06, right=0.97, top=block1_top, bottom=block1_bot),
    gridspec.GridSpec(1, 1, figure=fig, left=0.06, right=0.97, top=block2_top, bottom=block2_bot),
    gridspec.GridSpec(1, 1, figure=fig, left=0.06, right=0.97, top=block3_top, bottom=block3_bot),
]

# ═══════════════════════════════════════════════════════════════════════════
# Row A — Raster + FR heatmap
# ═══════════════════════════════════════════════════════════════════════════
gs_raster = gridspec.GridSpecFromSubplotSpec(
    2, 6, subplot_spec=gs_top[0][0, 0],
    height_ratios=[raster_height, fr_height],
    width_ratios=[1, 1, 1, 1, 1, 0.05],
    hspace=0.08, wspace=0.3,
)

cond_titles = {"D0": "0 µM", "D3": "3 µM", "D10": "10 µM", "D30": "30 µM", "D50": "50 µM"}
last_ax_fr = None
for j, cond in enumerate(CONDITIONS):
    ax_r = reg(fig.add_subplot(gs_raster[0, j]), "panel", f"A_{cond}_raster")
    ax_f = reg(fig.add_subplot(gs_raster[1, j]), "panel", f"A_{cond}_fr")

    plot_raster_fr_on_axes(
        ax_r, ax_f, raster_data[cond]["sd"], raster_data[cond]["fr_rates"],
        raster_sort, start_ms=start_times[cond], fr_vlim=fr_vlim,
    )
    last_ax_fr = ax_f

    ax_r.set_title(cond_titles[cond], fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=4)
    if j > 0:
        for ax in [ax_r, ax_f]:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    if j == 0:
        add_panel_label(ax_r, "A")

# FR heatmap colorbar
ax_cbar_top = reg(fig.add_subplot(gs_raster[0, 5]), "hidden", "A_cbar_empty")
ax_cbar_top.axis("off")
cax_fr = reg(fig.add_subplot(gs_raster[1, 5]), "colorbar", "A_fr_cbar")
cbar_fr = fig.colorbar(last_ax_fr.images[0], cax=cax_fr)
cbar_fr.set_label("Rate (Hz)", fontsize=FONT_SIZES["colorbar_label"])
cbar_fr.outline.set_linewidth(0.5)
cbar_fr.ax.tick_params(width=0.5, length=2)

print("Row A done")

# ═══════════════════════════════════════════════════════════════════════════
# Lower block — matrices, scatter, violins
# ═══════════════════════════════════════════════════════════════════════════
gs_lower = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_top[1][0, 0], width_ratios=[2, 1], wspace=0.2,
)

gs_left = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_lower[0],
    height_ratios=[matrix_row_height, matrix_row_height, scatter_row_height],
    hspace=0.02,
)
gs_right = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_lower[1], height_ratios=[0.1, 1, 1], hspace=0.3,
)

# --- FR correlation matrices ---
gs_fr_mat = gridspec.GridSpecFromSubplotSpec(
    1, 5, subplot_spec=gs_left[0], width_ratios=[1, 1, 0.07, 1, 0.07], wspace=0.5,
)

mat_fr_diff = mat_fr_d0 - mat_fr_d50

ax_fr0 = reg(fig.add_subplot(gs_fr_mat[0, 0]), "panel", "B_fr_d0")
im_fr = plot_corr_matrix(ax_fr0, mat_fr_d0, sort_indices, vmin=0, vmax=1, cmap="viridis")
ax_fr0.set_title("D0", fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=8)
add_panel_label(ax_fr0, "B")

ax_fr50 = reg(fig.add_subplot(gs_fr_mat[0, 1]), "panel", "B_fr_d50")
plot_corr_matrix(ax_fr50, mat_fr_d50, sort_indices, vmin=0, vmax=1, cmap="viridis")
ax_fr50.set_title("D50", fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=8)

cax_fr_mat = reg(fig.add_subplot(gs_fr_mat[0, 2]), "colorbar", "B_fr_cbar")
cbar = fig.colorbar(im_fr, cax=cax_fr_mat)
cbar.set_label("FR correlation", fontsize=FONT_SIZES["colorbar_label"], rotation=270, labelpad=10)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(width=0.5, length=2)
cbar.set_ticks([0, 0.5, 1])

ax_frd = reg(fig.add_subplot(gs_fr_mat[0, 3]), "panel", "B_fr_diff")
ax_frd.set_title("Difference", fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=8)
im_frd = plot_corr_matrix(ax_frd, mat_fr_diff, sort_indices, vmin=-0.5, vmax=0.5, cmap=CMAPS["diverging"])

cax_frd = reg(fig.add_subplot(gs_fr_mat[0, 4]), "colorbar", "B_fr_diff_cbar")
cbar_d = fig.colorbar(im_frd, cax=cax_frd)
cbar_d.set_label("Δ FR corr.", fontsize=FONT_SIZES["colorbar_label"], rotation=270, labelpad=10)
cbar_d.outline.set_linewidth(0.5)
cbar_d.ax.tick_params(width=0.5, length=2)
cbar_d.set_ticks([-0.5, 0, 0.5])

print("Row B done")

# --- STTC matrices ---
gs_sttc_mat = gridspec.GridSpecFromSubplotSpec(
    1, 5, subplot_spec=gs_left[1], width_ratios=[1, 1, 0.07, 1, 0.07], wspace=0.5,
)
mat_sttc_diff = mat_sttc_d0 - mat_sttc_d50

ax_s0 = reg(fig.add_subplot(gs_sttc_mat[0, 0]), "panel", "C_sttc_d0")
im_s = plot_corr_matrix(ax_s0, mat_sttc_d0, sort_indices, vmin=0, vmax=1, cmap="viridis")
add_panel_label(ax_s0, "C")

ax_s50 = reg(fig.add_subplot(gs_sttc_mat[0, 1]), "panel", "C_sttc_d50")
plot_corr_matrix(ax_s50, mat_sttc_d50, sort_indices, vmin=0, vmax=1, cmap="viridis")

cax_s = reg(fig.add_subplot(gs_sttc_mat[0, 2]), "colorbar", "C_sttc_cbar")
cb_s = fig.colorbar(im_s, cax=cax_s)
cb_s.set_label("STTC", fontsize=FONT_SIZES["colorbar_label"], rotation=270, labelpad=10)
cb_s.outline.set_linewidth(0.5)
cb_s.ax.tick_params(width=0.5, length=2)
cb_s.set_ticks([0, 0.5, 1])

ax_sd = reg(fig.add_subplot(gs_sttc_mat[0, 3]), "panel", "C_sttc_diff")
im_sd = plot_corr_matrix(ax_sd, mat_sttc_diff, sort_indices, vmin=-0.5, vmax=0.5, cmap=CMAPS["diverging"])

cax_sd = reg(fig.add_subplot(gs_sttc_mat[0, 4]), "colorbar", "C_sttc_diff_cbar")
cb_sd = fig.colorbar(im_sd, cax=cax_sd)
cb_sd.set_label("Δ STTC", fontsize=FONT_SIZES["colorbar_label"], rotation=270, labelpad=10)
cb_sd.outline.set_linewidth(0.5)
cb_sd.ax.tick_params(width=0.5, length=2)
cb_sd.set_ticks([-0.5, 0, 0.5])

print("Row C done")

# --- FR vs STTC scatter ---
gs_scatter = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_left[2], wspace=0.45)

fr_diff_sc = mat_fr_d50 - mat_fr_d0
sttc_diff_sc = mat_sttc_d50 - mat_sttc_d0

n_axes_before = len(fig.axes)
make_scatter_with_marginals(gs_scatter[0], fig, mat_fr_d0, mat_sttc_d0, "FR correlation", "STTC")
make_scatter_with_marginals(gs_scatter[1], fig, mat_fr_d50, mat_sttc_d50, "FR correlation", "STTC")
make_scatter_with_marginals(gs_scatter[2], fig, fr_diff_sc, sttc_diff_sc, "Δ FR corr. (D50−D0)", "Δ STTC (D50−D0)", show_zero_lines=True)

ax_scatter_first = fig.axes[n_axes_before + 1]
add_panel_label(ax_scatter_first, "D")

print("Row D done")

# --- Violins + normalized ---
ax_violin = reg(fig.add_subplot(gs_right[1]), "panel", "E_violin")
plot_pairwise_corr_violins(ax_violin, corr_data)
add_panel_label(ax_violin, "E")

ax_norm = reg(fig.add_subplot(gs_right[2]), "panel", "F_normalized")
plot_pairwise_corr_normalized(ax_norm, corr_data, show_legend=True)
add_panel_label(ax_norm, "F")

print("Panels E-F done")

# ═══════════════════════════════════════════════════════════════════════════
# Bottom block — MEA network + graph metrics
# ═══════════════════════════════════════════════════════════════════════════
gs_graph_block = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_top[2][0, 0], width_ratios=[1, 2.2], wspace=0.25,
)

# --- MEA networks ---
gs_mea = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_graph_block[0], height_ratios=[1, 0.12, 1], hspace=0.02,
)

sd_d0 = ws_get("D0", "spikedata")
sd_d50 = ws_get("D50", "spikedata")
fr_corr_d0_mat = ws_get("D0", "fr_corr_matrix").matrix
fr_corr_d50_mat = ws_get("D50", "fr_corr_matrix").matrix
edge_threshold = 0.8
shared_vmin = 0.0
shared_vmax = 0.5

ax_mea_d0 = reg(fig.add_subplot(gs_mea[0]), "panel", "G_mea_D0")
sc_d0 = plot_spatial_network(
    ax_mea_d0,
    np.column_stack([[a["x"] for a in sd_d0.neuron_attributes], [a["y"] for a in sd_d0.neuron_attributes]]),
    fr_corr_d0_mat, edge_threshold=edge_threshold,
    node_size_range=(2, 20), node_cmap="viridis", node_linewidth=0.2,
    edge_color="red", edge_linewidth=0.6, scale_bar_um=0, font_size=FONT_SIZES["axes_label"],
)
sc_d0.set_clim(shared_vmin, shared_vmax)
ax_mea_d0.set_title("0 µM", fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=4)
add_panel_label(ax_mea_d0, "G", y=1.08)

ax_mea_d50 = reg(fig.add_subplot(gs_mea[2]), "panel", "G_mea_D50")
sc_d50 = plot_spatial_network(
    ax_mea_d50,
    np.column_stack([[a["x"] for a in sd_d50.neuron_attributes], [a["y"] for a in sd_d50.neuron_attributes]]),
    fr_corr_d50_mat, edge_threshold=edge_threshold,
    node_size_range=(2, 20), node_cmap="viridis", node_linewidth=0.2,
    edge_color="red", edge_linewidth=0.6, scale_bar_um=0, font_size=FONT_SIZES["axes_label"],
)
sc_d50.set_clim(shared_vmin, shared_vmax)
ax_mea_d50.set_title("50 µM", fontsize=UNIFIED_FONTSIZE, fontweight="bold", pad=4)

# Middle strip: colorbar + edge legend
ax_mid = reg(fig.add_subplot(gs_mea[1]), "panel", "G_mid")
ax_mid.set_xlim(0, 1); ax_mid.set_ylim(0, 1); ax_mid.axis("off")

cax_mea = fig.add_axes([0, 0, 0.1, 0.01])
reg(cax_mea, "colorbar", "G_cbar")
cbar_mea = fig.colorbar(sc_d0, cax=cax_mea, orientation="horizontal")
cbar_mea.set_label("Mean FR corr.", fontsize=FONT_SIZES["colorbar_label"], labelpad=1)
cbar_mea.outline.set_linewidth(0.5)
cbar_mea.ax.tick_params(width=0.5, length=2, labelsize=FONT_SIZES["tick_label"])

legend_vals = [0.8, 0.9, 1.0]
legend_handles = [
    Line2D([0], [0], color="red", linewidth=1.2,
           alpha=0.15 + 0.85 * (v - edge_threshold) / (1.0 - edge_threshold), label=f"{v:.1f}")
    for v in legend_vals
]
ax_mid.legend(
    handles=legend_handles, fontsize=FONT_SIZES["legend"],
    loc="center left", bbox_to_anchor=(0.0, 0.5), ncol=1,
    columnspacing=0.8, handlelength=2.0, frameon=False,
    title="FR corr.", title_fontsize=FONT_SIZES["legend"],
)

# Scale bar
xlim = ax_mea_d50.get_xlim(); ylim = ax_mea_d50.get_ylim()
bar_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
bar_start = bar_end - 500
bar_y = ylim[0] - (ylim[1] - ylim[0]) * 0.06
ax_mea_d50.plot([bar_start, bar_end], [bar_y, bar_y], color="black", linewidth=2.0, clip_on=False, solid_capstyle="butt")
ax_mea_d50.text((bar_start + bar_end) / 2, bar_y - (ylim[1] - ylim[0]) * 0.03, "500 µm", ha="center", va="top", fontsize=FONT_SIZES["axes_label"])

print("Panel G done")

# --- Graph metrics 2x3 ---
gs_gm = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_graph_block[1], hspace=0.55, wspace=0.5)

graphs_raw = {}; graphs_inv = {}
for cond in CONDITIONS:
    pcm = ws_get(cond, "fr_corr_matrix")
    graphs_raw[cond] = pcm.to_networkx(threshold=0.0, invert_weights=False)
    graphs_inv[cond] = pcm.to_networkx(threshold=0.0, invert_weights=True)

color_list = [COLORS[c] for c in CONDITIONS]
cond_labels = ["0", "3", "10", "30", "50"]

# H: Node strength
strength = {c: np.array([d / (graphs_raw[c].number_of_nodes() - 1) for _, d in graphs_raw[c].degree(weight="weight")]) for c in CONDITIONS}
ax_h = reg(fig.add_subplot(gs_gm[0, 0]), "panel", "H_strength")
plot_distribution(ax_h, strength, labels=cond_labels, colors=color_list, ylabel="Mean FR corr.", xlabel="Diazepam (µM)", show_median=True, show_quartiles=True)
add_panel_label(ax_h, "H", x=-0.30, y=1.0)

# I: Clustering
clustering = {c: np.array(list(nx.clustering(graphs_raw[c], weight="weight").values())) for c in CONDITIONS}
ax_i = reg(fig.add_subplot(gs_gm[0, 1]), "panel", "I_clustering")
plot_distribution(ax_i, clustering, labels=cond_labels, colors=color_list, ylabel="Clustering coeff.", xlabel="Diazepam (µM)", show_median=True, show_quartiles=True)
add_panel_label(ax_i, "I", x=-0.30, y=1.0)

# J: Avg shortest path
avg_path = {}
for c in CONDITIONS:
    G = graphs_inv[c]
    comp = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()
    avg_path[c] = nx.average_shortest_path_length(comp, weight="weight")

ax_j = reg(fig.add_subplot(gs_gm[0, 2]), "panel", "J_path")
x_bar = np.arange(len(CONDITIONS))
ax_j.bar(x_bar, [avg_path[c] for c in CONDITIONS], color=color_list, width=0.7, edgecolor="black", linewidth=0.5)
ax_j.set_xticks(x_bar); ax_j.set_xticklabels(cond_labels)
ax_j.set_ylim(0.5, 0.75)
ax_j.set_xlabel("Diazepam (µM)"); ax_j.set_ylabel("Avg. shortest path")
add_panel_label(ax_j, "J", x=-0.30, y=1.0)

# K: Betweenness (normalized bands)
betweenness = {c: np.array(list(nx.betweenness_centrality(graphs_inv[c], weight="weight", normalized=True).values())) for c in CONDITIONS}
ax_k = reg(fig.add_subplot(gs_gm[1, 0]), "panel", "K_betweenness")
plot_percentile_bands(ax_k, betweenness, labels=cond_labels, normalize=True, style="bands", ylabel="Norm. betweenness", xlabel="Diazepam (µM)", show_legend=False)
handles, labels_leg = ax_k.get_legend_handles_labels()
ax_k.legend(handles=handles, labels=labels_leg, loc="upper left", ncol=2, fontsize=FONT_SIZES["legend"], columnspacing=0.8, handlelength=1.2)
add_panel_label(ax_k, "K", x=-0.30, y=1.0)

# L: Modularity
modularity = {}
for c in CONDITIONS:
    part = community_louvain.best_partition(graphs_raw[c], weight="weight", random_state=42)
    modularity[c] = community_louvain.modularity(part, graphs_raw[c], weight="weight")

ax_l = reg(fig.add_subplot(gs_gm[1, 1]), "panel", "L_modularity")
ax_l.bar(x_bar, [modularity[c] for c in CONDITIONS], color=color_list, width=0.7, edgecolor="black", linewidth=0.5)
ax_l.set_xticks(x_bar); ax_l.set_xticklabels(cond_labels)
ax_l.set_xlabel("Diazepam (µM)"); ax_l.set_ylabel("Modularity (Q)")
add_panel_label(ax_l, "L", x=-0.30, y=1.0)

# M: Rich-club
ax_m = reg(fig.add_subplot(gs_gm[1, 2]), "panel", "M_richclub")
for c in CONDITIONS:
    pct, phi = weighted_rich_club(graphs_raw[c])
    ax_m.plot(pct, phi, color=COLORS[c], linewidth=1.5, label=c)
ax_m.set_xlabel("Strength percentile"); ax_m.set_ylabel("Rich-club φ$^w$")
ax_m.legend(fontsize=5.5, ncol=1, loc="lower left")
add_panel_label(ax_m, "M", x=-0.30, y=1.0)

print("Panels H-M done")

# ═══════════════════════════════════════════════════════════════════════════
# Position adjustments
# ═══════════════════════════════════════════════════════════════════════════
fig.canvas.draw()

# Shrink FR heatmap colorbar
pos_hm = last_ax_fr.get_position()
pos_cb = cax_fr.get_position()
new_h = pos_hm.height * 0.8
y_off = (pos_hm.height - new_h) / 2
cax_fr.set_position([pos_cb.x0 - 0.01, pos_hm.y0 + y_off, pos_cb.width, new_h])

# MEA colorbar position
pos_d0 = ax_mea_d0.get_position()
cw = pos_d0.width * 0.60
cax_mea.set_position([pos_d0.x1 - cw, pos_d0.y0 - 0.012, cw, 0.005])

# Match matrix colorbar heights and nudge close to their heatmaps
cbar_gap = 0.005
for cax, ref in [(cax_fr_mat, ax_fr50), (cax_frd, ax_frd), (cax_s, ax_s50), (cax_sd, ax_sd)]:
    pr = ref.get_position()
    pc = cax.get_position()
    cax.set_position([pr.x1 + cbar_gap, pr.y0, pc.width, pr.height])

# ═══════════════════════════════════════════════════════════════════════════
# Enforce font sizes
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
preview_path = os.path.join(SCRIPT_DIR, "Figure5_preview.png")
os.makedirs(os.path.dirname(preview_path) or ".", exist_ok=True)
fig.savefig(preview_path, dpi=900)
plt.close(fig)
print("Preview saved as Figure5_preview.png")

from PIL import Image as PILImage
tif_path = os.path.join(SCRIPT_DIR, "Figure5.tif")
img = PILImage.open(preview_path).convert("RGB")
img.save(tif_path, format="TIFF", dpi=(900, 900))
print(f"TIFF saved as Figure5.tif ({img.size[0]}x{img.size[1]} px)")
