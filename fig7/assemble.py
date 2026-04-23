"""Assemble Figure 7 — GPLVM burst analysis.

Layout:
    Top block (rows A-B):
        Left 2/3:   Two burst raster panels (D0, D50) with pop rate and
                    GPLVM model states
        Right 1/3:  State transition matrices stacked vertically (5 conditions)
    Bottom block (rows C-I):
        Left:       Average pop rate + average P(continuous) around burst peaks
        Right 2x3:  State probability, entropy violins, PCA variance (top row);
                    PCA D0, PCA D50, PCA combined (bottom row)

Prerequisites:
    - compute_shared.py  (spikedata, tburst, burst_edges, pop_rate_acc, fr_rates)
    - fig7/compute.py    (gplvm_burst_result, rate_pca_embedding, etc.)

Usage:
    python -m fig7.assemble     (from the 200123_2953 directory)
    python fig7/assemble.py     (same)

Outputs:
    fig7/Figure7.tif            Submission-ready TIFF
"""

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib

matplotlib.use("Agg")

import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
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
    save_fig_submission,
    add_panel_label,
    FULL_PAGE_WIDTH,
    COLORS,
    FONT_SIZES,
    LINE_WIDTHS,
    CONDITIONS,
)

apply_style()

from fig7.panels import (
    stitch_burst_spikedata,
    stitch_pop_rate,
    stitch_gplvm_states,
    compute_transition_matrix,
    plot_transition_matrix,
    compute_closest_burst_edges,
    compute_avg_cont_prob_per_condition,
    plot_avg_poprate_with_edges,
    plot_avg_cont_prob,
    compute_avg_state_prob_per_condition,
    plot_avg_state_prob,
    compute_entropy_per_condition,
    plot_entropy_violins,
    plot_cumulative_variance,
    map_gplvm_states_to_ms,
    plot_pca_with_states,
    compute_avg_poprate_and_states,
    add_poprate_inset,
)

from spikelab.workspace.workspace import AnalysisWorkspace
from spikelab.spikedata.plot_utils import plot_recording
from spikelab.spikedata.ratedata import RateData

# ── Axes registry ──────────────────────────────────────────────────────────
axes_registry = {}


def reg(ax, role, desc):
    axes_registry[id(ax)] = {"role": role, "desc": desc, "ax": ax}
    return ax


# ── Load data ──────────────────────────────────────────────────────────────
results_dir = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
ws = AnalysisWorkspace.load(os.path.join(results_dir, "workspace"))
print("Workspace loaded.\n")

gplvm_all = ws.get("all", "gplvm_burst_result")
boundaries_bins = ws.get("all", "gplvm_burst_boundaries_bins")
condition_idx = ws.get("all", "gplvm_burst_condition_idx")

burst_counts = {
    rec: int(np.sum(condition_idx == i)) for i, rec in enumerate(CONDITIONS)
}
offsets = {}
o = 0
for rec in CONDITIONS:
    offsets[rec] = o
    o += burst_counts[rec]

n_states = np.array(gplvm_all["decode_res"]["posterior_latent_marg"]).shape[1]

PRE_MS = 250
POST_MS = 500
N_BURSTS = 4
GAP_MS = 150
BIN_SIZE_MS = 50
GAP_BINS = int(GAP_MS / BIN_SIZE_MS)

# ── Precompute data ────────────────────────────────────────────────────────
# Stitched burst data for D0 and D50
raster_conds = ["D0", "D50"]
stitched = {}
pr_global_max = 0.0
for cond in raster_conds:
    tburst = ws.get(cond, "tburst")
    pop_rate_full = ws.get(cond, "pop_rate_acc")
    sss = ws.get(cond, "burst_sss")
    burst_sds = [sss.spike_stack[i] for i in range(min(N_BURSTS, len(sss.spike_stack)))]
    sd_stitch, bounds = stitch_burst_spikedata(burst_sds, gap_ms=GAP_MS)
    pr_stitch = stitch_pop_rate(
        pop_rate_full, tburst, PRE_MS, POST_MS, list(range(N_BURSTS)), gap_ms=GAP_MS
    )
    global_idx = [offsets[cond] + i for i in range(N_BURSTS)]
    ms, cp = stitch_gplvm_states(
        gplvm_all, boundaries_bins, global_idx, gap_bins=GAP_BINS
    )
    pr_global_max = max(pr_global_max, np.nanmax(pr_stitch))
    stitched[cond] = dict(
        sd=sd_stitch,
        bounds=bounds,
        pr=pr_stitch,
        model_states=ms,
        cont_prob=cp,
        global_idx=global_idx,
    )
# Convert raw pop rate max to Hz/unit (plot_recording displays in Hz/unit)
_n_units = stitched["D0"]["sd"].N
_bin_s = 1.0 / 1000.0  # 1 ms bins
pr_global_max_hz = pr_global_max / (_bin_s * _n_units)
pop_rate_ylim = (0, pr_global_max_hz * 1.05)

# Transition matrices
trans_probs = {}
global_vmax_trans = 0
for i, rec in enumerate(CONDITIONS):
    tp, _ = compute_transition_matrix(
        gplvm_all, boundaries_bins, condition_idx, i, len(CONDITIONS), n_states
    )
    trans_probs[rec] = tp
    if tp.max() > global_vmax_trans:
        global_vmax_trans = tp.max()

# Avg pop rate & cont prob
sd_burst_data = {}
burst_edge_ranges = {}
for rec in CONDITIONS:
    pr = ws.get(rec, "pop_rate_acc")
    tb = ws.get(rec, "tburst")
    edges = ws.get(rec, "burst_edges")
    sd = ws.get(rec, "spikedata")
    sd_burst_data[rec] = {
        "sd": sd,
        "pop_rate": pr,
        "tburst": tb,
        "burst_edges": edges,
    }
    burst_edge_ranges[rec] = compute_closest_burst_edges(tb, edges)

avg_cont_list = compute_avg_cont_prob_per_condition(
    gplvm_all, boundaries_bins, condition_idx, len(CONDITIONS)
)
avg_cont_data = {rec: avg_cont_list[i] for i, rec in enumerate(CONDITIONS)}

# Avg state prob
avg_prob_list = compute_avg_state_prob_per_condition(
    gplvm_all, boundaries_bins, condition_idx, len(CONDITIONS)
)
avg_probs = {rec: avg_prob_list[i] for i, rec in enumerate(CONDITIONS)}

# Entropy
entropy_list = compute_entropy_per_condition(
    gplvm_all, boundaries_bins, condition_idx, len(CONDITIONS)
)
entropies = {rec: entropy_list[i] for i, rec in enumerate(CONDITIONS)}

# PCA variance (10 components)
var_data = {}
for rec in CONDITIONS:
    fr = ws.get(rec, "fr_rates")
    times = np.arange(fr.shape[1], dtype=float)
    rd = RateData(fr, times)
    _, vr, _ = rd.get_manifold(method="PCA", n_components=10)
    var_data[rec] = vr

# PCA embeddings + state maps for D0/D50
pca_data = {}
for cond in raster_conds:
    emb = ws.get(cond, "rate_pca_embedding")
    vr = ws.get(cond, "rate_pca_variance")
    tb = ws.get(cond, "tburst")
    pr = ws.get(cond, "pop_rate_acc")
    n_b = burst_counts[cond]
    gi = [offsets[cond] + j for j in range(n_b)]
    state_ms = map_gplvm_states_to_ms(tb, gplvm_all, boundaries_bins, gi, emb.shape[0])
    avg_pr, avg_st = compute_avg_poprate_and_states(
        pr, tb, gplvm_all, boundaries_bins, gi
    )
    pca_data[cond] = dict(
        emb=emb, var=vr, state_ms=state_ms, avg_pr=avg_pr, avg_state=avg_st
    )

# Combined PCA
pca_combined_emb = []
pca_combined_labels = []
var_ratio_combined = ws.get("all", "rate_pca_combined_variance")
for i, rec in enumerate(CONDITIONS):
    emb = ws.get(rec, "rate_pca_combined_embedding")
    tb = ws.get(rec, "tburst")
    n_b = burst_counts[rec]
    gi = [offsets[rec] + j for j in range(n_b)]
    state_ms = map_gplvm_states_to_ms(tb, gplvm_all, boundaries_bins, gi, emb.shape[0])
    cond_label = np.where(state_ms >= 0, i, -1)
    pca_combined_emb.append(emb)
    pca_combined_labels.append(cond_label)
pca_comb_emb = np.concatenate(pca_combined_emb, axis=0)
pca_comb_lbl = np.concatenate(pca_combined_labels)

sort_indices = ws.get("D0", "gplvm_burst_result")["reorder_indices"]

print("All data loaded.\n")

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FULL_PAGE_WIDTH, 10.0))

gs_top = gridspec.GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=[1.3, 0.8],
    left=0.05,
    right=0.97,
    top=0.98,
    bottom=0.04,
    hspace=0.12,
)

# ── TOP BLOCK: rasters + transitions ───────────────────────────────────────
gs_block0 = gridspec.GridSpecFromSubplotSpec(
    1,
    3,
    subplot_spec=gs_top[0],
    width_ratios=[2, 2, 1.2],
    wspace=0.25,
)


# -- Raster sub-panels for D0 and D50 --
def make_raster_panel(
    gs_slot,
    cond,
    panel_label,
    show_left_labels=True,
    show_right_labels=True,
    show_colorbar=True,
):
    """Render burst raster + pop rate + model states into a GridSpec slot."""
    if show_colorbar:
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            3,
            2,
            subplot_spec=gs_slot,
            height_ratios=[3, 1, 2],
            width_ratios=[1, 0.03],
            hspace=0.05,
            wspace=0.05,
        )
    else:
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            3,
            1,
            subplot_spec=gs_slot,
            height_ratios=[3, 1, 2],
            hspace=0.05,
        )

    ax_r = reg(fig.add_subplot(gs_inner[0, 0]), "panel", f"{panel_label}_raster")
    ax_p = reg(
        fig.add_subplot(gs_inner[1, 0], sharex=ax_r), "panel", f"{panel_label}_poprate"
    )
    ax_m = reg(
        fig.add_subplot(gs_inner[2, 0], sharex=ax_r), "panel", f"{panel_label}_model"
    )

    if show_colorbar:
        ax_cb = reg(
            fig.add_subplot(gs_inner[2, 1]), "colorbar", f"{panel_label}_model_cbar"
        )
        # Hide unused top-right cells
        ax_empty1 = fig.add_subplot(gs_inner[0, 1])
        ax_empty1.axis("off")
        ax_empty2 = fig.add_subplot(gs_inner[1, 1])
        ax_empty2.axis("off")
        reg(ax_empty1, "hidden", f"{panel_label}_empty1")
        reg(ax_empty2, "hidden", f"{panel_label}_empty2")
    else:
        ax_cb = fig.add_axes([0, 0, 0.001, 0.001])
        ax_cb.axis("off")
        ax_cb.set_visible(False)
        reg(ax_cb, "hidden", f"{panel_label}_model_cbar")

    # Hidden colorbar axes for raster and poprate
    cax_r = fig.add_axes([0, 0, 0.001, 0.001])
    cax_r.axis("off")
    cax_r.set_visible(False)
    cax_p = fig.add_axes([0, 0, 0.001, 0.001])
    cax_p.axis("off")
    cax_p.set_visible(False)
    reg(cax_r, "hidden", f"{panel_label}_raster_cbar")
    reg(cax_p, "hidden", f"{panel_label}_poprate_cbar")

    d = stitched[cond]
    plot_recording(
        d["sd"],
        show_raster=True,
        show_pop_rate=True,
        show_model_states=True,
        pop_rate=d["pr"],
        cont_prob=d["cont_prob"],
        model_states=d["model_states"],
        sort_indices=sort_indices,
        raster_style="eventplot",
        font_size=FONT_SIZES["axes_label"],
        show=False,
        axes=[(ax_r, cax_r), (ax_p, cax_p), (ax_m, ax_cb)],
    )

    # Restyle raster
    for coll in ax_r.collections:
        coll.set_linewidth(LINE_WIDTHS["raster_marker"])
    n_units = d["sd"].N
    ax_r.set_yticks([0, n_units - 1])
    ax_r.set_yticklabels(["1", str(n_units)])
    ax_r.spines["left"].set_bounds(0, n_units - 1)

    # Pop rate ylim
    ax_p.set_ylim(pop_rate_ylim)
    # Pin cont_prob right y-axis
    for child in fig.get_axes():
        if child.get_ylabel() == "P(continuous)":
            pos_c = child.get_position()
            pos_p = ax_p.get_position()
            if abs(pos_c.y0 - pos_p.y0) < 0.05:
                child.set_ylim(0, 1)

    # Style pop rate lines
    for line in ax_p.get_lines():
        line.set_linewidth(LINE_WIDTHS["data_trace"])

    # Remove x-ticks and bottom spines
    for ax in [ax_r, ax_p, ax_m]:
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.spines["bottom"].set_visible(False)

    # Colorbar styling
    if show_colorbar:
        ax_cb.tick_params(width=0.5, length=2)
        for spine in ax_cb.spines.values():
            spine.set_linewidth(0.5)

    # Strip left y-axis labels if not needed
    if not show_left_labels:
        for ax in [ax_r, ax_p, ax_m]:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    # Style right y-axis (P(continuous))
    for child in fig.get_axes():
        if child.get_ylabel() == "P(continuous)":
            pos_c = child.get_position()
            pos_p = ax_p.get_position()
            if abs(pos_c.y0 - pos_p.y0) < 0.05:
                if not show_right_labels:
                    child.set_ylabel("")
                    child.set_yticklabels([])
                    child.tick_params(right=False)
                else:
                    # Only 0 and 1 ticks, label between them
                    child.set_yticks([0, 1])
                    child.set_yticklabels(["0", "1"])
                    child.set_ylabel("")
                    child.spines["right"].set_visible(True)
                    child.spines["right"].set_color("red")
                    child.spines["right"].set_linewidth(LINE_WIDTHS["axis_spine"])
                    child.tick_params(axis="y", colors="red")
                    child.text(
                        1.05,
                        0.5,
                        "P(cont.)",
                        transform=child.transAxes,
                        fontsize=FONT_SIZES["axes_label"],
                        color="red",
                        ha="center",
                        va="center",
                        rotation=90,
                    )

    # Burst separators
    for b_start, b_end in d["bounds"]:
        for ax in [ax_r, ax_p, ax_m]:
            ax.axvline(b_start, color="0.7", lw=0.4, ls=":", zorder=0)
            ax.axvline(b_end, color="0.7", lw=0.4, ls=":", zorder=0)

    # Scale bar below model states panel, outside axes (axes fraction coords)
    window_ms = PRE_MS + POST_MS
    total_ms = N_BURSTS * window_ms + (N_BURSTS - 1) * GAP_MS
    bar_frac = 200 / total_ms  # 200 ms as fraction of x-axis
    ax_m.annotate(
        "",
        xy=(0.98, -0.04),
        xytext=(0.98 - bar_frac, -0.04),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
    )
    ax_m.text(
        0.98 - bar_frac / 2,
        -0.08,
        "200 ms",
        transform=ax_m.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZES["axes_label"],
    )

    ax_r.set_title(cond, fontsize=FONT_SIZES["axes_label"], pad=4)
    return ax_r


ax_A = make_raster_panel(
    gs_block0[0],
    "D0",
    "A",
    show_left_labels=True,
    show_right_labels=False,
    show_colorbar=True,
)
ax_A2 = make_raster_panel(
    gs_block0[1],
    "D50",
    "A2",
    show_left_labels=False,
    show_right_labels=True,
    show_colorbar=False,
)
add_panel_label(ax_A, "A", x=-0.12)

# -- Transition matrices stacked vertically --
gs_trans = gridspec.GridSpecFromSubplotSpec(
    5,
    2,
    subplot_spec=gs_block0[2],
    width_ratios=[1, 0.06],
    hspace=0.2,
    wspace=0.08,
)

cax_trans = reg(fig.add_subplot(gs_trans[0:2, 1]), "colorbar", "B_trans_cbar")
for k in range(2, 5):
    ax_empty = fig.add_subplot(gs_trans[k, 1])
    ax_empty.axis("off")
    reg(ax_empty, "hidden", f"B_trans_cbar_empty_{k}")

for k, rec in enumerate(CONDITIONS):
    ax_t = reg(fig.add_subplot(gs_trans[k, 0]), "panel", f"B_trans_{rec}")
    cax_t = cax_trans if k == 0 else None
    is_bottom = k == len(CONDITIONS) - 1
    plot_transition_matrix(
        ax_t,
        trans_probs[rec],
        cax=cax_t,
        vmax=global_vmax_trans,
        show_ylabel=True,
        show_xlabel=is_bottom,
    )
    ax_t.yaxis.set_label_coords(-0.08, 0.5)
    if not is_bottom:
        ax_t.set_xticklabels([])
    else:
        ax_t.xaxis.set_label_coords(0.5, -0.08)
    ax_t.set_title(rec, fontsize=FONT_SIZES["axes_label"], pad=2)
    if k == 0:
        add_panel_label(ax_t, "B", x=-0.25, y=1.15)

print("Top block done.")

# ── BOTTOM BLOCK ───────────────────────────────────────────────────────────
gs_block1 = gridspec.GridSpecFromSubplotSpec(
    1,
    2,
    subplot_spec=gs_top[1],
    width_ratios=[1, 3],
    wspace=0.25,
)

# -- Left: avg poprate + contprob --
gs_left = gridspec.GridSpecFromSubplotSpec(
    2,
    1,
    subplot_spec=gs_block1[0],
    height_ratios=[1, 1],
    hspace=0.35,
)

ax_pr = reg(fig.add_subplot(gs_left[0]), "panel", "D_poprate")
ax_cp = reg(fig.add_subplot(gs_left[1]), "panel", "E_contprob")

plot_avg_poprate_with_edges(ax_pr, sd_burst_data)
plot_avg_cont_prob(ax_cp, avg_cont_data, burst_edge_ranges)

ax_pr.set_xticks([])
ax_pr.set_xlabel("")
ax_cp.set_xticks([])
ax_cp.set_xlabel("Time from burst peak (ms)")

# Legend between subplots
handles, labels = ax_cp.get_legend_handles_labels()
ax_pr.legend(
    handles,
    labels,
    fontsize=FONT_SIZES["legend"],
    handlelength=1.0,
    handletextpad=0.3,
    columnspacing=0.5,
    loc="upper left",
    ncol=5,
    borderpad=0.2,
    bbox_to_anchor=(0.0, -0.02),
)

# Scale bar
trans = transforms.blended_transform_factory(ax_pr.transData, ax_pr.transAxes)
xlim = ax_pr.get_xlim()
bar_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
bar_start = bar_end - 100
ax_pr.plot(
    [bar_start, bar_end],
    [-0.1, -0.1],
    color="black",
    linewidth=1.5,
    clip_on=False,
    solid_capstyle="butt",
    transform=trans,
)
ax_pr.text(
    (bar_start + bar_end) / 2,
    -0.17,
    "100 ms",
    ha="center",
    va="top",
    fontsize=FONT_SIZES["axes_label"],
    transform=trans,
)

add_panel_label(ax_pr, "C")

# -- Right: 2x3 grid --
gs_right = gridspec.GridSpecFromSubplotSpec(
    2,
    3,
    subplot_spec=gs_block1[1],
    hspace=0.4,
    wspace=0.4,
)

# Top row: state prob, entropy, PCA variance
ax_sp = reg(fig.add_subplot(gs_right[0, 0]), "panel", "F_state_prob")
plot_avg_state_prob(ax_sp, avg_probs)
add_panel_label(ax_sp, "D", x=-0.25)

ax_ent = reg(fig.add_subplot(gs_right[0, 1]), "panel", "E_entropy")
plot_entropy_violins(ax_ent, entropies)
add_panel_label(ax_ent, "E", x=-0.25)

ax_var = reg(fig.add_subplot(gs_right[0, 2]), "panel", "F_pca_var")
plot_cumulative_variance(ax_var, var_data)
add_panel_label(ax_var, "F", x=-0.25)

# Bottom row: PCA D0, PCA D50, PCA combined
ax_pca0 = reg(fig.add_subplot(gs_right[1, 0]), "panel", "I_pca_D0")
sc0 = plot_pca_with_states(
    ax_pca0, pca_data["D0"]["emb"], pca_data["D0"]["state_ms"], n_states
)
add_poprate_inset(
    ax_pca0, pca_data["D0"]["avg_pr"], pca_data["D0"]["avg_state"], n_states
)
vr0 = pca_data["D0"]["var"]
ax_pca0.set_xlabel(f"PC1 ({vr0[0]:.1%})", fontsize=FONT_SIZES["axes_label"])
ax_pca0.set_ylabel(f"PC2 ({vr0[1]:.1%})", fontsize=FONT_SIZES["axes_label"])
ax_pca0.set_xticks([])
ax_pca0.set_yticks([])
ax_pca0.set_title("D0", fontsize=FONT_SIZES["axes_label"], pad=4)
add_panel_label(ax_pca0, "G")

ax_pca50 = reg(fig.add_subplot(gs_right[1, 1]), "panel", "H_pca_D50")
sc50 = plot_pca_with_states(
    ax_pca50, pca_data["D50"]["emb"], pca_data["D50"]["state_ms"], n_states
)
add_poprate_inset(
    ax_pca50, pca_data["D50"]["avg_pr"], pca_data["D50"]["avg_state"], n_states
)
vr50 = pca_data["D50"]["var"]
ax_pca50.set_xlabel(f"PC1 ({vr50[0]:.1%})", fontsize=FONT_SIZES["axes_label"])
ax_pca50.set_ylabel(f"PC2 ({vr50[1]:.1%})", fontsize=FONT_SIZES["axes_label"])
ax_pca50.set_xticks([])
ax_pca50.set_yticks([])
ax_pca50.set_title("D50", fontsize=FONT_SIZES["axes_label"], pad=4)
add_panel_label(ax_pca50, "H")

# Shared colorbar for G and H (GPLVM state), placed between them
if sc0 is not None or sc50 is not None:
    sc_ref = sc50 if sc50 is not None else sc0
    # Position between ax_pca0 and ax_pca50
    pos0 = ax_pca0.get_position()
    pos50 = ax_pca50.get_position()
    cbar_x = pos0.x1 + (pos50.x0 - pos0.x1) * 0.1
    cbar_w = 0.008
    cbar_y = pos0.y0 + pos0.height * 0.15
    cbar_h = pos0.height * 0.6
    cax_pca = reg(
        fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h]), "colorbar", "GH_state_cbar"
    )
    cb_pca = plt.colorbar(sc_ref, cax=cax_pca, alpha=1.0)
    cb_pca.solids.set_alpha(1.0)
    cb_pca.set_ticks([])
    cb_pca.outline.set_linewidth(0.5)
    cax_pca.text(
        0.5,
        -0.05,
        "1",
        transform=cax_pca.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZES["tick_label"],
    )
    cax_pca.text(
        0.5,
        1.05,
        str(n_states),
        transform=cax_pca.transAxes,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZES["tick_label"],
    )
    cax_pca.text(
        0.5,
        1.25,
        "State",
        transform=cax_pca.transAxes,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZES["colorbar_label"],
    )

ax_pcac = reg(fig.add_subplot(gs_right[1, 2]), "panel", "I_pca_combined")
pc1 = pca_comb_emb[:, 0]
pc2 = pca_comb_emb[:, 1]
bg = pca_comb_lbl == -1
ax_pcac.scatter(
    pc1[bg], pc2[bg], s=0.3, c="0.85", alpha=0.03, rasterized=True, edgecolors="none"
)
for i in reversed(range(len(CONDITIONS))):
    mask = pca_comb_lbl == i
    if mask.any():
        ax_pcac.scatter(
            pc1[mask],
            pc2[mask],
            s=0.8,
            c=COLORS[CONDITIONS[i]],
            alpha=0.3,
            rasterized=True,
            edgecolors="none",
            label=CONDITIONS[i],
        )
ax_pcac.set_xlabel(
    f"PC1 ({var_ratio_combined[0]:.1%})", fontsize=FONT_SIZES["axes_label"]
)
ax_pcac.set_ylabel(
    f"PC2 ({var_ratio_combined[1]:.1%})", fontsize=FONT_SIZES["axes_label"]
)
ax_pcac.set_xticks([])
ax_pcac.set_yticks([])
ax_pcac.set_title("Combined", fontsize=FONT_SIZES["axes_label"], pad=4)
leg_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=COLORS[r],
        markersize=3,
        label=r,
        linewidth=0,
    )
    for r in CONDITIONS
]
ax_pcac.legend(
    handles=leg_handles,
    fontsize=FONT_SIZES["legend"],
    handletextpad=0.2,
    loc="center left",
    bbox_to_anchor=(0.9, 0.5),
    ncol=1,
)
add_panel_label(ax_pcac, "I")

print("Bottom block done.")

# ── Enforce font sizes ────────────────────────────────────────────────────
UNIFIED_FONTSIZE = FONT_SIZES["axes_label"]
for entry in axes_registry.values():
    if entry["role"] == "hidden":
        continue
    fs = FONT_SIZES["annotation"] if entry["role"] == "inset" else UNIFIED_FONTSIZE
    ax = entry["ax"]
    ax.tick_params(labelsize=fs)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fs)
    ax.xaxis.label.set_fontsize(fs)
    ax.yaxis.label.set_fontsize(fs)

# ── Save ──────────────────────────────────────────────────────────────────
save_fig_submission(fig, os.path.join(SCRIPT_DIR, "Figure7.tif"))
print("\nFigure7.tif saved.")
