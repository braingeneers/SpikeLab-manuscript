"""Plot rubric item 14, method consistency across a cell's ten replicates.

Two views of the same cell scores:

    plot_consistency_maps     three heatmaps — with, without, difference —
                              laid out like the score overview. Supplementary.
    plot_consistency_summary  a compact paired plot, one line per cell, for the
                              main figure's narrow slot.

The score is max_agreeing / n_attempted: the largest group of a cell's ten
replicates that reported the population-coupling value in mutually commensurable
units. It is cell-scope, so it is reported on its own and never folded into the
13-item per-run totals — see grade_cell.py and ANALYSIS_PLAN.md section 8.

Item 14 was graded on the confirmatory grid only, so there are three effort
columns here against the score overview's five. The n=3 levels have no item-14
score at all, which is why the supplementary map cannot simply mirror panel B.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style, save_fig, add_panel_label, CMAPS, FONT_SIZES, FULL_PAGE_WIDTH,
)
apply_style()

from benchmark_v2_data import (
    CONFIRMATORY_EFFORTS, EFFORT_LABELS, MODEL_ORDER, MODEL_LABELS_SHORT,
    PHASE_COLORS, PHASE_LABELS, PHASE_LABELS_SHORT, consistency_matrices,
    load_cell_scores, model_colors,
)
from plot_effort_heatmap import (
    _text_color, _mark_ablation_cell, CELL_EDGE, CELL_EDGE_LW, SPINE_LW,
)
from panel_common import MARKER_SIZE, mean_ci, stars

# Matches the ablation bars: white dots with a dark rim over a filled bar.
CELL_DOT_EDGE = "#4D4D4D"
BAR_WIDTH = 0.62
JITTER = 0.14


def _draw(ax, mat, cmap, norm, label_rows, show_values=True):
    """One consistency heatmap. Values are printed: only 27 cells per map."""
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    if show_values:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    continue
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=FONT_SIZES["annotation"],
                        color=_text_color(im.cmap(im.norm(mat[i, j]))))

    for j in range(1, mat.shape[1]):
        ax.axvline(j - 0.5, color=CELL_EDGE, lw=CELL_EDGE_LW)
    for i in range(1, mat.shape[0]):
        ax.axhline(i - 0.5, color=CELL_EDGE, lw=CELL_EDGE_LW)

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([EFFORT_LABELS[e] for e in CONFIRMATORY_EFFORTS],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([MODEL_LABELS_SHORT[m] for m in MODEL_ORDER]
                       if label_rows else [])
    if label_rows:
        colors = model_colors()
        for tick, model in zip(ax.get_yticklabels(), MODEL_ORDER):
            tick.set_color(colors[model])
            tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(length=0)
    return im


def plot_consistency_maps(axes, caxes, with_a, without_b, diff):
    """Three maps sharing a row, in the grammar of the score overview."""
    score_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    lim = float(np.nanmax(np.abs(diff)))
    # Consistency differences run both ways — two models are more consistent
    # unaided — so this map is genuinely diverging and centred on zero.
    diff_norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)

    im_a = _draw(axes[0], with_a, CMAPS["score"], score_norm, True)
    _draw(axes[1], without_b, CMAPS["score"], score_norm, False)
    im_d = _draw(axes[2], diff, CMAPS["diverging"], diff_norm, True)

    # Same marker the score overview uses, so a reader moving between the two
    # figures can find the ablation cell in both. Column order is passed
    # explicitly: these maps carry three effort levels, not five.
    for ax in axes:
        _mark_ablation_cell(ax, col_order=CONFIRMATORY_EFFORTS)

    pad = 8
    axes[0].set_title(PHASE_LABELS["A"], fontsize=FONT_SIZES["axes_label"],
                      pad=pad)
    axes[1].set_title(PHASE_LABELS["B"], fontsize=FONT_SIZES["axes_label"],
                      pad=pad)
    axes[2].set_title(f"difference ({PHASE_LABELS['diff']})",
                      fontsize=FONT_SIZES["axes_label"], pad=pad)

    for cax, im, label in ((caxes[0], im_a, "Method consistency"),
                           (caxes[1], im_d, "Δ consistency")):
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=FONT_SIZES["colorbar_label"])
        cb.outline.set_linewidth(SPINE_LW)
        cb.ax.tick_params(labelsize=FONT_SIZES["tick_label"], width=SPINE_LW)


def plot_consistency_summary(ax, df, rng=None):
    """Mean consistency per arm as bars, with every cell overlaid.

    Built in the grammar of the ablation bars: a bar per condition, the
    individual observations scattered over it. Here an observation is one cell
    -- one model at one effort level -- not one run, because the measure only
    exists once per cell.
    """
    rng = rng or np.random.default_rng(0)
    piv = df.pivot_table(index=["model", "effort"], columns="phase",
                         values="score")

    for i, phase in enumerate(("A", "B")):
        vals = piv[phase].to_numpy()
        m, ci = mean_ci(vals)
        ax.bar(i, m, BAR_WIDTH, color=PHASE_COLORS[phase], edgecolor="none",
               zorder=2)
        ax.errorbar(i, m, yerr=ci, color=CELL_DOT_EDGE, capsize=2,
                    elinewidth=1.0, zorder=5, linestyle="none")
        ax.plot(i + rng.uniform(-JITTER, JITTER, size=len(vals)), vals,
                linestyle="none", marker="o", ms=2.6, mfc="white",
                mec=CELL_DOT_EDGE, mew=0.6, clip_on=False, zorder=4)

    # One comparison, so one bracket and no correction to make. The test is
    # paired by cell -- same model, same effort -- which is what makes 27
    # observations rather than two groups of 27.
    _, _, p_w, _, _ = _summarise(df)
    y = 1.05
    ax.plot([0, 0, 1, 1], [y - 0.03, y, y, y - 0.03], color=CELL_DOT_EDGE,
            lw=0.7, clip_on=False, zorder=6)
    ax.text(0.5, y + 0.01, stars(p_w), ha="center", va="bottom",
            clip_on=False, fontsize=FONT_SIZES["annotation"],
            color=CELL_DOT_EDGE)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([PHASE_LABELS_SHORT["A"], PHASE_LABELS_SHORT["B"]])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Method consistency")


def _summarise(df):
    piv = df.pivot_table(index=["model", "effort"], columns="phase",
                         values="score")
    w, p_w = stats.wilcoxon(piv["A"], piv["B"])
    t, p_t = stats.ttest_rel(piv["A"], piv["B"])
    return piv, w, p_w, t, p_t


if __name__ == "__main__":
    df = load_cell_scores()
    with_a, without_b, diff = consistency_matrices()

    fig = plt.figure(figsize=(FULL_PAGE_WIDTH, 2.9))
    gs = gridspec.GridSpec(1, 7, figure=fig,
                           width_ratios=[1, 1, 0.05, 0.55, 1, 0.05, 0.12],
                           wspace=0.14)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 4)]
    caxes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 5])]
    plot_consistency_maps(axes, caxes, with_a, without_b, diff)
    add_panel_label(axes[0], "A", x=-0.30, y=1.08)
    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "consistency_maps.png"))

    fig, ax = plt.subplots(figsize=(1.6, 2.3))
    plot_consistency_summary(ax, df)
    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "consistency_summary.png"))

    piv, w, p_w, t, p_t = _summarise(df)
    print(f"cells: {len(df)} ({len(piv)} paired)")
    print(f"  with    {piv['A'].mean():.3f}")
    print(f"  without {piv['B'].mean():.3f}")
    print(f"  difference {+(piv['A'] - piv['B']).mean():.3f}")
    print(f"  Wilcoxon W={w:.1f}, p={p_w:.3g} | paired t={t:.2f}, p={p_t:.3g}")
    print(f"  higher with SpikeLab in {(piv['A'] > piv['B']).sum()} of {len(piv)}"
          f" cells, tied in {(piv['A'] == piv['B']).sum()}")
    print(f"\n  replicates leaving units unstated, per cell of 10: "
          f"with {df[df.phase == 'A'].n_underspecified.mean():.2f}, "
          f"without {df[df.phase == 'B'].n_underspecified.mean():.2f}")
    print(f"  distinct unit classes per cell: "
          f"with {df[df.phase == 'A'].n_classes.mean():.2f}, "
          f"without {df[df.phase == 'B'].n_classes.mean():.2f}")
