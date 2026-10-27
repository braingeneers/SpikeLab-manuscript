"""Plot the model x effort score overview — Figure 2, heatmap panel.

Three heatmaps sharing a row: rubric score with SpikeLab (phase A), the same
models unaided (phase B), and their difference. Rows are models, columns are
reasoning-effort levels in their natural order.

The ``high`` and ``xhigh`` columns stand at n=3 against n=10 elsewhere and are
outside the pre-registered confirmatory grid. They are drawn here for the
overview but marked off by a dashed enclosure and grey italic tick labels.
The n itself belongs to the caption, not the panel. No inferential claim is
attached to these two columns.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style, save_fig, get_panel_width, add_panel_label,
    FONT_SIZES, FULL_PAGE_WIDTH, CMAPS,
)
apply_style()

from benchmark_v2_data import (
    RUBRIC_MAX, MODEL_ORDER, MODEL_LABELS_SHORT, EFFORT_ORDER, EFFORT_LABELS,
    EXPLORATORY_EFFORTS, PHASE_LABELS, ABLATION_CELL, effort_matrices,
    model_colors,
)

# Heatmap cell borders and the marking of the n=3 columns.
CELL_EDGE = "#9E9E9E"   # grey reads against both ends of parula (dark blue, yellow)
CELL_EDGE_LW = 0.5
SPINE_LW = 0.5           # house rule: heatmaps keep all four spines at 0.5 pt
MARK_COLOR = "#4D4D4D"
MARK_LW = 0.8


def _relative_luminance(rgb):
    """WCAG relative luminance of an RGB triple in 0-1."""
    c = np.asarray(rgb[:3], dtype=float)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _text_color(rgba):
    """Whichever of black or white contrasts better with the cell.

    Chosen by WCAG contrast ratio rather than a fixed luminance cutoff: parula
    turns over through a mid-range teal where a naive threshold picks white and
    lands near 2.4:1, which will not hold at 7 pt in print.
    """
    bg = _relative_luminance(rgba)
    on_black = (bg + 0.05) / 0.05
    on_white = 1.05 / (bg + 0.05)
    return "#000000" if on_black >= on_white else "#FFFFFF"


def _draw_map(ax, mat, cmap, norm, label_rows):
    """Draw one heatmap, colour only.

    Cell values are deliberately not printed: at 45 cells across three maps the
    numbers crowded the panel, and the map's job here is the pattern. The exact
    figures the caption needs are printed to stdout by this script instead.
    """
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")

    n_rows, n_cols = mat.shape

    # Thin separators between cells, drawn rather than left to the colormap.
    for j in range(1, n_cols):
        ax.axvline(j - 0.5, color=CELL_EDGE, lw=CELL_EDGE_LW)
    for i in range(1, n_rows):
        ax.axhline(i - 0.5, color=CELL_EDGE, lw=CELL_EDGE_LW)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([EFFORT_LABELS[e] for e in EFFORT_ORDER],
                       rotation=45, ha="right", rotation_mode="anchor")
    # Abbreviated and coloured to match the model legend in the cost panel, so
    # a model can be tracked between the two panels by its colour alone.
    ax.set_yticklabels([MODEL_LABELS_SHORT[m] for m in MODEL_ORDER]
                       if label_rows else [])
    if label_rows:
        colors = model_colors()
        for tick, model in zip(ax.get_yticklabels(), MODEL_ORDER):
            tick.set_color(colors[model])
            tick.set_fontweight("bold")

    # Grey italic labels flag the columns that are not at n=10.
    for tick, effort in zip(ax.get_xticklabels(), EFFORT_ORDER):
        if effort in EXPLORATORY_EFFORTS:
            tick.set_color(MARK_COLOR)
            tick.set_style("italic")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(length=0)

    return im


def _mark_exploratory(ax, n_rows):
    """Enclose the columns that are outside the confirmatory grid.

    The enclosure and the grey italic tick labels mark *which* columns differ;
    the caption carries the n. Nothing here states a number, so the two stay in
    one place and cannot drift apart.
    """
    cols = [EFFORT_ORDER.index(e) for e in EXPLORATORY_EFFORTS]
    x0, x1 = min(cols) - 0.5, max(cols) + 0.5

    ax.add_patch(Rectangle(
        (x0, -0.5), x1 - x0, n_rows,
        fill=False, edgecolor=MARK_COLOR, lw=MARK_LW,
        linestyle=(0, (2.5, 1.5)), zorder=6, clip_on=False,
    ))


def _mark_ablation_cell(ax, col_order=None):
    """Outline the one cell the component ablation was run in.

    Drawn as a white stroke with a black one inside it: a single-colour outline
    would disappear against one end or the other of these colormaps, since the
    cells it has to sit on run from near-black to near-white.

    ``col_order`` names the effort levels the axes actually shows, because the
    supplementary consistency maps carry only the three confirmatory levels
    against this overview's five. Both orderings happen to put ``medium`` at
    index 1, so passing it changes nothing today -- but the agreement is a
    coincidence, and a silently misplaced marker is the failure it would cause.
    """
    i = MODEL_ORDER.index(ABLATION_CELL[0])
    j = (col_order or EFFORT_ORDER).index(ABLATION_CELL[1])
    for color, lw in (("white", 2.2), ("black", 1.0)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=color, lw=lw, zorder=7))


def plot_effort_heatmap(axes, caxes, mean_a, mean_b, diff):
    """Draw the three-map overview onto pre-created axes.

    Parameters
    ----------
    axes : list of 3 Axes
        Phase A, phase B and difference maps, left to right.
    caxes : list of 2 Axes
        Colorbar axes: one shared by the two score maps, one for the difference.
    mean_a, mean_b, diff : (9, 5) arrays
        Cell means following MODEL_ORDER and EFFORT_ORDER.
    """
    n_rows = len(MODEL_ORDER)

    # Both score maps share one scale so they can be read against each other.
    score_norm = matplotlib.colors.Normalize(vmin=0, vmax=RUBRIC_MAX)
    # No cell difference is negative, so the difference map runs from zero
    # rather than straddling it: a symmetric scale would spend half the colormap
    # on an empty range and flatten the 27 cells into one shade.
    diff_cmap = CMAPS["heatmap"]
    diff_norm = matplotlib.colors.Normalize(
        vmin=0, vmax=np.ceil(np.nanmax(diff)),
    )

    im_a = _draw_map(axes[0], mean_a, CMAPS["score"], score_norm, True)
    _draw_map(axes[1], mean_b, CMAPS["score"], score_norm, False)
    im_d = _draw_map(axes[2], diff, diff_cmap, diff_norm, True)

    for ax in axes:
        _mark_exploratory(ax, n_rows)
        _mark_ablation_cell(ax)

    title_pad = 8
    axes[0].set_title(PHASE_LABELS["A"], fontsize=FONT_SIZES["axes_label"],
                      pad=title_pad)
    axes[1].set_title(PHASE_LABELS["B"], fontsize=FONT_SIZES["axes_label"],
                      pad=title_pad)
    axes[2].set_title(f"difference ({PHASE_LABELS['diff']})",
                      fontsize=FONT_SIZES["axes_label"], pad=title_pad)

    for cax, im, label in (
        (caxes[0], im_a, f"Rubric score (of {RUBRIC_MAX})"),
        (caxes[1], im_d, f"\u0394 score (of {RUBRIC_MAX})"),
    ):
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=FONT_SIZES["colorbar_label"])
        cb.outline.set_linewidth(SPINE_LW)
        cb.ax.tick_params(labelsize=FONT_SIZES["tick_label"], width=SPINE_LW)


if __name__ == "__main__":
    mean_a, mean_b, diff, n_a, n_b = effort_matrices()

    fig = plt.figure(figsize=(FULL_PAGE_WIDTH, 2.75))
    # Spacer columns keep each colorbar's label clear of the next heatmap.
    gs = gridspec.GridSpec(
        1, 7, figure=fig,
        width_ratios=[1, 1, 0.05, 0.62, 1, 0.05, 0.16],
        wspace=0.12,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 4)]
    caxes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 5])]

    plot_effort_heatmap(axes, caxes, mean_a, mean_b, diff)
    add_panel_label(axes[0], "A", x=-0.34, y=1.06)

    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "effort_heatmap.png"))

    # Numbers the caption has to carry.
    conf = [j for j, e in enumerate(EFFORT_ORDER) if e not in EXPLORATORY_EFFORTS]
    d_conf = diff[:, conf]
    print(f"confirmatory grid: {d_conf.size} cells, "
          f"diff range {d_conf.min():+.2f} to {d_conf.max():+.2f}, "
          f"{(d_conf <= 0).sum()} at or below zero")
    d_exp = diff[:, [EFFORT_ORDER.index(e) for e in EXPLORATORY_EFFORTS]]
    print(f"exploratory (n=3): {d_exp.size} cells, "
          f"diff range {d_exp.min():+.2f} to {d_exp.max():+.2f}, "
          f"{(d_exp <= 0).sum()} at or below zero")
    print(f"phase A range {np.nanmin(mean_a):.2f}-{np.nanmax(mean_a):.2f}; "
          f"phase B range {np.nanmin(mean_b):.2f}-{np.nanmax(mean_b):.2f}")
    print("cell n: A", sorted(set(n_a.ravel().tolist())),
          "B", sorted(set(n_b.ravel().tolist())))
