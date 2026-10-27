"""Assemble the revised Figure 2.

Layout, four rows:

    A                     task prompts (full width)
    B | C | D             with/without maps | difference map | consistency
    E | F                 effort response   | model response
    G | H | I             ablation | per-item map | cost vs score

Panels are rendered into this figure's own axes by calling each panel module's
plotting function — never stitched from the PNG previews.

D carries rubric item 14 (``reproducible_method``, cell-scope): whether a cell's
ten replicates computed population coupling the same way. Graded by
tools/grade_cell.py in a second pass, and deliberately NOT folded
into the 13-item score — it exists once per cell, not once per run, so adding it
to a per-run total would contribute between-cell signal with no within-cell
variance and make every per-run standard error here anti-conservative.

Usage:  python assemble_figure2.py
Output: Figure2_preview.png and Figure2.tif
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style, save_fig_submission, add_panel_label,
    FIG_DEFAULTS, FONT_SIZES, FULL_PAGE_WIDTH,
)
apply_style()

from benchmark_v2_data import (
    load_sweep_runs, load_sweep_with_cost, load_ablation_runs,
    load_ablation_items, load_ablation_consistency, effort_matrices,
    model_colors, load_cell_scores, ARM_ORDER,
)
from plot_task_prompts import load_prompts, plot_task_prompts
from plot_effort_heatmap import plot_effort_heatmap
from plot_effort_response import effort_response, plot_effort_response
from plot_model_response import model_response, plot_model_response
from plot_ablation import (
    ablation_stats, plot_ablation_scores, plot_ablation_items,
    plot_pvalue_matrix,
)
from plot_consistency import plot_consistency_summary
from plot_cost_score import (
    cost_score_cells, pareto_frontier, plot_cost_score_runs, build_legend,
)

# Row heights in inches. The prompt strip is text only and needs little; the
# three data rows are what the panels were designed at, trimmed slightly.
# Row heights and the gaps between them, both in inches. Explicit spacer rows
# rather than a gridspec hspace, because hspace applies one fraction to every
# gap and the map row and the bottom row each need a little more clearance for
# their column titles and panel letters than the middle row does.
ROW_H = [0.56, 0.66, 1.78, 0.50, 2.02, 0.66, 2.36]
FIG_H = sum(ROW_H) / 0.925           # top and bottom margins below


# Panel letters are positioned from the figure's own geometry rather than from
# per-axes offsets, which is the only way they align: an offset in axes
# fractions lands somewhere different for every panel, because the panels are
# different sizes.
LABEL_X_LEFT = 0.008      # figure fraction, shared by every row-leading label
LABEL_DX = 0.036          # how far left of its axes a non-leading label sits


def label_panels(fig, rows):
    """Place panel letters, aligned horizontally by row and vertically by column.

    ``rows`` is a list of (pad, [(letter, axes), ...]). Every label in a row
    shares one y, taken from the tallest axes in that row plus ``pad`` — the pad
    differs by row because some rows carry column titles above their axes and
    some do not. Every row-leading label shares one x. Labels further along a
    row sit a fixed distance left of their own axes, so they align vertically
    only where the underlying columns happen to align, which is as far as the
    layout allows.
    """
    for pad, entries in rows:
        y = max(ax.get_position().y1 for _, ax in entries) + pad
        for i, (letter, ax) in enumerate(entries):
            x = LABEL_X_LEFT if i == 0 else ax.get_position().x0 - LABEL_DX
            fig.text(x, y, letter, fontsize=FONT_SIZES["panel_label"],
                     fontweight="bold", va="bottom", ha="left")


def main():
    sweep = load_sweep_runs()
    cost_df = load_sweep_with_cost()
    abl = load_ablation_runs()

    fig = plt.figure(figsize=(FULL_PAGE_WIDTH, FIG_H))
    outer = gridspec.GridSpec(
        len(ROW_H), 1, figure=fig, height_ratios=ROW_H,
        left=0.065, right=0.985, top=0.975, bottom=0.05, hspace=0.0,
    )
    ROW_PROMPTS, ROW_MAPS, ROW_RESPONSE, ROW_BOTTOM = 0, 2, 4, 6

    # ---- A: task prompts ------------------------------------------------
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[ROW_PROMPTS],
                                            wspace=0.18)
    ax_prompts = [fig.add_subplot(gs_a[0, i]) for i in range(4)]
    tasks, prefix = load_prompts()
    plot_task_prompts(ax_prompts, tasks, prefix)

    # ---- B: model x effort overview | C: consistency placeholder --------
    # Column 6 is a spacer: D carries its own y-axis label and tick labels, and
    # without it they are drawn straight over the difference map's colorbar.
    gs_b = gridspec.GridSpecFromSubplotSpec(
        1, 8, subplot_spec=outer[ROW_MAPS],
        width_ratios=[1, 1, 0.055, 0.42, 1, 0.055, 0.62, 0.80], wspace=0.16,
    )
    ax_maps = [fig.add_subplot(gs_b[0, i]) for i in (0, 1, 4)]
    caxes = [fig.add_subplot(gs_b[0, 2]), fig.add_subplot(gs_b[0, 5])]
    mean_a, mean_b, diff, _, _ = effort_matrices()
    plot_effort_heatmap(ax_maps, caxes, mean_a, mean_b, diff)

    ax_consistency = fig.add_subplot(gs_b[0, 7])
    plot_consistency_summary(ax_consistency, load_cell_scores())

    # ---- D: effort response | E: model response -------------------------
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[ROW_RESPONSE], width_ratios=[1, 1.55],
        wspace=0.30,
    )
    gs_d = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_c[0, 0], height_ratios=[2.0, 1.0], hspace=0.14)
    ax_d1 = fig.add_subplot(gs_d[0])
    ax_d2 = fig.add_subplot(gs_d[1], sharex=ax_d1)
    plot_effort_response(ax_d1, ax_d2, effort_response(sweep))

    gs_e = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_c[0, 1], height_ratios=[2.0, 1.0], hspace=0.14)
    ax_e1 = fig.add_subplot(gs_e[0])
    ax_e2 = fig.add_subplot(gs_e[1], sharex=ax_e1)
    plot_model_response(ax_e1, ax_e2, model_response(sweep))

    # ---- F: ablation | G: cost vs score ---------------------------------
    gs_f = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[ROW_BOTTOM], width_ratios=[2.20, 1],
        wspace=0.28,
    )
    gs_abl = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_f[0, 0], width_ratios=[1, 2.05], wspace=0.26)
    gs_bars = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_abl[0, 0], height_ratios=[2.0, 1.5],
        hspace=0.42)
    ax_bars = fig.add_subplot(gs_bars[0])
    ax_pvals = fig.add_subplot(gs_bars[1])
    # The item map is shorter than its column: 4 rows over the full row
    # height made very elongated cells, and the rotated item labels hang
    # into the space left below.
    gs_items = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_abl[0, 1], height_ratios=[0.80, 0.20])
    ax_items = fig.add_subplot(gs_items[0])
    cax_items = ax_items.inset_axes([1.015, 0.0, 0.02, 1.0])
    res_abl = ablation_stats(abl)
    plot_ablation_scores(ax_bars, res_abl)
    plot_pvalue_matrix(ax_pvals, res_abl)
    consistency = load_ablation_consistency()
    plot_ablation_items(ax_items, cax_items, load_ablation_items(),
                        consistency=[consistency[a] for a in ARM_ORDER])

    # G is shorter than its column so the scatter is less elongated; its
    # model legend hangs into the space left below.
    gs_cost = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_f[0, 1], height_ratios=[0.80, 0.20])
    ax_cost = fig.add_subplot(gs_cost[0])
    cells = cost_score_cells(cost_df)
    colors = model_colors()
    plot_cost_score_runs(ax_cost, cost_df, cells, pareto_frontier(cells), colors)
    build_legend(ax_cost, colors)

    # Lower case: Nature Communications sets panel letters in lower case, and
    # every caption written for this figure refers to them that way. The
    # figure_code style guide still specifies upper case, but it was written
    # against eLife's guidelines and predates the change of target journal.
    label_panels(fig, [
        (0.010, [("a", ax_prompts[0])]),
        (0.026, [("b", ax_maps[0]), ("c", ax_maps[2]), ("d", ax_consistency)]),
        (0.008, [("e", ax_d1), ("f", ax_e1)]),
        (0.008, [("g", ax_bars), ("h", ax_items), ("i", ax_cost)]),
    ])

    # save_fig and save_fig_submission both close the figure, so they cannot be
    # chained: the preview is written directly and the TIFF helper takes the
    # figure last. The TIFF is opt-in because at 900 dpi it is a ~170 MB render
    # and the preview is what iteration needs.
    png = os.path.join(SCRIPT_DIR, "Figure2_preview.png")
    fig.savefig(png, dpi=FIG_DEFAULTS["dpi"], bbox_inches="tight")
    print(f"figure size: {FULL_PAGE_WIDTH:.2f} x {FIG_H:.2f} in")
    print(f"saved {os.path.basename(png)}")

    if "--tif" in sys.argv:
        tif = os.path.join(SCRIPT_DIR, "Figure2.tif")
        save_fig_submission(fig, tif)
        size_mb = os.path.getsize(tif) / 1024 / 1024
        print(f"saved {os.path.basename(tif)}  ({size_mb:.1f} MB, "
              f"{FIG_DEFAULTS['dpi']} dpi, 8-bit RGB)")
    else:
        plt.close(fig)
        print("(pass --tif to also write the submission TIFF)")


if __name__ == "__main__":
    main()
