"""Assemble Supplementary Figure 1: method consistency across the grid.

Per-cell detail behind main-figure panel d. Rubric item 14
(``reproducible_method``) is the fraction of a cell's ten replicates that
reported the population-coupling value in mutually commensurable units --
largest agreeing group divided by attempts.

Three maps in the grammar of main-figure panels b and c: with SpikeLab, the
same models unaided, and the difference. Values are printed in every cell,
which the main figure's maps cannot do -- 27 cells per map here against 45
there.

Item 14 was graded on the confirmatory grid only, so these maps carry three
effort columns against the score overview's five. The n=3 exploratory levels
have no item-14 score at all, which is why this cannot simply mirror panel b.

Usage:  python assemble_figure_s1.py          preview only
        python assemble_figure_s1.py --tif    also write the submission TIFF
Output: FigureS1_preview.png, FigureS1.tif
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

from benchmark_v2_data import consistency_matrices, load_cell_scores
from plot_consistency import plot_consistency_maps, _summarise

# Two-thirds page width: three 9x3 maps at full width would give cells wide
# enough to look like a different measure from the main figure's.
FIG_W = FULL_PAGE_WIDTH * 0.72
FIG_H = 2.95


def main():
    with_a, without_b, diff = consistency_matrices()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # Column 3 is a spacer: the difference map needs its own colorbar, and
    # without the gap its row labels land on the score maps' bar.
    gs = gridspec.GridSpec(
        1, 6, figure=fig, width_ratios=[1, 1, 0.055, 0.50, 1, 0.055],
        left=0.085, right=0.90, top=0.86, bottom=0.20, wspace=0.16,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 4)]
    caxes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 5])]
    plot_consistency_maps(axes, caxes, with_a, without_b, diff)

    # Letters in figure coordinates, matching the main figure's convention:
    # an axes-fraction offset lands somewhere different on each map, because
    # the left map carries row labels the middle one does not.
    y = max(ax.get_position().y1 for ax in axes) + 0.045
    for letter, ax, lead in (("a", axes[0], True), ("b", axes[2], False)):
        x = 0.008 if lead else ax.get_position().x0 - 0.055
        fig.text(x, y, letter, fontsize=FONT_SIZES["panel_label"],
                 fontweight="bold", va="bottom", ha="left")

    png = os.path.join(SCRIPT_DIR, "FigureS1_preview.png")
    fig.savefig(png, dpi=FIG_DEFAULTS["dpi"], bbox_inches="tight")
    print(f"figure size: {FIG_W:.2f} x {FIG_H:.2f} in")
    print(f"saved {os.path.basename(png)}")

    if "--tif" in sys.argv:
        tif = os.path.join(SCRIPT_DIR, "FigureS1.tif")
        save_fig_submission(fig, tif)
        print(f"saved {os.path.basename(tif)}  "
              f"({os.path.getsize(tif) / 1024 / 1024:.1f} MB, "
              f"{FIG_DEFAULTS['dpi']} dpi, 8-bit RGB)")
    else:
        plt.close(fig)
        print("(pass --tif to also write the submission TIFF)")

    piv, w, p_w, t, p_t = _summarise(load_cell_scores())
    d = piv["A"] - piv["B"]
    print(f"\ncells {len(piv)}   with {piv['A'].mean():.3f}   "
          f"without {piv['B'].mean():.3f}   diff {d.mean():+.3f}")
    print(f"Wilcoxon W={w:.1f}, p={p_w:.3g}   paired t={t:.2f}, p={p_t:.3g}")
    print(f"higher with SpikeLab in {(d > 0).sum()} of {len(piv)}, "
          f"tied in {(d == 0).sum()}, lower in {(d < 0).sum()}")


if __name__ == "__main__":
    main()
