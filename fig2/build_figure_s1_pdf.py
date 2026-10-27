"""Render Supplementary Figure 1 and its caption to a single A4 PDF.

The figure is drawn by the same panel code the TIFF uses, so the PDF carries
vector artwork rather than an embedded raster: 68 KB against the TIFF's 36 MB,
and sharp at any zoom.

The caption needs bold panel letters inside running text, which matplotlib has
no rich-text primitive for. It is laid out here by PREFIX MEASUREMENT: to place
a span, the text preceding it on its line is measured as one real string and the
span is drawn at that offset. Measuring words individually and summing does not
work -- a Text extent is the bounding box of rendered ink, not the advance
width, so the per-word error varies with the glyphs and words visibly collide.

Usage:  python build_figure_s1_pdf.py
Output: Supplementary_Figure_1.pdf
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import apply_style, FONT_SIZES
apply_style()

# mathtext carries the superscript minus that Arial lacks, but renders it in its
# own font by default. Point the math fontset at Arial so the one mathtext span
# in the caption matches the text around it.
matplotlib.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
})

from benchmark_v2_data import consistency_matrices, load_cell_scores
from plot_consistency import plot_consistency_maps, _summarise

A4_W, A4_H = 8.27, 11.69

TITLE = [("Supplementary Figure 1 | Method consistency per model and "
          "reasoning-effort level.", True)]
BODY = [
    ("Rubric item 14: the fraction of a cell's ten replicates reporting "
     "population coupling in commensurable units. ", False),
    ("a", True),
    (", Both arms, shared 0–1 scale. ", False),
    ("b", True),
    (", Difference, with − without; positive in 22 of 27 cells, tied in 2. "
     "Grid means 0.78 versus 0.53 (Wilcoxon signed-rank ", False),
    ("W", True),
    (" = 31.5, ", False),
    ("P", True),
    (" = 4 × 10$^{-4}$). Black outline, the ablation cell (Fig. 2g,h). "
     "Cell-scope, so never added to the 13-item totals; graded on the "
     "confirmatory grid only, hence three effort columns against five in "
     "Fig. 2b,c. H, S, O = haiku, sonnet, opus.", False),
]


def _tokenise(runs):
    """(text, bold) runs into whitespace-delimited tokens.

    A token is a maximal run of non-space characters, carrying its own
    (text, bold) segments: weight can change inside one token, because the
    caption's bold panel letter is followed immediately by a plain comma.
    """
    tokens, cur = [], []
    for text, bold in runs:
        for i, piece in enumerate(text.split(" ")):
            if i:
                if cur:
                    tokens.append(cur)
                cur = []
            if piece:
                cur.append((piece, bold))
    if cur:
        tokens.append(cur)
    return tokens


class Layout:
    """Wraps and draws (text, bold) runs using prefix measurement."""

    def __init__(self, fig, renderer, size):
        self.fig, self.renderer, self.size = fig, renderer, size
        self.px = fig.get_figwidth() * fig.dpi

    def width(self, text, bold=False):
        if not text:
            return 0.0
        t = self.fig.text(0, 0, text, fontsize=self.size,
                          fontweight="bold" if bold else "normal")
        w = t.get_window_extent(renderer=self.renderer).width / self.px
        t.remove()
        return w

    def draw(self, runs, x0, y0, avail, leading, bold_all=False):
        """Draw wrapped text. Returns the y of the last line drawn."""
        lines, cur = [], []
        for token in _tokenise(runs):
            trial = cur + [token]
            if cur and self.width(self._plain(trial), bold_all) > avail:
                lines.append(cur)
                cur = [token]
            else:
                cur = trial
        if cur:
            lines.append(cur)

        y = y0
        for line in lines:
            for i, token in enumerate(line):
                # Offset from the measured width of everything before this
                # token on this line, spaces included, so matplotlib's own
                # metrics set the spacing rather than a summed estimate.
                #
                # Trailing spaces DO count towards a Text extent here, verified
                # by measurement: width("Supplementary ") exceeds
                # width("Supplementary") by exactly the space advance, and
                # width(a) + space + width(b) equals width("a b") to six
                # decimals. So the prefix can be measured directly.
                prefix = self._plain(line[:i]) + " " if i else ""
                x = x0 + self.width(prefix, bold_all)
                for j, (piece, bold) in enumerate(token):
                    inner = "".join(p for p, _ in token[:j])
                    self.fig.text(x + self.width(inner, bold), y, piece,
                                  fontsize=self.size, va="top", ha="left",
                                  fontweight="bold" if bold else "normal")
            y -= leading
        return y + leading

    @staticmethod
    def _plain(tokens):
        return " ".join("".join(p for p, _ in tok) for tok in tokens)


def main():
    with_a, without_b, diff = consistency_matrices()

    fig = plt.figure(figsize=(A4_W, A4_H))
    gs = gridspec.GridSpec(
        1, 6, figure=fig, width_ratios=[1, 1, 0.055, 0.50, 1, 0.055],
        left=0.10, right=0.88, top=0.955, bottom=0.72, wspace=0.16,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 4)]
    caxes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 5])]
    plot_consistency_maps(axes, caxes, with_a, without_b, diff)

    y = max(ax.get_position().y1 for ax in axes) + 0.012
    for letter, ax, lead in (("a", axes[0], True), ("b", axes[2], False)):
        x = 0.035 if lead else ax.get_position().x0 - 0.055
        fig.text(x, y, letter, fontsize=FONT_SIZES["panel_label"],
                 fontweight="bold", va="bottom", ha="left")

    fig.canvas.draw()
    lay = Layout(fig, fig.canvas.get_renderer(), FONT_SIZES["axes_label"])

    leading = 0.0165
    y = lay.draw(TITLE, 0.10, 0.665, 0.78, leading, bold_all=True)
    lay.draw(BODY, 0.10, y - leading * 1.7, 0.78, leading)

    out = os.path.join(SCRIPT_DIR, "Supplementary_Figure_1.pdf")
    # Explicit full-page bbox: apply_style() sets savefig.bbox="tight", which
    # bbox_inches=None inherits rather than overrides, cropping the page.
    fig.savefig(out, format="pdf", bbox_inches=Bbox([[0, 0], [A4_W, A4_H]]))
    plt.close(fig)
    print(f"saved {os.path.basename(out)}  "
          f"({os.path.getsize(out) / 1024:.0f} KB, A4, vector)")

    piv, w, p_w, t, p_t = _summarise(load_cell_scores())
    d = piv["A"] - piv["B"]
    print(f"caption figures: with {piv['A'].mean():.2f}, "
          f"without {piv['B'].mean():.2f}, W={w:.1f}, p={p_w:.3g}, "
          f"higher in {(d > 0).sum()} of {len(piv)}, tied {(d == 0).sum()}")


if __name__ == "__main__":
    main()
