"""Shared drawing for the two-tier "with vs without" panels of Figure 2.

Both the effort panel and the model panel show the same thing against a
different x axis: the two arms above, the gap between them with its interval
below. They are drawn through these helpers so the two stay identical by
construction — the comparison between them is the point, and it only works if
the only difference is the axis.
"""

import os
import sys

import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import FONT_SIZES, LINE_WIDTHS

from benchmark_v2_data import RUBRIC_MAX, PHASE_LABELS, PHASE_COLORS

GAP_COLOR = "#4D4D4D"
REF_COLOR = "#9E9E9E"

# Both gap tiers share one scale, so a gap in the effort panel and a gap in the
# model panel are the same size on the page.
GAP_YLIM = (-0.8, 8.8)
GAP_YTICKS = [0, 2, 4, 6, 8]

MARKER_SIZE = 4

# The omnibus test governing each axis, quoted from stats_sweep.py. Held here as
# text rather than recomputed, so the panel and the tool cannot disagree; if the
# tool's output changes these must be updated with it.
# Exponents go through mathtext: Arial has no superscript-minus glyph, so a
# literal "10⁻²⁸" renders as a missing-glyph box.
# Only the p-value is shown; the F statistic and its degrees of freedom belong
# in the caption, where there is room to say which model produced them.
GLM_TESTS = {
    "effort": "phase × effort:  p = 0.86",
    "model": r"phase × model:  p = 3$\times$10$^{-28}$",
}

# The x-axis factor's main effect, tested inside each arm on its own. The
# interaction above says whether the factor moves the two arms differently; it
# is silent on whether the factor moves either arm at all, and for effort it
# does move both. Also from stats_sweep.py.
ARM_TESTS = {
    "effort": {
        "A": "effort: p = 0.011",
        "B": r"effort: p = 2$\times$10$^{-5}$",
    },
    "model": {
        "A": r"model: p = 4$\times$10$^{-86}$",
        "B": r"model: p = 1$\times$10$^{-64}$",
    },
}


def mean_ci(values):
    """Mean and half-width of its 95% CI."""
    v = np.asarray(values, dtype=float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def welch_ci(a, b):
    """Difference of means and its Welch 95% CI half-width.

    Welch rather than pooled: the two arms have markedly unequal variance, which
    is treated as a result rather than something to average away.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return a.mean() - b.mean(), stats.t.ppf(0.975, dof) * se


def welch_p(a, b):
    """Two-sided Welch p for the with-vs-without contrast at one point.

    Same test the plotted interval comes from, so a marked point and an interval
    clear of zero can never disagree.
    """
    return float(stats.ttest_ind(np.asarray(a, dtype=float),
                                 np.asarray(b, dtype=float),
                                 equal_var=False).pvalue)


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m, adj, prev = len(p), np.empty(m := len(p)), 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * m / (m - rank))
        adj[idx] = prev
    return adj


def stars(p):
    """Conventional significance marker for an adjusted p-value."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def plot_arms(ax, x, means, cis, groups=None, legend_loc="lower right",
              marks=None, arm_tests=None, arm_test_xy=(0.02, 0.99)):
    """Draw both arms as lines with 95% CI error bars.

    ``groups`` is a list of index lists. The line is drawn separately within
    each group and never across them, so an axis whose categories are only
    partly ordered — model versions within a family, but not between families —
    does not imply a continuum it does not have. Defaults to one group.
    """
    x = np.asarray(x)
    if groups is None:
        groups = [list(range(len(x)))]

    for phase in ("A", "B"):
        labelled = False
        for group in groups:
            idx = np.asarray(group)
            ax.errorbar(
                x[idx], np.asarray(means[phase])[idx],
                yerr=np.asarray(cis[phase])[idx],
                color=PHASE_COLORS[phase], lw=LINE_WIDTHS["mean_trace"],
                marker="o", ms=MARKER_SIZE, capsize=2, elinewidth=1.0, zorder=5,
                # Unclipped: the ceiling models sit at 12.8 of 13, so the top
                # half of their marker falls outside the axis and would be
                # sliced off by it.
                clip_on=False,
                label=None if labelled else PHASE_LABELS[phase],
            )
            labelled = True

    ax.set_ylim(0, RUBRIC_MAX)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_ylabel(f"Rubric score (of {RUBRIC_MAX})")
    ax.legend(loc=legend_loc, fontsize=FONT_SIZES["legend"], frameon=False,
              handlelength=1.4, borderaxespad=0.2)

    # The x-axis factor's own effect, tested separately inside each arm and
    # coloured to match it. The gap tier below carries the interaction; these
    # say whether each arm responds to the factor at all, which the interaction
    # is silent about.
    if arm_tests:
        x0, y0 = arm_test_xy
        for i, phase in enumerate(("A", "B")):
            if phase in arm_tests:
                ax.text(x0, y0 - 0.075 * i, arm_tests[phase],
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=FONT_SIZES["annotation"],
                        color=PHASE_COLORS[phase])

    # Per-point contrast markers, above the treated arm. Drawn unclipped: the
    # ceiling models sit at 12.8 of 13, so the marker falls outside the axis
    # rather than the axis being padded past the top of the rubric scale.
    if marks is not None:
        top = np.asarray(means["A"]) + np.asarray(cis["A"])
        for xi, hi, mark in zip(x, top, marks):
            ax.text(xi, hi + 0.30, mark, ha="center", va="bottom",
                    clip_on=False, zorder=6,
                    fontsize=FONT_SIZES["annotation"], color=GAP_COLOR)


def plot_gap_tier(ax, x, diff, ci_lo, ci_hi, pooled, test_label=None):
    """Draw the gap and its interval against zero and the pooled gap.

    The dashed line is the pooled gap across everything, left unlabelled: the
    caption names it, and the panel is quieter without a number floating in it.

    ``test_label`` is the one omnibus test that governs this axis, printed in
    the corner: the points carry estimates and intervals, and the single test
    says whether the pattern across them is real. The per-point contrast markers
    live on the arms tier above, next to the scores they compare.
    """
    x = np.asarray(x)
    diff = np.asarray(diff)

    ax.axhline(0, color=REF_COLOR, lw=0.8, zorder=1)
    ax.axhline(pooled, color=REF_COLOR, lw=0.8, linestyle=(0, (2.5, 1.5)),
               zorder=1)
    ax.errorbar(x, diff, yerr=[diff - np.asarray(ci_lo), np.asarray(ci_hi) - diff],
                linestyle="none", marker="o", ms=MARKER_SIZE, color=GAP_COLOR,
                capsize=2, elinewidth=1.0, zorder=5)

    ax.set_ylim(*GAP_YLIM)
    ax.set_yticks(GAP_YTICKS)
    ax.set_ylabel(f"\u0394 score (of {RUBRIC_MAX})")

    if test_label:
        # Sits in the empty band between the zero line and the lowest interval,
        # which is clear in both panels; the top of the axis is not, since the
        # largest per-model gap reaches into it.
        ax.text(0.015, 0.15, test_label, transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=FONT_SIZES["annotation"], color=GAP_COLOR)
