"""Plot the component ablation — Figure 2, ablation panel.

Four arms at sonnet-4-6 / medium effort, n=10 each, crossing the two things
SpikeLab supplies:

    A  instructions + tools     C  instructions only
    D  tools only               B  neither

Left    total score per arm, individual runs overlaid, with the six pairwise
        contrasts bracketed above.
Right   mean score per rubric item, split into the six instructed-behaviour and
        the seven analytical-correctness items.

The per-item map is DESCRIPTIVE ONLY. No statistical test is run on the panel
split; only the combined 13-item score is tested. Nothing on the right half
carries a p-value and it must not be described as if it did.

A vs C is the contrast that matters most and the only one that is not an upper
bound: both arms carry the instructions, so it isolates what the tools add.
Every contrast involving B or D compares against an arm that was never asked to
perform the six instructed-behaviour items.
"""

import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style, save_fig, add_panel_label,
    COLORS, CMAPS, FONT_SIZES, FULL_PAGE_WIDTH,
)
apply_style()

from benchmark_v2_data import (
    RUBRIC_MAX, ARM_ORDER, ARM_LABELS, RUBRIC_ITEMS, CATEGORY_LABELS,
    load_ablation_runs, load_ablation_items, load_ablation_consistency,
)
from panel_common import bh, mean_ci, stars, welch_ci, welch_p  # noqa: F401
from plot_effort_heatmap import SPINE_LW, CELL_EDGE, CELL_EDGE_LW

# Arms shade from the full library to nothing, reusing the two phase colours at
# the ends so the panel ties visually to the with/without panels.
ARM_COLORS = {
    "A": COLORS["palette"][0],
    "C": COLORS["palette"][1],
    "D": COLORS["palette"][3],
    "B": COLORS["palette"][4],
}
RUN_DOT_COLOR = "#4D4D4D"
# The p-value matrix keeps the figure-wide tick size; only its arm labels
# are shortened, because four of them have to fit across a narrow axis.
INSET_FONTSIZE = FONT_SIZES["tick_label"]
INSET_ARM_LABELS = {"A": "I&T", "C": "I", "D": "T", "B": "nei"}
BAR_WIDTH = 0.62
JITTER = 0.14


def ablation_stats(df, rng=None):
    """Arm means with CI, per-run values, and the six pairwise contrasts."""
    rng = rng or np.random.default_rng(0)
    arms = {a: df[df.arm == a].total.to_numpy() for a in ARM_ORDER}

    means, cis, ns, jitter = [], [], [], {}
    for a in ARM_ORDER:
        m, ci = mean_ci(arms[a])
        means.append(m)
        cis.append(ci)
        ns.append(len(arms[a]))
        jitter[a] = rng.uniform(-JITTER, JITTER, size=len(arms[a]))

    pairs, diff, p = [], [], []
    for x, y in combinations(ARM_ORDER, 2):
        d, _ = welch_ci(arms[x], arms[y])
        pairs.append((x, y))
        diff.append(d)
        p.append(welch_p(arms[x], arms[y]))

    return {
        "arms": arms, "jitter": jitter,
        "means": np.array(means), "cis": np.array(cis), "n": np.array(ns),
        "pairs": pairs, "diff": np.array(diff),
        "p": np.array(p), "p_bh": bh(p),
    }


def plot_pvalue_matrix(ax, res, cax=None):
    """All six pairwise contrasts as a -log10(p) matrix.

    Follows the idiom of the p-value matrices in Figure 6 — viridis, a black
    diagonal for the self-comparison, red dots where the contrast is
    significant, hairline spines and a slim colorbar. It replaces the stacked
    bracket-and-star notation, which needed six nested levels and rose a third
    of the way up the bar axis for something a 4 x 4 matrix says at a glance.

    Every contrast here survives correction, so every off-diagonal cell carries
    a dot. What the colour adds over the dots is the magnitude: p_BH ranges from
    5.9e-5 for instructions-plus-tools against instructions alone, to 4.1e-10
    for the full library against neither.
    """
    k = len(ARM_ORDER)
    idx = {a: i for i, a in enumerate(ARM_ORDER)}
    neg_log_p = np.full((k, k), np.nan)
    sig = np.zeros((k, k), dtype=bool)
    for (a, b), adj in zip(res["pairs"], res["p_bh"]):
        i, j = idx[a], idx[b]
        neg_log_p[i, j] = neg_log_p[j, i] = -np.log10(adj)
        sig[i, j] = sig[j, i] = adj < 0.05

    # Square cells shrink the axes box below the width it was given, and
    # matplotlib centres what is left. Anchor west so the matrix sits under the
    # left edge of the bars rather than floating in the middle of the column.
    ax.set_anchor("W")

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")
    vmax = float(np.nanmax(neg_log_p))
    im = ax.imshow(neg_log_p, cmap=cmap, aspect="equal",
                   interpolation="none", vmin=0, vmax=vmax)
    for i in range(k):
        for j in range(k):
            if i != j and sig[i, j]:
                ax.plot(j, i, "o", color="red", markersize=1.8,
                        markeredgewidth=0)

    labels = [INSET_ARM_LABELS[a] for a in ARM_ORDER]
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, fontsize=INSET_FONTSIZE)
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels, fontsize=INSET_FONTSIZE)
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    if cax is None:
        cax = ax.inset_axes([1.06, 0.0, 0.11, 1.0])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r"$-\log_{10}\,p$", fontsize=INSET_FONTSIZE,
                 labelpad=2)
    cb.set_ticks([0, round(vmax)])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=INSET_FONTSIZE, width=0.5, length=1.5, pad=1.0)
    return ax


def plot_ablation_scores(ax, res):
    """Bars per arm with the individual runs overlaid."""
    x = np.arange(len(ARM_ORDER))

    for i, arm in enumerate(ARM_ORDER):
        ax.bar(i, res["means"][i], BAR_WIDTH, color=ARM_COLORS[arm],
               edgecolor="none", zorder=2)
        # Unclipped: a run scoring the full 13 sits exactly on the axis top,
        # and the marker would otherwise be cut in half by it.
        ax.plot(i + res["jitter"][arm], res["arms"][arm], linestyle="none",
                marker="o", ms=2.6, mfc="white", mec=RUN_DOT_COLOR, mew=0.6,
                clip_on=False, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, RUBRIC_MAX)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_ylabel(f"Rubric score (of {RUBRIC_MAX})")


def plot_ablation_items(ax, cax, items, consistency=None):
    """Mean score per rubric item per arm. Descriptive; no test is applied.

    ``consistency`` appends rubric item 14 as a final column. It shares the
    0-1 scale but is NOT the same kind of quantity as the thirteen before
    it: those are means over ten runs of a 0/0.5/1 item, this is a single
    cell-level fraction with no replicates of its own. A heavier divider
    and its own block label keep the two from being read as one series.
    """
    labels = [label.replace(chr(10), " ") for _, _, label in RUBRIC_ITEMS]
    if consistency is not None:
        items = np.hstack([items, np.asarray(consistency)[:, None]])
        labels = labels + ["method consistency"]

    im = ax.imshow(items, cmap=CMAPS["score"], vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")

    for j in range(1, items.shape[1]):
        ax.axvline(j - 0.5, color=CELL_EDGE, lw=CELL_EDGE_LW)
    for i in range(1, items.shape[0]):
        ax.axhline(i - 0.5, color=CELL_EDGE, lw=CELL_EDGE_LW)

    # Divider between the two rubric categories, with a label over each block.
    cats = [cat for _, cat, _ in RUBRIC_ITEMS]
    split = cats.index("analytical_correctness")
    ax.axvline(split - 0.5, color="black", lw=1.2, zorder=6)
    blocks = [("instructed_behaviour", 0, split),
              ("analytical_correctness", split, len(cats))]
    if consistency is not None:
        # Heavier rule: what follows is cell-scope, not a per-run mean. It gets
        # no block label of its own -- one column is far narrower than the words
        # would be, and a centred "cell scope" ran straight into "analytical
        # correctness" next to it. The rule, the column name and the caption
        # carry the distinction instead.
        ax.axvline(len(cats) - 0.5, color="black", lw=1.8, zorder=6)
    for cat, lo, hi in blocks:
        ax.text((lo + hi - 1) / 2, -0.72, CATEGORY_LABELS[cat], ha="center",
                va="bottom", fontsize=FONT_SIZES["annotation"],
                fontstyle="italic")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       rotation_mode="anchor",
                       fontsize=FONT_SIZES["tick_label"])
    ax.set_yticks(range(len(ARM_ORDER)))
    ax.set_yticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(length=0)

    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Mean item score", fontsize=FONT_SIZES["colorbar_label"],
                 labelpad=2)
    cb.set_ticks([0, 1])
    cb.outline.set_linewidth(SPINE_LW)
    cb.ax.tick_params(labelsize=FONT_SIZES["tick_label"], width=SPINE_LW,
                      pad=1.5)


if __name__ == "__main__":
    df = load_ablation_runs()
    res = ablation_stats(df)
    items = load_ablation_items()
    consistency = load_ablation_consistency()

    fig = plt.figure(figsize=(FULL_PAGE_WIDTH, 3.3))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2.2],
                           wspace=0.34)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 0], height_ratios=[2.0, 1.0], hspace=0.62)
    ax_scores = fig.add_subplot(gs_left[0])
    ax_pvals = fig.add_subplot(gs_left[1])
    ax_items = fig.add_subplot(gs[0, 1])
    # Inset rather than a gridspec column: the colorbar should hug the heatmap,
    # and a column of its own inherits the figure-wide wspace.
    cax = ax_items.inset_axes([1.012, 0.0, 0.016, 1.0])

    plot_ablation_scores(ax_scores, res)
    plot_pvalue_matrix(ax_pvals, res)
    plot_ablation_items(ax_items, cax, items,
                        consistency=[consistency[a] for a in ARM_ORDER])
    add_panel_label(ax_scores, "D", x=-0.30, y=1.30)
    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "ablation.png"))

    # Numbers the caption has to carry.
    for i, a in enumerate(ARM_ORDER):
        print(f"  {a} {ARM_LABELS[a]:<10} "
              f"{res['means'][i]:5.2f} +-{res['cis'][i]:.2f}  (n={res['n'][i]})")
    print()
    for (a, b), d, p, adj in zip(res["pairs"], res["diff"], res["p"],
                                 res["p_bh"]):
        print(f"  {a}-{b}  {d:+.2f}  p={p:.3g}  p_BH={adj:.3g}  {stars(adj)}")

    # Per-item means, now that the map itself is colour only.
    print()
    header = "".join(f"{ARM_LABELS[a]:>9}" for a in ARM_ORDER)
    print(f"  {'rubric item':<32}{header}")
    for j, (item, _cat, _label) in enumerate(RUBRIC_ITEMS):
        row = "".join(f"{items[i, j]:>9.2f}" for i in range(len(ARM_ORDER)))
        print(f"  {item:<32}{row}")
