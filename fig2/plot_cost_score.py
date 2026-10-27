"""Plot score against cost — the cost-matched comparison.

Score against dollars spent, both arms, all efforts. If SpikeLab at `medium`
sits above the baseline's entire effort curve, the library is both cheaper
and better.

One point per model x phase x effort cell (9 x 2 x 3 = 54). Colour is the model,
family hue tinted by generation; marker is the arm. The three effort levels of
one model and arm are joined, so each thread shows what buying more reasoning
budget does to cost and score together.

The dashed staircase is the Pareto frontier: the cells that nothing else beats
on both axes at once. It is the direct answer to "what is worth paying for" —
anything below and right of it is dominated by some cell that is cheaper *and*
scores higher.

Cost is measured API spend per run, from the run manifests. It is not the cost
of the library, and it excludes the human time that the rubric's instructed
behaviours are meant to save, so it understates the case for the treated arm.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style, save_fig, add_panel_label, FONT_SIZES, FULL_PAGE_WIDTH,
)
apply_style()

from benchmark_v2_data import (
    RUBRIC_MAX, CONFIRMATORY_EFFORTS, MODEL_ORDER, MODEL_LABELS,
    MODEL_LABELS_SHORT, PHASE_LABELS_SHORT, model_colors,
    load_sweep_with_cost,
)

# Circle for the treated arm, square for the baseline. Shape alone carries the
# arm; both are filled with the model colour, because leaving the baseline
# unfilled threw away the colour channel for half the data.
PHASE_MARKERS = {"A": "o", "B": "s"}
THREAD_COLOR_ALPHA = 0.45
FRONTIER_COLOR = "#4D4D4D"
# Cell means take a dark ring, which separates them from the run cloud behind
# them in the per-run view and reads as deliberate in the means-only view.
MEAN_EDGE = "#333333"


def cost_score_cells(df):
    """Mean cost and mean score for every model x phase x effort cell."""
    cells = []
    for model in MODEL_ORDER:
        for phase in ("A", "B"):
            for effort in CONFIRMATORY_EFFORTS:
                sub = df[(df.model == model) & (df.phase == phase)
                         & (df.effort == effort)]
                if sub.empty:
                    continue
                cells.append({
                    "model": model, "phase": phase, "effort": effort,
                    "cost": sub.cost_usd.mean(), "score": sub.total.mean(),
                    "n": len(sub),
                })
    return cells


def pareto_frontier(cells):
    """Cells not dominated by any other on both cost (lower) and score (higher)."""
    keep = []
    for a in cells:
        dominated = any(
            b is not a and b["cost"] <= a["cost"] and b["score"] >= a["score"]
            and (b["cost"] < a["cost"] or b["score"] > a["score"])
            for b in cells
        )
        if not dominated:
            keep.append(a)
    return sorted(keep, key=lambda c: c["cost"])


def plot_cost_score(ax, cells, frontier, colors):
    """Draw the cost-score scatter with its effort threads and frontier."""
    # Threads first: the three effort levels of one model and arm, in order.
    for model in MODEL_ORDER:
        for phase in ("A", "B"):
            thread = [c for e in CONFIRMATORY_EFFORTS
                      for c in cells
                      if c["model"] == model and c["phase"] == phase
                      and c["effort"] == e]
            if len(thread) > 1:
                ax.plot([c["cost"] for c in thread], [c["score"] for c in thread],
                        color=colors[model], alpha=THREAD_COLOR_ALPHA, lw=0.7,
                        zorder=2)

    # The frontier staircase, drawn under the points.
    if frontier:
        xs, ys = [], []
        for c in frontier:
            if xs:
                xs.append(c["cost"])
                ys.append(ys[-1])
            xs.append(c["cost"])
            ys.append(c["score"])
        ax.plot(xs, ys, color=FRONTIER_COLOR, lw=0.9,
                linestyle=(0, (3, 1.6)), zorder=3)

    for c in cells:
        # Unclipped: opus-5 with SpikeLab at medium scores the full 13, landing
        # exactly on the axis top, and the marker would be cut in half by it.
        ax.plot(c["cost"], c["score"], marker=PHASE_MARKERS[c["phase"]],
                ms=5.5, mew=0.9, zorder=5, clip_on=False,
                mfc=colors[c["model"]], mec=MEAN_EDGE, linestyle="none")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per run (USD)")
    ax.set_ylabel(f"Rubric score (of {RUBRIC_MAX})")
    ax.set_ylim(0, RUBRIC_MAX)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.minorticks_off()


def plot_cost_score_runs(ax, df, cells, frontier, colors):
    """Every run as its own point, with the cell means over the top.

    Same encoding as the cell-mean version. The frontier is still the one
    computed on cell means: a frontier over individual runs is an order
    statistic over 539 draws, so it would trace the luckiest run in each cell
    and promise a price-performance nobody can buy on purpose.
    """
    # Individual runs are filled in both arms. At this size an unfilled square
    # is almost all outline, and the model colour disappeared; the marker shape
    # still carries the arm, and the cell means below keep the fill convention.
    for _, r in df.iterrows():
        ax.plot(r.cost_usd, r.total, marker=PHASE_MARKERS[r.phase],
                ms=2.6, mew=0.0, zorder=3, clip_on=False, alpha=0.5,
                mfc=colors[r.model], mec=colors[r.model], linestyle="none")

    if frontier:
        xs, ys = [], []
        for c in frontier:
            if xs:
                xs.append(c["cost"])
                ys.append(ys[-1])
            xs.append(c["cost"])
            ys.append(c["score"])
        ax.plot(xs, ys, color=FRONTIER_COLOR, lw=0.9,
                linestyle=(0, (3, 1.6)), zorder=4)

    # Cell means on top, ringed so they read against the cloud behind them.
    for c in cells:
        ax.plot(c["cost"], c["score"], marker=PHASE_MARKERS[c["phase"]],
                ms=6.5, mew=1.1, zorder=6, clip_on=False,
                mfc=colors[c["model"]], mec=MEAN_EDGE, linestyle="none")

    ax.set_xscale("log")
    # Log scale: costs span two orders of magnitude. The uneven tick
    # spacing is the only cue on the panel, so the caption must say so.
    ax.set_xlabel("Cost per run (USD, log scale)")
    ax.set_ylabel(f"Rubric score (of {RUBRIC_MAX})")
    ax.set_ylim(0, RUBRIC_MAX)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()


def build_legend(ax, colors):
    """Two legends: the arm by marker, the model by colour."""
    # Grey fill: the arm is the shape, so the swatch must not imply a colour.
    arm_handles = [
        Line2D([], [], linestyle="none", marker=PHASE_MARKERS[p], ms=5.5,
               mew=0.9, mfc="#BDBDBD", mec=MEAN_EDGE, label=PHASE_LABELS_SHORT[p])
        for p in ("A", "B")
    ]
    # H / S / O for haiku / sonnet / opus: the full names do not fit three
    # across inside the axes, and the caption expands them.
    model_handles = [
        Line2D([], [], linestyle="none", marker="o", ms=5,
               mfc=colors[m], mec=colors[m], label=MODEL_LABELS_SHORT[m])
        for m in MODEL_ORDER
    ]
    # The arm key sits inside the empty upper left -- nothing scores highly at
    # low cost -- and the model key runs below the axis. The inside legend is
    # the one added through add_artist: it lies within the axes, so the tight
    # bounding box picks it up regardless, while the legend that overhangs the
    # axis has to be the last ax.legend call to be captured at all.
    inside = ax.legend(handles=arm_handles, loc="upper left",
                       bbox_to_anchor=(0.005, 1.005), frameon=False,
                       fontsize=FONT_SIZES["legend"], handletextpad=0.35,
                       labelspacing=0.3, borderaxespad=0.0)
    ax.add_artist(inside)
    ax.legend(handles=model_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), frameon=False, ncol=3,
              fontsize=FONT_SIZES["legend"], handletextpad=0.35,
              columnspacing=1.0, labelspacing=0.35, borderaxespad=0.0)


if __name__ == "__main__":
    df = load_sweep_with_cost()
    cells = cost_score_cells(df)
    frontier = pareto_frontier(cells)
    colors = model_colors()

    fig, ax = plt.subplots(figsize=(FULL_PAGE_WIDTH * 0.62, 3.3))
    plot_cost_score(ax, cells, frontier, colors)
    build_legend(ax, colors)
    add_panel_label(ax, "E", x=-0.13, y=1.02)
    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "cost_score.png"))

    # Companion view: every run, with the cell means over the top, for judging
    # how much of the ordering survives within-cell spread.
    fig, ax = plt.subplots(figsize=(FULL_PAGE_WIDTH * 0.62, 3.3))
    plot_cost_score_runs(ax, df, cells, frontier, colors)
    build_legend(ax, colors)
    add_panel_label(ax, "E", x=-0.13, y=1.02)
    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "cost_score_runs.png"))

    print(f"{len(cells)} cells; cost "
          f"${min(c['cost'] for c in cells):.2f}-${max(c['cost'] for c in cells):.2f}")
    print("\nPareto frontier (nothing is both cheaper and better):")
    for c in frontier:
        print(f"  ${c['cost']:6.2f}  {c['score']:5.2f}/13  "
              f"{MODEL_LABELS[c['model']]:<12} {PHASE_LABELS_SHORT[c['phase']]:<9} "
              f"{c['effort']}")

    print("\nScore per dollar, top 5:")
    for c in sorted(cells, key=lambda c: -c["score"] / c["cost"])[:5]:
        print(f"  {c['score'] / c['cost']:5.2f} pts/$  {MODEL_LABELS[c['model']]:<12} "
              f"{PHASE_LABELS_SHORT[c['phase']]:<9} {c['effort']:<7} "
              f"(${c['cost']:.2f}, {c['score']:.2f}/13)")

    # Does SpikeLab at medium sit above the baseline's entire effort curve?
    a_med = [c for c in cells if c["phase"] == "A" and c["effort"] == "medium"]
    b_all = [c for c in cells if c["phase"] == "B"]
    for c in sorted(a_med, key=lambda c: c["cost"])[:3]:
        beaten = [b for b in b_all
                  if b["cost"] <= c["cost"] and b["score"] >= c["score"]]
        print(f"\nwith-SpikeLab {MODEL_LABELS[c['model']]} at medium "
              f"(${c['cost']:.2f}, {c['score']:.2f}/13): "
              f"{len(beaten)} of {len(b_all)} baseline cells match it "
              "on cost and score")
