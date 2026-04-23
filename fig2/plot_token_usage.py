"""Plot token usage per model with cost on secondary y-axis.

One bar per run per model, grouped. Left y-axis: tokens. Right y-axis: cost ($).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "..")))

from plot_config import apply_style, save_fig, get_panel_width, FONT_SIZES
apply_style()
from benchmark_data import CONDITIONS, TOKEN_USAGE, COST_PER_M_TOKENS


COND_COLORS = {
    "Opus plain": "#67A9CF",
    "Sonnet plain": "#F4A582",
    "Sonnet + SpikeLab": "#2D9F5D",
}

BAR_WIDTH = 0.22


def plot_token_usage(ax):
    """Bar chart: one bar per run per model, cost on secondary y-axis."""
    bar_idx = 0

    for ci, cond in enumerate(CONDITIONS):
        tokens = np.array(TOKEN_USAGE[cond], dtype=float)
        color = COND_COLORS[cond]

        for ri, val in enumerate(tokens):
            if np.isnan(val):
                bar_idx += 1
                continue
            x = bar_idx * BAR_WIDTH
            ax.bar(x, val / 1000, BAR_WIDTH * 0.9, color=color,
                   edgecolor="white", linewidth=0.3)
            bar_idx += 1

        bar_idx += 0.7  # gap between groups

    # x-axis labels at group centers
    group_centers = []
    pos = 0
    for ci, cond in enumerate(CONDITIONS):
        n = sum(1 for v in TOKEN_USAGE[cond] if not np.isnan(v))
        center = (pos + pos + (n - 1)) / 2 * BAR_WIDTH
        group_centers.append(center)
        pos += n + 0.7

    ax.set_xticks(group_centers)
    ax.set_xticklabels(["O", "S", "SL"], fontsize=FONT_SIZES["tick_label"])
    # y-label set by caller when embedded in combined figure
    ax.set_xlim(-BAR_WIDTH, bar_idx * BAR_WIDTH)

    # Secondary y-axis: cost
    ax2 = ax.twinx()

    # Compute cost ticks aligned to token ticks
    # We need different cost scales per model, so show cost bars overlaid
    # Instead, show cost as a separate set of markers
    for ci, cond in enumerate(CONDITIONS):
        tokens = np.array(TOKEN_USAGE[cond], dtype=float)
        cost_per_m = COST_PER_M_TOKENS[cond]

        pos_inner = 0
        for prev_ci in range(ci):
            n_prev = sum(1 for v in TOKEN_USAGE[CONDITIONS[prev_ci]]
                         if not np.isnan(v))
            pos_inner += n_prev + 0.7

        for ri, val in enumerate(tokens):
            if np.isnan(val):
                continue
            x = (pos_inner + ri) * BAR_WIDTH
            cost = val / 1e6 * cost_per_m
            ax2.plot(x, cost, "o", color="k", markersize=3, zorder=10)

    ax2.set_ylabel("Est. cost ($)", rotation=270, labelpad=12)
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(["1", "2"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)


if __name__ == "__main__":
    panels_dir = os.path.abspath(os.path.join(script_dir, "..", "panels"))

    width = get_panel_width(3)
    fig, ax = plt.subplots(figsize=(width, width * 0.85))
    plot_token_usage(ax)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COND_COLORS[c], label=c) for c in CONDITIONS
    ] + [
        Line2D([0], [0], marker="_", color="k", markersize=6,
               markeredgewidth=1.2, linewidth=0, label="Est. cost")
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZES["legend"],
              loc="upper right")

    save_fig(fig, os.path.join(panels_dir, "token_usage.png"))
    print("Saved: token_usage.png")
