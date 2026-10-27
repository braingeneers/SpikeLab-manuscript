"""Plot the four task prompts — Figure 2, prompt panel.

Read from benchmark_v2/prompts.json, which is what the 540 runs were actually
given. NOT from the v1 ``benchmark_data.py`` the original Figure 2 used: the
two differ. Tasks 2-4 gained a leading "Using the same IBL auditory cortex
dataset," clause in v2, and Task 2's shuffle count was raised to 100 and then
reverted to 10 on 2026-08-15 on feasibility grounds. Every run received 10,
verified against turns.json across all 540.

Only the SpikeLab arm's Task 1 carries the ``with /spikelab,`` prefix; the
baseline arm receives the prompts unprefixed and otherwise identical. That
prefix is the entire textual difference between the arms.
"""

import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style, save_fig, add_panel_label, COLORS, FONT_SIZES, FULL_PAGE_WIDTH,
)
apply_style()

from benchmark_v2_data import PHASE_COLORS, RUNS_DIR

PROMPTS_PATH = os.path.join(RUNS_DIR, "prompts.json")

WRAP = 33
BODY_COLOR = "#333333"
RULE_COLOR = "#BDBDBD"


def load_prompts():
    """Return (tasks, prefix) — the frozen v2 prompts and phase A's prefix."""
    with open(PROMPTS_PATH, encoding="utf-8") as fh:
        d = json.load(fh)
    return d["tasks"], d["phase_prefixes"]["A"]


def plot_task_prompts(axes, tasks, prefix, n_words=19):
    """One axes per task: a heading, then the opening of the prompt.

    The prompts run to 90 words; the panel shows the opening and the caption
    carries them in full. ``n_words`` is where the excerpt stops.
    """
    for ax, task in zip(axes, tasks):
        ax.axis("off")
        ax.text(0.0, 1.0, f"Task {task['n']}", transform=ax.transAxes,
                fontsize=FONT_SIZES["axes_label"], fontweight="bold",
                ha="left", va="top")

        body = task["prompt"]
        words = body.split()
        excerpt = " ".join(words[:n_words]) + (" …" if len(words) > n_words else "")

        y = 0.80
        if task["n"] == 1:
            # The prefix is the only textual difference between the arms, so it
            # is set apart rather than folded into the excerpt.
            ax.text(0.0, y, prefix.strip(), transform=ax.transAxes,
                    fontsize=FONT_SIZES["tick_label"], color=PHASE_COLORS["A"],
                    fontweight="bold", ha="left", va="top", family="monospace")
            y -= 0.17

        ax.text(0.0, y, textwrap.fill(excerpt, width=WRAP),
                transform=ax.transAxes, fontsize=FONT_SIZES["tick_label"],
                color=BODY_COLOR, ha="left", va="top", linespacing=1.35)


if __name__ == "__main__":
    tasks, prefix = load_prompts()

    fig = plt.figure(figsize=(FULL_PAGE_WIDTH, 1.25))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    plot_task_prompts(axes, tasks, prefix)
    add_panel_label(axes[0], "A", x=-0.04, y=1.20)
    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "task_prompts.png"))

    print(f"prefix applied to Task 1 of the treated arm: {prefix!r}")
    for t in tasks:
        print(f"  Task {t['n']:<2} {t['id']:<20} {len(t['prompt'].split()):>3} words")
