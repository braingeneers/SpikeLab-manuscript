"""Assemble Figure 2 — Benchmark: plain models vs SpikeLab-assisted analysis.

Layout:
    Row 0:  A: Task prompts (text)
    Row 1:  B: Token usage + cost  |  C: Tool call breakdown (4 task subplots)
    Row 2:  Legend for tool call bars
    Row 3:  D: Issue scorecard (5 category columns)

Prerequisites:
    - benchmark_data.py  (all quantified benchmark data)

Usage:
    python -m fig2.assemble     (from the figure_code directory)

Output:
    fig2/Figure2_preview.png    PNG preview
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys
import textwrap
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import (
    apply_style, save_fig, FONT_SIZES, FULL_PAGE_WIDTH, add_panel_label,
)
apply_style()

from benchmark_data import (
    CONDITIONS, TASKS, DOC_READS, INLINE_EXPLORATION, FAILED_EXECUTIONS,
    SCRIPT_WRITES, SUCCESSFUL_SCRIPT_RUNS, ENV_SETUP, FILE_READS,
    SCORECARD, SCORECARD_TASKS, TOKEN_USAGE, COST_PER_M_TOKENS,
    TASK_PROMPTS,
)
from plot_token_usage import plot_token_usage

from fig2.panels import (
    plot_tool_breakdown_by_task,
    plot_scorecard,
    C_FAILED, C_EXPLORE, C_WRITES, C_SCRIPTS, C_ENV, C_READS, C_DOC,
)

# ── Pack tool data for panels ─────────────────────────────────────────────
tool_data = {
    "DOC_READS": DOC_READS,
    "INLINE_EXPLORATION": INLINE_EXPLORATION,
    "FAILED_EXECUTIONS": FAILED_EXECUTIONS,
    "SCRIPT_WRITES": SCRIPT_WRITES,
    "SUCCESSFUL_SCRIPT_RUNS": SUCCESSFUL_SCRIPT_RUNS,
    "ENV_SETUP": ENV_SETUP,
    "FILE_READS": FILE_READS,
}


# ══════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    max_issues = max(len(SCORECARD[t]) for t in SCORECARD_TASKS)
    sc_height = max_issues * 0.65
    prompt_height = 1.0
    legend_row_height = 0.5
    bar_height = 2.0

    fig = plt.figure(figsize=(FULL_PAGE_WIDTH,
                               prompt_height + bar_height + legend_row_height + sc_height))
    gs = gridspec.GridSpec(4, 6, figure=fig,
                           width_ratios=[1, 0.3, 1, 1, 1, 1],
                           height_ratios=[prompt_height, bar_height, legend_row_height, sc_height],
                           hspace=0.3, wspace=0.25)

    # ── Row 0: Task prompts ──────────────────────────────────────────────
    ax_icon = fig.add_subplot(gs[0, 0])
    ax_icon.axis("off")
    add_panel_label(ax_icon, "A", x=-0.15, y=1.05)

    for ti, task in enumerate(TASKS):
        ax_p = fig.add_subplot(gs[0, ti + 2])
        ax_p.axis("off")
        ax_p.text(0.5, 1.15, task, fontsize=FONT_SIZES["axes_label"],
                  fontweight="bold", ha="center", va="top",
                  transform=ax_p.transAxes)
        # Show first ~10 words of the prompt followed by "..."
        words = TASK_PROMPTS[task].split()
        short_prompt = " ".join(words[:10]) + " ..."
        wrapped = textwrap.fill(short_prompt, width=25)
        ax_p.text(0.5, 0.80, wrapped,
                  fontsize=FONT_SIZES["tick_label"],
                  ha="center", va="top", transform=ax_p.transAxes,
                  color="#333333", fontstyle="italic",
                  linespacing=1.2)

    # ── Row 1: Token usage + tool call breakdown ─────────────────────────
    ax_tokens = fig.add_subplot(gs[1, 0])
    plot_token_usage(ax_tokens)
    ax_tokens.set_ylabel("Total tokens (K)")
    add_panel_label(ax_tokens, "B")

    bar_axes = [fig.add_subplot(gs[1, i + 2]) for i in range(4)]
    for ax in bar_axes[1:]:
        ax.sharey(bar_axes[0])

    plot_tool_breakdown_by_task(bar_axes, tool_data)
    add_panel_label(bar_axes[0], "C")

    # ── Row 2: Legend ────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[2, 2:6])
    ax_leg.axis("off")
    bar_legend = [
        Patch(facecolor=C_FAILED, label="Failed exec."),
        Patch(facecolor=C_EXPLORE, label="Exploration"),
        Patch(facecolor=C_WRITES, label="Script writes"),
        Patch(facecolor=C_SCRIPTS, label="Script runs"),
        Patch(facecolor=C_ENV, label="Env./setup"),
        Patch(facecolor=C_READS, label="File reads"),
        Patch(facecolor=C_DOC, label="Doc. reads"),
    ]
    ax_leg.legend(handles=bar_legend, loc="center",
                  ncol=4, fontsize=FONT_SIZES["legend"],
                  frameon=False)

    # ── Row 3: Scorecard ─────────────────────────────────────────────────
    gs_sc = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=gs[3, :], wspace=0.4,
        width_ratios=[1, 1, 1, 1, 1],
    )

    sc_axes_ordered = []
    for i in range(5):
        sc_axes_ordered.append(fig.add_subplot(gs_sc[0, i]))

    # Reorder: visual [Global, T1, T2, T3, T4] → data [T1, T2, T3, T4, Global]
    sc_axes = sc_axes_ordered[1:] + [sc_axes_ordered[0]]
    plot_scorecard(sc_axes, SCORECARD, max_issues=max_issues)

    sc_titles = ["Global", "Task 1", "Task 2", "Task 3", "Task 4"]
    for i, title in enumerate(sc_titles):
        sc_axes_ordered[i].set_title(title, fontsize=FONT_SIZES["axes_label"],
                                      fontweight="bold")
    add_panel_label(sc_axes_ordered[0], "D", x=-0.15, y=1.15)

    # ── Save ─────────────────────────────────────────────────────────────
    save_fig(fig, os.path.join(SCRIPT_DIR, "Figure2_preview.png"))
    print("Saved: Figure2_preview.png")

    from plot_config import save_fig_submission
    save_fig_submission(fig, os.path.join(SCRIPT_DIR, "Figure2.tif"))
    print("Saved: Figure2.tif")
