"""Figure 2 — panel plotting functions.

Reusable plotting functions for every panel in Figure 2. Each function
accepts pre-created Axes and pre-loaded data.

Panel groups:
    A     Task prompts (text-only)
    B     Token usage and estimated cost
    C     Tool call breakdown by task (stacked bars)
    D     Issue scorecard (colored circles)
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import FONT_SIZES

from benchmark_data import CONDITIONS, TASKS, SCORECARD_TASKS


# ── Colors ────────────────────────────────────────────────────────────────
C_DOC = "#2166AC"          # blue — documentation reads
C_READS = "#67A9CF"        # light blue — file reads
C_ENV = "#D9D9D9"          # light grey — environment/setup
C_SCRIPTS = "#FDE68A"      # light yellow — successful script runs
C_WRITES = "#E8B931"       # amber — script writes/updates
C_EXPLORE = "#F4A582"      # light orange — inline exploration
C_FAILED = "#B2182B"       # dark red — failed executions

BAR_WIDTH = 0.22

SC_COLORS = {"green": "#2ca02c", "orange": "#ff7f0e", "red": "#d62728"}
SC_MARKER_SIZE = 35


def plot_tool_breakdown_by_task(axes, tool_data):
    """One subplot per task, one bar per run per model, 7-layer stack.

    Parameters
    ----------
    axes : list of Axes
        One axes per task.
    tool_data : dict
        Keys: DOC_READS, INLINE_EXPLORATION, FAILED_EXECUTIONS,
        SCRIPT_WRITES, SUCCESSFUL_SCRIPT_RUNS, ENV_SETUP, FILE_READS.
    """
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        bar_idx = 0

        for ci, cond in enumerate(CONDITIONS):
            doc = np.array(tool_data["DOC_READS"][cond][task], dtype=float)
            explore = np.array(tool_data["INLINE_EXPLORATION"][cond][task], dtype=float)
            failed = np.array(tool_data["FAILED_EXECUTIONS"][cond][task], dtype=float)
            writes = np.array(tool_data["SCRIPT_WRITES"][cond][task], dtype=float)
            scripts = np.array(tool_data["SUCCESSFUL_SCRIPT_RUNS"][cond][task], dtype=float)
            env = np.array(tool_data["ENV_SETUP"][cond][task], dtype=float)
            reads = np.array(tool_data["FILE_READS"][cond][task], dtype=float)

            for ri in range(len(doc)):
                if np.isnan(doc[ri]):
                    bar_idx += 1
                    continue

                x = bar_idx * BAR_WIDTH
                vals = {
                    "failed": failed[ri] if not np.isnan(failed[ri]) else 0,
                    "explore": explore[ri] if not np.isnan(explore[ri]) else 0,
                    "writes": writes[ri] if not np.isnan(writes[ri]) else 0,
                    "scripts": scripts[ri] if not np.isnan(scripts[ri]) else 0,
                    "reads": reads[ri] if not np.isnan(reads[ri]) else 0,
                    "env": env[ri] if not np.isnan(env[ri]) else 0,
                    "doc": doc[ri] if not np.isnan(doc[ri]) else 0,
                }

                layers = [
                    ("failed", C_FAILED),
                    ("explore", C_EXPLORE),
                    ("writes", C_WRITES),
                    ("scripts", C_SCRIPTS),
                    ("env", C_ENV),
                    ("reads", C_READS),
                    ("doc", C_DOC),
                ]
                b = 0
                for key, color in layers:
                    v = vals[key]
                    if v > 0:
                        ax.bar(x, v, BAR_WIDTH * 0.9, bottom=b, color=color,
                               edgecolor="white", linewidth=0.3)
                    b += v

                bar_idx += 1

            bar_idx += 0.7

        group_centers = []
        pos = 0
        for ci, cond in enumerate(CONDITIONS):
            n = sum(1 for v in tool_data["DOC_READS"][cond][task] if not np.isnan(v))
            center = (pos + pos + (n - 1)) / 2 * BAR_WIDTH
            group_centers.append(center)
            pos += n + 0.7

        ax.set_xticks(group_centers)
        ax.set_xticklabels(["O", "S", "SL"], fontsize=FONT_SIZES["tick_label"])

        if ti == 0:
            ax.set_ylabel("Tool calls")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        ax.set_xlim(-BAR_WIDTH, bar_idx * BAR_WIDTH)


def plot_scorecard(axes, scorecard, max_issues=None):
    """One subplot per scorecard task, issue labels above triangles.

    Parameters
    ----------
    axes : list of Axes
        One axes per scorecard task category.
    scorecard : dict
        Nested dict: task → issue → condition → [run1, run2, run3].
    max_issues : int, optional
        Fixed max issues for uniform y-extent across columns.
    """
    n_cond = len(CONDITIONS)
    tri_dx = [-0.18, 0.18, 0.0]
    tri_dy = [0.08, 0.08, -0.18]

    if max_issues is None:
        max_issues = max(len(scorecard[t]) for t in SCORECARD_TASKS)

    for ti, task in enumerate(SCORECARD_TASKS):
        ax = axes[ti]
        issues = list(scorecard[task].keys())

        for ii, issue in enumerate(issues):
            y = max_issues - 1 - ii

            ax.text(n_cond / 2 - 0.5, y + 0.35, issue,
                    fontsize=FONT_SIZES["tick_label"], ha="center", va="bottom",
                    color="#333333")

            for ci, cond in enumerate(CONDITIONS):
                vals = scorecard[task][issue][cond]
                cx = ci
                for ri, val in enumerate(vals):
                    x = cx + tri_dx[ri]
                    yy = y + tri_dy[ri]
                    if val is None:
                        ax.scatter(x, yy, s=SC_MARKER_SIZE, facecolors="none",
                                   edgecolors="#999999", linewidths=0.6, zorder=5)
                    else:
                        ax.scatter(x, yy, s=SC_MARKER_SIZE,
                                   color=SC_COLORS[val], edgecolors="none",
                                   zorder=5)

        ax.set_yticks([])
        ax.set_ylim(-0.6, max_issues - 0.3)

        ax.set_xticks(range(n_cond))
        ax.set_xticklabels(["O", "S", "SL"], fontsize=FONT_SIZES["tick_label"])
        ax.set_xlim(-0.6, n_cond - 0.4)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)
