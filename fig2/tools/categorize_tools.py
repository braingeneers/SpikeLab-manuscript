#!/usr/bin/env python
"""Group benchmark v2 tool calls into the seven categories used in Fig. 2C of v1.

Categories and their v1 definitions (manuscript/supplementary_data/benchmark/
processed/benchmark_data.py):

    SUCCESSFUL_SCRIPT_RUNS  successful script executions (python script.py, exit 0)
    ENV_SETUP               environment/setup calls (pip, mkdir, skill, etc.)
    FILE_READS              file reads (Read tool, not Search/Grep)
    DOC_READS               documentation reads (Search/Grep to library docs)
    INLINE_EXPLORATION      inline exploration calls (python -c + heredoc blocks)
    FAILED_EXECUTIONS       failed executions (traceback / non-zero exit)
    SCRIPT_WRITES           script writes/updates

One deviation from v1, made deliberately and flagged in the output: v1 runs
consulted library documentation with Search/Grep, so DOC_READS was defined by
the tool used. The v2 harness has a Read tool that sessions use for SKILL.md and
REPO_MAP.md, so classifying purely by tool would file those as FILE_READS and
empty the DOC_READS category. Here DOC_READS is defined by TARGET -- any call
reading library documentation or library source, whatever tool it used -- which
preserves what the category measures. Pass --doc-by-tool for the literal v1 rule.

Anything unmatched lands in OTHER and is reported rather than dropped: silent
truncation makes a partial breakdown look complete.

Usage:
    python categorize_tools.py                     # summary + CSV
    python categorize_tools.py --plot fig2c.png    # stacked bars per task
    python categorize_tools.py --per-run           # one row per run, not per cell
"""

import os
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Benchmark data root: $BENCH_RUNS if set, else fig2/data (see prepare_data.py).
BENCH_ROOT = Path(os.environ.get("BENCH_RUNS")
                  or Path(__file__).resolve().parents[1] / "data")

RESULT_ROUNDS = ("round6", "round6B", "sweep")
EXCLUDE_RE = re.compile(r"\.(standby|aborted)")

TASKS = ["ibl_load", "sttc_cue_vs_nocue", "stpr_psth", "pca_state_space"]
TASK_LABEL = {t: "Task %d" % (i + 1) for i, t in enumerate(TASKS)}

# The seven v1 categories, plus three that v1 did not need. v1 counted tool
# calls "excluding sleep monitoring", so MONITORING reproduces that exclusion
# explicitly rather than letting polling calls inflate the totals. TASK_MGMT and
# CLARIFICATION are harness affordances that did not exist in v1 (todo tools and
# the ask_user channel); they are kept separate so the v1-comparable subtotal
# stays comparable.
V1_CATEGORIES = [
    "SUCCESSFUL_SCRIPT_RUNS",
    "SCRIPT_WRITES",
    "INLINE_EXPLORATION",
    "FILE_READS",
    "DOC_READS",
    "ENV_SETUP",
    "FAILED_EXECUTIONS",
]
V2_ONLY = ["MONITORING", "TASK_MGMT", "CLARIFICATION"]
CATEGORIES = V1_CATEGORIES + V2_ONLY + ["OTHER"]

# library documentation and library source: what a DOC_READS call is aimed at
DOC_TARGET = re.compile(
    r"SKILL\.md|REPO_MAP|site-packages/spikelab|/spikelab/[\w/]*\.py|"
    r"\bdocs?/|README|\.rst\b",
    re.I,
)
# writing a file from the shell rather than with the Write tool
HEREDOC_WRITE = re.compile(r"(cat|tee)\s*>+\s*\S+\.(py|md|json|txt|ya?ml)", re.I)
# python -c "..." or python <<EOF / python - <<EOF
INLINE_PY = re.compile(r"python\d?\s+-c\b|python\d?\s+-\s*<<|python\d?\s*<<", re.I)
# executing a script file
SCRIPT_RUN = re.compile(r"python\d?\s+[^|;&]*\.py\b", re.I)
ENV_CMD = re.compile(
    r"^\s*(pip|conda|mamba|apt|apt-get|mkdir|export|cd|which|nvidia-smi|"
    r"free|df|nproc|uname|whoami|source|chmod|touch)\b",
    re.I,
)
SEARCH_CMD = re.compile(r"^\s*(grep|rg|find|ls|head|tail|cat|wc|sed|awk)\b", re.I)
# polling a long-running job. v1 counted tool calls "excluding sleep monitoring",
# so these are separated out rather than left to inflate the totals.
MONITOR_CMD = re.compile(
    r"^\s*(sleep|watch|until|while)\b|\bdu\s+-|\bps\s+(aux|-ef)|"
    r"\btail\s+-f\b|nvidia-smi\s+-l|^\s*true\s*$",
    re.I,
)


def classify(call, doc_by_tool=False):
    """Return the Fig. 2C category for one tool call."""
    name = call.get("name") or ""
    cmd = call.get("command") or ""

    # a failed call is counted as a failure regardless of what it was trying to
    # do -- this is how v1 counted them
    if call.get("is_error"):
        return "FAILED_EXECUTIONS"

    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        return "SCRIPT_WRITES"

    if name in ("Skill", "ToolSearch"):
        return "ENV_SETUP"

    if name.startswith("Task") or name in ("ScheduleWakeup", "TodoWrite", "Monitor"):
        return "TASK_MGMT"

    if "ask_user" in name:
        return "CLARIFICATION"

    if name in ("Grep", "Glob", "Search"):
        return "DOC_READS" if (doc_by_tool or DOC_TARGET.search(cmd)) else "FILE_READS"

    if name == "Read":
        if not doc_by_tool and DOC_TARGET.search(cmd):
            return "DOC_READS"
        return "FILE_READS"

    if name == "Bash":
        if MONITOR_CMD.search(cmd):
            return "MONITORING"
        if HEREDOC_WRITE.search(cmd):
            return "SCRIPT_WRITES"
        if INLINE_PY.search(cmd):
            return "INLINE_EXPLORATION"
        if SCRIPT_RUN.search(cmd):
            return "SUCCESSFUL_SCRIPT_RUNS"
        if not doc_by_tool and DOC_TARGET.search(cmd) and SEARCH_CMD.search(cmd):
            return "DOC_READS"
        if ENV_CMD.search(cmd):
            return "ENV_SETUP"
        if SEARCH_CMD.search(cmd):
            return "FILE_READS"
        return "OTHER"

    return "OTHER"


def sessions(root):
    for rnd in RESULT_ROUNDS:
        rdir = root / rnd
        if not rdir.is_dir():
            continue
        for sdir in sorted(rdir.iterdir()):
            if not sdir.is_dir() or EXCLUDE_RE.search(sdir.name):
                continue
            mani, calls = sdir / "run_manifest.json", sdir / "toolcalls.json"
            if not (mani.exists() and calls.exists()):
                continue
            mf = json.loads(mani.read_text(encoding="utf-8"))
            if mf.get("exit_status") != "completed":
                print("  skipping %s: exit_status=%s" % (sdir.name, mf.get("exit_status")),
                      file=sys.stderr)
                continue
            yield sdir, mf, json.loads(calls.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(BENCH_ROOT))
    ap.add_argument("--out", default=None, help="CSV path")
    ap.add_argument("--plot", default=None, help="write a stacked-bar figure here")
    ap.add_argument("--per-run", action="store_true",
                    help="one row per run instead of means per cell")
    ap.add_argument("--doc-by-tool", action="store_true",
                    help="literal v1 rule: DOC_READS = Search/Grep calls only")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    unmatched = defaultdict(int)

    for sdir, mf, calls in sessions(root):
        model = re.sub(r"-\d{8}$", "",
                       re.sub(r"^claude-", "", mf.get("model_served") or ""))
        per_task = defaultdict(lambda: defaultdict(int))
        for c in calls:
            cat = classify(c, args.doc_by_tool)
            per_task[c.get("task") or "?"][cat] += 1
            if cat == "OTHER":
                unmatched["%s: %s" % (c.get("name"),
                                      (c.get("command") or "")[:60])] += 1
        for task in TASKS:
            for cat in CATEGORIES:
                rows.append(dict(
                    session=sdir.name, round=sdir.parent.name, phase=mf.get("phase"),
                    model=model, replicate=mf.get("replicate"),
                    task=TASK_LABEL[task], category=cat,
                    count=per_task[task][cat],
                ))

    if not rows:
        sys.exit("no sessions found under %s" % root)

    out = Path(args.out) if args.out else root / "tool_categories.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s  (%d rows)" % (out, len(rows)))

    # ---- summary: mean calls per run, by phase and category ----------------
    n_runs = defaultdict(set)
    totals = defaultdict(float)
    for r in rows:
        n_runs[r["phase"]].add(r["session"])
        totals[(r["phase"], r["category"])] += r["count"]

    print("\nmean tool calls per run, by category")
    print("  %-24s %8s %8s" % ("category", "phase A", "phase B"))
    for cat in CATEGORIES:
        a = totals[("A", cat)] / max(1, len(n_runs["A"]))
        b = totals[("B", cat)] / max(1, len(n_runs["B"]))
        print("  %-24s %8.1f %8.1f" % (cat, a, b))
    print("  %-24s %8.1f %8.1f" % (
        "TOTAL",
        sum(totals[("A", c)] for c in CATEGORIES) / max(1, len(n_runs["A"])),
        sum(totals[("B", c)] for c in CATEGORIES) / max(1, len(n_runs["B"]))))
    print("  (n = %d phase A runs, %d phase B runs)" % (len(n_runs["A"]), len(n_runs["B"])))

    if unmatched:
        print("\nOTHER -- %d unmatched call(s), most common:" % sum(unmatched.values()))
        for k, v in sorted(unmatched.items(), key=lambda x: -x[1])[:10]:
            print("  %3d  %s" % (v, k))

    if args.plot:
        make_plot(rows, Path(args.plot))


def make_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stacked = [c for c in CATEGORIES if c != "OTHER"]
    colors = plt.get_cmap("tab20")(np.linspace(0, 0.75, len(stacked)))
    phases = ["A", "B"]
    labels = {"A": "with SpikeLab", "B": "model alone"}

    agg = defaultdict(list)
    for r in rows:
        agg[(r["phase"], r["task"], r["category"], r["session"])].append(r["count"])

    fig, axes = plt.subplots(1, len(TASKS), figsize=(3.1 * len(TASKS), 4.2), sharey=True)
    for ax, task in zip(axes, [TASK_LABEL[t] for t in TASKS]):
        for xi, ph in enumerate(phases):
            bottom = 0.0
            for ci, cat in enumerate(stacked):
                sess = {k[3] for k in agg if k[0] == ph and k[1] == task}
                if not sess:
                    continue
                vals = [sum(agg[(ph, task, cat, s)]) for s in sess]
                m = float(np.mean(vals)) if vals else 0.0
                ax.bar(xi, m, bottom=bottom, color=colors[ci], width=0.62,
                       label=cat.replace("_", " ").lower() if (ax is axes[0] and xi == 0) else None,
                       edgecolor="white", linewidth=0.5)
                bottom += m
        ax.set_title(task, fontsize=10)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels([labels[p] for p in phases], rotation=20, ha="right", fontsize=8)
        ax.yaxis.grid(True, color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("tool calls (mean per run)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
