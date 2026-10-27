#!/usr/bin/env python
"""Plot scorecard total against reasoning effort, one panel per model.

Effort runs live in bench_runs/effort/; the `medium` level comes from the main
model grid (round6 / round6B / sweep), because medium was already collected
there at n=3 and re-running it would waste quota. Both sources are scored
against the same rubric, so they are directly comparable.

Usage:
    python plot_effort.py
    python plot_effort.py --panel correctness --out fig_effort_corr.png
"""

import os
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Benchmark data root: $BENCH_RUNS if set, else fig2/data (see prepare_data.py).
BENCH_ROOT = Path(os.environ.get("BENCH_RUNS")
                  or Path(__file__).resolve().parents[1] / "data")

EFFORTS = ["low", "medium", "high", "xhigh", "max"]
GRID_ROUNDS = ("round6", "round6B", "sweep")
EXCLUDE_RE = re.compile(r"\.(standby|aborted|crashed|orphaned|killed)")

POINTS = {"green": 1.0, "orange": 0.5, "red": 0.0}
PANEL_FILTER = {"total": None,
                "correctness": "analytical_correctness",
                "instructed": "instructed_behaviour"}

PHASES = {
    "A": dict(label="with SpikeLab", color="#1b6ca8", marker="o"),
    "B": dict(label="model alone", color="#e07b39", marker="s"),
}

# display order; anything not listed is appended alphabetically
MODEL_ORDER = ["haiku45", "sonnet45", "sonnet46", "sonnet5",
               "opus45", "opus46", "opus47", "opus48", "opus5"]
PRETTY = {"haiku45": "haiku-4-5", "sonnet45": "sonnet-4-5", "sonnet46": "sonnet-4-6",
          "sonnet5": "sonnet-5", "opus45": "opus-4-5", "opus46": "opus-4-6",
          "opus47": "opus-4-7", "opus48": "opus-4-8", "opus5": "opus-5"}


def score_of(card, want):
    verdicts = card.get("verdicts", [])
    if want:
        verdicts = [v for v in verdicts if v.get("category") == want]
    if not verdicts or any(v.get("verdict") is None for v in verdicts):
        return None, 0
    return sum(POINTS.get(v["verdict"], 0.0) for v in verdicts), len(verdicts)


def load(root, panel):
    """Return {(phase, model, effort): [scores]} plus the item count."""
    want = PANEL_FILTER[panel]
    runs, denom = {}, None

    def take(d, phase, model, effort):
        nonlocal denom
        mani, card = d / "run_manifest.json", d / "scorecard.json"
        if not (mani.exists() and card.exists()):
            return
        mf = json.loads(mani.read_text(encoding="utf-8"))
        # A completed run that produced nothing is a model failure and must be
        # scored; only a run that did not complete is an infrastructure artefact.
        if mf.get("exit_status") != "completed":
            return
        s, n = score_of(json.loads(card.read_text(encoding="utf-8")), want)
        if s is None:
            return
        denom = n if denom is None else max(denom, n)
        runs.setdefault((phase, model, effort), []).append(s)

    edir = root / "effort"
    if edir.is_dir():
        for d in sorted(edir.iterdir()):
            if not d.is_dir() or EXCLUDE_RE.search(d.name):
                continue
            m = re.match(r"^([AB])_(\w+?)_(low|high|xhigh|max)_rep(\d+)$", d.name)
            if m:
                take(d, m.group(1), m.group(2), m.group(3))

    # medium from the main grid
    for rnd in GRID_ROUNDS:
        rdir = root / rnd
        if not rdir.is_dir():
            continue
        for d in sorted(rdir.iterdir()):
            if not d.is_dir() or EXCLUDE_RE.search(d.name):
                continue
            m = re.match(r"^([AB])_(\w+?)_medium_rep(\d+)$", d.name)
            if m:
                take(d, m.group(1), m.group(2), "medium")
    return runs, denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(BENCH_ROOT))
    ap.add_argument("--panel", choices=list(PANEL_FILTER), default="total")
    ap.add_argument("--models", default=None,
                    help="comma-separated short names, e.g. haiku45 -- one panel each")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    runs, denom = load(root, args.panel)
    if not runs:
        sys.exit("no graded effort runs found under %s/effort" % root)

    # a model earns a panel only if it has a non-medium level; otherwise the
    # panel would be a single point carried over from the grid
    models = sorted({k[1] for k in runs
                     if any((k[0], k[1], e) in runs for e in EFFORTS if e != "medium")},
                    key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))
    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in want if m not in models]
        if missing:
            sys.exit("no effort data for: %s" % missing)
        models = want
    if not models:
        sys.exit("no model has a non-medium effort level yet")

    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.5 * nrow),
                             squeeze=False, sharey=True)
    flat = [a for row in axes for a in row]
    jitter = np.random.default_rng(0)
    rows = []

    for ax, model in zip(flat, models):
        for phase, style in PHASES.items():
            xs, ys, es, ns = [], [], [], []
            for i, eff in enumerate(EFFORTS):
                vals = runs.get((phase, model, eff))
                if not vals:
                    continue
                a = np.asarray(vals, dtype=float)
                xs.append(i)
                ys.append(a.mean())
                es.append(a.std(ddof=1) if a.size > 1 else 0.0)
                ns.append(a.size)
                ax.scatter(i + jitter.uniform(-0.06, 0.06, a.size), a, s=15,
                           color=style["color"], alpha=0.45, zorder=2, linewidths=0)
                rows.append((phase, model, eff, a.size, float(a.mean()),
                             float(a.std(ddof=1)) if a.size > 1 else 0.0,
                             ";".join("%g" % v for v in a)))
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=style["color"], marker=style["marker"],
                            markersize=6, linewidth=1.8, capsize=3.5, capthick=1.2,
                            elinewidth=1.2, zorder=3,
                            label=style["label"] if ax is flat[0] else None)
        ax.set_title(PRETTY.get(model, model), fontsize=11)
        ax.set_xticks(range(len(EFFORTS)))
        ax.set_xticklabels(EFFORTS, rotation=30, ha="right", fontsize=8.5)
        ax.set_xlim(-0.4, len(EFFORTS) - 0.6)
        ax.set_ylim(0, denom + 0.4)
        ax.yaxis.grid(True, color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in flat[len(models):]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r][0].set_ylabel({"total": "scorecard total (of %d)" % denom,
                               "correctness": "analytical correctness (of %d)" % denom,
                               "instructed": "instructed behaviour (of %d)" % denom}[args.panel])
    flat[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.supxlabel("reasoning effort", fontsize=10)
    fig.tight_layout()

    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else figdir / ("fig_effort_%s.png" % args.panel)
    fig.savefig(out, dpi=200)
    print("wrote %s" % out)

    csv = Path(args.csv) if args.csv else out.with_suffix(".csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("phase,model,effort,n,mean,sd,values\n")
        for r in rows:
            fh.write("%s,%s,%s,%d,%.3f,%.3f,%s\n" % r)
    print("wrote %s" % csv)

    for r in sorted(rows, key=lambda r: (r[1], r[0], EFFORTS.index(r[2]))):
        print("  %s  %-10s %-7s n=%d  mean=%5.2f  sd=%.2f   [%s]"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))


if __name__ == "__main__":
    main()
