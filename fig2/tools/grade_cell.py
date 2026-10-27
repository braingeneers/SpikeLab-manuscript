#!/usr/bin/env python
"""Rubric item 14, `reproducible_method` -- the cell-scope item, finally scored.

The rubric has always defined a fourteenth item that cannot be scored from one
session:

    "Whether the same coupling method was used across all replicates of a cell,
     producing comparable results in comparable units."
      green  -- same method and units across all replicates
      orange -- same method family, differing parameters
      red    -- fundamentally different metrics or units across replicates

It was never implemented, so every total reported so far is out of 13.

THE MEASURE. Per cell: the maximum number of replicates that produced the same
KIND of answer for the population-coupling value -- the largest mutually
commensurable group. A value in a different unit class is not a different answer,
it is a different quantity; 0.043 and 5.16 have not disagreed by a factor of 120.

Why this quantity and not another. Four quantities in the benchmark leave their
units open, and only this one measures what it appears to. `variance explained`
is near-universally reported in percent, so it has almost no variance to detect.
`firing / population rate` splits 5/5 between runs that name the raster first and
runs that name the rate first -- narration order, not units. `sttc` has its units
prescribed by the prompt. The coupling metric is the one the library encodes as a
published method (Bimbard et al., building on Okun et al.), and it is the one the
original manuscript used to make this argument at n=3.

SCORING. The score is max_agreeing / n_attempted -- 9 of 10 replicates producing
the same kind of quantity scores 0.9. A continuous fraction rather than the
rubric's green/orange/red, because any three-way cut needs thresholds and
thresholds picked after seeing the results are picked to suit them.

CONSEQUENCE, stated plainly. This item is cell-scope: one verdict shared by every
replicate in the cell. Folding it into per-run totals moves every total from /13
to /14 and adds the identical value to all ten runs of a cell, which inflates
apparent per-run precision because the item has no within-cell variance by
construction. This script therefore writes a per-CELL verdict and leaves per-run
scorecards untouched; whether to add it to the totals is a reporting decision,
recorded in ANALYSIS_PLAN.md section 8.

Usage:  python grade_cell.py [--write]      (--write emits cell_scores.json)
"""

import os
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_commensurability import ARMS, QUARANTINED, classify, load  # noqa: E402

# Benchmark data root: $BENCH_RUNS if set, else fig2/data (see prepare_data.py).
BENCH_ROOT = Path(os.environ.get("BENCH_RUNS")
                  or Path(__file__).resolve().parents[1] / "data")

TASK = "stpr_psth"          # population coupling only, by design

# The full grid: 9 models x 2 phases x 3 efforts. `medium` for sonnet-4-6 lives in
# the ablation rounds; every other medium cell is in sweep/.
MODELS = ["haiku45", "sonnet45", "sonnet46", "sonnet5",
          "opus45", "opus46", "opus47", "opus48", "opus5"]
PRETTY = {"haiku45": "haiku-4-5", "sonnet45": "sonnet-4-5", "sonnet46": "sonnet-4-6",
          "sonnet5": "sonnet-5", "opus45": "opus-4-5", "opus46": "opus-4-6",
          "opus47": "opus-4-7", "opus48": "opus-4-8", "opus5": "opus-5"}
EFFORTS = ["low", "medium", "max"]


def grid_cells():
    """(label, subdir, prefix) for every phase x effort x model cell."""
    for eff in EFFORTS:
        for ph in ("A", "B"):
            for mod in MODELS:
                if eff == "medium":
                    sub = ("round6" if ph == "A" else "round6B") if mod == "sonnet46" else "sweep"
                else:
                    sub = "effort"
                yield (f"{ph} {PRETTY[mod]} {eff}", sub, f"{ph}_{mod}_{eff}_rep")


def score_cell(runs):
    """Return (score, max_agreeing, n_attempted, distribution, n_underspecified).

    SCORE = max_agreeing / n_attempted. Ten replicates all producing the same kind
    of quantity scores 1.0; the largest group being three of ten scores 0.3.

    A continuous score rather than green/orange/red: any three-way cut needs
    thresholds, and thresholds chosen after seeing four numbers are chosen to suit
    them. The fraction carries the same information without that freedom, and it
    is what the item is actually measuring.

    `max_agreeing` is the largest class outright, including classes whose scaling
    is unstated. An earlier version excluded those and reported arm B as 2/10 when
    its largest group -- three runs reporting a bare "Hz" -- was 3. Excluding the
    biggest group from a maximum contradicts the measure's own definition. The
    under-specification is still counted and reported alongside, because a bare
    "Hz" covering values of 3.86 and 1241.8 is a weaker agreement than a shared
    normalised index, and the reader should see that.
    """
    classes = []
    for _name, r in runs:
        t = next((x for x in r["tasks"] if x["task"] == TASK), None)
        if t and t.get("attempted"):
            classes.append(classify(t.get("units"), 0))
    if not classes:
        return None, 0, 0, Counter(), 0
    c = Counter(classes)
    under = sum(v for k, v in c.items() if k.endswith("|unspec") or k == "unstated")
    top = c.most_common(1)[0][1]
    n = len(classes)
    return top / n, top, n, c, under


def main():
    root = BENCH_ROOT
    write = "--write" in sys.argv

    print("=" * 84)
    print("RUBRIC ITEM 14 -- reproducible_method (cell scope)")
    print("max replicates producing the same KIND of coupling value")
    print("=" * 84)
    print(f"\n  {'cell':<26}{'score':>8}{'max agreeing':>15}{'underspec':>11}"
          f"   distribution")
    out = {}
    cells = [(arm + " " + label, sub, prefix) for arm, label, sub, prefix in ARMS] \
        if "--grid" not in sys.argv else list(grid_cells())
    for label, sub, prefix in cells:
        runs = load(root, sub, prefix)
        if not runs:
            print(f"  {label:<26}{'-':>8}{'no fingerprints':>15}")
            continue
        sc, top, n, c, under = score_cell(runs)
        if sc is None:
            print(f"  {label:<26}{'-':>8}{'not attempted':>15}")
            continue
        dist = ", ".join(f"{b}x {a}" for a, b in c.most_common())
        print(f"  {label:<26}{sc:>8.2f}{f'{top}/{n}':>15}{under:>11}   {dist[:70]}")
        out[label] = {"score": sc, "max_agreeing": top, "n_attempted": n,
                      "n_underspecified": under,
                      "distribution": dict(c), "task": TASK,
                      "item": "reproducible_method", "scope": "cell"}

    print("\n  score = max agreeing / runs attempted. 1.00 = every replicate produced")
    print("  the same kind of quantity; 0.30 = the largest agreeing group was 3 of 10.")
    print("  underspec = runs whose units leave the scaling unstated. They still count")
    print('  toward a group, but a bare "Hz" spanning 3.86 to 1241.8 is a weaker')
    print("  agreement than a shared normalised index.")
    print("\n  Totals elsewhere remain out of 13: this item is per-cell, not per-run.")

    if write:
        p = root / "figures" / "cell_scores.json"
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
