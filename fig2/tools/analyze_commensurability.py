#!/usr/bin/env python
"""Commensurability of reported quantities across replicates.

This replicates, at n=10 and across four arms, the test the original manuscript
made at n=3:

    "Opus invented an ad-hoc approach from first principles each time, producing
     three fundamentally different metrics in different units across three runs:
     a dimensionless z-score, a unitless ratio, and a raw firing rate in spikes
     per second. These quantities cannot be compared across runs."

The question is NOT whether the numbers agree. It is whether they are the same
KIND of number -- whether a value from run 3 can be placed beside one from run 7
at all. Two runs reporting 0.043 and 5.16 have not disagreed by a factor of 120;
they have measured different quantities.

This also corrects a mistake in analyze_answers.py, which computed "fraction of
runs within +-10% of the cell median" for `median_coupling` while pooling runs
whose units differ. That statistic is meaningless wherever the unit class is not
constant, and it is reported here only within a class.

Classification is by unit string, with the rules fixed BEFORE looking at the
per-arm counts. A unit matching no rule is reported as `other`, never forced.

Usage:  python analyze_commensurability.py
"""

import os
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

# Benchmark data root: $BENCH_RUNS if set, else fig2/data (see prepare_data.py).
BENCH_ROOT = Path(os.environ.get("BENCH_RUNS")
                  or Path(__file__).resolve().parents[1] / "data")

QUARANTINED = (".crashed", ".killed", ".orphaned", ".aborted", ".standby", ".ratelimited")

ARMS = [("A", "instructions+tools", "round6", "A_sonnet46_medium_rep"),
        ("C", "instructions only", "ablation", "C_sonnet46_medium_rep"),
        ("D", "tools only", "ablation", "D_sonnet46_medium_rep"),
        ("B", "neither", "round6B", "B_sonnet46_medium_rep")]

# Every place the benchmark leaves a UNIT open. (task, clause index, label);
# clause 0 is the first quantity a run names in its `units` field, clause 1 the
# second -- runs state the primary quantity first and a secondary one after a
# semicolon ("coupling in normalised stPR units; PSTH in Hz").
#
# All four are reported whether or not they separate the arms. Choosing the ones
# that happen to discriminate would be selection on the analysis.
#
# NOT included, and why: `sttc_cue_vs_nocue` prescribes z-scoring in the prompt,
# so every run reports a z-score and the units cannot vary. That task fails
# commensurability differently -- a z-score against a degree-preserving swap and
# one against a circular shift share units but not a reference distribution, so
# they are dimensionally alike and semantically incomparable. That is measured by
# `null_method` in analyze_methods.py, not here.
TARGETS = [
    ("stpr_psth", 0, "population-coupling value"),
    ("stpr_psth", 1, "PSTH y-axis"),
    ("ibl_load", 0, "firing / population rate"),
    ("pca_state_space", 0, "variance explained"),
]

# Unit classes, in priority order: first match wins, so specific patterns lead.
RULES = [
    ("unstated",            r"not[_ ]stated|unknown"),
    ("zscore",              r"z-?scor"),
    ("ratio_to_baseline",   r"\bratio\b|baseline population rate|to baseline"),
    ("percent",             r"%|percent"),
    ("fraction",            r"\bfraction\b|\bproportion\b"),
    ("kilohertz",           r"\bkhz\b"),
    ("rate_hz",             r"\bhz\b|spikes?/s\b|sp/s\b|spikes per second|firing rate"),
    ("count_per_bin",       r"spikes?/(bin|ms)|spk/bin|per bin|spikes per bin"),
    ("normalised_index",    r"dimensionless|normalis|normaliz|a\.u\.|arbitrary unit|"
                            r"unitless|\[-1,\s*1\]|coefficient"),
    ("event_times",         r"spike times|event|raster|tick mark|scatter point"),
    ("count",               r"\bcount\b|n_units|integer"),
]

# the numeric answer belonging to each task's primary quantity
VALUE_FIELD = {"stpr_psth": "median_coupling", "ibl_load": "n_good_units",
               "pca_state_space": "pc_cumulative"}


def clause(units, idx):
    """The idx-th quantity named in a units field (0 = primary).

    Runs report two units in one field. Matching the whole string put every arm-A
    run in `rate_hz` on the strength of its PSTH clause, and the misclassification
    showed in the values: the supposed "Hz" group held numbers from 0.032 to
    0.053. Split first, then classify.
    """
    parts = re.split(r";|\|", (units or "").strip().lower())
    return parts[idx] if idx < len(parts) else ""


# Sharing a dimension is not sharing a quantity. Arm B's seven "rate_hz" runs
# held values from 0.070 to 1241.8 -- a raw TOTAL population rate summed over ~69
# units, a PER-UNIT rate, and a BASELINE-SUBTRACTED deviation are three different
# measurements that all happen to be frequencies.
#
# The two extra axes are the ones the manuscript itself names: SpikeLab's method
# carries "baseline subtraction and normalization by the sum of mean firing
# rates", and the plain model's was "equivalent to Okun et al. but lacking
# baseline subtraction and normalization". So they are given by the source, not
# chosen here.
BASELINE_RE = re.compile(r"baseline[- ]?(subtract|correct)|above baseline|"
                         r"minus baseline|excess|deviation")
PERUNIT_RE = re.compile(r"/unit|per[- ]unit|per neuron|/neuron|n_units")


def qualifiers(u):
    """(baseline, scaling) qualifiers, or 'unspec' when the string is silent."""
    base = "basesub" if BASELINE_RE.search(u) else "raw"
    scale = "perunit" if PERUNIT_RE.search(u) else "unspec"
    return base, scale


def classify(units, idx=0, refine=True):
    u = clause(units, idx)
    if not u.strip():
        return "unstated"
    dim = "other"
    for name, pat in RULES:
        if re.search(pat, u):
            dim = name
            break
    if not refine or dim in ("unstated", "other"):
        return dim
    # A normalised index is already dimensionless and self-scaled; qualifying it
    # further would split a class that is commensurable by construction.
    if dim in ("normalised_index", "zscore", "ratio_to_baseline", "percent",
               "fraction", "count", "event_times"):
        return dim
    base, scale = qualifiers(u)
    return f"{dim}|{base}|{scale}"


def load(root, sub, prefix):
    out = []
    for d in sorted((root / sub).glob(prefix + "*")):
        if not d.is_dir() or any(q in d.name for q in QUARANTINED):
            continue
        f = d / "methods.json"
        if f.exists():
            out.append((d.name, json.loads(f.read_text(encoding="utf-8"))))
    return out


def main():
    root = BENCH_ROOT
    print("=" * 92)
    print("COMMENSURABILITY -- can replicate results be placed beside one another at all?")
    print("every quantity whose unit the task prompt leaves open; none omitted")
    print("=" * 92)

    grid = {}
    for task, idx, label in TARGETS:
        print()
        print("=" * 92)
        print(f"## {label}    ({task}, clause {idx})")
        print("=" * 92)
        for arm, arm_label, sub, prefix in ARMS:
            runs = load(root, sub, prefix)
            if not runs:
                continue
            classes, vals = [], defaultdict(list)
            for _n, r in runs:
                tk = next((x for x in r["tasks"] if x["task"] == task), None)
                if not (tk and tk.get("attempted")):
                    continue
                u = tk.get("units") or ""
                if idx > 0 and len(re.split(r";|\|", u)) <= idx:
                    continue                       # this run named only one quantity
                cls = classify(u, idx)
                classes.append(cls)
                fld = VALUE_FIELD.get(task)
                if fld and idx == 0:
                    try:
                        vals[cls].append(float((tk.get("answer") or {}).get(fld, "")))
                    except (TypeError, ValueError):
                        pass
            if not classes:
                print(f"\n  arm {arm}: no run named this quantity")
                continue
            c = Counter(classes)
            top, ntop = c.most_common(1)[0]
            print(f"\n  arm {arm} {arm_label}  ({len(classes)} runs)")
            print(f"     {len(c)} class(es), largest {ntop}/{len(classes)}: "
                  + ", ".join(f"{n}x {k}" for k, n in c.most_common()))
            xs = vals.get(top, [])
            if len(xs) > 1:
                med = st.median(xs)
                within = sum(1 for x in xs if med and abs(x - med) <= 0.10 * abs(med))
                print(f"     within '{top}': median {med:g}, range {min(xs):g} to "
                      f"{max(xs):g}, {within}/{len(xs)} within +-10%")
            grid[(label, arm)] = (len(c), ntop, len(classes))

    print()
    print("=" * 92)
    print("SUMMARY -- distinct unit classes per arm (1 = all replicates commensurable)")
    print("=" * 92)
    print()
    print(f"  {'quantity':<30}" + "".join(f"{a[0]:>16}" for a in ARMS))
    print("  " + "-" * (30 + 16 * len(ARMS)))
    for _task, _idx, label in TARGETS:
        row = ""
        for a, _l, _s, _p in ARMS:
            g = grid.get((label, a))
            row += f"{f'{g[0]} ({g[1]}/{g[2]})':>16}" if g else f"{'-':>16}"
        print(f"  {label:<30}{row}")
    print()
    print("  cell reads: <distinct unit classes> (<runs in largest class>/<runs>)")
    print("  A value from a different class is not a different answer -- it is a")
    print("  different quantity, and the two cannot be pooled, compared or averaged.")


if __name__ == "__main__":
    main()
