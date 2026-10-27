"""Benchmark v2 data loading — Figure 2 (revised).

Reads the scored per-run results of the n=10 parameter sweep and the
sonnet-4-6 component ablation. Superseded the n=3 ``benchmark_data`` module.

Source of truth
---------------
``fig_effort_total.csv``   9 models x 5 efforts x 2 phases, cell means and the
                           per-run values behind them.
``stats_sweep.csv``        per-run totals for the confirmatory grid only
                           (low / medium / max, n=10).
``stats_ablation.csv``     per-run totals for the four ablation arms.

Every total is out of 13, not 14: rubric item 14 (``reproducible_method``)
is cell-scope. ``tools/grade_cell.py`` scores it once per cell, it is read
separately by ``load_cell_scores``, and it is never added to a per-run total.

Effort levels
-------------
``low``, ``medium`` and ``max`` form the pre-registered confirmatory grid at
n=10 per cell. ``high`` and ``xhigh`` stand at n=3 and are outside it: they are
displayed for the overview but carry no inferential claim.
"""

import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from plot_config import COLORS, CUD_HUES, tint

# Benchmark data root: $BENCH_RUNS if set, else fig2/data, which
# prepare_data.py builds from the Zenodo archive.
RUNS_DIR = os.environ.get("BENCH_RUNS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")
BENCH_DIR = os.path.join(RUNS_DIR, "figures")

# Rubric totals are out of 13 (item 14 is cell-scope); see module docstring.
RUBRIC_MAX = 13

# The three levels of the confirmatory grid, and the two that are not in it.
CONFIRMATORY_EFFORTS = ["low", "medium", "max"]
EXPLORATORY_EFFORTS = ["high", "xhigh"]

# Display order: family, then version.
MODEL_ORDER = [
    "haiku45", "sonnet45", "sonnet46", "sonnet5",
    "opus45", "opus46", "opus47", "opus48", "opus5",
]
MODEL_LABELS = {
    "haiku45": "haiku-4-5",
    "sonnet45": "sonnet-4-5",
    "sonnet46": "sonnet-4-6",
    "sonnet5": "sonnet-5",
    "opus45": "opus-4-5",
    "opus46": "opus-4-6",
    "opus47": "opus-4-7",
    "opus48": "opus-4-8",
    "opus5": "opus-5",
}

# Family initial plus version, for legends where the full names do not fit.
MODEL_LABELS_SHORT = {
    "haiku45": "H-4.5",
    "sonnet45": "S-4.5",
    "sonnet46": "S-4.6",
    "sonnet5": "S-5",
    "opus45": "O-4.5",
    "opus46": "O-4.6",
    "opus47": "O-4.7",
    "opus48": "O-4.8",
    "opus5": "O-5",
}

# One hue per model family, versions separated by tint: oldest lightest,
# newest at full strength, so a family reads as one colour and the generations
# order within it. Hues are the CUD reference set, not the diverging palette,
# because families are unordered categories.
FAMILY_HUES = {
    "haiku": CUD_HUES["orange"],
    "sonnet": CUD_HUES["blue"],
    "opus": CUD_HUES["bluish_green"],
}
# Generations run from lightened (oldest) through the base hue to darkened
# (newest). Going both ways rather than only lightening roughly doubles the
# separation, which the five opus generations need to be told apart.
MAX_TINT = 0.45
MAX_SHADE = 0.40


def model_colors():
    """Colour per model: family hue, shaded by generation within the family."""
    out = {}
    for group in model_family_groups():
        members = [MODEL_ORDER[i] for i in group]
        family = re.match(r"[a-z]+", members[0]).group()
        base = FAMILY_HUES[family]
        span = MAX_TINT + MAX_SHADE
        for rank, model in enumerate(members):
            # A lone member keeps the base hue; otherwise the family is spread
            # evenly from +MAX_TINT (oldest) to -MAX_SHADE (newest).
            amount = (MAX_TINT - span * rank / (len(members) - 1)
                      if len(members) > 1 else 0.0)
            out[model] = tint(base, amount)
    return out


def model_family_groups():
    """Contiguous index groups of MODEL_ORDER that share a family.

    Version order within a family is meaningful; order between families is not.
    Panels use this to break their lines at each family boundary rather than
    drawing haiku through sonnet through opus as one continuum.
    """
    groups, current, seen = [], [], None
    for i, model in enumerate(MODEL_ORDER):
        family = re.match(r"[a-z]+", model).group()
        if family != seen and current:
            groups.append(current)
            current = []
        seen = family
        current.append(i)
    if current:
        groups.append(current)
    return groups


# Effort levels in their natural order, low to max.
EFFORT_ORDER = ["low", "medium", "high", "xhigh", "max"]
EFFORT_LABELS = {
    "low": "low", "medium": "medium", "high": "high",
    "xhigh": "xhigh", "max": "max",
}

# Phase labels read as a with/without pair so the difference map's title
# names the two things being subtracted.
PHASE_LABELS = {
    "A": "with SpikeLab",
    "B": "without SpikeLab",
    "diff": "with − without",
}

# The two ends of the shared palette: the most separable pair it offers, and
# distinguishable under all three common forms of colour vision deficiency.
# Defined once here so every panel gives the two arms the same colour.
PHASE_COLORS = {"A": COLORS["palette"][0], "B": COLORS["palette"][4]}

# For legends with no room for the full phrase; the caption expands them.
PHASE_LABELS_SHORT = {"A": "with", "B": "without"}

# Ablation arms, ordered by score (high to low). Each names the directory its
# runs live in: run names are NOT unique across the tree -- A_sonnet46_medium_rep00
# exists in round2, round3, round4 and round6 with four different scores -- so a
# bare name must never be resolved by globbing every directory.
# Mirrors ARMS in benchmark_v2/tools/stats_ablation.py, which is authoritative.
# The ablation was run in exactly one cell of the grid. Named here so the
# overview maps can mark it rather than hardcoding a row and column.
ABLATION_CELL = ("sonnet46", "medium")
ARM_DIRS = {"A": "round6", "C": "ablation", "D": "ablation", "B": "round6B"}
ARM_ORDER = ["A", "C", "D", "B"]
# I = instructions, T = tools. Abbreviated so the labels fit on one line in
# both halves of the panel; the caption expands them.
ARM_LABELS = {
    "A": "I & T",
    "C": "I",
    "D": "T",
    "B": "neither",
}

# ---------------------------------------------------------------------------
# Disputed runs
#
# Two runs in the tree do not describe what happened. Both are being re-run;
# until the replacements land they are excluded, driven by their manifests
# rather than by a hardcoded flag (see disputed_run_status).
#
# A_opus46_medium_rep03  never reached the model (empty workdir, 0 tool calls,
#     $0, model_served=null; the turn died on a session-ID collision after an
#     84-min rate-limit wait). stats_sweep.csv scores it 0.0/13 and counts it as
#     a replicate; fig_effort_total.csv drops it (that cell is n=9).
#
# D_sonnet46_medium_rep03  was truncated, not null: it worked through part of
#     task 1 of 4, hit the five-hour rate limit, and the failed retry overwrote
#     its turn record. Scored 2.5/13 against a four-task rubric. Its tool-call
#     and cost telemetry is lost, so it cannot contribute to any cost panel.
# ---------------------------------------------------------------------------
NULL_SWEEP_RUN = "A_opus46_medium_rep03"
TRUNCATED_ABLATION_RUN = "D_sonnet46_medium_rep03"
DISPUTED_RUNS = (NULL_SWEEP_RUN, TRUNCATED_ABLATION_RUN)

# Which CSVs each disputed run can appear in, so a warning names only the runs
# that could actually have changed the file being read.
RUN_SOURCES = {
    NULL_SWEEP_RUN: ("stats_sweep.csv", "fig_effort_total.csv"),
    TRUNCATED_ABLATION_RUN: ("stats_ablation.csv",),
}


def disputed_run_status():
    """Return {run: (status, manifest_mtime)} for the disputed runs.

    Both runs are being re-run. Rather than hardcoding a decision that would
    silently discard the replacements once they land, the exclusion below is
    driven by what is on disk:

    ``absent``    no directory under RUNS_DIR — the run is not in the tree.
    ``running``   directory exists but holds no manifest yet: the re-run is
                  still in flight, so any CSV built now is provisional.
    otherwise     the manifest's own ``exit_status``. Only ``aborted`` is
                  dropped; a completed re-run is picked up automatically.

    Directories renamed ``<run>.aborted_*`` are deliberately not matched: that
    rename is how the harness retires a superseded attempt.
    """
    status = {}
    for run in DISPUTED_RUNS:
        dirs = glob.glob(os.path.join(RUNS_DIR, "*", run))
        if not dirs:
            status[run] = ("absent", 0.0)
            continue
        manifest = os.path.join(dirs[0], "run_manifest.json")
        if not os.path.exists(manifest):
            status[run] = ("running", os.path.getmtime(dirs[0]))
            continue
        with open(manifest, encoding="utf-8") as fh:
            state = json.load(fh).get("exit_status", "unknown")
        status[run] = (state, os.path.getmtime(manifest))
    return status


def _aborted_runs(csv_name):
    """Runs to exclude, plus a warning when a CSV no longer matches the tree.

    A re-run reaches a figure only once the CSVs in BENCH_DIR are regenerated
    from the new scorecards. Two ways that goes wrong silently, so both speak up:
    a re-run still in flight, and a landed re-run whose scores predate the CSV.
    """
    drop = set()
    csv_path = os.path.join(BENCH_DIR, csv_name)
    csv_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0.0

    for run, (state, mtime) in disputed_run_status().items():
        if csv_name not in RUN_SOURCES[run]:
            continue
        if state == "aborted":
            drop.add(run)
        elif state == "running":
            warnings.warn(
                f"{run} is being re-run now; {csv_name} is provisional. "
                f"Re-render this panel once the re-run is graded.",
                stacklevel=3,
            )
        elif state != "absent" and mtime > csv_mtime:
            warnings.warn(
                f"{run} was re-run ({state}) after {csv_name} was written. "
                f"Regenerate the CSVs in {BENCH_DIR} before using this figure.",
                stacklevel=3,
            )
    return drop


def _parse_values(s):
    """Turn a ``"9;7.5;9"`` values field into a float array."""
    return np.array([float(v) for v in str(s).split(";") if v != ""], dtype=float)


def load_effort_grid():
    """Return the 9 x 5 x 2 cell table: one row per model, effort and phase.

    Columns: phase, model, effort, n, mean, sd, values (array).

    ``fig_effort_total.csv`` already excludes the null sweep run. Once that run
    is no longer excluded — its re-run having landed — the affected cell is
    rebuilt from ``stats_sweep.csv``, which is regenerated more often.
    """
    df = pd.read_csv(os.path.join(BENCH_DIR, "fig_effort_total.csv"))
    df["values"] = df["values"].map(_parse_values)

    if NULL_SWEEP_RUN not in _aborted_runs("fig_effort_total.csv"):
        sweep = pd.read_csv(os.path.join(BENCH_DIR, "stats_sweep.csv"))
        cell = sweep[(sweep.phase == "A") & (sweep.model == "opus46")
                     & (sweep.effort == "medium")]
        vals = cell.total.to_numpy(dtype=float)
        mask = ((df.phase == "A") & (df.model == "opus46")
                & (df.effort == "medium"))
        df.loc[mask, "n"] = len(vals)
        df.loc[mask, "mean"] = vals.mean()
        df.loc[mask, "sd"] = vals.std(ddof=1)
        df.loc[mask, "values"] = pd.Series([vals], index=df.index[mask])

    return df


def effort_matrices():
    """Return (mean_A, mean_B, diff, n_A, n_B) as 9 x 5 arrays.

    Rows follow MODEL_ORDER, columns follow EFFORT_ORDER. ``diff`` is A - B.
    """
    df = load_effort_grid()
    shape = (len(MODEL_ORDER), len(EFFORT_ORDER))
    mean_a = np.full(shape, np.nan)
    mean_b = np.full(shape, np.nan)
    n_a = np.zeros(shape, dtype=int)
    n_b = np.zeros(shape, dtype=int)

    for i, model in enumerate(MODEL_ORDER):
        for j, effort in enumerate(EFFORT_ORDER):
            for phase, means, ns in (("A", mean_a, n_a), ("B", mean_b, n_b)):
                row = df[(df.phase == phase) & (df.model == model)
                         & (df.effort == effort)]
                if len(row) == 1:
                    means[i, j] = float(row["mean"].iloc[0])
                    ns[i, j] = int(row["n"].iloc[0])

    return mean_a, mean_b, mean_a - mean_b, n_a, n_b


def load_sweep_runs():
    """Return per-run totals for the confirmatory grid (low / medium / max)."""
    df = pd.read_csv(os.path.join(BENCH_DIR, "stats_sweep.csv"))
    return df[~df.run.isin(_aborted_runs("stats_sweep.csv"))]


# Rubric items in reporting order, with the short labels the panel prints.
# Instructed behaviour first, then analytical correctness — the split the
# analysis plan defines. Item 14 (reproducible_method) is cell-scope and is
# loaded by load_cell_scores, not listed here.
RUBRIC_ITEMS = [
    ("organized_project_structure", "instructed_behaviour", "project\nstructure"),
    ("initiated_data_cache", "instructed_behaviour", "data\ncache"),
    ("asked_for_clarification", "instructed_behaviour", "asked for\nclarification"),
    ("wrote_analysis_log", "instructed_behaviour", "analysis\nlog"),
    ("flagged_caveats", "instructed_behaviour", "flagged\ncaveats"),
    ("suggested_followups", "instructed_behaviour", "suggested\nfollow-ups"),
    ("correct_session", "analytical_correctness", "correct\nsession"),
    ("no_spikes_in_empty_gaps", "analytical_correctness", "no spikes\nin gaps"),
    ("no_shuffle_induced_rate_change", "analytical_correctness", "no rate\nchange"),
    ("published_method", "analytical_correctness", "published\nmethod"),
    ("fitted_to_spike_data", "analytical_correctness", "fitted to\nspikes"),
    ("no_scope_reduction", "analytical_correctness", "no scope\nreduction"),
    ("state_count_not_at_ceiling", "analytical_correctness", "state count\nnot at cap"),
]

CATEGORY_LABELS = {
    "instructed_behaviour": "instructed behaviour",
    "analytical_correctness": "analytical correctness",
    "cell_scope": "cell scope",
}

VERDICT_POINTS = {"green": 1.0, "orange": 0.5, "red": 0.0}


def load_ablation_items():
    """Mean score per rubric item per arm, as a (4, 13) array over ARM_ORDER.

    Read from each run's scorecard rather than from ``fig_ablation.csv``, which
    is regenerated on its own schedule and can lag the run set: the per-item
    display and the totals beside it must describe the same runs, or the panel
    contradicts itself.
    """
    runs = load_ablation_runs()
    ids = [item for item, _, _ in RUBRIC_ITEMS]
    out = np.full((len(ARM_ORDER), len(ids)), np.nan)

    for ai, arm in enumerate(ARM_ORDER):
        scores = {item: [] for item in ids}
        for run in runs[runs.arm == arm].run:
            path = os.path.join(RUNS_DIR, ARM_DIRS[arm], run, "scorecard.json")
            if not os.path.exists(path):
                warnings.warn(f"no scorecard at {path}; excluded from the "
                              "per-item display", stacklevel=2)
                continue
            with open(path, encoding="utf-8") as fh:
                card = json.load(fh)
            for verdict in card["verdicts"]:
                if verdict["id"] in scores:
                    scores[verdict["id"]].append(
                        VERDICT_POINTS[verdict["verdict"]])
        for ii, item in enumerate(ids):
            if scores[item]:
                out[ai, ii] = float(np.mean(scores[item]))

    return out


def load_ablation_runs():
    """Return per-run totals for the four ablation arms."""
    df = pd.read_csv(os.path.join(BENCH_DIR, "stats_ablation.csv"))
    return df[~df.run.isin(_aborted_runs("stats_ablation.csv"))]


# Sweep run directories and the run-name pattern, mirroring load() in
# benchmark_v2/tools/stats_sweep.py so both resolve the same runs.
SWEEP_DIRS = ("effort", "sweep", "round6", "round6B")
SWEEP_RUN_PAT = re.compile(r"^([AB])_([a-z0-9]+)_(low|medium|max)_rep(\d+)$")
QUARANTINE_MARKS = (".crashed", ".killed", ".orphaned", ".aborted",
                    ".standby", ".ratelimited")


def load_cell_scores():
    """Rubric item 14 (``reproducible_method``) — one score per cell.

    Written by benchmark_v2/tools/grade_cell.py as ``cell_scores.json``. The
    score is max_agreeing / n_attempted: the largest group of a cell's ten
    replicates that reported the population-coupling value in mutually
    commensurable units.

    Cell-scope, deliberately. It exists once per cell rather than once per run,
    so it is NOT added to the 13-item totals — doing so would paste one value
    onto all ten runs of a cell, adding between-cell signal with no within-cell
    variance and making every per-run standard error anti-conservative. Returns
    a frame of phase, model, effort, score, and the diagnostics behind it.
    """
    path = os.path.join(BENCH_DIR, "cell_scores.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    pretty_to_key = {v: k for k, v in MODEL_LABELS.items()}
    rows = []
    for label, v in raw.items():
        phase, pretty, effort = label.split(" ")
        if pretty not in pretty_to_key:
            warnings.warn(f"unknown model {pretty!r} in {path}", stacklevel=2)
            continue
        rows.append({
            "phase": phase, "model": pretty_to_key[pretty], "effort": effort,
            "score": float(v["score"]), "max_agreeing": v["max_agreeing"],
            "n_attempted": v["n_attempted"],
            "n_underspecified": v["n_underspecified"],
            "n_classes": len(v["distribution"]),
        })
    return pd.DataFrame(rows)


def load_ablation_consistency():
    """Rubric item 14 for the four ablation arms, as {arm: score}.

    Computed by calling grade_cell.py's own scorer rather than read from
    ``cell_scores.json``: that file currently holds the 54 grid cells, and the
    arm scores would only reach it by re-running the tool in its other mode,
    which overwrites the grid.

    One cell per arm, so one number each — four values, no replicates of the
    measure itself. Nothing inferential can be hung on them.
    """
    if TOOLS_DIR not in sys.path:
        sys.path.insert(0, TOOLS_DIR)
    from pathlib import Path

    from grade_cell import ARMS, load, score_cell

    out = {}
    for arm, label, sub, prefix in ARMS:
        # load() joins with pathlib, so it needs a Path, not a str.
        runs = load(Path(RUNS_DIR), sub, prefix)
        if not runs:
            warnings.warn(f"no coupling fingerprints for ablation arm {arm}",
                          stacklevel=2)
            continue
        score, _top, _n, _dist, _under = score_cell(runs)
        if score is not None:
            out[arm] = float(score)
    return out


def consistency_matrices():
    """Return (with, without, difference) as 9 x 3 arrays over the grid.

    Rows follow MODEL_ORDER, columns CONFIRMATORY_EFFORTS. Item 14 was graded
    only on the confirmatory grid, so there are no high/xhigh columns here —
    unlike the score overview, which shows all five.
    """
    df = load_cell_scores()
    shape = (len(MODEL_ORDER), len(CONFIRMATORY_EFFORTS))
    a = np.full(shape, np.nan)
    b = np.full(shape, np.nan)
    for i, model in enumerate(MODEL_ORDER):
        for j, effort in enumerate(CONFIRMATORY_EFFORTS):
            for phase, mat in (("A", a), ("B", b)):
                row = df[(df.phase == phase) & (df.model == model)
                         & (df.effort == effort)]
                if len(row) == 1:
                    mat[i, j] = row["score"].iloc[0]
    return a, b, a - b


def load_run_costs():
    """Per-run API cost in USD, read from each run manifest.

    No CSV carries cost, so it is taken from the manifests directly. Only
    completed runs are returned, matching the rule stats_sweep.py applies to
    scores: an aborted run is an infrastructure artefact, and its recorded cost
    of zero would drag any cost average toward zero.
    """
    rows = []
    for sub in SWEEP_DIRS:
        base = os.path.join(RUNS_DIR, sub)
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            d = os.path.join(base, name)
            if not os.path.isdir(d) or any(q in name for q in QUARANTINE_MARKS):
                continue
            m = SWEEP_RUN_PAT.match(name)
            manifest = os.path.join(d, "run_manifest.json")
            if not m or not os.path.exists(manifest):
                continue
            with open(manifest, encoding="utf-8") as fh:
                man = json.load(fh)
            if man.get("exit_status") != "completed":
                continue
            rows.append({
                "run": name, "phase": m.group(1), "model": m.group(2),
                "effort": m.group(3),
                "cost_usd": float(man.get("total_cost_usd") or 0.0),
                "tool_calls": man.get("total_tool_calls"),
            })
    return pd.DataFrame(rows)


def load_sweep_with_cost():
    """Sweep scores joined to their per-run cost, inner join on run name."""
    scores = load_sweep_runs()
    costs = load_run_costs()
    merged = scores.merge(costs[["run", "cost_usd", "tool_calls"]],
                          on="run", how="inner")
    missing = len(scores) - len(merged)
    if missing:
        warnings.warn(f"{missing} scored runs have no manifest cost and are "
                      "excluded from the cost analysis", stacklevel=2)
    return merged


# The benchmark's own tool-call classifier, reused rather than reimplemented so
# that "documentation read" means here exactly what it means everywhere else in
# the pipeline. Its CLI scans only round6/round6B/sweep, which is 180 of the 540
# runs; the loader below applies the same rules to all of them.
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")


def load_tool_summary():
    """Per-run tool-call summary for the confirmatory grid.

    Columns: run, phase, model, effort, total (rubric score), n_calls,
    doc_reads, failures, doc_frac, err_rate.

    ``doc_frac`` and ``err_rate`` are per-run rates. A run-weighted mean of
    those differs from a call-weighted rate over the pooled calls, and the two
    are not interchangeable — a figure has to say which it shows.
    """
    if TOOLS_DIR not in sys.path:
        sys.path.insert(0, TOOLS_DIR)
    from categorize_tools import classify

    scores = load_sweep_runs().set_index("run")
    rows = []
    for sub in SWEEP_DIRS:
        base = os.path.join(RUNS_DIR, sub)
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            m = SWEEP_RUN_PAT.match(name)
            if (not m or name not in scores.index
                    or any(q in name for q in QUARANTINE_MARKS)):
                continue
            calls_path = os.path.join(base, name, "toolcalls.json")
            if not os.path.exists(calls_path):
                warnings.warn(f"no toolcalls.json for {name}", stacklevel=2)
                continue
            with open(calls_path, encoding="utf-8") as fh:
                calls = json.load(fh)
            if not calls:
                continue
            cats = [classify(c) for c in calls]
            n = len(cats)
            rows.append({
                "run": name, "phase": m.group(1), "model": m.group(2),
                "effort": m.group(3), "total": float(scores.loc[name, "total"]),
                "n_calls": n,
                "doc_reads": cats.count("DOC_READS"),
                "failures": cats.count("FAILED_EXECUTIONS"),
                "doc_frac": cats.count("DOC_READS") / n,
                "err_rate": cats.count("FAILED_EXECUTIONS") / n,
            })
    return pd.DataFrame(rows)
