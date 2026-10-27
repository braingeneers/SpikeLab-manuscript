#!/usr/bin/env python
"""Ablation statistics: instructions vs tools, sonnet-4-6 medium, n=10 per arm.

Outcome. Each of 13 rubric items contributes 0, 1 or 2
half-points, so a run is k successes out of 26 Bernoulli trials. The score is
bounded, discrete and piles up on the ceiling, so it is modelled as binomial
rather than Gaussian -- 13/13 is a legitimate observation, not a boundary
violation.

Overdispersion. Runs within an arm vary more than binomial sampling allows,
so plain binomial errors would be too small. That is handled with a
quasi-binomial scale: the
point estimates are unchanged and the standard errors are inflated by the
estimated dispersion. This is the conservative direction.

Separation. Arm A contains runs at 26/26. A cell that is
entirely successful sends an unpenalised binomial coefficient to infinity. No
arm here is *completely* separated (arm A has non-perfect runs), so the fit is
finite, but the check is printed rather than assumed.

Panels. Only the combined 13-item score is tested. Correctness (7 items) and
instructed behaviour (6 items) are displayed descriptively in fig_ablation.png
but not tested separately. Pass "correctness" or "instructed" as argv[1] to run them.

Multiplicity. Benjamini-Hochberg across the six pairwise arm
contrasts.

Totals are out of 13, not 14: item 14 `reproducible_method` is cell-scope,
scored per cell by grade_cell.py, and never added to a per-run total.

Usage:  python stats_ablation.py
"""

import os
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Benchmark data root: $BENCH_RUNS if set, else fig2/data (see prepare_data.py).
BENCH_ROOT = Path(os.environ.get("BENCH_RUNS")
                  or Path(__file__).resolve().parents[1] / "data")

POINTS = {"green": 1.0, "orange": 0.5, "red": 0.0}
HALVES = {"green": 2, "orange": 1, "red": 0}          # half-points out of 2
QUARANTINED = (".crashed", ".killed", ".orphaned", ".aborted", ".standby", ".ratelimited")

ARMS = [
    ("A", "instructions+tools", "round6", "A_sonnet46_medium_rep"),
    ("C", "instructions only", "ablation", "C_sonnet46_medium_rep"),
    ("D", "tools only", "ablation", "D_sonnet46_medium_rep"),
    ("B", "neither", "round6B", "B_sonnet46_medium_rep"),
]


def load(root):
    rows = []
    for arm, label, sub, prefix in ARMS:
        for d in sorted((root / sub).glob(prefix + "*")):
            if not d.is_dir() or any(q in d.name for q in QUARANTINED):
                continue
            f = d / "scorecard.json"
            if not f.exists():
                continue
            # same rule as stats_sweep.py and plot_grid_heatmap.py: an aborted run
            # is an infrastructure artefact, a completed-but-empty run is a model
            # failure and counts. D_sonnet46_medium_rep03 was aborted mid-task-1
            # by a rate limit and scored 2.5/13 on one task's worth of output.
            mf = d / "run_manifest.json"
            if mf.exists():
                try:
                    if json.loads(mf.read_text(encoding="utf-8")).get("exit_status") != "completed":
                        continue
                except (ValueError, OSError):
                    pass
            verdicts = json.loads(f.read_text(encoding="utf-8"))["verdicts"]
            rec = {"arm": arm, "label": label, "run": d.name,
                   "total": sum(POINTS.get(v["verdict"], 0.0) for v in verdicts)}
            for panel, key in (("all", None),
                               ("correctness", "analytical_correctness"),
                               ("instructed", "instructed_behaviour")):
                sel = [v for v in verdicts if key is None or v["category"] == key]
                rec["k_" + panel] = sum(HALVES.get(v["verdict"], 0) for v in sel)
                rec["n_" + panel] = 2 * len(sel)
            rows.append(rec)
    return pd.DataFrame(rows)


def hedges_g(x, y):
    """Standardised difference with the small-sample correction."""
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / sp
    return d * (1 - 3 / (4 * (nx + ny) - 9))          # Hedges correction


def welch_ci(x, y, alpha=0.05):
    """Welch confidence interval on the mean difference (unequal variances)."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1) / nx, np.var(y, ddof=1) / ny
    se = np.sqrt(vx + vy)
    if se == 0:
        return (np.nan, np.nan)
    dof = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    diff = np.mean(x) - np.mean(y)
    return (diff - tcrit * se, diff + tcrit * se)


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = m - rank
        prev = min(prev, p[idx] * m / i)
        adj[idx] = prev
    return adj


def fit_panel(df, panel):
    """Quasi-binomial GLM: half-point successes out of trials, by arm."""
    d = df.copy()
    d["k"] = d["k_" + panel]
    d["n"] = d["n_" + panel]
    d["fail"] = d["n"] - d["k"]
    # arm B (the bare baseline) is the reference level
    d["arm"] = pd.Categorical(d["arm"], categories=["B", "D", "C", "A"])
    # Quasi-binomial done by hand, because fit(scale="X2") does NOT do it: on this
    # data it returns scale=0.056 and *shrinks* the standard errors, turning an
    # overdispersed fit into implausible p-values around 1e-90. Passing the Pearson
    # dispersion as an explicit float gives the intended inflation
    # (SE_quasi = SE_binomial * sqrt(dispersion)), which is the conservative
    # direction.
    spec = "k + fail ~ C(arm)"
    base = smf.glm(spec, data=d, family=sm.families.Binomial()).fit()
    disp = float(base.pearson_chi2 / base.df_resid)
    m = smf.glm(spec, data=d, family=sm.families.Binomial()).fit(scale=disp)
    # the scaling actually happened, rather than being assumed
    ratio = m.bse.iloc[-1] / base.bse.iloc[-1]
    assert abs(ratio - np.sqrt(disp)) < 1e-6, (
        "quasi-binomial scaling did not apply: SE ratio %.4f vs sqrt(disp) %.4f"
        % (ratio, np.sqrt(disp)))
    return d, m


def main():
    root = BENCH_ROOT
    df = load(root)
    if df.empty:
        sys.exit("no ablation scorecards found")

    print("=" * 78)
    print("ABLATION STATISTICS -- sonnet-4-6, medium")
    print("totals out of 13 (item 14 is cell-scope, see grade_cell.py)")
    print("=" * 78)

    print("\n## Descriptives\n")
    print(f"{'arm':<22}{'n':>3}{'mean':>8}{'sd':>7}{'min':>6}{'max':>6}   runs")
    for arm, label, _s, _p in ARMS:
        v = df.loc[df.arm == arm, "total"].values
        if not len(v):
            continue
        print(f"{arm + ' ' + label:<22}{len(v):>3}{v.mean():>8.2f}{v.std(ddof=1):>7.2f}"
              f"{v.min():>6.1f}{v.max():>6.1f}   {[float(x) for x in sorted(v)]}")

    # separation check, printed rather than assumed
    print("\n## Separation check (all 26 trials successful?)\n")
    for arm, label, _s, _p in ARMS:
        d = df[df.arm == arm]
        perfect = int((d.k_all == d.n_all).sum())
        print(f"  {arm}: {perfect}/{len(d)} runs at ceiling"
              + ("   <- fully separated, estimates unbounded" if perfect == len(d) else ""))

    # Pooled 13-item score only. The panel split is still shown descriptively in
    # fig_ablation.png, but no inferential test is run on it.
    wanted = sys.argv[1] if len(sys.argv) > 1 else "all"
    panels = [(p, n) for p, n in (("all", 13), ("correctness", 7), ("instructed", 6))
              if p in wanted.split(",")]
    for panel, nitems in panels:
        print("\n" + "=" * 78)
        print(f"## Panel: {panel}  ({nitems} items)")
        print("=" * 78)

        d, m = fit_panel(df, panel)
        print(f"\nQuasi-binomial GLM, reference = arm B."
              f"  dispersion = {m.scale:.2f}"
              + ("  (overdispersed; SEs inflated accordingly)" if m.scale > 1 else ""))
        print(f"\n{'term':<18}{'beta':>9}{'SE':>8}{'z':>8}{'p':>11}   odds ratio")
        for name in m.params.index:
            b, se, z, p = m.params[name], m.bse[name], m.tvalues[name], m.pvalues[name]
            print(f"{name:<18}{b:>9.3f}{se:>8.3f}{z:>8.2f}{p:>11.2e}   {np.exp(b):>7.2f}")

        # pairwise contrasts on the raw totals: Welch + Hedges g, BH across the six
        print(f"\nPairwise contrasts (Welch t on totals, BH-adjusted across 6):\n")
        pairs, res = list(combinations([a[0] for a in ARMS], 2)), []
        for a, b in pairs:
            x = d.loc[d.arm == a, "k_" + panel].values / 2.0
            y = d.loc[d.arm == b, "k_" + panel].values / 2.0
            t, p = stats.ttest_ind(x, y, equal_var=False)
            lo, hi = welch_ci(x, y)
            res.append((a, b, x.mean() - y.mean(), lo, hi, hedges_g(x, y), p))
        adj = bh([r[6] for r in res])
        print(f"{'contrast':<12}{'diff':>8}{'95% CI':>18}{'Hedges g':>11}{'p (BH)':>12}")
        for (a, b, diff, lo, hi, g, p), q in zip(res, adj):
            star = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
            ci = f"[{lo:+.2f}, {hi:+.2f}]"
            print(f"{a + ' vs ' + b:<12}{diff:>8.2f}{ci:>18}{g:>11.2f}{q:>12.2e} {star}")

    print("\n" + "=" * 78)
    print("Dispersion is reported as a result, not a nuisance.")
    print("=" * 78)
    print(f"\n{'arm':<22}{'sd (total)':>12}")
    for arm, label, _s, _p in ARMS:
        v = df.loc[df.arm == arm, "total"].values
        if len(v):
            print(f"{arm + ' ' + label:<22}{v.std(ddof=1):>12.2f}")

    out = root / "figures" / "stats_ablation.csv"
    df.to_csv(out, index=False)
    print(f"\nper-run data written to {out}")

    # Emit the pooled contrasts for plot_ablation.py to annotate from. The figure
    # must not recompute them: recomputation lets the brackets drift from the
    # numbers reported here, and it would make the plotting code depend on the
    # model-fitting stack.
    if "all" in wanted.split(","):
        d, _ = fit_panel(df, "all")
        rows = []
        for a, b in combinations([x[0] for x in ARMS], 2):
            x = d.loc[d.arm == a, "k_all"].values / 2.0
            y = d.loc[d.arm == b, "k_all"].values / 2.0
            lo, hi = welch_ci(x, y)
            rows.append([a, b, x.mean() - y.mean(), lo, hi, hedges_g(x, y),
                         stats.ttest_ind(x, y, equal_var=False).pvalue])
        qs = bh([r[6] for r in rows])
        cf = root / "figures" / "ablation_contrasts.csv"
        with open(cf, "w", encoding="utf-8") as fh:
            fh.write("a,b,diff,ci_lo,ci_hi,hedges_g,p,p_bh\n")
            for r, q in zip(rows, qs):
                fh.write("%s,%s,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g\n" % (*r, q))
        print(f"contrasts written to {cf}")


if __name__ == "__main__":
    main()
