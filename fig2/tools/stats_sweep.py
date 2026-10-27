#!/usr/bin/env python
"""Parameter-sweep statistics: phase x effort x model, n=10 per cell.

Design. 9 models x 2 phases (A = SpikeLab, B = baseline) x 3
effort levels (low, medium, max) x n=10 = 540 runs, fully balanced.

Outcome and model. Each of 13 rubric items contributes 0, 1 or 2
half-points, so a run is k successes out of 26. Fitted as a binomial GLM with a
quasi-binomial scale for overdispersion. The quasi-binomial scale inflates the
standard errors by the estimated dispersion, the conservative direction, and
leaves the point estimates unchanged.

Separation. Cells that score 26/26 in every replicate send an
unpenalised binomial coefficient to infinity. The check is printed per cell, and
the primary phase test is a Welch contrast on totals, which is finite regardless.

Panels. Not tested. Only the combined 13-item score is analysed here.

Multiplicity. One model with interactions rather than 27
per-cell tests. Where per-cell contrasts are reported they are descriptive, with
Benjamini-Hochberg applied across the 27 cells when read inferentially.

Totals are out of 13, not 14 (`reproducible_method` is cell-scope, scored per
cell by grade_cell.py, and never added to a per-run total).

Usage:  python stats_sweep.py
"""

import os
import json
import re
import sys
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
HALVES = {"green": 2, "orange": 1, "red": 0}
QUARANTINED = (".crashed", ".killed", ".orphaned", ".aborted", ".standby", ".ratelimited")
EFFORTS = ["low", "medium", "max"]
MODELS = ["haiku45", "sonnet45", "sonnet46", "sonnet5",
          "opus45", "opus46", "opus47", "opus48", "opus5"]
PRETTY = {"haiku45": "haiku-4-5", "sonnet45": "sonnet-4-5", "sonnet46": "sonnet-4-6",
          "sonnet5": "sonnet-5", "opus45": "opus-4-5", "opus46": "opus-4-6",
          "opus47": "opus-4-7", "opus48": "opus-4-8", "opus5": "opus-5"}
PAT = re.compile(r"^([AB])_([a-z0-9]+)_(low|medium|max)_rep(\d+)$")


def load(root):
    rows = []
    for sub in ("effort", "sweep", "round6", "round6B"):
        for d in (root / sub).glob("*"):
            if not d.is_dir() or any(q in d.name for q in QUARANTINED):
                continue
            m = PAT.match(d.name)
            f = d / "scorecard.json"
            if not m or not f.exists():
                continue
            # An aborted run is an infrastructure artefact, not a measurement --
            # scoring it 0/13 scores the harness. plot_grid_heatmap.py has always
            # applied this rule; this script did not, so the figures and the
            # statistics disagreed on A_opus46_medium_rep03 (n=9 vs n=10 for that
            # cell). A COMPLETED run that produced nothing is a genuine model
            # failure and still counts; only incompleteness is excluded.
            mf = d / "run_manifest.json"
            if mf.exists():
                try:
                    if json.loads(mf.read_text(encoding="utf-8")).get("exit_status") != "completed":
                        continue
                except (ValueError, OSError):
                    pass
            v = json.loads(f.read_text(encoding="utf-8"))["verdicts"]
            rows.append({
                "phase": m.group(1), "model": m.group(2), "effort": m.group(3),
                "rep": int(m.group(4)), "run": d.name,
                "total": sum(POINTS.get(x["verdict"], 0.0) for x in v),
                "k": sum(HALVES.get(x["verdict"], 0) for x in v),
                "n": 2 * len(v),
            })
    return pd.DataFrame(rows)


def hedges_g(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return np.inf if np.mean(x) != np.mean(y) else np.nan
    return (np.mean(x) - np.mean(y)) / sp * (1 - 3 / (4 * (nx + ny) - 9))


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m, adj, prev = len(p), np.empty(len(p)), 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * m / (m - rank))
        adj[idx] = prev
    return adj


def quasi(spec, d):
    """Binomial GLM with the Pearson dispersion applied as an explicit scale.

    fit(scale="X2") does NOT do this -- on this data it returns a dispersion far
    below 1 and shrinks the standard errors. Passing the float gives the intended
    SE_quasi = SE_binomial * sqrt(dispersion), verified by the assertion below.
    """
    base = smf.glm(spec, data=d, family=sm.families.Binomial()).fit()
    disp = float(base.pearson_chi2 / base.df_resid)
    m = smf.glm(spec, data=d, family=sm.families.Binomial()).fit(scale=disp)
    ratio = m.bse.iloc[-1] / base.bse.iloc[-1]
    assert abs(ratio - np.sqrt(disp)) < 1e-6, "quasi-binomial scaling did not apply"
    return m, disp


def quasi_f(full, reduced, dispersion):
    """Quasi-likelihood F test for a nested pair of quasi-binomial fits.

    Under a quasi-binomial scale the deviance difference is not chi-square: it is
    divided by the dispersion and referred to F, the standard quasi-likelihood
    analogue. The test depends only on fitted values, so it stays valid where a
    Wald test on individual coefficients would not -- which is what makes it
    usable on the three-way fit, whose separated cells send a coefficient to an
    arbitrary magnitude while leaving every fitted cell mean correct.

    `full` is the larger model, `reduced` the smaller, `dispersion` the Pearson
    dispersion of the larger.
    """
    num_df = int(reduced.df_resid - full.df_resid)
    f = ((reduced.deviance - full.deviance) / num_df) / dispersion
    return f, num_df, int(full.df_resid), float(stats.f.sf(f, num_df, full.df_resid))


def main():
    root = BENCH_ROOT
    df = load(root)
    if df.empty:
        sys.exit("no sweep scorecards found")
    df["fail"] = df["n"] - df["k"]

    print("=" * 78)
    print("PARAMETER SWEEP STATISTICS -- 9 models x 2 phases x 3 efforts")
    print("totals out of 13 (item 14 is cell-scope, see grade_cell.py)")
    print("=" * 78)

    cells = df.groupby(["effort", "phase", "model"]).size()
    print(f"\n{len(df)} runs, {len(cells)} cells, n per cell: "
          f"{cells.min()}-{cells.max()}"
          + ("  (BALANCED)" if cells.min() == cells.max() else "  (UNBALANCED)"))

    # ---- main effect of phase ----------------------------------------------
    print("\n" + "=" * 78)
    print("main effect of phase (SpikeLab raises the score)")
    print("=" * 78)
    a = df.loc[df.phase == "A", "total"].values
    b = df.loc[df.phase == "B", "total"].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"\n  phase A {a.mean():.2f} +- {a.std(ddof=1):.2f} (n={len(a)})")
    print(f"  phase B {b.mean():.2f} +- {b.std(ddof=1):.2f} (n={len(b)})")
    print(f"  difference {a.mean() - b.mean():+.2f}   Hedges g = {hedges_g(a, b):.2f}   "
          f"Welch p = {p:.3g}")

    m, disp = quasi("k + fail ~ C(phase) * C(effort) + C(model)", df)
    print(f"\n  quasi-binomial GLM, dispersion = {disp:.2f}")
    print(f"  {'term':<34}{'beta':>9}{'SE':>8}{'p':>12}")
    for name in m.params.index:
        if name.startswith("C(model)"):
            continue
        print(f"  {name:<34}{m.params[name]:>9.3f}{m.bse[name]:>8.3f}{m.pvalues[name]:>12.3g}")

    # per-cell direction: does the phase effect reverse anywhere?
    print("\n  per-cell A-B (descriptive; BH across the 27 cells):")
    rows = []
    for e in EFFORTS:
        for mod in MODELS:
            x = df[(df.effort == e) & (df.model == mod) & (df.phase == "A")].total.values
            y = df[(df.effort == e) & (df.model == mod) & (df.phase == "B")].total.values
            rows.append((e, mod, x.mean() - y.mean(), stats.ttest_ind(x, y, equal_var=False).pvalue))
    q = bh([r[3] for r in rows])
    neg = [r for r in rows if r[2] <= 0]
    print(f"    cells where A <= B: {len(neg)} of {len(rows)}"
          + ("" if neg else "   <- no reversal anywhere"))
    print(f"    smallest gap: {min(r[2] for r in rows):+.2f} "
          f"({[PRETTY[r[1]] + ' ' + r[0] for r in rows if r[2] == min(x[2] for x in rows)][0]})")
    print(f"    largest gap : {max(r[2] for r in rows):+.2f} "
          f"({[PRETTY[r[1]] + ' ' + r[0] for r in rows if r[2] == max(x[2] for x in rows)][0]})")
    print(f"    cells significant at BH q<0.05: {int((q < 0.05).sum())} of {len(rows)}")

    # ---- does effort substitute for the library? ---------------------------
    print("\n" + "=" * 78)
    print("phase x effort -- does reasoning budget substitute for the library?")
    print("=" * 78)
    print("\n  (a) interaction: does the A-B gap shrink as effort rises?\n")
    print(f"  {'effort':<10}{'phase A':>10}{'phase B':>10}{'A - B':>9}")
    for e in EFFORTS:
        x = df[(df.effort == e) & (df.phase == "A")].total
        y = df[(df.effort == e) & (df.phase == "B")].total
        print(f"  {e:<10}{x.mean():>10.2f}{y.mean():>10.2f}{x.mean() - y.mean():>9.2f}")
    inter = [n for n in m.params.index if "C(phase)" in n and "C(effort)" in n]
    for n in inter:
        print(f"    {n:<40} beta={m.params[n]:+.3f}  p={m.pvalues[n]:.3g}")

    # The two coefficients above are separate contrasts against `low`. The joint
    # test of the whole interaction is the one to quote, and it is the same kind
    # of nested F test used for the interactions below, so the tests stay comparable.
    _spec_2way = ("k + fail ~ C(phase)*C(effort) + C(phase)*C(model)"
                  " + C(effort)*C(model)")
    _spec_no_pe = ("k + fail ~ C(phase) + C(effort) + C(phase)*C(model)"
                   " + C(effort)*C(model)")
    _m2, _d2 = quasi(_spec_2way, df)
    _m_no_pe, _ = quasi(_spec_no_pe, df)
    f_pe, ndf_pe, ddf_pe, p_pe = quasi_f(_m2, _m_no_pe, _d2)
    print(f"\n    joint phase x effort           F({ndf_pe}, {ddf_pe}) = "
          f"{f_pe:.2f}   p = {p_pe:.3g}")

    print("\n  (b) directional contrast: is phase B at max >= phase A at low, per model?\n")
    print(f"  {'model':<13}{'A low':>8}{'B max':>8}{'B max - A low':>15}   substitutes?")
    subs = 0
    for mod in MODELS:
        al = df[(df.model == mod) & (df.effort == "low") & (df.phase == "A")].total.mean()
        bm = df[(df.model == mod) & (df.effort == "max") & (df.phase == "B")].total.mean()
        ok = bm >= al
        subs += ok
        print(f"  {PRETTY[mod]:<13}{al:>8.2f}{bm:>8.2f}{bm - al:>15.2f}   {'YES' if ok else 'no'}")
    print(f"\n    substitution holds in {subs} of {len(MODELS)} models"
          + ("   <- reasoning budget substitutes for the library" if subs else
             "   <- reasoning budget does not substitute for the library"))

    # ---- interactions ------------------------------------------------------
    print("\n" + "=" * 78)
    print("interactions")
    print("=" * 78)
    # phase x model and effort x model are tested in the all-two-way model, not
    # the three-way one.
    # Marginality forbids dropping phase:model while phase:effort:model, which
    # contains it, is still in the model. The three-way is tested separately,
    # against the two-way model.
    spec_3way = "k + fail ~ C(phase) * C(effort) * C(model)"
    spec_2way = ("k + fail ~ C(phase)*C(effort) + C(phase)*C(model)"
                 " + C(effort)*C(model)")
    spec_no_phase_model = "k + fail ~ C(phase)*C(effort) + C(effort)*C(model)"
    spec_no_effort_model = "k + fail ~ C(phase)*C(effort) + C(phase)*C(model)"

    m3, disp3 = quasi(spec_3way, df)
    m2, disp2 = quasi(spec_2way, df)
    m_no_pm, _ = quasi(spec_no_phase_model, df)
    m_no_em, _ = quasi(spec_no_effort_model, df)

    print(f"\n  two-way dispersion = {disp2:.2f};  three-way = {disp3:.2f}")
    print("\n  quasi-likelihood F tests (dispersion from the larger model):\n")
    for label, full_m, red_m, disp in (
        ("phase x model", m2, m_no_pm, disp2),
        ("effort x model", m2, m_no_em, disp2),
        ("phase x effort x model", m3, m2, disp3),
    ):
        f, ndf, ddf, p = quasi_f(full_m, red_m, disp)
        print(f"    {label:<28} F({ndf}, {ddf}) = {f:6.2f}   p = {p:.3g}")

    # Sensitivity: the three-way fit is saturated at the cell level, so its
    # dispersion is the within-cell overdispersion free of lack-of-fit. Re-run
    # phase x model against it, in case the choice of dispersion is what drives the result.
    f_alt, ndf_alt, ddf_alt, p_alt = quasi_f(m2, m_no_pm, disp3)
    print(f"\n    phase x model using the saturated-model dispersion ({disp3:.2f}): "
          f"F({ndf_alt}, {ddf_alt}) = {f_alt:.2f}, p = {p_alt:.3g}")

    ceiling = [f"{ph} {PRETTY[mod]} {ef}"
               for (ph, mod, ef), g in df.groupby(["phase", "model", "effort"])
               if (g.k == g.n).all()]
    if ceiling:
        print("\n  separation: every replicate at 26/26 in "
              f"{', '.join(ceiling)}.")
        print("    Cell means are still fitted exactly, so the deviance-based F")
        print("    tests above hold; individual three-way coefficients do not "
              f"(max |beta| = {np.abs(m3.params).max():.1f}).")
    print("\n  per-model A-B gap (does the benefit depend on the model?):\n")
    print(f"  {'model':<13}{'A':>8}{'B':>8}{'A - B':>9}")
    for mod in MODELS:
        x = df[(df.model == mod) & (df.phase == "A")].total
        y = df[(df.model == mod) & (df.phase == "B")].total
        print(f"  {PRETTY[mod]:<13}{x.mean():>8.2f}{y.mean():>8.2f}{x.mean() - y.mean():>9.2f}")

    # ---- dispersion is a result ---------------------------------------------
    print("\n" + "=" * 78)
    print("Within-cell SD (variance is a result, not a nuisance)")
    print("=" * 78)
    print(f"\n  {'model':<13}" + "".join(f"{e + ' A':>9}{e + ' B':>9}" for e in EFFORTS))
    for mod in MODELS:
        cellsd = []
        for e in EFFORTS:
            for ph in "AB":
                cellsd.append(df[(df.model == mod) & (df.effort == e) &
                                 (df.phase == ph)].total.std(ddof=1))
        print(f"  {PRETTY[mod]:<13}" + "".join(f"{s:>9.2f}" for s in cellsd))

    out = root / "figures" / "stats_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nper-run data written to {out}")


if __name__ == "__main__":
    main()
