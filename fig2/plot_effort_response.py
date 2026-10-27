"""Plot the effort response of both arms — Figure 2, effort panel.

Two tiers sharing an effort axis: the arms above, the gap with its 95% CI
below. Does a larger reasoning budget close the gap? The near-flat lower tier is the answer.

Only ``low``, ``medium`` and ``max`` appear. ``high`` and ``xhigh`` stand at n=3
and are outside the confirmatory grid, so they are shown in the overview panel
but take no part here.

The intervals are descriptive. Inference comes from the phase x effort
interaction terms of the quasi-binomial GLM fitted by tools/stats_sweep.py,
which the caption quotes.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from plot_config import apply_style, save_fig, get_panel_width, add_panel_label
apply_style()

from benchmark_v2_data import (
    CONFIRMATORY_EFFORTS, EFFORT_LABELS, load_sweep_runs,
)
from panel_common import (
    ARM_TESTS, GLM_TESTS, bh, mean_ci, plot_arms, plot_gap_tier, stars,
    welch_ci, welch_p,
)


def effort_response(df):
    """Per-effort arm means with 95% CI, and the gap with its Welch 95% CI."""
    means = {"A": [], "B": []}
    cis = {"A": [], "B": []}
    diff, ci_lo, ci_hi, n_a, n_b, pvals = [], [], [], [], [], []

    for effort in CONFIRMATORY_EFFORTS:
        arms = {}
        for phase in ("A", "B"):
            vals = df[(df.phase == phase) & (df.effort == effort)].total.to_numpy()
            arms[phase] = vals
            m, ci = mean_ci(vals)
            means[phase].append(m)
            cis[phase].append(ci)
        d, half = welch_ci(arms["A"], arms["B"])
        pvals.append(welch_p(arms["A"], arms["B"]))
        diff.append(d)
        ci_lo.append(d - half)
        ci_hi.append(d + half)
        n_a.append(len(arms["A"]))
        n_b.append(len(arms["B"]))

    return {
        "means": means, "cis": cis,
        "diff": np.array(diff), "ci_lo": np.array(ci_lo), "ci_hi": np.array(ci_hi),
        "n_a": np.array(n_a), "n_b": np.array(n_b),
        "p": np.array(pvals), "p_bh": bh(pvals),
        "pooled_gap": (df[df.phase == "A"].total.mean()
                       - df[df.phase == "B"].total.mean()),
    }


def plot_effort_response(ax_levels, ax_gap, res):
    """Draw both tiers onto two axes sharing an effort axis."""
    x = np.arange(len(CONFIRMATORY_EFFORTS))

    # The p-values are deliberately not drawn: they belong in the caption of
    # the assembled figure, and the slide deck reads better without them.
    # plot_arms and plot_gap_tier still accept them (ARM_TESTS, GLM_TESTS).
    plot_arms(ax_levels, x, res["means"], res["cis"], legend_loc="lower left",
              marks=[stars(v) for v in res["p_bh"]])
    plot_gap_tier(ax_gap, x, res["diff"], res["ci_lo"], res["ci_hi"],
                  res["pooled_gap"])

    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels([EFFORT_LABELS[e] for e in CONFIRMATORY_EFFORTS])
    for ax in (ax_levels, ax_gap):
        ax.set_xlim(-0.35, len(x) - 0.65)
    ax_levels.tick_params(labelbottom=False)


if __name__ == "__main__":
    df = load_sweep_runs()
    res = effort_response(df)

    fig = plt.figure(figsize=(get_panel_width(n_columns=3), 3.1))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.0, 1.0],
                           hspace=0.12)
    ax_levels = fig.add_subplot(gs[0])
    ax_gap = fig.add_subplot(gs[1], sharex=ax_levels)

    plot_effort_response(ax_levels, ax_gap, res)
    add_panel_label(ax_levels, "B", x=-0.22, y=1.02)
    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "effort_response.png"))

    # Numbers the caption has to carry.
    print(f"pooled gap {res['pooled_gap']:+.2f}")
    for i, effort in enumerate(CONFIRMATORY_EFFORTS):
        print(f"  {effort:<7} with {res['means']['A'][i]:6.2f} "
              f"(n={res['n_a'][i]})  without {res['means']['B'][i]:5.2f} "
              f"(n={res['n_b'][i]})  gap {res['diff'][i]:+.2f} "
              f"[{res['ci_lo'][i]:+.2f}, {res['ci_hi'][i]:+.2f}]")
