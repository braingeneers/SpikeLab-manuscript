"""Plot the per-model response of both arms — Figure 2, model panel.

Two tiers sharing a model axis, drawn in the same grammar as the effort panel so
the two can be read against each other: there the gap holds a constant width
across effort, here it visibly varies across models.

Lines break at each model-family boundary. Version order within a family is
meaningful; order between haiku, sonnet and opus is not, and a line drawn
straight through would imply a continuum that does not exist.

The intervals are descriptive. They are nine unadjusted contrasts, not the
27-cell family corrected across elsewhere, and the phase x model interaction
is not tested here; see the caption.
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
    MODEL_ORDER, MODEL_LABELS, MODEL_LABELS_SHORT, model_colors,
    model_family_groups, load_sweep_runs,
)
from panel_common import (
    GLM_TESTS, bh, mean_ci, plot_arms, plot_gap_tier, stars, welch_ci,
    welch_p,
)


def model_response(df):
    """Per-model arm means with 95% CI, and the gap with its Welch 95% CI."""
    means = {"A": [], "B": []}
    cis = {"A": [], "B": []}
    diff, ci_lo, ci_hi, n_a, n_b, pvals = [], [], [], [], [], []

    for model in MODEL_ORDER:
        arms = {}
        for phase in ("A", "B"):
            vals = df[(df.phase == phase) & (df.model == model)].total.to_numpy()
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


def plot_model_response(ax_levels, ax_gap, res):
    """Draw both tiers onto two axes sharing a model axis."""
    x = np.arange(len(MODEL_ORDER))
    groups = model_family_groups()

    plot_arms(ax_levels, x, res["means"], res["cis"], groups=groups,
              legend_loc="lower right",
              marks=[stars(v) for v in res["p_bh"]])
    plot_gap_tier(ax_gap, x, res["diff"], res["ci_lo"], res["ci_hi"],
                  res["pooled_gap"])

    # Abbreviated and coloured to match the overview panel and the cost
    # panel's legend, so one model reads the same way across all three.
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels([MODEL_LABELS_SHORT[m] for m in MODEL_ORDER],
                           rotation=45, ha="right", rotation_mode="anchor")
    colors = model_colors()
    for tick, model in zip(ax_gap.get_xticklabels(), MODEL_ORDER):
        tick.set_color(colors[model])
        tick.set_fontweight("bold")
    for ax in (ax_levels, ax_gap):
        ax.set_xlim(-0.6, len(x) - 0.4)
    ax_levels.tick_params(labelbottom=False)


if __name__ == "__main__":
    df = load_sweep_runs()
    res = model_response(df)

    fig = plt.figure(figsize=(get_panel_width(n_columns=2), 3.1))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.0, 1.0],
                           hspace=0.12)
    ax_levels = fig.add_subplot(gs[0])
    ax_gap = fig.add_subplot(gs[1], sharex=ax_levels)

    plot_model_response(ax_levels, ax_gap, res)
    add_panel_label(ax_levels, "C", x=-0.13, y=1.02)
    save_fig(fig, os.path.join(SCRIPT_DIR, "panels", "model_response.png"))

    # Numbers the caption has to carry.
    print(f"pooled gap {res['pooled_gap']:+.2f}")
    for i, m in enumerate(MODEL_ORDER):
        print(f"  {MODEL_LABELS[m]:<12} with {res['means']['A'][i]:5.2f} "
              f"(n={res['n_a'][i]})  without {res['means']['B'][i]:5.2f} "
              f"(n={res['n_b'][i]})  gap {res['diff'][i]:+.2f} "
              f"[{res['ci_lo'][i]:+.2f}, {res['ci_hi'][i]:+.2f}]")
    print(f"\ngap range {res['diff'].min():+.2f} to {res['diff'].max():+.2f}; "
          f"CIs excluding zero: {(res['ci_lo'] > 0).sum()} of {len(MODEL_ORDER)}")
