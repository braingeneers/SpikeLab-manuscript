"""Figure 6 — compute figure-specific analysis results.

Computes burst dynamics analyses that are only used by Figure 6:
    1. Burst-to-burst correlations — slice-to-slice correlations averaged
       across units on the combined RateSliceStack.
    2. Unit-to-unit PCA — pairwise unit correlations per burst, lower-
       triangle feature extraction, and PCA embedding.
    3. Slice-to-slice time correlations — per-recording temporal similarity
       of burst-aligned firing rate patterns.
    4. Burst rank order — rank-order correlations between burst pairs
       using firing rate peak timing.
    5. Unit burst groups — classify units by 100% burst participation and
       compute per-recording unit orderings.

Prerequisites (from compute_shared.py):
    - {cond}/spikedata, {cond}/tburst, {cond}/burst_edges
    - {cond}/burst_rss, all/burst_rss
    - {cond}/frac_bursts_active (from fig4/compute.py)

Usage:
    python -m fig6.compute          (from the 200123_2953 directory)
    python fig6/compute.py          (same)
"""

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))

from spikelab.workspace.workspace import AnalysisWorkspace
from spikelab.workspace.hdf5_io import load_workspace_item
from spikelab.spikedata.utils import PCA_reduction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
WS_PATH = os.path.join(RESULTS_DIR, "workspace")
CONDITIONS = ["D0", "D3", "D10", "D30", "D50"]

PRE_MS = 250
POST_MS = 500
N_PCA_COMPONENTS = 5
U2U_MAX_LAG = 10

# Rank order parameters
MIN_OVERLAP = 20
N_SHUFFLES = 100
MIN_RATE_THRESHOLD = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 1. Burst-to-burst correlations
# ═══════════════════════════════════════════════════════════════════════════
def compute_burst_to_burst_corr(ws):
    """Compute burst-to-burst (slice-to-slice) correlations.

    Uses the combined RateSliceStack to compute pairwise correlations
    between all burst slices, averaged per unit and across units.

    Stores in ``all`` namespace:
        - ``burst_corr_per_unit``: PairwiseCompMatrixStack (S, S, U)
        - ``burst_corr_avg``: ndarray (S, S)

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("1. Burst-to-burst correlations")
    print("=" * 60)

    if ws.get("all", "burst_corr_per_unit") is not None:
        print("  Already cached, skipping\n")
        return

    rss = ws.get("all", "burst_rss")
    print(f"  Combined RateSliceStack shape: {rss.event_stack.shape} (U, T, S)")

    all_corr, av_corr = rss.get_slice_to_slice_unit_corr_from_stack()
    print(f"  Per-unit: {all_corr.stack.shape}, Average: {av_corr.shape}")

    ws.store("all", "burst_corr_per_unit", all_corr, note="Burst-to-burst corr per unit (S,S,U)")
    ws.store("all", "burst_corr_avg", av_corr, note="Burst-to-burst corr average (S,S)")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Unit-to-unit PCA
# ═══════════════════════════════════════════════════════════════════════════
def compute_burst_u2u_pca(ws):
    """Compute unit-to-unit correlations per burst, extract features, run PCA.

    Steps:
        1. Unit-to-unit correlation on combined burst RSS
        2. Extract lower-triangle features per slice
        3. PCA on the feature matrix
        4. Build condition label array for coloring

    Stores in ``all`` namespace:
        - ``burst_u2u_pca``: embedding (S, n_components)
        - ``burst_u2u_pca_variance``: variance ratio
        - ``burst_slice_condition_idx``: integer condition label per slice

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("2. Unit-to-unit PCA")
    print("=" * 60)

    existing = ws.list_keys("all")

    # Condition labels
    if "burst_slice_condition_idx" not in existing:
        slice_counts = []
        for rec in CONDITIONS:
            info = ws.get_info(rec, "burst_rss")
            n_slices = info["shape"][2]
            slice_counts.append(n_slices)
            print(f"  {rec}: {n_slices} slices")
        condition_labels = np.concatenate(
            [np.full(n, i, dtype=int) for i, n in enumerate(slice_counts)]
        )
        ws.store(
            "all", "burst_slice_condition_idx", condition_labels,
            note="Integer condition index per burst slice (0=D0..4=D50)",
        )

    if "burst_u2u_pca" in existing:
        print("  Already cached, skipping\n")
        return

    # Step 1: unit-to-unit correlations
    if "burst_u2u_corr" not in existing:
        rss = ws.get("all", "burst_rss")
        print(f"  Computing unit-to-unit correlations (max_lag={U2U_MAX_LAG})...")
        t0 = time.time()
        u2u_corr, u2u_lag, av_corr, av_lag = rss.unit_to_unit_correlation(max_lag=U2U_MAX_LAG)
        print(f"  Done in {time.time() - t0:.1f}s")
        ws.store("all", "burst_u2u_corr", u2u_corr, note="U2U corr per slice (U,U,S)")
        ws.store("all", "burst_u2u_lag", u2u_lag, note="U2U lag per slice (U,U,S)")
        ws.save(WS_PATH)
    else:
        u2u_corr = ws.get("all", "burst_u2u_corr")

    # Step 2: lower-triangle features
    if "burst_u2u_lower_tri" not in existing:
        print("  Extracting lower-triangle features...")
        features = u2u_corr.extract_lower_triangle_features()
        ws.store("all", "burst_u2u_lower_tri", features, note=f"Lower tri features {features.shape}")
        ws.save(WS_PATH)
    else:
        features = ws.get("all", "burst_u2u_lower_tri")

    # Step 3: PCA
    print(f"  Running PCA (n_components={N_PCA_COMPONENTS})...")
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"  Filling {nan_count} NaN values with 0")
        features = np.nan_to_num(features, nan=0.0)

    embedding, var_ratio, components = PCA_reduction(features, n_components=N_PCA_COMPONENTS)
    print(f"  Cumulative variance: {np.cumsum(var_ratio)}")

    ws.store("all", "burst_u2u_pca", embedding, note=f"PCA embedding ({N_PCA_COMPONENTS} comp)")
    ws.store("all", "burst_u2u_pca_variance", var_ratio, note="Explained variance ratio")
    ws.store("all", "burst_u2u_pca_components", components, note="PC loadings")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Slice-to-slice time correlations
# ═══════════════════════════════════════════════════════════════════════════
def compute_s2s_time_corr(ws):
    """Compute slice-to-slice time correlations per recording.

    For each recording, computes the temporal similarity between all pairs
    of burst-aligned slices at each time point.

    Stores per recording:
        - ``burst_s2s_time_corr``: PairwiseCompMatrixStack (S, S, T)
        - ``burst_s2s_time_corr_avg``: ndarray (T,)

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("3. Slice-to-slice time correlations")
    print("=" * 60)

    for rec in CONDITIONS:
        if ws.get(rec, "burst_s2s_time_corr") is not None:
            print(f"  {rec}: already cached, skipping")
            continue

        rss = ws.get(rec, "burst_rss")
        if rss is None:
            print(f"  {rec}: WARNING — burst_rss not found, skipping")
            continue

        print(f"  {rec}: computing...")
        t0 = time.time()
        s2s_corr, av_corr = rss.get_slice_to_slice_time_corr_from_stack()
        print(f"  {rec}: done in {time.time() - t0:.1f}s")

        ws.store(rec, "burst_s2s_time_corr", s2s_corr, note="S2S time corr (S,S,T)")
        ws.store(rec, "burst_s2s_time_corr_avg", av_corr, note="S2S time corr avg (T,)")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Burst rank order
# ═══════════════════════════════════════════════════════════════════════════
def compute_burst_rank_order(ws):
    """Compute rank-order correlations between burst pairs using rate peaks.

    Uses the combined RateSliceStack to compute per-unit firing rate peak
    timing, then correlates the rank orders across all burst pairs.

    Stores in ``all`` namespace:
        - ``burst_rank_rate_corr``: rank correlation matrix (S, S)
        - ``burst_rank_rate_overlap``: overlap fraction matrix (S, S)

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("4. Burst rank order correlations")
    print("=" * 60)

    existing = ws.list_keys("all")

    if "burst_rank_rate_corr" in existing:
        print("  Already cached, skipping\n")
        return

    rss = ws.get("all", "burst_rss")

    print("  Computing rate-peak timing matrix...")
    t0 = time.time()
    timing_rate = rss.get_unit_timing_per_slice(MIN_RATE_THRESHOLD=MIN_RATE_THRESHOLD)
    print(f"  Shape: {timing_rate.shape}, done in {time.time() - t0:.1f}s")

    ws.store("all", "burst_timing_rate", timing_rate, note="Rate-peak timing (U,S)")

    print(f"  Computing rank-order correlations (min_overlap={MIN_OVERLAP})...")
    t0 = time.time()
    corr, av_corr, overlap = rss.rank_order_correlation(
        timing_matrix=timing_rate,
        min_overlap=MIN_OVERLAP,
        n_shuffles=N_SHUFFLES,
    )
    print(f"  Done in {time.time() - t0:.1f}s, av_corr={av_corr:.4f}")

    ws.store(
        "all", "burst_rank_rate_corr", corr,
        note=f"Rank order corr (rate peaks, min_overlap={MIN_OVERLAP})",
    )
    ws.store("all", "burst_rank_rate_overlap", overlap, note="Rank order overlap fraction")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Unit burst groups
# ═══════════════════════════════════════════════════════════════════════════
def compute_unit_burst_groups(ws):
    """Classify units into burst groups and compute per-recording orderings.

    Group 0: units that fire >=2 spikes in 100% of bursts in at least
    one recording. Group 1: all remaining units.

    Within each group, units are ordered by median firing rate peak time
    relative to burst peak.

    Stores:
        - ``all/unit_burst_group_id``: per-unit group (0 or 1)
        - ``{cond}/burst_unit_order``: sorted unit indices per recording

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("5. Unit burst groups")
    print("=" * 60)

    if ws.get("all", "unit_burst_group_id") is not None:
        print("  Already cached, skipping\n")
        return

    sd_d0 = load_workspace_item(WS_PATH, "D0", "spikedata")
    n_units = sd_d0.N
    any_100pct = np.zeros(n_units, dtype=bool)

    for rec in CONDITIONS:
        frac = load_workspace_item(WS_PATH, rec, "frac_bursts_active")
        any_100pct |= frac >= 1.0

    group_id = np.where(any_100pct, 0, 1)
    group0 = np.where(group_id == 0)[0]
    group1 = np.where(group_id == 1)[0]
    print(f"  Group 0: {len(group0)} units, Group 1: {len(group1)} units")

    for rec in CONDITIONS:
        rss = load_workspace_item(WS_PATH, rec, "burst_rss")

        timing_matrix = rss.get_unit_timing_per_slice(MIN_RATE_THRESHOLD=0.0)
        peak_times = np.nanmedian(timing_matrix - PRE_MS, axis=1)

        g0_order = group0[np.argsort(peak_times[group0])]
        g1_order = group1[np.argsort(peak_times[group1])]
        unit_order = np.concatenate([g0_order, g1_order])

        if ws.get(rec, "burst_unit_order") is not None:
            ws.delete(rec, "burst_unit_order")
        ws.store(
            rec, "burst_unit_order", unit_order,
            note=f"Unit order: {len(g0_order)} group0 + {len(g1_order)} group1",
        )
        print(
            f"  {rec}: group0 peaks "
            f"{peak_times[g0_order[0]]:.0f}–{peak_times[g0_order[-1]]:.0f} ms"
        )

    ws.store(
        "all", "unit_burst_group_id", group_id,
        note="Per-unit group: 0=100% active in >=1 rec, 1=rest",
    )
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Run all Figure 6 computations."""
    print(f"Loading workspace from {WS_PATH}\n")
    ws = AnalysisWorkspace.load(WS_PATH)

    compute_burst_to_burst_corr(ws)
    compute_burst_u2u_pca(ws)
    compute_s2s_time_corr(ws)
    compute_burst_rank_order(ws)
    compute_unit_burst_groups(ws)

    ws.save(WS_PATH)
    print("=" * 60)
    print("Figure 6 computations complete. Workspace saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
