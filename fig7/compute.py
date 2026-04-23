"""Figure 7 — compute figure-specific analysis results.

Computes and stores GPLVM and PCA results used by Figure 7:
    1. Burst GPLVM — fit a GPLVM on burst-period firing rates across all
       recordings concatenated with silence gaps. Stores per-recording
       sliced results, combined result, bin boundaries, and condition
       indices. Requires optional JAX dependencies (jax, jaxlib, numpyro).
    2. Rate PCA — per-recording and combined PCA on instantaneous firing
       rate traces (3 components).

Prerequisites (from compute_shared.py):
    - {cond}/spikedata, {cond}/tburst
    - {cond}/fr_rates

Usage:
    python -m fig7.compute          (from the 200123_2953 directory)
    python fig7/compute.py          (same)
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
from spikelab.spikedata.ratedata import RateData

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
WS_PATH = os.path.join(RESULTS_DIR, "workspace")
CONDITIONS = ["D0", "D3", "D10", "D30", "D50"]

# GPLVM parameters
PRE_MS = 250
POST_MS = 500
SILENCE_MS = 100
BIN_SIZE_MS = 50  # GPLVM default

# PCA parameters
N_PCA_COMPONENTS = 3


# ===========================================================================
# Helper
# ===========================================================================
def _split_dict(d, start, end, T_total):
    """Split arrays in a dict along first axis where dim matches T_total."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T_total:
            out[k] = v[start:end]
        else:
            out[k] = v
    return out


# ===========================================================================
# 1. Burst GPLVM
# ===========================================================================
def compute_burst_gplvm(ws):
    """Fit GPLVM on burst periods across all recordings.

    Extracts fixed-size windows (PRE_MS/POST_MS) around every detected
    burst peak, concatenates them with SILENCE_MS gaps, fits a single
    GPLVM model, then slices results back per recording.

    NOTE: This requires optional JAX dependencies (jax, jaxlib, numpyro).
    If they are not installed the GPLVM fit will fail.

    Stores:
        - ``all/gplvm_burst_result``: full combined GPLVM result
        - ``all/gplvm_burst_boundaries_bins``: (n_total, 2) bin boundaries
        - ``all/gplvm_burst_condition_idx``: condition index per burst
        - ``{cond}/gplvm_burst_result``: per-recording GPLVM slice
        - ``{cond}/burst_sss``: SpikeSliceStack for burst windows

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("1. Burst GPLVM")
    print("=" * 60)

    if ws.get("all", "gplvm_burst_result") is not None:
        print("  Already cached, skipping\n")
        return

    # --- Step 1: Extract burst windows as SpikeData per recording ---
    all_burst_sds = []
    burst_counts_per_rec = {}

    for rec in CONDITIONS:
        sd = ws.get(rec, "spikedata")
        tburst = ws.get(rec, "tburst")
        n_bursts = len(tburst)

        sss = sd.align_to_events(tburst, pre_ms=PRE_MS, post_ms=POST_MS, kind="spike")

        # Store the SpikeSliceStack for later use (raster panels)
        if ws.get(rec, "burst_sss") is None:
            ws.store(
                rec,
                "burst_sss",
                sss,
                note=f"Burst SpikeSliceStack, {n_bursts} slices, pre={PRE_MS}ms post={POST_MS}ms",
            )

        for burst_sd in sss.spike_stack:
            all_burst_sds.append(burst_sd)
        burst_counts_per_rec[rec] = n_bursts
        print(f"  {rec}: {n_bursts} bursts")

    n_total = len(all_burst_sds)
    print(f"  Total bursts: {n_total}")

    # --- Step 2: Concatenate all bursts with silence gaps ---
    window_ms = PRE_MS + POST_MS
    combined = all_burst_sds[0]
    for i in range(1, n_total):
        combined = combined.append(all_burst_sds[i], offset=SILENCE_MS)

    total_length_ms = combined.length
    print(
        f"  Combined: {combined.N} units, {total_length_ms:.0f} ms "
        f"({total_length_ms / 1000:.1f} s)"
    )

    # --- Step 3: Compute bin boundaries per burst ---
    boundaries_ms = []
    for i in range(n_total):
        start_ms = i * (window_ms + SILENCE_MS)
        end_ms = start_ms + window_ms
        boundaries_ms.append((start_ms, end_ms))

    boundaries_bins = [
        (int(s / BIN_SIZE_MS), int(e / BIN_SIZE_MS)) for s, e in boundaries_ms
    ]

    # --- Step 4: Fit GPLVM ---
    print("  Fitting GPLVM on burst-only data...")
    t0 = time.time()
    gplvm_result = combined.fit_gplvm(bin_size_ms=BIN_SIZE_MS)
    elapsed = time.time() - t0
    print(f"  GPLVM fit complete in {elapsed:.1f}s")

    bin_size = gplvm_result["bin_size_ms"]
    T_total = gplvm_result["binned_spike_counts"].shape[0]

    # --- Step 5: Split results per recording and store ---
    decode_res = gplvm_result["decode_res"]
    reorder_indices = gplvm_result["reorder_indices"]
    binned_counts = gplvm_result["binned_spike_counts"]

    burst_idx = 0
    for rec in CONDITIONS:
        n_b = burst_counts_per_rec[rec]
        b_start = boundaries_bins[burst_idx][0]
        b_end = boundaries_bins[burst_idx + n_b - 1][1]

        rec_result = {
            "decode_res": _split_dict(decode_res, b_start, b_end, T_total),
            "reorder_indices": reorder_indices,
            "binned_spike_counts": binned_counts[b_start:b_end],
            "bin_size_ms": bin_size,
            "n_bursts": n_b,
        }
        ws.store(
            rec,
            "gplvm_burst_result",
            rec_result,
            note=f"Burst GPLVM result, bins [{b_start},{b_end}), {n_b} bursts",
        )
        print(f"  Stored gplvm_burst_result for {rec}: bins [{b_start},{b_end})")
        burst_idx += n_b

    # Store full combined result (exclude non-serialisable model)
    gplvm_to_store = {k: v for k, v in gplvm_result.items() if k != "model"}
    ws.store(
        "all",
        "gplvm_burst_result",
        gplvm_to_store,
        note="Full GPLVM result on burst-only concatenated data",
    )
    ws.store(
        "all",
        "gplvm_burst_boundaries_bins",
        np.array(boundaries_bins),
        note=f"Per-burst bin boundaries ({n_total}, 2) for splitting GPLVM result",
    )

    # Condition index per burst
    burst_condition_idx = np.array(
        [
            CONDITIONS.index(r)
            for r in CONDITIONS
            for _ in range(burst_counts_per_rec[r])
        ],
        dtype=int,
    )
    ws.store(
        "all",
        "gplvm_burst_condition_idx",
        burst_condition_idx,
        note="Condition index per burst (0=D0..4=D50), matches gplvm_burst_boundaries_bins",
    )
    print()


# ===========================================================================
# 2. Rate PCA
# ===========================================================================
def compute_rate_pca(ws):
    """Run PCA on instantaneous firing rate data: per-recording and combined.

    Computes a 3-component PCA embedding of firing rate traces for each
    recording individually, then a combined PCA across all recordings with
    per-recording slices stored for overlay plotting.

    Stores per recording:
        - ``rate_pca_embedding``: (T, 3) PCA embedding
        - ``rate_pca_variance``: explained variance ratio
        - ``rate_pca_components``: PC loadings
        - ``rate_pca_combined_embedding``: slice of the combined embedding

    Stores in ``all`` namespace:
        - ``rate_pca_combined_embedding``: full combined embedding
        - ``rate_pca_combined_variance``: combined explained variance
        - ``rate_pca_combined_components``: combined PC loadings
        - ``rate_pca_combined_boundaries``: cumulative time bin boundaries

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("2. Rate PCA")
    print("=" * 60)

    if ws.get("all", "rate_pca_combined_embedding") is not None:
        print("  Already cached, skipping\n")
        return

    # --- Per-recording PCA ---
    all_rates = []
    all_lengths = []

    for rec in CONDITIONS:
        fr = ws.get(rec, "fr_rates")  # (U, T)
        times = np.arange(fr.shape[1], dtype=float)
        rd = RateData(fr, times)

        embedding, var_ratio, components = rd.get_manifold(
            method="PCA",
            n_components=N_PCA_COMPONENTS,
        )

        ws.store(
            rec,
            "rate_pca_embedding",
            embedding,
            note=f"PCA embedding of firing rates ({N_PCA_COMPONENTS} components), shape (T, {N_PCA_COMPONENTS})",
        )
        ws.store(
            rec,
            "rate_pca_variance",
            var_ratio,
            note=f"PCA explained variance ratio ({N_PCA_COMPONENTS} components)",
        )
        ws.store(
            rec,
            "rate_pca_components",
            components,
            note=f"PCA components ({N_PCA_COMPONENTS}, U)",
        )

        print(
            f"  {rec}: T={fr.shape[1]}, var explained={var_ratio.sum():.3f} "
            f"({', '.join(f'{v:.3f}' for v in var_ratio)})"
        )

        all_rates.append(fr)
        all_lengths.append(fr.shape[1])

    # --- Combined PCA ---
    combined_fr = np.concatenate(all_rates, axis=1)  # (U, T_total)
    times_combined = np.arange(combined_fr.shape[1], dtype=float)
    rd_combined = RateData(combined_fr, times_combined)

    embedding_all, var_ratio_all, components_all = rd_combined.get_manifold(
        method="PCA",
        n_components=N_PCA_COMPONENTS,
    )

    # Split back per recording
    boundaries = np.cumsum([0] + all_lengths)
    for i, rec in enumerate(CONDITIONS):
        start = boundaries[i]
        end = boundaries[i + 1]
        ws.store(
            rec,
            "rate_pca_combined_embedding",
            embedding_all[start:end],
            note=f"Combined PCA embedding slice ({N_PCA_COMPONENTS} components)",
        )

    ws.store(
        "all",
        "rate_pca_combined_embedding",
        embedding_all,
        note=f"Combined PCA embedding ({N_PCA_COMPONENTS} comp), shape ({embedding_all.shape[0]}, {N_PCA_COMPONENTS})",
    )
    ws.store(
        "all",
        "rate_pca_combined_variance",
        var_ratio_all,
        note=f"Combined PCA explained variance ({N_PCA_COMPONENTS} components)",
    )
    ws.store(
        "all",
        "rate_pca_combined_components",
        components_all,
        note=f"Combined PCA components ({N_PCA_COMPONENTS}, U)",
    )
    ws.store(
        "all",
        "rate_pca_combined_boundaries",
        boundaries,
        note="Cumulative time bin boundaries for splitting combined PCA per recording",
    )

    print(
        f"\n  Combined: T={combined_fr.shape[1]}, var explained={var_ratio_all.sum():.3f} "
        f"({', '.join(f'{v:.3f}' for v in var_ratio_all)})"
    )
    print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    """Run all Figure 7 computations."""
    print(f"Loading workspace from {WS_PATH}\n")
    ws = AnalysisWorkspace.load(WS_PATH)

    compute_burst_gplvm(ws)
    compute_rate_pca(ws)

    ws.save(WS_PATH)
    print("=" * 60)
    print("Figure 7 computations complete. Workspace saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
