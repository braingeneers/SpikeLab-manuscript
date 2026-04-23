"""Figure 5 — compute figure-specific analysis results.

Computes and stores pairwise correlation matrices that are only used by
Figure 5:
    1. Pairwise firing-rate correlations — cross-correlation of
       instantaneous firing rate traces between all unit pairs.
    2. Pairwise STTC — spike time tiling coefficient for all unit pairs.

Prerequisites (from compute_shared.py):
    - {cond}/spikedata
    - {cond}/fr_rates

Usage:
    python -m fig5.compute          (from the 200123_2953 directory)
    python fig5/compute.py          (same)
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

# FR correlation parameters
MAX_LAG = 350  # ms — maximum lag for cross-correlation

# STTC parameters
DELT = 20.0  # ms — coincidence window


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pairwise firing-rate correlations
# ═══════════════════════════════════════════════════════════════════════════
def compute_pairwise_fr_corr(ws):
    """Compute pairwise FR cross-correlations for each recording.

    Uses the pre-computed instantaneous firing rates (fr_rates) to build
    a RateData object and compute pairwise cross-correlations up to
    MAX_LAG ms.

    Stores per recording:
        - ``fr_corr_matrix``: PairwiseCompMatrix of peak correlations
        - ``fr_lag_matrix``: PairwiseCompMatrix of corresponding lags

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("1. Pairwise firing-rate correlations")
    print("=" * 60)

    for rec in CONDITIONS:
        existing = ws.list_keys(rec)
        if "fr_corr_matrix" in existing and "fr_lag_matrix" in existing:
            print(f"  {rec}: already cached, skipping")
            continue

        fr_rates = ws.get(rec, "fr_rates")  # (U, T)
        if fr_rates is None:
            print(f"  {rec}: WARNING — fr_rates not found, skipping")
            continue

        U, T = fr_rates.shape
        times = np.arange(T, dtype=np.float64)  # 1 ms bins

        print(f"  {rec}: building RateData ({U} units, {T} timepoints)...")
        rd = RateData(fr_rates, times)

        print(f"  {rec}: computing pairwise FR correlations (max_lag={MAX_LAG})...")
        t0 = time.time()
        corr_matrix, lag_matrix = rd.get_pairwise_fr_corr(max_lag=MAX_LAG)
        elapsed = time.time() - t0
        print(f"  {rec}: done in {elapsed:.1f}s")

        ws.store(
            rec,
            "fr_corr_matrix",
            corr_matrix,
            note=f"Pairwise FR correlation matrix (max_lag={MAX_LAG}ms)",
        )
        ws.store(
            rec,
            "fr_lag_matrix",
            lag_matrix,
            note=f"Pairwise FR lag matrix (max_lag={MAX_LAG}ms)",
        )

    print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pairwise STTC
# ═══════════════════════════════════════════════════════════════════════════
def compute_sttc(ws):
    """Compute pairwise spike time tiling coefficients for each recording.

    Stores per recording:
        - ``sttc_matrix``: PairwiseCompMatrix of STTC values

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("2. Pairwise STTC")
    print("=" * 60)

    for rec in CONDITIONS:
        existing = ws.list_keys(rec)
        if "sttc_matrix" in existing:
            print(f"  {rec}: already cached, skipping")
            continue

        sd = ws.get(rec, "spikedata")
        if sd is None:
            print(f"  {rec}: WARNING — spikedata not found, skipping")
            continue

        print(f"  {rec}: computing STTC (delt={DELT}ms, {sd.N} units)...")
        t0 = time.time()
        sttc_pcm = sd.spike_time_tilings(delt=DELT)
        elapsed = time.time() - t0
        print(f"  {rec}: done in {elapsed:.1f}s")

        ws.store(
            rec,
            "sttc_matrix",
            sttc_pcm,
            note=f"STTC pairwise matrix (delt={DELT}ms)",
        )

    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Run all Figure 5 computations."""
    print(f"Loading workspace from {WS_PATH}\n")
    ws = AnalysisWorkspace.load(WS_PATH)

    compute_pairwise_fr_corr(ws)
    compute_sttc(ws)

    ws.save(WS_PATH)
    print("=" * 60)
    print("Figure 5 computations complete. Workspace saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
