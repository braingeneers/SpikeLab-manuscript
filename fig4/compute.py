"""Figure 4 — compute figure-specific analysis results.

Computes and stores per-unit metrics that are only used by Figure 4:
    1. Per-unit firing rate (Hz), ISI coefficient of variation, and
       population coupling (zero-lag and max).
    2. Per-unit burst participation: fraction of spikes inside bursts and
       fraction of bursts in which the unit fires >=2 spikes.
    3. Burst detection threshold sensitivity: burst count as a function
       of the RMS threshold parameter.

Prerequisites (from compute_shared.py):
    - {cond}/spikedata
    - {cond}/pop_rate_acc
    - {cond}/tburst, {cond}/burst_edges

Usage:
    python -m fig4.compute          (from the 200123_2953 directory)
    python fig4/compute.py          (same)
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))

from spikelab.workspace.workspace import AnalysisWorkspace

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
WS_PATH = os.path.join(RESULTS_DIR, "workspace")
CONDITIONS = ["D0", "D3", "D10", "D30", "D50"]

# ISI CV: only consider ISIs up to this value (ms)
ISI_FILTER_MS = 200

# Burst sensitivity: range of RMS multiplier thresholds to test
SENSITIVITY_THRESHOLDS = np.arange(1.0, 4.25, 0.25)
MIN_BURST_DIFF = 1000  # ms
BURST_EDGE_MULT_THRESH = 0.2


# ═══════════════════════════════════════════════════════════════════════════
# 1. Per-unit metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_unit_metrics(ws):
    """Compute per-unit firing rate, ISI CV, and population coupling.

    Stores per recording:
        - ``firing_rates_hz``: mean firing rate per unit (Hz), shape (N,)
        - ``isi_cv``: ISI coefficient of variation (ISI <= 200 ms), shape (N,)
        - ``pop_coupling_zero``: population coupling at zero lag, shape (N,)
        - ``pop_coupling_max``: population coupling at max lag, shape (N,)

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("1. Per-unit metrics (firing rate, ISI CV, population coupling)")
    print("=" * 60)

    for rec in CONDITIONS:
        if ws.get(rec, "firing_rates_hz") is not None:
            print(f"  {rec}: already cached, skipping")
            continue

        sd = ws.get(rec, "spikedata")

        # Firing rate (Hz)
        fr = sd.rates(unit="Hz")
        ws.store(rec, "firing_rates_hz", fr, note="Mean firing rate per unit (Hz)")

        # ISI CV — only ISIs up to ISI_FILTER_MS
        isis = sd.interspike_intervals()
        isi_cv = np.full(sd.N, np.nan)
        for i, isi in enumerate(isis):
            isi_filt = isi[isi <= ISI_FILTER_MS]
            if len(isi_filt) > 1:
                isi_cv[i] = np.std(isi_filt) / np.mean(isi_filt)
        ws.store(
            rec,
            "isi_cv",
            isi_cv,
            note=f"ISI CV per unit (ISI <= {ISI_FILTER_MS} ms only)",
        )

        # Population coupling
        _, coupling_zero, coupling_max, _, _ = sd.compute_spike_trig_pop_rate()
        ws.store(
            rec,
            "pop_coupling_zero",
            coupling_zero,
            note="Population coupling (zero-lag) per unit",
        )
        ws.store(
            rec,
            "pop_coupling_max",
            coupling_max,
            note="Population coupling (max) per unit",
        )

        print(
            f"  {rec}: FR mean={np.nanmean(fr):.2f} Hz, "
            f"ISI CV mean={np.nanmean(isi_cv):.3f}, "
            f"coupling mean={np.nanmean(coupling_zero):.4f}"
        )

    print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-unit burst participation
# ═══════════════════════════════════════════════════════════════════════════
def compute_burst_unit_metrics(ws):
    """Compute per-unit burst participation metrics.

    For each unit, computes:
        - What fraction of its spikes fall inside burst windows.
        - In what fraction of bursts the unit fires at least 2 spikes.

    Stores per recording:
        - ``frac_spikes_in_burst``: shape (N,)
        - ``frac_bursts_active``: shape (N,)

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("2. Per-unit burst participation")
    print("=" * 60)

    for rec in CONDITIONS:
        if ws.get(rec, "frac_spikes_in_burst") is not None:
            print(f"  {rec}: already cached, skipping")
            continue

        sd = ws.get(rec, "spikedata")
        burst_edges = ws.get(rec, "burst_edges")  # (B, 2) in raster bin coords

        # Fraction of bursts where each unit fires >=2 spikes
        frac_bursts_active, _, _ = sd.get_frac_active(
            burst_edges, MIN_SPIKES=2, backbone_threshold=1.0
        )

        # Fraction of each unit's total spikes inside burst windows
        frac_spikes_in_burst = sd.get_frac_spikes_in_burst(burst_edges)
        B = burst_edges.shape[0]

        ws.store(
            rec,
            "frac_spikes_in_burst",
            frac_spikes_in_burst,
            note="Per-unit fraction of total spikes inside burst windows",
        )
        ws.store(
            rec,
            "frac_bursts_active",
            frac_bursts_active,
            note="Per-unit fraction of bursts with >=2 spikes",
        )

        print(
            f"  {rec}: B={B}, "
            f"frac_spikes median={np.nanmedian(frac_spikes_in_burst):.3f}, "
            f"frac_bursts median={np.nanmedian(frac_bursts_active):.3f}"
        )

    print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Burst threshold sensitivity
# ═══════════════════════════════════════════════════════════════════════════
def compute_burst_sensitivity(ws):
    """Sweep burst detection thresholds and record burst counts.

    For each recording and each RMS multiplier in SENSITIVITY_THRESHOLDS,
    runs burst detection and records how many bursts are found.

    Stores in the ``all`` namespace:
        - ``burst_sensitivity_thresholds``: 1-D array of thresholds
        - ``burst_sensitivity_counts``: dict mapping condition to count array

    Parameters
    ----------
    ws : AnalysisWorkspace
    """
    print("=" * 60)
    print("3. Burst threshold sensitivity")
    print("=" * 60)

    if ws.get("all", "burst_sensitivity_thresholds") is not None:
        print("  Already cached, skipping\n")
        return

    counts = {}
    for rec in CONDITIONS:
        sd = ws.get(rec, "spikedata")
        pop_rate_acc = ws.get(rec, "pop_rate_acc")
        rec_counts = []

        for thr in SENSITIVITY_THRESHOLDS:
            tburst, _, _ = sd.get_bursts(
                thr_burst=thr,
                min_burst_diff=MIN_BURST_DIFF,
                burst_edge_mult_thresh=BURST_EDGE_MULT_THRESH,
                pop_rate=pop_rate_acc,
            )
            rec_counts.append(len(tburst))

        counts[rec] = np.array(rec_counts)
        print(
            f"  {rec}: {counts[rec][0]} bursts at thr={SENSITIVITY_THRESHOLDS[0]:.1f} "
            f"→ {counts[rec][-1]} at thr={SENSITIVITY_THRESHOLDS[-1]:.1f}"
        )

    ws.store(
        "all",
        "burst_sensitivity_thresholds",
        SENSITIVITY_THRESHOLDS,
        note=f"RMS multiplier thresholds ({len(SENSITIVITY_THRESHOLDS)} values, "
        f"{SENSITIVITY_THRESHOLDS[0]:.1f} to {SENSITIVITY_THRESHOLDS[-1]:.1f})",
    )
    ws.store(
        "all",
        "burst_sensitivity_counts",
        counts,
        note="Burst counts per threshold per condition",
    )
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Run all Figure 4 computations."""
    print(f"Loading workspace from {WS_PATH}\n")
    ws = AnalysisWorkspace.load(WS_PATH)

    compute_unit_metrics(ws)
    compute_burst_unit_metrics(ws)
    compute_burst_sensitivity(ws)

    ws.save(WS_PATH)
    print("=" * 60)
    print("Figure 4 computations complete. Workspace saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
