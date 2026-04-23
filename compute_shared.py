"""Shared computations used by multiple figures.

This script computes and stores results that are prerequisites for two or
more figure scripts.  It should be run after ``load_recordings.py`` and
before any ``fig*.py`` script.

Computations (in order):
    1. Population rates — smoothed population firing rates at 1 ms resolution
       (two variants: display-smoothed and accurate).
       Used by: fig4, fig6, fig7.

    2. Burst detection — network burst times, edges, and peak amplitudes via
       threshold-crossing on the population rate.
       Used by: fig4, fig6, fig7.

    3. Burst-aligned RateSliceStacks — event-aligned firing rate matrices
       centered on each burst peak, per recording and combined across all
       recordings.
       Used by: fig4, fig6, fig7.

    4. Instantaneous firing rates — per-unit firing rate traces at 1 ms
       resolution via resampled ISI.
       Used by: fig5, fig6, fig7.

    5. Burst-aligned SpikeSliceStacks — event-aligned spike data centered
       on each burst peak, per recording and combined.
       Used by: fig6, fig7.

All results are stored in the AnalysisWorkspace and skipped if already present.

Usage:
    python compute_shared.py
"""

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))

from spikelab.workspace.workspace import AnalysisWorkspace
from spikelab.spikedata.spikeslicestack import SpikeSliceStack

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
WS_PATH = os.path.join(RESULTS_DIR, "workspace")
CONDITIONS = ["D0", "D3", "D10", "D30", "D50"]

# Population rate parameters
POPRATE_DISPLAY = {"square_width": 20, "gauss_sigma": 100, "raster_bin_size_ms": 1.0}
POPRATE_ACCURATE = {"square_width": 10, "gauss_sigma": 10, "raster_bin_size_ms": 1.0}

# Burst detection parameters
THR_BURST = 2.5  # RMS multiplier for peak height threshold
MIN_BURST_DIFF = 1000  # minimum distance between consecutive peaks (ms)
BURST_EDGE_MULT_THRESH = 0.2  # edge threshold as fraction of peak amplitude

# Event alignment parameters
PRE_MS = 250  # time before burst peak (ms)
POST_MS = 500  # time after burst peak (ms)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Population rates
# ═══════════════════════════════════════════════════════════════════════════
def compute_population_rates(ws):
    """Compute smoothed population rates for each recording.

    Stores two variants per recording:
        - ``pop_rate``: display-smoothed (sq=20, gauss=100) — for overview plots
        - ``pop_rate_acc``: accurate (sq=10, gauss=10) — for burst detection and
          time-resolved analyses

    Parameters
    ----------
    ws : AnalysisWorkspace
        Workspace containing SpikeData objects at ``{cond}/spikedata``.
    """
    print("=" * 60)
    print("1. Population rates")
    print("=" * 60)

    for rec in CONDITIONS:
        sd = ws.get(rec, "spikedata")

        if ws.get(rec, "pop_rate") is None:
            pop_rate = sd.get_pop_rate(**POPRATE_DISPLAY)
            ws.store(
                rec,
                "pop_rate",
                pop_rate,
                note=f"Pop rate display (sq={POPRATE_DISPLAY['square_width']}, "
                f"gauss={POPRATE_DISPLAY['gauss_sigma']}, bin=1ms)",
            )
            print(f"  {rec}: pop_rate shape={pop_rate.shape}")
        else:
            print(f"  {rec}: pop_rate already cached, skipping")

        if ws.get(rec, "pop_rate_acc") is None:
            pop_rate_acc = sd.get_pop_rate(**POPRATE_ACCURATE)
            ws.store(
                rec,
                "pop_rate_acc",
                pop_rate_acc,
                note=f"Pop rate accurate (sq={POPRATE_ACCURATE['square_width']}, "
                f"gauss={POPRATE_ACCURATE['gauss_sigma']}, bin=1ms)",
            )
            print(f"  {rec}: pop_rate_acc shape={pop_rate_acc.shape}")
        else:
            print(f"  {rec}: pop_rate_acc already cached, skipping")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Burst detection
# ═══════════════════════════════════════════════════════════════════════════
def compute_burst_detection(ws):
    """Detect network bursts in each recording using the accurate population rate.

    Stores per recording:
        - ``tburst``: burst peak times (ms)
        - ``burst_edges``: burst start/end times, shape (B, 2)
        - ``burst_peak_amp``: burst peak amplitudes

    Parameters
    ----------
    ws : AnalysisWorkspace
        Workspace with ``pop_rate_acc`` already computed.
    """
    print("=" * 60)
    print("2. Burst detection")
    print("=" * 60)

    for rec in CONDITIONS:
        if ws.get(rec, "tburst") is not None:
            print(f"  {rec}: burst data already cached, skipping")
            continue

        sd = ws.get(rec, "spikedata")
        pop_rate_acc = ws.get(rec, "pop_rate_acc")

        tburst, edges, peak_amp = sd.get_bursts(
            thr_burst=THR_BURST,
            min_burst_diff=MIN_BURST_DIFF,
            burst_edge_mult_thresh=BURST_EDGE_MULT_THRESH,
            pop_rate=pop_rate_acc,
        )

        ws.store(rec, "tburst", tburst, note=f"Burst peak times (ms), n={len(tburst)}")
        ws.store(
            rec,
            "burst_edges",
            edges,
            note=f"Burst start/end (ms), shape ({edges.shape[0]}, 2)",
        )
        ws.store(rec, "burst_peak_amp", peak_amp, note="Burst peak amplitudes")
        print(f"  {rec}: {len(tburst)} bursts detected")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Burst-aligned RateSliceStacks
# ═══════════════════════════════════════════════════════════════════════════
def compute_burst_rate_stacks(ws):
    """Create burst-aligned RateSliceStacks for each recording and combined.

    Centers a [-PRE_MS, +POST_MS] window around each burst peak and computes
    the instantaneous firing rate matrix.

    Stores:
        - ``{cond}/burst_rss``: per-recording RateSliceStack
        - ``all/burst_rss``: combined RateSliceStack across all recordings

    Parameters
    ----------
    ws : AnalysisWorkspace
        Workspace with burst times already computed.
    """
    print("=" * 60)
    print("3. Burst-aligned RateSliceStacks")
    print("=" * 60)

    all_rss = []

    for rec in CONDITIONS:
        if ws.get(rec, "burst_rss") is not None:
            print(f"  {rec}: burst_rss already cached, loading for combination")
            all_rss.append(ws.get(rec, "burst_rss"))
            continue

        sd = ws.get(rec, "spikedata")
        tburst = ws.get(rec, "tburst")

        rss = sd.align_to_events(tburst, pre_ms=PRE_MS, post_ms=POST_MS, kind="rate")

        ws.store(
            rec,
            "burst_rss",
            rss,
            note=f"Burst-aligned RateSliceStack ({PRE_MS}/{POST_MS} ms), "
            f"{len(tburst)} bursts, shape {rss.event_stack.shape}",
        )
        all_rss.append(rss)
        print(f"  {rec}: burst_rss shape={rss.event_stack.shape}")

    # Combined stack across all recordings
    if ws.get("all", "burst_rss") is None:
        combined_stack = np.concatenate(
            [rss.event_stack for rss in all_rss], axis=2
        )
        from spikelab.spikedata.rateslicestack import RateSliceStack

        combined_rss = RateSliceStack(combined_stack, pre_ms=PRE_MS, post_ms=POST_MS)
        ws.store(
            "all",
            "burst_rss",
            combined_rss,
            note=f"Combined burst RateSliceStack, shape {combined_stack.shape}",
        )
        print(f"  Combined: shape={combined_stack.shape}")
    else:
        print("  Combined: already cached, skipping")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Instantaneous firing rates
# ═══════════════════════════════════════════════════════════════════════════
def compute_firing_rates(ws):
    """Compute per-unit instantaneous firing rates at 1 ms resolution.

    Uses resampled ISI interpolation to produce a (U, T) firing rate matrix
    for each recording, where U is the number of units and T is the number
    of 1 ms time bins.

    Stores:
        - ``{cond}/fr_rates``: ndarray of shape (U, T)

    Parameters
    ----------
    ws : AnalysisWorkspace
        Workspace with SpikeData objects.
    """
    print("=" * 60)
    print("4. Instantaneous firing rates")
    print("=" * 60)

    for rec in CONDITIONS:
        if ws.get(rec, "fr_rates") is not None:
            print(f"  {rec}: fr_rates already cached, skipping")
            continue

        sd = ws.get(rec, "spikedata")
        times = np.arange(0, sd.length, 1.0)  # 1 ms resolution
        fr = sd.resampled_isi(times)  # shape (U, T)

        ws.store(
            rec,
            "fr_rates",
            fr,
            note=f"Instantaneous firing rates (U={fr.shape[0]}, T={fr.shape[1]}) "
            "at 1 ms, resampled ISI",
        )
        print(f"  {rec}: fr_rates shape={fr.shape}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Burst-aligned SpikeSliceStacks
# ═══════════════════════════════════════════════════════════════════════════
def compute_burst_spike_stacks(ws):
    """Create burst-aligned SpikeSliceStacks for each recording and combined.

    Centers a [-PRE_MS, +POST_MS] window around each burst peak and extracts
    the raw spike times as a SpikeSliceStack.

    Stores:
        - ``{cond}/burst_sss``: per-recording SpikeSliceStack
        - ``all/burst_sss``: combined SpikeSliceStack across all recordings

    Parameters
    ----------
    ws : AnalysisWorkspace
        Workspace with burst times already computed.
    """
    print("=" * 60)
    print("5. Burst-aligned SpikeSliceStacks")
    print("=" * 60)

    all_burst_sds = []

    for rec in CONDITIONS:
        sd = ws.get(rec, "spikedata")
        tburst = ws.get(rec, "tburst")
        n_bursts = len(tburst)

        if ws.get(rec, "burst_sss") is not None:
            print(f"  {rec}: burst_sss already cached, loading for combination")
            sss = ws.get(rec, "burst_sss")
            all_burst_sds.extend(sss.spike_stack)
            continue

        sss = sd.align_to_events(tburst, pre_ms=PRE_MS, post_ms=POST_MS, kind="spike")

        ws.store(
            rec,
            "burst_sss",
            sss,
            note=f"Burst-aligned SpikeSliceStack ({PRE_MS}/{POST_MS} ms), "
            f"{n_bursts} bursts",
        )
        all_burst_sds.extend(sss.spike_stack)
        print(f"  {rec}: {n_bursts} bursts")

    # Combined SpikeSliceStack
    if ws.get("all", "burst_sss") is None:
        n_total = len(all_burst_sds)
        all_times = [(0.0, float(PRE_MS + POST_MS))] * n_total
        all_sss = SpikeSliceStack(
            spike_stack=all_burst_sds,
            times_start_to_end=all_times,
            neuron_attributes=ws.get("D0", "spikedata").neuron_attributes,
        )
        ws.store(
            "all",
            "burst_sss",
            all_sss,
            note=f"Combined burst SpikeSliceStack ({n_total} bursts from all recordings)",
        )
        print(f"  Combined: {n_total} bursts")
    else:
        print("  Combined: already cached, skipping")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Run all shared computations in dependency order."""
    print(f"Loading workspace from {WS_PATH}\n")
    ws = AnalysisWorkspace.load(WS_PATH)

    compute_population_rates(ws)
    compute_burst_detection(ws)
    compute_burst_rate_stacks(ws)
    compute_firing_rates(ws)
    compute_burst_spike_stacks(ws)

    ws.save(WS_PATH)
    print("=" * 60)
    print("All shared computations complete. Workspace saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
