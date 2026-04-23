"""Figure 3 — compute all data needed by the panels.

Pipeline (run steps in order):
    1. load_and_split    — load raw data, split cue/no-cue, compute firing rates
    2. detect_bursts     — population rates + burst detection
    3. burst_sensitivity — burst count vs threshold sweep → .npz
    4. coupling_nocue    — sliding-window coupling on human no-cue → .npz
    5. coupling_full     — extend coupling to cue period → .npz
    6. coupling_shuffle  — z-score coupling against 100 shuffles → .npz

Prerequisites:
    - Raw data in manuscript/data/different_samples/{human,mouse,D0}_spikedata/
    - SpikeLab installed

Usage:
    python -m fig3.compute                  (run full pipeline)
    python -m fig3.compute --step 2         (run only step 2)

Output:
    manuscript/data/in_vivo/workspace.h5    — workspace with all recordings
    manuscript/data/in_vivo/*.npz           — precomputed results
"""

import os
import sys
import pickle
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))

from spikelab.workspace.workspace import AnalysisWorkspace
from spikelab.workspace.hdf5_io import load_workspace_item
from spikelab.spikedata.spikedata import SpikeData

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(REPO_ROOT, "manuscript", "data", "different_samples")
WS_PATH = os.path.join(DATA_DIR, "workspace")

NAMESPACES = ["human_cue", "human_nocue", "mouse_cue", "mouse_nocue", "D0"]

# ── Parameters ────────────────────────────────────────────────────────────
SPLIT_BUFFER_MS = 5000  # 5 s after last stimulus for cue/no-cue boundary
AUD_REGIONS = {"AUDv", "AUDd"}  # mouse auditory cortex subregions

# Burst detection
BURST_THRESHOLDS = {"human": 1.5, "mouse": 1.25, "D0": 2.5}
MIN_BURST_DIFF = 1000  # ms
EDGE_THRESH = 0.2
POPRATE_DISPLAY = {"square_width": 20, "gauss_sigma": 100, "raster_bin_size_ms": 1.0}
POPRATE_ACCURATE = {"square_width": 8, "gauss_sigma": 8, "raster_bin_size_ms": 1.0}

# Burst sensitivity sweep
SENSITIVITY_THRESHOLDS = np.arange(1.0, 5.05, 0.1)

# Sliding-window coupling
FRAME_LENGTH_MS = 120000  # 2 minutes
FRAME_STEP_MS = 10000     # 10 s step
FRAME_OVERLAP_MS = FRAME_LENGTH_MS - FRAME_STEP_MS
MIN_SPIKES = 12           # minimum spikes per unit per frame

# Shuffle coupling
N_SHUFFLES = 100
SHUFFLE_SEED = 42


def _get_burst_threshold(ns):
    if ns.startswith("human"):
        return BURST_THRESHOLDS["human"]
    elif ns.startswith("mouse"):
        return BURST_THRESHOLDS["mouse"]
    return BURST_THRESHOLDS["D0"]


def _compute_coupling(sd):
    """Zero-lag population coupling per unit. Returns (N,) array."""
    return sd.compute_spike_trig_pop_rate()[1]


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Load raw data, split cue/no-cue, compute firing rates
# ══════════════════════════════════════════════════════════════════════════

def step_load_and_split():
    print("=" * 60)
    print("STEP 1: Load and split")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    ws = AnalysisWorkspace(name="in_vivo_cue_nocue")

    # ── Human ──
    print("\nHUMAN")
    with open(os.path.join(DATA_DIR, "human_spikedata", "spikedata.pkl"), "rb") as f:
        sd_human = pickle.load(f)
    if not hasattr(sd_human, "start_time"):
        sd_human.start_time = 0.0

    cue_onsets = sd_human.metadata["all_cue_onset_times"]
    image_dur = sd_human.metadata["image_duration_ms"]
    cue_end = cue_onsets[-1] + image_dur + SPLIT_BUFFER_MS

    sd_human_cue = sd_human.subtime(0, cue_end)
    sd_human_nocue = sd_human.subtime(cue_end, sd_human.length)
    sd_human_cue.metadata = sd_human.metadata.copy()

    print(f"  {sd_human.N} units, split at {cue_end/1000:.1f} s")
    print(f"  Cue: {sd_human_cue.length/1000:.1f} s, No-cue: {sd_human_nocue.length/1000:.1f} s")

    ws.store("human_cue", "spikedata", sd_human_cue)
    ws.store("human_nocue", "spikedata", sd_human_nocue)

    # ── Mouse (IBL) ──
    print("\nMOUSE (IBL)")
    mouse_pkl = os.path.join(DATA_DIR, "mouse_spikedata", "spikedata.pkl")
    with open(mouse_pkl, "rb") as f:
        sd_mouse_full = pickle.load(f)

    aud_idx = [
        i for i, attr in enumerate(sd_mouse_full.neuron_attributes)
        if attr.get("region") in AUD_REGIONS
    ]
    sd_mouse = sd_mouse_full.subset(aud_idx)
    sd_mouse.metadata = sd_mouse_full.metadata.copy()

    stim_on = sd_mouse.metadata["stim_on_times"]
    cue_end_mouse = min(sd_mouse.length, np.nanmax(stim_on) + SPLIT_BUFFER_MS)

    sd_mouse_cue = sd_mouse.subtime(0, cue_end_mouse)
    sd_mouse_nocue = sd_mouse.subtime(cue_end_mouse, sd_mouse.length)
    sd_mouse_cue.metadata = sd_mouse.metadata.copy()

    print(f"  {sd_mouse.N} AUD units, split at {cue_end_mouse/1000:.1f} s")
    ws.store("mouse_cue", "spikedata", sd_mouse_cue)
    ws.store("mouse_nocue", "spikedata", sd_mouse_nocue)

    # ── Organoid D0 ──
    print("\nORGANOID D0")
    d0_pkl = os.path.join(DATA_DIR, "D0_spikedata", "spikedata.pkl")
    with open(d0_pkl, "rb") as f:
        sd_d0 = pickle.load(f)
    print(f"  {sd_d0.N} units, {sd_d0.length/1000:.1f} s")
    ws.store("D0", "spikedata", sd_d0)

    # ── Firing rates ──
    print("\nCOMPUTING FIRING RATES")
    for ns in NAMESPACES:
        sd = ws.get(ns, "spikedata")
        times = np.arange(0, sd.length, 1.0)
        fr = sd.resampled_isi(times)
        ws.store(ns, "fr_rates", fr)
        print(f"  {ns}: {fr.shape}")

    ws.save(WS_PATH)
    print(f"\nWorkspace saved to {WS_PATH}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Detect population bursts
# ══════════════════════════════════════════════════════════════════════════

def step_detect_bursts():
    print("=" * 60)
    print("STEP 2: Detect bursts")
    print("=" * 60)

    ws = AnalysisWorkspace.load(WS_PATH)

    for ns in NAMESPACES:
        sd = ws.get(ns, "spikedata")
        thr = _get_burst_threshold(ns)

        pop_rate = sd.get_pop_rate(**POPRATE_DISPLAY)
        pop_rate_acc = sd.get_pop_rate(**POPRATE_ACCURATE)
        ws.store(ns, "pop_rate", pop_rate)
        ws.store(ns, "pop_rate_acc", pop_rate_acc)

        tburst, edges, peak_amp = sd.get_bursts(
            thr_burst=thr,
            min_burst_diff=MIN_BURST_DIFF,
            burst_edge_mult_thresh=EDGE_THRESH,
            pop_rate=pop_rate,
            pop_rate_acc=pop_rate_acc,
        )
        ws.store(ns, "tburst", tburst)
        ws.store(ns, "burst_edges", edges)
        ws.store(ns, "burst_peak_amp", peak_amp)

        rate = len(tburst) / (sd.length / 1000 / 60)
        print(f"  {ns}: {len(tburst)} bursts ({rate:.1f}/min), thr={thr}")

    ws.save(WS_PATH)
    print("Workspace saved.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Burst sensitivity sweep
# ══════════════════════════════════════════════════════════════════════════

def step_burst_sensitivity():
    print("=" * 60)
    print("STEP 3: Burst sensitivity")
    print("=" * 60)

    burst_counts = {}
    durations_min = {}

    for ns in NAMESPACES:
        sd = load_workspace_item(WS_PATH, ns, "spikedata")
        duration_min = sd.length / 1000 / 60
        durations_min[ns] = duration_min
        counts = []
        for thr in SENSITIVITY_THRESHOLDS:
            tburst, _, _ = sd.get_bursts(
                thr_burst=thr,
                min_burst_diff=MIN_BURST_DIFF,
                burst_edge_mult_thresh=EDGE_THRESH,
            )
            counts.append(len(tburst))
        burst_counts[ns] = np.array(counts)
        print(f"  {ns}: {min(counts)}–{max(counts)} bursts ({duration_min:.1f} min)")

    ws = AnalysisWorkspace.load(WS_PATH)
    for ns in NAMESPACES:
        ws.store(ns, "burst_sensitivity", {
            "thresholds": SENSITIVITY_THRESHOLDS,
            "counts": burst_counts[ns],
            "duration_min": float(durations_min[ns]),
        })
    ws.save(WS_PATH)
    print("Saved to workspace: {ns}/burst_sensitivity for each recording")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Sliding-window coupling (human no-cue)
# ══════════════════════════════════════════════════════════════════════════

def step_coupling_nocue():
    print("=" * 60)
    print("STEP 4: Sliding-window coupling (no-cue)")
    print("=" * 60)

    sd = load_workspace_item(WS_PATH, "human_nocue", "spikedata")
    print(f"  {sd.N} units, {sd.length/1000:.1f} s")

    sss = sd.frames(length=FRAME_LENGTH_MS, overlap=FRAME_OVERLAP_MS)
    n_slices = len(sss.spike_stack)
    print(f"  {n_slices} frames")

    coupling_matrix = np.zeros((n_slices, sd.N))
    for i, sd_slice in enumerate(sss.spike_stack):
        coupling_matrix[i, :] = _compute_coupling(sd_slice)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i+1}/{n_slices}")

    time_centers_s = np.array([
        (i * FRAME_STEP_MS + FRAME_LENGTH_MS / 2) / 1000
        for i in range(n_slices)
    ])

    ws = AnalysisWorkspace.load(WS_PATH)
    ws.store("human_nocue", "coupling_over_time", {
        "coupling_matrix": coupling_matrix,
        "time_centers_s": time_centers_s,
    })
    ws.save(WS_PATH)
    print("Saved to workspace: human_nocue/coupling_over_time")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Extend coupling to cue period
# ══════════════════════════════════════════════════════════════════════════

def step_coupling_full():
    print("=" * 60)
    print("STEP 5: Full coupling (cue + no-cue)")
    print("=" * 60)

    # Load existing no-cue results from workspace
    nocue_data = load_workspace_item(WS_PATH, "human_nocue", "coupling_over_time")
    coupling_nocue = nocue_data["coupling_matrix"]
    time_nocue_rel = nocue_data["time_centers_s"]
    n_nocue, n_units = coupling_nocue.shape
    print(f"  Existing no-cue: {n_nocue} frames, {n_units} units")

    # Load recordings
    sd_cue = load_workspace_item(WS_PATH, "human_cue", "spikedata")
    sd_nocue = load_workspace_item(WS_PATH, "human_nocue", "spikedata")
    cue_end_ms = sd_cue.length

    # Alignment: clip no-cue start so frame boundaries align
    remainder_ms = cue_end_ms % FRAME_STEP_MS
    if remainder_ms > 0:
        sd_nocue_clipped = sd_nocue.subtime(remainder_ms, sd_nocue.length)
        print(f"  Clipping {remainder_ms:.0f} ms from no-cue start")
    else:
        sd_nocue_clipped = sd_nocue

    # Concatenate
    concat_trains = []
    for i in range(sd_cue.N):
        concat_trains.append(np.concatenate([
            sd_cue.train[i],
            sd_nocue_clipped.train[i] + cue_end_ms,
        ]))

    sd_full = SpikeData(
        concat_trains, N=sd_cue.N,
        length=cue_end_ms + sd_nocue_clipped.length,
        neuron_attributes=sd_cue.neuron_attributes,
    )
    print(f"  Concatenated: {sd_full.length/1000:.1f} s")

    # Create frames, identify which need computing
    sss = sd_full.frames(length=FRAME_LENGTH_MS, overlap=FRAME_OVERLAP_MS)
    n_total = len(sss.spike_stack)
    frame_centers_ms = np.array([
        FRAME_LENGTH_MS / 2 + i * FRAME_STEP_MS for i in range(n_total)
    ])
    time_nocue_full_ms = cue_end_ms + time_nocue_rel * 1000

    new_mask = np.ones(n_total, dtype=bool)
    for i in range(n_total):
        if np.any(np.abs(time_nocue_full_ms - frame_centers_ms[i]) < FRAME_STEP_MS / 2):
            new_mask[i] = False

    n_new = new_mask.sum()
    print(f"  {n_total} total frames, {n_new} new, {n_total - n_new} reused")

    # Compute coupling for new frames
    coupling_new = np.zeros((n_new, n_units))
    for j, idx in enumerate(np.where(new_mask)[0]):
        sd_slice = sss.spike_stack[idx]
        vals = _compute_coupling(sd_slice)
        for u in range(n_units):
            if len(sd_slice.train[u]) < MIN_SPIKES:
                vals[u] = np.nan
        coupling_new[j, :] = vals
        if (j + 1) % 10 == 0 or j == 0:
            print(f"  New frame {j+1}/{n_new}")

    # Spike counts for all frames
    spike_counts = np.array([
        sum(len(t) for t in sss.spike_stack[i].train) for i in range(n_total)
    ])

    # Combine
    coupling_full = np.zeros((n_total, n_units))
    coupling_full[new_mask] = coupling_new

    # Fill reused no-cue frames with spike count masking
    sss_nocue = sd_nocue.frames(length=FRAME_LENGTH_MS, overlap=FRAME_OVERLAP_MS)
    for k in range(n_nocue):
        abs_center_ms = time_nocue_full_ms[k]
        idx = np.argmin(np.abs(frame_centers_ms - abs_center_ms))
        vals = coupling_nocue[k].copy()
        sd_slice = sss_nocue.spike_stack[k]
        for u in range(n_units):
            if len(sd_slice.train[u]) < MIN_SPIKES:
                vals[u] = np.nan
        coupling_full[idx] = vals

    ws = AnalysisWorkspace.load(WS_PATH)
    ws.store("human_nocue", "coupling_over_time_full", {
        "coupling_matrix": coupling_full,
        "time_centers_s": frame_centers_ms / 1000,
        "cue_boundary_s": float(cue_end_ms / 1000),
        "spike_counts": spike_counts,
    })
    ws.save(WS_PATH)
    print("Saved to workspace: human_nocue/coupling_over_time_full")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Shuffle-based z-scored coupling (human no-cue)
# ══════════════════════════════════════════════════════════════════════════

def step_coupling_shuffle():
    print("=" * 60)
    print("STEP 6: Shuffle coupling")
    print("=" * 60)

    sd = load_workspace_item(WS_PATH, "human_nocue", "spikedata")
    print(f"  {sd.N} units, {sd.length/1000:.1f} s")

    coupling_original = _compute_coupling(sd)
    print(f"  Original range: {coupling_original.min():.4f} – {coupling_original.max():.4f}")

    sss = sd.spike_shuffle_stack(n_shuffles=N_SHUFFLES, seed=SHUFFLE_SEED)
    coupling_shuffled = np.zeros((N_SHUFFLES, sd.N))
    for i, sd_shuf in enumerate(sss.spike_stack):
        coupling_shuffled[i, :] = _compute_coupling(sd_shuf)
        if (i + 1) % 10 == 0:
            print(f"  Shuffle {i+1}/{N_SHUFFLES}")

    shuf_mean = np.mean(coupling_shuffled, axis=0)
    shuf_std = np.std(coupling_shuffled, axis=0)
    shuf_std[shuf_std == 0] = np.nan
    coupling_zscore = (coupling_original - shuf_mean) / shuf_std

    ws = AnalysisWorkspace.load(WS_PATH)
    ws.store("human_nocue", "coupling_shuffle", {
        "coupling_original": coupling_original,
        "coupling_zscore": coupling_zscore,
        "coupling_shuffled": coupling_shuffled,
    })
    ws.save(WS_PATH)
    print("Saved to workspace: human_nocue/coupling_shuffle")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

STEPS = {
    1: ("load_and_split", step_load_and_split),
    2: ("detect_bursts", step_detect_bursts),
    3: ("burst_sensitivity", step_burst_sensitivity),
    4: ("coupling_nocue", step_coupling_nocue),
    5: ("coupling_full", step_coupling_full),
    6: ("coupling_shuffle", step_coupling_shuffle),
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, help="Run a single step (1-6)")
    args = parser.parse_args()

    if args.step:
        name, func = STEPS[args.step]
        print(f"Running step {args.step}: {name}")
        func()
    else:
        for i, (name, func) in STEPS.items():
            func()
            print()
        print("All steps complete.")
