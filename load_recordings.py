"""Load diazepam dose-response recordings into an AnalysisWorkspace.

Dataset: 200123_2953 — Human brain organoid, MEA, 5 diazepam concentrations.
Each recording contains 177 units over ~180 s at 1 ms spike resolution.

Conditions:
    D0  — 0 µM diazepam (baseline)
    D3  — 3 µM diazepam
    D10 — 10 µM diazepam
    D30 — 30 µM diazepam
    D50 — 50 µM diazepam

Data source:
    Pre-computed SpikeData pickle files. Download ``sorted-recordings.zip``
    (Supplementary Data 3) from Zenodo: https://doi.org/10.5281/zenodo.22681642
    and unzip it into data/ (see data_paths.py).

Usage:
    python load_recordings.py

    This creates an AnalysisWorkspace at data/diazepam_casestudy/workspace
    containing one namespace per condition, each with a 'spikedata' key
    holding the SpikeData object.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from spikelab.workspace.workspace import AnalysisWorkspace

from data_paths import DIAZEPAM_DIR, load_recording

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = DIAZEPAM_DIR
CONDITIONS = ["D0", "D3", "D10", "D30", "D50"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_all_recordings(results_dir=RESULTS_DIR):
    """Load all SpikeData pickle files and store them in a workspace.

    Parameters
    ----------
    results_dir : str
        Directory where the workspace will be saved.

    Returns
    -------
    AnalysisWorkspace
        Workspace with one namespace per condition, each containing
        the SpikeData object at key ``"spikedata"``.
    """
    os.makedirs(results_dir, exist_ok=True)

    ws = AnalysisWorkspace(name="200123_2953_diazepam")

    for rec in CONDITIONS:
        sd = load_recording("organoid", rec)
        ws.store(rec, "spikedata", sd)
        print(f"{rec}: {sd.N} units, {sd.length:.0f} ms ({sd.length / 1000:.1f} s)")

    print()
    ws.describe()

    ws_path = os.path.join(results_dir, "workspace")
    ws.save(ws_path)
    print(f"\nWorkspace saved to {ws_path}")

    return ws


if __name__ == "__main__":
    load_all_recordings()
