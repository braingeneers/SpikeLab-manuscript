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
    Pre-computed SpikeData pickle files. Download from Zenodo:
    <TODO: insert Zenodo DOI link>

Usage:
    python load_recordings.py

    This creates an AnalysisWorkspace at results/workspace containing one
    namespace per condition, each with a 'spikedata' key holding the
    SpikeData object.
"""

import os
import pickle
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SpikeLab", "src"))

from spikelab.workspace.workspace import AnalysisWorkspace

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
RESULTS_DIR = os.path.join(REPO_ROOT, "manuscript", "data", "diazepam_casestudy")
CONDITIONS = ["D0", "D3", "D10", "D30", "D50"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_all_recordings(data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    """Load all SpikeData pickle files and store them in a workspace.

    Parameters
    ----------
    data_dir : str
        Directory containing condition subdirectories (D0/, D3/, etc.),
        each with a ``spikedata.pkl`` file.
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
        pkl_path = os.path.join(data_dir, f"{rec}_spikedata", "spikedata.pkl")
        with open(pkl_path, "rb") as f:
            sd = pickle.load(f)
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
