"""Where the figure code finds its data (Figures 3-7).

``DATA_DIR`` is ``$SPIKELAB_MANUSCRIPT_DATA`` if set, else ``data/`` in this
repository. Unzip ``sorted-recordings.zip`` from the Zenodo deposit
(https://doi.org/10.5281/zenodo.22681642, Supplementary Data 3) into it, so the
recordings sit in ``DATA_DIR/sorted_recordings/``. The compute steps write their
workspaces next to them.
"""

import os
import pickle

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SPIKELAB_MANUSCRIPT_DATA") or os.path.join(REPO_DIR, "data")

RECORDINGS_DIR = os.path.join(DATA_DIR, "sorted_recordings")
DIAZEPAM_DIR = os.path.join(DATA_DIR, "diazepam_casestudy")  # Figs. 4-7
SAMPLES_DIR = os.path.join(DATA_DIR, "different_samples")    # Fig. 3


class _Unpickler(pickle.Unpickler):
    # Four organoid recordings (D3, D10, D30, D50) were pickled when the package
    # was named ``SpikeLab``; resolve that name to ``spikelab``.
    def find_class(self, module, name):
        if module == "SpikeLab" or module.startswith("SpikeLab."):
            module = "spikelab" + module[len("SpikeLab"):]
        return super().find_class(module, name)


def load_recording(group, name):
    """Load ``sorted_recordings/<group>/<name>_spikedata.pkl`` as a SpikeData.

    ``group`` is ``"organoid"`` or ``"in_vivo"``; ``name`` is e.g. ``"D0"`` or
    ``"human"``.
    """
    path = os.path.join(RECORDINGS_DIR, group, f"{name}_spikedata.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Unzip sorted-recordings.zip from "
            f"https://doi.org/10.5281/zenodo.22681642 into {DATA_DIR}")
    with open(path, "rb") as f:
        return _Unpickler(f).load()
