# SpikeLab Installation Guide

## Clone the repository

```bash
git clone https://github.com/braingeneers/SpikeLab.git
```

## Install the library

**Option A — Conda environment (recommended for full functionality):**

The repository includes an `environment.yml` that installs Python, all core dependencies, and all optional packages (matplotlib, scikit-learn, umap-learn, networkx, neo, boto3, etc.):

```bash
conda env create -f SpikeLab/environment.yml
conda activate spikelab
pip install -e SpikeLab/
```

This is the recommended approach because `umap-learn` depends on `numba`/`llvmlite`, which install cleanly from conda-forge but often fail when built from source via pip.

**Option B — Pip only (core + selected extras):**

Install the core library (numpy, scipy, pandas, h5py, mcp):

```bash
pip install -e SpikeLab/
```

Install with optional dependency groups using extras:

```bash
# Core + S3 support (boto3)
pip install -e "SpikeLab/[s3]"

# Core + GPLVM latent state modelling (jax, jaxlib, jaxopt, optax, poor-man-gplvm)
pip install -e "SpikeLab/[gplvm]"

# Core + plotting (matplotlib)
pip install -e "SpikeLab/[plotting]"

# Core + ML (scikit-learn, umap-learn, networkx, python-louvain)
pip install -e "SpikeLab/[ml]"

# Core + Neo/NWB interop (neo, quantities, pynwb)
pip install -e "SpikeLab/[neo]"

# Core + numba-accelerated kernels
pip install -e "SpikeLab/[numba]"

# Core + dev tools (pytest, black)
pip install -e "SpikeLab/[dev]"

# Everything (all of the above)
pip install -e "SpikeLab/[all]"
```

Note: `umap-learn` depends on `numba`/`llvmlite`. If pip installation fails, install via conda instead: `conda install -c conda-forge umap-learn`.

## Updating SpikeLab

```bash
cd SpikeLab && git pull && cd ..
pip install -e SpikeLab/
```

Or if using the conda environment: `conda env update -f SpikeLab/environment.yml` followed by `pip install -e SpikeLab/`.

After pulling new code, regenerate the repo maps (see below).

## Generating the Repo Maps (required)

The repo maps do not ship with the repository — they must be generated after installation and after every library update. Without them, the analysis-implementer, educator, developer, and spike-sorter skills cannot orient themselves to the library API and will produce unreliable results.

Run the `spikelab-map-updater` skill to generate `REPO_MAP.md` and `REPO_MAP_DETAILED.md`:

```
SpikeLab/src/spikelab/agent/skills/spikelab-map-updater/SKILL.md
```

This reads the library source and produces two files in the same directory:
- `REPO_MAP.md` — condensed quick reference
- `REPO_MAP_DETAILED.md` — full API reference with signatures and return types

**When to regenerate:**
- After first installation
- After `git pull` with library changes
- After adding or modifying methods in `src/spikelab/`
- If any skill reports that the repo maps are missing or outdated
