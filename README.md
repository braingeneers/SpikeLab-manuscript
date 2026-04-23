# SpikeLab Manuscript — Figure Code

Code to reproduce all figures from the manuscript: *"SpikeLab: Agentic tools for spike data analysis"*.

## Figures

| Figure | Description | Script |
|---|---|---|
| Fig. 2 | Benchmark: plain models vs SpikeLab-assisted analysis | `fig2/` |
| Fig. 3 | Cross-species recordings: mouse, human, and organoid | `fig3/` |
| Fig. 4 | Neuronal firing and burst properties across diazepam doses | `fig4/` |
| Fig. 5 | Pairwise correlations and network structure | `fig5/` |
| Fig. 6 | Burst structure and reproducibility | `fig6/` |
| Fig. 7 | GPLVM latent state analysis of burst dynamics | `fig7/` |

## Structure

Each figure directory contains:

- **`compute.py`** — data processing and intermediate result computation
- **`panels.py`** — individual panel plotting functions
- **`assemble.py`** — multi-panel figure assembly

Shared utilities at the root:

- **`load_recordings.py`** — loads spike-sorted recordings into an AnalysisWorkspace
- **`compute_shared.py`** — shared computations used across multiple figures
- **`plot_config.py`** — matplotlib style configuration (eLife formatting, color palette, figure sizing)

## Prerequisites

- Python 3.10+
- [SpikeLab](https://github.com/braingeneers/SpikeLab) (`pip install -e SpikeLab/`)
- matplotlib, numpy, scipy, scikit-learn, Pillow
- Spike-sorted recording data (SpikeData `.pkl` files)

## Usage

Run from the repository root:

```bash
# Compute intermediate results for a figure
python -m fig3.compute

# Assemble the final figure
python -m fig3.assemble
```

Figures that depend on shared computations should run `compute_shared.py` and the figure-specific `compute.py` before assembly.

## SpikeLab Skill

The `spikelab-skill/` directory contains the distributable SpikeLab skill for agentic coding tools. Skill files are plain markdown documents that can be used with any agentic coding tool that supports custom instructions — simply provide the skill file as context. See `spikelab-skill/INSTALL.md` for library installation instructions.
