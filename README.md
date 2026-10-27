# SpikeLab Manuscript — Figure Code

Code to reproduce all figures from the manuscript: *"SpikeLab: Agentic tools for spike data analysis"*.

## Figures

| Figure | Description | Script |
|---|---|---|
| Fig. 2, Supp. Fig. 1 | Benchmark: nine models with and without SpikeLab across reasoning effort, plus a component ablation | `fig2/` |
| Fig. 3 | Cross-species recordings: mouse, human, and organoid | `fig3/` |
| Fig. 4 | Neuronal firing and burst properties across diazepam doses | `fig4/` |
| Fig. 5 | Pairwise correlations and network structure | `fig5/` |
| Fig. 6 | Burst structure and reproducibility | `fig6/` |
| Fig. 7 | GPLVM latent state analysis of burst dynamics | `fig7/` |

## Structure

Each figure directory from Fig. 3 onward contains (Fig. 2 is laid out differently, see [Figure 2](#figure-2) below):

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

## Figure 2

Figure 2 and Supplementary Figure 1 are built from the benchmark archive on Zenodo, not from recordings, and do not need SpikeLab installed.

1. Download `benchmark.zip` (Supplementary Data 2) from [doi.org/10.5281/zenodo.22681642](https://doi.org/10.5281/zenodo.22681642).
2. Install the requirements: `pip install numpy pandas scipy statsmodels matplotlib Pillow`
3. Run from `fig2/`:

```bash
# Unpack the per-run records into fig2/data/ and build the summary tables
python prepare_data.py path/to/benchmark.zip

# Figure 2 (add --tif for the submission TIFF)
python assemble_figure2.py

# Supplementary Figure 1, as a figure and as the captioned PDF
python assemble_figure_s1.py
python build_figure_s1_pdf.py
```

To keep the data somewhere else, set `BENCH_RUNS` to that folder before running any of these.

Layout of `fig2/`:

- **`prepare_data.py`**: unpacks the archive and runs the tools below
- **`tools/`**: the benchmark's scoring and statistics scripts. `stats_sweep.py` and `stats_ablation.py` fit the models and write the per-run tables; `grade_cell.py` scores rubric item 14 (method consistency within a cell); `plot_effort.py` writes the effort summary table
- **`benchmark_v2_data.py`**: loads the tables and run records for the panels
- **`plot_*.py`**: one module per panel; each also runs on its own and writes a preview to `fig2/panels/`
- **`assemble_figure2.py`**, **`assemble_figure_s1.py`**, **`build_figure_s1_pdf.py`**: figure assembly

## SpikeLab Skill

The `spikelab-skill/` directory contains the distributable SpikeLab skill for agentic coding tools. Skill files are plain markdown documents that can be used with any agentic coding tool that supports custom instructions — simply provide the skill file as context. See `spikelab-skill/INSTALL.md` for library installation instructions.
