"""Build the Figure 2 inputs from the Zenodo benchmark archive.

Download ``benchmark.zip`` from the manuscript's Zenodo deposit
(https://doi.org/10.5281/zenodo.22681642, Supplementary Data 2), then run:

    python prepare_data.py path/to/benchmark.zip

Two steps:

1. Unpack the per-run records the figure code reads (scorecard, run manifest,
   methods and tool calls, about 95 MB) plus ``prompts.json``. The archive
   stores every run flat under ``runs/<run_id>/``; ``runs_index.csv`` gives the
   batch each run came from, and the tools below expect that layout, so runs
   are written to ``<data>/<source_dir>/<run_id>/``.
2. Run the scoring and statistics tools in ``tools/``, which write the summary
   tables to ``<data>/figures/``: ``stats_sweep.csv``, ``stats_ablation.csv``,
   ``ablation_contrasts.csv``, ``cell_scores.json`` and
   ``fig_effort_total.csv``. Nothing is graded again: every per-run verdict is
   read from its ``scorecard.json``.

``<data>`` is ``$BENCH_RUNS`` if set, else ``fig2/data``. After this,
``assemble_figure2.py`` and ``assemble_figure_s1.py`` run as they are.
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(SCRIPT_DIR, "tools")
DATA_DIR = os.environ.get("BENCH_RUNS") or os.path.join(SCRIPT_DIR, "data")

ARCHIVE_ROOT = "benchmark_v2"
RUN_FILES = ("scorecard.json", "run_manifest.json", "methods.json", "toolcalls.json")

# Order matters only for readability of the log; each tool reads runs, not
# another tool's output.
TOOLS = (
    ["stats_sweep.py"],
    ["stats_ablation.py"],
    ["grade_cell.py", "--grid", "--write"],
    ["plot_effort.py", "--panel", "total"],
)


def unpack(archive, data_dir):
    """Extract the per-run records into ``<data_dir>/<source_dir>/<run_id>/``."""
    with zipfile.ZipFile(archive) as z:
        index = z.read(f"{ARCHIVE_ROOT}/runs_index.csv").decode("utf-8")
        source = {r["run_id"]: r["source_dir"] for r in csv.DictReader(io.StringIO(index))}
        n = 0
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) != 4 or parts[1] != "runs" or parts[3] not in RUN_FILES:
                continue
            run = parts[2]
            out = os.path.join(data_dir, source[run], run)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, parts[3]), "wb") as fh:
                fh.write(z.read(name))
            n += 1
        with open(os.path.join(data_dir, "prompts.json"), "wb") as fh:
            fh.write(z.read(f"{ARCHIVE_ROOT}/prompts.json"))
    print(f"unpacked {n} files for {len(source)} runs into {data_dir}")


def build_tables(data_dir):
    """Run each tool against ``data_dir``; they write to ``data_dir/figures``."""
    os.makedirs(os.path.join(data_dir, "figures"), exist_ok=True)
    env = dict(os.environ, BENCH_RUNS=data_dir)
    for args in TOOLS:
        print(f"\n--- {' '.join(args)}")
        subprocess.run([sys.executable, *args], cwd=TOOLS_DIR, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive", help="benchmark.zip from the Zenodo deposit")
    args = ap.parse_args()
    unpack(args.archive, DATA_DIR)
    build_tables(DATA_DIR)
    print(f"\nFigure 2 inputs are in {DATA_DIR}")


if __name__ == "__main__":
    main()
