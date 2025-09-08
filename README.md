# arborstats

- A small command-line tool to run a flattening pipeline (e.g., flatone) and/or compute arbor statistics for a list of neuron segment IDs
- `flatone` automatically downloads the mesh of an EyeWire II neuron as `.obj` with [CaveClient/CloudVolume](https://github.com/seung-lab/cloud-volume), skeletonizes it as an `.swc` with [skeliner](https://github.com/berenslab/skeliner) and flattens it with [pywarper](https://github.com/berenslab/pywarper)
- Segment IDs can come from the CLI, a Google Sheet (CSV export), or a local CSV or a list of Segment IDs

## Features 
- Two-stage pipeline:
  1. Run external flattener (e.g., flatone) per segment
  2. Compute arbor statistics from emitted skeletons (SWC)
- Flexible input: --segids, Google Sheet (--google-sheet-id), or CSV (--csv)
- Schema from CLI: choose which columns to read and their dtypes (no hard-coding)
- Smart recompute: --overwrite-all or --new-only
- Mode control: run both, only flatone, or only arbor stats
- Parallelism: -j/--jobs for multiprocessing
- Robust output layout with per-segment folders and error markers

> __NOTE__
> 
> `flatone` relies on SuiteSparse, which does **NOT** run on native Windows. Use it on Unix-like enviroment or Windows Subsystem for Linux (WSL 2) instead.
>
> Instructions to install prerequisites for flatone
> ```bash
> # prerequisites
> ## mac
> brew update
> brew install suite-sparse
> 
> ## debian/ubuntu/WSL
> sudo apt-get update
> sudo apt-get install build-essential # if not already installed
> sudo apt-get install libsuitesparse-dev
>```


## Installation Instructions for ArborStats

> System prerequisites for `flatone` (e.g., SuiteSparse) still apply; follow the steps mentioned above.

```bash
# clone this repo
git clone git@github.com:Schwartz-Lab-NU/arborStats.git
cd arborStats

# install conda environment with Python=3.13
conda create -n arborstats python=3.13

# activate conda environment
conda activate arborstats

# installing dependencies using the following command
pip install -e .

# check if the installation worked
arborstats -h
```
## Quickstart 

- Compute for specific segment IDs, writing outputs under ./out:
```bash
arborstats \
  --segids 720575940550176551 720575940550176552 \
  --output-dir ./out -j 2
```

- Read IDs from a Google Sheet (public or you have access), with explicit schema:
```bash
arborstats \
  --google-sheet-id 1o4i53h92oyzsBc8jEWKmF8ZnfyXKXtFCTaYSecs8tBk \
  --read-columns "Status" "Final SegID" "Cell Requires Review" \
  --dtypes "Final SegID=Int64" "Status=string" "Cell Requires Review=string" \
  --segid-col "Final SegID" \
  --status-col "Status" --status-filter Complete "Complete (cut off)" \
  --cell-review-col "Cell Requires Review" --cell-review-filter FALSE \
  --output-dir ./out -j 8
```

- Read IDs from a CSV:
```bash
arborstats \
  --csv data/segids.csv \
  --read-columns "SegID,Status" \
  --dtypes "SegID=Int64,Status=string" \
  --segid-col "SegID" \
  --status-col "Status" --status-filter Complete \
  --output-dir ./out
```

## CLI 
```bash
usage: arborstats (--segids ... | --google-sheet-id ... | --csv CSV) --output-dir PATH [options]

Input source (choose exactly one) — mutually exclusive
  --segids SEGID [SEGID ...]   One or more segment IDs
  --google-sheet-id ID         Google Sheet ID to read (CSV export URL is inferred)
  --csv PATH                   CSV path containing segment IDs

Schema controls
  --read-columns COL [COL ...]   Columns to read (space or comma separated)
  --dtypes COL=DTYPE [...]       Per-column dtypes (e.g., Final SegID=Int64, Status=string)

Column names & filters
  --segid-col NAME               Column containing segment IDs (default: "Final SegID")
  --status-col NAME              Status column name (default: "Status")
  --cell-review-col NAME         Cell-review column name (default: "Cell Requires Review")
  --status-filter ...            Values to include from the status column (default: Complete, "Complete (cut off)")
  --cell-review-filter ...       Values to include from the cell-review column (default: FALSE)
  --csv-col NAME                 (deprecated) overrides --segid-col when using --csv

Common
  --output-dir PATH              Root output directory (per-seg subfolders are created)
  -j, --jobs N                   Parallel workers (default: 1)

Overwrite policy — mutually exclusive
  --overwrite-all                Force recompute even if outputs exist
  --new-only                     Only compute when expected outputs are missing

Which tasks to run — mutually exclusive
  --flatone-arbor-stats-both     Run flatone and compute arbor stats (default)
  --arbor-stats-only             Compute only arbor stats (expects existing SWC)
  --flatone-only                 Run flatone only (skip arbor stats)

```

## Output layout

```bash
out/
├─ 720575940550176551/
│  ├─ mesh.obj                      # required: mesh of an EyeWire II neuron
│  ├─ skeleton.swc                  # required: output from flatone
│  ├─ skeleton_warped.swc           # required: output from flatone
│  ├─ arbor_stats.pkl               # required: computed statistics
│  ├─ arbor_stats_error.txt         # present only if an error occurred
│  ├─ *                             # other files from flatone outputs (not so important)
├─ arbor_stats_error_seg_ids.txt    # segIDs which errored during arbor stats computation
├─ not_processed_seg_ids.txt        # segIDs skipped (e.g., no meshes found)
```
