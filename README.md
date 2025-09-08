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
arborstats \
  --csv data/segids.csv \
  --read-columns "SegID,Status" \
  --dtypes "SegID=Int64,Status=string" \
  --segid-col "SegID" \
  --status-col "Status" --status-filter Complete \
  --output-dir ./out

