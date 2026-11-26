# arborstats

- A command-line tool that computes arbor statistics for a list of EyeWire II neuron segment IDs.
- Per segment it downloads the mesh with [CaveClient/CloudVolume](https://github.com/seung-lab/cloud-volume), skeletonizes it with [skeliner](https://github.com/berenslab/skeliner) (>= 0.4.1, raw mesh, no mesh preprocessing), flattens it with [pywarper](https://github.com/berenslab/pywarper) (>= 0.3) and computes statistics with [MorphoPy](https://github.com/berenslab/MorphoPy) and its own features.
- Segment IDs can come from the CLI, a Google Sheet (CSV export) or a local CSV.

## Features

- One in-repo pipeline (`--stats-method morphopy`); the former `flatone` backend was removed in 0.2.
- Flexible input: `--segids`, Google Sheet (`--google-sheet-id`) or CSV (`--csv`)
- Schema from CLI: choose which columns to read and their dtypes (no hard-coding)
- Smart recompute: `--overwrite-all` or `--new-only`
- Parallelism: `-j/--jobs` for multiprocessing
- Export to SQLite (`--export-sqlite`, `--export-only`) and a compact Parquet file for the webapp (`arborstats-bulk-export`)
- Robust output layout with per-segment folders and error markers

## Installation

No system libraries are needed (skeliner ships binary wheels, the flattening needs no SuiteSparse), so it also runs on native Windows.

```bash
# clone this repo
git clone git@github.com:Schwartz-Lab-NU/arborStats.git
cd arborStats

# install conda environment with Python=3.13
conda create -n arborstats python=3.13
conda activate arborstats

# install
pip install -e .

# check if the installation worked
arborstats -h
```

A CAVE token is needed to download meshes. Without one, `arborstats` prints CAVEclient's instructions; save the token with
`python -c "from caveclient import CAVEclient; CAVEclient().auth.save_token(token='<TOKEN>')"`.

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
  --gid 0 \
  --read-columns "Status" "Final SegID" "Cell Class" \
  --dtypes "Final SegID=Int64" "Status=string" "Cell Class=string" \
  --segid-col "Final SegID" --cell-class-col "Cell Class" \
  --status-col "Status" --status-filter Complete "Complete (cut off)" \
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

- Export existing per-segment stats to SQLite without running processing:

```bash
arborstats \
  --csv data/segids.csv \
  --read-columns "SegID" \
  --dtypes "SegID=Int64" \
  --segid-col "SegID" \
  --output-dir ./out \
  --export-only \
  --export-file custom_stats.sqlite3
```

## CLI

```bash
usage: arborstats (--segids ... | --google-sheet-id ... | --csv CSV) --output-dir PATH [options]

Input source (choose exactly one) — mutually exclusive
  --segids SEGID [SEGID ...]   One or more segment IDs
  --google-sheet-id ID         Google Sheet ID to read (CSV export URL is inferred)
  --google-sheet-gid GID       Google Sheet tab gid to read; alias: --gid
  --csv PATH                   CSV path containing segment IDs

Schema controls
  --read-columns COL [COL ...]   Columns to read (space or comma separated)
  --dtypes COL=DTYPE [...]       Per-column dtypes (e.g., Final SegID=Int64, Status=string)

Column names & filters
  --segid-col NAME               Column containing segment IDs
  --status-col NAME              Status column name (default: "Status")
  --cell-review-col NAME         Cell-review column name
  --cell-class-col NAME          Column containing cell-class labels (default: "Cell Class")
  --cell-class NAME              Cell class label to use for every segment ID when no cell-class column is present
  --status-filter ...            Values to include from the status column
  --cell-review-filter ...       Values to include from the cell-review column (default: FALSE)

Common
  --output-dir PATH              Root output directory (per-seg subfolders are created)
  -j, --jobs N                   Parallel workers (default: 1)
  --stats-method {morphopy}      Only 'morphopy' remains; kept so existing command lines keep working
  --export-sqlite                Export per-segment stats to SQLite after processing
  --export-file PATH             SQLite filename/path to write during export; relative paths use --output-dir
  --full-export                  Check every segID's pickle during export (default after processing: only segIDs
                                  recomputed in this run or missing from the database)
  --run-label NAME               Prefix for this run's .txt marker files (e.g. "All Cells"); old files with the
                                  same label are deleted at the start of each run
  --marker-dir PATH              Where to write the .txt marker files (default: --output-dir)

Overwrite policy — mutually exclusive
  --overwrite-all                Force recompute even if outputs exist
  --new-only                     Only compute when expected outputs are missing

Which tasks to run — mutually exclusive
  --arbor-stats-only             Compute arbor stats (the default; kept for existing command lines)
  --export-only                  Skip processing and export existing arbor stats to SQLite
```

`--new-only` skips segments whose outputs exist, so cells processed before 0.2 keep their old
statistics (without `arbor_features`) until they are recomputed with `--overwrite-all`.

## Pipeline

Per segment (`morphopyStats.process_cell`):

1. download the mesh (CloudVolume),
2. skeletonize the raw mesh with skeliner (`postprocess=False`; no `sk.preprocess`),
3. detect the soma, merge soma nodes, bridge gaps and prune tiny neurites on the skeleton,
4. flatten with pywarper and the bundled global map (`arborstats/cached/global_map_j1_a16.npz`;
   maps of pywarper < 0.3 and the layered maps of `pywarper.layers.save_map` both load),
5. label the axon (cell-class dependent) and reroot cells without a plausible soma,
6. compute the statistics (`compute_stats`).

## Statistics

Each segment's `arbor_stats_morphopy.pkl` holds `{"segment_id", "stats", "units", "stats_method"}`;
`stats` contains:

| key | content |
| --- | --- |
| `morpho_stats` | MorphoPy's morphometric statistics (tips, branch points, lengths, angles, ...) |
| `morph_dists` | MorphoPy distributions (branch order, Strahler order, angles, thickness, segment and path length, radial distance) and the Sholl profile |
| `hull_stats` | convex hull of the dendrites (area, perimeter, equivalent diameter, points) |
| `z_dict` | depth profile (flat IPL units: ON SAC band = 0, OFF = 12) and its percentiles |
| `xy_dict` | xy density, a 2D Gaussian fit and its position relative to the soma |
| `arbor_features` | added in 0.2, see below |

### `arbor_features` (0.2)

Features the EyeWire II cell typing found informative (copied from eyewire2-datajoint). All are
scalars and are exported to the bulk Parquet file as `arbor_features.<key>`:

| keys | what | why |
| --- | --- | --- |
| `pruned_tips`, `pruned_branch_points`, `pruned_stems`, `pruned_total_length`, `pruned_max_branch_order`, `pruned_median_intermediate_segment_pl`, `pruned_median_terminal_segment_pl` | MorphoPy statistics after removing terminal branches ≤ 10 µm (two passes) | robust to segmentation spurs; unlike the raw counts they transfer between datasets |
| `on_hull_{perimeter,area,diameter}`, `off_hull_{...}` | convex hulls of the dendrites in the ON (z −12…5) and OFF (z 5…22) windows | mono- vs. bistratified, layer-specific field sizes |
| `wavy_band_rms_xy`, `wavy_band_rms_fine_xy`, `wavy_band_rms_z`, `wavy_decorr`, `wavy_excess_{mean,median,q95}`, `wavy_n_long_paths`, `wavy_frac_cable_long` | waviness of long smoothed dendrite paths (bends of ~6–30 µm wavelength) | separates medium/wide-field amacrine types; robust to node jitter, unlike path angles |
| `asym_c1`, `asym_c2`, `asym_bipolar`, `asym_c{1,2}_debiased`, `asym_lean_angle_deg`, `asym_axis_angle_deg`, `asym_n_eff`, `asym_total_length_um` | circular harmonics of the dendritic cable around the soma (one-sidedness, elongation and their directions in volume xy) | asymmetric and direction-selective types |
| `coverage_cut_below`, `coverage_cut_above`, `coverage_on_unsupported`, `coverage_off_unsupported` (+ `coverage_<layer>_unsupported`) | share of the cable where the imaged volume ends inside the IPL window, or where a SAC band was not observed | size features of cut-off cells are lower bounds; needs a pywarper >= 0.3 layered map, None with the bundled older map |

`feature_errors` lists a feature group that failed for a cell (the others are still computed).

Radii are skeliner's own (raw mesh, not calibrated), so radius-based statistics (`thickness`,
MorphoPy's radius stats) include organelles and are less reliable than the rest.

## Output layout

```bash
out/
├─ 720575940550176551/
│  ├─ mesh.obj                      # mesh of the EyeWire II neuron
│  ├─ skeleton_raw_morphopy.swc     # skeleton before flattening
│  ├─ skeleton_warped_morphopy.swc  # flattened skeleton
│  ├─ arbor_stats_morphopy.pkl      # computed statistics
│  ├─ arbor_stats_error.txt         # present only if an error occurred
├─ <label>_arbor_stats_error_seg_ids_<YYYY-MM-DD_HH-MM-SS>.txt  # segIDs which errored during arbor stats computation
├─ <label>_morphopy_failed_seg_ids_<YYYY-MM-DD_HH-MM-SS>.txt    # segIDs whose pipeline failed
├─ <label>_not_processed_seg_ids_<YYYY-MM-DD_HH-MM-SS>.txt      # segIDs skipped, e.g., no meshes found
```

The run-level `.txt` files go to `--marker-dir` instead when it is set.
