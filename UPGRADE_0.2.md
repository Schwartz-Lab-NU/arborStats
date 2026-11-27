# Upgrading arborstats to 0.2

arborstats 0.2 moves the pipeline to skeliner 0.4.1 and pywarper 0.3.0, removes the `flatone`
backend and adds new per-cell features (`arbor_features`). The changes are on the branch
`skeliner-0.4`.

## What changes for you

- **Same command line.** The nightly cron job (`--new-only --arbor-stats-only --stats-method morphopy`)
  keeps working without edits. `--stats-method flatone`, `--flatone-only` and
  `--flatone-arbor-stats-both` no longer exist.
- **No columns removed.** Everything in `morpho_stats`, `morph_dists`, `hull_stats`, `z_dict` and
  `xy_dict` is still computed. New values are only added.
- **`morpho_stats` without the axon.** The MorphoPy statistics (`morpho_stats` and the new
  `pruned_*` values) are now computed on the dendrites only, like the hulls, density profiles and
  distributions. Before, they included the axon. For cells with an axon (mainly RGCs), `total_length`,
  `tips`, `branch_points`, `width`/`depth`/`height`, tortuosity, angles and radii change. Recompute
  all cells (step 4) so that old and new cells are comparable.
- **No system libraries needed.** flatone needed SuiteSparse; arborstats 0.2 does not.
- **Old cells keep their old stats.** `--new-only` skips cells that already have outputs, so they
  get the new features only after a recompute (step 4).

## 1. Update the code

```bash
cd arborStats
git fetch
git checkout skeliner-0.4
```

## 2. Reinstall

The dependencies changed (flatone, shapely and concave-hull are gone; skeliner, pywarper, MorphoPy,
CAVEclient and CloudVolume are now direct dependencies), so reinstall in the existing environment:

```bash
conda activate arborstats
pip install -e .
```

Check the versions:

```bash
python -c "import skeliner, pywarper, importlib.metadata as m; print(skeliner.__version__, pywarper.__version__, m.version('morphopy'), m.version('arborstats'))"
# expected: 0.4.1 0.3.0 0.7.6 0.2.0 (or newer)
```

Optionally uninstall the old backend: `pip uninstall flatone`.

## 3. Test on a few cells

Use a separate output folder so production outputs stay untouched:

```bash
arborstats --segids 720575940550176551 720575940548230344 \
  --output-dir ./test_out --overwrite-all -j 2 \
  --export-sqlite --export-file test.sqlite3
arborstats-bulk-export ./test_out/test.sqlite3
```

Each cell folder should contain `mesh.obj`, `skeleton_raw_morphopy.swc`,
`skeleton_warped_morphopy.swc` and `arbor_stats_morphopy.pkl`, and no `arbor_stats_error.txt`.
Inspect the new features:

```python
import pickle
stats = pickle.load(open("test_out/720575940550176551/arbor_stats_morphopy.pkl", "rb"))["stats"]
stats["arbor_features"]
```

`feature_errors` should be `None`. The `coverage_*` values are `None` until the new map is in
place (step 5).

## 4. Recompute existing cells

To give every cell the new features, run the usual job once with `--overwrite-all` instead of
`--new-only`. This recomputes every cell, so expect a run as long as a first full run. Add
`--full-export` so every cell's row in the SQLite database is updated, then rebuild the Parquet file:

```bash
arborstats <your usual input options> --output-dir "$OUTPUT_DIR" \
  --overwrite-all --export-sqlite --export-file "$SQLITE_FILE" --full-export -j <N>
arborstats-bulk-export "$SQLITE_FILE"
```

After that, the cron job continues with `--new-only` as before.

The skeletons change slightly with the new skeliner version, so existing statistics of
recomputed cells can shift a little as well.

## 5. New global map (later)

The coverage features need the new layered SAC map (pywarper 0.3 `layers`), which is built on the
Berens lab server. Once it is available:

1. replace `arborstats/cached/global_map_j1_a16.npz` with the new map file (keep the file name, or
   update `_MORPHOPY_GLOBAL_MAP_PATH` in `arborstats/runner.py`),
2. recompute all cells as in step 4, since the flattening (and so every depth-based statistic)
   changes with the map.

Old and new map formats both load.

## New values for the webapp

In the SQLite `stats_json` they are under `arbor_features`. In the bulk Parquet file they are columns
named `arbor_features.<key>` (float, null for cells computed before 0.2). The Parquet format version
is unchanged, since columns were only added.

| columns | meaning |
| --- | --- |
| `pruned_tips`, `pruned_branch_points`, `pruned_stems`, `pruned_total_length`, `pruned_max_branch_order`, `pruned_median_intermediate_segment_pl`, `pruned_median_terminal_segment_pl` | MorphoPy statistics after removing terminal branches shorter than 10 µm; more robust than the raw counts |
| `on_hull_perimeter`, `on_hull_area`, `on_hull_diameter`, `off_hull_*` | convex hull of the dendrites in the ON (depth −12 to 5) and OFF (5 to 22) layer |
| `wavy_band_rms_xy`, `wavy_band_rms_fine_xy`, `wavy_band_rms_z`, `wavy_decorr`, `wavy_excess_mean`, `wavy_excess_median`, `wavy_excess_q95`, `wavy_n_long_paths`, `wavy_frac_cable_long` | how wavy the dendrites are (bends of about 6–30 µm wavelength) |
| `asym_c1`, `asym_c2`, `asym_bipolar`, `asym_c1_debiased`, `asym_c2_debiased`, `asym_lean_angle_deg`, `asym_axis_angle_deg`, `asym_n_eff`, `asym_total_length_um` | asymmetry of the dendrites around the soma: one-sidedness (`c1`, direction `lean_angle_deg`) and elongation (`c2`, axis `axis_angle_deg`), angles in volume coordinates |
| `coverage_cut_below`, `coverage_cut_above`, `coverage_on_unsupported`, `coverage_off_unsupported` | share of the dendrites where the imaged volume ends or a SAC band was not observed; a high value means size features are lower bounds. `None` until step 5 |

## If something goes wrong

- **Errors per cell:** `<cell>/arbor_stats_error.txt` and the marker files
  `<label>_morphopy_failed_seg_ids_<timestamp>.txt` / `<label>_arbor_stats_error_seg_ids_<timestamp>.txt`.
- **No CAVE token:** arborstats prints CAVEclient's instructions; save the token with
  `python -c "from caveclient import CAVEclient; CAVEclient().auth.save_token(token='<TOKEN>')"`.
  (The old `flatone add-token` command is gone.)
- **Roll back:** `git checkout main && pip install -e .` restores 0.1 with flatone.
