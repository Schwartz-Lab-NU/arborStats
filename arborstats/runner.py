# arborStats/runner.py
from __future__ import annotations
import os
import json
import pickle
import subprocess
from pathlib import Path
from typing import Iterable
import multiprocessing as mp

from .core import load_swc, arborStatsFromSkeleton

class ArborRunError(RuntimeError):
    pass

def run_flattener(seg_id: int, root_output: Path, overwrite: bool = False) -> Path:
    """
    Run `flatone SEG_ID --output-dir ROOT`, return the segment's output folder.
    """
    seg_dir = root_output / str(seg_id)
    seg_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["flatone", str(seg_id), "--output-dir", str(root_output)]
    if overwrite:
        cmd.append("--overwrite")

    result = subprocess.run(cmd, capture_output=True, text=True)
    # flatone prints to stderr/stdout; detect common 'no mesh' case:
    if "No meshes found." in (result.stderr or ""):
        # write a tiny marker file for consistency with your script
        (seg_dir / "error_msg.txt").write_text("No meshes found.\n")
        raise ArborRunError("No meshes found.")

    return seg_dir

def compute_and_save_stats(seg_id: int, seg_dir: Path) -> dict:
    """
    Load skeleton_warped.swc (preferred), else skeleton.swc; compute stats; save .pkl .
    Returns the stats dict (with 'units' as a sibling).
    """
    swc = (seg_dir / "skeleton_warped.swc")
    if not swc.exists():
        swc = (seg_dir / "skeleton.swc")
    if not swc.exists():
        raise FileNotFoundError(f"SWC not found for {seg_id} in {seg_dir}")

    coords, radii, edges = load_swc(str(swc))
    stats, units = arborStatsFromSkeleton(coords, edges, radii=radii)

    out_pkl = seg_dir / "arbor_stats.pkl"
    payload = {"segment_id": seg_id, "stats": stats, "units": units}
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload

def _one_worker(args):
    """Top-level worker so it can be pickled with 'spawn' start method."""
    sid, root_output_str, overwrite = args
    root_output = Path(root_output_str)  # paths are picklable, but str is safest
    try:
        seg_dir = run_flattener(sid, root_output, overwrite=overwrite)
        compute_and_save_stats(sid, seg_dir)
        return ("ok", sid)
    except ArborRunError:
        return ("no-mesh", sid)
    except Exception as e:
        return ("error", sid, str(e))

def process_many(seg_ids, root_output: Path, overwrite: bool, jobs: int = 1) -> None:
    """
    Run the full pipeline for many segments. Parallelism is optional.
    """
    seg_ids = list(seg_ids)
    root_output = Path(root_output)

    if jobs <= 1:
        for sid in seg_ids:
            try:
                seg_dir = run_flattener(sid, root_output, overwrite=overwrite)
                compute_and_save_stats(sid, seg_dir)
            except ArborRunError:
                (root_output / "not_processed_seg_ids.txt").open("a").write(f"{sid}\n")
            except Exception as e:
                (root_output / str(sid) / "arbor_stats_error.txt").write_text(str(e))
                (root_output / "arbor_stats_error_seg_ids.txt").open("a").write(f"{sid}\n")
        return

    # Use an explicit context for cross-platform consistency
    ctx = mp.get_context("spawn")  # good on macOS/Windows; fine on Linux too
    work = ((sid, str(root_output), overwrite) for sid in seg_ids)

    with ctx.Pool(processes=jobs) as pool:
        for res in pool.imap_unordered(_one_worker, work):
            kind = res[0]
            if kind == "ok":
                continue
            if kind == "no-mesh":
                _, sid = res
                (root_output / "not_processed_seg_ids.txt").open("a").write(f"{sid}\n")
            else:
                _, sid, msg = res
                (root_output / str(sid) / "arbor_stats_error.txt").write_text(msg)
                (root_output / "arbor_stats_error_seg_ids.txt").open("a").write(f"{sid}\n")

