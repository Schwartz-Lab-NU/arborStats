# arborStats/runner.py
from __future__ import annotations
import sys
import pickle
import subprocess
from pathlib import Path
from typing import Iterable, Tuple
import multiprocessing as mp

from .core import load_swc, arborStatsFromSkeleton

class ArborRunError(RuntimeError):
    pass


# ---------------------------
# HELPER: detect existing outputs
# ---------------------------

def _seg_dir(root_output: Path, seg_id: int) -> Path:
    return Path(root_output) / str(seg_id)


def _flatone_exists(root_output: Path, seg_id: int) -> bool:
    """
    look for flatone outputs: mesh.obj + skeleton_warped.swc
    """
    d = _seg_dir(root_output, seg_id)
    if not d.exists():
        return False
    mesh, skel = d / "mesh.obj", d / "skeleton_warped.swc"
    if mesh.exists() and skel.exists():
        return True
    return False


def _arbor_stats_exists(root_output: Path, seg_id: int) -> bool:
    """
    look for an arbor stats file: arbor_stats.pkl   
    """
    d = _seg_dir(root_output, seg_id)
    if not d.exists():
        return False
    stats = d / "arbor_stats.pkl"
    if stats.exists():
        return True
    return False


def _find_swc_for_stats(root_output: Path, seg_id: int) -> Path | None:
    """
    Try to locate a skeleton SWC file for computing arbor stats. Prefer skeleton_warped.swc if present.
    Returns Path or None if not found.
    """
    d = _seg_dir(root_output, seg_id)
    if not d.exists():
        return None
    
    skel = d / "skeleton_warped.swc"
    if skel.exists():
        return skel
    
    return None


# ---------------------------
# TASK IMPLEMENTATIONS
# ---------------------------

def run_flattener(seg_id: int, root_output: Path, overwrite: bool = False) -> Path:
    """
    Run `flatone SEG_ID --output-dir ROOT`, return the segment's output folder.
    """
    seg_dir = root_output / str(seg_id)
    seg_dir.mkdir(parents=True, exist_ok=True)

    error_file = seg_dir / "flatone_error.txt"
    if error_file.exists():
        error_file.unlink()
    
    cmd = ["flatone", str(seg_id), "--output-dir", str(root_output)]
    if overwrite:
        cmd.append("--overwrite")

    # result = subprocess.run(cmd, capture_output=True, text=True)
    # # flatone prints to stderr/stdout; detect common 'no mesh' case:
    # if "No meshes found." in (result.stderr or ""):
    #     # write a tiny marker file for consistency with your script
    #     (seg_dir / "flatone_error.txt").write_text("No meshes found.\n")
    #     raise ArborRunError("No meshes found.")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )

    lines = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
    finally:
        if proc.stdout:
            proc.stdout.close()

    proc.wait()
    full_output = "".join(lines)

    # Preserve original behavior: detect 'no mesh' case and raise.
    if "No meshes found." in full_output:
        (seg_dir / "flatone_error.txt").write_text("No meshes found.\n")
        raise ArborRunError("No meshes found.")
    if "No CAVEclient token found." in full_output:
        (seg_dir / "flatone_error.txt").write_text("No CAVEclient token found.\n")
        raise ArborRunError("No CAVEclient token found. Please add token using the instructions above.")


    return seg_dir


def compute_arbor_stats_for_seg(seg_id: int, root_output: Path, overwrite: bool = False) -> Path:
    """
    Compute arbor stats from an SWC found in the segment's output directory.
    Writes Pickle (arbor_stats.pkl) file by default
    """
    segdir = _seg_dir(root_output, seg_id)
    segdir.mkdir(parents=True, exist_ok=True)

    error_file = segdir / "arbor_stats_error.txt"
    if error_file.exists():
        error_file.unlink()
    
    out_pkl = segdir / "arbor_stats.pkl"
    if out_pkl.exists() and not overwrite:
        return out_pkl

    swc_path = _find_swc_for_stats(root_output, seg_id)
    if swc_path is None:
        raise ArborRunError(f"No skeleton_warped.swc found for seg {seg_id}. Run flatone first or provide a skeleton.")

    coords, radii, edges = load_swc(str(swc_path))
    stats, units = arborStatsFromSkeleton(coords, edges, radii=radii)
    payload = {"segment_id": seg_id, "stats": stats, "units": units}

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_pkl


# ---------------------------
# DISPATCH + PARALLEL DRIVER
# ---------------------------

def _decide_tasks_for_seg(
    seg_id: int,
    root_output: Path,
    mode: str,
    overwrite: bool,
    new_only: bool,
) -> Tuple[bool, bool]:
    """
    Returns (need_flatone, need_arbor_stats) booleans for this seg_id.
    - mode: "both" | "flatone-only" | "arbor-only"
    - overwrite: force recompute
    - new_only: only compute when outputs are missing
    """
    want_flat   = mode in ("both", "flatone-only")
    want_arbor  = mode in ("both", "arbor-only")

    if overwrite:
        return (want_flat, want_arbor)

    # not overwriting: skip if existing (and if new_only is requested)
    if new_only:
        need_flat  = want_flat  and not _flatone_exists(root_output, seg_id)
        need_arbor = want_arbor and not _arbor_stats_exists(root_output, seg_id)
        return (need_flat, need_arbor)

    # default non-overwrite: we let the underlying tasks short-circuit if they see outputs
    return (want_flat, want_arbor)


def _one_worker(args: tuple) -> tuple:
    seg_id, root_output, mode, overwrite, new_only = args
    try:
        need_flat, need_arbor = _decide_tasks_for_seg(seg_id, root_output, mode, overwrite, new_only)
        
        # Run flatone first if requested (stats may depend on SWC emitted by flatone)
        if need_flat:
            run_flattener(seg_id, root_output, overwrite=overwrite)

        if need_arbor:
            compute_arbor_stats_for_seg(seg_id, root_output, overwrite=overwrite)

        return ("ok", seg_id)

    except ArborRunError as e:
        # Something expected but absent (e.g., no mesh / no SWC).
        print("Error : ", e)
        if str(e) == "No meshes found.":
            return ("no-mesh", seg_id, str(e))
        elif str(e).startswith("No CAVEclient token found."):
            return ("No-CAVEclient-token-found", seg_id, str(e))
    except Exception as e:
        return ("err", seg_id, f"{type(e).__name__}: {e}")


def process_many(
    seg_ids: Iterable[int],
    root_output: Path,
    overwrite: bool = False,
    jobs: int = 1,
    mode: str = "both",       # NEW: "both" | "flatone-only" | "arbor-only"
    new_only: bool = False,   # NEW: process only missing outputs when not overwriting
) -> None:
    """
    Backward-compatible entry point with two new keyword args:
      - mode: which tasks to run
      - new_only: only process segIDs missing required artifacts (when overwrite=False)
    """
    root_output = Path(root_output)
    root_output.mkdir(parents=True, exist_ok=True)

    for f in root_output.glob("*.txt"):
        if f.name == "not_processed_seg_ids.txt" or f.name == "arbor_stats_error_seg_ids.txt":
            f.unlink()

    # Prepare work items for the pool
    work = [(int(sid), root_output, mode, bool(overwrite), bool(new_only)) for sid in seg_ids]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs) as pool:
        for res in pool.imap_unordered(_one_worker, work):
            kind = res[0]
            
            if kind == "No-CAVEclient-token-found":
                _, sid, msg = res
                print(f"SegID {sid} skipped: {msg}", file=sys.stderr)
                break
            if kind == "ok":
                # success for this seg_id
                continue

            
            if kind == "no-mesh":
                _, sid, msg = res
                # Track segIDs that could not be processed due to missing mesh/SWC
                (root_output / "not_processed_seg_ids.txt").open("a").write(f"{sid}\n")
                # Optionally keep a per-seg note:
                (root_output / str(sid) / "arbor_stats_error.txt").write_text(msg)
                continue

            # kind == "err"
            _, sid, msg = res
            (root_output / str(sid) / "arbor_stats_error.txt").write_text(msg)
            (root_output / "arbor_stats_error_seg_ids.txt").open("a").write(f"{sid}\n")
