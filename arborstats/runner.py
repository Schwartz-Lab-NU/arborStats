# arborStats/runner.py
from __future__ import annotations
import sys
import pickle
import subprocess
from pathlib import Path
from typing import Any, Iterable, Tuple
import multiprocessing as mp

import numpy as np

from .core import load_swc, arborStatsFromSkeleton

class ArborRunError(RuntimeError):
    """
    Exception raised for anticipated runtime issues when orchestrating arbor stats.
    Optional `code` attribute enables callers to branch on specific failure kinds.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


_MORPHOPY_HINT = (
    "Install the Morphopy extras (e.g., pip install morphopy skeliner cloud-volume pywarper trimesh)."
)
_MORPHOPY_GLOBAL_MAP_PATH = Path(__file__).resolve().parent / "cached" / "global_map_j1_a16.npz"
_MORPHOPY_FUNCS: dict[str, Any] | None = None
_MORPHOPY_GLOBAL_MAP: dict | None = None
_MORPHOPY_CV = None

MORPHOPY_MESH_NAME = "mesh.obj"
MORPHOPY_SWC_NAME = "skeleton_warped_morphopy.swc"
MORPHOPY_STATS_NAME = "arbor_stats_morphopy.pkl"


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


def _morphopy_stats_exists(root_output: Path, seg_id: int) -> bool:
    d = _seg_dir(root_output, seg_id)
    if not d.exists():
        return False
    return (d / MORPHOPY_STATS_NAME).exists()


def _ensure_morphopy_modules() -> dict[str, Any]:
    global _MORPHOPY_FUNCS
    if _MORPHOPY_FUNCS is not None:
        return _MORPHOPY_FUNCS
    try:
        from . import morphopyStats as morph_mod
        from skeliner.io import to_swc as skeliner_to_swc  # type: ignore
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or "morphopy dependencies"
        raise ArborRunError(
            f"Stats method 'morphopy' requires optional dependency '{missing}'. {_MORPHOPY_HINT}",
            code="missing-morphopy-deps",
        ) from exc
    except ImportError as exc:
        raise ArborRunError(
            f"Failed to import morphopy stack: {exc}",
            code="missing-morphopy-deps",
        ) from exc

    _MORPHOPY_FUNCS = {
        "process_cell": morph_mod.process_cell,
        "compute_stats": morph_mod.compute_stats,
        "get_cv": morph_mod.get_cv,
        "to_swc": skeliner_to_swc,
    }
    return _MORPHOPY_FUNCS


def _load_global_map_from_disk(path: Path) -> dict:
    if not path.exists():
        raise ArborRunError(
            f"Morphopy global_map file not found: {path}",
            code="morphopy-global-map-missing",
        )
    try:
        with np.load(path, allow_pickle=True) as data:
            mapping = {key: data[key] for key in data.files}
    except Exception as exc:
        raise ArborRunError(
            f"Failed to load morphopy global_map: {exc}",
            code="morphopy-global-map-error",
        ) from exc
    return mapping


def _get_global_map() -> dict:
    global _MORPHOPY_GLOBAL_MAP
    if _MORPHOPY_GLOBAL_MAP is None:
        _MORPHOPY_GLOBAL_MAP = _load_global_map_from_disk(_MORPHOPY_GLOBAL_MAP_PATH)
    return _MORPHOPY_GLOBAL_MAP


def _get_cloudvolume():
    global _MORPHOPY_CV
    if _MORPHOPY_CV is not None:
        return _MORPHOPY_CV
    funcs = _ensure_morphopy_modules()
    get_cv = funcs["get_cv"]
    try:
        cv = get_cv()
    except SystemExit as exc:
        raise ArborRunError(
            "No CAVEclient token found. Please add one before using the morphopy backend.",
            code="No-CAVEclient-token-found",
        ) from exc
    except ArborRunError:
        raise
    except Exception as exc:
        raise ArborRunError(
            f"Failed to initialize CloudVolume: {exc}",
            code="morphopy-cv-error",
        ) from exc
    _MORPHOPY_CV = cv
    return _MORPHOPY_CV


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

    return_code = proc.wait()
    full_output = "".join(lines)

    # Preserve original behavior: detect 'no mesh' case and raise.
    if "No meshes found." in full_output:
        (seg_dir / "flatone_error.txt").write_text("No meshes found.\n")
        raise ArborRunError("No meshes found.", code="no-mesh")
    if "No CAVEclient token found." in full_output:
        (seg_dir / "flatone_error.txt").write_text("No CAVEclient token found.\n")
        raise ArborRunError(
            "No CAVEclient token found. Please add token using the instructions above.",
            code="no-cave-token",
        )

    if return_code != 0:
        if full_output:
            error_file.write_text(full_output)
        else:
            error_file.write_text(f"flatone exited with code {return_code} without output.\n")
        raise ArborRunError(
            f"flatone exited with code {return_code} for seg {seg_id}",
            code="flatone-failed",
        )


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
        raise ArborRunError(
            f"No skeleton_warped.swc found for seg {seg_id}. Run flatone first or provide a skeleton.",
            code="missing-skeleton",
        )

    coords, radii, edges = load_swc(str(swc_path))
    stats, units = arborStatsFromSkeleton(coords, edges, radii=radii)
    payload = {"segment_id": seg_id, "stats": stats, "units": units}

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_pkl


def compute_morphopy_stats_for_seg(
    seg_id: int,
    root_output: Path,
    *,
    cell_class: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Compute meshes/skeletons using the morphopy pipeline (process_cell) and persist
    stats computed via compute_stats from morphopyStats.py.
    """
    segdir = _seg_dir(root_output, seg_id)
    segdir.mkdir(parents=True, exist_ok=True)

    error_file = segdir / "arbor_stats_error.txt"
    if error_file.exists():
        error_file.unlink()

    mesh_path = segdir / MORPHOPY_MESH_NAME
    swc_path = segdir / MORPHOPY_SWC_NAME
    out_pkl = segdir / MORPHOPY_STATS_NAME
    if out_pkl.exists() and not overwrite:
        return out_pkl

    funcs = _ensure_morphopy_modules()
    process_cell = funcs["process_cell"]
    compute_stats_fn = funcs["compute_stats"]
    to_swc = funcs["to_swc"]
    cv = _get_cloudvolume()
    global_map = _get_global_map()

    try:
        result = process_cell(
            cell=str(seg_id),
            cv=cv,
            global_map=global_map,
            always_detect_axon=True,
            cellclass=cell_class,
            add_stats=False,
            verbose=True
        )
    except ArborRunError:
        raise
    except Exception as exc:
        raise ArborRunError(
            f"Morphopy pipeline failed for seg {seg_id}: {exc}",
            code="morphopy-failed",
        ) from exc

    mesh = result.get("mesh")
    skel = result.get("skel")
    if mesh is None or skel is None:
        raise ArborRunError(
            f"Morphopy pipeline did not provide mesh and skeleton for seg {seg_id}.",
            code="morphopy-failed",
        )

    try:
        mesh_path.write_bytes(mesh.to_obj())
    except Exception as exc:
        raise ArborRunError(
            f"Failed to write {mesh_path.name} for seg {seg_id}: {exc}",
            code="morphopy-write-failed",
        ) from exc

    try:
        to_swc(skel, swc_path, include_header=True, include_meta=True)
    except Exception as exc:
        raise ArborRunError(
            f"Failed to write {swc_path.name} for seg {seg_id}: {exc}",
            code="morphopy-write-failed",
        ) from exc

    try:
        stats = compute_stats_fn(skel=skel)
    except Exception as exc:
        raise ArborRunError(
            f"Failed to compute morphopy stats for seg {seg_id}: {exc}",
            code="morphopy-failed",
        ) from exc

    payload = {
        "segment_id": seg_id,
        "stats": stats,
        "units": None,
        "stats_method": "morphopy",
    }
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
    stats_method: str,
) -> Tuple[bool, bool]:
    """
    Returns (need_flatone, need_arbor_stats) booleans for this seg_id.
    - mode: "both" | "flatone-only" | "arbor-only"
    - overwrite: force recompute
    - new_only: only compute when outputs are missing
    """
    want_flat = stats_method == "flatone" and mode in ("both", "flatone-only")
    want_arbor = mode in ("both", "arbor-only")
    stats_exists_fn = _arbor_stats_exists if stats_method == "flatone" else _morphopy_stats_exists

    if overwrite:
        return (want_flat, want_arbor)

    # not overwriting: skip if existing (and if new_only is requested)
    if new_only:
        need_flat  = want_flat  and not _flatone_exists(root_output, seg_id)
        need_arbor = want_arbor and not stats_exists_fn(root_output, seg_id)
        return (need_flat, need_arbor)

    # default non-overwrite: we let the underlying tasks short-circuit if they see outputs
    return (want_flat, want_arbor)


def _one_worker(args: tuple) -> tuple:
    seg_id, root_output, mode, overwrite, new_only, stats_method, cell_class = args
    try:
        need_flat, need_arbor = _decide_tasks_for_seg(
            seg_id, root_output, mode, overwrite, new_only, stats_method
        )

        if stats_method == "flatone":
            if need_flat:
                run_flattener(seg_id, root_output, overwrite=overwrite)
            if need_arbor:
                compute_arbor_stats_for_seg(seg_id, root_output, overwrite=overwrite)
        else:
            if need_arbor:
                print("#####################################################################\n" \
                "Cell Class:", cell_class)
                compute_morphopy_stats_for_seg(
                    seg_id,
                    root_output,
                    cell_class=cell_class,
                    overwrite=overwrite,
                )

        return ("ok", seg_id)

    except ArborRunError as e:
        # Something expected but absent (e.g., no mesh / no SWC).
        print("Error : ", e)
        msg = str(e)
        code = getattr(e, "code", None)
        if code == "no-mesh" or msg == "No meshes found.":
            return ("no-mesh", seg_id, msg)
        if code == "no-cave-token" or msg.startswith("No CAVEclient token found."):
            return ("No-CAVEclient-token-found", seg_id, msg)
        if code == "missing-skeleton" or "No skeleton_warped.swc found" in msg:
            return ("missing-skeleton", seg_id, msg)
        if code == "flatone-failed":
            return ("flatone-failed", seg_id, msg)
        if code in ("missing-morphopy-deps", "morphopy-global-map-missing", "morphopy-global-map-error", "morphopy-cv-error"):
            return (code, seg_id, msg)
        if code in ("morphopy-failed", "morphopy-write-failed"):
            return ("morphopy-failed", seg_id, msg)
        return ("err", seg_id, msg)
    except Exception as e:
        return ("err", seg_id, f"{type(e).__name__}: {e}")


def process_many(
    seg_ids: Iterable[int],
    root_output: Path,
    overwrite: bool = False,
    jobs: int = 1,
    mode: str = "both",       # "both" | "flatone-only" | "arbor-only"
    new_only: bool = False,   # process only missing outputs when not overwriting
    stats_method: str = "flatone",
    cell_classes: dict[int, str | None] | None = None,
) -> None:
    """
    Backward-compatible entry point with two new keyword args:
      - mode: which tasks to run
      - new_only: only process segIDs missing required artifacts (when overwrite=False)
      - stats_method: 'flatone' (default) or 'morphopy'
    """
    if stats_method not in ("flatone", "morphopy"):
        raise ValueError(f"Unknown stats_method '{stats_method}'")
    if stats_method == "morphopy" and mode == "flatone-only":
        raise ValueError("stats_method='morphopy' is incompatible with mode='flatone-only'")

    root_output = Path(root_output)
    root_output.mkdir(parents=True, exist_ok=True)

    tracked_markers = {
        "not_processed_seg_ids.txt",
        "arbor_stats_error_seg_ids.txt",
        "flatone_failed_seg_ids.txt",
        "morphopy_failed_seg_ids.txt",
    }
    for f in root_output.glob("*.txt"):
        if f.name in tracked_markers:
            f.unlink()

    # Prepare work items for the pool
    class_map = {int(k): v for k, v in (cell_classes or {}).items()}
    work = [
        (
            int(sid),
            root_output,
            mode,
            bool(overwrite),
            bool(new_only),
            stats_method,
            class_map.get(int(sid)),
        )
        for sid in seg_ids
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs) as pool:
        for res in pool.imap_unordered(_one_worker, work):
            kind = res[0]
            
            if kind == "No-CAVEclient-token-found":
                _, sid, msg = res
                print(f"SegID {sid} skipped: {msg}", file=sys.stderr)
                break
            if kind in (
                "missing-morphopy-deps",
                "morphopy-global-map-missing",
                "morphopy-global-map-error",
                "morphopy-cv-error",
            ):
                _, sid, msg = res
                print(msg, file=sys.stderr)
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
            if kind in ("missing-skeleton", "flatone-failed"):
                _, sid, msg = res
                (root_output / "flatone_failed_seg_ids.txt").open("a").write(f"{sid}\n")
                (root_output / str(sid) / "arbor_stats_error.txt").write_text(msg)
                continue
            if kind == "morphopy-failed":
                _, sid, msg = res
                (root_output / "morphopy_failed_seg_ids.txt").open("a").write(f"{sid}\n")
                (root_output / str(sid) / "arbor_stats_error.txt").write_text(msg)
                continue

            # kind == "err"
            _, sid, msg = res
            (root_output / str(sid) / "arbor_stats_error.txt").write_text(msg)
            (root_output / "arbor_stats_error_seg_ids.txt").open("a").write(f"{sid}\n")
