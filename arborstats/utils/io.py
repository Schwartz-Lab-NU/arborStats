from collections import deque
from typing import List

import numpy as np
import pandas as pd
from morphopy.computation.file_manager import check_neurontree
from morphopy.neurontree.NeuronTree import NeuronTree
from morphopy.neurontree.utils import get_standardized_swc
from skeliner import Skeleton


def read_swc(swc_file, contains_zraw=False):
    f = lambda x: float(x.replace(",", "."))

    col_names = ["n", "type", "x", "y", "z", "radius", "parent"]
    if contains_zraw:
        col_names.append("zraw")

    return pd.read_csv(swc_file, comment="#", sep=r"\s+",
                       converters={"x": f, "y": f, "z": f, "radius": f},
                       names=col_names, index_col=False)


def reconstruct_warped_arbor(swc_file):
    df = read_swc(swc_file, contains_zraw=True)

    # Read first two line for metadata
    with open(swc_file, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2 or not lines[0].startswith("# medVZmin:") or not lines[1].startswith("# medVZmax:"):
            raise ValueError(f"SWC file does not contain expected metadata lines: {lines[:5]}")

        # Extract metadata
        med_vzmin = float(lines[0].split(":")[1].strip())
        med_vzmax = float(lines[1].split(":")[1].strip())

    # Extract edges from "n" and "parent" columns
    edges = df[["n", "parent"]].to_numpy()

    # Extract nodes from "x", "y", "z" columns
    nodes = df[["x", "y", "zraw"]].to_numpy()

    # Extract radii from the "radius" column
    radii = df["radius"].to_numpy()

    # Reconstruct the dictionary
    warped_arbor_normed = {
        "edges": edges,
        "nodes": nodes,
        "radii": radii,
        "medVZmin": med_vzmin,
        "medVZmax": med_vzmax,
    }

    return warped_arbor_normed


def skel_to_morphopy(skel: Skeleton, soma_center: bool = True) -> NeuronTree:
    """Convert skeliner.Skeleton to morphopy.NeuronTree"""
    radius_metric = 'calibrated' if 'calibrated' in skel.radii.keys() else None
    df_swc = skel_to_df(skel=skel, include_meta=False, scale=1.0, radius_metric=radius_metric)
    df_swc = get_standardized_swc(swc=df_swc, scaling=1.0, soma_radius=None, soma_center=soma_center, pca_rot=False)
    morph = NeuronTree(swc=df_swc)
    check_neurontree(morph)
    return morph


def _bfs_parents(edges: np.ndarray, n_nodes: int, *, root: int = 0) -> List[int]:
    # Copied from skeliner
    """Return parent[] array of BFS tree from *root* given an undirected edge list."""
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    parent = [-1] * n_nodes
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v != root and parent[v] == -1:
                parent[v] = u
                q.append(v)
    return parent


def skel_to_df(
        skel: Skeleton,
        *,
        include_meta: bool = False,
        scale: float = 1.0,
        radius_metric: str | None = None,
        axis_order: tuple[int, int, int] | str = (0, 1, 2),
):
    """
    Convert a skeliner Skeleton into a pandas DataFrame compatible with morphopy.

    Parameters
    ----------
    skel : skeliner.Skeleton
        Input skel object.
    include_meta : bool
        If True, metadata is included in DataFrame.attrs["meta"].
    scale : float
        Scaling factor applied to coordinates and radii.
    radius_metric : str or None
        Key for skel.radii[...] if alternative radius estimator is used.
    axis_order : tuple or 'xyz' string
        Permutation of coordinate axes.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns:
        ['n', 'type', 'x', 'y', 'z', 'radius', 'parent']
    """

    # --- normalize axis_order ---
    if isinstance(axis_order, str):
        axis_map = {"x": 0, "y": 1, "z": 2}
        try:
            axis_order = tuple(axis_map[c.lower()] for c in axis_order)
        except KeyError:
            raise ValueError("axis_order string must be a permutation of 'xyz'")
    axis_order = tuple(map(int, axis_order))
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError("axis_order must be a permutation of (0,1,2)")

    # --- BFS parents (same logic as to_swc) ---
    parent = np.asarray(_bfs_parents(skel.edges, len(skel.nodes), root=0), dtype=int)


    # --- choose radii ---
    if radius_metric is None:
        radii = skel.r
    else:
        if radius_metric not in skel.radii:
            raise ValueError(f"Unknown radius estimator '{radius_metric}'")
        radii = skel.radii[radius_metric]

    # --- node types ---
    if skel.ntype is not None:
        ntype = skel.ntype.astype(int, copy=False)
    else:
        ntype = np.full(len(skel.nodes), 0, dtype=int)
        if len(ntype) > 0:
            ntype[0] = -1  # root

    # --- coordinates, reordered + scaled ---
    coords = skel.nodes[:, axis_order] * scale
    radii = radii * scale

    # --- construct dataframe ---
    n = np.arange(1, len(coords) + 1)
    parent_adj = np.where(parent == -1, -1, parent + 1)  # 1-based parent

    df = pd.DataFrame({
        "n": n,
        "type": ntype,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "radius": radii,
        "parent": parent_adj,
    })

    # attach metadata if needed
    if include_meta and skel.meta:
        df.attrs["meta"] = skel.meta

    return df
