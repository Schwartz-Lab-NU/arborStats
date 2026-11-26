"""Arbor features the Eyewire II cell typing found informative (added 2026-11).

Copied from eyewire2-datajoint (``datajoint/tables/morphopy_stats.py``, ``convex_hull.py``,
``dendrite_waviness.py``, ``dendrite_asymmetry.py``, ``utils/xyz.py``) so this package stays
independent of it. What the clustering learned (celltype_explorer presets):

- *pruned* branch features (tiny terminal neurites <= 10 um removed) transfer between datasets
  and segmentation qualities; raw tips/branch counts, terminal segment lengths, path angles and
  tortuosity do not,
- separate ON and OFF hull diameters (z windows around the two SAC bands) tell mono- from
  bistratified cells and are used for every type,
- waviness (``wavy_band_rms_xy``) separates medium/wide-field amacrine types; it is measured on
  long smoothed dendrite paths and so is robust to node jitter (unlike path angles),
- angular asymmetry (circular harmonics) helps with direction-selective / asymmetric types,
- coverage: how much of the cable lies where the volume or the SAC band observations end
  (pywarper >= 0.3 maps); size features of a cut-off cell are lower bounds.
"""
from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
from morphopy.neurontree.NeuronTree import NeuronTree
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull

# ----------------------------------------------------------------------------------------------
# Pruned MorphoPy statistics (eyewire2-datajoint MorphoPyStatPruned)
# ----------------------------------------------------------------------------------------------

TINY_NEURITE_MAX_LEN_UM = 10.0
TINY_NEURITE_PRUNE_STEPS = 2
PRUNED_STAT_KEYS = (
    'tips', 'branch_points', 'stems', 'total_length', 'max_branch_order',
    'median_intermediate_segment_pl', 'median_terminal_segment_pl',
)


def _tiny_neurite_nodes(neurontree, max_len_um):
    """Node ids of terminal branches (tip back to the nearest branch point or root) shorter than
    ``max_len_um``. Single pass: branch points exposed by trimming are not re-evaluated."""
    root = neurontree.get_root()
    other_points = set(np.unique(np.append(neurontree.get_branchpoints(), root)).tolist())
    tips_raw = neurontree.get_tips()
    tips = tips_raw.tolist() if hasattr(tips_raw, "tolist") else list(tips_raw)
    G = neurontree.get_graph()

    to_remove = set()
    for tip in tips:
        if tip == root or tip in other_points:
            continue
        segment = [tip]
        node = tip
        pred = next(iter(G.predecessors(node)))
        length = G[pred][node]["path_length"]
        while pred not in other_points:
            segment.append(pred)
            node = pred
            pred = next(iter(G.predecessors(node)))
            length += G[pred][node]["path_length"]
        if length < max_len_um:
            to_remove.update(segment)
    return to_remove


def _drop_nodes(neurontree, nodes):
    G = neurontree.get_graph()
    keep_nodes = [n for n in G.nodes() if n not in nodes]
    return NeuronTree(graph=nx.subgraph(G, keep_nodes))


def remove_tiny_neurites(neurontree, max_len_um=TINY_NEURITE_MAX_LEN_UM, n_steps=TINY_NEURITE_PRUNE_STEPS):
    """Copy of ``neurontree`` with short terminal branches trimmed, the threshold ramped up over
    ``n_steps`` passes so that siblings of removed branches are judged on the merged topology."""
    tree = neurontree
    for threshold in np.linspace(max_len_um / n_steps, max_len_um, n_steps):
        to_remove = _tiny_neurite_nodes(tree, threshold)
        if to_remove:
            tree = _drop_nodes(tree, to_remove)
    return tree


def compute_pruned_stats(morph, keys=PRUNED_STAT_KEYS) -> dict:
    """``pruned_<key>`` MorphoPy statistics of ``morph`` after removing tiny neurites."""
    import warnings
    from morphopy.computation.feature_presentation import compute_morphometric_statistics

    pruned = remove_tiny_neurites(morph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stats = dict(compute_morphometric_statistics(pruned).iloc[0])
    return {f'pruned_{k}': _float_or_none(stats.get(k)) for k in keys}


# ----------------------------------------------------------------------------------------------
# ON / OFF layer hulls (eyewire2-datajoint ConvexHullONOFF)
# ----------------------------------------------------------------------------------------------

ZLIM_ON = (-12.0, 5.0)
ZLIM_OFF = (5.0, 22.0)


def _hull_size(xy):
    if xy.shape[0] < 3:
        return dict(perimeter=0.0, area=0.0, diameter=0.0)
    try:
        hull = ConvexHull(xy)
    except Exception:  # degenerate (collinear) points
        return dict(perimeter=0.0, area=0.0, diameter=0.0)
    return dict(perimeter=float(hull.area), area=float(hull.volume),
                diameter=float(np.sqrt(4 * hull.volume / np.pi)))


def compute_onoff_hulls(skel, zlim_on=ZLIM_ON, zlim_off=ZLIM_OFF) -> dict:
    """Convex hulls of the dendrite nodes (ntype 3/4) inside the ON and the OFF z window (flat
    IPL units, ON SAC band = 0, OFF = 12). A plain z mask, so a thin slab may hold several
    disconnected pieces of the arbor. Size 0 with fewer than 3 nodes."""
    out = {}
    dend = (skel.ntype == 3) | (skel.ntype == 4)
    for name, (z0, z1) in (('on', zlim_on), ('off', zlim_off)):
        mask = dend & (skel.nodes[:, 2] >= z0) & (skel.nodes[:, 2] <= z1)
        for k, v in _hull_size(skel.nodes[mask, :2]).items():
            out[f'{name}_hull_{k}'] = v
    return out


# ----------------------------------------------------------------------------------------------
# Waviness (eyewire2-datajoint DendriteWaviness)
# ----------------------------------------------------------------------------------------------

def _long_paths(nodes, edges, ntype, root=0, dendrite_ntypes=(3,)):
    """Long paths as node-index arrays: at each branch point the path continues into the child
    with the most downstream cable; split into runs of >= 3 dendrite nodes."""
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    parent = np.full(n, -1)
    order = []
    seen = np.zeros(n, bool)
    seen[root] = True
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                q.append(v)

    cable = np.zeros(n)
    for v in reversed(order):
        p = parent[v]
        if p >= 0:
            cable[p] += cable[v] + np.linalg.norm(nodes[v] - nodes[p])
    children = [[] for _ in range(n)]
    for v in order[1:]:
        children[parent[v]].append(v)
    heavy = np.full(n, -1)
    for u in range(n):
        if children[u]:
            heavy[u] = max(children[u], key=lambda c: cable[c])

    is_dendrite = np.isin(ntype, dendrite_ntypes)
    paths = []
    starts = list(children[root])
    while starts:
        s = starts.pop()
        p = [] if parent[s] == root else [parent[s]]
        v = s
        while v >= 0:
            p.append(v)
            starts.extend(c for c in children[v] if c != heavy[v])
            v = heavy[v]
        p = np.array(p)
        idx = np.flatnonzero(np.diff(np.r_[0, is_dendrite[p].astype(int), 0]))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a >= 3:
                paths.append(p[a:b])
    return paths


def _arclen(P):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def _resample(P, step):
    s = _arclen(P)
    if s[-1] < 2 * step:
        return None
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])])


def _smooth_curve(P, sigma, step):
    R = _resample(P, step)
    if R is None or len(R) < 5:
        return None
    S = gaussian_filter1d(R, sigma / step, axis=0, mode='nearest')
    return _resample(S, step)


def _smooth_curve_reparam(P, sigma, step, n_iter=2):
    t = _arclen(P)
    if t[-1] < 4 * step:
        return None
    for _ in range(n_iter):
        u = np.arange(0, t[-1] + 1e-9, step)
        R = np.column_stack([np.interp(u, t, P[:, k]) for k in range(P.shape[1])])
        S = gaussian_filter1d(R, sigma / step, axis=0, mode='nearest')
        t = np.interp(t, u, _arclen(S))
    return _resample(S, step)


def _band(C, sigma_hi, step, trim):
    B = C - gaussian_filter1d(C, sigma_hi / step, axis=0, mode='nearest')
    a = int(round((trim + 2 * sigma_hi) / step))
    return B[a:len(B) - a]


def _window_excess(C, window, step, trim):
    a = int(round(trim / step))
    C = C[a:len(C) - a]
    n = int(round(window / step))
    if len(C) <= n:
        return np.array([])
    j = np.arange(0, len(C) - n, max(1, n // 2))
    chord = np.linalg.norm(C[j + n] - C[j], axis=1)
    return window / np.maximum(chord, 1e-6) - 1


def _divider(P, chord, h=0.05):
    R = _resample(P, h)
    if R is None:
        return None
    pts = [R[0]]
    i = 0
    n = len(R)
    w = int(4 * chord / h) + 10
    while True:
        j_end = min(n, i + w)
        d = np.linalg.norm(R[i + 1:j_end] - R[i], axis=1)
        k = np.flatnonzero(d >= chord)
        if len(k) == 0:
            if j_end == n:
                break
            w *= 2
            continue
        i = i + 1 + k[0]
        pts.append(R[i])
    return np.array(pts) if len(pts) >= 3 else None


def _tangent_corr(D, lag):
    t = np.diff(D, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return (t[:-lag] * t[lag:]).sum(1)


def compute_waviness_stats(
        nodes, edges, ntype, root=0, dendrite_ntypes=(3,), step_um=0.25, smooth_sigma_um=1.0,
        trim_um=2.0, min_path_len_um=12.0, band_sigma_hi_um=5.0, fine_band_sigma_lo_um=0.5,
        fine_band_sigma_hi_um=2.0, excess_window_um=10.0, divider_step_um=2.0,
        decorr_short_um=4.0, decorr_long_um=10.0) -> dict:
    """Waviness of long dendrite paths (``nodes`` in um), prefixed ``wavy_``:

    - ``band_rms_xy``: RMS (um) of bends with wavelengths of roughly 6-30 um (xy curve smoothed
      at 1 um minus a copy smoothed at 5 um); ``band_rms_fine_xy`` a finer band, ``band_rms_z``
      the z component,
    - ``decorr``: 1 - C(10 um) / C(4 um), C the correlation of path directions that far apart,
    - ``excess_*``: extra path length (arc / chord - 1) of the smoothed curve in 10 um windows.

    Only paths of at least ``min_path_len_um`` count. None where no path qualifies.
    """
    lag_short = int(round(decorr_short_um / divider_step_um))
    lag_long = int(round(decorr_long_um / divider_step_um))

    band_xy, band_fine_xy, band_z, excess, corr_short, corr_long = [], [], [], [], [], []
    cable, cable_long, n_paths = 0.0, 0.0, 0
    for path in _long_paths(nodes, edges, ntype, root=root, dendrite_ntypes=dendrite_ntypes):
        P = nodes[path]
        length = _arclen(P)[-1]
        cable += length
        if length < min_path_len_um:
            continue
        cable_long += length
        n_paths += 1

        C = _smooth_curve(P, smooth_sigma_um, step_um)
        if C is not None:
            band_z.append(_band(C, band_sigma_hi_um, step_um, trim_um)[:, 2] ** 2)
        Cxy = _smooth_curve(P[:, :2], smooth_sigma_um, step_um)
        if Cxy is not None:
            band_xy.append((_band(Cxy, band_sigma_hi_um, step_um, trim_um) ** 2).sum(1))
        Cxy_fine = _smooth_curve(P[:, :2], fine_band_sigma_lo_um, step_um)
        if Cxy_fine is not None:
            band_fine_xy.append((_band(Cxy_fine, fine_band_sigma_hi_um, step_um, trim_um) ** 2).sum(1))

        Cr = _smooth_curve_reparam(P, smooth_sigma_um, step_um)
        if Cr is not None:
            excess.append(_window_excess(Cr, excess_window_um, step_um, trim_um))

        D = _divider(P, divider_step_um)
        if D is not None and len(D) >= lag_long + 2:
            corr_short.append(_tangent_corr(D, lag_short))
            corr_long.append(_tangent_corr(D, lag_long))

    def _cat(x):
        return np.concatenate(x) if len(x) else np.array([])

    def _rms(x):
        x = _cat(x)
        return float(np.sqrt(np.mean(x))) if len(x) else None

    excess = _cat(excess)
    corr_short, corr_long = _cat(corr_short), _cat(corr_long)
    out = dict(
        n_long_paths=n_paths,
        frac_cable_long=cable_long / cable if cable > 0 else None,
        band_rms_xy=_rms(band_xy),
        band_rms_fine_xy=_rms(band_fine_xy),
        band_rms_z=_rms(band_z),
        decorr=float(1 - np.mean(corr_long) / np.mean(corr_short)) if len(corr_long) else None,
        excess_mean=float(np.mean(excess)) if len(excess) else None,
        excess_median=float(np.median(excess)) if len(excess) else None,
        excess_q95=float(np.percentile(excess, 95)) if len(excess) else None,
    )
    return {f'wavy_{k}': v for k, v in out.items()}


# ----------------------------------------------------------------------------------------------
# Angular asymmetry (eyewire2-datajoint DendriteAsymmetry)
# ----------------------------------------------------------------------------------------------

def _segment_ids(edges, n_nodes, root=0):
    """Id of the unbranched segment each node belongs to (node and the edge to its parent)."""
    adj = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    parent = np.full(n_nodes, -1)
    n_children = np.zeros(n_nodes, int)
    order = []
    seen = np.zeros(n_nodes, bool)
    seen[root] = True
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                n_children[u] += 1
                q.append(v)

    seg = np.full(n_nodes, -1)
    n_seg = 0
    for v in order[1:]:
        p = parent[v]
        if p == root or n_children[p] != 1:
            seg[v] = n_seg
            n_seg += 1
        else:
            seg[v] = seg[p]
    return seg


def _soma_mass_mask(skel, positions):
    """Per-edge masses (stored at the child node) belonging to the soma: node 0, everything
    inside the soma ellipsoid and the edges from the soma node."""
    is_soma = skel.soma.contains(positions)
    is_soma[0] = True
    is_soma[skel.edges[(skel.edges == 0).any(axis=1)].ravel()] = True
    return is_soma


def compute_asymmetry_stats(skel, ntypes=(3, 4), z_range_um=(-15.25, 24.75), r_range_um=(0.0, 250.0),
                            exclude_soma=True) -> dict:
    """Circular harmonics of the dendritic cable over the polar angle around the soma (xy),
    prefixed ``asym_``: ``c1`` one-sidedness (``lean_angle_deg`` its direction), ``c2``
    elongation (``axis_angle_deg``, mod 180), ``bipolar = c2 * (1 - c1)``, ``n_eff`` effective
    number of segments and ``c*_debiased`` with the 1/n_eff noise floor removed. Angles are in
    volume xy coordinates (counter-clockwise from +x). None where no cable is left."""
    from pywarper.warpers import segment_lengths

    weights, mids = segment_lengths(skel)
    center = np.asarray(skel.soma.center[:2], dtype=float)
    dx, dy = mids[:, 0] - center[0], mids[:, 1] - center[1]
    r = np.hypot(dx, dy)

    mask = (weights > 0) & np.isin(skel.ntype, ntypes)
    if exclude_soma:
        mask &= ~_soma_mass_mask(skel, mids)
    if z_range_um is not None:
        mask &= (mids[:, 2] >= z_range_um[0]) & (mids[:, 2] <= z_range_um[1])
    if r_range_um is not None:
        mask &= (r >= r_range_um[0]) & (r <= r_range_um[1])

    w = weights[mask]
    total = float(w.sum())
    if total <= 0:
        out = dict(total_length_um=0.0, n_eff=None, c1=None, c2=None, bipolar=None,
                   c1_debiased=None, c2_debiased=None, lean_angle_deg=None, axis_angle_deg=None)
    else:
        theta = np.arctan2(dy[mask], dx[mask])
        z1 = np.sum(w * np.exp(1j * theta)) / total
        z2 = np.sum(w * np.exp(2j * theta)) / total
        c1, c2 = float(np.abs(z1)), float(np.abs(z2))

        seg = _segment_ids(skel.edges, len(skel.nodes))[mask]
        seg_w = np.bincount(seg, weights=w)
        n_eff = float(total ** 2 / np.sum(seg_w ** 2))
        out = dict(
            total_length_um=total,
            n_eff=n_eff,
            c1=c1,
            c2=c2,
            bipolar=c2 * (1 - c1),
            c1_debiased=float(np.sqrt(max(c1 ** 2 - 1 / n_eff, 0.0))),
            c2_debiased=float(np.sqrt(max(c2 ** 2 - 1 / n_eff, 0.0))),
            lean_angle_deg=float(np.degrees(np.angle(z1))),
            axis_angle_deg=float(np.degrees(np.angle(z2)) / 2),
        )
    return {f'asym_{k}': v for k, v in out.items()}


# ----------------------------------------------------------------------------------------------
# Coverage (pywarper >= 0.3)
# ----------------------------------------------------------------------------------------------

COVERAGE_EXTENT = (-12.0, 22.0)  # IPL window (flat units) the cut_below/cut_above shares refer to
COVERAGE_KEYS = ('cut_below', 'cut_above', 'on_unsupported', 'off_unsupported')


def compute_coverage_summary(skel_warped, nodes_raw, global_map, extent=COVERAGE_EXTENT,
                             on_sac_pos=0.0, off_sac_pos=12.0) -> dict:
    """Shares of the cable (``coverage_<key>``) lying where the data stop covering it:

    - ``cut_below`` / ``cut_above``: the imaged volume ends inside the IPL window ``extent``,
      below / above it, in that cable's column (the arbor may continue outside the volume),
    - ``on_unsupported`` / ``off_unsupported`` (and ``<layer>_unsupported`` for extra layers):
      that SAC band was not observed in the cable's column, so its depth there is extrapolated.

    ``skel_warped`` is the flattened skeleton and ``nodes_raw`` its nodes before warping (same
    order, the map's units). None when the map records none of this (pywarper < 0.3 maps).
    The per-node record is not kept on ``skel_warped`` (a later reroot would misalign it)."""
    import copy

    from pywarper.warpers import get_coverage, node_coverage

    coverage = node_coverage(np.asarray(nodes_raw, dtype=float), global_map)
    if coverage is None:
        return {f'coverage_{k}': None for k in COVERAGE_KEYS}
    if 'depth_rel' in coverage:
        with np.errstate(invalid='ignore'):
            depth = on_sac_pos + np.asarray(coverage['depth_rel']) * (off_sac_pos - on_sac_pos)
        coverage['depth'] = np.sort(depth, axis=1)
    view = copy.copy(skel_warped)
    view.extra = {**(skel_warped.extra or {}), 'coverage': coverage}
    summary = get_coverage(view, extent)
    return {f'coverage_{k}': _float_or_none(v) for k, v in summary.items()}


def _float_or_none(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
