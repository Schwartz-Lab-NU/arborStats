import time
from copy import deepcopy
from warnings import warn

import numpy as np
import skeliner as sk
import trimesh
from matplotlib import pyplot as plt
from pywarper.warpers import warp_nodes, normalize_nodes
from skeliner import Skeleton


def detect_soma_and_connect_skeleton(
        skel: Skeleton,
        mesh: trimesh.Trimesh,
        *,
        soma_radius_percentile_threshold=99.9,
        soma_radius_distance_factor=4.0,
        soma_min_nodes=3,
        inside_tol=0.0,
        near_factor=1.2,
        fat_factor=0.20,
        tip_extent_factor=1.2,
        stem_extent_factor=3.0,
        verbose=False,
) -> tuple[Skeleton, dict]:
    """Post-process the skeleton based on cell type and other info."""
    post_info = dict()

    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    skel.ntype[:] = 3
    skel = skel.detect_soma(
        soma_radius_percentile_threshold=soma_radius_percentile_threshold,
        soma_radius_distance_factor=soma_radius_distance_factor,
        soma_min_nodes=soma_min_nodes,
        mesh_vertices=mesh_vertices,
        verbose=verbose,
    )

    skel = skel.merge_near_soma_nodes(
        mesh_vertices=mesh_vertices,
        inside_tol=inside_tol,
        near_factor=near_factor,
        fat_factor=fat_factor,
        verbose=verbose,
    )

    skel.bridge_gaps(
        bridge_max_factor=None,  # adaptive heuristic
        bridge_recalc_after=None,
        rebuild_mst=False,  # rebuild happens in the next stage
        verbose=verbose,
    )

    skel = skel.rebuild_mst(verbose=verbose)

    skel = skel.prune_neurites(
        mesh_vertices=mesh_vertices,
        tip_extent_factor=tip_extent_factor,
        stem_extent_factor=stem_extent_factor,
        drop_single_node_branches=True,
        verbose=verbose,
    )

    return skel, post_info


def warp_skeleton(
        skel: Skeleton,
        surface_mapping: dict,
        *,
        voxel_resolution: float | list[float | int] = [1.0, 1.0, 1.0],
        on_sac_pos: float = 0.0,
        off_sac_pos: float = 12.0,
        skeleton_nodes_scale: float = 1.0,
        conformal_jump: int | None = None,
        backward_compatible: bool = False,
        verbose: bool = False,
) -> tuple[Skeleton, dict]:
    """
    Flatten skeleton using global mapping.
    See pywarper.warp_skeleton for details.
    """
    post_info = dict()

    nodes = (
            skel.nodes.astype(float) * skeleton_nodes_scale
    )  # scale to the surface unit, which is often μm

    if verbose:
        print("[pywarper] Warping skeleton...")
        start_time = time.time()

    warped_nodes, med_z_on, med_z_off = warp_nodes(
        nodes,
        surface_mapping,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
    )

    normalized_nodes = normalize_nodes(
        warped_nodes,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
    )

    normalized_nodes /= skeleton_nodes_scale

    if verbose:
        print(f"    done in {time.time() - start_time:.2f} seconds.")

    normalized_soma = deepcopy(skel.soma)
    normalized_soma.center = (
            normalized_nodes[0] * voxel_resolution
    )  # soma is at the first node

    skel_norm = Skeleton(
        soma=normalized_soma,
        nodes=normalized_nodes * voxel_resolution,
        edges=skel.edges,
        radii=skel.radii,
        ntype=skel.ntype,
        node2verts=skel.node2verts,
        vert2node=skel.vert2node,
        meta=skel.meta.copy(),
    )

    skel_norm.extra['nodes_raw'] = nodes

    return skel_norm, post_info


def class_postprocess_skeleton(
        skel: Skeleton,
        *,
        cellclass: str | None,
        ais_xyz_um: tuple[float, float, float] | None,
        verbose: bool = False,
):
    if cellclass is not None:
        cellclass = cellclass.lower()
        if not cellclass in ['rgc', 'bc', 'hc', 'ac', 'glia']:
            raise ValueError(f"Unknown cellclass: {cellclass}")
    post_info = {}
    if ais_xyz_um is not None:
        post_info['input_ais'] = ais_xyz_um

    if cellclass in ['rgc']:
        detected_axon, ais, status = detect_axon(skel=skel, ais_xyz_um=ais_xyz_um, verbose=verbose)
        post_info['detected_axon'] = detected_axon
        post_info['detected_axon_status'] = status
        post_info['detected_axon_ais'] = ais

    return skel, post_info


def order_branchpoints(
        skel: Skeleton,
) -> tuple[np.ndarray, np.ndarray]:
    """Find all branchpoints (degree >= 3) and get their on-graph distance
    to the soma. Returns branchpoints and (unitless) distances."""
    all_nodes = np.arange(len(skel.nodes))

    # exclude all degree=1 and =2 nodes from the subtree
    deg1_nodes = skel.nodes_of_degree(k=1)
    deg2_nodes = skel.nodes_of_degree(k=2)
    nonbranching_nodes = np.concatenate((deg1_nodes, deg2_nodes))
    branchpoints = all_nodes[~np.isin(all_nodes, nonbranching_nodes)]

    # order branchpoints by graph distance to soma
    if sum(skel.ntype == 1) != 1:
        raise ValueError('More than one soma node!')
    skelgraph = skel._igraph()
    soma_node = np.where(skel.ntype == 1)[0][0]
    dists_soma_to_branchpoints = np.array(skelgraph.distances(source=soma_node, target=branchpoints)[0])

    return branchpoints, dists_soma_to_branchpoints


def filter_backward_neighbors(
        skel: Skeleton,
        neighbors: np.ndarray,
        branchpoint: np.ndarray,
):
    """For the neighbors of a branchpoint, remove those that are closer
    to the soma than the branchpoint itself (to avoid walking backwards
    on the skeleton)"""
    if sum(skel.ntype == 1) != 1:
        raise ValueError('More than one soma node!')
    skelgraph = skel._igraph()
    soma_node = np.where(skel.ntype == 1)[0][0]

    dist_branchpoint_to_soma = skelgraph.distances(source=soma_node, target=branchpoint)[0][0]
    dist_neighbors_to_soma = np.array(skelgraph.distances(source=soma_node, target=neighbors)[0])
    neighbors = neighbors[dist_neighbors_to_soma > dist_branchpoint_to_soma]

    return neighbors


def plot_skel_highlight_nodes(
        skel: Skeleton,
        *,
        highlight_nodes: list[int] | np.ndarray,
        label: str | None = None,
        max_meanz_axon: float | None = None,
):
    """Plots YZ view of the skeleton, with selected nodes highlighted
    and the meanz threshold for axon detection as a hline"""
    _, ax = plt.subplots(1, 1, figsize=(10, 5))

    sk.plot2d(skel=skel, plane='yz', ax=ax)

    highlight_nodes_yz = skel.nodes[highlight_nodes, 1:]
    ax.scatter(*highlight_nodes_yz.T, s=10, lw=2, c='tab:blue', label=label)

    allnodes_y = skel.nodes[:, 1]
    if max_meanz_axon is not None:
        ax.hlines(y=max_meanz_axon, xmin=min(allnodes_y), xmax=max(allnodes_y), colors='k')

    ax.legend(loc=(1, 0.2))
    plt.show()


def find_deepest_subtree(
        skel: Skeleton,
        *,
        branchpoints: list[int] | np.ndarray,
        dists_soma_to_branchpoints: list[int] | np.ndarray,
        processed_branchpoints: list[int] | None = None,
        minnodes_axon: int = 10,
        max_meanz_axon: float = -15,
        verbose: bool = False,
        plot_normal_outcomes: bool = False,
        plot_edgecases: bool = False,
) -> tuple[int | list[int] | None, list[int] | list[list[int]] | None, str]:
    """Starting from the soma, iterates over the branchpoints of a skeleton and
    looks for the first subtrees that lies below `max_meanz_axon` on average.
    Short branches are ignored, multiple axon-like subtrees are returned
    with a warning."""

    if processed_branchpoints is None:
        processed_branchpoints = []

    # select branchpoint closest to soma on graph for processing in this call
    sort_idx = np.argsort(dists_soma_to_branchpoints)
    branchpoint = branchpoints[sort_idx][0]
    neighbors = np.array(skel.neighbors(branchpoint))
    neighbors = filter_backward_neighbors(
        skel=skel,
        neighbors=neighbors,
        branchpoint=branchpoint)

    # check all subtrees starting at the branchpoint's neighbors
    if verbose:
        print(f'Looking for axon around branchpoint {branchpoint} '
              f'(remaining branchpoint list length: {len(branchpoints)})')
    subtree_nodes_not_promising = []
    subtrees_axon = []
    rootnodes_axon = []
    for neighbor in neighbors:
        if verbose:
            print(f'--- subtree for neighbor {neighbor}', end='')

        subtree = skel.extract_neurites(root=neighbor)
        subtree_z = skel.nodes[subtree, 2]
        subtree_meanz = np.mean(subtree_z)

        ## all promising subtrees (with at least some deep-lying nodes) will be processed
        ispromising = sum(subtree_z < max_meanz_axon) >= 10
        ## axon subtrees will be returned immediately
        isaxon = (len(subtree) >= 10) and (subtree_meanz < max_meanz_axon)

        if verbose:
            print(f' - z={subtree_meanz:.2f} n={len(subtree)}- ', end='')

        if ispromising:
            if verbose:
                print('..is promising')
        else:
            subtree_nodes_not_promising += subtree
            if verbose:
                print('..is NOT promising')

        if isaxon:
            subtrees_axon.append(subtree)
            rootnodes_axon.append(neighbor)

    processed_branchpoints.append(branchpoint)

    # if exactly one axon found, return
    if len(subtrees_axon) == 1:
        if verbose:
            print('Axon found!')

        axon = subtrees_axon[0]
        ais = rootnodes_axon[0]

        if plot_normal_outcomes:
            plot_skel_highlight_nodes(
                skel=skel,
                highlight_nodes=axon,
                label='axon',
                max_meanz_axon=max_meanz_axon)

        return axon, ais, 'Axon found'

    # multiple axon-subtrees: warn
    elif len(subtrees_axon) > 1:
        if plot_edgecases:
            for i, axon in enumerate(subtrees_axon):
                plot_skel_highlight_nodes(
                    skel=skel,
                    highlight_nodes=axon,
                    label=f'axon {i}',
                    max_meanz_axon=max_meanz_axon)

        warn('Multiple or branching axon found... --- returning both')
        return subtrees_axon, rootnodes_axon, 'Multiple axons found'

    # no axons yet: continue with the next-closest branchpoint that is on a promising subtree
    else:
        ## remove the processed branchpoint and all non-promising subtrees from branchpoint list
        nodes_to_remove = subtree_nodes_not_promising + [branchpoint]
        branchpoints_to_remove = np.isin(branchpoints, nodes_to_remove)
        branchpoints = branchpoints[~branchpoints_to_remove]
        dists_soma_to_branchpoints = dists_soma_to_branchpoints[~branchpoints_to_remove]

        # if no promising branchpoints left, call it a day
        if len(branchpoints) == 0:
            if verbose:
                print('No more promising branches left - probably no axon on this cell?')

            if plot_normal_outcomes:
                plot_skel_highlight_nodes(
                    skel=skel,
                    highlight_nodes=processed_branchpoints,
                    label='processed branchpoints',
                    max_meanz_axon=max_meanz_axon)

            return None, None, 'No Axon found'


        # otherwise, got to next branchpoint
        else:
            return find_deepest_subtree(
                skel=skel,
                branchpoints=branchpoints,
                processed_branchpoints=processed_branchpoints,
                dists_soma_to_branchpoints=dists_soma_to_branchpoints,
                minnodes_axon=minnodes_axon,
                max_meanz_axon=max_meanz_axon,
                verbose=verbose,
                plot_normal_outcomes=plot_normal_outcomes,
                plot_edgecases=plot_edgecases)


def detect_axon(
        skel: Skeleton,
        *,
        ais_xyz_um: tuple[float, float, float] | None = None,
        minnodes_axon: int = 10,
        max_meanz_axon: float = -15,
        verbose: bool = False,
        plot_normal_outcomes: bool = False,
        plot_edgecases: bool = False,
) -> tuple[bool, list, str]:
    """Detect axon in the skeleton and set the types of the axon nodes accordingly."""

    if ais_xyz_um is not None:
        raise NotImplementedError("Manual annotation of AIS not implemented yet.")

    if len(skel.nodes) < 20:
        if verbose:
            warn('Aborting axon detection: Skeleton has too few nodes.')
        detected_axon = False
        return detected_axon, [], "ERR1"

    somax, somay, somaz = skel.soma.center
    allnodes_z = skel.nodes[:, 2]
    if somaz > np.mean(allnodes_z):
        if verbose:
            warn('Aborting axon detection: Soma seems to be above dendritic tree, not below as expected for GCL cells.')
        detected_axon = False
        return detected_axon, [], 'ERR2'

    branchpoints, dists_soma_to_branchpoints = order_branchpoints(skel)
    axon, ais, status = find_deepest_subtree(
        skel=skel,
        branchpoints=branchpoints,
        dists_soma_to_branchpoints=dists_soma_to_branchpoints,
        minnodes_axon=minnodes_axon,
        max_meanz_axon=max_meanz_axon,
        verbose=verbose,
        plot_normal_outcomes=plot_normal_outcomes,
        plot_edgecases=plot_edgecases)

    if status == 'No axon found':
        detected_axon = False
    elif status == 'Axon found':
        sk.post.set_ntype(skel, root=ais, code=2)
        detected_axon = True
    elif status == 'Multiple axons found':
        for single_ais in ais:
            sk.post.set_ntype(skel, root=single_ais, code=2)
        detected_axon = True
    elif status == 'INL cell?':
        detected_axon = False
    else:
        detected_axon = False

    return detected_axon, ais, status


def calibrate_radii(skel, mesh, verbose: bool = False):
    skel.convert_unit(target_unit='nm')
    sk.post.calibrate_radii(
        skel=skel,
        mesh=mesh,
        store_key="calibrated",
        verbose=verbose,
        aggregate="trim",
        min_n_outer=20,
        min_frac_outer=0.33,
        min_verts_q_outer=90.,
        rays_num_outer=30,
        rays_thresh_outer=0.2,
    )

def reroot_cellclass_based(skel, cellclass):
    soma_layer_plausible = (skel.soma.center[2] > 12) or (skel.soma.center[2] < 0)
    if cellclass is not None:
        if cellclass.lower() in ['bc']:
            soma_layer_plausible = skel.soma.center[2] > 20
        elif cellclass.lower() in ['hc']:
            soma_layer_plausible = skel.soma.center[2] > 30
        elif cellclass.lower() in ['rgc', 'ac']:
            soma_layer_plausible = (skel.soma.center[2] > 20) or (skel.soma.center[2] < -5)

    post_has_soma = (
            (1 in skel.ntype)  # Has soma type
            & soma_layer_plausible  # Soma is in nucleus layer
            & (len(skel.node2verts[0]) > 10)  # Soma has a minimum number of mesh vertices
            & (skel.soma.equiv_radius > 1.7)  # Soma has a minimum size
    )

    if not post_has_soma:  # If soma is not present, set soma to max z-value
        skel = skel.reroot(
            axis="z", mode="max", prefer_leaves=True, rebuild_mst=True, verbose=False)
        skel.ntype[skel.ntype == 1] = 3
        skel.ntype[0] = -1

    return skel