from types import MappingProxyType
from copy import deepcopy
import numpy as np
from pathlib import Path
import skeliner as sk
from cloudvolume import CloudVolume
import trimesh
from arborstats.utils.finalize_skeleton import finalize_skel
from arborstats.utils.io import skel_to_df, skel_to_morphopy
from morphopy.computation.feature_presentation import compute_morphometric_statistics
from arborstats.compute import compute_convex_hull_features, compute_density_xy, compute_density_z, \
    compute_percentiles_from_histogram, compute_features_xy_gauss, compute_gauss_xy_stats
from arborstats.utils.postprocess import detect_axon
from arborstats.features import (
    COVERAGE_KEYS, compute_asymmetry_stats, compute_onoff_hulls, compute_pruned_stats,
    compute_waviness_stats,
)

SKEL_KWS_RAW = MappingProxyType(dict(
    detect_soma=False,
    collapse_soma=False,
    bridge_gaps=False,
    prune_tiny_neurites=False,
    prune_drop_single_node_branches=False,
    soma_init_guess_axis="z",
    soma_init_guess_mode="max",
))

MORPHO_DIST_BINS = MappingProxyType({
    'branch_order': np.arange(0.5, 200.5 + 0.1, 10),
    'strahler_order': np.arange(0.5, 20.5 + 0.1, 1),
    'branch_angle': np.arange(0, 180 + 0.1, 10),
    'path_angle': np.arange(0, 180 + 0.1, 10),
    'root_angle': np.arange(0, 180 + 0.1, 10),
    'thickness': np.arange(0, 3 + 0.1, 0.1),
    'segment_length': 10. ** np.arange(-1, 3 + 0.1, 0.5),
    'path_length': 10. ** np.arange(-1, 3 + 0.1, 0.5),
    'radial_dist': 10. ** np.arange(-1, 3 + 0.1, 0.5),
    'sholl': 10 ** np.arange(0.1, 3 + 0.1, 0.2),
})

Z_DENS_KWS = MappingProxyType(dict(bin_size=0.5, measure='length', extent=(-20., 60.)))
Z_PERCS = (2, 5, 25, 50, 75, 95, 98)

XY_DENS_KWS = MappingProxyType(dict(bin_size=25.0, margin_bins=2, blur_sigma=1.))

def _prompt_for_token(client):
    """If the user has no CAVE token, print CAVEclient's instructions for getting and saving
    one (``CAVEclient().auth.save_token(token=...)``) and exit."""
    if client.auth.token is not None:
        return

    print("No CAVEclient token found.\n")
    client.auth.get_new_token()
    raise SystemExit

def get_cv() -> CloudVolume:
    """Get a CloudVolume instance for accessing meshes and skeletons.
    """
    from caveclient import CAVEclient
    from cloudvolume import CloudVolume
    client = CAVEclient()
    _prompt_for_token(client)
    cv = CloudVolume(
        "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/stroeh_mouse_retina/",
        use_https=True,
        progress=True,
    )
    return cv

def process_cell(
        cell: str | trimesh.Trimesh,
        cv: CloudVolume,
        global_map: dict,
        *,
        cellclass: str | None = None,
        always_detect_axon : bool = True,
        soma_seed_point_nm: tuple[float, float, float] | None = None,
        add_stats: bool = True,
        skel_kws_raw: dict | MappingProxyType = SKEL_KWS_RAW,
        morph_dist_bins: dict | MappingProxyType = MORPHO_DIST_BINS,
        z_dens_kws: dict | MappingProxyType = Z_DENS_KWS,
        z_percentiles : tuple = Z_PERCS,
        xy_dens_kws: dict | MappingProxyType = XY_DENS_KWS,
        verbose: bool = False
) -> dict:
    """Process a cell's skeleton by finalizing it with various stages.
    """

    # Mesh
    if isinstance(cell, trimesh.Trimesh):
        mesh = mesh_trimesh = cell
    else:
        cell = int(cell)
        mesh = cv.mesh.get(cell, remove_duplicate_vertices=True)[cell]
        mesh_trimesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            process=False,                          # skip trimesh’s cleanup
        )
        #mesh = mesh_dict[int(cell)].to_obj()

    # Skel
    skel_raw = sk.skeletonize(
        mesh_trimesh,
        unit="nm",
        verbose=verbose,
        soma_seed_point=soma_seed_point_nm,
        postprocess=False,
        **skel_kws_raw
    )

    # Skel Final
    skel, modification_info = finalize_skel(
        skel=skel_raw,
        mesh=mesh_trimesh,
        set_ntypes=True,
        ais_xyz_um=None,
        global_map=global_map,
        cellclass=cellclass,
        verbose=verbose,
        run_stage_soma=True,
        run_stage_warp=True,
        run_stage_class=True,
        run_stage_radii=False,
    )

    if always_detect_axon and (cellclass != 'RGC'):
        detect_axon(skel=skel, ais_xyz_um=None, verbose=verbose)

    skel_raw.convert_unit(target_unit="um")
    results = {
        'mesh': mesh,
        'skel': skel,
        'skel_raw': skel_raw,
    }

    if add_stats:
        stats = compute_stats(
            skel=skel,
            morph_dist_bins=morph_dist_bins,
            z_dens_kws=z_dens_kws,
            z_percentiles=z_percentiles,
            xy_dens_kws=xy_dens_kws,
        )
        results.update(stats)
    return results



def compute_stats(
        skel: sk.Skeleton,
        *,
        morph_dist_bins: dict | MappingProxyType = MORPHO_DIST_BINS,
        z_dens_kws: dict | MappingProxyType = Z_DENS_KWS,
        z_percentiles: tuple = Z_PERCS,
        xy_dens_kws: dict | MappingProxyType = XY_DENS_KWS,
) -> dict:
    skel = deepcopy(skel)
    skel.convert_unit('um')

    # Morphopy - Start
    skel_root_ntype = skel.ntype[0]
    skel.ntype[0] = 1  # Set root to soma
    morph = skel_to_morphopy(skel=skel, soma_center=True)
    skel.ntype[0] = skel_root_ntype  # Restore original ntype
    dend_tree = morph.get_dendritic_tree()
    dend_tree_minor = dend_tree.get_topological_minor()

    ## MorphoPy statistics
    if len(morph.get_dendrite_nodes()) < 20:
        morpho_stats = None
    else:
        morpho_stats = dict(compute_morphometric_statistics(morph).iloc[0])

    ## Morphopy distributions + sholl
    if len(morph.get_dendrite_nodes()) < 20:
        morph_dists = None
    else:
        morph_dists = dict()
        for stat, bins in morph_dist_bins.items():
            if stat == 'sholl':
                dist, radii = dend_tree.get_sholl_intersection_profile(
                    proj='xy', steps=None, centroid='soma', radii=bins)
                morph_dists[stat] = {'dist': dist, 'radii': bins}
            else:
                tree_compute = dend_tree_minor if stat in [
                    'branch_order', 'strahler_order', 'root_angle'] else dend_tree
                dist, edges = tree_compute.get_histogram(stat, bins=bins)
                morph_dists[stat] = {'dist': dist, 'edges': edges}
    # Morphopy - End

    # Features the Eyewire II typing found informative (see arborstats/features.py)
    arbor_features = compute_arbor_features(skel, morph=morph if morpho_stats is not None else None)

    # Convex hull
    dend_mask = (skel.ntype == 3) | (skel.ntype == 4)
    dend_nodes = skel.nodes[dend_mask, :2]
    if dend_nodes.shape[0] < 3:  # Need at least 3 points to compute a convex hull
        hull_stats = None
    else:
        hull_stats = compute_convex_hull_features(dend_nodes)

    # Remove axon nodes
    sk.post.prune(
        skel=skel,
        kind="nodes",
        nodes=np.where(skel.ntype == 2)[0]
    )

    # Z-profile
    z_dens, zmin, zmax = compute_density_z(skel=skel, **z_dens_kws)
    z_dict = dict(z_dens=z_dens, z_min=zmin, z_max=zmax)
    z_perc_results = compute_percentiles_from_histogram(
        bin_edges=np.linspace(zmin, zmax, z_dens.shape[0] + 1),
        frequencies=z_dens,
        percentiles=z_percentiles,
    )
    z_dict['z_percentiles'] = z_perc_results

    # XY-profile
    xy_dens, xmin, xmax, ymin, ymax = compute_density_xy(
        skel=skel, bounds=(0, 1200, 0, 1200), **xy_dens_kws)
    xy_dict = dict(xy_dens=xy_dens, x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax)
    xy_gauss_results = compute_features_xy_gauss(
        xy_dens.T, (xmin, xmax, ymin, ymax), plot_init=False, plot_fit=False)
    xy_dict['xy_gauss'] = xy_gauss_results

    gauss_xy_stats_results = compute_gauss_xy_stats(
        xy_soma=np.array(skel.soma.center[:2]),
        xy_gauss_mean=np.array([xy_gauss_results['model_params']['x_mean_um'], xy_gauss_results['model_params']['y_mean_um']]),
        xy_gauss_std=np.array([xy_gauss_results['model_params']['x_stddev_um'], xy_gauss_results['model_params']['y_stddev_um']]),
        xy_gauss_theta=xy_gauss_results['model_params']['theta']
    )
    xy_dict['xy_gauss_stats'] = gauss_xy_stats_results

    stats = {
        'morpho_stats': morpho_stats,
        'morph_dists': morph_dists,
        'hull_stats': hull_stats,
        'z_dict': z_dict,
        'xy_dict': xy_dict,
        'arbor_features': arbor_features,
    }

    return stats


def compute_arbor_features(skel: sk.Skeleton, morph=None) -> dict:
    """Flat dict of the added features (skeleton in um, flattened, axon still labelled 2):
    ``pruned_*`` (MorphoPy stats without tiny neurites; None without ``morph``),
    ``on_hull_*`` / ``off_hull_*``, ``wavy_*``, ``asym_*`` and ``coverage_*`` (from
    ``skel.extra['coverage_summary']``, set when the skeleton was warped). A group that fails
    is None and its error is listed in ``feature_errors``, so the other stats still come out."""
    out, errors = {}, {}

    def _run(name, fn):
        try:
            out.update(fn())
        except Exception as exc:  # keep the established stats even if a new feature fails
            errors[name] = f"{type(exc).__name__}: {exc}"

    if morph is not None:
        _run('pruned', lambda: compute_pruned_stats(morph))
    _run('onoff_hull', lambda: compute_onoff_hulls(skel))
    _run('waviness', lambda: compute_waviness_stats(
        nodes=np.asarray(skel.nodes, dtype=float), edges=skel.edges, ntype=skel.ntype, root=0))
    _run('asymmetry', lambda: compute_asymmetry_stats(skel))
    coverage = (skel.extra or {}).get('coverage_summary') or {f'coverage_{k}': None for k in COVERAGE_KEYS}
    out.update(coverage)
    out['feature_errors'] = errors or None
    return out
