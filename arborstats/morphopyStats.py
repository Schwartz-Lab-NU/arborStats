from types import MappingProxyType
from copy import deepcopy
import numpy as np
import io
import re
from contextlib import redirect_stdout
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
    """
    If the user has no token, show CaveClient’s instructions *minus* steps 3a/3b
    and inject a Flatone-specific step 3.  Exits the program afterward.
    """
    if client.auth.token is not None:
        return

    print("No CAVEclient token found.\n")
    # Capture CaveClient’s standard instructions without hard-coding them
    buf = io.StringIO()
    with redirect_stdout(buf):
        client.auth.get_new_token()
    cave_text = buf.getvalue().splitlines()

    # Drop CaveClient’s “3a” and “3b” lines
    filtered = [
            ln for ln in cave_text
            if not re.match(r'\s*3[ab]\)', ln)          # remove 3a/3b
            and not re.match(r'\s*or\s*$', ln, re.I)    # remove “or”
        ]
    
    # Insert our own single step 3 right after step 2
    for idx, ln in enumerate(filtered):
        if re.match(r'\s*2\)', ln):
            filtered.insert(idx + 1,
                "                3) Add it to `flatone` with:"
            )
            filtered.insert(idx + 2,
                "                       flatone add-token <TOKEN>"
            )
            break

    # Re-print the modified message and quit
    print("\n".join(filtered))
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
        mesh = cell
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

    results = {
        'mesh': mesh,
        'skel': skel,
        'skel_raw': skel_raw.convert_unit(target_unit="μm"),
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
    }

    return stats
