import trimesh
from skeliner import Skeleton

from .postprocess import (
    detect_soma_and_connect_skeleton,
    warp_skeleton,
    class_postprocess_skeleton,
    calibrate_radii,
    reroot_cellclass_based,
)


def finalize_skel(
        skel: Skeleton,
        mesh: trimesh.Trimesh,
        *,
        global_map: dict,
        cellclass: str = None,
        ais_xyz_um: tuple[float, float, float] | None = None,
        set_ntypes: bool = True,
        run_stage_radii: bool = True,
        run_stage_soma: bool = True,
        run_stage_warp: bool = True,
        run_stage_class: bool = True,
        run_stage_reroot: bool = True,
        verbose: bool = False,
) -> tuple[Skeleton, dict]:
    post_info = {"cellclass": cellclass}

    if run_stage_soma:
        if verbose:
            print("[finalize_skel] Finalizing skeleton - detect_soma_and_connect_skeleton...")

        skel.convert_unit(target_unit='nm')
        skel, post_info_soma = detect_soma_and_connect_skeleton(
            skel=skel,
            mesh=mesh,
            verbose=verbose,
        )
        post_info['soma'] = post_info_soma

    if set_ntypes:  # Set root to soma and everything else to dendrites
        skel.ntype[0] = 1
        skel.ntype[1:] = 3

    if run_stage_radii:
        if verbose:
            print("[finalize_skel] Finalizing skeleton - calibrate_radii...")
        calibrate_radii(skel=skel, mesh=mesh, verbose=verbose)

        post_info['radii'] = 'calibrated'

    if run_stage_warp:
        if verbose:
            print("[finalize_skel] Finalizing skeleton - warp_skeleton...")

        skel.convert_unit(target_unit='um')
        skel, post_info_warp = warp_skeleton(
            skel=skel,
            surface_mapping=global_map,
            verbose=verbose,
        )
        post_info['warp'] = post_info_warp

    if run_stage_class:
        if verbose:
            print("[finalize_skel] Finalizing skeleton - class_postprocess_skeleton...")

        skel.convert_unit(target_unit='um')
        skel, post_info_class = class_postprocess_skeleton(
            skel=skel,
            cellclass=cellclass,
            ais_xyz_um=ais_xyz_um,
            verbose=verbose,
        )
        post_info['class'] = post_info_class

    if run_stage_reroot:
        assert run_stage_warp, "Rerooting requires warped skeleton."
        if verbose:
            print("[finalize_skel] Finalizing skeleton - rerooting based on soma presence...")
        skel.convert_unit(target_unit='um')
        skel = reroot_cellclass_based(skel, cellclass)

    skel.convert_unit(target_unit='um')
    return skel, post_info
