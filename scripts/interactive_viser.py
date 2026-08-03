"""
Interactive 3D playground for MPM simulation using viser.

Replaces the 2D OpenCV-based interactive_playground with a 3D web interface.
- Gaussian splats + particle point cloud visualization
- Transform gizmos to select and drag boundary controllers
- GUI controls for play/pause and visualization toggles

Usage:
    python scripts/test_viser.py
    Then open http://localhost:8080 in your browser.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append("./scripts/gs")

import numpy as np
import torch
import time
import yaml
import pickle
import json
import viser
import warp as wp

from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.utils import logger, cfg
from phys_sim.engine.utils import (
    get_param, inv_transform_points, transform_points,
)
from third_party.gaussian_splatting.scene.gaussian_model import GaussianModel
from third_party.gaussian_splatting.utils.sh_utils import SH2RGB
from gs_render import remove_gaussians_with_low_opacity
from third_party.gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from dataclasses import dataclass
import tyro
from scripts.utils import set_all_seeds


def load_config(case_name):
    with open("configs/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_path = config["data_path"]

    cfg.load_from_yaml("configs/mpm_config.yaml")

    base_path = f"{data_path}/data/different_types"

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        T_wc_list = pickle.load(f)
    cfg.c2ws = np.array(T_wc_list)
    cfg.w2cs = np.array([np.linalg.inv(T) for T in T_wc_list])
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        meta = json.load(f)
    cfg.intrinsics = np.array(meta["intrinsics"])
    cfg.WH = meta["WH"]
    cfg.case_name = case_name

    return data_path, base_path


def to_sim(world_pts, exp_name):
    """Convert world-space points to simulation space."""
    t = world_pts if isinstance(world_pts, torch.Tensor) else torch.tensor(
        world_pts, dtype=torch.float32)
    return transform_points(t, name=exp_name)


def to_world(sim_pts, exp_name):
    """Convert simulation-space points to world space."""
    return inv_transform_points(sim_pts, name=exp_name)


def load_gaussians_from_ply(gs_path):
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(gs_path)
    gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
    gaussians.isotropic = True
    return gaussians


def get_gs_colors(gaussians):
    features_dc = gaussians._features_dc.squeeze(1)
    return SH2RGB(features_dc).clamp(0, 1).detach().cpu().numpy()


@dataclass
class Config:
    load_gaussians: bool = True
    seq_config_path: str = "configs/experiments.yaml"


if __name__ == "__main__":
    config = tyro.cli(Config)
    
    set_all_seeds(42)

    with open(config.seq_config_path, "r") as f:
        exp_config = yaml.safe_load(f)
    case_name = exp_config["experiments"][0]["case_name"]

    data_path, base_path = load_config(case_name)
    dataset_name = data_path.split("/")[-1]
    gaussian_dir = f"{data_path}/gaussian_output"
    base_dir = f"{data_path}/experiments/{case_name}"

    # --- Trainer + MPM ---
    trainer = InvPhyTrainerWarpMPM(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    exp_name = trainer.exp_name
    params = get_param(name=exp_name)
    mpm = trainer.initialize_mpm(
        parameters=params, diff_sim=False, update_ctrl=False)

    n_ctrl = trainer.n_ctrl_parts
    if n_ctrl == 1:
        cmap = [1]
    elif n_ctrl == 2:
        cmap = [1, 2] if dataset_name == "phystwin" else [2, 1]

    # --- Gaussians ---
    if config.load_gaussians:
        gs_exp = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
        gs_path = (f"{gaussian_dir}/{case_name}/{gs_exp}"
                f"/point_cloud/iteration_500/point_cloud.ply")
        gaussians = load_gaussians_from_ply(gs_path)
        gs_pos = gaussians.get_xyz
        gs_rot = gaussians.get_rotation

    # --- Viser server ---
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    # GUI
    show_gs = server.gui.add_checkbox("Gaussians", initial_value=True)
    show_pcd = server.gui.add_checkbox("Particles", initial_value=True)
    reposition_mode = server.gui.add_checkbox(
        "Reposition Mode (move without force)", initial_value=False)
    vel_gain = server.gui.add_slider(
        "Controller gain", min=10.0, max=500.0, step=10.0, initial_value=100.0)

    # Controller gizmos in world space
    gizmos = []
    gizmo_prev_sim = []
    for i in range(n_ctrl):
        sim_pos = torch.tensor(
            trainer.ctrl_param["handler_pos"][i], dtype=torch.float32)
        world_pos = to_world(sim_pos.unsqueeze(0), exp_name).squeeze(0).numpy()
        handle = server.scene.add_transform_controls(
            f"/ctrl/controller_{i}",
            position=tuple(world_pos),
            scale=0.03,
        )
        gizmos.append(handle)
        gizmo_prev_sim.append(sim_pos.numpy().copy())

    # Show initial gaussians
    if config.load_gaussians:
        gs_colors = get_gs_colors(gaussians)
        server.scene.add_gaussian_splats(
            "/world/gaussians",
            centers=gaussians.get_xyz.detach().cpu().numpy(),
            covariances=gaussians.get_real_covariance().detach().cpu().numpy(),
            rgbs=gs_colors,
            opacities=gaussians.get_opacity.detach().cpu().numpy(),
        )

    # Show initial particles with per-point color
    init_world = to_world(trainer.obj_points[0], exp_name)
    n_pts = init_world.shape[0]
    obj_colors = trainer.dataset.original_object_colors
    if obj_colors is not None:
        # original_object_colors is (num_frames, num_original_points, 3); use frame 0
        frame0_colors = obj_colors[0].cpu().numpy() if isinstance(
            obj_colors, torch.Tensor) else np.array(obj_colors[0])
        if frame0_colors.max() > 1.0:
            frame0_colors = frame0_colors / 255.0
        n_colored = frame0_colors.shape[0]
        if n_colored < n_pts:
            extra = np.full((n_pts - n_colored, 3), [0.5, 0.5, 0.5])
            pcd_colors = np.concatenate([frame0_colors, extra], axis=0)
        else:
            pcd_colors = frame0_colors[:n_pts]
    else:
        pcd_colors = np.full((n_pts, 3), [0.3, 0.6, 1.0])
    server.scene.add_point_cloud(
        "/world/particles",
        points=init_world.cpu().numpy(),
        colors=pcd_colors,
        point_size=0.002,
        point_shape="circle",
    )

    # Motion interpolation state
    prev_x = None
    relations = None
    weights = None
    weights_indices = None

    print(f"\n  Viser running at http://localhost:8080")
    print(f"  Case: {case_name}")
    print(f"  Controllers: {n_ctrl}")
    print(f"  Simulation runs continuously. Drag gizmos to apply force.\n")

    # --- Main loop (always running) ---
    while True:
        for i in range(n_ctrl):
            cur_world = np.array(gizmos[i].position, dtype=np.float32)
            cur_sim = to_sim(
                torch.tensor(cur_world), exp_name).numpy()
            delta = cur_sim - gizmo_prev_sim[i]
            moved = np.linalg.norm(delta) > 1e-6

            if reposition_mode.value:
                if moved:
                    mpm.mpm_solver.collider_params[cmap[i]].point = wp.vec3(
                        float(cur_sim[0]), float(cur_sim[1]),
                        float(cur_sim[2]))
                mpm.mpm_solver.collider_params[cmap[i]].velocity = wp.vec3(
                    0.0, 0.0, 0.0)
            else:
                if moved:
                    vel = delta * vel_gain.value
                    mpm.mpm_solver.collider_params[cmap[i]].velocity = wp.vec3(
                        float(vel[0]), float(vel[1]), float(vel[2]))
                else:
                    mpm.mpm_solver.collider_params[cmap[i]].velocity = wp.vec3(
                        0.0, 0.0, 0.0)
            gizmo_prev_sim[i] = cur_sim

        # Step MPM
        vertices = mpm.advance_to(n_frames=1)
        sim_pts = vertices[-1]
        world_pts = to_world(sim_pts, exp_name)

        # Update particle point cloud
        if show_pcd.value:
            server.scene.add_point_cloud(
                "/world/particles",
                points=world_pts.cpu().numpy(),
                colors=pcd_colors,
                point_size=0.002,
                point_shape="circle",
            )

        # Update gaussians via motion interpolation
        if config.load_gaussians and prev_x is not None and show_gs.value:
            with torch.no_grad():
                if relations is None:
                    relations = get_topk_indices(prev_x, K=16)
                if weights is None:
                    weights, weights_indices = knn_weights_sparse(
                        prev_x, gs_pos, K=16)

                weights = calc_weights_vals_from_indices(
                    prev_x, gs_pos, weights_indices)

                gs_pos, gs_rot, _ = interpolate_motions_speedup(
                    bones=prev_x,
                    motions=world_pts - prev_x,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=gs_pos,
                    quat=gs_rot,
                )
                gaussians._xyz = gs_pos
                gaussians._rotation = gs_rot

            if config.load_gaussians:
                gs_colors = get_gs_colors(gaussians)
                server.scene.add_gaussian_splats(
                    "/world/gaussians",
                    centers=gaussians.get_xyz.detach().cpu().numpy(),
                    covariances=gaussians.get_real_covariance(
                    ).detach().cpu().numpy(),
                    rgbs=gs_colors,
                    opacities=gaussians.get_opacity.detach().cpu().numpy(),
                )

        prev_x = world_pts.clone()
        torch.cuda.synchronize()
