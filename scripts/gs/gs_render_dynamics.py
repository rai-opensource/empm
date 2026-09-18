#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from third_party.gaussian_splatting.scene import Scene
from tqdm import tqdm
from os import makedirs
from third_party.gaussian_splatting.gaussian_renderer import render
import torchvision
from third_party.gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from third_party.gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from third_party.gaussian_splatting.gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import numpy as np
from kornia import create_meshgrid
import copy
from gs_render import (
    remove_gaussians_with_mask,
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from third_party.gaussian_splatting.dynamic_utils import (
    interpolate_motions,
    create_relation_matrix,
    knn_weights,
    get_topk_indices,
    quat2mat,
    mat2quat,
)
import pickle


def render_set(
    output_path,
    name,
    views,
    gaussians_list,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    disable_sh=False,
):

    render_path = os.path.join(output_path, name)
    makedirs(render_path, exist_ok=True)

    # view_indices = [0, 25, 50, 75, 100, 125]
    view_indices = [0, 50, 100]
    selected_views = [views[i] for i in view_indices]

    for idx, view in enumerate(tqdm(selected_views, desc="Rendering progress")):

        # view_idx = view_indices[idx]
        # view_render_path = os.path.join(render_path, '{0:05d}'.format(view_idx))
        view_render_path = os.path.join(render_path, f"{idx}")
        makedirs(view_render_path, exist_ok=True)

        for frame_idx, gaussians in enumerate(gaussians_list):

            if disable_sh:
                override_color = gaussians.get_features_dc.squeeze()
                results = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    override_color=override_color,
                    use_trained_exp=train_test_exp,
                    separate_sh=separate_sh,
                )
            else:
                results = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=train_test_exp,
                    separate_sh=separate_sh,
                )

            rendering = results["render"]

            torchvision.utils.save_image(
                rendering,
                os.path.join(view_render_path, "{0:05d}".format(frame_idx) + ".png"),
            )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
    remove_gaussians: bool = False,
    name: str = "dynamic",
    ctlpts_dir: str = "./experiments",
    output_dir: str = "./gaussian_output_dynamic",
):
    with torch.no_grad():
        output_path = output_dir

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # remove gaussians that are outside the mask
        if remove_gaussians:
            gaussians = remove_gaussians_with_mask(gaussians, scene.getTrainCameras())

        # remove gaussians that are low opacity
        gaussians = remove_gaussians_with_low_opacity(gaussians)

        # remove gaussians that are far from the mesh
        # gaussians = remove_gaussians_with_point_mesh_distance(gaussians, scene.mesh_sampled_points, dist_threshold=0.01)

        # rollout
        exp_name = dataset.source_path.split("/")[-1]

        ctrl_pts_path = f"{ctlpts_dir}/{exp_name}/inference_MPM.pkl"
        with open(ctrl_pts_path, "rb") as f:
            ctrl_pts = pickle.load(f)  # (n_frames, n_ctrl_pts, 3) ndarray
        ctrl_pts = torch.tensor(ctrl_pts, dtype=torch.float32, device="cuda")

        xyz_0 = gaussians.get_xyz
        rgb_0 = gaussians.get_features_dc.squeeze(1)
        quat_0 = gaussians.get_rotation
        opa_0 = gaussians.get_opacity
        scale_0 = gaussians.get_scaling

        # print(gaussians.get_features_dc.shape)   # (N, 1, 3)
        # print(gaussians.get_features_rest.shape) # (N, 15, 3)

        print("===== Number of steps: ", ctrl_pts.shape[0])
        print("===== Number of control points: ", ctrl_pts.shape[1])
        print("===== Number of gaussians: ", gaussians.get_xyz.shape[0])

        n_steps = ctrl_pts.shape[0]

        # rollout
        xyz, rgb, quat, opa = rollout(xyz_0, rgb_0, quat_0, opa_0, ctrl_pts, n_steps)

        # interpolate smoothly
        change_points = (
            (xyz - torch.cat([xyz[0:1], xyz[:-1]], dim=0))
            .norm(dim=-1)
            .sum(dim=-1)
            .nonzero()
            .squeeze(1)
        )
        change_points = torch.cat([torch.tensor([0]), change_points])
        for i in range(1, len(change_points)):
            start = change_points[i - 1]
            end = change_points[i]
            if end - start < 2:  # 0 or 1
                continue
            xyz[start:end] = torch.lerp(
                xyz[start][None],
                xyz[end][None],
                torch.linspace(0, 1, end - start + 1).to(xyz.device)[:, None, None],
            )[:-1]
            rgb[start:end] = torch.lerp(
                rgb[start][None],
                rgb[end][None],
                torch.linspace(0, 1, end - start + 1).to(rgb.device)[:, None, None],
            )[:-1]
            quat[start:end] = torch.lerp(
                quat[start][None],
                quat[end][None],
                torch.linspace(0, 1, end - start + 1).to(quat.device)[:, None, None],
            )[:-1]
            opa[start:end] = torch.lerp(
                opa[start][None],
                opa[end][None],
                torch.linspace(0, 1, end - start + 1).to(opa.device)[:, None, None],
            )[:-1]
        quat = torch.nn.functional.normalize(quat, dim=-1)

        gaussians_list = []
        for i in range(n_steps):
            gaussians_i = copy.deepcopy(gaussians)
            gaussians_i._xyz = xyz[i].to("cuda")
            gaussians_i._features_dc = rgb[i].unsqueeze(1).to("cuda")
            gaussians_i._rotation = quat[i].to("cuda")
            gaussians_i._opacity = gaussians_i.inverse_opacity_activation(opa[i]).to(
                "cuda"
            )
            gaussians_i._scaling = gaussians._scaling
            gaussians_list.append(gaussians_i)

        views = scene.getTestCameras()

        render_set(
            output_path,
            name,
            views,
            gaussians_list,
            pipeline,
            background,
            dataset.train_test_exp,
            separate_sh,
            disable_sh=dataset.disable_sh,
        )


def rollout(xyz_0, rgb_0, quat_0, opa_0, ctrl_pts, n_steps, device="cuda"):
    # store results
    xyz = xyz_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 3)
    rgb = rgb_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 3)
    quat = quat_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 4)
    opa = opa_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 1)

    # init relation matrix
    init_particle_pos = ctrl_pts[0]
    relations = get_topk_indices(init_particle_pos, K=16)

    all_pos = xyz_0
    all_rot = quat_0

    for i in tqdm(range(1, n_steps), desc="Rollout progress", dynamic_ncols=True):

        prev_particle_pos = ctrl_pts[i - 1]
        cur_particle_pos = ctrl_pts[i]

        # relations = get_topk_indices(prev_particle_pos, K=16)

        # interpolate all_pos and particle_pos
        chunk_size = 20_000
        num_chunks = (len(all_pos) + chunk_size - 1) // chunk_size
        for j in range(num_chunks):
            start = j * chunk_size
            end = min((j + 1) * chunk_size, len(all_pos))
            all_pos_chunk = all_pos[start:end]
            all_rot_chunk = all_rot[start:end]
            weights = knn_weights(prev_particle_pos, all_pos_chunk, K=16)
            all_pos_chunk, all_rot_chunk, _ = interpolate_motions(
                bones=prev_particle_pos,
                motions=cur_particle_pos - prev_particle_pos,
                relations=relations,
                weights=weights,
                xyz=all_pos_chunk,
                quat=all_rot_chunk,
            )
            all_pos[start:end] = all_pos_chunk
            all_rot[start:end] = all_rot_chunk

        quat[i] = all_rot.cpu()
        xyz[i] = all_pos.cpu()
        rgb[i] = rgb[i - 1].clone()
        opa[i] = opa[i - 1].clone()

    return xyz, rgb, quat, opa


if __name__ == "__main__":
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    import yaml
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument("--exp_config", "--exp-config", default="configs/experiments.yaml",
                        help="Path to the experiments YAML file.")
    config_args, _ = config_parser.parse_known_args()
    with open(os.path.join(REPO_ROOT, config_args.exp_config), "r") as f:
        config = yaml.safe_load(f)
    data_path = os.path.join(REPO_ROOT, config["data_path"])
    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    ctlpts_dir = f"{data_path}/experiments"
    use_white_bg = "--white_background" in sys.argv

    if use_white_bg:
        dyn_output_dir = f"{data_path}/gaussian_output_dynamic_white"
    else:
        dyn_output_dir = f"{data_path}/gaussian_output_dynamic"

    parser = ArgumentParser(description="Testing script parameters", parents=[config_parser])
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--remove_gaussians", action="store_true")
    parser.add_argument("--name", type=str, required=False)
    parser.add_argument("--ctlpts_dir", default=ctlpts_dir, type=str)
    parser.add_argument("--output_dir", default=dyn_output_dir, type=str)
    parser.add_argument("--scene", type=str, default=None,
                        help="Run a single scene by name (default: all from experiments.yaml)")

    if "--scene" in sys.argv:
        scene_idx = sys.argv.index("--scene")
        scene_name = sys.argv[scene_idx + 1]
        experiments = [{"case_name": scene_name}]
        sys.argv = [sys.argv[0]] + sys.argv[1:scene_idx] + sys.argv[scene_idx+2:]
    else:
        experiments = config["experiments"]

    extra_args = ["--white_background"] if use_white_bg else []

    from third_party.gaussian_splatting.img2video import images_to_video

    for exp in experiments:
        scene = exp["case_name"]
        print(f"\n{'='*60}\nRendering dynamics: {scene}\n{'='*60}")
        sys.argv = [sys.argv[0],
                    "-s", f"{data_path}/data/gaussian_data/{scene}",
                    "-m", f"{data_path}/gaussian_output/{scene}/{exp_name}",
                    "--name", scene,
                    "--ctlpts_dir", ctlpts_dir,
                    "--output_dir", dyn_output_dir] + extra_args
        args = get_combined_args(parser)
        safe_state(args.quiet)
        render_sets(
            model.extract(args), args.iteration, pipeline.extract(args),
            args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,
            args.remove_gaussians, args.name, args.ctlpts_dir, args.output_dir,
        )

        for view in ["0", "1", "2"]:
            img_folder = f"{dyn_output_dir}/{scene}/{view}"
            if os.path.isdir(img_folder):
                images_to_video(img_folder, f"{dyn_output_dir}/{scene}/{view}.mp4")

        print(f"Dynamic rendering complete for {scene}.")
