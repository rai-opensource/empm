from phys_sim.data_loaders import RealData, SimpleData
from phys_sim.utils import logger, visualize_pc, cfg
from phys_sim.diff_simulator import WarpMPMWrapper
import open3d as o3d
import numpy as np
import torch
import os
from tqdm import tqdm
import warp as wp
from scipy.spatial import KDTree
import pickle
import cv2
from pynput import keyboard
import pyrender
import trimesh
import matplotlib.pyplot as plt
import pyvista as pv
from pytorch3d.loss import chamfer_distance

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
# 2025-12, YH, add gs_render import
import sys
sys.path.append("./scripts/gs")
# 2025-12, YH, add gs_render import
from gs_render import (
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from gaussian_splatting.rotation_utils import quaternion_multiply, matrix_to_quaternion

from sklearn.cluster import KMeans
import copy
import time
import threading
import time
from .utils import transform_points, inv_transform_points, modify_handler, initial_param_guess, get_param
from phys_sim.diff_simulator.warp_mpm.warp_mpm_wrapper import save_ply_seq, save_ply_file, material_gallery

import sys
import yaml
sys.path.append("../..")

with open("configs/experiments.yaml", "r") as f:
    config = yaml.safe_load(f)
    DATASET = config["data_path"].split("/")[-1]

# 2025-09, YH, add viser visualization
import viser
import viser.transforms as tf

viser_server = viser.ViserServer()


class InvPhyTrainerWarpMPM:

    def __init__(
        self,
        data_path,
        base_dir,
        train_frame=None,
        mask_path=None,
        velocity_path=None,
        pure_inference_mode=False,
        device="cuda:0",
        online=False,
    ):
        if not online:
            self.init_offline(
                data_path,
                base_dir,
                train_frame,
                mask_path,
                velocity_path,
                pure_inference_mode,
                device,
            )
        else:
            self.init_online(data_path, base_dir, device)

    def init_offline(self,
                     data_path,
                     base_dir,
                     train_frame=None,
                     mask_path=None,
                     velocity_path=None,
                     pure_inference_mode=False,
                     device="cuda:0"):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=True)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            first_frame_controller_points = None
        else:
            first_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            first_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        if not pure_inference_mode:
            if not os.path.exists(f"{cfg.base_dir}/train"):
                # Create directory if it doesn't exist
                os.makedirs(f"{cfg.base_dir}/train")
        """ [MPM]
        """
        self.vis_path = "outputs"
        os.makedirs(f"{self.vis_path}/sim", exist_ok=True)
        os.makedirs(f"{self.vis_path}/optim", exist_ok=True)
        self.exp_name = cfg.case_name

        # [TODO]:
        self.ctrl_points = transform_points(self.controller_points.clone(),
                                            name=self.exp_name)
        self.obj_points = transform_points(self.object_points.clone(),
                                           name=self.exp_name)
        # [TODO]: handlers
        self.n_ctrl_parts = 2 if "double" in cfg.case_name else 1 if "single" in cfg.case_name else 0
        ctrl_index = []
        handler_pos = []  # handler center position
        if self.n_ctrl_parts > 0:
            ctrl_y = self.ctrl_points[0, :, 1]
            y_mid = (torch.min(ctrl_y) + torch.max(ctrl_y)) / 2
            ctrl_mask = (self.ctrl_points[0, :, 1] < y_mid)
            ctrl_pts = self.ctrl_points[0].clone()
            obj_pts = self.obj_points[0]
            dists = torch.cdist(ctrl_pts, obj_pts, p=2)
            for i in range(self.n_ctrl_parts):
                mask = (ctrl_mask == i)
                m_dists = dists.clone()
                m_dists[~mask, :] = 1e6
                # closest control point to object
                closest_ctrl_pts = torch.argmin(m_dists, dim=0)
                ctrl_idx = torch.bincount(closest_ctrl_pts).argmax().item()
                ctrl_index.append(ctrl_idx)
                # closest object point to controller
                # m_dists = m_dists[mask]
                # closest_cobj_pts = torch.argmin(m_dists, dim=1)
                # obj_idx = torch.bincount(closest_cobj_pts).argmax().item()
                m_dists = m_dists[ctrl_idx]
                obj_idx = torch.argmin(m_dists).item()
                handler_pos.append(obj_pts[obj_idx].cpu().numpy())
        self.ctrl_param = {
            "ctrl_index": ctrl_index,
            "handler_pos": handler_pos,
            "vscale": [[100.0, 100.0, 100.0] for _ in range(self.n_ctrl_parts)
                       ],  # controller speed (default: 100.0)
            "csize": [0.015 for _ in range(self.n_ctrl_parts)
                      ],  # controller size (default: 0.015)
            "ctrl_mask": ctrl_mask
        }
        if False:  # Deubugging
            for i in range(self.ctrl_points.shape[1]):
                print(i, self.ctrl_points[0, i], ctrl_mask[i])

        save_ply_seq(self.ctrl_points[:], f"{self.vis_path}/sim", "ctrl_")
        save_ply_seq(self.obj_points, f"{self.vis_path}/sim", "obj_")

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx,
                 _] = pcd_tree.search_hybrid_vector_3d(points[i],
                                                       object_radius,
                                                       object_max_neighbours)
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (spring_flags[i, j] == 0 and spring_flags[j, i] == 0
                            and rest_length > 1e-4):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(
                            np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j]))

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths,
                             dtype=torch.float32,
                             device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros(
                    (len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours)
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] -
                                                     temp_points[j])
                        if (temp_spring_flags[i, j] == 0
                                and temp_spring_flags[j, i] == 0
                                and rest_length > 1e-4):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths,
                             dtype=torch.float32,
                             device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def denormalize_param(self, param):
        new_param = {
            "type": "jelly",
            "E": 5e4,
            "nu": 0.3,
            "rho": 2e3,
            "ys": 1e5,
            "vol": 1e-5
        }
        for key, value in param.items():
            if key == "E":
                new_param[key] = value * 1e6
            # elif key == "rho":
            #     new_param[key] = value * 1e3
            else:
                new_param[key] = value
        return new_param

    def initialize_mpm(self, parameters, diff_sim=False, update_ctrl=False):
        mpm = self.initialize_mpm_test(parameters,
                                       diff_sim=diff_sim,
                                       update_ctrl=update_ctrl)
        if mpm is not None:
            return mpm

        # initialize parameters
        parameters = self.denormalize_param(parameters)
        g_E = parameters["E"]
        g_nu = parameters["nu"]
        g_rho = parameters["rho"]
        g_ys = parameters["ys"]
        g_vol = parameters["vol"]

        init_x = self.object_points[0].clone()
        # init_x = self.structure_points.clone()
        init_x = transform_points(x=init_x, name=self.exp_name)
        if init_x[:, 2].min() < 0.5 + 1e-4:
            init_x[:, 2] += 0.001  # lift up a bit to avoid initial penetration

        mpm = WarpMPMWrapper(requires_grad=diff_sim)

        # material
        mpm.material_params = material_gallery[parameters["type"]]
        if "y_s" in parameters:
            mpm.material_params["yield_stress"] = g_ys
        mpm.material_params["grid_v_damping_scale"] = 0.98

        mpm.add_particles(init_x, None, g_vol, g_E, g_nu, g_rho)

        # boundary conditions
        # ground
        mpm.surface_colliders.append([(0., 0., 0.5), (0., 0., 1.), "sticky",
                                      0.])
        # controllers
        ctrl_config_file = f"{cfg.base_dir}/ctrl_param.pkl"
        if os.path.exists(ctrl_config_file):
            with open(ctrl_config_file, "rb") as f:
                self.ctrl_param = pickle.load(f)
        for i in range(self.n_ctrl_parts):
            csize = self.ctrl_param["csize"][i]
            point = self.ctrl_param["handler_pos"][i]
            # print("Fixed point:", point)
            mpm.moving_boundaries.append([point, csize, (0., 0., 0.), 0.0])

        # training/inference: update controller position
        def sim_callback(mpm, f):
            if (f < self.ctrl_points.shape[0]):
                indices = self.ctrl_param["ctrl_index"]
                for i in range(self.n_ctrl_parts):
                    if indices[i] == -1:
                        mask = (self.ctrl_param["ctrl_mask"] == i)
                        vel = (self.ctrl_points[f + 1, mask, :].mean(dim=0) -
                               self.ctrl_points[f, mask, :].mean(dim=0))
                    else:
                        vel = (self.ctrl_points[f + 1, indices[i], :] -
                               self.ctrl_points[f, indices[i], :])
                    vscale = self.ctrl_param["vscale"][i]
                    for j in range(3):
                        vel[j] *= vscale[j]
                    mpm.mpm_solver.collider_params[i +
                                                   1].velocity = wp.vec3(vel)

        if update_ctrl:
            mpm.frame_callback = sim_callback

        mpm.initialize()
        return mpm

    def initialize_mpm_test(self,
                            parameters,
                            diff_sim=False,
                            update_ctrl=False):
        """
            Test initialization for custom data; to be removed later
        """
        exp_name = self.exp_name
        if not exp_name in [
                "double_tear_pita_test", "single_poke_bluey_test"
        ]:
            return None

        print(f"Use the special initialization for {exp_name}")

        # initialize parameters
        parameters = self.denormalize_param(parameters)
        print("Initializing MPM with:", parameters)
        g_E = parameters["E"]
        g_nu = parameters["nu"]
        g_rho = parameters["rho"]
        g_ys = parameters["ys"]
        g_vol = parameters["vol"]

        from .utils import densify_point_cloud
        init_x = self.structure_points.clone()
        # init_x = densify_point_cloud(init_x, K=1000)
        init_x = transform_points(x=init_x, name=self.exp_name)
        if init_x[:, 2].min() < 0.5 + 1e-4:
            init_x[:, 2] += 0.001  # lift up a bit to avoid initial penetration

        if self.exp_name == "double_tear_pita_test":
            self.obj_mask = [(init_x[:, 1] > 0.45) & (init_x[:, 0] < 0.57),
                             (init_x[:, 1] < 0.45) & (init_x[:, 0] < 0.57)]
            init_x[self.obj_mask[0], 1] += 0.0015
            init_x[self.obj_mask[1], 1] -= 0.0015

        mpm = WarpMPMWrapper(requires_grad=diff_sim)

        # material
        mpm.material_params = material_gallery[parameters["type"]]
        if "y_s" in parameters:
            mpm.material_params["yield_stress"] = g_ys

        mpm.add_particles(init_x, None, g_vol, g_E, g_nu, g_rho)

        # boundary conditions
        # ground
        mpm.surface_colliders.append([(0., 0., 0.5), (0., 0., 1.), "sticky",
                                      0.])

        # controllers
        ctrl_config_file = f"{cfg.base_dir}/ctrl_param.pkl"
        with open(ctrl_config_file, "wb") as f:
            pickle.dump(self.ctrl_param, f)
        for i in range(self.n_ctrl_parts):
            csize = self.ctrl_param["csize"][i]
            point = self.ctrl_param["handler_pos"][i]
            # print("Fixed point:", point)
            mpm.moving_boundaries.append([point, csize, (0., 0., 0.), 0.0])

        # training/inference: update controller position
        def sim_callback(mpm, f):
            if (f + 1 < self.ctrl_points.shape[0]):
                indices = self.ctrl_param["ctrl_index"]
                for i in range(self.n_ctrl_parts):
                    mask = (self.ctrl_param["ctrl_mask"] == i)
                    vel = (self.ctrl_points[f + 1, mask, :].mean(dim=0) -
                           self.ctrl_points[f, mask, :].mean(dim=0))
                    # vel = (self.ctrl_points[f + 1, indices[i], :] -
                    #        self.ctrl_points[f, indices[i], :])
                    vscale = self.ctrl_param["vscale"][i]
                    for j in range(3):
                        vel[j] *= vscale[j]
                    mpm.mpm_solver.collider_params[i +
                                                   1].velocity = wp.vec3(vel)

        if update_ctrl:
            mpm.frame_callback = sim_callback

        if self.exp_name == "single_poke_bluey_test":
            poker = mpm.add_grid_geometry_mover(
                geo_args={
                    "shape": "cube",
                    "center": self.ctrl_param["handler_pos"][0],
                    "size": [0.005, 0.005, 0.02],
                    "velocity": [0., 0., 0.]
                })

            def sim_callback_2(mpm, f):
                ip = 2  # poker
                if (f + 1 < self.ctrl_points.shape[0]):
                    vel = (self.ctrl_points[f + 1].mean(dim=0) -
                           self.ctrl_points[f].mean(dim=0)) * 100.
                vel[2] *= 1.75
                mpm.mpm_solver.collider_params[ip].velocity = wp.vec3(vel)
                if (f >= 220):
                    mpm.mpm_solver.collider_params[ip].active = False

                if f <= 160:
                    pos = [0.0, 0.0, 0.511]
                else:
                    pos = [0, 0, max(0.511 - (f - 160) * 0.0001, 0.509)]
                mpm.mpm_solver.collider_params[0].point = wp.vec3(pos)

            mpm.frame_callback = sim_callback_2

        mpm.initialize()
        return mpm

    def visualize_forward_mpm(self, parameters, video_path, track_path=None):
        """ visualize the forward simulation using MPM with given parameters
        """
        mpm = self.initialize_mpm(parameters, diff_sim=False, update_ctrl=True)
        # mpm.advance_to_static(n_frames=10)
        n_frames = self.object_points.shape[0] - 1
        vertices = mpm.advance_to(n_frames=n_frames)
        if self.exp_name == "double_tear_pita_test":
            vertices[:, self.obj_mask[0], 1] -= 0.0015
            vertices[:, self.obj_mask[1], 1] += 0.0015
        save_ply_seq(vertices[:], f"{self.vis_path}/sim", "test_")
        vertices = inv_transform_points(x=vertices, name=self.exp_name)

        if True:
            visualize_pc(
                vertices[:, :self.num_original_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

        if track_path is not None:
            logger.info(f"Save the trajectory to {track_path}")
            vertices_to_save = vertices.cpu().numpy()
            with open(track_path, "wb") as f:
                pickle.dump(vertices_to_save, f)

        if True:  # print loss in forward sim
            vertices = vertices[:, :self.num_original_points]

            total_loss_chamfer = 0.0
            for t in range(self.object_points.shape[0]):
                # Chamfer loss for visible points
                vis_mask = self.object_visibilities[t]
                pts_pred_vis = vertices[t][vis_mask].unsqueeze(0)
                pts_target_vis = self.object_points[t][vis_mask].unsqueeze(0)
                loss_chamfer, _ = chamfer_distance(pts_pred_vis,
                                                   pts_target_vis)
                total_loss_chamfer += loss_chamfer

            total_loss_tracking = 0.0
            for t in range(self.object_points.shape[0]):
                # Tracking loss for valid motion points
                valid_mask = self.object_motions_valid[t]
                pts_pred_valid = vertices[t][valid_mask]
                pts_target_valid = self.object_points[t][valid_mask]
                if pts_pred_valid.shape[0] > 0:
                    loss_tracking = (pts_pred_valid - pts_target_valid
                                     ).norm() / pts_pred_valid.shape[0]
                    total_loss_tracking += loss_tracking

            print(
                f"[Inference Time]:\n Chamfer loss: {total_loss_chamfer * 1e3:.4f} \n Tracking loss: {total_loss_tracking * 1e3:.4f}"
            )

    def train_cma(self, max_iter=20):
        """
            Optimize the physical parameters using CMA-ES
        """
        # [TODO]: clean up
        import cma
        os.makedirs(f"{cfg.base_dir}/optimizeCMA", exist_ok=True)

        losses = []
        n_train_frames = cfg.train_frame
        target_points = self.object_points[:n_train_frames + 1]
        target_points = transform_points(target_points, self.exp_name)
        save_ply_file(target_points[-1], f"{self.vis_path}/optim/gt_final.ply")

        # Parameters to optimize
        params = initial_param_guess(name=self.exp_name)

        self.visualize_forward_mpm(
            parameters=params,
            video_path=f"{cfg.base_dir}/optimizeCMA/init_MPM.mp4")

        # --- Run CMA-ES optimization ---
        # cma requires a list input of parameters
        param_keys = ["E", "nu"]
        init_param_list = [params[key] for key in param_keys]
        std = 1 / 6
        es = cma.CMAEvolutionStrategy(init_param_list, std, {
            "bounds": [0.0, 1.0],
            "seed": 42
        })

        def error_func_cma(param_list):
            exp_name = self.exp_name
            n_train_frames = cfg.train_frame
            target_points = self.object_points[:n_train_frames + 1]

            # Parameters to optimize
            for i in range(len(param_list)):
                params[param_keys[i]] = param_list[i]
            mpm = self.initialize_mpm(parameters=params,
                                      diff_sim=False,
                                      update_ctrl=True)

            vertices = mpm.advance_to(n_frames=n_train_frames)
            self.current_vertices = vertices[-1].cpu().numpy()
            vertices = inv_transform_points(vertices, name=exp_name)

            # compute loss
            points = vertices[:]

            def loss_func(points):
                points = points[:, :self.num_original_points, :]

                total_loss_chamfer = 0.0
                for t in range(n_train_frames):
                    # Chamfer loss for visible points
                    vis_mask = self.object_visibilities[t]
                    pts_pred_vis = points[t][vis_mask].unsqueeze(0)
                    pts_target_vis = self.object_points[t][vis_mask].unsqueeze(
                        0)
                    loss_chamfer, _ = chamfer_distance(pts_pred_vis,
                                                       pts_target_vis)
                    total_loss_chamfer += loss_chamfer

                total_loss_tracking = 0.0
                for t in range(n_train_frames):
                    # Tracking loss for valid motion points
                    valid_mask = self.object_motions_valid[t]
                    pts_pred_valid = points[t][valid_mask]
                    pts_target_valid = target_points[t][valid_mask]
                    loss_tracking = (pts_pred_valid - pts_target_valid).norm()
                    total_loss_tracking += loss_tracking / pts_pred_valid.shape[
                        0]

                loss = (total_loss_chamfer + total_loss_tracking) * 1e3
                return loss

            loss = loss_func(points).item()

            return loss

        def iteration_callback(es):
            iter_num = es.countiter
            best_param = es.best.x
            best_error = es.best.f
            # print(
            #     f"Iteration {iter_num}: best param = {best_param}, best error = {best_error}"
            # )
            losses.append(best_error)
            plt.plot(losses)
            plt.savefig(f"{self.vis_path}/loss_plot.png")
            save_ply_file(self.current_vertices,
                          f"{self.vis_path}/optim/optim_{iter_num:04d}.ply")

        es.optimize(error_func_cma,
                    iterations=max_iter,
                    callback=iteration_callback)

        res = es.result
        optimal_param_list = res[0]
        optimal_error = res[1]
        logger.info(
            f"Optimal x: {optimal_param_list}, Optimal error: {optimal_error}")

        for i in range(len(param_keys)):
            params[param_keys[i]] = optimal_param_list[i]

        print(params)

        self.visualize_forward_mpm(
            parameters=params,
            video_path=f"{cfg.base_dir}/optimizeCMA/optimal_MPM_cma.mp4")

        torch.save(params, f"{cfg.base_dir}/optimizeCMA/optimal_param.pt")

    def train(self, max_iter=50):
        """ [MPM]
            MS optimizable params: cfg.init_spring_Y, cfg.collide_elas, cfg.collide_fric, cfg.collide_object_elas, cfg.collide_object_fric
            ['controller_mask', 'controller_points', 'object_points', 'object_colors', 'object_visibilities', 'object_motions_valid', 'surface_points', 'interior_points']
            structure_points(6405) = object_points(4895=num_original_points) + other_surface_points(+538=num_surface_points) + interior_points(+972=num_all_points)
        """
        n_train_frames = cfg.train_frame
        n_train_frames = min(n_train_frames, 35)
        n_track_points = self.num_original_points
        print("--train_frames", n_train_frames)
        print("--num_tracked_points", n_track_points)

        params = initial_param_guess(name=self.exp_name)
        self.visualize_forward_mpm(
            parameters=params, video_path=f"{cfg.base_dir}/train/init_MPM.mp4")

        mpm = self.initialize_mpm(params, diff_sim=True, update_ctrl=True)

        # mpm.advance_to_static(n_frames=10)
        # mpm.x_t = wp.to_torch(mpm.mpm_state.particle_x, requires_grad=False)

        # ------ train step callback -----
        def iter_callback(mpm):
            # [TODO]: re-initialization required per iteration
            mpm.mpm_solver.time = 0.0
            for i in range(self.n_ctrl_parts):
                point = self.ctrl_param["handler_pos"][i]
                mpm.mpm_solver.collider_params[i + 1].point = wp.vec3(*point)

        mpm.optim_callback = iter_callback

        # current workaround for the callback in training
        def step_callback(step):
            f = step // int(mpm.frame_dt / mpm.sim_dt)
            mpm.frame_callback(mpm, f)

        mpm.mpm_solver.step_callback = step_callback

        # ------ define loss func -----
        # object_visibilities = self.object_visibilities[:n_train_frames]
        # object_motions_valid = self.object_motions_valid[:n_train_frames]
        target_points = transform_points(x=self.object_points[:n_train_frames],
                                         name=self.exp_name)
        save_ply_file(target_points[-1], f"{self.vis_path}/optim/gt_final.ply")

        def loss_func(points):
            points = points[:, :n_track_points, :]
            # loss, _ = chamfer_distance(points, target_points)
            # loss = (points - target_points).norm()

            # # chamfer loss (only consider visible points)
            # points_valid = points[object_visibilities]
            # target_points_valid = target_points[object_visibilities]
            # loss_chamfer, _ = chamfer_distance(
            #     points_valid.unsqueeze(0), target_points_valid.unsqueeze(0))

            # # tracking loss (only consider points with valid motions)
            # points_valid = points[object_motions_valid]
            # target_points_valid = target_points[object_motions_valid]
            # loss = (points_valid - target_points_valid).norm()

            # loss = loss_chamfer + loss_tracking

            total_loss_chamfer = 0.0
            for t in range(n_train_frames):
                # Chamfer loss for visible points
                vis_mask = self.object_visibilities[t]
                pts_pred_vis = points[t][vis_mask].unsqueeze(0)
                pts_target_vis = target_points[t][vis_mask].unsqueeze(0)
                loss_chamfer, _ = chamfer_distance(pts_pred_vis,
                                                   pts_target_vis)
                total_loss_chamfer += loss_chamfer

            total_loss_tracking = 0.0
            for t in range(n_train_frames):
                # Tracking loss for valid motion points
                valid_mask = self.object_motions_valid[t]
                pts_pred_valid = points[t][valid_mask]
                pts_target_valid = target_points[t][valid_mask]
                loss_tracking = (pts_pred_valid - pts_target_valid).norm()
                total_loss_tracking += loss_tracking / pts_pred_valid.shape[0]

            # loss = total_loss_tracking * 1e3
            loss = total_loss_chamfer * 1e3 + total_loss_tracking * 1e3
            return loss

        # ------ run optimization -----
        logger.info(f"Initial parameters: {params}")
        optimal_param = mpm.run_diff_optimization(
            parameters=params,
            n_train_frames=n_train_frames,
            max_iters=max_iter,
            loss_func=loss_func)
        logger.info(f"Optimal parameters: {optimal_param}")
        for key in optimal_param.keys():
            params[key] = optimal_param[key]

        self.visualize_forward_mpm(
            parameters=params,
            video_path=f"{cfg.base_dir}/train/optimal_MPM.mp4")

        torch.save(params, f"{cfg.base_dir}/train/optimal_param.pt")

    def test(self, model_path=None):
        # model_path = f"{cfg.base_dir}/optimizeCMA/optimal_param.pt"
        # model_path = f"{cfg.base_dir}/train/optimal_param.pt"
        model_path = f"{cfg.base_dir}/best_model_param.pt"
        if model_path is not None and os.path.exists(model_path):
            param = torch.load(model_path)
            print(f"Load the model from {model_path}")
            print(param)
        else:
            param = get_param(name=self.exp_name)
            torch.save(param, f"{cfg.base_dir}/best_model_param.pt")
            print(f"Use the default model {param}")

        video_path = f"{cfg.base_dir}/inference_MPM.mp4"
        save_path = f"{cfg.base_dir}/inference_MPM.pkl"
        self.visualize_forward_mpm(parameters=param,
                                   video_path=video_path,
                                   track_path=save_path)

    def visualize_sim(self,
                      save_only=True,
                      video_path=None,
                      save_trajectory=False,
                      save_path=None):
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(self.simulator.wp_init_vertices,
                                      self.simulator.wp_init_velocities)
        vertices = [
            wp.to_torch(self.simulator.wp_states[0].wp_x,
                        requires_grad=False).cpu()
        ]

        with wp.ScopedTimer("simulate"):
            for i in tqdm(range(1, frame_len)):
                if cfg.data_type == "real":
                    self.simulator.set_controller_target(i,
                                                         pure_inference=True)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(self.simulator.forward_graph)
                else:
                    self.simulator.step()
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x,
                                requires_grad=False)
                vertices.append(x.cpu())
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

        vertices = torch.stack(vertices, dim=0)

        if save_trajectory:
            logger.info(f"Save the trajectory to {save_path}")
            vertices_to_save = vertices.cpu().numpy()
            with open(save_path, "wb") as f:
                pickle.dump(vertices_to_save, f)

        if not save_only:
            visualize_pc(
                vertices[:, :self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=True,
            )
        else:
            assert video_path is not None, "Please provide the video path to save"
            visualize_pc(
                vertices[:, :self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

    def on_press(self, key):
        try:
            # Add physical keyboard key to pressed_keys set
            key_char = key.char.lower()  # Normalize to lowercase
            if key_char in self.key_mappings:
                self.pressed_keys.add(key_char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            # Remove physical keyboard key from pressed_keys set
            key_char = key.char.lower()  # Normalize to lowercase
            if key_char in self.pressed_keys:
                self.pressed_keys.remove(key_char)
        except (KeyError, AttributeError):
            try:
                key_str = str(key).lower()
                if key_str in self.pressed_keys:
                    self.pressed_keys.remove(key_str)
            except KeyError:
                pass

    def get_target_change(self):
        target_change = np.zeros((self.n_ctrl_parts, 3))
        # Process all currently active keys (both physical and virtual)
        for key in self.pressed_keys:
            if key in self.key_mappings:
                idx, change = self.key_mappings[key]
                target_change[idx] += change
        return target_change

    def init_control_ui(self):

        height = cfg.WH[1]
        width = cfg.WH[0]

        self.arrow_size = 30

        self.arrow_empty_orig = cv2.imread("./assets/arrow_empty.png",
                                           cv2.IMREAD_UNCHANGED)[:, :,
                                                                 [2, 1, 0, 3]]
        self.arrow_1_orig = cv2.imread("./assets/arrow_1.png",
                                       cv2.IMREAD_UNCHANGED)[:, :,
                                                             [2, 1, 0, 3]]
        self.arrow_2_orig = cv2.imread("./assets/arrow_2.png",
                                       cv2.IMREAD_UNCHANGED)[:, :,
                                                             [2, 1, 0, 3]]

        spacing = self.arrow_size + 5

        self.bottom_margin = 25  # Margin from bottom of screen
        bottom_y = height - self.bottom_margin
        top_y = height - self.bottom_margin - spacing

        self.edge_buffer = self.bottom_margin
        set1_margin_x = self.edge_buffer  # Add buffer from left edge
        set2_margin_x = width - self.edge_buffer

        self.arrow_positions_set1 = {
            "q": (set1_margin_x + spacing * 3, top_y),  # Up
            "w": (set1_margin_x + spacing, top_y),  # Forward
            "a": (set1_margin_x, bottom_y),  # Left
            "s": (set1_margin_x + spacing, bottom_y),  # Backward
            "d": (set1_margin_x + spacing * 2, bottom_y),  # Right
            "e": (set1_margin_x + spacing * 3, bottom_y),  # Down
        }

        self.arrow_positions_set2 = {
            "u": (set2_margin_x - spacing * 3, top_y),  # Up
            "i": (set2_margin_x - spacing * 1, top_y),  # Forward
            "j": (set2_margin_x - spacing * 2, bottom_y),  # Left
            "k": (set2_margin_x - spacing * 1, bottom_y),  # Backward
            "l": (set2_margin_x, bottom_y),  # Right
            "o": (set2_margin_x - spacing * 3, bottom_y),  # Down
        }

        self.interm_size = 512
        self.rotations = {
            "w":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0,
                1),  # Forward
            "a":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1),  # Left
            "s":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180,
                1),  # Backward
            "d":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270,
                1),  # Right
            "q":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1),  # Up
            "e":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180,
                1),  # Down
            "i":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0,
                1),  # Forward
            "j":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1),  # Left
            "k":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180,
                1),  # Backward
            "l":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270,
                1),  # Right
            "u":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1),  # Up
            "o":
            cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180,
                1),  # Down
        }

        self.hand_left = cv2.imread("./assets/Picture2.png",
                                    cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0, 3]]
        self.hand_right = cv2.imread("./assets/Picture1.png",
                                     cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0, 3]]

        self.hand_left_pos = torch.tensor([0.0, 0.0, 0.0], device=cfg.device)
        self.hand_right_pos = torch.tensor([0.0, 0.0, 0.0], device=cfg.device)

        # pre-compute all rotated arrows to avoid aliasing
        self.arrow_rotated_filled = {}
        self.arrow_rotated_empty = {}
        for key in self.arrow_positions_set1:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_1_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_empty_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
        for key in self.arrow_positions_set2:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_2_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_empty_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )

    def _rotate_arrow(self, arrow, key):
        rotation_matrix = self.rotations[key]
        rotated = cv2.warpAffine(
            arrow,
            rotation_matrix,
            (self.interm_size, self.interm_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        return rotated

    def _overlay_arrow(self, background, arrow, position, key, filled=True):
        x, y = position

        if filled:
            rotated_arrow = self.arrow_rotated_filled[key].copy()
        else:
            rotated_arrow = self.arrow_rotated_empty[key].copy()

        h, w = rotated_arrow.shape[:2]

        roi_x = max(0, x - w // 2)
        roi_y = max(0, y - h // 2)
        roi_w = min(w, background.shape[1] - roi_x)
        roi_h = min(h, background.shape[0] - roi_y)

        arrow_x = max(0, w // 2 - x)
        arrow_y = max(0, h // 2 - y)

        roi = background[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        arrow_roi = rotated_arrow[arrow_y:arrow_y + roi_h,
                                  arrow_x:arrow_x + roi_w]

        alpha = arrow_roi[:, :, 3] / 255.0

        for c in range(3):  # Apply for RGB channels
            roi[:, :,
                c] = roi[:, :, c] * (1 - alpha) + arrow_roi[:, :, c] * alpha

        background[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi

        return background

    def _overlay_hand_at_position(self,
                                  frame,
                                  target_points,
                                  x_axis,
                                  hand_size,
                                  hand_icon,
                                  align="center"):
        result = frame.copy()

        mean_pos = target_points.cpu().numpy().mean(axis=0)

        pixel_mean = self.projection @ np.append(mean_pos, 1)
        pixel_mean = pixel_mean[:2] / pixel_mean[2]

        pos_1 = np.append(mean_pos + hand_size * x_axis, 1)
        pixel_1 = self.projection @ pos_1
        pixel_1 = pixel_1[:2] / pixel_1[2]

        pos_2 = np.append(mean_pos - hand_size * x_axis, 1)
        pixel_2 = self.projection @ pos_2
        pixel_2 = pixel_2[:2] / pixel_2[2]

        icon_size = int(np.linalg.norm(pixel_1[:2] - pixel_2[:2]) / 2)
        icon_size = max(1, min(icon_size, 100))

        resized_icon = cv2.resize(hand_icon, (icon_size, icon_size))
        h, w = resized_icon.shape[:2]
        x, y = int(pixel_mean[0]), int(pixel_mean[1])

        if align == "top-left":
            roi_x = int(max(0, x - w * 0.15))
            roi_y = int(max(0, y - h * 0.1))
        if align == "top-right":
            roi_x = int(max(0, x - w + w * 0.15))
            roi_y = int(max(0, y - h * 0.1))
        if align == "center":
            roi_x = int(max(0, x - w // 2))
            roi_y = int(max(0, y - h // 2))
        roi_w = min(w, result.shape[1] - roi_x)
        roi_h = min(h, result.shape[0] - roi_y)

        if roi_w <= 0 or roi_h <= 0:
            return result

        icon_x = max(0, w // 2 - x)
        icon_y = max(0, h // 2 - y)

        roi = result[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        icon_roi = resized_icon[icon_y:icon_y + roi_h, icon_x:icon_x + roi_w]

        if icon_roi.size == 0 or roi.shape[:2] != icon_roi.shape[:2]:
            return result

        if icon_roi.shape[2] == 4:
            alpha = icon_roi[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :,
                    c] = roi[:, :, c] * (1 - alpha) + icon_roi[:, :, c] * alpha
            result[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi
        else:
            result[roi_y:roi_y + roi_h,
                   roi_x:roi_x + roi_w] = icon_roi[:, :, :3]

        return result

    def _overlay_hand_icons(self, frame):
        if self.n_ctrl_parts not in [1, 2]:
            raise ValueError("Only support 1 or 2 control parts")

        result = frame.copy()

        c2w = np.linalg.inv(self.w2c)
        x_axis = c2w[:3, 0]
        self.projection = self.intrinsic @ self.w2c[:3, :]
        hand_size = 0.1  # size in physical space (in meters)

        if self.n_ctrl_parts == 1:
            current_target = self.hand_left_pos.unsqueeze(0)
            # align = 'top-right'
            align = "center"
            result = self._overlay_hand_at_position(result, current_target,
                                                    x_axis, hand_size,
                                                    self.hand_left, align)
        else:
            for i in range(2):
                current_target = (self.hand_left_pos.unsqueeze(0) if i == 0
                                  else self.hand_right_pos.unsqueeze(0))
                # align = 'top-right' if i == 0 else 'top-left'
                align = "center"
                hand_icon = self.hand_left if i == 0 else self.hand_right
                result = self._overlay_hand_at_position(
                    result, current_target, x_axis, hand_size, hand_icon,
                    align)

        return result

    def update_frame(self, frame, pressed_keys):
        result = frame.copy()

        result = self._overlay_hand_icons(result)

        # Add text to show active keys
        font = cv2.FONT_HERSHEY_SIMPLEX
        active_keys = ", ".join(
            sorted(pressed_keys)) if pressed_keys else "None"
        cv2.putText(result, f"Active keys: {active_keys}", (10, 30), font, 0.7,
                    (0, 0, 0), 2)

        # Add text to show which keys are being handled by physical vs virtual keyboard
        physical_keys = set(
            [k for k in pressed_keys if k not in self.virtual_keys])
        virtual_keys = set([k for k in pressed_keys if k in self.virtual_keys])
        physical_text = ", ".join(
            sorted(physical_keys)) if physical_keys else "None"
        virtual_text = ", ".join(
            sorted(virtual_keys)) if virtual_keys else "None"
        cv2.putText(result, f"Physical: {physical_text}", (10, 60), font, 0.7,
                    (0, 0, 0), 2)
        cv2.putText(result, f"Virtual: {virtual_text}", (10, 90), font, 0.7,
                    (0, 0, 0), 2)

        # overlay an transparent white mask on the bottom left and bottom right corners with width trans_width, and height trans_height
        trans_width = 160
        trans_height = 120
        overlay = result.copy()

        bottom_left_pt1 = (0, cfg.WH[1] - trans_height)
        bottom_left_pt2 = (trans_width, cfg.WH[1])
        cv2.rectangle(overlay, bottom_left_pt1, bottom_left_pt2,
                      (255, 255, 255), -1)

        if self.n_ctrl_parts == 2:
            bottom_right_pt1 = (cfg.WH[0] - trans_width,
                                cfg.WH[1] - trans_height)
            bottom_right_pt2 = (cfg.WH[0], cfg.WH[1])
            cv2.rectangle(overlay, bottom_right_pt1, bottom_right_pt2,
                          (255, 255, 255), -1)

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        # Draw all buttons for Set 1 (left side)
        for key, pos in self.arrow_positions_set1.items():
            if key in pressed_keys:
                result = self._overlay_arrow(result,
                                             None,
                                             pos,
                                             key,
                                             filled=True)
            else:
                result = self._overlay_arrow(result,
                                             None,
                                             pos,
                                             key,
                                             filled=False)

        # Draw all buttons for Set 2 (right side)
        if self.n_ctrl_parts == 2:
            for key, pos in self.arrow_positions_set2.items():
                if key in pressed_keys:
                    result = self._overlay_arrow(result,
                                                 None,
                                                 pos,
                                                 key,
                                                 filled=True)
                else:
                    result = self._overlay_arrow(result,
                                                 None,
                                                 pos,
                                                 key,
                                                 filled=False)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        control1_x = self.edge_buffer  # hard coded for now
        control2_x = cfg.WH[0] - self.edge_buffer - 113  # hard coded for now
        text_y = (cfg.WH[1] - self.arrow_size * 2 - self.bottom_margin - 10
                  )  # hard coded for now
        cv2.putText(
            result,
            "Left Hand",
            (control1_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        if self.n_ctrl_parts == 2:
            cv2.putText(
                result,
                "Right Hand",
                (control2_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return result

    def _find_closest_point(self, target_points):
        """Find the closest structure point to any of the target points."""
        dist_matrix = torch.sum(
            (target_points.unsqueeze(1) -
             self.structure_points.unsqueeze(0))**2,
            dim=2,
        )
        min_dist_per_ctrl_pts, min_indices = torch.min(dist_matrix, dim=1)
        min_idx = min_indices[torch.argmin(min_dist_per_ctrl_pts)]
        return self.structure_points[min_idx].unsqueeze(0)

    def interactive_playground(self,
                               model_path,
                               gs_path,
                               n_ctrl_parts=1,
                               inv_ctrl=False):
        # Load the model
        logger.info(f"Load model from {model_path}")
        # checkpoint = torch.load(model_path, map_location=cfg.device)

        logger.info("Party Time Start!!!!")
        ###########################################################################
        # [MPM] initialization
        # params = {"E": 0.008, "nu": 0.3}
        params = get_param(name=self.exp_name)
        mpm = self.initialize_mpm(parameters=params,
                                  diff_sim=False,
                                  update_ctrl=False)
        ###########################################################################

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        current_target = self.controller_points[0]
        prev_target = current_target

        vis_controller_points = current_target.cpu().numpy()

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)

        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            N = vis_controller_points.shape[0]
            masks_ctrl_pts = []
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                masks_ctrl_pts.append(torch.from_numpy(mask))
            # project the center of the cluster to the object to the image space, those on the left will be mask 1
            center1 = np.mean(vis_controller_points[masks_ctrl_pts[0]], axis=0)
            center2 = np.mean(vis_controller_points[masks_ctrl_pts[1]], axis=0)
            center1 = np.concatenate([center1, [1]])
            center2 = np.concatenate([center2, [1]])
            proj_mat = intrinsic @ w2c[:3, :]
            center1 = proj_mat @ center1
            center2 = proj_mat @ center2
            center1 = center1 / center1[-1]
            center2 = center2 / center2[-1]
            if center1[0] > center2[0]:
                print("Switching the control parts")
                masks_ctrl_pts = [masks_ctrl_pts[1], masks_ctrl_pts[0]]
        else:
            masks_ctrl_pts = None
        self.n_ctrl_parts = n_ctrl_parts
        self.mask_ctrl_pts = masks_ctrl_pts
        self.scale_factors = 1.0
        assert n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        print("UI Controls:")
        print("- Set 1: WASD (XY movement), QE (Z movement)")
        print("- Set 2: IJKL (XY movement), UO (Z movement)")
        print(
            "- Supports both physical keyboard and virtual keyboard input simultaneously"
        )
        print("- Multiple keys can be pressed at the same time")
        print("- Press ESC to exit")
        self.inv_ctrl = -1.0 if inv_ctrl else 1.0
        self.key_mappings = {
            # Set 1 controls
            "w": (0, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "s": (0, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "a": (0, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "d": (0, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            # Set 2 controls
            "i": (1, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "k": (1, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "j": (1, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "l": (1, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "o": (1, np.array([0, 0, 0.005])),
            "u": (1, np.array([0, 0, -0.005])),
        }
        self.pressed_keys = set()
        self.w2c = w2c
        self.intrinsic = intrinsic
        self.init_control_ui()
        if n_ctrl_parts > 1:
            hand_positions = []
            for i in range(2):
                target_points = torch.from_numpy(
                    vis_controller_points[self.mask_ctrl_pts[i]]).to("cuda")
                hand_positions.append(self._find_closest_point(target_points))
            self.hand_left_pos, self.hand_right_pos = hand_positions
        else:
            target_points = torch.from_numpy(vis_controller_points).to("cuda")
            self.hand_left_pos = self._find_closest_point(target_points)

        # Initialize keyboard tracking variables
        self.pressed_keys = set(
        )  # Set to track all active keys (both physical and virtual)
        self.virtual_keys = {
        }  # Dictionary to track virtual keys with timestamps
        self.virtual_key_duration = 0.03  # Virtual key press duration in seconds
        self.target_change = np.zeros((n_ctrl_parts, 3))

        # Start physical keyboard listener
        listener = keyboard.Listener(on_press=self.on_press,
                                     on_release=self.on_release)
        listener.start()

        ############## Temporary timer ##############
        import time

        class Timer:

            def __init__(self, name):
                self.name = name
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None
                self.use_cuda = torch.cuda.is_available()

            def start(self):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    self.cuda_start_event = torch.cuda.Event(
                        enable_timing=True)
                    self.cuda_end_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_start_event.record()
                self.start_time = time.time()

            def stop(self):
                if self.use_cuda:
                    self.cuda_end_event.record()
                    torch.cuda.synchronize()
                    self.elapsed = (self.cuda_start_event.elapsed_time(
                        self.cuda_end_event) / 1000)  # convert ms to seconds
                else:
                    self.elapsed = time.time() - self.start_time
                return self.elapsed

            def reset(self):
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None

        sim_timer = Timer("Simulator")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        interp_timer = Timer("Full Motion Interpolation")
        total_timer = Timer("Total Loop")
        knn_weights_timer = Timer("KNN Weights")
        motion_interp_timer = Timer("Motion Interpolation")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "rendering": [],
            "frame_compositing": [],
            "full_motion_interpolation": [],
            "total": [],
            "knn_weights": [],
            "motion_interp": [],
        }

        # Number of frames to average over for stats
        STATS_WINDOW = 10
        frame_count = 0

        ############## End Temporary timer ##############

        while True:

            total_timer.start()

            # 1. Simulator step

            sim_timer.start()

            ########################################################################
            # x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            vertices = mpm.advance_to(n_frames=1)
            x = inv_transform_points(vertices[-1], name=self.exp_name)

            sim_time = sim_timer.stop()
            component_times["simulator"].append(sim_time)

            torch.cuda.synchronize()

            # 2. Frame initialization and setup

            frame_timer.start()

            frame = overlay.clone()

            frame_setup_time = (
                frame_timer.stop()
            )  # We'll accumulate times for frame compositing

            torch.cuda.synchronize()

            # 3. Rendering
            render_timer.start()

            # render with gaussians and paste the image on top of the frame
            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach()

            render_time = render_timer.stop()
            component_times["rendering"].append(render_time)

            torch.cuda.synchronize()

            # Continue frame compositing
            frame_timer.start()

            image = image.clamp(0, 1)
            if use_white_background:
                image_mask = torch.logical_and((image != 1.0).any(dim=2),
                                               image[:, :, 3] > 100 / 255)
            else:
                image_mask = torch.logical_and((image != 0.0).any(dim=2),
                                               image[:, :, 3] > 100 / 255)
            image[..., 3].masked_fill_(~image_mask, 0.0)

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.cpu().numpy()
            image_mask = image_mask.cpu().numpy()
            frame = frame.astype(np.uint8)

            frame = self.update_frame(frame, self.pressed_keys)

            # Add shadows
            final_shadow = get_simple_shadow(x,
                                             intrinsic,
                                             w2c,
                                             width,
                                             height,
                                             image_mask,
                                             light_point=[0, 0, -3])
            frame[final_shadow] = (frame[final_shadow] * 0.95).astype(np.uint8)
            final_shadow = get_simple_shadow(x,
                                             intrinsic,
                                             w2c,
                                             width,
                                             height,
                                             image_mask,
                                             light_point=[1, 0.5, -2])
            frame[final_shadow] = (frame[final_shadow] * 0.97).astype(np.uint8)
            final_shadow = get_simple_shadow(x,
                                             intrinsic,
                                             w2c,
                                             width,
                                             height,
                                             image_mask,
                                             light_point=[-3, -0.5, -5])
            frame[final_shadow] = (frame[final_shadow] * 0.98).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Interactive Playground", frame)
            key = cv2.waitKey(1)

            # Handle virtual keyboard input through OpenCV window
            if key != -1:
                key_char = chr(key & 0xFF).lower()
                if key_char in self.key_mappings:
                    # Store virtual key with timestamp - refresh timestamp if already pressed
                    self.virtual_keys[key_char] = time.time()
                    self.pressed_keys.add(key_char)
                elif key == 27:  # ESC key to exit
                    break

            # Process all keyboard inputs (both physical and virtual)
            # For virtual keys, check if they're still active based on timestamp
            current_time = time.time()
            keys_to_remove = []
            for k, press_time in self.virtual_keys.items():
                if current_time - press_time > self.virtual_key_duration:
                    keys_to_remove.append(k)

            # Remove expired virtual keys
            for k in keys_to_remove:
                if k in self.pressed_keys:
                    self.pressed_keys.discard(k)
                if k in self.virtual_keys:
                    del self.virtual_keys[k]

            frame_comp_time = (frame_timer.stop() + frame_setup_time
                               )  # Total frame compositing time
            component_times["frame_compositing"].append(frame_comp_time)

            torch.cuda.synchronize()

            if prev_x is not None:
                with torch.no_grad():

                    prev_particle_pos = prev_x
                    cur_particle_pos = x

                    if relations is None:
                        relations = get_topk_indices(
                            prev_x,
                            K=16)  # only computed in the first iteration

                    if weights is None:
                        weights, weights_indices = knn_weights_sparse(
                            prev_particle_pos, current_pos,
                            K=16)  # only computed in the first iteration

                    interp_timer.start()

                    weights = calc_weights_vals_from_indices(
                        prev_particle_pos, current_pos, weights_indices)

                    current_pos, current_rot, _ = interpolate_motions_speedup(
                        bones=prev_particle_pos,
                        motions=cur_particle_pos - prev_particle_pos,
                        relations=relations,
                        weights=weights,
                        weights_indices=weights_indices,
                        xyz=current_pos,
                        quat=current_rot,
                    )

                    # update gaussians with the new positions and rotations
                    gaussians._xyz = current_pos
                    gaussians._rotation = current_rot

                interp_time = interp_timer.stop()
                component_times["full_motion_interpolation"].append(
                    interp_time)

            torch.cuda.synchronize()

            prev_x = x.clone()

            prev_target = current_target
            target_change = self.get_target_change()

            if masks_ctrl_pts is not None:
                for i in range(n_ctrl_parts):
                    if masks_ctrl_pts[i].sum() > 0:
                        current_target[masks_ctrl_pts[i]] += torch.tensor(
                            target_change[i],
                            dtype=torch.float32,
                            device=cfg.device)
                        if i == 0:
                            self.hand_left_pos += torch.tensor(
                                target_change[i],
                                dtype=torch.float32,
                                device=cfg.device)
                        if i == 1:
                            self.hand_right_pos += torch.tensor(
                                target_change[i],
                                dtype=torch.float32,
                                device=cfg.device)
            else:
                current_target += torch.tensor(target_change,
                                               dtype=torch.float32,
                                               device=cfg.device)
                self.hand_left_pos += torch.tensor(target_change,
                                                   dtype=torch.float32,
                                                   device=cfg.device)

            ###############################################
            # update the handler velocity with keyboard input
            if self.n_ctrl_parts == 1:
                cmap = [1]
            elif self.n_ctrl_parts == 2:
                cmap = [1, 2] if DATASET == "phystwin" else [2, 1]
            for i in cmap:
                mpm.mpm_solver.collider_params[i].velocity = wp.vec3(0, 0, 0)

            if DATASET == "phystwin":
                vscale = 0.05 if inv_ctrl else -0.05
            else:
                vscale = -0.05 if inv_ctrl else 0.05
            for key in self.pressed_keys:
                if key in self.key_mappings:
                    idx, change = self.key_mappings[key]
                    mpm.mpm_solver.collider_params[
                        cmap[idx]].velocity = wp.vec3(change) / vscale
            ###############################################

            ############### Temporary timer ###############
            # 2025-09, YH, add viser visualization
            from gaussian_splatting.utils.sh_utils import SH2RGB

            # Total loop time
            total_time = total_timer.stop()
            component_times["total"].append(total_time)

            # Calculate FPS
            fps = 1.0 / total_time
            fps_history.append(fps)

            # Display performance stats periodically
            frame_count += 1
            if frame_count % 10 == 0:
                # Limit stats to last STATS_WINDOW frames
                if len(fps_history) > STATS_WINDOW:
                    fps_history = fps_history[-STATS_WINDOW:]
                    for key in component_times:
                        component_times[key] = component_times[key][
                            -STATS_WINDOW:]

                avg_fps = np.mean(fps_history)
                print(
                    f"\n--- Performance Stats (avg over last {len(fps_history)} frames) ---"
                )
                print(f"FPS: {avg_fps:.2f}")

                # Calculate percentages for pie chart
                total_avg = np.mean(component_times["total"])
                print(f"Total Frame Time: {total_avg*1000:.2f} ms")

                # Display individual component times
                for key in [
                        "simulator",
                        "rendering",
                        "frame_compositing",
                        "full_motion_interpolation",
                        "knn_weights",
                        "motion_interp",
                ]:
                    avg_time = np.mean(component_times[key])
                    percentage = (avg_time / total_avg) * 100
                    print(
                        f"{key.capitalize()}: {avg_time*1000:.2f} ms ({percentage:.1f}%)"
                    )

            # 2025-09, YH, update the gaussians and pcd here
            features_dc_rgb = gaussians._features_dc.squeeze(1)
            gaussian_colors = SH2RGB(features_dc_rgb).clamp(
                0, 1).detach().cpu().numpy()
            viser_server.scene.add_gaussian_splats(
                name=f"/world/gaussians",
                centers=gaussians.get_xyz.detach().cpu().numpy(),
                covariances=gaussians.get_real_covariance().detach().cpu().
                numpy(),
                rgbs=gaussian_colors,
                opacities=gaussians.get_opacity.detach().cpu().numpy(),
                visible=True,
            )

            viser_server.scene.add_point_cloud(
                name=f"/world/gaussians_points",
                points=gaussians.get_xyz.detach().cpu().numpy(),
                colors=gaussian_colors,
                point_size=0.001,
                point_shape="circle",
                visible=True,
            )

        listener.stop()

    def _transform_gs(self, gaussians, M, majority_scale=1):

        new_gaussians = copy.copy(gaussians)

        new_xyz = gaussians.get_xyz.clone()
        ones = torch.ones((new_xyz.shape[0], 1),
                          device=new_xyz.device,
                          dtype=new_xyz.dtype)
        new_xyz = torch.cat((new_xyz, ones), dim=1)
        print("inside:", new_xyz.max(), new_xyz.min())
        new_xyz = new_xyz @ M.T
        print("outside:", new_xyz.max(), new_xyz.min())

        new_rotation = gaussians.get_rotation.clone()
        new_rotation = quaternion_multiply(matrix_to_quaternion(M[:3, :3]),
                                           new_rotation)

        new_scales = gaussians._scaling.clone()
        new_scales += torch.log(
            torch.tensor(majority_scale,
                         device=new_scales.device,
                         dtype=new_scales.dtype))

        new_gaussians._xyz = new_xyz[:, :3]
        new_gaussians._rotation = new_rotation
        new_gaussians._scaling = new_scales

        return new_gaussians

    def _create_gs_view(self, w2c, intrinsic, height, width):
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        view = Camera(
            (width, height),
            colmap_id="0000",
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name="0000",
            uid="0000",
            data_device="cuda",
            train_test_exp=None,
            is_test_dataset=None,
            is_test_view=None,
            K=K,
            normal=None,
            depth=None,
            occ_mask=None,
        )
        return view

    # ----------------- Online control and optimization ----------------- #
    def init_online(self, data_path, base_dir, device="cuda:0"):
        cfg.device = device
        self.object_points = None
        self.ctrl_param = None
        self.obj_name = "dough"  # "rope" or "dough"
        self.exp_name = f"{self.obj_name}_app"  # "opt" or "app"
        self.vis_path = "outputs"
        self.enable_online_opt = False
        self.ctrl_mode = "preset"  # "teleop" or "hand" or "preset"
        self.n_ctrl_parts = 2
        self.ctrl_start_step = 30
        self.ctrl_end_step = 999999
        self.replay = True

        # with open("outputs/init_points.pkl", "rb") as f:
        #     data = pickle.load(f)

        with open(f"outputs/online_data/final_data_{self.obj_name}.pkl",
                  "rb") as f:
            data = pickle.load(f)
        self.object_points = torch.tensor(data["object_points"],
                                          dtype=torch.float32,
                                          device=cfg.device)

        if self.obj_name == "rope":
            self.ctrl_param = {
                "handler_pos": [(.65, .25, .6), (.65, .50, .6)],
                "csize": [0.015, 0.015],
            }
        elif self.obj_name == "dough":
            from .utils import densify_point_cloud
            object_points = densify_point_cloud(self.object_points[0], 20000)
            self.object_points = object_points.unsqueeze(0)
            self.ctrl_param = {
                "handler_pos": [(.72, .325, .51), (.72, .405, .51)],
                # "csize": [0.02, 0.02],
                "csize": [0.01, 0.01],
            }

        self.prepare_realtime_streaming()

    def prepare_realtime_streaming(self):
        print("Preparing real-time streaming...")

        # initial point cloud
        init_x = self.object_points[0].clone()
        init_x = transform_points(x=init_x)

        # initialize mpm
        if self.exp_name == "rope_opt":
            init_param = {"type": "jelly", "E": 0.05}
        elif self.exp_name == "dough_opt":
            init_param = {
                "type": "plasticine",
                "E": 0.0015,  #0.0015,
                "ys": 900.,
            }
        elif self.exp_name == "rope_app":
            init_param = {"type": "jelly", "E": 0.1}
        elif self.exp_name == "dough_app":
            init_param = {
                "type": "plasticine",
                "E": 0.0001,  #0.0015,
                "ys": 900.,
            }
        if not self.enable_online_opt:
            mpm = self.initialize_mpm(parameters=init_param,
                                      diff_sim=False,
                                      update_ctrl=False)
        else:
            mpm = self.initialize_mpm(parameters=init_param,
                                      diff_sim=True,
                                      update_ctrl=False)
            mpm.mpm_solver.step_callback = None
            mpm.prepare_online_optimization(x=init_x, parameters=init_param)
            self.trainable_params = init_param  # store optimizable parameters
            self.losses = []
            self.param_hist = []
            self.total_opt_steps = 0

        if self.obj_name == "dough":
            mpm.mpm_solver.collider_params[0].point = wp.vec3(0, 0, 0.505)

        self.mpm = mpm
        self.prev_hand_pos = None
        self.prev_grip_pos = np.array([[0., 0, 0], [0, 0, 0]])
        self.total_stream_steps = 0
        print("Finished initializing MPM")

    def realtime_streaming_step(self, real_data_list=None):
        from .utils import project_point_cloud, unproject_point_cloud
        print(f"Real-time streaming step... {self.total_stream_steps}")
        cur_step = self.total_stream_steps
        opt_step = self.enable_online_opt and (cur_step
                                               >= 100) and (cur_step % 10 == 0)
        exp_name = self.exp_name

        #---------------------- load data ---------------------
        # load real data
        if real_data_list is not None:
            real_data = real_data_list[0]
            if "color" in real_data:
                color = real_data["color"]
            else:
                color = cv2.imread(cfg.bg_img_path)
            if "depth" in real_data:
                depth = real_data["depth"]
            else:
                depth = None
            if "object_mask" in real_data:
                object_mask = real_data["object_mask"]
            else:
                object_mask = None
            if "handler_mark" in real_data:
                hand_pos = real_data["handler_mark"]
            else:
                hand_pos = None

        n_ctrl_parts = self.n_ctrl_parts
        cmap = [1, 2]  # right - left
        # if self.ctrl_mode == "teleop":
        #     with open(f"/home/yunuochen/Workspace/code/PET/ee_position.txt",
        #               "r") as f:  # load teleop result
        #         line = f.readline().split()
        #         ee_pos = [float(x) for x in line]
        #         if not len(ee_pos) == 6:
        #             grip_pos = self.prev_grip_pos
        #         else:
        #             grip_pos = np.array([[ee_pos[0], ee_pos[1], ee_pos[2]],
        #                                  [ee_pos[3], ee_pos[4], ee_pos[5]]])
        #         grip_pos = grip_pos + np.array([[0.0, -0.495, 0.0],
        #                                         [0.0, 0.0, 0.0]])
        #         grip_pos = grip_pos / 5 + np.array([0.5, 0.49, 0.49])
        #         self.grip_pos = grip_pos
        # elif self.ctrl_mode == "hand":
        #     prev_hand_pos = self.prev_hand_pos
        #     self.prev_hand_pos = hand_pos
        #     if (hand_pos is None) or (prev_hand_pos is None) or (
        #             len(hand_pos) != n_ctrl_parts) or (len(prev_hand_pos)
        #                                                != n_ctrl_parts):
        #         hand_pos = [[0, 0] for _ in range(n_ctrl_parts)]
        #         prev_hand_pos = [[0, 0] for _ in range(n_ctrl_parts)]

        # load camera parameters
        cam_id = 0
        W, H = cfg.WH
        intrs = cfg.intrinsics[cam_id]
        w2c = cfg.w2cs[cam_id]
        intrs = torch.tensor(intrs, dtype=torch.float32, device=cfg.device)
        w2c = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)

        if self.replay:
            with open(
                    f"{self.vis_path}/online_data/{exp_name}/data_{(cur_step // 2):04d}.pkl",
                    "rb") as f:
                online_data = pickle.load(f)

            color = online_data["color"]
            depth = online_data["depth"]
            object_mask = online_data["object_mask"]
            gt_points_3d = torch.tensor(online_data["points_3d"],
                                        dtype=torch.float32,
                                        device=cfg.device)
            hand_pos = None
        else:
            # reconstruct 3d
            ys, xs = np.where(object_mask > 0)  # (N,) arrays
            points_2d = np.stack([xs, ys], axis=1)  # (N, 2)
            object_depth = depth[ys, xs]  # (N,)
            points_2d = torch.tensor(points_2d,
                                     dtype=torch.float32,
                                     device=cfg.device)
            object_depth = torch.tensor(object_depth,
                                        dtype=torch.float32,
                                        device=cfg.device)
            points_3d = unproject_point_cloud(points_2d, object_depth, intrs,
                                              w2c)
            gt_points_3d = transform_points(points_3d)

            online_data = {
                "color": color,
                "depth": depth,
                "object_mask": object_mask,
                "points_3d": gt_points_3d.cpu().numpy()
            }
            # with open(
            #         f"{self.vis_path}/online_data/{exp_name}/data_{cur_step:04d}.pkl",
            #         "wb") as f:
            #     pickle.dump(online_data, f)

        save_ply_file(gt_points_3d.cpu().numpy(),
                      f"{self.vis_path}/online/gt_pcd_{cur_step:04d}.ply")

        #--------------------- simulation control ---------------------
        # set simulation control
        mpm_solver = self.mpm.mpm_solver
        ctrl_pos = np.array(
            [mpm_solver.collider_params[i].point for i in cmap])
        save_ply_file(ctrl_pos,
                      f"{self.vis_path}/online/ctrl_{cur_step:04d}.ply")

        ctrl_vel = [[0, 0, 0] for _ in range(n_ctrl_parts)]
        # if cur_step == self.ctrl_start_step:
        #     print("Starting control...")
        #     for i in range(n_ctrl_parts):
        #         mpm_solver.collider_params[cmap[i]].point = wp.vec3(
        #             self.grip_pos[i])

        # elif cur_step > self.ctrl_start_step and cur_step < self.ctrl_end_step:
        #     if self.ctrl_mode == "teleop":
        #         for i in range(n_ctrl_parts):
        #             # diff_pos = grip_pos[i] - prev_grip_pos[i]
        #             diff_pos = self.grip_pos[i] - ctrl_pos[i]
        #             v = diff_pos * 100.0
        #             ctrl_vel[i] = [v[0], v[1], v[2]]
        #     elif self.ctrl_mode == "hand":
        #         for i in range(n_ctrl_parts):
        #             diff_pos = (np.array(hand_pos[i]) -
        #                         np.array(prev_hand_pos[i])).astype(np.float32)
        #             v = diff_pos[1] / (-10.0)
        #             ctrl_vel[i] = [0, 0, v]
        #     elif self.ctrl_mode == "preset":
        #         ctrl_vel = [[0, 0, 0.1], [0, 0, 0.1]]

        if self.ctrl_mode == "preset":
            if exp_name == "rope_opt":
                mvs = 10
                if cur_step <= 50:
                    ctrl_vel = [[0, 0, 0.1], [0, 0, 0.1]]
                elif cur_step > 300 and cur_step <= 300 + mvs:
                    ctrl_vel = [[-0.1, 0, -0.1], [0, 0, 0]]
                elif cur_step > 500 and cur_step <= 500 + mvs:
                    ctrl_vel = [[0, 0, 0], [-0.1, 0, -0.1]]
                elif cur_step > 700 and cur_step <= 700 + mvs:
                    ctrl_vel = [[0.1, 0, 0.1], [0, 0, 0]]
                elif cur_step > 900 and cur_step <= 900 + mvs:
                    ctrl_vel = [[0, 0, 0], [0.1, 0, 0.1]]
                elif cur_step > 1100 and cur_step <= 1100 + mvs:
                    ctrl_vel = [[0.1, 0, 0.1], [0, 0, 0]]
                elif cur_step > 1300 and cur_step <= 1300 + mvs:
                    ctrl_vel = [[0, 0, 0], [0.1, 0, 0.1]]
                elif cur_step > 1500 and cur_step <= 1500 + mvs:
                    ctrl_vel = [[-0.1, 0, -0.1], [0, 0, 0]]
                elif cur_step > 1700 and cur_step <= 1700 + mvs:
                    ctrl_vel = [[0, 0, 0], [-0.1, 0, -0.1]]
                else:
                    ctrl_vel = [[0, 0, 0], [0, 0, 0]]
            elif exp_name == "dough_opt":
                mvs = 10
                if cur_step <= 50:
                    ctrl_vel = [[0, -0.05, 0], [0, 0.05, 0]]
                elif cur_step > 200 and cur_step <= 200 + mvs:
                    ctrl_vel = [[0, -0.08, 0], [0, 0, 0]]
                elif cur_step > 400 and cur_step <= 400 + mvs:
                    ctrl_vel = [[0, 0.08, 0], [0, 0, 0]]
                elif cur_step > 600 and cur_step <= 600 + mvs:
                    ctrl_vel = [[0.1, 0, 0], [0, 0, 0]]
                elif cur_step > 800 and cur_step <= 800 + mvs:
                    ctrl_vel = [[-0.05, 0, 0], [0, 0, 0]]
                elif cur_step > 1000 and cur_step <= 1000 + mvs:
                    ctrl_vel = [[0, 0, 0], [0, -0.05, 0]]
                elif cur_step > 1300 and cur_step <= 1300 + mvs:
                    ctrl_vel = [[0, 0, 0], [0.1, 0, 0]]
                elif cur_step > 1500 and cur_step <= 1500 + mvs:
                    ctrl_vel = [[0, 0, 0], [-0.05, 0, 0]]
                else:
                    ctrl_vel = [[0, 0, 0], [0, 0, 0]]
            elif exp_name == "rope_app":
                mvs = 10
                if cur_step <= 40:
                    ctrl_vel = [[0, 0, 0.125], [0, 0, 0.125]]
                elif cur_step >= 85 and cur_step <= 85 + mvs:
                    ctrl_vel = [[-0.3, 0.5, 0], [0, 0, 0]]
                elif cur_step >= 125 and cur_step <= 125 + mvs:
                    ctrl_vel = [[0, 0, 0], [0, -0.3, 0]]
            elif exp_name == "dough_app":
                mvs = 10
                if cur_step >= 60 and cur_step <= 60 + mvs:
                    ctrl_vel = [[0, 0, 0], [0.15, 0.5, 0]]
                if cur_step >= 105 and cur_step <= 105 + mvs:
                    ctrl_vel = [[0.25, -0.5, 0], [0, 0, 0]]
                # if cur_step <= 50:
                #     ctrl_vel = [[0, -0.05, 0], [0, 0.05, 0]]
                # elif cur_step > 200 and cur_step <= 200 + mvs:
                #     ctrl_vel = [[0, -0.08, 0], [0, 0, 0]]
                # elif cur_step > 400 and cur_step <= 400 + mvs:
                #     ctrl_vel = [[0, 0.08, 0], [0, 0, 0]]
                # elif cur_step > 600 and cur_step <= 600 + mvs:
                #     ctrl_vel = [[0.1, 0, 0], [0, 0, 0]]

        if np.linalg.norm(np.array(ctrl_vel)) > 0:
            opt_step = False

        # elif cur_step >= 100 and cur_step <= 110:
        #     opt_step = False
        #     ctrl_vel = [[-0.05, 0, -0.05], [0, 0, 0]]

        for i in range(n_ctrl_parts):
            mpm_solver.collider_params[cmap[i]].velocity = wp.vec3(ctrl_vel[i])

        # ------------------- simulation / optimization -------------------
        def loss_func(points):
            # Compute the loss based on the output points
            # return (points).norm()

            # 2d mask loss
            points_3d = inv_transform_points(points[-1])
            pixel_2d = project_point_cloud(points_3d, w2c, intrs)
            pixel_2d = pixel_2d.unsqueeze(0)
            mask_2d = torch.nonzero(torch.tensor(object_mask) > 0,
                                    as_tuple=False).to(pixel_2d.device)
            mask_2d = mask_2d[:, [1, 0]].float().unsqueeze(0)

            loss_2d, _ = chamfer_distance(pixel_2d, mask_2d)

            # 3d point cloud loss
            loss_3d, _ = chamfer_distance(points[-1].unsqueeze(0),
                                          gt_points_3d.unsqueeze(0))

            loss = loss_2d + loss_3d
            return loss

        if opt_step:
            # Online Optimization should not interfere with forward sim
            init_x = wp.to_torch(self.mpm.mpm_state.particle_x)
            init_param = self.trainable_params
            result = self.mpm.run_online_optimization_step(
                x=init_x,
                parameters=init_param,
                n_train_frames=10,
                n_iters=1,
                loss_func=loss_func)
            opt_param = result["param"]
            points_3d = result["points"][1]

            # apply changes to forward mpm
            self.trainable_params = opt_param
            new_param = {"E": opt_param["E"] * 1e6}
            self.mpm.set_paramters(new_param)
            # visualize
            self.losses.append(result["loss"])
            plt.clf()
            plt.plot(self.losses)
            plt.savefig(f"{self.vis_path}/loss_plot.png")
            self.param_hist.append(opt_param["E"])
            plt.clf()
            plt.plot(self.param_hist)
            plt.savefig(f"{self.vis_path}/param_plot.png")
            print(self.trainable_params)
        else:
            # one-frame forward sim
            with torch.no_grad():
                vertices = self.mpm.advance_to(n_frames=1)
                points_3d = vertices[-1]

        save_ply_file(
            points_3d.cpu().numpy(),
            f"{self.vis_path}/online/sim_pcd_{self.total_stream_steps:04d}.ply"
        )
        points_3d = inv_transform_points(points_3d)

        #------------------- visualization step -------------------
        vis_type = "pointcloud"
        if vis_type == "pointcloud":
            with torch.no_grad():
                pixels = project_point_cloud(points_3d, w2c, intrs)
                pixels = pixels.cpu().numpy()

            # Round and clip to image size
            pixels = np.round(pixels).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, W - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, H - 1)
            pixels[:, 0] -= 8

            img = color.copy()

            # Draw the simulated points
            overlay = img.copy()
            for x, y in pixels:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            alpha = 0.5
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Draw the object mask
            if object_mask is not None:
                alpha = 0.3
                mask = (object_mask > 0).astype(np.uint8)
                overlay = img.copy()
                overlay[mask == 1] = [0, 0, 255]
                img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

            # Draw the hand selections
            if hand_pos is not None:
                hsize = 25
                for [x, y] in hand_pos:
                    cv2.rectangle(img, (x - hsize, y - hsize),
                                  (x + hsize, y + hsize), (255, 0, 0), 2)

            cv2.imshow("Simulation", img)
            cv2.imwrite(
                f"{self.vis_path}/online/{self.total_stream_steps}.png", img)
            if opt_step:
                cv2.imwrite(f"{self.vis_path}/opt/{self.total_opt_steps}.png",
                            img)
                self.total_opt_steps += 1

            if self.replay:
                cv2.waitKey(1)
        else:
            raise NotImplementedError

        self.total_stream_steps += 1

    def initialize_pointcloud(self, real_data_list=None):
        from .utils import unproject_point_cloud, densify_point_cloud
        print(f"Real-time streaming step... {self.total_stream_steps}")

        W, H = cfg.WH

        # load real data
        fused_points = []
        for cam_id in range(len(real_data_list)):
            print(f"Processing camera {cam_id}")
            real_data = real_data_list[cam_id]
            depth = real_data["depth"]
            object_mask = real_data["object_mask"]

            intrs = cfg.intrinsics[cam_id]
            w2c = cfg.w2cs[cam_id]
            intrs = torch.tensor(intrs, dtype=torch.float32, device=cfg.device)
            w2c = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)

            ys, xs = np.where(object_mask > 0)  # (N,) arrays
            points_2d = np.stack([xs, ys], axis=1)  # (N, 2)
            object_depth = depth[ys, xs]  # (N,)
            points_2d = torch.tensor(points_2d,
                                     dtype=torch.float32,
                                     device=cfg.device)
            object_depth = torch.tensor(object_depth,
                                        dtype=torch.float32,
                                        device=cfg.device)
            points_3d = unproject_point_cloud(points_2d, object_depth, intrs,
                                              w2c)
            fused_points.append(points_3d)

        points_3d = torch.cat(fused_points, dim=0)
        print(f"Total {points_3d.shape[0]} points before processing")
        mask = points_3d[:, 2] <= 0.5
        points_3d = points_3d[mask]
        print(f"Filtered {points_3d.shape[0] - mask.sum()} points")
        points_3d = densify_point_cloud(points_3d, K=5000)
        print(f"Total {points_3d.shape[0]} points after densification")

        save_ply_file(points_3d.cpu().numpy(),
                      f"{self.vis_path}/fused_points.ply")
        data = {"object_points": points_3d.unsqueeze(0).cpu().numpy()}
        pickle.dump(data, open(f"{self.vis_path}/init_points.pkl", "wb"))

        exit(0)

    def evaluate_online(self, real_data_list=None):
        from .utils import project_point_cloud, unproject_point_cloud

        # load camera parameters
        cam_id = 0
        W, H = cfg.WH
        intrs = cfg.intrinsics[cam_id]
        w2c = cfg.w2cs[cam_id]
        intrs = torch.tensor(intrs, dtype=torch.float32, device=cfg.device)
        w2c = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)

        total_loss_3d = 0
        total_loss_2d = 0
        i_start = 1500
        i_end = 1600
        for i in range(i_start, i_end):
            with open(f"{self.vis_path}/online_data/dough/data_{i:04d}.pkl",
                      "rb") as f:
                online_data = pickle.load(f)

            object_mask = online_data["object_mask"]
            gt_points_3d = torch.tensor(online_data["points_3d"],
                                        dtype=torch.float32,
                                        device=cfg.device)

            # # filter out the invalid point
            # gt_points_3d = gt_points_3d[gt_points_3d[:, 2] < 0.6]

            import pyvista as pv
            # gt_pcd = pv.read(f"{self.vis_path}/online/gt_pcd_{i:04d}.ply")
            sim_pcd = pv.read(
                f"{self.vis_path}/online_data/results/dough/sim_pcd_{i:04d}.ply"
            )
            sim_points_3d = torch.tensor(sim_pcd.points,
                                         dtype=torch.float32,
                                         device=cfg.device)

            loss_3d, _ = chamfer_distance(gt_points_3d.unsqueeze(0),
                                          sim_points_3d.unsqueeze(0))
            total_loss_3d += loss_3d.item()

            points_3d = inv_transform_points(sim_points_3d)
            pixel_2d = project_point_cloud(points_3d, w2c, intrs)
            pixel_2d = pixel_2d.unsqueeze(0)
            mask_2d = torch.nonzero(torch.tensor(object_mask) > 0,
                                    as_tuple=False).to(pixel_2d.device)
            mask_2d = mask_2d[:, [1, 0]].float().unsqueeze(0)

            loss_2d, _ = chamfer_distance(pixel_2d, mask_2d)
            total_loss_2d += loss_2d.item()

        print(f"Avg 3D Chamfer Distance: {total_loss_3d / (i_end - i_start)}")
        print(f"Avg 2D Chamfer Distance: {total_loss_2d / (i_end - i_start)}")


def get_simple_shadow(
    points,
    intrinsic,
    w2c,
    width,
    height,
    image_mask,
    kernel_size=7,
    light_point=[0, 0, -3],
):
    points = points.cpu().numpy()

    t = -points[:, 2] / light_point[2]
    points_on_table = points + t[:, None] * light_point

    points_homogeneous = np.hstack([
        points_on_table,
        np.ones((points_on_table.shape[0], 1))
    ])  # Convert to homogeneous coordinates
    points_camera = (w2c @ points_homogeneous.T).T

    points_pixels = (intrinsic @ points_camera[:, :3].T).T
    points_pixels /= points_pixels[:, 2:3]
    pixel_coords = points_pixels[:, :2]

    valid_mask = ((pixel_coords[:, 0] >= 0)
                  & (pixel_coords[:, 0] < width)
                  & (pixel_coords[:, 1] >= 0)
                  & (pixel_coords[:, 1] < height))

    valid_pixel_coords = pixel_coords[valid_mask]
    valid_pixel_coords = valid_pixel_coords.astype(int)

    shadow_image = np.zeros((height, width), dtype=np.uint8)
    shadow_image[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_1 = np.ones((3, 3), np.uint(8))
    dilated_shadow = cv2.dilate(shadow_image, kernel, iterations=1)
    dilated_shadow = cv2.dilate(dilated_shadow, kernel_1, iterations=1)
    final_shadow = cv2.erode(dilated_shadow, kernel, iterations=1)

    final_shadow[image_mask] = 0
    final_shadow = final_shadow == 255
    return final_shadow
