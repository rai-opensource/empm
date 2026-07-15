"""
    Wrapper for Differentiable MPM in WARP
    Warp Version: 1.8.0
    File Created: July 23, 2025
    Last Modified: August 12, 2025
"""
import warp as wp
from .mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
)
from .mpm_solver_diff import MPMWARPDiff
from .interface import (MPMDifferentiableSimulationTrail,
                        MPMDifferentiableSimulation,
                        MPMDifferentiableSimulationWCheckpoint,
                        MPMDifferentiableSimulationClean)

import os
from tqdm import tqdm
import torch
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

DTYPE = torch.float32
material_gallery = {
    "jelly": {
        "material": "jelly"
    },
    "metal": {
        "material": "metal"
    },
    "sand": {
        "material": "sand",
        "friction_angle": 35
    },
    "foam": {
        "material": "foam"
    },
    "snow": {
        "material": "snow"
    },
    "plasticine": {
        "material": "plasticine",
        "yield_stress": 5000.,
        "softening": 0.1
    },
    "neo-hookean": {
        "material": "neo-hookean"
    },
    "fluid": {
        "material": "fluid"
    }
}


def sample_cube(lower_corner, cube_size, N):
    lower_corner = np.array(lower_corner)
    cube_size = np.array(cube_size)
    points = np.random.rand(N, 3) * cube_size + lower_corner
    return points


def sample_sphere(center, radius, N):
    center = np.array(center)
    vecs = np.random.normal(size=(N, 3))
    vecs /= np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    r = np.random.rand(N)**(1 / 3) * radius
    points = vecs * r[:, np.newaxis] + center
    return points


def save_ply_seq(vertices, save_path="./outputs/sim", prefix="sim_"):
    os.makedirs(save_path, exist_ok=True)
    for f in range(len(vertices)):
        x_np = vertices[f].detach().cpu().numpy()
        point_cloud = pv.PolyData(x_np)
        point_cloud.save(f"{save_path}/{prefix}{f:04d}.ply", binary=False)


def save_ply_file(vertices, filename=""):
    if isinstance(vertices, torch.Tensor):
        x_np = vertices.detach().cpu().numpy()
    else:
        x_np = np.array(vertices)
    point_cloud = pv.PolyData(x_np)
    point_cloud.save(filename, binary=False)


class WarpMPMWrapper:

    def __init__(self,
                 grid_size=100,
                 grid_lim=1.0,
                 frame_dt=0.01,
                 sim_dt=1e-3,
                 device="cuda:0",
                 requires_grad=False,
                 frame_callback=None,
                 step_callback=None,
                 optim_callback=None):
        self.n_particles = 0
        self.grid_size = grid_size
        self.grid_lim = grid_lim
        self.frame_dt = frame_dt
        self.sim_dt = sim_dt
        self.device = device
        self.requires_grad = requires_grad

        self.gravity = [0.0, 0.0, -9.8]
        # TO BE DEPRECATED
        self.surface_colliders = []
        # [point=(0., 0., .1), normal=(0., 0., 1.), surface="sticky", friction=0.]
        self.moving_boundaries = []
        # [point=(.5, .5, .5), size=0.01, velocity=(0., 0., 0.), start_time=0.]
        self.geometry_colliders = []
        self.grid_geometry_movers = []
        self.particle_geometry_movers = []

        self.frame_callback = frame_callback  # per-frame callback
        self.step_callback = step_callback  # per-step callback
        self.optim_callback = optim_callback  # per-optimization callback

        self.x_t = torch.empty((0, 3), dtype=DTYPE, device=self.device)
        self.v_t = torch.empty((0, 3), dtype=DTYPE, device=self.device)
        self.vol_t = torch.empty(0, dtype=DTYPE, device=self.device)
        self.E_t = torch.empty(0, dtype=DTYPE, device=self.device)
        self.nu_t = torch.empty(0, dtype=DTYPE, device=self.device)
        self.rho_t = torch.empty(0, dtype=DTYPE, device=self.device)

        # [TODO]:
        self.initialized = False
        self.material_params = material_gallery["jelly"]

    def add_particles(self, x, v=None, vol=1e-5, E=5e4, nu=0.3, rho=2000.0):
        n_p = x.shape[0]
        ##
        if isinstance(x, np.ndarray):
            x_t = torch.tensor(x)
        elif isinstance(x, torch.Tensor):
            x_t = x
        x_t = x_t.to(dtype=DTYPE).to(self.device)
        self.x_t = torch.cat((self.x_t, x_t))
        ##
        if v is None:
            v_t = torch.zeros((n_p, 3))
        elif isinstance(v, np.ndarray):
            v_t = torch.tensor(v)
        elif isinstance(v, torch.Tensor):
            v_t = v
        v_t = v_t.to(dtype=DTYPE).to(self.device)
        self.v_t = torch.cat((self.v_t, v_t))

        def append_attributes(q_tensor, q):
            """ Append scalar attribute tensor
            """
            if isinstance(q, float):
                q_t = torch.ones(n_p) * q
            elif isinstance(q, np.ndarray):
                q_t = torch.tensor(q)
            elif isinstance(q, torch.Tensor):
                if q.ndim == 0:
                    q_t = torch.ones(n_p) * q.item()
                else:
                    q_t = q
            q_t = q_t.to(dtype=DTYPE).to(q_tensor.device)
            q_tensor = torch.cat((q_tensor, q_t))
            return q_tensor

        self.vol_t = append_attributes(self.vol_t, vol)
        self.E_t = append_attributes(self.E_t, E)
        self.nu_t = append_attributes(self.nu_t, nu)
        self.rho_t = append_attributes(self.rho_t, rho)
        ##

        assert (x_t.min().item() > 0. and x_t.max().item() < self.grid_lim)
        self.n_particles += n_p

    def set_paramters(self, params):
        assert (self.initialized)
        if "E" in params:
            self.E_t = torch.ones(self.n_particles,
                                  dtype=DTYPE,
                                  device=self.device) * params["E"]

        if "E" in params or "nu" in params:
            self.mpm_solver.set_E_nu_from_torch(self.mpm_model,
                                                self.E_t,
                                                self.nu_t,
                                                device=self.device)
            self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state,
                                           self.device)

    def add_grid_geometry_mover(self, geo_args):
        self.grid_geometry_movers.append(geo_args)
        return len(self.grid_geometry_movers) - 1

    def add_particle_mover(self, geo_args):
        self.mpm_solver.enforce_particle_velocity_rotation(
            self.mpm_state,
            point=geo_args["center"],
            normal=geo_args["axis"],
            half_height_and_radius=[
                geo_args["half_height"], geo_args["radius"]
            ],
            rotation_scale=0.0,
            translation_scale=0.0,
            start_time=0.0,
            end_time=999.0)

    def initialize(self):
        self.mpm_state = MPMStateStruct()
        self.mpm_state.init(self.n_particles,
                            device=self.device,
                            requires_grad=self.requires_grad)
        self.mpm_state.from_torch(
            self.x_t,
            self.vol_t,
            None,
            self.v_t,
            n_grid=self.grid_size,
            grid_lim=self.grid_lim,
            device=self.device,
            requires_grad=self.requires_grad,
        )

        self.mpm_state.reset_density(tensor_density=self.rho_t,
                                     selection_mask=None,
                                     device=self.device,
                                     update_mass=True)

        self.mpm_model = MPMModelStruct()
        self.mpm_model.init(self.n_particles,
                            device=self.device,
                            requires_grad=self.requires_grad)
        self.mpm_model.init_other_params(n_grid=self.grid_size,
                                         grid_lim=self.grid_lim,
                                         device=self.device)

        self.mpm_model.gravitational_accelaration = wp.vec3(
            self.gravity[0], self.gravity[1], self.gravity[2])

        self.mpm_solver = MPMWARPDiff()
        self.mpm_solver.initialize(self.n_particles,
                                   n_grid=self.grid_size,
                                   grid_lim=self.grid_lim,
                                   device=self.device)
        self.mpm_solver.set_parameters_dict(self.mpm_model, self.mpm_state,
                                            self.material_params)
        self.mpm_solver.set_E_nu_from_torch(self.mpm_model,
                                            self.E_t,
                                            self.nu_t,
                                            device=self.device)
        self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state,
                                       self.device)

        # TO BE DEPRECATED
        for sc in self.surface_colliders:
            self.mpm_solver.add_surface_collider(sc[0], sc[1], sc[2], sc[3])

        for mb in self.moving_boundaries:
            self.mpm_solver.set_velocity_on_cuboid(mb[0], mb[1], mb[2], mb[3])

        for gc in self.geometry_colliders:
            self.mpm_solver.add_geometry_collider(geo_args=gc)

        for gm in self.grid_geometry_movers:
            self.mpm_solver.set_grid_velocity_on_geometry(geo_args=gm)

        self.initialized = True

    def advance_to_static(self, n_frames=10):
        frame_callback = self.frame_callback
        self.frame_callback = None
        self.advance_to(n_frames=n_frames)
        self.frame_callback = frame_callback

    def advance_to(self, n_frames):
        """
            Return postions as a torch tensor (F, N, 3)
        """
        n_substeps = int(self.frame_dt / self.sim_dt)
        prev_state = self.mpm_state
        vertices = [wp.to_torch(prev_state.particle_x)]
        step_count = 0
        for frame in range(n_frames):
            if self.frame_callback:
                self.frame_callback(self, frame)
            for substep in range(n_substeps):
                if self.step_callback:
                    self.step_callback(self, step_count)
                next_state = prev_state.partial_clone(requires_grad=False)
                self.mpm_solver.p2g2p_differentiable(
                    self.mpm_model,
                    prev_state,
                    next_state,
                    self.sim_dt,
                    device=self.device,
                )
                prev_state = next_state
                step_count += 1
            vertices.append(wp.to_torch(prev_state.particle_x))

        self.mpm_state = prev_state
        vertices = torch.stack(vertices, dim=0)
        return vertices

    def prepare_online_optimization(self, x, parameters):
        """
            Prepare for online DiffMPM optimization
        """
        if not self.initialized:
            self.initialize()

        n_p = self.n_particles
        # [TODO]: initialize trainable parameters
        init_E = parameters["E"]
        E_t = torch.tensor(1., dtype=DTYPE, device=self.device) * init_E
        E_t.requires_grad_()
        trainable_params = [E_t]

        self.optim = {
            "trainable_params": trainable_params,
        }

        print("Finished preparing online optimization")

    def run_online_optimization_step(self,
                                     x,
                                     parameters,
                                     n_train_frames=1,
                                     n_iters=1,
                                     loss_func=None):
        """
            Run online DiffMPM optimization
        """
        print("Running Dummy diff optimization online")

        # [TODO]
        x_t = x.clone()
        rho_mask = torch.ones(self.n_particles,
                              dtype=torch.bool,
                              device=self.device)
        init_E = parameters["E"]
        E_t = torch.tensor(1., dtype=DTYPE, device=self.device) * init_E
        E_t.requires_grad_()

        losses = []
        n_substeps = int(self.frame_dt / self.sim_dt)
        n_step_with_grad = n_train_frames * n_substeps
        extra_no_grad_step = 0

        optimizer = torch.optim.AdamW([E_t], lr=1e-4, weight_decay=0.)
        mpm_state = self.mpm_state.partial_clone(requires_grad=True)
        for _ in range(n_iters):
            optimizer.zero_grad()
            if self.optim_callback:
                self.optim_callback(self)
            particle_pos = MPMDifferentiableSimulationTrail.apply(
                self.mpm_solver,
                mpm_state,
                self.mpm_model,
                n_substeps,
                self.sim_dt,
                n_step_with_grad,
                x_t,
                self.v_t,
                E_t,
                self.nu_t,
                self.rho_t,
                rho_mask,
                None,
                self.device,
                True,
                extra_no_grad_step,
            )

            points = particle_pos
            loss = loss_func(points)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        optimal_param = {"E": E_t.item()}
        return {
            "param": optimal_param,
            "loss": np.mean(losses),
            "points": particle_pos.clone().detach()
        }

    def run_diff_optimization(self,
                              parameters,
                              n_train_frames,
                              max_iters,
                              loss_func=None):
        """
            Run DiffMPM optimization
        """
        if not self.initialized:
            self.initialize()

        # initialize parameters
        n_p = self.n_particles
        x_t = self.x_t.clone()
        v_t = self.v_t.clone()
        E_t = self.E_t.clone()
        nu_t = self.nu_t.clone()
        rho_t = self.rho_t.clone()
        rho_mask = torch.ones(n_p, dtype=torch.bool, device=self.device)

        trainable_params = []
        optimal_param = {}
        if "E" in parameters:
            init_E = parameters["E"]
            E_t = torch.tensor(1., dtype=DTYPE, device=self.device) * init_E
            E_t.requires_grad_()
            trainable_params.append(E_t)
            optimal_param["E"] = E_t
        if "nu" in parameters:
            init_nu = parameters["nu"]
            nu_t = torch.tensor(1., dtype=DTYPE, device=self.device) * init_nu
            nu_t.requires_grad_()
            trainable_params.append(nu_t)
            optimal_param["nu"] = nu_t
        if "rho" in parameters:
            init_rho = parameters["rho"]
            rho_t = torch.ones(n_p, dtype=DTYPE, device=self.device) * init_rho
            rho_t.requires_grad_()
            trainable_params.append(rho_t)
            optimal_param["rho"] = rho_t

        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=1e-3,
                                      weight_decay=0.)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=50,
        #                                                        eta_min=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=5)

        losses = []
        n_substeps = int(self.frame_dt / self.sim_dt)
        n_step_with_grad = n_train_frames * n_substeps
        extra_no_grad_step = 0
        with tqdm(range(max_iters), desc=f"") as pbar:
            for optim_step in pbar:
                optimizer.zero_grad()
                if self.optim_callback:
                    self.optim_callback(self)
                particle_pos = MPMDifferentiableSimulationTrail.apply(
                    self.mpm_solver,
                    self.mpm_state,
                    self.mpm_model,
                    n_substeps,
                    self.sim_dt,
                    n_step_with_grad,
                    x_t,
                    v_t,
                    E_t,
                    nu_t,
                    rho_t,
                    rho_mask,
                    None,
                    self.device,
                    True,
                    extra_no_grad_step,
                )

                # points = particle_pos
                # target = self.target_points[:n_train_frames]
                # loss = (points - target).norm()
                loss = loss_func(particle_pos)
                losses.append(loss.item())
                loss.backward()

                optimizer.step()
                scheduler.step(loss.item())

                # for group in optimizer.param_groups:
                #     for param in group['params']:
                #         # print(param.shape, param.requires_grad, param.grad)
                #         if param.ndim > 0:
                #             print(param.shape, param.requires_grad, param.grad.norm())
                points_np = particle_pos[-1].detach().cpu().numpy()
                point_cloud = pv.PolyData(points_np)
                point_cloud.save(f"outputs/optim/optim_{optim_step:04d}.ply")

                pbar.set_postfix(loss=f"{loss:.4f}")
                plt.plot(losses)
                plt.savefig('outputs/loss_plot.png')

                current_lr = optimizer.param_groups[0]['lr']
                if current_lr < 1e-5:
                    print("Early stopping criteria met")
                    break  # early stopping

        return optimal_param
