import torch
import numpy as np
import sys
import yaml

with open("configs/experiments.yaml", "r") as f:
    config = yaml.safe_load(f)
    DATASET = config["data_path"].split("/")[-1]

cscale = 1.0  # 2.5 for bluey and online exps
SLOTH_CROP_CENTER = 0.505
SLOTH_CROP_TRANSITION = 0.005


def linear_crop(x, xbar, theta):
    """
        Moves x[:, 2] to above xbar
    """
    mask = x[:, 2] < (xbar + theta)
    if mask.any():
        xmin = x[mask, 2].min()
        x[mask, 2] = xbar + (x[mask, 2] - xmin) / (xbar + theta - xmin) * theta

    return x


def transform_points(x, name=""):
    if DATASET == "ours":  # custom data
        xx = (torch.tensor([.5, .5, .5], device=x.device) + x / 5. * cscale)
        if "sloth" in name:
            xx = linear_crop(
                xx, xbar=SLOTH_CROP_CENTER, theta=SLOTH_CROP_TRANSITION
            )
        return xx
    elif DATASET == "phystwin":  # phystwin data
        return (torch.tensor([.5, .5, .5], device=x.device) - x / 5.)
    else:
        print("[Error] dataset not defined.")


def inv_transform_points(x, name=""):
    if DATASET == "ours":  # custom data
        return (-torch.tensor([.5, .5, .5], device=x.device) + x) * 5. / cscale
    elif DATASET == "phystwin":  # phystwin data
        return (torch.tensor([.5, .5, .5], device=x.device) - x) * 5.
    else:
        print("[Error] dataset not defined.")


def initial_param_guess(name=""):
    if "dough" in name:
        return {
            "type": "plasticine",
            "E": 0.001,
            "nu": 0.2,
            "ys": 1.5e3,
        }
    elif "playdoh" in name:
        return {
            "type": "plasticine",
            "E": 0.008,
            "nu": 0.3,
            "rho": 2e3,
            "ys": 100.,
        }
    elif "pita" in name:
        return {
            "type": "plasticine",
            "E": 0.01,
            "nu": 0.3,
            "rho": 2e3,
            "ys": 100.,
        }
    elif "cloth" in name:
        return {
            "type": "jelly",
            "E": 0.03,
            "nu": 0.3,
            "rho": 2e3,
        }
    else:
        return {
            "type": "jelly",
            "E": 0.008,
            "nu": 0.3,
            "rho": 2e3,
        }


def get_param(name=""):
    if "dough" in name:
        return {
            "type": "plasticine",
            "E": 0.001,
            "nu": 0.2,
            "ys": 1.5e3,  # 900. for fracture
        }
    elif "playdoh" in name:
        return {
            "type": "plasticine",
            "E": 0.01,
            "nu": 0.2,
            "ys": 10000.,
            "rho": 500.,
        }
    elif "pita" in name:
        return {
            "type": "plasticine",
            "E": 0.01,
            "nu": 0.3,
            "rho": 2e3,
            "ys": 100.,
        }
    else:
        return {
            "type": "jelly",
            "E": 0.05,
            "nu": 0.3,
            "rho": 2e3,
        }


def modify_handler(ctrl_param, name=""):
    if name == "double_stretch_dough_test":
        ctrl_param["handler_pos"] = [(0.59, 0.462, 0.53), (0.59, 0.43, 0.53)]
        ctrl_param["vscale"] = [[80, 80, 80], [80, 80, 80]]
    elif name == "double_lift_sloth_test":
        ctrl_param["vscale"][1] = [100, 100, 100]
    elif name == "double_compress_playdoh_test":
        ctrl_param["handler_pos"] = [(0.58, 0.461, 0.50), (0.59, 0.429, 0.50)]
        ctrl_param["csize"] = [[0.03, 0.01, 0.015] for _ in range(2)]
        ctrl_param["vscale"] = [[80, 80, 80] for _ in range(2)]
    elif name == "double_tear_pita_test":
        ctrl_param["handler_pos"] = [(0.556, 0.47, 0.50), (0.556, 0.43, 0.50)]
        ctrl_param["csize"] = [[0.015, 0.02, 0.015] for _ in range(2)]
    elif name == "double_lift_cloth_test":
        ctrl_param["handler_pos"] = [(.545, .482, .505), (.545, .416, .505)]
        ctrl_param["vscale"][0] = [100, 100, 110]
    elif name == "double_hand_rope_test":
        ctrl_param["handler_pos"] = [(.56, .50, .54), (.56, .39, .54)]
    elif name == "single_poke_bluey_test":
        ctrl_param["handler_pos"] = [(0.68, 0.38, 0.6)]
    print("Handler Pos:", ctrl_param["handler_pos"])
    # if name == "double_lift_sloth" or name == "double_lift_sloth_2":
    #     handler_pos[0] = (.425, .495, .51)
    #     handler_pos[1] = (.425, .445, .51)
    return ctrl_param


def project_point_cloud(points_3d, w2c, intrinsic):
    N = points_3d.shape[0]

    # Step 1: Homogeneous coordinates
    ones = torch.ones((N, 1), dtype=points_3d.dtype, device=points_3d.device)
    pos_h = torch.cat([points_3d, ones], dim=1)  # (N, 4)

    # Step 2: World to camera
    cam_points_h = (w2c @ pos_h.T).T  # (N, 4)
    cam_points = cam_points_h[:, :3]  # (N, 3)

    # Step 3: Project to image plane
    pixels_h = (intrinsic @ cam_points.T).T  # (N, 3)
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]  # (N, 2)

    # # Step 4: Round and clip
    # pixels = torch.round(pixels).to(torch.int64)
    # pixels[:, 0] = torch.clamp(pixels[:, 0], 0, W - 1)
    # pixels[:, 1] = torch.clamp(pixels[:, 1], 0, H - 1)

    # #----- Numpy version (deprecated)
    # points_3d = points_3d.detach().cpu().numpy()
    # N = points_3d.shape[0]
    # # Homogeneous coordinates
    # pos_h = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
    # # World to camera
    # cam_points = (w2c @ pos_h.T).T[:, :3]  # (N, 3)
    # # Project to image plane
    # pixels = (intrs @ cam_points.T).T  # (N, 3)
    # pixels = pixels[:, :2] / pixels[:, 2:3]  # (N, 2)
    # # Round and clip to image size
    # pixels = np.round(pixels).astype(int)
    # pixels[:, 0] = np.clip(pixels[:, 0], 0, W - 1)
    # pixels[:, 1] = np.clip(pixels[:, 1], 0, H - 1)

    return pixels  # (N, 2) pixel coordinates


def unproject_point_cloud(points_2d, depth, intrinsic, w2c):
    """
    points_2d: torch.Tensor of shape (N, 2) -- pixel coordinates
    depth: torch.Tensor of shape (N,) -- depth value for each pixel
    intrinsic: torch.Tensor of shape (3, 3)
    w2c: torch.Tensor of shape (4, 4) (world-to-camera)
    Returns: torch.Tensor of shape (N, 3) -- 3D points in world coordinates
    """
    depth /= 1000.  # convert mm to meter
    N = points_2d.shape[0]
    # Step 1: Convert 2D pixels to normalized camera coordinates
    ones = torch.ones((N, 1), dtype=points_2d.dtype, device=points_2d.device)
    pixels_h = torch.cat([points_2d, ones], dim=1)  # (N, 3)
    # Step 2: Multiply by inverse intrinsic to get direction
    intrinsic_inv = torch.inverse(intrinsic)
    cam_dirs = (intrinsic_inv @ pixels_h.T).T  # (N, 3)
    # Step 3: Scale by depth to get camera coordinates
    cam_points = cam_dirs * depth.unsqueeze(1)  # (N, 3)
    # Step 4: Convert to homogeneous coordinates
    cam_points_h = torch.cat(
        [cam_points, torch.ones(
            (N, 1), device=cam_points.device)], dim=1)  # (N, 4)
    # Step 5: Transform from camera to world coordinates
    c2w = torch.inverse(w2c)
    world_points_h = (c2w @ cam_points_h.T).T  # (N, 4)
    world_points = world_points_h[:, :3] / world_points_h[:, 3:4]
    return world_points  # (N, 3)


# def getPcdFromDepth(depth, intrinsic):
#     # Depth in meters
#     height, width = np.shape(depth)
#     # Reshape the depth array to invert the depth values
#     depth = -depth
#     # Create a grid of (x, y) coordinates
#     x_coords = np.arange(width)
#     y_coords = np.arange(height)
#     # Create a meshgrid for x and y coordinates
#     X, Y = np.meshgrid(x_coords, y_coords)
#     # Calculate points using vectorized operations
#     old_points = np.stack([(width - X) * depth, Y * depth, depth], axis=-1)
#     # Flatten the old_points array and calculate the new points using matrix multiplication
#     points = np.dot(np.linalg.inv(intrinsic),
#                     old_points.reshape(-1, 3).T).T.reshape(old_points.shape)
#     points[:, :, 1] *= -1
#     points[:, :, 2] *= -1
#     return points


def densify_point_cloud(points, K):
    """
    points: torch.Tensor of shape (N, 3)
    K: number of additional points to sample
    Returns: torch.Tensor of shape (N+K, 3)
    """
    N = points.shape[0]
    # Randomly sample pairs of points
    idx1 = torch.randint(0, N, (K, ))
    idx2 = torch.randint(0, N, (K, ))
    # Interpolate between pairs with random weights
    alpha = torch.rand(K, 1).to(points.device)
    new_points = points[idx1] * alpha + points[idx2] * (1 - alpha)
    # Concatenate original and new points
    dense_points = torch.cat([points, new_points], dim=0)
    return dense_points
