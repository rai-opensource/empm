"""
Convert the hand tracking data to gripper data format
"""
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import glob
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

# 2025-08, YH, added viser visualization, nicer
vis_tool = "viser" # "o3d"
if vis_tool == "viser":
    import viser
    import viser.transforms as tf
    from scipy.spatial.transform import Rotation
    import time
    import sys
    import select
    viser_server = viser.ViserServer()

base_path = args.base_path
case_name = args.case_name


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


# Based on the valid mask, filter out the bad tracking data
def filter_track(track_path, pcd_path, mask_path, frame_num, num_cam):
    with open(f"{mask_path}/processed_masks.pkl", "rb") as f:
        processed_masks = pickle.load(f)

    # Filter out the points not valid in the first frame
    object_points = []
    object_colors = []
    object_visibilities = []
    controller_points = []
    controller_colors = []
    controller_visibilities = []
    for i in range(num_cam):
        current_track_data = np.load(f"{track_path}/{i}.npz")
        # Filter out the track data
        tracks = current_track_data["tracks"]
        tracks = np.round(tracks).astype(int)
        visibility = current_track_data["visibility"]
        assert tracks.shape[0] == frame_num
        num_points = np.shape(tracks)[1]

        # Locate the track points in the object mask of the first frame
        object_mask = processed_masks[0][i]["object"]
        track_object_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                track_object_idx[j] = object_mask[tracks[0, j, 0], tracks[0, j, 1]]
        # Locate the controller points in the controller mask of the first frame
        controller_mask = processed_masks[0][i]["controller"]
        track_controller_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                track_controller_idx[j] = controller_mask[
                    tracks[0, j, 0], tracks[0, j, 1]
                ]

        # Filter out bad tracking in other frames
        for frame_idx in range(1, frame_num):
            # Filter based on object_mask
            object_mask = processed_masks[frame_idx][i]["object"]
            for j in range(num_points):
                try:
                    if track_object_idx[j] == 1 and visibility[frame_idx, j] == 1:
                        if not object_mask[
                            tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                        ]:
                            visibility[frame_idx, j] = 0
                except:
                    # Sometimes the track coordinate is out of image
                    visibility[frame_idx, j] = 0
            # Filter based on controller_mask
            controller_mask = processed_masks[frame_idx][i]["controller"]
            for j in range(num_points):
                if track_controller_idx[j] == 1 and visibility[frame_idx, j] == 1:
                    if not controller_mask[
                        tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                    ]:
                        visibility[frame_idx, j] = 0

        # Get the track point cloud
        track_points = np.zeros((frame_num, num_points, 3))
        track_colors = np.zeros((frame_num, num_points, 3))
        for frame_idx in range(frame_num):
            data = np.load(f"{pcd_path}/{frame_idx}.npz")
            points = data["points"]
            colors = data["colors"]

            visible_indices = np.where(visibility[frame_idx])[0]
            if len(visible_indices) > 0:
                try:
                    # Get track coordinates for visible points
                    track_coords_y = tracks[frame_idx, visible_indices, 0]
                    track_coords_x = tracks[frame_idx, visible_indices, 1]
                    
                    # Clip coordinates to valid bounds
                    H, W = points[i].shape[:2]
                    track_coords_y = np.clip(track_coords_y, 0, H-1)
                    track_coords_x = np.clip(track_coords_x, 0, W-1)
                    
                    track_points[frame_idx][visible_indices] = points[i][track_coords_y, track_coords_x]
                    track_colors[frame_idx][visible_indices] = colors[i][track_coords_y, track_coords_x]
                except IndexError as e:
                    print(f"IndexError at frame {frame_idx}, camera {i}: {e}")
                    print(f"Points shape: {points[i].shape}, Track coords range: y={track_coords_y.min()}-{track_coords_y.max()}, x={track_coords_x.min()}-{track_coords_x.max()}")
                    # Set visibility to 0 for problematic points
                    visibility[frame_idx, visible_indices] = 0

        object_points.append(track_points[:, np.where(track_object_idx)[0], :])
        object_colors.append(track_colors[:, np.where(track_object_idx)[0], :])
        object_visibilities.append(visibility[:, np.where(track_object_idx)[0]])
        controller_points.append(track_points[:, np.where(track_controller_idx)[0], :])
        controller_colors.append(track_colors[:, np.where(track_controller_idx)[0], :])
        controller_visibilities.append(visibility[:, np.where(track_controller_idx)[0]])

    object_points = np.concatenate(object_points, axis=1)
    object_colors = np.concatenate(object_colors, axis=1)
    object_visibilities = np.concatenate(object_visibilities, axis=1)
    controller_points = np.concatenate(controller_points, axis=1)
    controller_colors = np.concatenate(controller_colors, axis=1)
    controller_visibilities = np.concatenate(controller_visibilities, axis=1)

    track_data = {}
    track_data["object_points"] = object_points
    track_data["object_colors"] = object_colors
    track_data["object_visibilities"] = object_visibilities
    track_data["controller_points"] = controller_points
    track_data["controller_colors"] = controller_colors
    track_data["controller_visibilities"] = controller_visibilities

    return track_data


def filter_motion(track_data, neighbor_dist=0.01):
    # Calculate the motion of each point
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions = np.zeros_like(object_points)
    object_motions[:-1] = object_points[1:] - object_points[:-1]
    object_motions_valid = np.zeros_like(object_visibilities)
    object_motions_valid[:-1] = np.logical_and(
        object_visibilities[:-1], object_visibilities[1:]
    )

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    num_frames = object_points.shape[0]
    num_points = object_points.shape[1]

    # 2025-08, YH, viz tool setup
    if vis_tool == "o3d":
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    print(f"filtering object points...")
    for i in tqdm(range(num_frames - 1)):
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points[i])
        pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        # modified_points = []
        # new_points = []
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):
            if object_motions_valid[i, j] == 0:
                continue
            # Get the neighbors within neighbor_dist
            [k, idx, _] = kdtree.search_radius_vector_3d(
                object_points[i, j], neighbor_dist
            )
            neighbors = [index for index in idx if object_motions_valid[i, index] == 1]
            if len(neighbors) < 5:
                object_motions_valid[i, j] = 0
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])
            motion_diff = np.linalg.norm(
                object_motions[i, j] - object_motions[i, neighbors], axis=1
            )
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                object_motions_valid[i, j] = 0
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])

        motion_pcd = o3d.geometry.PointCloud()
        motion_pcd.points = o3d.utility.Vector3dVector(
            object_points[i][np.where(object_motions_valid[i])]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            object_colors[i][np.where(object_motions_valid[i])]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_motions_valid[i])]
        )

        # modified_pcd = o3d.geometry.PointCloud()
        # modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
        # modified_pcd.colors = o3d.utility.Vector3dVector(
        #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
        # )

        # new_pcd = o3d.geometry.PointCloud()
        # new_pcd.points = o3d.utility.Vector3dVector(new_points)
        # new_pcd.colors = o3d.utility.Vector3dVector(
        #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
        # )
        if i == 0:
            render_motion_pcd = motion_pcd
            if vis_tool == "o3d":
                # render_modified_pcd = modified_pcd
                # render_new_pcd = new_pcd
                vis.add_geometry(render_motion_pcd)
                # vis.add_geometry(render_modified_pcd)
                # vis.add_geometry(render_new_pcd)
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
        else:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)
            # render_modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
            # render_modified_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
            # )
            # render_new_pcd.points = o3d.utility.Vector3dVector(new_points)
            # render_new_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
            # )
            if vis_tool == "o3d":
                vis.update_geometry(render_motion_pcd)
                # vis.update_geometry(render_modified_pcd)
                # vis.update_geometry(render_new_pcd)
                vis.poll_events()
                vis.update_renderer()
        # modified_num = len(modified_points)
        # print(f"Object Frame {i}: {modified_num} points are modified")

        if vis_tool == "viser":
            viser_server.scene.add_point_cloud(
                name="/world/object_motion_pcd",
                points=np.array(render_motion_pcd.points),
                colors=np.array(render_motion_pcd.colors),
                point_size=0.0005
            )

    if vis_tool == "o3d":
        vis.destroy_window()
         
    track_data["object_motions_valid"] = object_motions_valid

    controller_points = track_data["controller_points"]
    controller_colors = track_data["controller_colors"]
    controller_visibilities = track_data["controller_visibilities"]
    controller_motions = np.zeros_like(controller_points)
    controller_motions[:-1] = controller_points[1:] - controller_points[:-1]
    controller_motions_valid = np.zeros_like(controller_visibilities)
    controller_motions_valid[:-1] = np.logical_and(
        controller_visibilities[:-1], controller_visibilities[1:]
    )
    num_points = controller_points.shape[1]
    # Filter all points that disappear in the sequence
    mask = np.prod(controller_visibilities, axis=0)

    y_min, y_max = np.min(controller_points[0, :, 1]), np.max(
        controller_points[0, :, 1]
    )
    y_normalized = (controller_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    if vis_tool == "o3d":
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    print(f"filtering controller points...")
    for i in tqdm(range(num_frames - 1)):
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points[i])
        pcd.colors = o3d.utility.Vector3dVector(controller_colors[i])
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):
            if mask[j] == 0:
                controller_motions_valid[i, j] = 0
            if controller_motions_valid[i, j] == 0:
                continue
            # Get the neighbors within neighbor_dist
            [k, idx, _] = kdtree.search_radius_vector_3d(
                controller_points[i, j], neighbor_dist
            )
            neighbors = [
                index for index in idx if controller_motions_valid[i, index] == 1
            ]
            if len(neighbors) < 5:
                controller_motions_valid[i, j] = 0
                mask[j] = 0

            motion_diff = np.linalg.norm(
                controller_motions[i, j] - controller_motions[i, neighbors], axis=1
            )
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                controller_motions_valid[i, j] = 0
                mask[j] = 0

        motion_pcd = o3d.geometry.PointCloud()
        motion_pcd.points = o3d.utility.Vector3dVector(
            controller_points[i][np.where(mask)]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            controller_colors[i][np.where(controller_motions_valid[i])]
        )

        if i == 0:
            render_motion_pcd = motion_pcd
            if vis_tool == "o3d":
                vis.add_geometry(render_motion_pcd)
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
        else:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)
            if vis_tool == "o3d":
                vis.update_geometry(render_motion_pcd)
                vis.poll_events()
                vis.update_renderer()
        
        if vis_tool == "viser":
            viser_server.scene.add_point_cloud(
                name="/world/controller_motion_pcd",
                points=np.array(render_motion_pcd.points),
                colors=np.array(render_motion_pcd.colors),
                point_size=0.0005
            )

    track_data["controller_mask"] = mask
    return track_data


def get_final_track_data(track_data, controller_threhsold=0.01):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]
    mask = track_data["controller_mask"]

    new_controller_points = controller_points[:, np.where(mask)[0], :]
    assert len(new_controller_points[0]) >= 30
    # Do farthest point sampling on the valid controller points to select the final controller points
    valid_indices = np.arange(len(new_controller_points[0]))
    points_map = {}
    sample_points = []
    for i in valid_indices:
        points_map[tuple(new_controller_points[0, i])] = i
        sample_points.append(new_controller_points[0, i])
    sample_points = np.array(sample_points)
    sample_pcd = o3d.geometry.PointCloud()
    sample_pcd.points = o3d.utility.Vector3dVector(sample_points)
    fps_pcd = sample_pcd.farthest_point_down_sample(30)
    final_indices = []
    for point in fps_pcd.points:
        final_indices.append(points_map[tuple(point)])

    print(f"Controller Point Number: {len(final_indices)}")

    # Get the nearest controller points and their colors
    nearest_controller_points = new_controller_points[:, final_indices]

    # object_pcd = o3d.geometry.PointCloud()
    # object_pcd.points = o3d.utility.Vector3dVector(valid_object_points)
    # object_pcd.colors = o3d.utility.Vector3dVector(
    #     object_colors[0][np.where(object_motions_valid[0])]
    # )
    # controller_meshes = []
    # for j in range(nearest_controller_points.shape[1]):
    #     origin = nearest_controller_points[0, j]
    #     origin_color = [1, 0, 0]
    #     controller_meshes.append(
    #         getSphereMesh(origin, color=origin_color, radius=0.005)
    #     )
    # o3d.visualization.draw_geometries([object_pcd])
    # o3d.visualization.draw_geometries([object_pcd] + controller_meshes)

    track_data.pop("controller_points")
    track_data.pop("controller_colors")
    track_data.pop("controller_visibilities")
    track_data["controller_points"] = nearest_controller_points

    return track_data


def visualize_track(track_data):
    print(f"visualizing final track data...")

    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    if vis_tool == "o3d":
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_motions_valid[i])[0], :]
        )
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_motions_valid[i])[0]]
        )

        if i == 0:
            render_object_pcd = object_pcd
            if vis_tool == "o3d":
                vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                if vis_tool == "o3d":
                    vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            if vis_tool == "o3d":
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            if vis_tool == "o3d":
                vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                if vis_tool == "o3d":
                    vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            if vis_tool == "o3d":
                vis.poll_events()
                vis.update_renderer()
        
        # 2025-09, YH, viz final filtered tracked pcd and mesh
        if vis_tool == "viser":
            viser_server.scene.add_point_cloud(
                name="/world/final_object_pcd",
                points=np.array(render_object_pcd.points),
                colors=np.array(render_object_pcd.colors),
                point_size=0.0005
            )

            for j in range(controller_points.shape[1]):
                viser_server.scene.add_mesh_simple(
                    name=f"/world/controller_meshes/{j}",
                    vertices=np.array(controller_meshes[j].vertices),
                    faces=np.array(controller_meshes[j].triangles)
                )
    

    # ====================Get two hand pose from two hand points and save for PGND training==================== #
    controller_points = track_data["controller_points"]
    
    # K-means clustering to separate two hands
    from sklearn.cluster import KMeans
    
    # Cluster the controller points into 2 groups (left/right hand)
    points_first_frame = controller_points[0]  # Use first frame for clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    hand_labels = kmeans.fit_predict(points_first_frame)
    
    # Split points by hand
    hand1_indices = np.where(hand_labels == 0)[0]
    hand2_indices = np.where(hand_labels == 1)[0]
    
    hand1_points = controller_points[:, hand1_indices, :]
    hand2_points = controller_points[:, hand2_indices, :]
    
    print(f"Hand 1: {len(hand1_indices)} points, Hand 2: {len(hand2_indices)} points")
    
    hand_poses = []
    hand_velocities = []
    
    for frame_idx in range(controller_points.shape[0]):
        frame_poses = []
        
        # Calculate pose for each hand
        for hand_points in [hand1_points[frame_idx], hand2_points[frame_idx]]:
            # Position: center of points
            position = np.mean(hand_points, axis=0)
            
            # Orientation: find main direction of hand
            distances = np.linalg.norm(hand_points - position, axis=1)
            farthest_idx = np.argmax(distances)
            main_direction = hand_points[farthest_idx] - position
            main_direction = main_direction / np.linalg.norm(main_direction)
            
            # Simple rotation matrix
            x_axis = main_direction
            z_axis = np.array([0, 0, 1])  
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            frame_poses.append({
                'position': position,
                'rotation_matrix': rotation_matrix
            })
        
        hand_poses.append(frame_poses)
        
        # Compute velocities (skip first frame)
        if frame_idx > 0:
            frame_velocities = []
            
            for hand_idx in range(2):  # For both hands
                current_pose = hand_poses[frame_idx - 1][hand_idx]
                next_pose = hand_poses[frame_idx][hand_idx]
                
                # Linear velocity
                linear_vel = next_pose['position'] - current_pose['position']
                
                # Angular velocity
                R1 = current_pose['rotation_matrix']
                R2 = next_pose['rotation_matrix']
                R_diff = R2 @ R1.T
                
                from scipy.spatial.transform import Rotation as R_scipy
                r = R_scipy.from_matrix(R_diff)
                angular_vel = r.as_rotvec()
                
                frame_velocities.append({
                    'linear_velocity': linear_vel,
                    'angular_velocity': angular_vel
                })
            
            hand_velocities.append(frame_velocities)
    
    print(f"Computed {len(hand_poses)} poses and {len(hand_velocities)} velocities")
    
    # Convert to grippers format - vectorized
    n_frames = len(hand_poses)
    grippers = np.zeros((n_frames, 2, 15))
    
    # Extract positions and rotations
    positions = np.array([[pose['position'] for pose in frame] for frame in hand_poses])  # (n_frames, 2, 3)
    rotations = np.array([[pose['rotation_matrix'] for pose in frame] for frame in hand_poses])  # (n_frames, 2, 3, 3)
    
    # Fill grippers array
    grippers[:, :, :3] = positions  # positions
    grippers[:, :, 13] = 0.02       # radius
    grippers[:, :, 14] = 0.0        # gripper state (always open)
    
    # Convert rotations to quaternions
    from scipy.spatial.transform import Rotation as R_scipy
    for frame_idx in range(n_frames):
        for hand_idx in range(2):
            quat = R_scipy.from_matrix(rotations[frame_idx, hand_idx]).as_quat(scalar_first=True)
            grippers[frame_idx, hand_idx, 6:10] = quat
    
    # Fill velocities
    if hand_velocities:
        velocities = np.array([[vel['linear_velocity'] for vel in frame] for frame in hand_velocities])
        angular_vels = np.array([[vel['angular_velocity'] for vel in frame] for frame in hand_velocities])
        grippers[1:, :, 3:6] = velocities    # linear velocities
        grippers[1:, :, 10:13] = angular_vels # angular velocities
    
    # ========== save the data to PGND format ========== #
    import shutil
    import json
    import pickle
    
    source_dir = os.path.join(base_path, case_name)
    converted_data_dir = os.path.join(base_path, case_name + "_pgnd")
    if not os.path.exists(converted_data_dir):
        os.makedirs(converted_data_dir)

    # copy rgbd images
    for i in range(3):
        curr_source_rgb_dir = os.path.join(source_dir, "color", f"{i}")
        curr_source_depth_dir = os.path.join(source_dir, "depth", f"{i}")
        curr_converted_rgb_dir = os.path.join(converted_data_dir, "recording_1", "0000000000", f"camera_{i}", "rgb")
        curr_converted_depth_dir = os.path.join(converted_data_dir, "recording_1", "0000000000", f"camera_{i}", "depth")

        if not os.path.exists(curr_converted_rgb_dir):
            shutil.copytree(curr_source_rgb_dir, curr_converted_rgb_dir)
        if not os.path.exists(curr_converted_depth_dir):
            shutil.copytree(curr_source_depth_dir, curr_converted_depth_dir)
    
    # convert and copy calibration data
    calibration_dir = os.path.join(converted_data_dir, "recording_1", "0000000000", "calibration")
    if not os.path.exists(calibration_dir):
        os.makedirs(calibration_dir)
    
    # Load original calibration data
    with open(os.path.join(source_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    with open(os.path.join(source_dir, "calibrate.pkl"), "rb") as f:
        T_wcs = pickle.load(f)
    
    # Convert to expected format
    intrinsics = np.array(metadata["intrinsics"]).astype(np.float32)  # (num_cams, 3, 3)
    
    # Convert c2w to rvec and tvec format
    num_cams = len(T_wcs)
    rvecs = np.zeros((num_cams, 3, 1))
    tvecs = np.zeros((num_cams, 3, 1))
    
    for i in range(num_cams):
        T_wc = T_wcs[i]
        T_cw = np.linalg.inv(T_wc)
        R = T_cw[:3, :3]
        t = T_cw[:3, 3]
        
        # Convert rotation matrix to rodrigues vector
        import cv2
        rvec, _ = cv2.Rodrigues(R)
        rvecs[i] = rvec
        tvecs[i] = t.reshape(3, 1)
    
    # Save calibration files to npy
    np.save(os.path.join(calibration_dir, "intrinsics.npy"), intrinsics)
    np.save(os.path.join(calibration_dir, "rvecs.npy"), rvecs)
    np.save(os.path.join(calibration_dir, "tvecs.npy"), tvecs)

    # also save to pickle as rvecs and tvecs.pkl
    with open(os.path.join(calibration_dir, "rvecs.pkl"), "wb") as f:
        pickle.dump(rvecs, f)
    with open(os.path.join(calibration_dir, "tvecs.pkl"), "wb") as f:
        pickle.dump(tvecs, f)
    
    print(f"Converted calibration data: {num_cams} cameras")
    print(f"Intrinsics shape: {intrinsics.shape}")
    print(f"Rvecs shape: {rvecs.shape}")
    print(f"Tvecs shape: {tvecs.shape}")
    
    # Convert and save robot data
    robot_dir = os.path.join(converted_data_dir, "recording_1", "robot")
    if not os.path.exists(robot_dir):
        os.makedirs(robot_dir)
    
    # Load gripper data
    n_frames, num_grippers = grippers.shape[:2]
    
    print(f"Converting robot data: {n_frames} frames, {num_grippers} grippers")
    
    for frame_idx in range(n_frames):
        robot_data = []
        
        for gripper_idx in range(num_grippers):
            # Extract position and quaternion
            position = grippers[frame_idx, gripper_idx, :3]  # (3,)
            quat_wxyz = grippers[frame_idx, gripper_idx, 6:10]  # (w, x, y, z)
            
            # Convert quaternion to rotation matrix
            from scipy.spatial.transform import Rotation as R_scipy
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # Convert to (x,y,z,w)
            rotation_matrix = R_scipy.from_quat(quat_xyzw).as_matrix()  # (3, 3)
            
            # Add position (1 row of 3 values)
            robot_data.append(position)
            
            # Add rotation matrix (3 rows of 3 values each)
            for row in range(3):
                robot_data.append(rotation_matrix[row])
        
        # Add gripper states (1 row with 3 values)
        gripper_states = grippers[frame_idx, :, 14]  # Gripper states for both hands
        gripper_line = np.array([gripper_states[0], gripper_states[1], 0.0])  # 3 values
        robot_data.append(gripper_line)
        
        # Stack all rows to get 9x3 matrix
        robot_matrix = np.vstack(robot_data)  # 9 rows x 3 cols
        
        # Save robot data for this frame
        robot_file = os.path.join(robot_dir, f"{frame_idx:06d}.txt")
        np.savetxt(robot_file, robot_matrix, fmt='%.6f')
    
    print(f"Saved robot data: {n_frames} files in {robot_dir}")
    print(f"Robot data format: 9 rows x 3 cols per frame")

    # user viser to visualize the hand poses
    if vis_tool == "viser":
        for frame_idx, frame_poses in enumerate(hand_poses):
            for hand_idx, pose in enumerate(frame_poses):
                viser_server.scene.add_frame(
                    name=f"/world/hand_{hand_idx}_frame_{frame_idx}",
                    wxyz=Rotation.from_matrix(pose['rotation_matrix']).as_quat(scalar_first=True),
                    position=pose['position'],
                    axes_length=0.05,
                    axes_radius=0.005,
                )


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"
    track_path = f"{base_path}/{case_name}/cotracker"

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))

    # Filter the track data using the semantic mask of object and controller
    track_data = filter_track(track_path, pcd_path, mask_path, frame_num, num_cam)
    # Filter motion
    track_data = filter_motion(track_data)
    # # Save the filtered track data
    # with open(f"test2.pkl", "wb") as f:
    #     pickle.dump(track_data, f)

    # with open(f"test2.pkl", "rb") as f:
    #     track_data = pickle.load(f)

    track_data = get_final_track_data(track_data)

    with open(f"{base_path}/{case_name}/track_process_data.pkl", "wb") as f:
        pickle.dump(track_data, f)

    visualize_track(track_data)

    # 2025-08, YH, put a while lool and press enter to exit and continue to other processing
    if vis_tool == "viser":
        print("Press Enter to exit and continue...")
        while True:
            time.sleep(0.1)
            # check if user pressed Enter
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                break