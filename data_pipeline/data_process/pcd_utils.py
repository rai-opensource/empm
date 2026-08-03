import numpy as np
import open3d as o3d
import json
import pickle
import cv2
from tqdm import tqdm
import os
import glob


vis_tool = "viser" # "o3d"


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/camera/cameraHelper.py
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/model/pcdHelper.py
def getPcdFromDepth(
    depth,
    intrinsic,
):
    # Depth in meters
    height, width = np.shape(depth)

    # Reshape the depth array to invert the depth values
    depth = -depth

    # Create a grid of (x, y) coordinates
    x_coords = np.arange(width)
    y_coords = np.arange(height)

    # Create a meshgrid for x and y coordinates
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate points using vectorized operations
    old_points = np.stack([(width - X) * depth, Y * depth, depth], axis=-1)

    # Flatten the old_points array and calculate the new points using matrix multiplication
    points = np.dot(np.linalg.inv(intrinsic), old_points.reshape(-1, 3).T).T.reshape(
        old_points.shape
    )

    points[:, :, 1] *= -1
    points[:, :, 2] *= -1

    return points


def get_pcd_from_data(path, frame_idx, num_cam, intrinsics, c2ws):
    total_points = []
    total_colors = []
    total_masks = []
    for i in range(num_cam):
        color = cv2.imread(f"{path}/color/{i}/{frame_idx}.png")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0
        depth = np.load(f"{path}/depth/{i}/{frame_idx}.npy") / 1000.0

        points = getPcdFromDepth(
            depth,
            intrinsic=intrinsics[i],
        )
        if i == 1:  # [TODO]
            points += [0.025, 0.01, 0.0]
        masks = np.logical_and(points[:, :, 2] > 0.2, points[:, :, 2] < 1.5)
        points_flat = points.reshape(-1, 3)
        # Transform points to world coordinates using homogeneous transformation
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )
        points_world = np.dot(c2ws[i], homogeneous_points.T).T[:, :3]
        points_final = points_world.reshape(points.shape)
        total_points.append(points_final)
        total_colors.append(color)
        total_masks.append(masks)

    total_points = np.asarray(total_points)
    total_colors = np.asarray(total_colors)
    total_masks = np.asarray(total_masks)
    return total_points, total_colors, total_masks




def lift_pcd(base_path: str, case_name: str):
    """
        merge the RGB-D data from multiple cameras into a single point cloud in world coordinate,
        do some depth filtering to make the point cloud more clean
    """
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    WH = data["WH"]
    frame_num = data["frame_num"]
    num_cam = len(intrinsics)

    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))

    # get the height and width of the images
    depth = np.load(f"{base_path}/{case_name}/depth/0/0.npy") / 1000.0
    height, width = np.shape(depth)

    os.makedirs(f"{base_path}/{case_name}/pcd", exist_ok=True)

    if vis_tool == "o3d":
        cameras = []
        # Visualize the cameras
        for i in range(num_cam):
            camera = getCamera(
                c2ws[i],
                intrinsics[i, 0, 0],
                intrinsics[i, 1, 1],
                intrinsics[i, 0, 2],
                intrinsics[i, 1, 2],
                z_flip=True,
                scale=0.2,
            )
            cameras += camera

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for camera in cameras:
            vis.add_geometry(camera)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(coordinate)
    
    elif vis_tool == "viser":
        import viser
        import viser.transforms as tf
        from scipy.spatial.transform import Rotation
        import time
        import sys
        import select

        viser_server = viser.ViserServer()

        for i in range(num_cam):
            T_wc_i = c2ws[i]
            # add cam frame
            viser_server.scene.add_frame(
                f"/world/cameras_{i}",
                wxyz=Rotation.from_matrix(T_wc_i[:3, :3]).as_quat(scalar_first=True),
                position=T_wc_i[:3, 3],
                axes_length=0.1,
                axes_radius=0.005,
           )
        
            # Add the visual frustum for the camera
            K = intrinsics[i]
            fov = 2 * np.arctan2(height, 2 * K[1,1]) * 180 / np.pi
            aspect = width / height
            viser_server.scene.add_camera_frustum(
                name=f"/world/camera_frustum_{i}",
                fov=fov,
                aspect=aspect,
                scale=0.1,
                line_width=6.0,
                wxyz=Rotation.from_matrix(T_wc_i[:3, :3]).as_quat(scalar_first=True),
                position=T_wc_i[:3, 3],
                visible=True)

    pcd = None
    for i in tqdm(range(frame_num)):
        points, colors, masks = get_pcd_from_data(
            f"{base_path}/{case_name}", i, num_cam, intrinsics, c2ws
        )

        if i == 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )
            if vis_tool == "o3d":
                vis.add_geometry(pcd)
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
        else:
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )
            if vis_tool == "o3d":
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
        
        if vis_tool == "viser": 
            viser_server.scene.add_point_cloud(
                name="/world/point_cloud",
                # points=np.array(voxel_grid.vox_centers),
                points=np.array(pcd.points),
                colors=np.array(pcd.colors),
                point_size=0.0005
            )


        np.savez(
            f"{base_path}/{case_name}/pcd/{i}.npz",
            points=points,
            colors=colors,
            masks=masks,
        )

    if vis_tool == "viser":
        print("Press Enter to exit and continue...")
        while True:
            time.sleep(0.1)
            # check if user pressed Enter
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                break


processed_masks = {}

def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def process_pcd_mask(frame_idx, pcd_path, mask_path, mask_info, num_cam):
    global processed_masks
    processed_masks[frame_idx] = {}

    # Load the pcd data
    data = np.load(f"{pcd_path}/{frame_idx}.npz")
    points = data["points"]
    colors = data["colors"]
    masks = data["masks"]

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()

    for i in range(num_cam):
        # Load the object mask
        object_idx = mask_info[i]["object"]
        mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
        object_mask = np.logical_and(masks[i], mask)
        object_points = points[i][object_mask]
        object_colors = colors[i][object_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd.colors = o3d.utility.Vector3dVector(object_colors)
        object_pcd += pcd

        # Load the controller mask
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        controller_colors = colors[i][controller_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points)
        pcd.colors = o3d.utility.Vector3dVector(controller_colors)
        controller_pcd += pcd

    # Apply the outlier removal
    cl, ind = object_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_object_points = np.asarray(
        object_pcd.select_by_index(ind, invert=True).points
    )
    object_pcd = object_pcd.select_by_index(ind)

    cl, ind = controller_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_controller_points = np.asarray(
        controller_pcd.select_by_index(ind, invert=True).points
    )
    controller_pcd = controller_pcd.select_by_index(ind)

    # controller_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])
    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()
    for i in range(num_cam):
        processed_masks[frame_idx][i] = {}
        # Load the object mask
        object_idx = mask_info[i]["object"]
        mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
        object_mask = np.logical_and(masks[i], mask)
        object_points = points[i][object_mask]
        indices = np.nonzero(object_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the object_points in the filtered points
        object_indices = []
        for j, point in enumerate(object_points):
            if tuple(point) in filtered_object_points:
                object_indices.append(j)
        original_indices = [indices_list[j] for j in object_indices]
        # Update the object mask
        for idx in original_indices:
            object_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["object"] = object_mask

        # Load the controller mask
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        indices = np.nonzero(controller_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the controller_points in the filtered points
        controller_indices = []
        for j, point in enumerate(controller_points):
            if tuple(point) in filtered_controller_points:
                controller_indices.append(j)
        original_indices = [indices_list[j] for j in controller_indices]
        # Update the controller mask
        for idx in original_indices:
            controller_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["controller"] = controller_mask

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][object_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][object_mask])

        object_pcd += pcd

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][controller_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][controller_mask])

        controller_pcd += pcd

    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])

    return object_pcd, controller_pcd


def process_mask(base_path: str, case_name: str, controller_name: str):
    """
        process the mask data to filter out the outliers and generate the processed masks
    """
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))
    # Load the mask metadata
    mask_info = {}
    for i in range(num_cam):
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        mask_info[i] = {}
        for key, value in data.items():
            if value != controller_name:
                if "object" in mask_info[i]:
                    raise ValueError(
                        f"Multiple objects found in mask metadata for camera {i}"
                    )
                mask_info[i]["object"] = int(key)
            if value == controller_name:
                if "controller" in mask_info[i]:
                    mask_info[i]["controller"].append(int(key))
                else:
                    mask_info[i]["controller"] = [int(key)]

    if vis_tool == "viser":
        import viser
        import viser.transforms as tf
        from scipy.spatial.transform import Rotation
        import time
        import sys
        import select
        viser_server = viser.ViserServer()

    elif vis_tool == "o3d":
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    object_pcd = None
    controller_pcd = None
    for i in tqdm(range(frame_num)):
        temp_object_pcd, temp_controller_pcd = process_pcd_mask(
            i, pcd_path, mask_path, mask_info, num_cam
        )
        if i == 0:
            object_pcd = temp_object_pcd
            controller_pcd = temp_controller_pcd
            if vis_tool == "o3d":
                vis.add_geometry(object_pcd)
                vis.add_geometry(controller_pcd)
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
            
            # save the initial object and controller pcd
            o3d.io.write_point_cloud(f"{base_path}/{case_name}/pcd/object_pcd_0.ply",
                                    object_pcd)
            o3d.io.write_point_cloud(f"{base_path}/{case_name}/pcd/controller_pcd_0.ply", 
                                    controller_pcd)

        else:
            object_pcd.points = o3d.utility.Vector3dVector(temp_object_pcd.points)
            object_pcd.colors = o3d.utility.Vector3dVector(temp_object_pcd.colors)
            controller_pcd.points = o3d.utility.Vector3dVector(
                temp_controller_pcd.points
            )
            controller_pcd.colors = o3d.utility.Vector3dVector(
                temp_controller_pcd.colors
            )
            if vis_tool == "o3d":
                vis.update_geometry(object_pcd)
                vis.update_geometry(controller_pcd)
                vis.poll_events()
                vis.update_renderer()
        
        if vis_tool == "viser": 
            viser_server.scene.add_point_cloud(
                name="/world/object_pcd",
                points=np.array(object_pcd.points),
                colors=np.array(object_pcd.colors),
                point_shape="circle",
                point_size=0.002
            )
            # add a little red to the controller pcd
            controller_pcd_colors = np.array(controller_pcd.colors) + np.array([1.0, 0, 0])
            controller_pcd_colors = np.clip(controller_pcd_colors, 0, 1)
            viser_server.scene.add_point_cloud(
                name="/world/controller_pcd",
                points=np.array(controller_pcd.points),
                colors=controller_pcd_colors,
                point_size=0.0005
            )

    # Save the processed masks considering both depth filter, semantic filter and outlier filter
    with open(f"{base_path}/{case_name}/mask/processed_masks.pkl", "wb") as f:
        pickle.dump(processed_masks, f)
    
    if vis_tool == "viser":
        print("Press Enter to exit and continue...")
        while True:
            time.sleep(0.1)
            # check if user pressed Enter
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                break
