import os
import json
import pickle
import numpy as np
import open3d as o3d
import yaml
import sys
import shutil
from pathlib import Path
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_pipeline.data_process.segment_utils import segment_image

parser = ArgumentParser(description="Export Gaussian scene data.")
parser.add_argument("-exp_config", "--exp_config", "--exp-config", default="configs/experiments.yaml",
                    help="Path to the experiments YAML file.")
args = parser.parse_args()

with open(args.exp_config, "r") as f:
    _config = yaml.safe_load(f)
    DATA_PATH = _config["data_path"]

base_path = f"{DATA_PATH}/data/different_types"
output_path = f"{DATA_PATH}/data/gaussian_data"
CONTROLLER_NAME = "hand"
num_cams = 3

os.makedirs(output_path, exist_ok=True)


if __name__ == "__main__":
    for exp in _config["experiments"]:
        case_name = exp["case_name"]
        category = exp["category"]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        print(f"Processing {case_name} ...")

        # Create the directory for the case
        os.makedirs(f"{output_path}/{case_name}", exist_ok=True)
        for i in range(num_cams):
            # Copy the original RGB image
            shutil.copy2(
                Path(base_path) / case_name / "color" / str(i) / "0.png",
                Path(output_path) / case_name / f"{i}.png",
            )

            # Copy the original mask image
            # Get the mask path for the image
            with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_idx = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)
            mask_path = f"{base_path}/{case_name}/mask/{i}/{obj_idx}/0.png"
            shutil.copy2(
                mask_path, Path(output_path) / case_name / f"mask_{i}.png"
            )

            segment_image(img_path=f"{output_path}/{case_name}/{i}.png", 
                        text_prompt=category,
                        output_path=f"{output_path}/{case_name}/mask_{i}.png")

            # Copy the original depth image
            shutil.copy2(
                Path(base_path) / case_name / "depth" / str(i) / "0.npy",
                Path(output_path) / case_name / f"{i}_depth.npy",
            )

            # prepare the human mask
            segment_image(img_path=f"{output_path}/{case_name}/{i}.png", 
                        text_prompt="person",
                        output_path=f"{output_path}/{case_name}/mask_human_{i}.png")

    # Prepare the intrinsic and extrinsic parameters
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        intrinsics = json.load(f)["intrinsics"]
    data = {}
    data["c2ws"] = c2ws
    data["intrinsics"] = intrinsics
    with open(f"{output_path}/{case_name}/camera_meta.pkl", "wb") as f:
        pickle.dump(data, f)

    # Save the original pcd data into the world coordinate system for 3DGS
    obs_points = []
    obs_colors = []
    pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
    processed_mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
    data = np.load(pcd_path)
    with open(processed_mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    for i in range(num_cams):
        points = data["points"][i]
        colors = data["colors"][i]
        mask = processed_masks[0][i]["object"]
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])

    obs_points = np.vstack(obs_points)
    obs_colors = np.vstack(obs_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obs_points)
    pcd.colors = o3d.utility.Vector3dVector(obs_colors)
    viz_pcd = False
    if viz_pcd:
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, coordinate])
    o3d.io.write_point_cloud(f"{output_path}/{case_name}/observation.ply", pcd)
