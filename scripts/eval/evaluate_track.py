import pickle
import glob
import csv
import json
import yaml
import numpy as np
from scipy.spatial import KDTree
from argparse import ArgumentParser

with open("configs/experiments.yaml", "r") as f:
    _config = yaml.safe_load(f)
    DATA_PATH = _config["data_path"]

prediction_path = f"{DATA_PATH}/experiments"
base_path = f"{DATA_PATH}/data/different_types"
output_file = f"{DATA_PATH}/results/final_track.csv"


parser = ArgumentParser()
parser.add_argument("--simulator", type=str, choices=["springmass", "mpm"], default="mpm")
args = parser.parse_args()


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return np.mean(track_errors)


file = open(output_file, mode="w", newline="", encoding="utf-8")
writer = csv.writer(file)
writer.writerow(
    [
        "Case Name",
        "Train Track Error",
        "Test Track Error",
    ]
)

for case_name in [exp["case_name"] for exp in _config["experiments"]]:
    # if case_name != "single_lift_dinosor":
    #     continue
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]
    train_frame = split["train"][1]
    test_frame = split["test"][1]

    if args.simulator == "mpm":
        ctrl_pts_path = f"{prediction_path}/{case_name}/inference_MPM.pkl"
    else:
        ctrl_pts_path = f"{prediction_path}/{case_name}/inference.pkl"

    with open(ctrl_pts_path, "rb") as f:
        vertices = pickle.load(f)

    with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
        data = pickle.load(f)
        object_points = data["object_points"]
        gt_track_3d = np.full(vertices.shape, np.nan)
        gt_track_3d[:, :object_points.shape[1]] = object_points

    # Locate the index of corresponding point index in the vertices, if nan, then ignore the points
    mask = ~np.isnan(gt_track_3d[0]).any(axis=1)

    kdtree = KDTree(vertices[0])
    dis, idx = kdtree.query(gt_track_3d[0][mask])

    train_track_error = evaluate_prediction(
        1, train_frame, vertices, gt_track_3d, idx, mask
    )
    test_track_error = evaluate_prediction(
        train_frame, test_frame, vertices, gt_track_3d, idx, mask
    )
    writer.writerow([case_name, train_track_error, test_track_error])
file.close()
