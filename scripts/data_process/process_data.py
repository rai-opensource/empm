import os
import sys
import time
import json
import glob
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_pipeline.data_process.segment_utils import segment_video, segment_image
from data_pipeline.data_process.video_track_utils import get_dense_track
from data_pipeline.data_process.pcd_utils import lift_pcd, process_mask
from data_pipeline.data_process.pcd_track_utils import filter_track_data
from data_pipeline.data_process.final_process import get_final_data


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.task_name}: Starting...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"{self.task_name}: Completed in {elapsed_time:.2f}s")


def process_data(base_path: str, case_name: str, category: str, shape_prior: bool):
    # data process flags
    process_seg = True
    process_shape_prior = True
    process_track = True
    process_3d = True
    process_align = True
    process_final = True
    text_prompt = f"{category}.hand"
    controller_name = "hand"
    n_cam = 3
    train_test_split = 0.7


    # segment the videos, get the masks of the controller and the object using GroundedSAM2
    if process_seg:
        assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == n_cam
       
        with Timer("Video Segmentation"):
            print(f"Processing {case_name}")
            for camera_idx in range(n_cam):
                print(f"Processing {case_name} camera {camera_idx}")
                segment_video(base_path, case_name, text_prompt, camera_idx, None)
                os.system(f"rm -rf {base_path}/{case_name}/tmp_data")


    # get the dense 2d tracking of the object using Co-tracker 3 for rgb videos
    if process_track:
        with Timer("Dense Tracking"):
            get_dense_track(base_path=base_path, case_name=case_name, num_cam=n_cam)


    # lift to 3D and track the 3D data
    if process_3d:
        # Get the pcd in the world coordinate from the raw observations
        with Timer("Lift to 3D"):
            lift_pcd(base_path=base_path, case_name=case_name)

        # Further process and filter the noise of object and controller masks
        with Timer("Mask Post-Processing"):
            process_mask(base_path=base_path, case_name=case_name, controller_name=controller_name)

        # Process the 3D data tracking
        with Timer("Data Tracking"):
            filter_track_data(base_path=base_path, case_name=case_name)


    # get the final 3d data
    if process_final:
        with Timer("Final Data Generation"):
            get_final_data(base_path=base_path, case_name=case_name, shape_prior=shape_prior)

        # Save the train test split
        frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
        split = {}
        split["frame_len"] = frame_len
        split["train"] = [0, int(frame_len * train_test_split)]
        split["test"] = [int(frame_len * train_test_split), frame_len]
        with open(f"{base_path}/{case_name}/split.json", "w") as f:
            json.dump(split, f)


if __name__ == "__main__":
    with open("configs/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)
    base_path = f"{config['data_path']}/data/different_types"

    os.system("rm -f timer.log")

    for exp in config["experiments"]:
        case_name = exp["case_name"]
        category = exp["category"]
        shape_prior = exp.get("shape_prior", False)

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        print(f"Processing {case_name}...")
        process_data(base_path, case_name, category, shape_prior)