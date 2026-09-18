import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import glob
import yaml
import shutil
from pathlib import Path
from argparse import ArgumentParser
from data_pipeline.data_process.segment_utils import segment_video

parser = ArgumentParser(description="Export human masks for the configured experiments.")
parser.add_argument("-exp_config", "--exp_config", "--exp-config",
                    default="configs/experiments.yaml",
                    help="Path to the experiments YAML file.")
args = parser.parse_args()

with open(args.exp_config, "r") as f:
    _config = yaml.safe_load(f)
    DATA_PATH = _config["data_path"]

base_path = f"{DATA_PATH}/data/different_types"
output_path = f"{DATA_PATH}/data/different_types_human_mask"

for exp in _config["experiments"]:
    case_name = exp["case_name"]
    print(f"Processing {case_name}")
    os.makedirs(f"{output_path}/{case_name}", exist_ok=True)

    TEXT_PROMPT = "human"
    camera_num = 3
    assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num

    for camera_idx in range(camera_num):
        print(f"  Camera {camera_idx}")
        segment_video(base_path, case_name, TEXT_PROMPT, camera_idx, f"{output_path}/{case_name}")
        tmp_dir = Path(base_path) / case_name / "tmp_data"
        shutil.rmtree(tmp_dir, ignore_errors=True)
