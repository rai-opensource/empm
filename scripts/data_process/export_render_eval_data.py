import os
import json
import yaml
import shutil
from pathlib import Path

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--exp_config", "--exp-config", default="configs/experiments.yaml",
                    help="Path to the experiments YAML file.")
args = parser.parse_args()

with open(args.exp_config, "r") as f:
    _config = yaml.safe_load(f)
    DATA_PATH = _config["data_path"]

base_path = f"{DATA_PATH}/data/different_types"
output_path = f"{DATA_PATH}/data/render_eval_data"
CONTROLLER_NAME = "hand"

os.makedirs(output_path, exist_ok=True)

for exp in _config["experiments"]:
    case_name = exp["case_name"]
    category = exp["category"]
        
    if not os.path.exists(f"{base_path}/{case_name}"):
        continue
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    # Create the directory for the case
    os.makedirs(f"{output_path}/{case_name}", exist_ok=True)
    os.makedirs(f"{output_path}/{case_name}/mask", exist_ok=True)
    for i in range(3):
        # Copy the original RGB image
        shutil.copytree(
            Path(base_path) / case_name / "color",
            Path(output_path) / case_name / "color",
            dirs_exist_ok=True,
        )
        # Copy only the object mask image
        # Get the mask path for the image
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        os.makedirs(f"{output_path}/{case_name}/mask/{i}", exist_ok=True)
        source_mask_dir = Path(base_path) / case_name / "mask" / str(i) / str(obj_idx)
        destination_mask_dir = Path(output_path) / case_name / "mask" / str(i)
        for source in source_mask_dir.iterdir():
            destination = destination_mask_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
    
    # Copy the split.json
    shutil.copy2(
        Path(base_path) / case_name / "split.json",
        Path(output_path) / case_name / "split.json",
    )
