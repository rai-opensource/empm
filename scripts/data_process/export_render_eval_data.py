import os
import json
import yaml

with open("configs/example_experiments.yaml", "r") as f:
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
        os.system(
            f"cp -r {base_path}/{case_name}/color {output_path}/{case_name}/"
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
        os.system(f"cp -r {base_path}/{case_name}/mask/{i}/{obj_idx}/* {output_path}/{case_name}/mask/{i}/")
    
    # Copy the split.json
    os.system(f"cp {base_path}/{case_name}/split.json {output_path}/{case_name}/")