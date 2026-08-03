import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.utils import logger, cfg
from datetime import datetime
import numpy as np
import pickle
import json
import time
from scripts.utils import set_all_seeds, load_config


def run_inference(base_path, case_name, output_path, optimal_path, skip_cma):
    cfg.load_from_yaml("configs/mpm_config.yaml")

    base_dir = f"{output_path}/{case_name}"

    if not skip_cma:
        # Read the first-satage optimized parameters to set the indifferentiable parameters
        optimal_path = f"{optimal_path}/{case_name}/optimal_params.pkl"
        logger.info(f"Load optimal parameters from: {optimal_path}")
        assert os.path.exists(
            optimal_path
        ), f"{case_name}: Optimal parameters not found: {optimal_path}"
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"
    cfg.case_name = case_name

    logger.set_log_file(path=base_dir, name="inference_log")

    trainer = InvPhyTrainerWarpMPM(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    trainer.test()



if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)

    exp_config = "configs/experiments.yaml"
    config = load_config(exp_config)
    base_path = config["base_path"]
    output_path = config["output_path"]
    optimal_path = config["optimal_path"]

    for exp in config["experiments"]:
        case_name = exp["case_name"]
        print(f"--case_name: {case_name}")
        start_time = time.time()
        run_inference(base_path, case_name, output_path, optimal_path, skip_cma=False)
        elapsed = time.time() - start_time
        print(f"============================== Inference time: {elapsed:.2f}s")
