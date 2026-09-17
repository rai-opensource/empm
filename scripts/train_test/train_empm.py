import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import pickle
import json
import time
from scripts.utils import set_all_seeds, load_config


def run_train(base_path, case_name, train_frame, output_path, do_cma_optimization=False):
    cfg.load_from_yaml("configs/mpm_config.yaml")
    base_dir = f"{output_path}/{case_name}"

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

    # init MPM trainer
    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarpMPM(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )
    if do_cma_optimization:
        trainer.train_cma()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser(description="Train EMPM for the configured experiments.")
    parser.add_argument(
        "--exp_config", "--exp-config",
        default="configs/experiments.yaml",
        help="Path to the experiments YAML file (default: %(default)s).",
    )
    args = parser.parse_args()

    seed = 42
    set_all_seeds(seed)
    config = load_config(args.exp_config)
    base_path = config["base_path"]
    output_path = config["output_path"]

    for exp in config["experiments"]:
        case_name = exp["case_name"]
        print(f"--case_name: {case_name}")

        # Read the train test split
        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)

        train_frame = split["train"][1]

        start_time = time.time()
        run_train(base_path, case_name, train_frame, output_path)
        elapsed = time.time() - start_time
        print(f"============================== Training time: {elapsed:.2f}s")
