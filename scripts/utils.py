import random
import yaml
import numpy as np
import torch


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path="configs/experiments.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_path = config["data_path"]
    return {
        "base_path": f"{data_path}/data/different_types",
        "output_path": f"{data_path}/experiments",
        "optimal_path": f"{data_path}/experiments_optimization",
        "data_path": data_path,
        "experiments": config["experiments"],
    }
