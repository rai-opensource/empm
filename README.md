# EMPM: Embodied MPM for Modeling and Simulation of Deformable Objects

[Yunuo Chen*](https://yunuoch.github.io/), [Yafei Hu*](https://jeffreyyh.github.io/), [Lingfeng Sun](https://lingfeng.moe/), [Tushar Kusnur](https://ktushar14.github.io/), [Laura Herlant](https://www.linkedin.com/in/lauraherlant/), [Chenfanfu Jiang](https://www.math.ucla.edu/~cffjiang/)

*Equal contribution | Work done at Robotics and AI Institute

[paper](https://arxiv.org/abs/2511.13216) | [project site](https://embodied-mpm.github.io/)

### Overview
Embodied MPM (EMPM) is a deformable object modeling and simulation framework built on a
differentiable Material Point Method (MPM) simulator that captures the dynamics of challenging materials. From
multi-view RGB-D videos, our approach reconstructs geometry and appearance, then uses an MPM physics engine
to simulate object behavior by minimizing the mismatch between predicted and observed visual data.

<p align="center">
  <img src="assets/empm_teaser.png" alt="Embodied MPM teaser" width="90%">
</p>

### Enviroment Setup
#### Linux Setup (CUDA 12.8 + Python 3.10 Specific)
```bash
# Create conda environment
conda create -y -n empm python=3.10
conda activate empm

# install cudatoolkit 12.8 inside the conda environment
conda install -c "nvidia/label/cuda-12.8.0" -c nvidia "cuda-toolkit=12.8"
export CUDA_HOME="$CONDA_PREFIX"          
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH}"
export PYTHONNOUSERSITE=1

# run environment setup script
bash ./env_setup/env_setup.sh
```

### Data Preparation
We provide processed dataset, you can download them [here](https://drive.google.com/drive/folders/1bCdLZuH8y4COb146BcaHR5SWcwcCIQf-?usp=drive_link). after downloading and unzipping, place these data files in in `data_custom/ours/data/different_types`. You can edit `configs/experiments_sample.yaml` to adpat the data files you downloaded.

Alternative, you can process your own RGB-D recordings plus camera intrinsics and calibration with these steps:
#### Data Processing
In `configs/experiments.yaml`, set `data_path` and list the experiments to run.
```bash
python3 scripts/data_process/process_data.py
python3 scripts/data_process/export_gaussian_data.py
python3 scripts/data_process/export_video_human_mask.py
```

### Training and Optimization
Physics simulation training:
```bash
# physics simulation training with cma-es (optional) and gradient-based optimization
python3 scripts/train_test/train_empm.py -exp_config configs/experiments.yaml

# model rollout after params optimization
python3 scripts/train_test/test_empm.py -exp_config configs/experiments.yaml
```

Use `--exp_config` to select a different experiment list for training or inference, e.g.
```bash
python3 scripts/train_test/train_empm.py --exp_config configs/experiments_sample.yaml
python3 scripts/train_test/test_empm.py --exp_config configs/experiments_sample.yaml
```
The default is `configs/experiments.yaml`.

We provided trained 3DGS files in.

You can also do the 3DGS training
You can choose `configs/experiments.yaml` or `configs/experiments_sample.yaml`.
```bash
python3 third_party/gaussian_splatting/generate_interp_poses.py --base_path ./data_custom/ours
cp data_custom/ours/data/different_types/double_lift_sloth_test/shape/matching/final_mesh.glb \
   data_custom/ours/data/gaussian_data/double_lift_sloth_test/shape_prior.glb
python3 scripts/data_process/export_gaussian_data.py --exp_config configs/experiments.yaml
python3 scripts/gs/gs_train.py --exp_config configs/experiments.yaml
```

to render the 3DGS:
```bash
python3 scripts/gs/gs_render.py --exp_config configs/experiments.yaml
```

### Evaluations
```bash
# Use LBS to render the dynamic videos and export the evaluation data(The final videos in ./gaussian_output_dynamic folder)
python3 scripts/gs/gs_render_dynamics.py
# White bg renders → gaussian_output_dynamic_white/
python3 scripts/gs/gs_render_dynamics.py --white_background
python3 scripts/data_process/export_render_eval_data.py

# Get the quantative results
python3 scripts/eval/evaluate_chamfer.py
python3 scripts/eval/evaluate_track.py
python3 scripts/eval/evaluate_render.py

# Get the qualitative results
python3 scripts/viz/visualize_render_results.py
```

### Visualization on viser
NOTE: visualization code is being constantly updated and may not be fully stable.
```bash
python3 scripts/interactive_viser.py
```

## Maintenance

This repository is released as-is with no maintenance commitment or support
guarantee. Users should expect to diagnose issues themselves and fork the
repository for continued development or project-specific changes.

## License

Code authored by the Robotics and AI Institute is licensed under the [RAI Institute Research License](LICENSE.md).
`third_party/gaussian_splatting/` and the Inria-derived files under `scripts/gs/`
are governed by the Inria Gaussian-Splatting License, which restricts use to
non-commercial research and evaluation. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the applicable paths, license locations, and additional notices.

## Citation

If you find this work helpful, please kindly cite our paper

```bibtex
@article{chen_hu2026empm,
    title = {EMPM: Embodied MPM for Modeling and Simulation of Deformable Objects},
    author={Yunuo Chen* and Yafei Hu* and Lingfeng Sun and Tushar Kusnur and Laura Herlant and Chenfanfu Jiang},
    year={2026},
    booktitle={IEEE Robotics and Automation Letters (RA-L)}
}
```

## Acknowledgments

Parts of this codebase are adapted from [PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos](https://github.com/Jianghanxiao/PhysTwin). We thank the PhysTwin authors for releasing their code.
