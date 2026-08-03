# install pytorch
pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# basic dependencies
pip3 install -r env_setup/requirements.txt

# grounded-sam-2
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2/checkpoints/
bash download_ckpts.sh
cd ../gdino_checkpoints/
bash download_ckpts.sh
cd ../
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ../

mkdir -p data_pipeline/data_process/groundedSAM_checkpoints
cd data_pipeline/data_process/groundedSAM_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../..

# gaussian splatting
cd third_party/gaussian_splatting/submodules/diff-gaussian-rasterization/
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
cd ../simple-knn/
pip install -e . --no-build-isolation
cd ../../../../

# pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d/
pip install -e . --no-build-isolation
cd ../
