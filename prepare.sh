#!/bin/bash
# Run on the login node (it has internet; GPU nodes don't). Safe to re-run.
#   bash prepare.sh
#
# Why uv and not "module load python": the cluster's python module is compiled for the
# login node's Intel CPU and crashes ("Illegal instruction") on the GPU nodes' CPUs.
# uv downloads a portable Python that runs on every node.
set -e
cd "$(dirname "$0")"

UV=~/.local/bin/uv
if [ ! -x $UV ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv into ~/.local/bin
fi

# Python virtual environment inside the project folder
if [ ! -d venv ]; then
    $UV venv venv --python 3.11
fi
source venv/bin/activate

# CUDA 12.8 build: works on all Grete nodes (some still have the CUDA 12.8 driver, CUDA 13 builds fail there)
$UV pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
$UV pip install numpy scipy matplotlib scikit-learn pytorch-fid diffusers huggingface_hub pyarrow

mkdir -p checkpoints outputs logs data

# Download everything the GPU nodes will need
python - <<'EOF'
from library.data_images import download_all
download_all()                                   # MNIST + CIFAR-10 -> data/
from pytorch_fid.inception import InceptionV3
InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])   # FID Inception weights -> ~/.cache/torch
print("datasets + FID weights downloaded")
# Stage 3: CelebA faces (parquet, 1.4 GB) + pretrained Stable Diffusion VAE (320 MB)
from huggingface_hub import snapshot_download
snapshot_download("nielsr/CelebA-faces", repo_type="dataset", local_dir="data/celeba_hf")
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").save_pretrained("data/sd-vae-ft-mse")
print("CelebA + SD VAE downloaded")
EOF

# Sanity check
python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda build', torch.version.cuda)"
echo "Environment ready."
