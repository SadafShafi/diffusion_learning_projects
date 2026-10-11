#!/bin/bash
#SBATCH --job-name=ddpm_test
#SBATCH --partition=grete
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_%j.out

# Re-test an already trained run (run name = folder name in outputs/).
# Usage:  sbatch [slurm options] test.sh <script.py> --run_name <name> [script args]
#   e.g.  sbatch test.sh ddpm_images_Sg1_2.py --run_name UNet_cifar10_20261011-120000 --n_fid 10000
if [ $# -lt 1 ]; then echo "usage: sbatch test.sh <script.py> --run_name <name> [args]"; exit 1; fi
SCRIPT=$1
shift

source ${VENV:-venv}/bin/activate   # other env: sbatch --export=ALL,VENV=<dir> test.sh ...

python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| GPU usable:', ok); sys.exit(0 if ok else 1)" \
    || { echo "ERROR: no usable GPU, stopping"; exit 1; }

python "$SCRIPT" --mode test "$@"
