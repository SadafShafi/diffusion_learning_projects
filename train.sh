#!/bin/bash
#SBATCH --job-name=ddpm_train
#SBATCH --partition=grete
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out

# Trains, then tests the same run (--mode all). Works for every stage's script.
# Usage:  sbatch [slurm options] train.sh <script.py> [script args]
#   e.g.  sbatch train.sh ddpm_images_Sg1_2.py --dataset mnist
#         sbatch train.sh ddpm_images_Sg1_2.py --dataset mnist --run_name my_name
# No --run_name -> automatic: <model>_<dataset>_<timestamp>
# All commands to run: see notes.txt
if [ $# -lt 1 ]; then echo "usage: sbatch train.sh <script.py> [args]"; exit 1; fi
SCRIPT=$1
shift

source ${VENV:-venv}/bin/activate   # other env: sbatch --export=ALL,VENV=<dir> train.sh ...

nvidia-smi
# Fail loudly instead of silently training on CPU (e.g. GPU driver too old for this torch build)
python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| GPU usable:', ok); sys.exit(0 if ok else 1)" \
    || { echo "ERROR: no usable GPU, stopping"; exit 1; }

python "$SCRIPT" --mode all "$@"
