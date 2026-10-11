# Diffusion Learning Projects

Small hands-on projects for learning diffusion models with PyTorch.

The project includes:

- DDPMs on 2D datasets such as spirals and two moons
- DDPMs on MNIST and CIFAR-10 images
- DDIM, Euler, and Heun samplers
- Class conditioning and classifier-free guidance

## Setup

On a machine with internet access, run:

```bash
bash prepare.sh
```

This creates the Python environment, installs the dependencies, and downloads
the datasets and FID model weights.

## Usage

Activate the environment:

```bash
source venv/bin/activate
```

Run a small 2D experiment locally:

```bash
python ddpm_2d_Sg1_1.py --mode all --dataset spiral
```

For GPU/Slurm runs, use the provided scripts:

```bash
sbatch train.sh ddpm_images_Sg1_2.py --dataset mnist
sbatch train.sh ddpm_images_Sg1_2.py --dataset cifar10
```

See [notes.txt](notes.txt) for more experiment commands and options.

## Project layout

- `ddpm_2d_Sg1_1.py` - DDPM on 2D toy data
- `ddpm_images_Sg1_2.py` - DDPM on image datasets
- `samplers_Sg2_1.py` - alternative sampling methods
- `cfg_Sg2_2.py` - classifier-free guidance
- `library/` - reusable models, data loaders, samplers, and utilities
- `checkpoints/` - saved model checkpoints
- `outputs/` - experiment results
