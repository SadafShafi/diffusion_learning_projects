# Diffusion Learning Projects

Small hands-on projects for learning diffusion models with PyTorch.

The project includes:

- DDPMs on 2D datasets such as spirals and two moons
- DDPMs on MNIST and CIFAR-10 images
- DDIM, Euler, and Heun samplers
- Class conditioning and classifier-free guidance

## Experiments completed so far

### 1. DDPM on 2D toy data

An MLP-based DDPM was trained on a spiral dataset. The model learns the
distribution and generates samples with a similar spiral shape. The run also
includes reverse-process visualizations, denoising trajectories, and a learned
vector field.

![Spiral samples](outputs/spiral_20261011-105835/samples.png)

### 2. DDPM on MNIST

A small U-Net was trained to generate handwritten digits. The output below
compares real test images with generated samples. Additional experiments
inspect the reverse diffusion process and the similarity between generated and
training images.

![MNIST samples](outputs/UNet_mnist_20261011-120515/samples.png)

### 3. DDPM on CIFAR-10

The same diffusion workflow was applied to CIFAR-10 using a U-Net. The images
are more difficult to generate than MNIST because they contain natural-image
textures and multiple object classes, but the run produces recognizable
low-resolution samples and FID evaluation outputs.

![CIFAR-10 samples](outputs/UNet_cifar10_20261011-115606/samples.png)

### 4. Classifier-free guidance

Class-conditional MNIST generation was implemented by randomly dropping labels
during training and changing the guidance scale during sampling. The grid
below shows how samples change as the guidance scale `w` increases.

![Classifier-free guidance](outputs/UNetCFG_mnist_20261011-192420/class_grid.png)

### 5. Alternative samplers

DDIM, Euler, and Heun samplers were tested on trained CIFAR-10 models. Their
results are saved in `outputs/Samplers_*`, including comparisons of FID against
the number of function evaluations and grids generated from the same initial
noise.

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
