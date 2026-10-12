"""Pretrained Stable Diffusion VAE (Stage 3, project 6). Downloaded once by prepare.sh into data/.

The VAE compresses an image 8x in each direction:   3 x 128 x 128  ->  4 x 16 x 16   (48x fewer numbers)
Diffusion then runs on these small latents instead of pixels; the decoder turns latents back into images.

SCALE: raw SD latents have std ~5. Multiplying by 0.18215 (the value used by Stable Diffusion)
makes them std ~1, the same scale as our Gaussian noise.
"""
import os

import torch
from diffusers import AutoencoderKL

from library.data_images import DATA_DIR

VAE_DIR = os.path.join(DATA_DIR, "sd-vae-ft-mse")
SCALE = 0.18215


def load_vae(device):
    return AutoencoderKL.from_pretrained(VAE_DIR).to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode(vae, x):
    """images in [-1, 1] (B, 3, H, W) -> latents (B, 4, H/8, W/8), scaled to std ~1"""
    return vae.encode(x).latent_dist.sample() * SCALE


@torch.no_grad()
def decode(vae, z, batch=64):
    """latents -> images in [-1, 1]. Batched, because the decoder is memory hungry."""
    return torch.cat([vae.decode(c / SCALE).sample.clamp(-1, 1) for c in z.split(batch)])
