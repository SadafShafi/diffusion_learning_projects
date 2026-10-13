"""Stable Diffusion 1.5 taken apart (Stage 3, project 8). Downloaded once by prepare.sh into data/sd15.

We load the 4 parts SEPARATELY (no diffusers pipeline):
    tokenizer + text_encoder (CLIP ViT-L/14): prompt -> 77 tokens -> 77 x 768 embeddings   (frozen)
    unet (860M): predicts the noise eps(x_t, t, text) on 4 x 64 x 64 latents              (frozen, + our LoRA)
    vae (84M): 512 x 512 image <-> 4 x 64 x 64 latent, scaling 0.18215                    (frozen)
    noise schedule: SD's "scaled linear" betas, 1000 steps, eps-prediction (like our Stage 1.2 DDPM)
The sampler, CFG, img2img, inpainting and LoRA are OUR code (library/samplers.py, sd_Sg3_3.py, library/lora.py).
"""
import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from library.data_images import DATA_DIR

SD_DIR = os.path.join(DATA_DIR, "sd15")
VAE_SCALE = 0.18215


def load_sd(device, dtype=torch.float16):
    kw = dict(variant="fp16", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(SD_DIR, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD_DIR, subfolder="text_encoder", **kw).to(device)
    unet = UNet2DConditionModel.from_pretrained(SD_DIR, subfolder="unet", **kw).to(device)
    vae = AutoencoderKL.from_pretrained(SD_DIR, subfolder="vae", **kw).to(device)
    for m in (text_encoder, unet, vae):
        m.eval().requires_grad_(False)
    return tokenizer, text_encoder, unet, vae


class SDSchedule:
    """SD's training noise schedule in OUR format (alpha_bars, T), so library/samplers.py works unchanged.
    'scaled linear': betas = linspace(sqrt(0.00085), sqrt(0.012), 1000) ** 2"""

    def __init__(self, device, T=1000, beta_start=0.00085, beta_end=0.012):
        self.T, self.device = T, device
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T, device=device, dtype=torch.float64) ** 2
        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0).float()

    def q_sample(self, x0, t, noise):
        a = self.alpha_bars[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompts):
    """list of strings -> (B, 77, 768) CLIP embeddings. '' gives the unconditional embedding for CFG."""
    ids = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(text_encoder.device)
    return text_encoder(ids)[0]


class SDGuided(torch.nn.Module):
    """The SD U-Net + classifier-free guidance, wrapped as a plain model(x, t) -> eps,
    so OUR samplers (DDIM / Euler / Heun from Stage 2.1) can drive it.
        eps = eps_uncond + w * (eps_text - eps_uncond)        (uncond = empty or negative prompt)"""

    def __init__(self, unet, cond, uncond, w):
        super().__init__()
        self.unet, self.cond, self.uncond, self.w = unet, cond, uncond, w

    def forward(self, x, t):
        dtype = self.unet.dtype
        if self.w == 1:
            return self.unet(x.to(dtype), t, encoder_hidden_states=self.cond).sample.float()
        eps = self.unet(torch.cat([x, x]).to(dtype), torch.cat([t, t]),
                        encoder_hidden_states=torch.cat([self.cond, self.uncond])).sample.float()
        eps_text, eps_uncond = eps.chunk(2)
        return eps_uncond + self.w * (eps_text - eps_uncond)


@torch.no_grad()
def encode_image(vae, x):
    """images in [-1, 1] (B, 3, 512, 512) -> latents (B, 4, 64, 64)"""
    return vae.encode(x.to(vae.dtype)).latent_dist.sample().float() * VAE_SCALE


@torch.no_grad()
def decode_latents(vae, z):
    return torch.cat([vae.decode((c / VAE_SCALE).to(vae.dtype)).sample.float().clamp(-1, 1) for c in z.split(4)])
