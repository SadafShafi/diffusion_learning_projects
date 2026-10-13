"""Diagrams of Stable Diffusion 1.5 (Stage 3, project 8): the whole system and its U-Net.
Drawn from the loaded model, so the numbers (shapes, parameters) are the real ones.
    python -m library.viz_sd        -> viz/StableDiffusion_system.png, viz/StableDiffusion_unet.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from library.viz_arch import VIZ_DIR, nparams

FROZEN, OURS, COND, DATA = "#bbdefb", "#c8e6c9", "#f8bbd0", "white"


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    def box(x, y, text, fc, bw=3.4, bh=0.9, ec="gray", lw=1, fs=9, ls="-"):
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.05",
                                    fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, family="monospace")

    def arrow(x1, y1, x2, y2, text="", color="gray", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls=ls))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, fontsize=8, color=color, ha="center")
    return fig, ax, box, arrow


def draw_system(te, unet, vae, out_dir=VIZ_DIR):
    fig, ax, box, arrow = _canvas(19, 11)
    ax.text(9.5, 10.6, "Stable Diffusion 1.5, taken apart: blue = pretrained & frozen, green = OUR code",
            ha="center", fontsize=14)

    # ---- text path ----
    box(1.9, 9.3, 'prompt\n"a photo of an\nastronaut riding a horse"', DATA, bw=3.0, bh=1.2)
    box(5.6, 9.3, "CLIP tokenizer\n-> 77 token ids\n(padded)", FROZEN, bw=2.8, bh=1.2)
    box(9.6, 9.3, f"CLIP text encoder ViT-L/14\n{nparams(te) / 1e6:.0f}M params\n-> 77 x 768 embeddings",
        FROZEN, bw=3.8, bh=1.2)
    box(13.6, 7.75, 'same encoder on "" (or a negative\nprompt) -> unconditional 77 x 768', FROZEN, bw=4.4, bh=0.9)
    arrow(3.4, 9.3, 4.2, 9.3)
    arrow(7.0, 9.3, 7.7, 9.3)

    # ---- start latent ----
    box(1.9, 5.6, "pure noise\nz ~ N(0, I)\n4 x 64 x 64", DATA, bw=3.0, bh=1.2)
    box(1.9, 3.7, "img2img (OURS):\nVAE-encode a photo,\nnoise it to 'strength' t", OURS, bw=3.0, bh=1.2)
    box(1.9, 7.4, "inpainting (OURS):\nkeep the un-masked part,\nre-noised, at every step", OURS, bw=3.0, bh=1.2)

    # ---- sampling loop ----
    ax.add_patch(FancyBboxPatch((4.4, 2.6), 9.0, 4.5, boxstyle="round,pad=0.08", fc="#f1f8e9", ec="green", lw=2))
    ax.text(8.9, 6.8, "OUR sampling loop  (replaces diffusers' PNDM scheduler + pipeline)", ha="center",
            fontsize=11, color="green")
    box(8.9, 5.65, f"U-Net  {nparams(unet) / 1e6:.0f}M params  (frozen)\n"
                   "predicts noise eps(x_t, t, text)\ntext enters through 16 cross-attention layers",
        FROZEN, bw=5.6, bh=1.2)
    box(8.9, 4.45, "+ LoRA on all attention projections (OURS, trainable, starts at 0)", OURS, bw=6.4, bh=0.6)
    box(8.9, 3.35, "CFG (OURS): eps = eps_uncond + w (eps_text - eps_uncond)\n"
                   "sampler (OURS): DDIM (eta) / Euler / Heun from Stage 2.1,\n"
                   "on SD's noise schedule: scaled-linear betas, 1000 steps", OURS, bw=7.6, bh=1.2)
    arrow(3.4, 5.6, 6.1, 5.6, "x_T")
    arrow(3.4, 3.7, 5.1, 3.4, "x_t (start)")
    arrow(3.4, 7.4, 6.0, 6.3, "", color="green", ls="--")
    arrow(9.6, 8.7, 9.6, 6.3, "cond", color="deeppink", ls="--")
    arrow(12.6, 7.3, 11.2, 6.3, "uncond", color="deeppink", ls="--")

    # ---- decode ----
    box(15.5, 6.1, "final latent\n4 x 64 x 64", DATA, bw=2.4, bh=0.9)
    box(15.5, 3.9, f"VAE decoder\n{(nparams(vae.decoder) + nparams(vae.post_quant_conv)) / 1e6:.0f}M params\n"
                   "latent / 0.18215 -> image", FROZEN, bw=2.8, bh=1.2)
    box(15.5, 2.6, "image\n3 x 512 x 512", DATA, bw=2.4, bh=0.8)
    arrow(13.4, 5.6, 14.3, 6.0)
    arrow(15.5, 5.65, 15.5, 4.5)
    arrow(15.5, 3.3, 15.5, 3.0)

    # ---- LoRA training path ----
    ax.add_patch(FancyBboxPatch((0.4, 0.2), 18.2, 1.5, boxstyle="round,pad=0.05", fc="#fffde7", ec="goldenrod", lw=1.5))
    ax.text(9.5, 1.35, "LoRA fine-tuning (OURS, DreamBooth style)", ha="center", fontsize=11, color="goldenrod")
    ax.text(9.5, 0.7, '5 photos of one dog --VAE encoder--> latents --+noise--> U-Net + LoRA predicts eps,  prompt '
                      '"a photo of sks dog" --CLIP-->   loss = MSE(eps, eps_pred),  ONLY the LoRA weights are trained',
            ha="center", fontsize=9, family="monospace")

    fig.legend(handles=[Patch(fc=FROZEN, ec="gray", label="pretrained, frozen"),
                        Patch(fc=OURS, ec="gray", label="our code"),
                        Patch(fc=DATA, ec="gray", label="data")], loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "StableDiffusion_system.png"), dpi=120)
    plt.close()


def draw_unet(unet, out_dir=VIZ_DIR):
    cfg = unet.config
    chs, L = cfg.block_out_channels, len(cfg.block_out_channels)
    sizes = [cfg.sample_size // 2 ** i for i in range(L)]
    fig, ax, box, arrow = _canvas(20, 12)
    ax.text(7.2, 11.5, f"SD 1.5 U-Net ({nparams(unet) / 1e6:.0f}M params) on 4 x 64 x 64 latents", ha="center", fontsize=14)
    XL, XR, y0, dy = 2.6, 11.8, 9.6, 2.0
    has_attn = lambda b: "CrossAttn" in type(b).__name__
    n_down = len(unet.down_blocks[0].resnets)
    n_up = len(unet.up_blocks[0].resnets)
    box(XL, 11.0, "noisy latent x_t  4 x 64 x 64", DATA, bw=3.6, bh=0.6)
    box(XR, 11.0, "predicted noise  4 x 64 x 64", DATA, bw=3.6, bh=0.6)
    box(7.2, 11.0, "t -> sinusoidal(320) -> MLP -> 1280\nadded inside EVERY ResBlock", COND, bw=4.4, bh=0.8)
    for i in range(L):
        y = y0 - i * dy
        d, u = unet.down_blocks[i], unet.up_blocks[L - 1 - i]
        tag = lambda b, n: f"{n} x (ResBlock{' + Transformer' if has_attn(b) else ''})"
        box(XL, y, f"Down {i}: {tag(d, n_down)}\n{chs[i]}ch x {sizes[i]}x{sizes[i]}\n{nparams(d) / 1e6:.0f}M p",
            "#ffe0b2" if has_attn(d) else "#c8e6c9", bw=4.4, bh=1.2, ec="darkorange" if has_attn(d) else "gray",
            lw=2 if has_attn(d) else 1)
        box(XR, y, f"Up {i}: {tag(u, n_up)}\n{chs[i]}ch x {sizes[i]}x{sizes[i]}\n{nparams(u) / 1e6:.0f}M p",
            "#ffe0b2" if has_attn(u) else "#bbdefb", bw=4.4, bh=1.2, ec="darkorange" if has_attn(u) else "gray",
            lw=2 if has_attn(u) else 1)
        arrow(XL + 2.3, y, XR - 2.3, y, "skip connections", color="tab:purple", ls="--")
        if i < L - 1:
            arrow(XL, y - 0.65, XL, y - dy + 0.65, "downsample")
            arrow(XR, y - dy + 0.65, XR, y - 0.65, "upsample")
    arrow(XL, 10.7, XL, y0 + 0.65)
    arrow(XR, y0 + 0.65, XR, 10.7)
    ym = y0 - (L - 1) * dy - 1.9
    box(7.2, ym, f"Middle: ResBlock - Transformer - ResBlock\n{chs[-1]}ch x {sizes[-1]}x{sizes[-1]}\n"
                 f"{nparams(unet.mid_block) / 1e6:.0f}M p", "#ffe0b2", bw=4.8, bh=1.2, ec="darkorange", lw=2)
    arrow(XL, y0 - (L - 1) * dy - 0.65, 4.8, ym)
    arrow(9.6, ym, XR, y0 - (L - 1) * dy - 0.65)

    # zoom: one transformer block
    XZ = 17.3
    ax.add_patch(FancyBboxPatch((14.75, 0.4), 5.1, 10.2, boxstyle="round,pad=0.05", fc="#fff8ef", ec="darkorange", lw=2))
    ax.text(XZ, 10.25, "inside one Transformer\n(orange blocks)", ha="center", fontsize=11, color="darkorange")
    tb = unet.down_blocks[0].attentions[0].transformer_blocks[0]
    rows = [("image tokens: H*W x C\n(64*64 = 4096 tokens at level 0)", DATA, 8.9),
            (f"LayerNorm -> SELF-attention (attn1)\nimage tokens look at each other\n{nparams(tb.attn1) / 1e6:.2f}M p", "#ffe0b2", 7.4),
            (f"LayerNorm -> CROSS-attention (attn2)\nQ from image tokens,\nK, V from TEXT (77 x 768)\n{nparams(tb.attn2) / 1e6:.2f}M p", COND, 5.6),
            (f"LayerNorm -> GEGLU MLP\n{nparams(tb.ff) / 1e6:.2f}M p", "#ffe0b2", 3.9),
            ("+ residual after each part", DATA, 2.35)]
    for text, fc, y in rows:
        box(XZ, y, text, fc, bw=4.6, bh=1.25 if "\n\n" not in text else 1.5, fs=8)
    for a, b in zip([8.9, 7.4, 5.6, 3.9], [7.4, 5.6, 3.9, 2.35]):
        arrow(XZ, a - 0.62, XZ, b + 0.7)
    box(XZ, 1.05, "LoRA (ours) goes on to_q, to_k, to_v, to_out\nof attn1 AND attn2 in every block",
        OURS, bw=4.6, bh=0.9, fs=8)
    ax.text(7.2, 0.5, "orange border = has Transformer blocks (self-attention + cross-attention to the text)",
            ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "StableDiffusion_unet.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    from library.sd import load_sd
    _, te, unet, vae = load_sd("cpu")
    draw_system(te, unet, vae)
    draw_unet(unet)
    print("saved viz/StableDiffusion_system.png, viz/StableDiffusion_unet.png")
