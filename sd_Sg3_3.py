"""
Stage 3, project 8 - Taking apart Stable Diffusion 1.5.

We load SD's parts separately (library/sd.py) and write the rest ourselves:
    generate  our sampling loop + CFG (DDIM / Euler / Heun from Stage 2.1 on SD's noise schedule)
              -> check against diffusers, guidance sweep, sampler sweep, x0 predictions, negative prompt
    img2img   start from a noised photo instead of pure noise ("strength" = how much noise)
    inpaint   regenerate only a masked region, keep the rest at every step
    lora      fine-tune with OUR LoRA on 5 photos of one dog (DreamBooth style), compare before / after
    all       everything above

    python sd_Sg3_3.py --mode all
Diagrams: viz/StableDiffusion_system.png, viz/StableDiffusion_unet.png   (python -m library.viz_sd)
"""
import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ddpm_images_Sg1_2 import show, plot_loss, CKPT_DIR, OUT_DIR
from library.data_images import DATA_DIR
from library.lora import inject_lora, lora_state_dict
from library.samplers import ddim, ode, timesteps
from library.sd import load_sd, SDSchedule, SDGuided, encode_prompt, encode_image, decode_latents, SD_DIR
from library.utils import get_device, set_seed, make_run_name

LATENT = (4, 64, 64)


def small(x, size=256):
    """Shrink 512x512 images for the figures (keeps the PNGs small)."""
    return F.interpolate(x, size=size, mode="area")


def noise(n, seed, device):
    return torch.randn((n, *LATENT), generator=torch.Generator().manual_seed(seed)).to(device)


def load_image(path, size=512):
    """Resize shorter side to 512, centre crop, -> (1, 3, 512, 512) in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    s = size / min(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return torch.tensor(np.array(img)).permute(2, 0, 1)[None].float() / 127.5 - 1


class SD:
    """Holds the frozen parts + helpers, so every experiment reads like the maths."""

    def __init__(self, device):
        self.device = device
        self.tok, self.te, self.unet, self.vae = load_sd(device)
        self.sched = SDSchedule(device)

    def text(self, prompts):
        return encode_prompt(self.tok, self.te, prompts)

    def model(self, prompt, w=7.5, negative=""):
        """eps-model with CFG for ONE prompt (the batch dimension is handled by repeating the embeddings)."""
        return CFGModel(self.unet, self.text([prompt]), self.text([negative]), w)

    def sample(self, prompt, z, steps=30, w=7.5, sampler="ddim", negative="", t_start=None):
        m = self.model(prompt, w, negative)
        if sampler in ("euler", "heun"):
            x = ode(m, self.sched, z, steps, sampler)
        else:
            x = ddim(m, self.sched, z, steps, eta=1.0 if sampler == "ddim_eta1" else 0.0, t_start=t_start)
        return decode_latents(self.vae, x)


class CFGModel(SDGuided):
    """SDGuided for a whole batch: the (1, 77, 768) embeddings are repeated to the batch size."""

    def forward(self, x, t):
        B = x.shape[0]
        g = SDGuided(self.unet, self.cond.expand(B, -1, -1), self.uncond.expand(B, -1, -1), self.w)
        return g(x, t)


# =====================================================================
# 1) GENERATE: our sampling loop + CFG
# =====================================================================
@torch.no_grad()
def generate(sd, args, run_dir):
    dev = sd.device
    results = {}

    # (a) correctness: our DDIM == diffusers' DDIMScheduler (same noise, prompt and timesteps).
    # SD's default "leading" timesteps are 958, 925, ..., 1 (steps of 1000 // 30 = 33, offset 1);
    # we give our DDIM exactly these, so both walk the same path.
    from diffusers import DDIMScheduler
    prompt = "a photo of an astronaut riding a horse on mars"
    z = noise(1, 0, dev)
    ref_sched = DDIMScheduler.from_pretrained(SD_DIR, subfolder="scheduler", set_alpha_to_one=True, clip_sample=False)
    ref_sched.set_timesteps(30)
    ours = ddim(sd.model(prompt), sd.sched, z, 30, ts=ref_sched.timesteps.tolist())
    x, m = z.clone(), sd.model(prompt)
    for t in ref_sched.timesteps:
        x = ref_sched.step(m(x, t.to(dev).expand(1)), t, x).prev_sample
    diff = (ours - x).abs().max().item()
    results["max_latent_diff_ours_vs_diffusers_ddim"] = diff
    print(f"our DDIM vs diffusers DDIM: max |latent diff| = {diff:.2e}", flush=True)
    imgs = decode_latents(sd.vae, torch.cat([ours, x]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    show(axes[0], small(imgs[:1]), 1, "OUR loop (library/samplers.ddim + CFG)")
    show(axes[1], small(imgs[1:]), 1, f"diffusers DDIMScheduler   max latent diff {diff:.1e}")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "check_vs_diffusers.png"), dpi=110)
    plt.close()

    # (b) guidance sweep: rows = prompts, columns = w (same noise in a row)
    prompts = ["a photo of an astronaut riding a horse on mars",
               "a red sports car parked in front of a castle, golden hour",
               "a bowl of ramen, studio photo, top view",
               "an oil painting of a lighthouse in a storm"]
    ws = [1, 3, 5, 7.5, 10, 15]
    fig, axes = plt.subplots(len(prompts), len(ws), figsize=(2.6 * len(ws), 2.75 * len(prompts) + 0.6))
    t0 = time.time()
    for r, p in enumerate(prompts):
        for c, w in enumerate(ws):
            show(axes[r, c], small(sd.sample(p, noise(1, r, dev), 30, w), 192), 1, f"w={w}" if r == 0 else "")
        axes[r, 0].set_ylabel(p[:28], fontsize=8)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    results["seconds_per_image_30_steps_cfg"] = (time.time() - t0) / (len(prompts) * len(ws))
    plt.suptitle("guidance scale w (columns), DDIM 30 steps, same noise per row", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(run_dir, "guidance_sweep.png"), dpi=110)
    plt.close()
    print("saved guidance_sweep.png", flush=True)

    # (c) sampler sweep: rows = sampler, columns = steps (same noise everywhere)
    p, z = prompts[0], noise(1, 0, dev)
    samplers, steps = ["ddim", "ddim_eta1", "euler", "heun"], [5, 10, 20, 50]
    fig, axes = plt.subplots(len(samplers), len(steps), figsize=(2.6 * len(steps), 2.75 * len(samplers) + 0.6))
    for r, s in enumerate(samplers):
        for c, n in enumerate(steps):
            nfe = 2 * n - 1 if s == "heun" else n
            show(axes[r, c], small(sd.sample(p, z, n, 7.5, s), 192), 1, f"{s}, {n} steps (NFE {nfe})")
    plt.suptitle("our samplers from Stage 2.1 on Stable Diffusion (w=7.5)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(run_dir, "sampler_sweep.png"), dpi=110)
    plt.close()
    print("saved sampler_sweep.png", flush=True)

    # (d) what happens during sampling: decoded x_t and the model's guess x0_hat
    m, x = sd.model(p), noise(1, 0, dev)
    ts = timesteps(sd.sched.T, 30)
    keep, rows_xt, rows_x0 = {0, 5, 10, 15, 20, 25, 29}, [], []
    ab = sd.sched.alpha_bars
    for i, t in enumerate(ts):
        a_t = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        eps = m(x, torch.full((1,), t, device=dev, dtype=torch.long))
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        if i in keep:
            rows_xt.append((t, x.clone()))
            rows_x0.append(x0.clone())
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    fig, axes = plt.subplots(2, len(keep), figsize=(2.3 * len(keep), 5.2))
    for c, ((t, xt), x0) in enumerate(zip(rows_xt, rows_x0)):
        show(axes[0, c], small(decode_latents(sd.vae, xt), 160), 1, f"x_t, t={t}")
        show(axes[1, c], small(decode_latents(sd.vae, x0), 160), 1, "guess x0_hat")
    plt.suptitle("during sampling: the noisy latent (top) and the model's one-step guess of the result (bottom)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "x0_predictions.png"), dpi=110)
    plt.close()

    # (e) negative prompt (replaces "" in CFG) and seeds
    p = "a portrait photo of an old fisherman"
    neg = "blurry, low quality, cartoon, beard"
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.8))
    for c in range(4):
        show(axes[0, c], small(sd.sample(p, noise(1, 10 + c, dev)), 192), 1, f"seed {10 + c}")
        show(axes[1, c], small(sd.sample(p, noise(1, 10 + c, dev), negative=neg), 192), 1, f"seed {10 + c} + negative")
    plt.suptitle(f'"{p}"   top: uncond = ""   bottom: uncond = "{neg}"')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "negative_prompt.png"), dpi=110)
    plt.close()
    print("saved x0_predictions.png, negative_prompt.png", flush=True)
    return results


# =====================================================================
# 2) IMG2IMG
# =====================================================================
@torch.no_grad()
def img2img(sd, args, run_dir):
    """Encode a photo, noise it to t = strength * 999 (Stage 1.2 q_sample), then denoise with a prompt."""
    dev = sd.device
    jobs = [(os.path.join(DATA_DIR, "sd_examples", "sketch_mountains.jpg"),
             "a fantasy landscape with mountains and a lake, highly detailed, trending on artstation"),
            (sorted(glob.glob(os.path.join(DATA_DIR, "dreambooth_dog", "*.jpeg")))[0],
             "a watercolor painting of a dog")]
    strengths = [0.2, 0.4, 0.6, 0.8, 1.0]
    fig, axes = plt.subplots(len(jobs), 1 + len(strengths), figsize=(2.6 * (1 + len(strengths)), 3.0 * len(jobs) + 0.4))
    for r, (path, prompt) in enumerate(jobs):
        x = load_image(path).to(dev)
        z0 = encode_image(sd.vae, x)
        show(axes[r, 0], small(x, 192), 1, "input")
        for c, s in enumerate(strengths):
            t = min(int(s * sd.sched.T), sd.sched.T - 1)
            z_t = sd.sched.q_sample(z0, torch.tensor([t], device=dev), noise(1, 0, dev))
            out = sd.sample(prompt, z_t, steps=max(2, round(50 * s)), t_start=t)
            show(axes[r, 1 + c], small(out, 192), 1, f"strength {s}  (t={t})")
        axes[r, 0].set_ylabel(prompt[:30], fontsize=8)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    plt.suptitle("img2img: low strength keeps the input, high strength follows the prompt", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(run_dir, "img2img.png"), dpi=110)
    plt.close()
    print("saved img2img.png", flush=True)


# =====================================================================
# 3) INPAINTING
# =====================================================================
@torch.no_grad()
def inpaint(sd, args, run_dir):
    """Blended inpainting: after EVERY DDIM step, the un-masked area is overwritten with the original
    latent noised to the same level. Only the masked area is really generated, but it sees the context."""
    dev = sd.device
    img = load_image(os.path.join(DATA_DIR, "sd_examples", "inpaint_dog.png")).to(dev)
    mask_px = (load_image(os.path.join(DATA_DIR, "sd_examples", "inpaint_dog_mask.png")).to(dev)[:, :1] > 0).float()
    mask = F.interpolate(mask_px, size=LATENT[1:], mode="nearest")        # 1 = generate here
    z0 = encode_image(sd.vae, img)
    prompts = ["a white cat sitting on a bench", "a teddy bear sitting on a bench", "an astronaut sitting on a bench"]
    ab = sd.sched.alpha_bars
    ts = timesteps(sd.sched.T, 50)

    def run(prompt, seed):
        m = sd.model(prompt)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn((1, *LATENT), generator=g).to(dev)
        for i, t in enumerate(ts):
            a_t = ab[t]
            last = i + 1 == len(ts)
            a_prev = torch.tensor(1.0, device=dev) if last else ab[ts[i + 1]]
            eps = m(x, torch.full((1,), t, device=dev, dtype=torch.long))
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps                       # DDIM step
            known = z0 if last else a_prev.sqrt() * z0 + (1 - a_prev).sqrt() * torch.randn(z0.shape, generator=g).to(dev)
            x = mask * x + (1 - mask) * known                                         # keep the rest
        out = decode_latents(sd.vae, x)
        return out, mask_px * out + (1 - mask_px) * img                               # + paste original pixels

    fig, axes = plt.subplots(len(prompts), 4, figsize=(11, 2.9 * len(prompts) + 0.4))
    for r, p in enumerate(prompts):
        show(axes[r, 0], small(img, 192), 1, "original" if r == 0 else "")
        show(axes[r, 1], small(mask_px.repeat(1, 3, 1, 1) * 2 - 1, 192), 1, "mask (white = regenerate)" if r == 0 else "")
        for c, seed in enumerate([0, 1]):
            _, pasted = run(p, seed)
            show(axes[r, 2 + c], small(pasted, 192), 1, f"seed {seed}")
        axes[r, 2].set_ylabel(p, fontsize=8)
        axes[r, 2].axis("on"); axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
    plt.suptitle("inpainting with the normal SD model: only the masked region is generated", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(run_dir, "inpainting.png"), dpi=110)
    plt.close()
    print("saved inpainting.png", flush=True)


# =====================================================================
# 4) LoRA fine-tuning (DreamBooth style)
# =====================================================================
LORA_PROMPTS = ["a photo of sks dog", "a photo of sks dog on the beach", "a photo of sks dog in a bucket",
                "an oil painting of sks dog", "a photo of sks dog wearing a red hat",
                "a photo of a dog"]          # last one: did we break the normal word "dog"? (language drift)


@torch.no_grad()
def lora_grid(sd, seeds):
    return {(p, s): small(sd.sample(p, noise(1, s, sd.device)), 192).cpu() for p in LORA_PROMPTS for s in seeds}


SNAPSHOT_PROMPTS = ["a photo of sks dog on the beach", "a photo of sks dog in a bucket",
                    "a photo of sks dog wearing a red hat", "a photo of a dog"]


@torch.no_grad()
def lora_snapshot(sd):
    """A few prompts, fixed seed: to watch WHEN the LoRA starts to overfit."""
    return [small(sd.sample(p, noise(1, 0, sd.device)), 160).cpu() for p in SNAPSHOT_PROMPTS]


def plot_lora_progress(snapshots, run_dir, prior):
    steps = sorted(snapshots)
    fig, axes = plt.subplots(len(SNAPSHOT_PROMPTS), len(steps), figsize=(2.3 * len(steps), 2.5 * len(SNAPSHOT_PROMPTS) + 0.5))
    for c, st in enumerate(steps):
        for r, img in enumerate(snapshots[st]):
            show(axes[r, c], img, 1, ("base model" if st == 0 else f"step {st}") if r == 0 else "")
    for r, p in enumerate(SNAPSHOT_PROMPTS):
        axes[r, 0].set_ylabel(p.replace("a photo of ", ""), fontsize=8)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    plt.suptitle(f"LoRA during training{prior} (same seed)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(run_dir, "lora_progress.png"), dpi=110)
    plt.close()


def lora(sd, args, run_dir):
    dev = sd.device
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "dreambooth_dog", "*.jpeg")))
    images = torch.cat([load_image(p) for p in paths]).to(dev)
    images = torch.cat([images, images.flip(3)])                      # + mirrored copies
    with torch.no_grad():
        dist = sd.vae.encode(images.to(sd.vae.dtype)).latent_dist
        mean, std = dist.mean.float(), dist.std.float()               # sample a fresh latent each step
        cond = sd.text([args.lora_prompt])

    before = lora_grid(sd, seeds=[0])                                 # the base model, same seeds

    # Prior preservation (DreamBooth): the BASE model draws generic "a photo of a dog" images.
    # During training we also ask the LoRA model to reproduce those -> "dog" keeps its general meaning
    # and the model can't just memorise the 5 photos. We keep the latents, no need to decode them.
    if args.prior:
        with torch.no_grad():
            class_cond = sd.text([args.class_prompt])
            m = sd.model(args.class_prompt)
            class_latents = torch.cat([ddim(m, sd.sched, noise(8, 1000 + i, dev), 30)
                                       for i in range(0, args.n_class_images, 8)])[:args.n_class_images]
        print(f"prior preservation: {len(class_latents)} class images of '{args.class_prompt}'", flush=True)

    eval_steps = [int(v) for v in args.eval_steps.split(",")]
    snapshots = {0: lora_snapshot(sd)}                                # step 0 = base model
    params = inject_lora(sd.unet, rank=args.lora_rank, alpha=args.lora_rank)
    n_lora = sum(p.numel() for p in params)
    print(f"LoRA: {n_lora / 1e6:.2f}M trainable params = {100 * n_lora / sum(p.numel() for p in sd.unet.parameters()):.2f}% of the U-Net", flush=True)
    optim = torch.optim.AdamW(params, lr=args.lora_lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    ab = sd.sched.alpha_bars

    losses, t0 = [], time.time()
    for step in range(1, args.lora_steps + 1):
        idx = torch.randint(0, len(mean), (args.lora_batch,), device=dev)
        z0 = (mean[idx] + std[idx] * torch.randn_like(mean[idx])) * 0.18215
        t = torch.randint(0, sd.sched.T, (args.lora_batch,), device=dev)
        eps = torch.randn_like(z0)
        a = ab[t].view(-1, 1, 1, 1)
        x_t = a.sqrt() * z0 + (1 - a).sqrt() * eps                    # Stage 1.2 forward process
        pred = sd.unet(x_t.half(), t, encoder_hidden_states=cond.expand(args.lora_batch, -1, -1)).sample.float()
        loss = F.mse_loss(pred, eps)                                  # Stage 1.2 loss on the 5 photos
        if args.prior:                                                # + the same loss on class images
            zc = class_latents[torch.randint(0, len(class_latents), (args.lora_batch,), device=dev)]
            tc = torch.randint(0, sd.sched.T, (args.lora_batch,), device=dev)
            ec = torch.randn_like(zc)
            ac = ab[tc].view(-1, 1, 1, 1)
            pc = sd.unet((ac.sqrt() * zc + (1 - ac).sqrt() * ec).half(), tc,
                         encoder_hidden_states=class_cond.expand(args.lora_batch, -1, -1)).sample.float()
            loss = loss + args.prior_weight * F.mse_loss(pc, ec)
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        losses.append(loss.item())
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d} | loss {np.mean(losses[-100:]):.4f} | {step / (time.time() - t0):.1f} it/s", flush=True)
        if step in eval_steps:
            snapshots[step] = lora_snapshot(sd)
    train_minutes = (time.time() - t0) / 60
    torch.save({"lora": lora_state_dict(sd.unet), "config": vars(args)}, os.path.join(CKPT_DIR, f"{args.run_name}.pt"))
    plot_loss(losses, run_dir)

    with torch.no_grad():
        after = lora_grid(sd, seeds=[0, 1])
    cols = ["base model (seed 0)", "+ LoRA (seed 0)", "+ LoRA (seed 1)"]
    fig, axes = plt.subplots(len(LORA_PROMPTS) + 1, 3, figsize=(8.4, 2.75 * (len(LORA_PROMPTS) + 1)))
    show(axes[0, 0], small(images[:5].cpu(), 96), 5, "the 5 training photos")
    axes[0, 1].axis("off"); axes[0, 2].axis("off")
    for r, p in enumerate(LORA_PROMPTS, start=1):
        for c, img in enumerate([before[(p, 0)], after[(p, 0)], after[(p, 1)]]):
            show(axes[r, c], img, 1, cols[c] if r == 1 else "")
        axes[r, 0].set_ylabel(p.replace("a photo of ", ""), fontsize=8)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    prior = f", prior preservation x{args.prior_weight:g}" if args.prior else ", no prior preservation"
    plt.suptitle(f'LoRA (rank {args.lora_rank}, {args.lora_steps} steps{prior}) on "{args.lora_prompt}"', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(run_dir, "lora_before_after.png"), dpi=110)
    plt.close()
    print("saved lora_before_after.png", flush=True)
    plot_lora_progress(snapshots, run_dir, prior)
    return {"lora_params_M": n_lora / 1e6, "lora_train_minutes": train_minutes,
            "lora_final_loss": float(np.mean(losses[-100:]))}


# =====================================================================
# MAIN
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generate", "img2img", "inpaint", "lora", "all", "test"], required=True,
                   help="all (or test) = every experiment")
    p.add_argument("--tag", default=None)
    p.add_argument("--run_name", default=None, help="default: S3.3_<mode>_sd15[_<tag>]_<time>")
    p.add_argument("--lora_prompt", default="a photo of sks dog")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_steps", type=int, default=800)
    p.add_argument("--lora_batch", type=int, default=4)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--prior", action="store_true", help="DreamBooth prior-preservation loss")
    p.add_argument("--class_prompt", default="a photo of a dog")
    p.add_argument("--n_class_images", type=int, default=100)
    p.add_argument("--prior_weight", type=float, default=1.0)
    p.add_argument("--eval_steps", default="100,200,400,800", help="LoRA snapshots at these steps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.mode == "test":
        args.mode = "all"
    if args.run_name is None:
        args.run_name = make_run_name("S3.3", args.mode, "sd15", "text2img", args.tag)

    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(OUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    print(f"mode={args.mode} run={args.run_name} device={device}", flush=True)

    sd = SD(device)
    results = {}
    if args.mode in ("generate", "all"):
        results.update(generate(sd, args, run_dir))
    if args.mode in ("img2img", "all"):
        img2img(sd, args, run_dir)
    if args.mode in ("inpaint", "all"):
        inpaint(sd, args, run_dir)
    if args.mode in ("lora", "all"):
        results.update(lora(sd, args, run_dir))                      # last: it changes the U-Net
    json.dump(results, open(os.path.join(run_dir, "results.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
