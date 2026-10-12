"""
Stage 3, project 6 - Latent diffusion on CelebA faces (128x128) with the pretrained Stable Diffusion VAE.

Idea: don't diffuse 3x128x128 pixels. Compress each face with a pretrained VAE to a 4x16x16 latent
(48x fewer numbers), train the diffusion model on latents, decode the generated latents back to faces.
The diffusion part is our flow matching from Stage 2.3 - only the data changes.

Modes, in this order:
    vae_check       encode -> decode a few faces + show the latent channels (best quality we can reach)
    encode          encode the training set ONCE -> data/celeba128_latents.pt (training never sees pixels)
    train/test/all  flow matching on latents; samples are decoded for plots and FID

    python latent_Sg3_1.py --mode vae_check
    python latent_Sg3_1.py --mode encode
    python latent_Sg3_1.py --mode all
"""
import argparse
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ddpm_images_Sg1_2 import ints, update_ema, show, save_grid, plot_loss, CKPT_DIR, OUT_DIR   # Stage 1.2 helpers
from library.data_images import DATA_DIR, CelebA
from library.flow import FlowMatching
from library.models import UNet
from library.vae import load_vae, encode, decode
from library.fid import FID
from library.utils import get_device, set_seed, save_checkpoint, load_checkpoint, make_run_name
from library.viz_arch import refresh as refresh_arch_viz

LATENTS = os.path.join(DATA_DIR, "celeba128_latents.pt")
LATENT_SHAPE = (4, 16, 16)


def build(config, device):
    model = UNet(in_ch=4, base_ch=config["base_ch"], ch_mults=ints(config["ch_mults"]), n_res=config["n_res"],
                 attn_levels=ints(config["attn_levels"]), dropout=config["dropout"]).to(device)
    return model, FlowMatching(T=config["T"], device=device)


# =====================================================================
# 1) VAE CHECK
# =====================================================================
@torch.no_grad()
def vae_check(args, run_dir, device):
    vae = load_vae(device)
    test_set = CelebA(train=False)
    x = torch.stack([test_set[i][0] for i in range(8)]).to(device)
    z = encode(vae, x)
    rec = decode(vae, z)
    mse = ((rec - x) ** 2).mean().item()
    psnr = 10 * np.log10(4 / mse)                      # pixel range [-1, 1] -> peak-to-peak 2, squared 4

    fig, axes = plt.subplots(3, 1, figsize=(14, 6.5))
    show(axes[0], x, 8, "original 3x128x128 (49,152 numbers)")
    show(axes[1], rec, 8, f"encoded to 4x16x16 (1,024 numbers) and decoded   -  PSNR {psnr:.1f} dB")
    show(axes[2], (rec - x).abs().mean(1, keepdim=True) * 4 - 1, 8, "|difference| x4 (black = same)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "vae_reconstruction.png"), dpi=130)
    plt.close()

    # the 4 latent channels of 4 faces, each channel as a gray image
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for r in range(4):
        show(axes[r, 0], x[r:r + 1], 1, "image" if r == 0 else "")
        for c in range(4):
            axes[r, c + 1].imshow(z[r, c].cpu(), cmap="gray")
            axes[r, c + 1].axis("off")
            if r == 0:
                axes[r, c + 1].set_title(f"latent channel {c}")
    plt.suptitle(f"what the diffusion model sees: 4 x 16 x 16   (scaled latents: mean {z.mean():.2f}, std {z.std():.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "latent_channels.png"), dpi=130)
    plt.close()
    out = {"psnr_db": psnr, "latent_mean": z.mean().item(), "latent_std": z.std().item()}
    json.dump(out, open(os.path.join(run_dir, "results.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


# =====================================================================
# 2) ENCODE THE DATASET ONCE
# =====================================================================
@torch.no_grad()
def encode_dataset(args, device):
    """Every training face, plus its mirror image (free data augmentation), -> one fp16 tensor."""
    vae = load_vae(device)
    data = CelebA(train=True, flip=False)
    loader = torch.utils.data.DataLoader(data, batch_size=256, num_workers=args.num_workers)
    latents, t0 = [], time.time()
    for i, (x, _) in enumerate(loader):
        x = x.to(device)
        latents.append(encode(vae, x).half().cpu())
        latents.append(encode(vae, x.flip(3)).half().cpu())       # mirrored face
        if i % 50 == 0:
            print(f"  {i * 256}/{len(data)} images | {time.time() - t0:.0f}s", flush=True)
    latents = torch.cat(latents)
    torch.save(latents, LATENTS)
    print(f"saved {tuple(latents.shape)} latents -> {LATENTS} ({latents.numel() * 2 / 1e9:.2f} GB) "
          f"in {(time.time() - t0) / 60:.1f} min")


# =====================================================================
# 3) TRAIN on latents
# =====================================================================
def train(args, run_dir, device):
    config = vars(args).copy()
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    latents = torch.load(LATENTS).to(device)              # whole dataset fits on the GPU
    print(f"latents: {tuple(latents.shape)}", flush=True)
    model, fm = build(config, device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, (s + 1) / args.warmup))
    vae = load_vae(device)                                 # only for the progress pictures
    os.makedirs(os.path.join(run_dir, "progress"), exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{args.run_name}.pt")
    z_show = torch.randn((16, *LATENT_SHAPE), generator=torch.Generator().manual_seed(0)).to(device)

    losses, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x0 = latents[torch.randint(0, len(latents), (args.batch_size,), device=device)].float()
        loss = fm.loss(model, x0)                          # exactly the Stage 2.3 loss, on latents
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        warmup.step()
        update_ema(ema, model, step, args.ema_decay)
        losses.append(loss.item())

        minutes = (time.time() - t0) / 60
        done = step == args.steps or (args.time_limit_min and minutes > args.time_limit_min)
        if step % 500 == 0 or step == 1:
            print(f"step {step:7d} | loss {np.mean(losses[-500:]):.4f} | {step / (time.time() - t0):.1f} it/s", flush=True)
        if step % args.sample_every == 0 or done:
            with torch.no_grad():
                x = decode(vae, fm.sample(ema, z_show.shape, x_start=z_show, t_start=fm.T - 1)).cpu()
            save_grid(x, 4, f"step {step}", os.path.join(run_dir, "progress", f"step_{step:07d}.png"))
            save_checkpoint(ckpt_path, model, config, losses, ema=ema.state_dict(), step=step, train_minutes=minutes)
        if done:
            break
    plot_loss(losses, run_dir)


# =====================================================================
# 4) TEST: decode samples, FID, steps sweep
# =====================================================================
@torch.no_grad()
def test(args, run_dir, device):
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    config = ckpt["config"]
    model, fm = build(config, device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    vae = load_vae(device)
    test_set = CelebA(train=False)
    real = torch.stack([test_set[i][0] for i in range(64)])

    def generate(n, steps):
        """n faces with `steps` Euler steps: sample latents, then decode."""
        f = FlowMatching(T=steps, device=device)
        return decode(vae, f.sample(model, (n, *LATENT_SHAPE)))

    # samples + timing
    t0 = time.time()
    samples = generate(64, config["T"]).cpu()
    sec_per_64 = time.time() - t0
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
    show(axes[0], real, 8, "real (test set)")
    show(axes[1], samples, 8, f"generated (latent flow matching, {config['T']} steps)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "samples.png"), dpi=120)
    plt.close()
    save_grid(samples, 8, "output examples", os.path.join(run_dir, "output_examples.png"))

    # reverse process: decode the latent at several points of the trajectory
    _, traj = fm.sample(model, (6, *LATENT_SHAPE), return_trajectory=True)
    T = fm.T
    ks = [0, int(T * 0.25), int(T * 0.5), int(T * 0.75), int(T * 0.9), T]
    fig, axes = plt.subplots(1, len(ks), figsize=(2.2 * len(ks), 13))
    for ax, k in zip(axes, ks):
        show(ax, decode(vae, traj[k].to(device)).cpu(), 1, f"step {k}/{T}")
    plt.suptitle("decoded latents during sampling (noise -> face)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "reverse_process.png"), dpi=110)
    plt.close()

    results = {"final_train_loss": float(np.mean(ckpt["losses"][-1000:])), "train_steps": ckpt["step"],
               "train_minutes": ckpt["train_minutes"], "seconds_per_64_samples": sec_per_64}

    if args.n_fid > 0:
        fid = FID(device)
        mu_r, sig_r = fid.real_stats("celeba", CelebA(train=True, flip=False), args.n_fid)
        sweep = []
        for steps in [int(s) for s in args.fm_steps.split(",")]:
            feats = [fid.features(generate(min(args.sample_batch, args.n_fid - i), steps))
                     for i in range(0, args.n_fid, args.sample_batch)]
            score = FID.score(mu_r, sig_r, *FID.stats(np.concatenate(feats)))
            sweep.append({"steps": steps, "fid": score})
            print(f"  Euler steps={steps:4d} | FID {score:.2f}", flush=True)
        results["fid_vs_steps"] = sweep
        results["n_fid"] = args.n_fid
    json.dump(results, open(os.path.join(run_dir, "results.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))


# =====================================================================
# 5) SANITY CHECKS: memorisation + VAE ceiling
# =====================================================================
@torch.no_grad()
def sanity(args, run_dir, device):
    """Is it too good to be true?
    a) Memorisation: nearest TRAINING latent of each generated latent, compared with real TEST faces
       (never seen in training). Copies would sit at distance ~0, far below the test faces.
    b) Ceiling: FID of plain VAE reconstructions of real faces. No generator can beat its decoder."""
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    model, fm = build(ckpt["config"], device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    vae = load_vae(device)
    train = torch.load(LATENTS).to(device).float().flatten(1)          # (385k, 1024)
    n = args.n_nn

    gen = torch.cat([fm.sample(model, (min(250, n - i), *LATENT_SHAPE)) for i in range(0, n, 250)])
    test_set = CelebA(train=False)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, range(n)), batch_size=250, num_workers=4)
    test = torch.cat([encode(vae, x.to(device)) for x, _ in loader])

    def nearest(z):
        d, idx = [], []
        for c in z.flatten(1).split(250):
            dist = torch.cdist(c, train)
            m = dist.min(1)
            d.append(m.values); idx.append(m.indices)
        return torch.cat(d), torch.cat(idx)

    d_gen, i_gen = nearest(gen)
    d_test, _ = nearest(test)
    out = {"nn_dist_generated": d_gen.mean().item(), "nn_dist_test": d_test.mean().item(),
           "nn_dist_generated_min": d_gen.min().item(), "nn_dist_test_min": d_test.min().item(), "n": n}

    bins = np.linspace(0, max(d_gen.max().item(), d_test.max().item()), 60)
    plt.figure(figsize=(8, 4.5))
    plt.hist(d_test.cpu().numpy(), bins, alpha=0.5, label=f"real test faces, mean {out['nn_dist_test']:.2f}")
    plt.hist(d_gen.cpu().numpy(), bins, alpha=0.5, label=f"generated, mean {out['nn_dist_generated']:.2f}")
    plt.xlabel("L2 distance to nearest TRAINING latent")
    plt.ylabel("count")
    plt.title("Memorisation check (copies would pile up near 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "nearest_hist.png"), dpi=130)
    plt.close()

    order = d_gen.argsort()[:12]                     # the 12 most copy-like samples + their training match
    pairs = torch.stack([gen[order], train[i_gen[order]].view(-1, *LATENT_SHAPE)], 1).flatten(0, 1)
    save_grid(decode(vae, pairs).cpu(), 8, "12 most copy-like generated | nearest TRAINING face (pairs)",
              os.path.join(run_dir, "nearest_closest.png"))

    if args.n_fid > 0:                               # FID of VAE reconstructions = best reachable FID
        fid = FID(device)
        mu_r, sig_r = fid.real_stats("celeba", CelebA(train=True, flip=False), args.n_fid)
        sub = torch.utils.data.Subset(CelebA(train=True, flip=False), range(len(CelebA(train=True)) - args.n_fid, len(CelebA(train=True))))
        feats = [fid.features(decode(vae, encode(vae, x.to(device))))
                 for x, _ in torch.utils.data.DataLoader(sub, batch_size=250, num_workers=4)]
        out["fid_vae_reconstruction"] = FID.score(mu_r, sig_r, *FID.stats(np.concatenate(feats)))
    json.dump(out, open(os.path.join(run_dir, "sanity.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


# =====================================================================
# MAIN
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["vae_check", "encode", "train", "test", "all", "sanity"], required=True)
    p.add_argument("--tag", default=None, help="short label for the run name")
    p.add_argument("--run_name", default=None, help="default: S3.1_<mode>_..._celeba[_<tag>]_<time>")
    p.add_argument("--T", type=int, default=100, help="Euler steps for samples/plots")
    # model on 4x16x16 latents: levels 16 -> 8 -> 4, attention at 8x8 and 4x4
    p.add_argument("--base_ch", type=int, default=128)
    p.add_argument("--ch_mults", default="1,2,2")
    p.add_argument("--attn_levels", default="1,2")
    p.add_argument("--n_res", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    # training
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--time_limit_min", type=float, default=None, help="stop training after this many minutes")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--sample_every", type=int, default=10000)
    p.add_argument("--num_workers", type=int, default=8)
    # testing
    p.add_argument("--n_fid", type=int, default=5000)
    p.add_argument("--fm_steps", default="10,20,50,100", help="Euler steps for the FID sweep")
    p.add_argument("--sample_batch", type=int, default=250)
    p.add_argument("--n_nn", type=int, default=2000, help="sanity: samples for the memorisation check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.run_name is None:
        if args.mode in ("test", "sanity"):
            p.error(f"--run_name is required for {args.mode}")
        what = {"vae_check": ("vaecheck", "sdvae")}.get(args.mode, ("flow", "latentunet"))
        args.run_name = make_run_name("S3.1", *what, "celeba", args.tag)

    set_seed(args.seed)
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"mode={args.mode} device={device}")

    if args.mode == "encode":
        encode_dataset(args, device)
        return
    run_dir = os.path.join(OUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"run={args.run_name}")
    refresh_arch_viz()
    if args.mode == "vae_check":
        vae_check(args, run_dir, device)
    if args.mode in ("train", "all"):
        train(args, run_dir, device)
    if args.mode in ("test", "all"):
        test(args, run_dir, device)
    if args.mode == "sanity":
        sanity(args, run_dir, device)


if __name__ == "__main__":
    main()
