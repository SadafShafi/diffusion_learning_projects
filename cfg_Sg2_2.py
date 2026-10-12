"""
Stage 2, project 4 - Class conditioning + classifier-free guidance (CFG).

Only 3 things are new compared to Stage 1.2:
  1. the U-Net gets a class embedding           (library/models.py, num_classes=10)
  2. training drops 10% of labels -> null class   (library/ddpm.py, loss(..., y, p_uncond))
  3. sampling mixes both predictions              (library/guidance.py)
         eps = eps_uncond + w * (eps_cond - eps_uncond)

Train+test:  python cfg_Sg2_2.py --mode all --dataset cifar10
Test only:   python cfg_Sg2_2.py --mode test --run_name <run>

Outputs in outputs/<run_name>/   (run_name default: UNetCFG_<dataset>_<timestamp>)
    class_grid.png         rows = classes, columns = guidance scale w (same noise in each row)
    fid_vs_guidance.png    FID and diversity vs w  -> the fidelity / diversity trade-off
    progress/              guided samples during training
    results.json
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

# reuse helpers from Stage 1.2
from ddpm_images_Sg1_2 import PRESETS, ints, update_ema, show, save_grid, plot_loss, CKPT_DIR, OUT_DIR
from library.ddpm import DDPM
from library.models import UNet
from library.data_images import IMAGE_SHAPE, CLASS_NAMES, get_dataset, infinite_batches
from library.guidance import Guided
from library.samplers import ddim
from library.fid import FID
from library.utils import get_device, set_seed, save_checkpoint, load_checkpoint, make_run_name
from library.viz_arch import refresh as refresh_arch_viz

N_CLASSES = 10


def build(config, device):
    C, H, W = IMAGE_SHAPE[config["dataset"]]
    model = UNet(in_ch=C, base_ch=config["base_ch"], ch_mults=ints(config["ch_mults"]), n_res=config["n_res"],
                 attn_levels=ints(config["attn_levels"]), dropout=config["dropout"], num_classes=N_CLASSES).to(device)
    return model, DDPM(T=config["T"], device=device)


@torch.no_grad()
def sample(model, ddpm, y, w, steps, shape, seed=None):
    """Class-conditional samples (labels y) with guidance scale w, using DDIM (Stage 2.1) for speed.
    seed: fixed starting noise, so different w can be compared on the same noise."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    z = torch.randn((len(y), *shape), generator=g).to(y.device)
    return ddim(Guided(model, y, w), ddpm, z, steps).clamp(-1, 1)


# =====================================================================
# TRAIN
# =====================================================================
def train(args, run_dir, device):
    config = vars(args).copy()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model, ddpm = build(config, device)
    shape = IMAGE_SHAPE[args.dataset]
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, (s + 1) / args.warmup))
    batches = infinite_batches(get_dataset(args.dataset, train=True), args.batch_size, args.num_workers, labels=True)
    os.makedirs(os.path.join(run_dir, "progress"), exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{args.run_name}.pt")
    y_show = torch.arange(N_CLASSES, device=device).repeat_interleave(2)    # 2 samples per class

    losses, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x0, y = next(batches)
        x0, y = x0.to(device, non_blocking=True), y.to(device, non_blocking=True)

        loss = ddpm.loss(model, x0, y, p_uncond=args.p_uncond)     # <- the only change to training
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        warmup.step()
        update_ema(ema, model, step, args.ema_decay)

        losses.append(loss.item())
        if step % 500 == 0 or step == 1:
            print(f"step {step:7d} | loss {np.mean(losses[-500:]):.4f} | {step / (time.time() - t0):.1f} it/s", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            x = sample(ema, ddpm, y_show, w=3.0, steps=50, shape=shape, seed=0).cpu()
            save_grid(x, 4, f"step {step}, w=3 (2 per class)", os.path.join(run_dir, "progress", f"step_{step:07d}.png"))
            save_checkpoint(ckpt_path, model, config, losses, ema=ema.state_dict(), step=step,
                            train_minutes=(time.time() - t0) / 60)

    plot_loss(losses, run_dir)


# =====================================================================
# TEST: guidance sweep
# =====================================================================
@torch.no_grad()
def test(args, run_dir, device):
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    config = ckpt["config"]
    dataset = config["dataset"]
    model, ddpm = build(config, device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    shape = IMAGE_SHAPE[dataset]
    steps = args.sampling_steps
    ws = [float(w) for w in args.ws.split(",")]
    names = CLASS_NAMES[dataset]

    # 1) class grid: rows = classes, columns = w. Same noise within a row -> only w changes.
    fig, axes = plt.subplots(N_CLASSES, len(ws), figsize=(1.9 * len(ws) + 1, 1.2 * N_CLASSES + 0.8))
    for c in range(N_CLASSES):
        y = torch.full((args.n_grid,), c, device=device)
        for j, w in enumerate(ws):
            show(axes[c, j], sample(model, ddpm, y, w, steps, shape, seed=c).cpu(), args.n_grid,
                 f"w={w:g}" if c == 0 else "")
        axes[c, 0].text(-4, 16, names[c], ha="right", va="center", fontsize=11)
    plt.suptitle("rows = class, columns = guidance scale w")
    plt.tight_layout(rect=[0.04, 0, 1, 0.97])
    plt.savefig(os.path.join(run_dir, "class_grid.png"), dpi=130)
    plt.close()
    print("saved class_grid.png", flush=True)

    # 2) FID + diversity for every w (equal number of samples per class)
    results = {"sampling_steps": steps, "n_fid": args.n_fid, "sweep": []}
    if args.n_fid > 0:
        fid = FID(device)
        mu_r, sig_r = fid.real_stats(dataset, get_dataset(dataset, train=True), args.n_fid)
        for w in ws:
            t0 = time.time()
            feats, labels = [], []
            for i in range(0, args.n_fid, args.sample_batch):
                y = torch.arange(i, min(i + args.sample_batch, args.n_fid), device=device) % N_CLASSES
                feats.append(fid.features(sample(model, ddpm, y, w, steps, shape)))
                labels.append(y.cpu().numpy())
            feats, labels = np.concatenate(feats), np.concatenate(labels)
            score = FID.score(mu_r, sig_r, *FID.stats(feats))
            # diversity: how spread out the images of ONE class are (avg feature std within a class)
            diversity = float(np.mean([feats[labels == c].std(axis=0).mean() for c in range(N_CLASSES)]))
            results["sweep"].append({"w": w, "fid": score, "diversity": diversity, "minutes": (time.time() - t0) / 60})
            print(f"w={w:4.1f} | FID {score:7.2f} | diversity {diversity:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
            with open(os.path.join(run_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
        plot_sweep(results["sweep"], run_dir)

    results["final_train_loss"] = float(np.mean(ckpt["losses"][-1000:]))
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


def plot_sweep(sweep, run_dir):
    ws = [r["w"] for r in sweep]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ws, [r["fid"] for r in sweep], "o-", color="tab:blue")
    ax1.set_xlabel("guidance scale w")
    ax1.set_ylabel("FID (lower = better)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ws, [r["diversity"] for r in sweep], "s--", color="tab:orange")
    ax2.set_ylabel("within-class diversity (higher = more varied)", color="tab:orange")
    plt.title("Fidelity / diversity trade-off of classifier-free guidance")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "fid_vs_guidance.png"), dpi=130)
    plt.close()


# =====================================================================
# RELABEL TEST: same noised image, different labels
# =====================================================================
@torch.no_grad()
def relabel(args, run_dir, device):
    """Noise real test images to t, then denoise with EVERY class label.
    Small t: the image barely changes, whatever the label. Large t: the label decides what comes out."""
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    config = ckpt["config"]
    dataset = config["dataset"]
    model, ddpm = build(config, device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    names = CLASS_NAMES[dataset]
    ts = [int(t) for t in args.relabel_ts.split(",")]

    test_set = get_dataset(dataset, train=False)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(test_set), generator=g)[:args.n_images]

    rows = len(idx) * len(ts)
    fig, axes = plt.subplots(rows, 2 + N_CLASSES, figsize=(1.25 * (2 + N_CLASSES), 1.35 * rows + 0.8))
    r = 0
    for i in idx.tolist():
        x0, label = test_set[i]
        x0 = x0[None].to(device)
        for t in ts:
            # noise once, then denoise the SAME x_t with all 10 labels
            x_t = ddpm.q_sample(x0, torch.full((1,), t, device=device, dtype=torch.long),
                                torch.randn(x0.shape, generator=g).to(device))
            y = torch.arange(N_CLASSES, device=device)
            out = ddim(Guided(model, y, args.relabel_w), ddpm, x_t.repeat(N_CLASSES, 1, 1, 1),
                       args.relabel_steps, t_start=t).clamp(-1, 1).cpu()

            show(axes[r, 0], x0, 1, f"input: {names[label]}" if r % len(ts) == 0 else "")
            show(axes[r, 1], x_t, 1, f"noised t={t}")
            for c in range(N_CLASSES):
                show(axes[r, 2 + c], out[c:c + 1], 1, f"-> {names[c]}" if r == 0 else "")
                if c == label:      # green frame = the image's true label
                    axes[r, 2 + c].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[r, 2 + c].transAxes,
                                                           fill=False, ec="limegreen", lw=4))
            r += 1
    plt.suptitle(f"same noised image, denoised with each label (green = true label), w={args.relabel_w:g}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(run_dir, "relabel.png"), dpi=130)
    plt.close()
    print(f"saved relabel.png -> {run_dir}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test", "all", "relabel"], required=True,
                   help="all = train then test; relabel = same noised image with every label")
    p.add_argument("--tag", default=None, help="short label for the run name, e.g. bigmodel or noema")
    p.add_argument("--run_name", default=None, help="default: S2.2_cfg_unet_<dataset>[_<tag>]_<time>")
    p.add_argument("--dataset", choices=list(PRESETS), default="cifar10")
    p.add_argument("--T", type=int, default=1000)
    # model (None -> PRESETS from Stage 1.2)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--ch_mults", default=None)
    p.add_argument("--attn_levels", default=None)
    p.add_argument("--n_res", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    # training
    p.add_argument("--p_uncond", type=float, default=0.1, help="how often the label is dropped")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--sample_every", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    # testing
    p.add_argument("--ws", default="0,1,1.5,2,3,4,6", help="guidance scales to sweep")
    p.add_argument("--sampling_steps", type=int, default=100, help="DDIM steps used for testing")
    p.add_argument("--n_grid", type=int, default=4, help="images per cell in class_grid.png")
    p.add_argument("--n_fid", type=int, default=5000, help="0 = skip FID")
    p.add_argument("--sample_batch", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    # relabel test
    p.add_argument("--n_images", type=int, default=4, help="real test images in relabel.png")
    p.add_argument("--relabel_ts", default="300,500,700", help="noise levels to start from")
    p.add_argument("--relabel_w", type=float, default=3.0, help="guidance scale")
    p.add_argument("--relabel_steps", type=int, default=50, help="DDIM steps from t down to 0")
    args = p.parse_args()

    for k, v in PRESETS[args.dataset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.run_name is None:
        if args.mode in ("test", "relabel"):
            p.error(f"--run_name is required for {args.mode}")
        args.run_name = make_run_name("S2.2", "cfg", "unet", args.dataset, args.tag)

    set_seed(args.seed)
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = os.path.join(OUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"mode={args.mode} run={args.run_name} device={device}")
    refresh_arch_viz()

    if args.mode in ("train", "all"):
        train(args, run_dir, device)
    if args.mode in ("test", "all"):
        test(args, run_dir, device)
    if args.mode == "relabel":
        relabel(args, run_dir, device)


if __name__ == "__main__":
    main()
