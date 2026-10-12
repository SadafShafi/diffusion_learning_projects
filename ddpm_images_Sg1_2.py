"""
Stage 1.2 - DDPM on images (MNIST, then CIFAR-10) with our own U-Net.

Train:  python ddpm_images_Sg1_2.py --mode train --dataset mnist --run_name mnist_v1
Test:   python ddpm_images_Sg1_2.py --mode test  --run_name mnist_v1

Same DDPM as Stage 1.1 (library/ddpm.py works for any shape), only the network and data change.
New things here:
    - U-Net (library/models.py)
    - EMA: a slowly-moving average of the weights. Sampling uses it -> much cleaner images.
    - progress samples during training (same noise every time -> you see the model improve)
    - FID (library/fid.py)

Files of a run:
    checkpoints/<run_name>.pt          model + EMA weights + config + loss history
    outputs/<run_name>/                plots, progress/, config.json, results.json
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
from torchvision.utils import make_grid

from library.ddpm import DDPM
from library.flow import FlowMatching
from library.models import UNet
from library.data_images import IMAGE_SHAPE, get_dataset, infinite_batches, load_all
from library.fid import FID
from library.classifier import get_mnist_classifier
from library.utils import get_device, set_seed, save_checkpoint, load_checkpoint, make_run_name
from library.viz_arch import refresh as refresh_arch_viz

CKPT_DIR = "checkpoints"
OUT_DIR = "outputs"

# Per-dataset defaults (any of them can be overridden on the command line)
PRESETS = {
    "mnist":   dict(base_ch=32, ch_mults="1,2,2", attn_levels="2",   steps=20000,  sample_every=2000),
    "cifar10": dict(base_ch=64, ch_mults="1,2,2", attn_levels="1,2", steps=100000, sample_every=10000),
}


def ints(s):
    return tuple(int(v) for v in s.split(","))


def build(config, device):
    C, H, W = IMAGE_SHAPE[config["dataset"]]
    model = UNet(in_ch=C, base_ch=config["base_ch"], ch_mults=ints(config["ch_mults"]),
                 n_res=config["n_res"], attn_levels=ints(config["attn_levels"]), dropout=config["dropout"]).to(device)
    # --method ddpm (Stage 1.2) or flow (Stage 2.3): same network, different training target + sampler
    if config.get("method", "ddpm") == "flow":
        ddpm = FlowMatching(T=config["T"], device=device)
    else:
        ddpm = DDPM(T=config["T"], device=device)
    return model, ddpm


@torch.no_grad()
def update_ema(ema, model, step, max_decay):
    """ema = decay * ema + (1 - decay) * model. Decay starts small so EMA isn't stuck at the random init."""
    decay = min(max_decay, (1 + step) / (10 + step))
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - decay)


@torch.no_grad()
def sample_fixed_noise(ddpm, model, n, shape, seed=0):
    """Sample with a fixed seed, without disturbing the training random state."""
    with torch.random.fork_rng(devices=[ddpm.device] if str(ddpm.device).startswith("cuda") else []):
        torch.manual_seed(seed)
        return ddpm.sample(model, (n, *shape))


# =====================================================================
# TRAIN
# =====================================================================
def train(args, run_dir, device):
    config = vars(args).copy()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    shape = IMAGE_SHAPE[args.dataset]
    model, ddpm = build(config, device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, (s + 1) / args.warmup))
    batches = infinite_batches(get_dataset(args.dataset, train=True), args.batch_size, args.num_workers)

    plot_input_examples(ddpm, next(batches)[:8].to(device), run_dir)
    progress_dir = os.path.join(run_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    progress = []                       # (step, 8 samples) for the overview plot
    ckpt_path = os.path.join(CKPT_DIR, f"{args.run_name}.pt")

    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x0 = next(batches).to(device, non_blocking=True)

        loss = ddpm.loss(model, x0)               # exactly the same loss as in Stage 1.1
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
            x = sample_fixed_noise(ddpm, ema, 16, shape).cpu()
            save_grid(x, 4, f"EMA samples at step {step}", os.path.join(progress_dir, f"step_{step:07d}.png"))
            progress.append((step, x[:8]))
            save_checkpoint(ckpt_path, model, config, losses, ema=ema.state_dict(), step=step,
                            train_minutes=(time.time() - t0) / 60)
            print(f"  saved samples + checkpoint at step {step}", flush=True)

    plot_loss(losses, run_dir)
    plot_progress(progress, run_dir)


# =====================================================================
# TEST
# =====================================================================
def test(args, run_dir, device):
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    config = ckpt["config"]
    shape = IMAGE_SHAPE[config["dataset"]]
    model, ddpm = build(config, device)
    model.load_state_dict(ckpt["ema"])         # sample with the EMA weights
    model.eval()

    test_set = get_dataset(config["dataset"], train=False)
    idx = torch.randperm(len(test_set))[:max(64, args.n_nn)]
    real_all = torch.stack([test_set[i][0] for i in idx])     # n_nn real test images (for NN histogram)
    real = real_all[:64]

    plot_loss(ckpt["losses"], run_dir)

    # Generate samples (+ FID if n_fid > 0)
    t0 = time.time()
    n_keep = max(64, args.n_nn)                 # samples we keep for plots / NN histogram
    n_gen = max(args.n_fid, n_keep)
    fid = FID(device) if args.n_fid > 0 else None
    feats, samples = [], []
    for i in range(0, n_gen, args.sample_batch):
        x = ddpm.sample(model, (min(args.sample_batch, n_gen - i), *shape))
        if fid is not None:
            feats.append(fid.features(x))
        if sum(len(s) for s in samples) < n_keep:
            samples.append(x.cpu())
        print(f"  sampled {i + len(x)}/{n_gen}", flush=True)
    samples_all = torch.cat(samples)[:n_keep]
    samples = samples_all[:64]
    sample_minutes = (time.time() - t0) / 60

    plot_samples(real, samples, run_dir)
    save_grid(samples, 8, f"output examples ({config['dataset']})", os.path.join(run_dir, "output_examples.png"))

    _, traj = ddpm.sample(model, (8, *shape), return_trajectory=True)
    plot_reverse_process(traj, ddpm.T, run_dir)
    nn_results = plot_nearest_neighbours(model, ddpm, traj, samples_all, real_all, config["dataset"], device, run_dir)
    plot_input_output_grid(model, ddpm, real[:8].to(device), run_dir)
    plot_x0_prediction(model, ddpm, real[:1].to(device), run_dir)
    print(f"saved plots -> {run_dir}")

    results = {
        "final_train_loss": float(np.mean(ckpt["losses"][-1000:])),
        "train_steps": ckpt.get("step"),
        "train_minutes": ckpt.get("train_minutes"),
        "sample_minutes": sample_minutes,
        **nn_results,
    }
    if fid is not None:
        mu_r, sig_r = fid.real_stats(config["dataset"], get_dataset(config["dataset"], train=True), args.n_fid)
        mu_g, sig_g = FID.stats(np.concatenate(feats))
        results["fid"] = FID.score(mu_r, sig_r, mu_g, sig_g)
        results["n_fid"] = args.n_fid
        if config.get("method") == "flow":
            results["fid_vs_euler_steps"] = flow_steps_sweep(model, config, fid, mu_r, sig_r, shape, args, device, run_dir)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


@torch.no_grad()
def flow_steps_sweep(model, config, fid, mu_r, sig_r, shape, args, device, run_dir):
    """Flow matching only: how many Euler steps are needed? (compare with outputs/Samplers_* of Stage 2.1)"""
    rows = []
    for n in [int(v) for v in args.fm_steps.split(",")]:
        fm = FlowMatching(T=n, device=device)
        feats = [fid.features(fm.sample(model, (min(args.sample_batch, args.n_fid - i), *shape)))
                 for i in range(0, args.n_fid, args.sample_batch)]
        score = FID.score(mu_r, sig_r, *FID.stats(np.concatenate(feats)))
        rows.append({"steps": n, "fid": score})
        print(f"  flow Euler steps={n:4d} | FID {score:.2f}", flush=True)
    plt.figure(figsize=(6, 4))
    plt.plot([r["steps"] for r in rows], [r["fid"] for r in rows], "o-")
    plt.xscale("log")
    plt.xlabel("Euler steps (= model calls)")
    plt.ylabel("FID")
    plt.title("Flow matching: quality vs number of steps")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "fid_vs_steps.png"), dpi=130)
    plt.close()
    return rows


# =====================================================================
# PLOTS
# =====================================================================
def to_img(x, nrow):
    """Images in [-1, 1] -> one HxWx3 numpy grid in [0, 1]."""
    g = make_grid((x.detach().cpu().clamp(-1, 1) + 1) / 2, nrow=nrow, padding=1, pad_value=1.0)
    return g.permute(1, 2, 0).numpy()


def show(ax, x, nrow, title):
    ax.imshow(to_img(x, nrow))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_grid(x, nrow, title, path):
    plt.figure(figsize=(nrow * 0.9 + 1, (len(x) // nrow) * 0.9 + 1))
    show(plt.gca(), x, nrow, title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_loss(losses, run_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, alpha=0.3, label="loss")
    w = 500
    if len(losses) > w:
        plt.plot(np.arange(w - 1, len(losses)), np.convolve(losses, np.ones(w) / w, "valid"), label=f"avg {w}")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE (noise)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=120)
    plt.close()


def plot_progress(progress, run_dir):
    """Rows = training step, columns = the same 8 noise seeds. Shows how samples improve with training."""
    fig, axes = plt.subplots(len(progress), 1, figsize=(8, 1.1 * len(progress) + 0.5))
    axes = np.atleast_1d(axes)
    for ax, (step, x) in zip(axes, progress):
        ax.imshow(to_img(x, 8))
        ax.set_ylabel(f"{step}", rotation=0, labelpad=30, va="center")
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("EMA samples during training (same noise each row)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "progress_overview.png"), dpi=120)
    plt.close()


@torch.no_grad()
def plot_input_examples(ddpm, x0, run_dir):
    """What the model is trained on: clean images and their noised versions x_t."""
    T = ddpm.T
    ts = [0, int(T * 0.05), int(T * 0.1), int(T * 0.25), int(T * 0.5), int(T * 0.75), T - 1]
    fig, axes = plt.subplots(1, len(ts), figsize=(1.6 * len(ts), 8))
    for ax, t in zip(axes, ts):
        tt = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
        show(ax, x0 if t == 0 else ddpm.q_sample(x0, tt), 1, "clean x_0" if t == 0 else f"t={t}")
    plt.suptitle("input examples (forward process)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "input_examples.png"), dpi=120)
    plt.close()


def plot_samples(real, samples, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    show(axes[0], real, 8, "real (test set)")
    show(axes[1], samples, 8, "generated")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "samples.png"), dpi=120)
    plt.close()


def plot_reverse_process(traj, T, run_dir):
    """Rows = 8 samples, columns = timesteps of sampling from pure noise to image."""
    ts = [T, int(T * 0.75), int(T * 0.5), int(T * 0.25), int(T * 0.1), int(T * 0.05), 0]
    fig, axes = plt.subplots(1, len(ts), figsize=(1.6 * len(ts), 8))
    for ax, t in zip(axes, ts):
        show(ax, traj[T - t], 1, f"t={t}")
    plt.suptitle("reverse process (noise -> image)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "reverse_process.png"), dpi=120)
    plt.close()


@torch.no_grad()
def plot_input_output_grid(model, ddpm, x0, run_dir):
    """One row per noise level t:  clean x_0 | noised x_t | denoised output | |output - x_0|
    Small t: output ~ input (denoising). Large t: output is a NEW image (generation)."""
    T = ddpm.T
    ts = [int(T * 0.1), int(T * 0.25), int(T * 0.5), int(T * 0.75), T - 1]
    trajs = {}
    fig, axes = plt.subplots(len(ts), 4, figsize=(16, 2.4 * len(ts) + 0.5))
    for row, t in zip(axes, ts):
        tt = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
        x_t = ddpm.q_sample(x0, tt)
        out, trajs[t] = ddpm.sample(model, x0.shape, x_start=x_t, t_start=t, return_trajectory=True)
        diff = (out - x0).abs().mean(1, keepdim=True)          # 0 (black) = identical
        show(row[0], x0, 4, "input: clean x_0")
        show(row[1], x_t, 4, f"input: noised x_t (t={t})")
        show(row[2], out, 4, f"output: denoised from t={t}")
        show(row[3], diff - 1, 4, "|output - x_0|  (black = same)")
    plt.suptitle("inputs vs outputs at different noise levels", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(run_dir, "input_output_grid.png"), dpi=150)
    plt.close()

    plot_input_output_steps(trajs, run_dir)


def plot_input_output_steps(trajs, run_dir, n_cols=7):
    """Same rows as the grid, columns = intermediate denoising steps from x_t to the output (4 images per cell)."""
    fig, axes = plt.subplots(len(trajs), n_cols, figsize=(2.2 * n_cols, 2.4 * len(trajs) + 0.5))
    for row, (t, traj) in zip(axes, trajs.items()):
        n_steps = traj.shape[0] - 1
        for ax, k in zip(row, np.linspace(0, n_steps, n_cols).round().astype(int)):
            title = f"input x_t, t={t}" if k == 0 else "output" if k == n_steps else f"t={t - k} ({k}/{n_steps})"
            show(ax, traj[k][:4], 2, title)
    plt.suptitle("intermediate denoising steps (each row starts from a different noise level)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(run_dir, "input_output_steps.png"), dpi=150)
    plt.close()


@torch.no_grad()
def plot_x0_prediction(model, ddpm, x0, run_dir):
    """What the network sees and predicts at different t, for one image.
    From the predicted noise we can compute the model's guess of the clean image in ONE step:
        x0_hat = (x_t - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar)
    At large t this guess is a blurry 'average' image -> that's why sampling needs many small steps."""
    T = ddpm.T
    ts = [int(T * 0.05), int(T * 0.1), int(T * 0.25), int(T * 0.5), int(T * 0.75), T - 1]
    fig, axes = plt.subplots(4, len(ts), figsize=(2 * len(ts), 8.5))
    for col, t in zip(axes.T, ts):
        tt = torch.full((1,), t, device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        x_t = ddpm.q_sample(x0, tt, eps)
        out = model(x_t, ddpm.model_time(tt))         # DDPM: predicted noise, flow: predicted velocity
        x0_hat = ddpm.predict_x0(model, x_t, t)
        show(col[0], x_t, 1, f"x_t  (t={t})")
        show(col[1], ddpm.target(x0, eps) / 3, 1, "true target")
        show(col[2], out / 3, 1, "model output")
        show(col[3], x0_hat, 1, "predicted x_0")
    plt.suptitle("one-step prediction of x_0 from x_t")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "x0_prediction.png"), dpi=120)
    plt.close()


# =====================================================================
# NEAREST TRAINING IMAGES (does the model copy its training data?)
# =====================================================================
@torch.no_grad()
def nearest(x, train):
    """Closest training image (pixel L2 distance) for each image in x. Batched to save memory."""
    flat = train.flatten(1)
    dists, idxs = [], []
    for chunk in x.split(500):
        d = torch.cdist(chunk.flatten(1).to(train.device), flat)
        dist, idx = d.min(dim=1)
        dists.append(dist); idxs.append(idx)
    idx = torch.cat(idxs)
    return train[idx], torch.cat(dists)


@torch.no_grad()
def plot_nearest_neighbours(model, ddpm, traj, samples, real, dataset, device, run_dir):
    """traj: sampling trajectory of 8 images (traj[k] = state after k steps, next step uses t = T-1-k).
    At each t we take the model's current guess of the final image,
        x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t),
    and find the closest TRAINING image to it. (x_t itself is mostly noise, its neighbour would be meaningless.)
    Also returns a memorisation check: NN distance of generated images vs. of real test images."""
    T = ddpm.T
    train = load_all(dataset, train=True).to(device)
    ts = [int(T * 0.75), int(T * 0.5), int(T * 0.25), int(T * 0.1), 0]

    fig, axes = plt.subplots(1, 2 * len(ts), figsize=(2.0 * 2 * len(ts), 9))
    for j, t in enumerate(ts):
        x_t = traj[T - 1 - t].to(device)
        x0_hat = ddpm.predict_x0(model, x_t, t).clamp(-1, 1)
        nn_img, dist = nearest(x0_hat, train)
        show(axes[2 * j], x0_hat, 1, f"t={t}\nguess x0_hat")
        show(axes[2 * j + 1], nn_img, 1, f"nearest train\nd={dist.mean():.1f}")
    plt.suptitle("model's guess of the final image during sampling, and its nearest TRAINING image (pixel L2)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "nearest_neighbours.png"), dpi=130)
    plt.close()

    # Memorisation check. If generated images were copies, their NN distance would be ~0,
    # much smaller than for real test images (which the model never saw).
    _, d_gen = nearest(samples, train)
    _, d_test = nearest(real, train)
    out = {"nn_dist_generated": d_gen.mean().item(), "nn_dist_test": d_test.mean().item(),
           "nn_dist_generated_min": d_gen.min().item(), "nn_dist_test_min": d_test.min().item(),
           "n_nn": len(samples)}

    # Histogram: memorisation would show up as a separate bump of generated images near 0
    d_gen_np, d_test_np = d_gen.cpu().numpy(), d_test.cpu().numpy()
    bins = np.linspace(0, max(d_gen_np.max(), d_test_np.max()), 60)
    plt.figure(figsize=(8, 4.5))
    plt.hist(d_test_np, bins, alpha=0.5, label=f"real test images (n={len(d_test_np)}), mean {d_test_np.mean():.2f}")
    plt.hist(d_gen_np, bins, alpha=0.5, label=f"generated (n={len(d_gen_np)}), mean {d_gen_np.mean():.2f}")
    plt.axvline(d_test_np.mean(), color="tab:blue", ls="--")
    plt.axvline(d_gen_np.mean(), color="tab:orange", ls="--")
    plt.xlabel("pixel L2 distance to nearest TRAINING image")
    plt.ylabel("count")
    plt.title("Memorisation check: copies would pile up near 0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "nearest_hist.png"), dpi=130)
    plt.close()

    # The 16 generated images closest to the training set (most 'copy-like') + their match
    order = d_gen.argsort()[:16].cpu()
    closest, cd = nearest(samples[order], train)
    pairs = torch.stack([samples[order].to(device), closest], dim=1).flatten(0, 1)
    save_grid(pairs.cpu(), 8, f"16 most copy-like generated | nearest train (pairs), d = "
              f"{cd.min():.1f} .. {cd.max():.1f}", os.path.join(run_dir, "nearest_closest.png"))

    # final samples next to their nearest training image
    nn_img, _ = nearest(samples[:32], train)
    pairs = torch.stack([samples[:32].to(device), nn_img], dim=1).flatten(0, 1)   # gen, nn, gen, nn, ...
    save_grid(pairs.cpu(), 8, f"generated | nearest train image (pairs)   "
              f"mean NN dist: generated {out['nn_dist_generated']:.1f}, real test {out['nn_dist_test']:.1f}",
              os.path.join(run_dir, "nearest_final.png"))
    return out


def nearest_only(args, run_dir, device):
    """--mode nearest: only the nearest-neighbour plots, for an already trained run."""
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    config = ckpt["config"]
    shape = IMAGE_SHAPE[config["dataset"]]
    model, ddpm = build(config, device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    test_set = get_dataset(config["dataset"], train=False)
    n = max(64, args.n_nn)
    real = torch.stack([test_set[i][0] for i in torch.randperm(len(test_set))[:n]])
    samples = torch.cat([ddpm.sample(model, (min(args.sample_batch, n - i), *shape)).cpu()
                         for i in range(0, n, args.sample_batch)])
    _, traj = ddpm.sample(model, (8, *shape), return_trajectory=True)
    out = plot_nearest_neighbours(model, ddpm, traj, samples, real, config["dataset"], device, run_dir)
    print(json.dumps(out, indent=2))
    print(f"saved nearest_neighbours.png / nearest_final.png / nearest_hist.png / nearest_closest.png -> {run_dir}")


# =====================================================================
# EXPERIMENT: where is the digit decided?
# =====================================================================
@torch.no_grad()
def pick_test_digits(clf, per_class, device):
    """per_class test images of each digit 0-9, only ones the classifier gets right."""
    test_set = get_dataset("mnist", train=False)
    xs, ys, count = [], [], [0] * 10
    for x, y in test_set:
        if count[y] < per_class and clf(x[None].to(device)).argmax(1).item() == y:
            xs.append(x); ys.append(y); count[y] += 1
        if min(count) == per_class:
            break
    return torch.stack(xs).to(device), torch.tensor(ys, device=device)


@torch.no_grad()
def identity_curve(args, run_dir, device):
    """For each t: noise real digits to t, denoise back, and check (with a classifier)
    whether the digit is still the same. Repeat with different noise -> P(same digit | t)."""
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    config = ckpt["config"]
    assert config["dataset"] == "mnist", "identity_curve needs an MNIST run"
    model, ddpm = build(config, device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    clf = get_mnist_classifier(device)

    x0, y = pick_test_digits(clf, args.per_class, device)
    xb, yb = x0.repeat(args.repeats, 1, 1, 1), y.repeat(args.repeats)     # each image R times
    T = ddpm.T
    ts = list(range(0, T, args.t_stride)) + [T - 1]
    example_ts = [t for t in (250, 400, 500, 600, 750) if t in ts]
    examples = {}                                    # t -> (outputs of image 0, predicted labels)
    ex_idx = torch.arange(args.repeats, device=device) * len(x0)    # the R copies of image 0

    kept = np.zeros((len(ts), 10))                   # P(same digit) per t, per digit
    t0 = time.time()
    for i, t in enumerate(ts):
        x_t = ddpm.q_sample(xb, torch.full((len(xb),), t, device=device, dtype=torch.long))
        out = ddpm.sample(model, xb.shape, x_start=x_t, t_start=t)
        pred = clf(out).argmax(1)
        same = (pred == yb).float()
        for d in range(10):
            kept[i, d] = same[yb == d].mean().item()
        if t in example_ts:
            examples[t] = (out[ex_idx].cpu(), pred[ex_idx].tolist())
        print(f"t={t:4d} | P(same digit) = {kept[i].mean():.3f} | {time.time() - t0:.0f}s", flush=True)

    overall = kept.mean(1)
    below = [t for t, p in zip(ts, overall) if p < 0.5]
    t_decided = below[0] if below else None
    with open(os.path.join(run_dir, "identity_curve.json"), "w") as f:
        json.dump({"ts": ts, "overall": overall.tolist(), "per_digit": kept.tolist(),
                   "t_decided": t_decided, "per_class": args.per_class, "repeats": args.repeats}, f, indent=2)

    # curve
    plt.figure(figsize=(9, 5.5))
    for d in range(10):
        plt.plot(ts, kept[:, d], lw=1, alpha=0.7, color=plt.cm.tab10(d), label=f"digit {d}")
    plt.plot(ts, overall, lw=3, color="black", label="all digits")
    plt.axhline(0.5, ls="--", color="gray")
    plt.axhline(0.1, ls=":", color="gray")
    plt.text(5, 0.11, "chance", color="gray")
    if t_decided is not None:
        plt.axvline(t_decided, ls="--", color="red", alpha=0.6)
        plt.text(t_decided + 8, 0.95, f"decided ~ t={t_decided}", color="red")
    plt.xlabel("t (how far the real digit was noised)")
    plt.ylabel("P(output is the same digit)")
    plt.title(f"Where is the digit decided?  ({args.per_class}/digit x {args.repeats} repeats)")
    plt.legend(fontsize=8, ncol=2, loc="center right")
    plt.ylim(0, 1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "identity_curve.png"), dpi=130)
    plt.close()

    # examples: one real digit, its R outputs at a few t
    fig, axes = plt.subplots(len(examples), args.repeats + 1, figsize=(1.3 * (args.repeats + 1), 1.5 * len(examples) + 0.6))
    for row, (t, (outs, preds)) in zip(np.atleast_2d(axes), examples.items()):
        show(row[0], x0[:1], 1, f"input (t={t})")
        for ax, o, pr in zip(row[1:], outs, preds):
            show(ax, o[None], 1, f"-> {pr}")
            ax.title.set_color("green" if pr == y[0].item() else "red")
    plt.suptitle(f"same real digit '{y[0].item()}', noised to t and denoised {args.repeats} times (label = classifier)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "identity_examples.png"), dpi=130)
    plt.close()
    print(f"saved identity_curve.png / identity_examples.png / identity_curve.json -> {run_dir}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test", "all", "identity_curve", "nearest"], required=True,
                   help="all = train then test; identity_curve / nearest = extra experiments on a trained run")
    p.add_argument("--tag", default=None, help="short label for the run name, e.g. bigmodel or noema")
    p.add_argument("--run_name", default=None, help="default: S1.2|S2.3_<method>_unet_<dataset>[_<tag>]_<time>")
    p.add_argument("--dataset", choices=list(PRESETS), default="mnist")
    # diffusion
    p.add_argument("--method", choices=["ddpm", "flow"], default="ddpm", help="flow = Stage 2.3 flow matching")
    p.add_argument("--T", type=int, default=None, help="DDPM steps (default 1000) or flow Euler steps (default 100)")
    p.add_argument("--fm_steps", default="5,10,20,50,100", help="flow only: Euler steps for the FID sweep")
    # model (None -> use PRESETS)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--ch_mults", default=None, help="e.g. 1,2,2")
    p.add_argument("--attn_levels", default=None, help="e.g. 2  or  1,2")
    p.add_argument("--n_res", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    # training
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--sample_every", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    # testing
    p.add_argument("--n_fid", type=int, default=5000, help="0 = skip FID")
    p.add_argument("--sample_batch", type=int, default=500)
    p.add_argument("--n_nn", type=int, default=2000, help="samples for the nearest-neighbour histogram")
    p.add_argument("--seed", type=int, default=0)
    # identity_curve experiment
    p.add_argument("--per_class", type=int, default=10, help="test images per digit")
    p.add_argument("--repeats", type=int, default=10, help="noise+denoise repeats per image")
    p.add_argument("--t_stride", type=int, default=50, help="spacing of the tested t values")
    args = p.parse_args()

    for k, v in PRESETS[args.dataset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.T is None:
        args.T = 100 if args.method == "flow" else 1000
    if args.run_name is None:
        if args.mode in ("test", "identity_curve", "nearest"):
            p.error(f"--run_name is required for {args.mode}")
        stage = "S2.3" if args.method == "flow" else "S1.2"
        args.run_name = make_run_name(stage, args.method, "unet", args.dataset, args.tag)

    set_seed(args.seed)
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True     # faster matmuls/convs on A100/H100
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
    if args.mode == "identity_curve":
        identity_curve(args, run_dir, device)
    if args.mode == "nearest":
        nearest_only(args, run_dir, device)


if __name__ == "__main__":
    main()
