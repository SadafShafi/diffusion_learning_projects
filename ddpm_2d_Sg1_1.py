"""
Stage 1.1 - DDPM on 2D toy data.

Train:  python ddpm_2d_Sg1_1.py --mode train --dataset spiral --run_name spiral_v1
Test:   python ddpm_2d_Sg1_1.py --mode test  --run_name spiral_v1

Files of a run:
    checkpoints/<run_name>.pt          model weights + config + loss history
    outputs/<run_name>/                all plots and the config of that run
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
import torch

from library.ddpm import DDPM
from library.flow import FlowMatching
from library.models import MLP
from library.data2d import get_data
from library.utils import get_device, set_seed, save_checkpoint, load_checkpoint, make_run_name
from library.viz_arch import refresh as refresh_arch_viz

CKPT_DIR = "checkpoints"
OUT_DIR = "outputs"


def build(config, device):
    model = MLP(data_dim=2, hidden=config["hidden"], n_layers=config["n_layers"]).to(device)
    # --method ddpm (Stage 1.1) or flow (Stage 2.3): same MLP, different training target + sampler
    if config.get("method", "ddpm") == "flow":
        ddpm = FlowMatching(T=config["T"], device=device)
    else:
        ddpm = DDPM(T=config["T"], device=device)
    return model, ddpm


# =====================================================================
# TRAIN
# =====================================================================
def train(args, run_dir, device):
    config = vars(args).copy()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model, ddpm = build(config, device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Toy data is tiny: generate one big dataset and sample random batches from it.
    data = get_data(args.dataset, args.n_data).to(device)
    plot_input_examples(ddpm, data[:2000], run_dir)

    losses = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, data.shape[0], (args.batch_size,), device=device)
        x0 = data[idx]

        loss = ddpm.loss(model, x0)
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())
        if step % 1000 == 0 or step == 1:
            print(f"step {step:6d} | loss {np.mean(losses[-1000:]):.4f}", flush=True)

    ckpt_path = os.path.join(CKPT_DIR, f"{args.run_name}.pt")
    save_checkpoint(ckpt_path, model, config, losses)
    print(f"saved checkpoint -> {ckpt_path}")

    plot_loss(losses, run_dir)


# =====================================================================
# TEST (sampling + visualisations)
# =====================================================================
def test(args, run_dir, device):
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    config = ckpt["config"]
    model, ddpm = build(config, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    real = get_data(config["dataset"], args.n_samples)
    samples, traj = ddpm.sample(model, (args.n_samples, 2), return_trajectory=True)
    samples = samples.cpu()
    # traj[k] is x at reverse step k: traj[0] = pure noise (t=T), traj[T] = final sample (t=0)

    plot_loss(ckpt["losses"], run_dir)
    plot_samples(real, samples, run_dir)
    plot_output_examples(samples, run_dir)
    plot_input_output_grid(model, ddpm, real.to(device), run_dir)
    plot_reverse_snapshots(traj, ddpm.T, run_dir)
    plot_trajectories(traj, samples, run_dir)
    plot_vector_field(model, ddpm, samples, device, run_dir)
    print(f"saved plots -> {run_dir}")

    results = {
        "final_train_loss": float(np.mean(ckpt["losses"][-1000:])),
        **sample_quality(real, samples),
    }
    if config.get("method") == "flow":
        results["chamfer_vs_euler_steps"] = flow_steps_sweep(model, real, args, device, run_dir)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


@torch.no_grad()
def flow_steps_sweep(model, real, args, device, run_dir):
    """Flow matching only: sample quality (chamfer) and sample paths for few Euler steps.
    Straight paths -> even a handful of steps gives a good spiral."""
    steps = [int(v) for v in args.fm_steps.split(",")]
    z = torch.randn(len(real), 2, device=device)            # same noise for every step count
    rows = []
    fig, axes = plt.subplots(1, len(steps), figsize=(3.2 * len(steps), 3.4))
    for ax, n in zip(axes, steps):
        fm = FlowMatching(T=n, device=device)
        x, traj = fm.sample(model, z.shape, return_trajectory=True, x_start=z, t_start=n - 1)
        x = x.cpu()
        q = sample_quality(real, x)
        rows.append({"steps": n, **q})
        ax.scatter(x[:, 0], x[:, 1], s=1, c="tab:orange")
        for i in range(15):
            ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.8, color="black", alpha=0.6)
        _square(ax, f"{n} Euler steps\nchamfer {q['chamfer']:.3f}")
    plt.suptitle("flow matching with few steps (black = paths of 15 points)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "flow_few_steps.png"), dpi=120)
    plt.close()
    return rows


def sample_quality(real, samples):
    """Simple numbers to compare runs (lower = better).
    gen_to_real: avg distance from each generated point to its nearest real point -> are samples ON the shape?
    real_to_gen: avg distance from each real point to its nearest generated point -> is the WHOLE shape covered?
    chamfer:     sum of both."""
    d = torch.cdist(samples, real)
    gen_to_real = d.min(dim=1).values.mean().item()
    real_to_gen = d.min(dim=0).values.mean().item()
    return {"gen_to_real": gen_to_real, "real_to_gen": real_to_gen, "chamfer": gen_to_real + real_to_gen}


# =====================================================================
# PLOTS
# =====================================================================
LIM = 2.5  # axis limits for all 2D plots


def _square(ax, title):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_loss(losses, run_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, alpha=0.3, label="loss")
    w = 200  # moving average to see the trend
    if len(losses) > w:
        plt.plot(np.arange(w - 1, len(losses)), np.convolve(losses, np.ones(w) / w, "valid"), label=f"avg {w}")
    plt.xlabel("step")
    plt.ylabel("MSE (noise)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=120)
    plt.close()


@torch.no_grad()
def plot_input_examples(ddpm, x0, run_dir):
    """What the model is trained on: clean data x_0 and its noised versions x_t = q_sample(x_0, t)."""
    T = ddpm.T
    ts = [0, int(T * 0.1), int(T * 0.25), int(T * 0.5), T - 1]
    fig, axes = plt.subplots(1, len(ts), figsize=(3 * len(ts), 3))
    for ax, t in zip(axes, ts):
        tt = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
        x = (x0 if t == 0 else ddpm.q_sample(x0, tt)).cpu()
        ax.scatter(x[:, 0], x[:, 1], s=1)
        _square(ax, "clean x_0" if t == 0 else f"noised x_t, t={t}")
    plt.suptitle("input examples (forward process)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "input_examples.png"), dpi=120)
    plt.close()


def plot_output_examples(samples, run_dir):
    """What the model generates from pure noise."""
    plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=2, c="tab:orange")
    _square(plt.gca(), f"output examples ({len(samples)} generated points)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "output_examples.png"), dpi=120)
    plt.close()


@torch.no_grad()
def plot_input_output_grid(model, ddpm, x0, run_dir):
    """Big grid. One row per noise level t:
        clean input x_0 | noised input x_t | model output (denoised from t) | overlay
    Points are coloured by their angle in x_0, so you can see if a point returns to where it came from.
    Last row (t = T-1) starts from (almost) pure noise = normal generation."""
    T = ddpm.T
    ts = [int(T * 0.1), int(T * 0.25), int(T * 0.5), int(T * 0.75), T - 1]
    colour = torch.atan2(x0[:, 1], x0[:, 0]).cpu()
    x0_cpu = x0.cpu()

    trajs = {}  # start t -> trajectory, reused for the intermediate-steps figure
    fig, axes = plt.subplots(len(ts), 4, figsize=(16, 4 * len(ts)))
    for row, t in zip(axes, ts):
        tt = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
        x_t = ddpm.q_sample(x0, tt)
        out, trajs[t] = ddpm.sample(model, x0.shape, x_start=x_t, t_start=t, return_trajectory=True)
        out, x_t = out.cpu(), x_t.cpu()

        for ax, x, title in zip(row[:3], [x0_cpu, x_t, out],
                                ["input: clean x_0", f"input: noised x_t (t={t})", f"output: denoised from t={t}"]):
            ax.scatter(x[:, 0], x[:, 1], s=2, c=colour, cmap="hsv")
            _square(ax, title)
        row[3].scatter(x0_cpu[:, 0], x0_cpu[:, 1], s=2, c="lightgray", label="input x_0")
        row[3].scatter(out[:, 0], out[:, 1], s=2, c="tab:orange", label="output")
        row[3].legend(loc="upper right", markerscale=4)
        _square(row[3], "overlay")

    plt.suptitle("inputs vs outputs at different noise levels", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(run_dir, "input_output_grid.png"), dpi=150)
    plt.close()

    plot_input_output_steps(trajs, colour, run_dir)


def plot_input_output_steps(trajs, colour, run_dir, n_cols=7):
    """Same rows as the grid, but columns = intermediate denoising steps from x_t to the output.
    trajs[t][k] = x after k denoising steps starting from x_t (k=0 is the noised input)."""
    fig, axes = plt.subplots(len(trajs), n_cols, figsize=(3 * n_cols, 3 * len(trajs)))
    for row, (t, traj) in zip(axes, trajs.items()):
        n_steps = traj.shape[0] - 1
        ks = np.linspace(0, n_steps, n_cols).round().astype(int)   # evenly spaced snapshots
        for ax, k in zip(row, ks):
            x = traj[k]
            ax.scatter(x[:, 0], x[:, 1], s=1, c=colour, cmap="hsv")
            if k == 0:
                title = f"input x_t, t={t}"
            elif k == n_steps:
                title = "output"
            else:
                title = f"t={t - k}  (step {k}/{n_steps})"
            _square(ax, title)
            ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("intermediate denoising steps (each row starts from a different noise level)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(run_dir, "input_output_steps.png"), dpi=150)
    plt.close()


def plot_samples(real, samples, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(real[:, 0], real[:, 1], s=2)
    _square(axes[0], "real data")
    axes[1].scatter(samples[:, 0], samples[:, 1], s=2, c="tab:orange")
    _square(axes[1], "generated samples")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "samples.png"), dpi=120)
    plt.close()


def plot_reverse_snapshots(traj, T, run_dir):
    """Point cloud at several timesteps of the reverse process: noise -> data."""
    ts = [T, int(T * 0.75), int(T * 0.5), int(T * 0.25), int(T * 0.1), 0]
    fig, axes = plt.subplots(1, len(ts), figsize=(3 * len(ts), 3))
    for ax, t in zip(axes, ts):
        x = traj[T - t]
        ax.scatter(x[:, 0], x[:, 1], s=1)
        _square(ax, f"t = {t}")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "reverse_process.png"), dpi=120)
    plt.close()


def plot_trajectories(traj, samples, run_dir, n_paths=30):
    """Path of individual points during sampling. The zig-zag is the noise DDPM adds back each step."""
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, c="lightgray")
    for i in range(n_paths):
        path = traj[:, i]
        plt.plot(path[:, 0], path[:, 1], lw=0.6, alpha=0.7)
        plt.scatter(path[0, 0], path[0, 1], s=10, c="black")    # start (noise)
        plt.scatter(path[-1, 0], path[-1, 1], s=10, c="red")    # end (sample)
    _square(plt.gca(), "sampling trajectories (black=start, red=end)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "trajectories.png"), dpi=120)
    plt.close()


@torch.no_grad()
def plot_vector_field(model, ddpm, samples, device, run_dir):
    """Arrows = -model(x, t): the direction the model wants to push x towards the data.
    DDPM: -eps (proportional to the score).  Flow: -velocity (the straight-line direction to the data)."""
    T = ddpm.T
    ts = [int(T * 0.75), int(T * 0.25), int(T * 0.05), 0]
    g = torch.linspace(-LIM, LIM, 25)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(device)

    fig, axes = plt.subplots(1, len(ts), figsize=(4.5 * len(ts), 4.5))
    for ax, t in zip(axes, ts):
        tt = torch.full((grid.shape[0],), t, device=device, dtype=torch.long)
        v = -model(grid, ddpm.model_time(tt)).cpu()
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c="lightgray")
        ax.quiver(grid[:, 0].cpu(), grid[:, 1].cpu(), v[:, 0], v[:, 1], color="tab:blue")
        _square(ax, f"-model(x, t={t})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "vector_field.png"), dpi=120)
    plt.close()


# =====================================================================
# MAIN
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test", "all"], required=True, help="all = train then test")
    p.add_argument("--tag", default=None, help="short label for the run name, e.g. bigmodel or noema")
    p.add_argument("--run_name", default=None, help="default: S1.1|S2.3_<method>_mlp_<dataset>[_<tag>]_<time>")
    p.add_argument("--dataset", choices=["spiral", "moons"], default="moons")
    # diffusion
    p.add_argument("--method", choices=["ddpm", "flow"], default="ddpm", help="flow = Stage 2.3 flow matching")
    p.add_argument("--T", type=int, default=None, help="DDPM steps (default 1000) or flow Euler steps (default 100)")
    p.add_argument("--fm_steps", default="2,5,10,20,50,100", help="flow only: Euler steps for the quality sweep")
    # model
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    # training
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_data", type=int, default=50000)
    # testing
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.T is None:
        args.T = 100 if args.method == "flow" else 1000
    if args.run_name is None:
        if args.mode == "test":
            p.error("--run_name is required for test")
        stage = "S2.3" if args.method == "flow" else "S1.1"
        args.run_name = make_run_name(stage, args.method, "mlp", args.dataset, args.tag)

    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(OUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"mode={args.mode} run={args.run_name} device={device}")
    refresh_arch_viz()  # redraws viz/ only if library/models.py changed

    if args.mode in ("train", "all"):
        train(args, run_dir, device)
    if args.mode in ("test", "all"):
        test(args, run_dir, device)


if __name__ == "__main__":
    main()
