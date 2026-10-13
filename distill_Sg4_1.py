"""
Stage 4, project 9 - Fast sampling by PROGRESSIVE DISTILLATION on our CIFAR-10 model (Stage 1.2).
The logic is in library/distill.py.

    round 0     : fine-tune the eps-model into a v-model (better behaved for few-step sampling) = teacher
    rounds 1..8 : student learns to do in 1 step what the teacher does in 2:  256 -> 128 -> ... -> 2 -> 1 steps
Same network size all the time. Only the number of sampling steps shrinks.

Run:  sbatch --time=06:00:00 train.sh distill_Sg4_1.py              (train + test)
      sbatch test.sh distill_Sg4_1.py --run_name <run>               (test again)

Outputs in outputs/<run_name>/
    progress/round_<N>.png    samples of the N-step model after each round (same noise)
    loss_per_round.png        distillation loss of every round
    fid_vs_steps.png          FID vs number of steps: teacher (DDIM) vs distilled students
    same_noise_grid.png       rows = teacher / student, columns = number of steps, same starting noise
    test_input_output.png     TEST images noised to u, then denoised by the teacher and by the students
    test_mse.png              on 500 test images: error to the clean image, and student vs teacher output
    results.json              all numbers
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

from ddpm_images_Sg1_2 import build, show, save_grid, CKPT_DIR, OUT_DIR   # reuse Stage 1.2 helpers
from library.data_images import IMAGE_SHAPE, get_dataset, infinite_batches
from library.distill import Schedule, U_MAX, grid, from_u, ddim_v, v_loss, distill_loss
from library.fid import FID
from library.samplers import ddim
from library.utils import get_device, set_seed, load_checkpoint, make_run_name

SHAPE = IMAGE_SHAPE["cifar10"]


def fixed_noise(n, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((n, *SHAPE), generator=g).to(device)


def load_teacher(run, device):
    """The Stage 1.2 eps-model (EMA weights). dropout=0: we want exact, repeatable outputs while distilling."""
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{run}.pt"), device)
    config = {**ckpt["config"], "dropout": 0.0}
    model, ddpm = build(config, device)
    model.load_state_dict(ckpt["ema"])
    return model.eval(), ddpm, config


def train_loop(model, loss_fn, data, steps, lr, name, device):
    """Adam, lr decays linearly to 0 within the round (as in the paper). Returns the loss history."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 - i / steps)
    model.train()
    losses, t0 = [], time.time()
    for step in range(1, steps + 1):
        loss = loss_fn(next(data).to(device, non_blocking=True))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lr_sched.step()
        losses.append(loss.item())
        if step % 1000 == 0 or step == 1:
            print(f"{name} step {step:6d} | loss {np.mean(losses[-1000:]):.4f} | {step / (time.time() - t0):.1f} it/s", flush=True)
    return model.eval(), losses


# =====================================================================
# TRAIN
# =====================================================================
def train(args, run_dir, device):
    eps_teacher, ddpm, config = load_teacher(args.teacher_run, device)
    sched = Schedule(ddpm)
    data = infinite_batches(get_dataset("cifar10", train=True), args.batch_size, args.num_workers)
    progress_dir = os.path.join(run_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    z_fixed = fixed_noise(16, 0, device)
    ckpt_path = os.path.join(CKPT_DIR, f"{args.run_name}.pt")
    ckpt = {"config": vars(args), "teacher_config": config, "students": {}, "losses": {}}

    # ---- round 0: eps-model -> v-model (starts from the eps weights, so it learns fast) ----
    teacher = copy.deepcopy(eps_teacher)
    teacher, ckpt["losses"]["v"] = train_loop(teacher, lambda x: v_loss(teacher, sched, x), data,
                                              args.v_steps, args.lr, "v-teacher", device)
    n = args.start_steps
    ckpt["students"][2 * n] = teacher.state_dict()                 # the teacher samples with 2n steps
    save_grid(ddim_v(teacher, sched, z_fixed, grid(2 * n)), 4, f"v-teacher, {2 * n} steps",
              os.path.join(progress_dir, f"round_{2 * n:03d}.png"))
    torch.save(ckpt, ckpt_path)

    # ---- distillation rounds: 2n teacher steps -> n student steps ----
    while n >= 1:
        student = copy.deepcopy(teacher)
        student, ckpt["losses"][n] = train_loop(student, lambda x: distill_loss(student, teacher, sched, x, n), data,
                                                args.round_steps, args.lr, f"student {n:3d} steps", device)
        ckpt["students"][n] = student.state_dict()
        torch.save(ckpt, ckpt_path)                               # save after every round
        save_grid(ddim_v(student, sched, z_fixed, grid(n)), 4, f"student, {n} step{'s' * (n > 1)}",
                  os.path.join(progress_dir, f"round_{n:03d}.png"))
        teacher, n = student, n // 2

    plot_losses(ckpt["losses"], run_dir)


# =====================================================================
# TEST
# =====================================================================
def test(args, run_dir, device):
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.run_name}.pt"), device)
    eps_teacher, ddpm, config = load_teacher(ckpt["config"]["teacher_run"], device)
    sched = Schedule(ddpm)
    models = {}
    for n, state in sorted(ckpt["students"].items(), reverse=True):
        m, _ = build(config, device)
        m.load_state_dict(state)
        models[n] = m.eval()
    v_teacher = models[max(models)]                     # round 0 result, the start of the chain
    ns = sorted(models, reverse=True)                   # 512, 256, ..., 1
    print("models (steps):", ns, flush=True)
    results = {"run": args.run_name, "teacher_run": ckpt["config"]["teacher_run"], "n_fid": args.n_fid}

    # samplers we compare. Each: (number of steps) -> function(z) -> images
    samplers = {
        "original eps-teacher, DDIM (Stage 2.1)": lambda n, z: ddim(eps_teacher, ddpm, z, n),
        "v-teacher, DDIM": lambda n, z: ddim_v(v_teacher, sched, z, grid(n)),
        "distilled student": lambda n, z: ddim_v(models[n], sched, z, grid(n)),
    }

    # ---- 1) same starting noise: teacher vs student, every step count ----
    z = fixed_noise(args.n_grid, 1, device)
    plot_same_noise(samplers, ns, z, run_dir)

    # ---- 2) test set: noise real test images to u, denoise with teacher and students ----
    test_set(models, v_teacher, sched, args, results, run_dir, device)

    # ---- 3) FID vs number of steps ----
    if args.n_fid > 0:
        fid = FID(device)
        mu_r, sig_r = fid.real_stats("cifar10", get_dataset("cifar10", train=True), args.n_fid)
        results["fid"] = {}
        for name, fn in samplers.items():
            for n in ns:
                t0 = time.time()
                g = torch.Generator(device="cpu").manual_seed(123)          # same noise for every sampler
                feats = []
                for i in range(0, args.n_fid, args.sample_batch):
                    b = min(args.sample_batch, args.n_fid - i)
                    zb = torch.randn((b, *SHAPE), generator=g).to(device)
                    feats.append(fid.features(fn(n, zb)))
                score = FID.score(mu_r, sig_r, *FID.stats(np.concatenate(feats)))
                results["fid"].setdefault(name, []).append({"steps": n, "fid": score})
                print(f"{name:40s} steps={n:4d} | FID {score:7.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)
                save_json(results, run_dir)
        plot_fid(results["fid"], run_dir)
    save_json(results, run_dir)
    print(json.dumps(results, indent=2))


@torch.no_grad()
def test_set(models, v_teacher, sched, args, results, run_dir, device):
    """Real TEST images (never seen in training), noised to time u, then denoised.
    Teacher: v-teacher with its fine grid. Students: their own few steps from u (see library/distill.from_u)."""
    x_all = torch.stack([get_dataset("cifar10", train=False)[i][0] for i in range(args.n_test)]).to(device)
    us = [250.0, 500.0, 750.0, 1000.0]
    teacher_n = max(models)
    show_ns = [n for n in (16, 8, 4, 2, 1) if n in models]
    g = torch.Generator(device="cpu").manual_seed(7)
    noise = torch.randn(x_all.shape, generator=g).to(device)

    mse = {"to clean": {}, "to teacher": {}}
    fig, axes = plt.subplots(len(us), 3 + len(show_ns), figsize=(2.6 * (3 + len(show_ns)), 3.0 * len(us) + 0.8))
    for r, u in enumerate(us):
        a, s = sched.alpha_sigma(torch.tensor([u], device=device))
        x_u = a * x_all + s * noise
        outs = {f"teacher {teacher_n}": ddim_v(v_teacher, sched, x_u, from_u(teacher_n, u))}
        for n in show_ns:
            outs[f"student {n}"] = ddim_v(models[n], sched, x_u, from_u(n, u))
        ref = outs[f"teacher {teacher_n}"]
        for name, out in outs.items():
            mse["to clean"].setdefault(name, []).append(((out - x_all) ** 2).mean().item())
            mse["to teacher"].setdefault(name, []).append(((out - ref) ** 2).mean().item())

        k = args.n_show
        show(axes[r, 0], x_all[:k], 2, "test image x0")
        show(axes[r, 1], x_u[:k], 2, f"noised to u={u:.0f}")
        for c, (name, out) in enumerate(outs.items()):
            who, n = name.split()
            n_used = len(from_u(int(n), u)) - 1             # steps actually taken from u down to 0
            show(axes[r, 2 + c], out[:k], 2, f"{n}-step {who}: {n_used} step{'s' * (n_used > 1)} from u\n"
                                              f"MSE to x0 {mse['to clean'][name][-1]:.3f}")
    plt.suptitle("TEST images: noise to u (rows), denoise with the teacher and the distilled students\n"
                 "u=1000 is pure noise -> a NEW image, so there MSE to x0 is large for every model", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(run_dir, "test_input_output.png"), dpi=130)
    plt.close()
    results["test_mse"] = {"u": us, "n_images": args.n_test, **mse}
    plot_test_mse(us, mse, run_dir)


# =====================================================================
# PLOTS
# =====================================================================
@torch.no_grad()
def plot_same_noise(samplers, ns, z, run_dir):
    rows = ["v-teacher, DDIM", "distilled student"]
    fig, axes = plt.subplots(len(rows), len(ns), figsize=(2.2 * len(ns), 2.6 * len(rows) + 0.6))
    for r, name in enumerate(rows):
        for c, n in enumerate(ns):
            show(axes[r, c], samplers[name](n, z), 2, f"{name.split(',')[0]}\n{n} step{'s' * (n > 1)}")
    plt.suptitle("same starting noise everywhere: teacher (DDIM, top) vs distilled student (bottom)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(run_dir, "same_noise_grid.png"), dpi=130)
    plt.close()


def plot_losses(losses, run_dir):
    rounds = [k for k in losses if k != "v"]
    fig, axes = plt.subplots(1, len(rounds) + 1, figsize=(2.6 * (len(rounds) + 1), 2.8), sharey=False)
    for ax, k in zip(axes, ["v"] + rounds):
        l = np.array(losses[k])
        ax.plot(l, alpha=0.3)
        w = 200
        if len(l) > w:
            ax.plot(np.arange(w - 1, len(l)), np.convolve(l, np.ones(w) / w, "valid"))
        ax.set_yscale("log")
        ax.set_title("round 0: v-teacher" if k == "v" else f"student {k} steps", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_per_round.png"), dpi=120)
    plt.close()


def plot_fid(fid, run_dir):
    plt.figure(figsize=(7.5, 5))
    for name, rows in fid.items():
        plt.plot([r["steps"] for r in rows], [r["fid"] for r in rows], "o-", label=name)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sampling steps (= model calls)")
    plt.ylabel("FID (lower = better)")
    plt.title("Progressive distillation: quality vs number of steps")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "fid_vs_steps.png"), dpi=130)
    plt.close()


def plot_test_mse(us, mse, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (what, d) in zip(axes, mse.items()):
        for name, vals in d.items():
            if what == "to teacher" and name.startswith("teacher"):
                continue
            ax.plot(us, vals, "o-", label=name)
        ax.set_xlabel("noise level u of the input (1000 = pure noise)")
        ax.set_ylabel("MSE")
        ax.set_title("output vs clean test image" if what == "to clean" else "student output vs teacher output")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "test_mse.png"), dpi=130)
    plt.close()


def save_json(results, run_dir):
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test", "all"], default="all")
    p.add_argument("--teacher_run", default="UNet_cifar10_20261011-115606", help="trained Stage 1.2 CIFAR run")
    p.add_argument("--tag", default=None)
    p.add_argument("--run_name", default=None, help="default: S4.1_progdistill_unet_cifar10[_<tag>]_<time>")
    p.add_argument("--start_steps", type=int, default=256, help="first student uses this many steps (teacher 2x)")
    p.add_argument("--v_steps", type=int, default=20000, help="round 0: eps-model -> v-model fine-tuning")
    p.add_argument("--round_steps", type=int, default=10000, help="training steps per halving round")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_fid", type=int, default=5000, help="0 = skip FID (5000 = same as Stage 1.2 / 2.1)")
    p.add_argument("--sample_batch", type=int, default=500)
    p.add_argument("--n_grid", type=int, default=4, help="images per cell in same_noise_grid.png")
    p.add_argument("--n_test", type=int, default=500, help="test images for test_mse")
    p.add_argument("--n_show", type=int, default=4, help="test images shown per cell")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = make_run_name("S4.1", "progdistill", "unet", "cifar10", args.tag)

    set_seed(args.seed)
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = os.path.join(OUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"run={args.run_name} device={device}", flush=True)
    if args.mode in ("train", "all"):
        train(args, run_dir, device)
    if args.mode in ("test", "all"):
        test(args, run_dir, device)


if __name__ == "__main__":
    main()
