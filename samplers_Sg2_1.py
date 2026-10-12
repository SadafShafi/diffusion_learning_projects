"""
Stage 2, project 3 - Samplers: DDIM, Euler and Heun on the trained CIFAR-10 (or MNIST) model.
No training: we only change HOW we sample from an already trained model.

Run:  python samplers_Sg2_1.py --mode test --ckpt_run UNet_cifar10_20261011-115606
      (slurm: sbatch test.sh samplers_Sg2_1.py --ckpt_run UNet_cifar10_20261011-115606)

Outputs in outputs/<run_name>/   (run_name default: Samplers_<dataset>_<timestamp>)
    fid_vs_nfe.png        quality vs cost (number of model calls) for every sampler
    same_noise_grid.png   rows = sampler, columns = number of steps, SAME starting noise everywhere
    results.json          FID per sampler/steps, NFE, time, Euler-vs-DDIM check
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ddpm_images_Sg1_2 import build, show, CKPT_DIR, OUT_DIR   # reuse the Stage 1.2 model builder + plot helper
from library.data_images import IMAGE_SHAPE, get_dataset
from library.fid import FID
from library.samplers import SAMPLERS
from library.utils import get_device, set_seed, load_checkpoint, make_run_name
from library.viz_arch import refresh as refresh_arch_viz


class CountCalls(torch.nn.Module):
    """Wraps the model and counts how often it is called (= NFE, number of function evaluations)."""

    def __init__(self, model):
        super().__init__()
        self.model, self.calls = model, 0

    def forward(self, x, t):
        self.calls += 1
        return self.model(x, t)


def compare(args, run_dir, device):
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{args.ckpt_run}.pt"), device)
    config = ckpt["config"]
    shape = IMAGE_SHAPE[config["dataset"]]
    model, ddpm = build(config, device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    model = CountCalls(model)

    steps = [int(s) for s in args.steps.split(",")]
    results = {"ckpt_run": args.ckpt_run, "n_fid": args.n_fid, "fid": {}}

    # ---------------------------------------------------------------
    # 1) Same starting noise for every sampler / step count
    # ---------------------------------------------------------------
    g = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn((args.n_grid, *shape), generator=g).to(device)
    grid = {}
    for name, (fn, nfe) in SAMPLERS.items():
        for n in steps:
            model.calls = 0
            grid[(name, n)] = fn(model, ddpm, z, n).cpu()
            assert model.calls == nfe(n), (name, n, model.calls)     # check our NFE formula
    plot_grid(grid, steps, run_dir)

    # Euler on the ODE should give the SAME images as DDIM(eta=0) -> check numerically
    diffs = {n: (grid[("euler", n)] - grid[("ddim", n)]).abs().max().item() for n in steps}
    results["max_abs_diff_euler_vs_ddim"] = diffs
    print("max |euler - ddim| per steps:", {n: f"{d:.1e}" for n, d in diffs.items()}, flush=True)

    # ---------------------------------------------------------------
    # 2) FID vs NFE
    # ---------------------------------------------------------------
    if args.n_fid > 0:
        fid = FID(device)
        mu_r, sig_r = fid.real_stats(config["dataset"], get_dataset(config["dataset"], train=True), args.n_fid)
        for name in args.fid_samplers.split(","):
            fn, nfe = SAMPLERS[name]
            for n in steps:
                if name == "heun" and n > args.heun_max_steps:
                    continue                      # Heun converges early; big n is just expensive
                t0 = time.time()
                feats = []
                for i in range(0, args.n_fid, args.sample_batch):
                    b = min(args.sample_batch, args.n_fid - i)
                    feats.append(fid.features(fn(model, ddpm, torch.randn((b, *shape), device=device), n)))
                score = FID.score(mu_r, sig_r, *FID.stats(np.concatenate(feats)))
                results["fid"].setdefault(name, []).append(
                    {"steps": n, "nfe": nfe(n), "fid": score, "minutes": (time.time() - t0) / 60})
                print(f"{name:10s} steps={n:5d} NFE={nfe(n):5d} | FID {score:7.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)
                with open(os.path.join(run_dir, "results.json"), "w") as f:   # save as we go
                    json.dump(results, f, indent=2)
        plot_fid(results["fid"], run_dir)

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


def plot_grid(grid, steps, run_dir):
    names = list(SAMPLERS)
    fig, axes = plt.subplots(len(names), len(steps), figsize=(2.3 * len(steps), 2.6 * len(names) + 0.6))
    for r, name in enumerate(names):
        for c, n in enumerate(steps):
            show(axes[r, c], grid[(name, n)], 2, f"{name}, {n} steps\nNFE={SAMPLERS[name][1](n)}")
    plt.suptitle("same starting noise in every cell", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(run_dir, "same_noise_grid.png"), dpi=130)
    plt.close()


def plot_fid(fid, run_dir):
    plt.figure(figsize=(7.5, 5))
    for name, rows in fid.items():
        plt.plot([r["nfe"] for r in rows], [r["fid"] for r in rows], "o-", label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("NFE (number of model calls) - log scale")
    plt.ylabel("FID (lower = better) - log scale")
    plt.title("Sample quality vs sampling cost")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "fid_vs_nfe.png"), dpi=130)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test"], default="test", help="only sampling, no training")
    p.add_argument("--ckpt_run", required=True, help="trained Stage 1.2 run to sample from")
    p.add_argument("--tag", default=None, help="short label for the run name, e.g. bigmodel or noema")
    p.add_argument("--run_name", default=None, help="default: S2.1_samplers_unet_<dataset>[_<tag>]_<time>")
    p.add_argument("--steps", default="10,20,50,100,250,1000")
    p.add_argument("--n_grid", type=int, default=4, help="images per cell in same_noise_grid.png")
    p.add_argument("--n_fid", type=int, default=5000, help="0 = skip FID (5000 = same as Stage 1.2)")
    p.add_argument("--fid_samplers", default="ddim_eta1,ddim,heun",
                   help="euler is skipped by default: it equals ddim (checked numerically)")
    p.add_argument("--heun_max_steps", type=int, default=250)
    p.add_argument("--sample_batch", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.run_name is None:
        dataset = load_checkpoint(os.path.join(CKPT_DIR, f"{args.ckpt_run}.pt"), "cpu")["config"]["dataset"]
        args.run_name = make_run_name("S2.1", "samplers", "unet", dataset, args.tag)

    set_seed(args.seed)
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = os.path.join(OUT_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"run={args.run_name} sampling from {args.ckpt_run} device={device}")
    refresh_arch_viz()
    compare(args, run_dir, device)


if __name__ == "__main__":
    main()
