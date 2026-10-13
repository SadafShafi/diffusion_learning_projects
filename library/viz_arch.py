"""
Draw every architecture in library.models.ARCHITECTURES and save to viz/.

    python -m library.viz_arch            # redraw only if library/models.py changed
    python -m library.viz_arch --force    # always redraw

How it works: run the model once on a dummy input with "forward hooks" on every
leaf layer. Hooks fire in execution order, so we get the real data flow:
layer name, type, output shape and number of parameters.
"""
import argparse
import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from library import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR = os.path.join(ROOT, "viz")
HASH_FILE = os.path.join(VIZ_DIR, ".models_hash")


def trace(model, inputs):
    """Return a list of (name, type, output shape, n_params) in execution order."""
    rows, hooks = [], []
    for name, m in model.named_modules():
        if len(list(m.children())) > 0 or isinstance(m, torch.nn.Identity):   # only real leaf layers
            continue
        def hook(mod, inp, out, name=name):
            shape = tuple(out.shape) if torch.is_tensor(out) else "?"
            n = sum(p.numel() for p in mod.parameters())
            rows.append((name, type(mod).__name__, shape, n))
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(*inputs)
    for h in hooks:
        h.remove()
    return rows


def draw(name, rows, inputs, total_params, out_dir, suffix=""):
    in_shapes = ", ".join(str(tuple(x.shape)) for x in inputs)
    groups = sorted({r[0].split(".")[0] for r in rows})            # top-level block -> colour
    colours = {g: plt.cm.Pastel1(i % 9) for i, g in enumerate(groups)}

    nw = max(len(r[0]) for r in rows)              # column widths
    tw = max(len(r[1]) for r in rows)
    sw = max(len(str(r[2])) for r in rows)
    n = len(rows) + 2
    fig, ax = plt.subplots(figsize=(max(7, 0.09 * (nw + tw + sw + 16)), 0.55 * n + 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(-n, 1)
    ax.axis("off")

    def box(y, text, colour):
        ax.text(5, y, text, ha="center", va="center", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc=colour, ec="gray"))

    box(0, f"input  {in_shapes}", "white")
    for i, (lname, ltype, shape, nparams) in enumerate(rows, start=1):
        ax.annotate("", xy=(5, -i + 0.25), xytext=(5, -i + 0.75), arrowprops=dict(arrowstyle="->", color="gray"))
        box(-i, f"{lname:<{nw}} {ltype:<{tw}} {str(shape):<{sw}} {nparams:>9,} p", colours[lname.split('.')[0]])
    ax.annotate("", xy=(5, -n + 1.25), xytext=(5, -n + 1.75), arrowprops=dict(arrowstyle="->", color="gray"))
    box(-n + 1, "output", "white")

    ax.set_title(f"{name}   ({total_params:,} parameters)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}{suffix}.png"), dpi=120)
    plt.close()


def models_hash():
    """Fingerprint of models.py AND this file, so changing a drawing also triggers a redraw."""
    h = hashlib.md5()
    for path in (models.__file__, __file__):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


# =====================================================================
# U-Net: block-level diagram in a U shape
# =====================================================================
def nparams(m):
    return sum(p.numel() for p in m.parameters())


def unet_shapes(model, inputs):
    """Run the model once and record the output shape (C, H, W) of each level / the middle."""
    shapes, hooks = {}, []
    def rec(key):
        return lambda mod, inp, out: shapes.__setitem__(key, tuple(out.shape[1:]))
    for i, level in enumerate(model.down):
        hooks.append(level.blocks[-1].register_forward_hook(rec(("down", i))))
    for j, level in enumerate(model.up):
        i = len(model.up) - 1 - j                      # model.up is stored bottom -> top
        hooks.append(level.blocks[-1].register_forward_hook(rec(("up", i))))
    hooks.append(model.mid2.register_forward_hook(rec("mid")))
    with torch.no_grad():
        model(*inputs)
    for h in hooks:
        h.remove()
    return shapes


def draw_unet(name, model, inputs, out_dir):
    from matplotlib.patches import FancyBboxPatch
    shapes = unet_shapes(model, inputs)
    L = len(model.down)
    n_res = len(model.down[0].blocks)
    has_attn = lambda level: not isinstance(level.attn, torch.nn.Identity)
    fmt = lambda s: f"{s[0]}ch × {s[1]}×{s[2]}"
    C_in, H, W = inputs[0].shape[1:]

    fig, ax = plt.subplots(figsize=(13, 2.3 * L + 3.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(-2.3 * (L - 1) - 3.0, 4.0)
    ax.axis("off")

    def box(x, y, text, fc, w=3.6, h=1.3, ec="gray", lw=1):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=lw))
        ax.text(x, y, text, ha="center", va="center", fontsize=9, family="monospace")

    def arrow(x1, y1, x2, y2, text="", color="gray", style="-|>", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5, ls=ls))
        if text:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, text, fontsize=8, color=color, va="center")

    XL, XR, XM = 2.6, 10.4, 6.5
    y_of = lambda i: -2.3 * i
    ATTN = "#ffe0b2"                                   # orange-ish = has attention

    # time embedding
    t_dim = model.time_emb[-1].out_features
    if getattr(model, "num_classes", None) is None:
        box(XM, 3.3, f"t  ->  sinusoidal  ->  MLP  ->  t_emb ({t_dim})\n{nparams(model.time_emb):,} p\n"
                     f"added inside EVERY ResBlock", "#f8bbd0", w=5.6, h=1.2)
    else:   # class-conditional (Stage 2.2)
        box(XM, 3.3, f"t -> sinusoidal -> MLP ({nparams(model.time_emb):,} p)\n"
                     f"+  class y -> Embedding({model.num_classes} classes + 1 null) ({nparams(model.class_emb):,} p)\n"
                     f"= t_emb ({t_dim}), added inside EVERY ResBlock", "#f8bbd0", w=7.4, h=1.2)

    # input / output
    box(XL, 1.9, f"input x_t\n{C_in}ch × {H}×{W}", "white", h=0.9)
    box(XR, 1.9, f"output (predicted noise)\n{C_in}ch × {H}×{W}", "white", h=0.9)
    arrow(XL, 1.45, XL, y_of(0) + 0.7, f"conv_in ({nparams(model.conv_in):,} p)")
    arrow(XR, y_of(0) + 0.7, XR, 1.45, f"norm+conv_out ({nparams(model.norm_out) + nparams(model.conv_out):,} p)")

    for i in range(L):
        d, u = model.down[i], model.up[L - 1 - i]
        y = y_of(i)
        box(XL, y, f"Down {i}: {n_res}× ResBlock{' + Attn' if has_attn(d) else ''}\n"
                   f"{fmt(shapes[('down', i)])}\n{nparams(d.blocks) + nparams(d.attn):,} p",
            ATTN if has_attn(d) else "#c8e6c9", ec="darkorange" if has_attn(d) else "gray", lw=2 if has_attn(d) else 1)
        box(XR, y, f"Up {i}: {n_res}× ResBlock{' + Attn' if has_attn(u) else ''}\n"
                   f"{fmt(shapes[('up', i)])}\n{nparams(u.blocks) + nparams(u.attn):,} p",
            ATTN if has_attn(u) else "#bbdefb", ec="darkorange" if has_attn(u) else "gray", lw=2 if has_attn(u) else 1)
        arrow(XL + 1.85, y, XR - 1.85, y, "", color="tab:purple", ls="--")
        ax.text(XM, y + 0.18, f"skip: concat {shapes[('down', i)][0]}ch", ha="center", fontsize=8, color="tab:purple")
        if i < L - 1:
            arrow(XL, y - 0.7, XL, y_of(i + 1) + 0.7, f"Downsample ({nparams(d.downsample):,} p)")
            arrow(XR, y_of(i + 1) + 0.7, XR, y - 0.7, f"Upsample ({nparams(model.up[L - 2 - i].upsample):,} p)")

    # middle
    ym = y_of(L - 1) - 2.0
    mid_p = nparams(model.mid1) + nparams(model.mid_attn) + nparams(model.mid2)
    box(XM, ym, f"Middle: ResBlock - Attn - ResBlock\n{fmt(shapes['mid'])}\n{mid_p:,} p", ATTN,
        w=4.4, ec="darkorange", lw=2)
    arrow(XL, y_of(L - 1) - 0.7, XM - 2.2, ym)
    arrow(XM + 2.2, ym, XR, y_of(L - 1) - 0.7)

    total = nparams(model)
    ax.set_title(f"{name}   ({total:,} parameters)\n"
                 f"green = encoder, blue = decoder, orange border = self-attention, dashed purple = skip connection",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=130)
    plt.close()


# =====================================================================
# DiT: token pipeline (left) + zoom into one adaLN-Zero block (right)
# =====================================================================
def draw_dit(name, model, inputs, out_dir):
    from matplotlib.patches import FancyBboxPatch
    C, H, W = inputs[0].shape[1:]
    p, g = model.patch, model.grid
    blk = model.blocks[0]
    dim = blk.norm1.normalized_shape[0]
    heads, mlp_h, depth = blk.attn.heads, blk.mlp[0].out_features, len(model.blocks)

    fig, ax = plt.subplots(figsize=(17, 11))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(x, y, text, fc, w=4.6, h=0.8, ec="gray", lw=1, fs=9):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=lw))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, family="monospace")

    def arrow(x1, y1, x2, y2, color="gray", ls="-", text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, ls=ls))
        if text:
            ax.text((x1 + x2) / 2 + 0.08, (y1 + y2) / 2, text, fontsize=8, color=color, va="center")

    COND = "#f8bbd0"
    # ---------------- left: the pipeline ----------------
    X = 3.0
    ys = [10.2, 8.9, 7.6, 5.6, 3.6, 2.3]
    box(X, ys[0], f"input: noisy latent  {C} x {H} x {W}", "white")
    box(X, ys[1], f"patchify: {p}x{p} conv, stride {p}  ({nparams(model.patch_embed):,} p)\n"
                  f"-> {g}x{g} = {g * g} tokens of size {dim}", "#c8e6c9", h=0.9)
    box(X, ys[2], f"+ fixed 2D sin-cos position embedding\n(each token knows where its patch is)", "#c8e6c9", h=0.9)
    box(X, ys[3], f"{depth} x DiT block (adaLN-Zero)\n{depth} x {nparams(blk):,} = {nparams(model.blocks):,} p\n"
                  f"tokens stay {g * g} x {dim}", "#ffe0b2", h=1.3, ec="darkorange", lw=2)
    box(X, ys[4], f"final layer: LN + adaLN (shift, scale)\n+ Linear {dim} -> {p}*{p}*{C} = {p * p * C}  "
                  f"({nparams(model.final_adaLN) + nparams(model.final_linear):,} p)", "#bbdefb", h=0.9)
    box(X, ys[5], f"unpatchify -> output {C} x {H} x {W}\n(predicted noise / velocity)", "white", h=0.9)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(X, a - 0.45, X, b + 0.5)
    # condition
    cy = 6.9
    box(X + 4.3, cy + 1.2, f"t -> sinusoidal(256) -> MLP\n-> condition c ({dim})\n({nparams(model.time_emb):,} p)",
        COND, w=3.2, h=1.1)
    if model.num_classes is not None:
        ax.text(X + 4.3, cy + 0.45, f"+ class embedding ({model.num_classes}+1 null)", ha="center", fontsize=8)
    arrow(X + 4.3, cy + 0.65, X + 2.35, ys[3] + 0.2, color="deeppink", ls="--", text="c")
    arrow(X + 4.3, cy + 0.65, X + 2.35, ys[4] + 0.2, color="deeppink", ls="--")
    ax.text(X, 0.9, f"{name}: {nparams(model):,} parameters  |  {g * g} tokens  |  patch {p}  |  "
                    f"{heads} heads", ha="center", fontsize=11)

    # ---------------- right: one block ----------------
    XB = 12.6
    ax.add_patch(FancyBboxPatch((9.3, 0.5), 7.4, 10.3, boxstyle="round,pad=0.05", fc="#fff8ef", ec="darkorange", lw=2))
    ax.text(XB, 10.5, "inside ONE DiT block (adaLN-Zero)", ha="center", fontsize=12, color="darkorange")
    box(XB, 9.65, f"c -> SiLU -> Linear({dim} -> 6x{dim})   ({nparams(blk.adaLN):,} p, starts at 0)\n"
                  f"= shift1 scale1 gate1 | shift2 scale2 gate2", COND, w=6.9, h=0.85)
    yb = [8.5, 7.6, 6.7, 5.8, 4.9, 3.6, 2.7, 1.8, 0.95]
    labels = [("x  (tokens)", "white"),
              ("LayerNorm (no own gamma/beta)", "#e3f2fd"),
              ("x (1 + scale1) + shift1", COND),
              (f"self-attention, {heads} heads  ({nparams(blk.attn):,} p)", "#ffe0b2"),
              ("x gate1   ->   + residual", COND),
              ("LayerNorm", "#e3f2fd"),
              ("x (1 + scale2) + shift2", COND),
              (f"MLP {dim} -> {mlp_h} -> {dim}, GELU  ({nparams(blk.mlp):,} p)", "#ffe0b2"),
              ("x gate2   ->   + residual  ->  x out", COND)]
    for (t, fc), y in zip(labels, yb):
        box(XB, y, t, fc, w=5.2, h=0.6)
    for a, b in zip(yb[:-1], yb[1:]):
        arrow(XB, a - 0.3, XB, b + 0.3)
    # residual connections
    for top, bottom in [(yb[0], yb[4]), (yb[4], yb[8])]:
        ax.plot([XB - 2.75, XB - 3.0, XB - 3.0, XB - 2.75], [top, top, bottom, bottom], color="tab:purple", lw=1.5)
    ax.text(XB - 3.15, (yb[0] + yb[4]) / 2, "residual", rotation=90, color="tab:purple", va="center", fontsize=8)
    ax.text(XB - 3.15, (yb[4] + yb[8]) / 2, "residual", rotation=90, color="tab:purple", va="center", fontsize=8)
    # condition feeds the modulation boxes
    for y in (yb[2], yb[4], yb[6], yb[8]):
        ax.annotate("", xy=(XB + 2.65, y), xytext=(XB + 3.4, 9.25),
                    arrowprops=dict(arrowstyle="-|>", color="deeppink", lw=1, ls="--"))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=130)
    plt.close()


def refresh(force=False):
    """Redraw all architectures if library/models.py changed since last time."""
    os.makedirs(VIZ_DIR, exist_ok=True)
    h = models_hash()
    if not force and os.path.exists(HASH_FILE) and open(HASH_FILE).read() == h:
        return
    for name, (build, make_inputs) in models.ARCHITECTURES.items():
        model, inputs = build().eval(), make_inputs()
        rows = trace(model, inputs)
        total = sum(p.numel() for p in model.parameters())
        if isinstance(model, models.UNet):
            draw_unet(name, model, inputs, VIZ_DIR)                    # U-shaped block diagram
            draw(name, rows, inputs, total, VIZ_DIR, suffix="_layers")  # full per-layer list
        elif isinstance(model, models.DiT):
            draw_dit(name, model, inputs, VIZ_DIR)                     # token pipeline + one block
            draw(name, rows, inputs, total, VIZ_DIR, suffix="_layers")
        else:
            draw(name, rows, inputs, total, VIZ_DIR)
        with open(os.path.join(VIZ_DIR, f"{name}.txt"), "w") as f:
            f.write(f"{model}\n\nTotal parameters: {total:,}\n")
        print(f"[viz] drew {name} -> viz/{name}.png")
    with open(HASH_FILE, "w") as f:
        f.write(h)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    refresh(force=p.parse_args().force)
