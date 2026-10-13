"""Denoising networks. eps_theta(x_t, t) -> predicted noise (same shape as x_t)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Turn an integer timestep t into a vector (same idea as Transformer positional encoding).
    Lets the network know *how noisy* its input is."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class MLP(nn.Module):
    """Small MLP for 2D toy data. Input: point (x, y) + time embedding. Output: predicted noise (2D)."""

    def __init__(self, data_dim=2, hidden=128, time_dim=32, n_layers=3):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        layers = [nn.Linear(data_dim + time_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        h = torch.cat([x, self.time_emb(t)], dim=1)
        return self.net(h)


# =====================================================================
# U-Net for images (Stage 1.2)
# =====================================================================
class ResBlock(nn.Module):
    """Two 3x3 convs + skip connection. The timestep embedding is added in the middle,
    so every block knows how noisy the image is."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]   # add time, broadcast over H, W
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention over pixels: every pixel looks at every other pixel.
    Cost grows with (H*W)^2, so we only use it at low resolution."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)   # each (B, C, N)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)   # (B, N, N)
        out = v @ attn.transpose(1, 2)                                         # (B, C, N)
        return x + self.proj(out.reshape(B, C, H, W))


class Downsample(nn.Module):
    """Halve H and W (28 -> 14 -> 7, or 32 -> 16 -> 8)."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Double H and W: nearest-neighbour resize, then a conv."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """
    Encoder: at each level, n_res ResBlocks (+ attention), save a skip, then downsample.
    Middle:  ResBlock - Attention - ResBlock at the lowest resolution.
    Decoder: at each level, concat the skip, n_res ResBlocks (+ attention), then upsample.

    ch_mults:    channels at each level = base_ch * mult (one level per entry)
    attn_levels: which levels (0 = full resolution) get self-attention
    num_classes: None = unconditional. Otherwise a class embedding is ADDED to the time embedding
                 (Stage 2.2). Label num_classes means "no label" (null class), used for CFG.
    """

    def __init__(self, in_ch=1, base_ch=32, ch_mults=(1, 2, 2), n_res=2, attn_levels=(2,), dropout=0.1,
                 num_classes=None):
        super().__init__()
        time_dim = base_ch * 4
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes + 1, time_dim)   # +1 for the null class
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ---- encoder ----
        self.down = nn.ModuleList()
        skip_chs = []
        ch = base_ch
        for i, m in enumerate(ch_mults):
            out = base_ch * m
            level = nn.Module()
            level.blocks = nn.ModuleList([ResBlock(ch if j == 0 else out, out, time_dim, dropout) for j in range(n_res)])
            level.attn = AttentionBlock(out) if i in attn_levels else nn.Identity()
            level.downsample = Downsample(out) if i < len(ch_mults) - 1 else nn.Identity()
            self.down.append(level)
            skip_chs.append(out)
            ch = out

        # ---- middle ----
        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        # ---- decoder ----
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mults))):
            out = base_ch * m
            level = nn.Module()
            level.blocks = nn.ModuleList(
                [ResBlock(ch + skip_chs[i] if j == 0 else out, out, time_dim, dropout) for j in range(n_res)])
            level.attn = AttentionBlock(out) if i in attn_levels else nn.Identity()
            level.upsample = Upsample(out) if i > 0 else nn.Identity()
            self.up.append(level)
            ch = out

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t, y=None):
        t_emb = self.time_emb(t)
        if self.num_classes is not None:
            if y is None:                                  # no label given -> null class
                y = torch.full_like(t, self.num_classes)
            t_emb = t_emb + self.class_emb(y)              # "which class" is told the same way as "which t"
        h = self.conv_in(x)

        skips = []
        for level in self.down:
            for block in level.blocks:
                h = block(h, t_emb)
            h = level.attn(h)
            skips.append(h)
            h = level.downsample(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for level in self.up:
            h = torch.cat([h, skips.pop()], dim=1)   # skip connection from the same resolution
            for block in level.blocks:
                h = block(h, t_emb)
            h = level.attn(h)
            h = level.upsample(h)

        return self.conv_out(F.silu(self.norm_out(h)))


# =====================================================================
# Diffusion Transformer, DiT (Peebles & Xie, 2023)   (Stage 3, project 7)
# =====================================================================
def sincos_2d(dim, grid):
    """Fixed 2D sin-cos position embedding (grid*grid, dim): half the dims encode the row, half the column."""
    def sincos_1d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2))
        out = pos[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return torch.cat([sincos_1d(dim // 2, rows.flatten().float()), sincos_1d(dim // 2, cols.flatten().float())], dim=1)


def modulate(x, shift, scale):
    """adaLN: normalised tokens are scaled and shifted with numbers computed from the condition."""
    return x * (1 + scale[:, None]) + shift[:, None]


class SelfAttention(nn.Module):
    """Multi-head self-attention over tokens:  softmax(Q K^T / sqrt(d)) V   per head."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)  # (B, heads, N, d)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning. From the condition c it computes 6 vectors:
        x = x + gate1 * Attention( LN(x) * (1 + scale1) + shift1 )
        x = x + gate2 * MLP(       LN(x) * (1 + scale2) + shift2 )
    The layer producing them is initialised to ZERO -> gates start at 0 -> every block starts as the identity."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)   # no own gamma/beta: adaLN provides them
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate="tanh"),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=1)
        x = x + gate1[:, None] * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2[:, None] * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class DiT(nn.Module):
    """
    latent (C, H, W) --patchify--> N = (H/p)^2 tokens of size dim --+ 2D position embedding
        --> depth x DiTBlock (each conditioned on c = time embedding [+ class embedding])
        --> final adaLN + Linear: each token -> a p x p x C patch --unpatchify--> output (C, H, W)
    Same call as the U-Net: model(x, t, y=None) -> predicted noise / velocity.
    """

    def __init__(self, in_ch=4, input_size=16, patch=2, dim=384, depth=8, heads=6, num_classes=None):
        super().__init__()
        self.in_ch, self.patch, self.grid = in_ch, patch, input_size // patch
        self.num_classes = num_classes
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)   # each p x p patch -> 1 token
        self.register_buffer("pos_emb", sincos_2d(dim, self.grid)[None], persistent=False)
        self.time_emb = nn.Sequential(SinusoidalTimeEmbedding(256), nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes + 1, dim)                       # +1 = null class (CFG)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_linear = nn.Linear(dim, patch * patch * in_ch)
        for layer in (self.final_adaLN[1], self.final_linear):                     # output starts at 0
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t, y=None):
        B = x.shape[0]
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_emb          # (B, N, dim)
        c = self.time_emb(t)                                                       # the condition
        if self.num_classes is not None:
            y = torch.full((B,), self.num_classes, device=x.device, dtype=torch.long) if y is None else y
            c = c + self.class_emb(y)
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        h = self.final_linear(modulate(self.final_norm(h), shift, scale))         # (B, N, p*p*C)
        p, g, C = self.patch, self.grid, self.in_ch                                # unpatchify
        return h.reshape(B, g, g, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)


# =====================================================================
# Helper classifier (only used to measure things, not for generation)
# =====================================================================
class MNISTClassifier(nn.Module):
    """Small CNN (~99% test accuracy). Only used to *measure* things, e.g. which digit a sample is."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14 -> 7
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Registry used by library/viz_arch.py to draw every architecture into viz/.
# When you add a model, add an entry: name -> (build the model, build a dummy input tuple).
# ---------------------------------------------------------------------
ARCHITECTURES = {
    "MLP": (lambda: MLP(), lambda: (torch.randn(4, 2), torch.randint(0, 1000, (4,)))),
    # keep these in sync with PRESETS in ddpm_images_Sg1_2.py
    "UNet_mnist": (lambda: UNet(in_ch=1, base_ch=32, ch_mults=(1, 2, 2), attn_levels=(2,)),
                   lambda: (torch.randn(2, 1, 28, 28), torch.randint(0, 1000, (2,)))),
    "UNet_cifar10": (lambda: UNet(in_ch=3, base_ch=64, ch_mults=(1, 2, 2), attn_levels=(1, 2)),
                     lambda: (torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,)))),
    "UNet_cifar10_cond": (lambda: UNet(in_ch=3, base_ch=64, ch_mults=(1, 2, 2), attn_levels=(1, 2), num_classes=10),
                          lambda: (torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,)))),
    "UNet_mnist_cond": (lambda: UNet(in_ch=1, base_ch=32, ch_mults=(1, 2, 2), attn_levels=(2,), num_classes=10),
                        lambda: (torch.randn(2, 1, 28, 28), torch.randint(0, 1000, (2,)))),
    # Stage 3 (latent CelebA, 4x16x16): keep in sync with latent_Sg3_1.py defaults
    "UNet_celeba_latent": (lambda: UNet(in_ch=4, base_ch=128, ch_mults=(1, 2, 2), attn_levels=(1, 2)),
                           lambda: (torch.randn(2, 4, 16, 16), torch.randint(0, 1000, (2,)))),
    "DiT_celeba_latent_p2": (lambda: DiT(in_ch=4, input_size=16, patch=2, dim=384, depth=8, heads=6),
                             lambda: (torch.randn(2, 4, 16, 16), torch.randint(0, 1000, (2,)))),
    "MNISTClassifier": (lambda: MNISTClassifier(), lambda: (torch.randn(2, 1, 28, 28),)),
}
