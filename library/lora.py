"""LoRA: Low-Rank Adaptation (Hu et al., 2021).   (Stage 3, project 8)

Fine-tuning all 860M U-Net weights is expensive. LoRA freezes a Linear layer W and learns a small update:
    y = W x  +  (alpha / r) * B (A x)          A: r x in,   B: out x r,   r << in, out  (e.g. r = 8)
B starts at ZERO, so at the start the model is exactly the pretrained one.
We put LoRA on the attention projections (to_q, to_k, to_v, to_out) of every attention layer in the U-Net.
"""
import math

import torch
import torch.nn as nn

TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=8):
        super().__init__()
        self.base = base                                          # frozen pretrained layer
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        lora = (x.float() @ self.A.t() @ self.B.t()) * self.scale   # LoRA math in fp32
        return self.base(x) + lora.to(x.dtype)


def inject_lora(unet, rank=8, alpha=8):
    """Replace every attention projection in the U-Net by a LoRALinear. Returns the new trainable params."""
    params = []
    for name, module in list(unet.named_modules()):
        if not name.endswith(("attn1", "attn2")):                # attn1 = self-attention, attn2 = cross-attention
            continue
        for target in TARGETS:
            parent_name, _, child = target.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            if isinstance(getattr(parent, child, None), nn.Linear):
                lora = LoRALinear(getattr(parent, child), rank, alpha).to(next(unet.parameters()).device)
                setattr(parent, child, lora)
                params += [lora.A, lora.B]
    return params


def lora_state_dict(unet):
    return {k: v for k, v in unet.state_dict().items() if k.endswith((".A", ".B"))}


def load_lora(unet, state, rank=8, alpha=8):
    inject_lora(unet, rank, alpha)
    missing = set(state) - set(unet.state_dict())
    assert not missing, f"unknown LoRA keys: {list(missing)[:3]}"
    unet.load_state_dict(state, strict=False)
