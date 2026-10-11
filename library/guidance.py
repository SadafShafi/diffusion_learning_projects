"""Classifier-free guidance (Ho & Salimans, 2022).   (Stage 2.2)

The model is trained to predict noise both WITH a label (eps_cond) and WITHOUT (eps_uncond, null class).
At sampling time we push the prediction further in the direction the label points to:

    eps = eps_uncond + w * (eps_cond - eps_uncond)

    w = 0  -> unconditional model (ignores the label)
    w = 1  -> plain conditional model
    w > 1  -> guided: more "typical" for the class (higher fidelity), but less variety (lower diversity)

Guided looks like a normal model(x, t) from the outside, so every sampler from Stage 2.1 works unchanged.
"""
import torch


class Guided(torch.nn.Module):
    def __init__(self, model, y, w):
        super().__init__()
        self.model, self.y, self.w = model, y, w

    def forward(self, x, t):
        if self.w == 1:                                    # no guidance needed: one forward pass
            return self.model(x, t, self.y)
        null = torch.full_like(self.y, self.model.num_classes)
        # one forward pass for both: first half with labels, second half with the null class
        eps = self.model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([self.y, null]))
        eps_cond, eps_uncond = eps.chunk(2)
        return eps_uncond + self.w * (eps_cond - eps_uncond)
