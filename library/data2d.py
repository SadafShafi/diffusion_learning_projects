"""2D toy datasets. Each returns a float tensor of shape [n, 2], roughly zero-mean, unit-ish scale."""
import numpy as np
import torch
from sklearn.datasets import make_moons


def spiral(n, noise=0.05):
    theta = np.sqrt(np.random.rand(n)) * 3 * np.pi       # angle grows along the spiral
    r = theta / (3 * np.pi) * 2                           # radius grows with angle
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x += noise * np.random.randn(n, 2)
    return torch.tensor(x, dtype=torch.float32)


def moons(n, noise=0.05):
    x, _ = make_moons(n_samples=n, noise=noise)
    x = (x - x.mean(0)) / x.std(0)                        # normalise
    return torch.tensor(x, dtype=torch.float32)


DATASETS = {"spiral": spiral, "moons": moons}


def get_data(name, n):
    return DATASETS[name](n)
