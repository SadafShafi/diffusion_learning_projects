"""
FID (Frechet Inception Distance). Lower = better.

1. Push real and generated images through an Inception network, take the 2048-d features.
2. Fit a Gaussian (mean mu, covariance sigma) to each set of features.
3. FID = distance between the two Gaussians:
       |mu_r - mu_g|^2 + Tr(sigma_r + sigma_g - 2 sqrt(sigma_r sigma_g))

Notes: FID depends on the number of samples (fewer samples -> higher FID), so only compare
runs that used the same n. Papers use 50k samples. Inception was trained on photos, so FID on
MNIST is only useful to compare your own runs.
"""
import os

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from library.data_images import DATA_DIR


class FID:
    def __init__(self, device):
        self.device = device
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    @torch.no_grad()
    def features(self, x):
        """x: images in [-1, 1], shape (B, C, H, W). Returns (B, 2048) numpy array."""
        x = ((x.to(self.device) + 1) / 2).clamp(0, 1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)          # grayscale -> RGB
        return self.net(x)[0].squeeze(-1).squeeze(-1).cpu().numpy()   # net resizes to 299x299 itself

    @staticmethod
    def stats(feats):
        return feats.mean(axis=0), np.cov(feats, rowvar=False)

    def real_stats(self, dataset_name, dataset, n, batch_size=500):
        """Stats of n real training images. Cached in data/ because they never change."""
        path = os.path.join(DATA_DIR, f"fid_stats_{dataset_name}_{n}.npz")
        if os.path.exists(path):
            f = np.load(path)
            return f["mu"], f["sigma"]
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        feats = []
        for x, _ in loader:
            feats.append(self.features(x))
            if sum(len(f) for f in feats) >= n:
                break
        mu, sigma = self.stats(np.concatenate(feats)[:n])
        np.savez(path, mu=mu, sigma=sigma)
        return mu, sigma

    @staticmethod
    def score(mu1, sigma1, mu2, sigma2):
        return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))
