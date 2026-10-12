"""
DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020).

Notation (same as the paper):
    beta_t        : how much noise we add at step t           (small, e.g. 1e-4 .. 0.02)
    alpha_t       = 1 - beta_t
    alpha_bar_t   = alpha_1 * alpha_2 * ... * alpha_t        (how much signal is left at step t)

Forward process (adding noise) - we can jump straight to any t:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,     eps ~ N(0, I)

Training: the network eps_theta(x_t, t) learns to predict eps. Loss = MSE(eps, eps_theta).

Sampling (reverse process), one step from x_t to x_{t-1}:
    mean    = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) )
    x_{t-1} = mean + sigma_t * z,      z ~ N(0, I),  sigma_t^2 = beta_t   (no noise at the last step)
"""
import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        # Noise schedule. Index 0 here means paper's t=1.
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _gather(self, values, t, x):
        """Pick values[t] for each sample and reshape so it broadcasts with x."""
        out = values[t]
        return out.view(-1, *([1] * (x.dim() - 1)))

    # ---------- forward process ----------
    def q_sample(self, x0, t, noise=None):
        """Noise clean data x0 to step t in one shot."""
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self._gather(self.alpha_bars, t, x0)
        return a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise

    # ---------- small helpers (same names in library/flow.py, so plots work for both) ----------
    def model_time(self, t):
        """What the network gets as 'time' for integer step t."""
        return t

    def target(self, x0, noise):
        """What the network should output: the noise."""
        return noise

    @torch.no_grad()
    def predict_x0(self, model, x_t, t_int):
        """One-step guess of the clean image from the predicted noise."""
        a = self.alpha_bars[t_int]
        eps = model(x_t, torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long))
        return (x_t - (1 - a).sqrt() * eps) / a.sqrt()

    # ---------- training ----------
    def loss(self, model, x0, y=None, p_uncond=0.0):
        """Pick random t, noise x0, ask the model to predict the noise.
        y: class labels (Stage 2.2). With probability p_uncond a label is replaced by the
           null class (= model.num_classes), so the same model also learns the unconditional case."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if y is None:
            noise_pred = model(x_t, t)
        else:
            drop = torch.rand(y.shape[0], device=y.device) < p_uncond
            y = torch.where(drop, torch.full_like(y, model.num_classes), y)
            noise_pred = model(x_t, t, y)
        return F.mse_loss(noise_pred, noise)

    # ---------- reverse process ----------
    @torch.no_grad()
    def p_sample(self, model, x_t, t_int):
        """One denoising step: x_t -> x_{t-1}. t_int is a python int."""
        t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
        beta = self.betas[t_int]
        alpha = self.alphas[t_int]
        alpha_bar = self.alpha_bars[t_int]

        eps = model(x_t, t)
        # 1) denoise: remove the predicted noise -> mean of x_{t-1}
        mean = (x_t - beta / (1 - alpha_bar).sqrt() * eps) / alpha.sqrt()
        if t_int == 0:
            return mean
        # 2) add a bit of fresh noise back (this is what makes DDPM stochastic)
        return mean + beta.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model, shape, return_trajectory=False, x_start=None, t_start=None):
        """Start from pure noise and denoise T times.
        If x_start/t_start are given, start from that x_t instead and denoise t_start+1 times
        (noise a real sample, then denoise it back).
        If return_trajectory: also returns a tensor [n_steps+1, *shape] with every x_t."""
        if x_start is None:
            x, t_start = torch.randn(shape, device=self.device), self.T - 1
        else:
            x = x_start.to(self.device)
        traj = [x.cpu()] if return_trajectory else None
        for t in reversed(range(t_start + 1)):
            x = self.p_sample(model, x, t)
            if return_trajectory:
                traj.append(x.cpu())
        if return_trajectory:
            return x, torch.stack(traj)
        return x
