"""
Flow matching / rectified flow (Lipman et al. 2022; Liu et al. 2022).   (Stage 2, project 5)

Draw a STRAIGHT LINE between a data point x0 and a noise sample z:

    x_s = (1 - s) * x0 + s * z          s in [0, 1]   (s = 0: data, s = 1: pure noise)

Moving along this line has a constant velocity:   d x_s / d s = z - x0

Training:  the network learns that velocity:   loss = MSE( v_theta(x_s, s),  z - x0 )
Sampling:  start at pure noise (s = 1) and walk back to s = 0 with Euler steps:
               x_{s - ds} = x_s - ds * v_theta(x_s, s)

Compare with DDPM: no betas, no alpha_bars, no posterior variance. Just a line and its slope.

To reuse ALL Stage-1 plots, this class has the same interface as library/ddpm.DDPM:
an integer t = 0 .. T-1 is a sampling step and means noise level s = (t + 1) / T.
Here T = number of Euler steps used for sampling (100 is plenty; DDPM needs 1000).
"""
import torch
import torch.nn.functional as F


class FlowMatching:
    TIME_SCALE = 1000      # our networks' time embedding was designed for t in [0, 1000]

    def __init__(self, T=100, device="cpu"):
        self.T = T
        self.device = device

    def _s(self, t, x):
        """integer step t -> noise level s = (t + 1) / T, shaped to broadcast with x."""
        s = (torch.as_tensor(t, device=x.device).float() + 1) / self.T
        return s.view(-1, *([1] * (x.dim() - 1))) if s.dim() > 0 else s

    def model_time(self, t):
        """What the network gets as 'time' for integer step t."""
        return (t.float() + 1) / self.T * self.TIME_SCALE

    # ---------- forward process: a point on the straight line ----------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s = self._s(t, x0)
        return (1 - s) * x0 + s * noise

    def target(self, x0, noise):
        """What the network should output: the velocity z - x0."""
        return noise - x0

    # ---------- training ----------
    def loss(self, model, x0):
        s = torch.rand(x0.shape[0], device=x0.device)          # continuous s in [0, 1]
        z = torch.randn_like(x0)
        s_ = s.view(-1, *([1] * (x0.dim() - 1)))
        x_s = (1 - s_) * x0 + s_ * z
        v_pred = model(x_s, s * self.TIME_SCALE)
        return F.mse_loss(v_pred, z - x0)

    # ---------- sampling ----------
    @torch.no_grad()
    def predict_x0(self, model, x_t, t_int):
        """One-step guess of the data point: follow the velocity all the way to s = 0."""
        s = (t_int + 1) / self.T
        v = model(x_t, torch.full((x_t.shape[0],), s * self.TIME_SCALE, device=x_t.device))
        return x_t - s * v

    @torch.no_grad()
    def p_sample(self, model, x, t_int):
        """One Euler step from s = (t+1)/T to s = t/T."""
        s, s_next = (t_int + 1) / self.T, t_int / self.T
        v = model(x, torch.full((x.shape[0],), s * self.TIME_SCALE, device=x.device))
        return x + (s_next - s) * v

    @torch.no_grad()
    def sample(self, model, shape, return_trajectory=False, x_start=None, t_start=None):
        """Same as DDPM.sample: from pure noise (or from x_start at step t_start) to data."""
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
