"""
Samplers for a trained noise-prediction model eps_theta(x_t, t).   (Stage 2, project 3)

Key idea: TRAINING only teaches the model to predict noise. HOW we walk from noise to an image
is a separate choice. All samplers here use the same trained model.

We pick a subsequence of n timesteps out of the T=1000 training steps, e.g. n=10: 999, 888, ..., 0.

1) DDIM (Song et al., 2020). From x_t:
       x0_hat = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)              (guess of the clean image)
       x_prev = sqrt(a_prev) * x0_hat                                 (move the guess to the next noise level)
              + sqrt(1 - a_prev - sigma^2) * eps                      (re-use the predicted noise direction)
              + sigma * z                                             (fresh noise, z ~ N(0, I))
   with sigma = eta * sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).
   eta = 0 -> deterministic.   eta = 1 -> stochastic, like DDPM.
   (a_t = alpha_bar_t)

2) Probability-flow ODE. Rescale: x~ = x_t / sqrt(a_t), and use the noise level
       sigma(t) = sqrt((1 - a_t) / a_t)       (0 = clean, ~157 at t=999)
   Then the whole sampling process is the ODE      d x~ / d sigma = eps_theta(x_t, t)
   which we solve from sigma_max down to 0 with a standard ODE solver:
     Euler: one model call per step.        (Mathematically IDENTICAL to DDIM with eta=0!)
     Heun:  Euler step, evaluate the slope again at the end point, use the average slope.
            2 model calls per step (1 for the last step), but 2nd-order accurate -> fewer steps needed.
   (This sigma-form is the EDM view of Karras et al., 2022.)
"""
import torch


def timesteps(T, n, t_start=None):
    """n timesteps from t_start (default T-1) down to 0, evenly spaced."""
    t_start = T - 1 if t_start is None else t_start
    return torch.linspace(t_start, 0, n).round().long().tolist()


def _t(x, t):
    return torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)


@torch.no_grad()
def ddim(model, ddpm, z, n_steps, eta=0.0, t_start=None, ts=None):
    """z: starting noise x_T. Returns the generated images.
    t_start: start from a partly noised image z = x_{t_start} instead of pure noise (img2img).
    ts: optional explicit list of timesteps (overrides n_steps / t_start), e.g. to match another library."""
    ab = ddpm.alpha_bars
    ts = timesteps(ddpm.T, n_steps, t_start) if ts is None else [int(t) for t in ts]
    x = z
    for i, t in enumerate(ts):
        a_t = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
        eps = model(x, _t(x, t))
        x0_hat = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
        x = a_prev.sqrt() * x0_hat + (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if eta > 0 and i + 1 < len(ts):
            x = x + sigma * torch.randn_like(x)
    return x


def _sigma(ddpm, t):
    a = ddpm.alpha_bars[t]
    return ((1 - a) / a).sqrt()


@torch.no_grad()
def ode(model, ddpm, z, n_steps, method="euler"):
    """Solve d x~ / d sigma = eps from sigma(T-1) to 0 with Euler or Heun."""
    ab = ddpm.alpha_bars
    ts = timesteps(ddpm.T, n_steps)
    sigmas = [_sigma(ddpm, t) for t in ts] + [torch.tensor(0.0, device=z.device)]

    def slope(x_tilde, i):                       # eps_theta at noise level sigmas[i] (= timestep ts[i])
        t = ts[i]
        return model(x_tilde * ab[t].sqrt(), _t(x_tilde, t))   # the model wants x_t = sqrt(a_t) * x~

    x = z / ab[ts[0]].sqrt()                     # x~ at the start (same noise z as DDIM uses)
    for i in range(len(ts)):
        h = sigmas[i + 1] - sigmas[i]            # step size (negative: we go DOWN in noise)
        d = slope(x, i)
        x_next = x + h * d                       # Euler step
        if method == "heun" and i + 1 < len(ts): # Heun: correct with the slope at the end point
            d2 = slope(x_next, i + 1)
            x_next = x + h * (d + d2) / 2
        x = x_next
    return x                                     # sigma = 0 -> x~ = x_0


# name -> (function(model, ddpm, z, n_steps), model calls (NFE) for n_steps)
SAMPLERS = {
    "ddim_eta1": (lambda m, d, z, n: ddim(m, d, z, n, eta=1.0), lambda n: n),
    "ddim":      (lambda m, d, z, n: ddim(m, d, z, n, eta=0.0), lambda n: n),
    "euler":     (lambda m, d, z, n: ode(m, d, z, n, "euler"), lambda n: n),
    "heun":      (lambda m, d, z, n: ode(m, d, z, n, "heun"),  lambda n: 2 * n - 1),
}
