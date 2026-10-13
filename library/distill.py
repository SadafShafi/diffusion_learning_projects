"""
Progressive distillation (Salimans & Ho, 2022).   (Stage 4, project 9)

Goal: the SAME network, but sampling in N = 4, 2, 1 steps instead of hundreds.

One round:
    teacher = model that samples well with 2N DDIM steps
    student = copy of the teacher
    train the student so that ONE of its DDIM steps lands where TWO teacher steps land
    -> student samples well with N steps. It becomes the next teacher. Repeat: 256 -> 128 -> ... -> 1.

Time. We use a continuous time u in [0, 1000]: u = 0 is the clean image, u = 1000 is pure noise
(= training step t = 999). alpha_bar(u) is interpolated from the DDPM schedule, so we can halve the
number of steps as often as we like. The network gets t = u - 1 as its time input.
    z_u = alpha * x0 + sigma * eps,       alpha = sqrt(alpha_bar(u)),  sigma = sqrt(1 - alpha_bar(u))

v-prediction. Our Stage 1.2 model predicts the noise eps. Then x0 = (z - sigma * eps) / alpha, and at
u = 1000 alpha is 0.006, so tiny eps errors become 150x bigger in x0. That is fine with many small steps,
but a 1-step student jumps straight to x0. So (like the paper) the network predicts the "velocity"
    v = alpha * eps - sigma * x0           ->   x0 = alpha * z - sigma * v,   eps = sigma * z + alpha * v
which is well behaved at every noise level. Round 0 fine-tunes our eps-model into a v-model.
"""
import torch
import torch.nn.functional as F

U_MAX = 1000.0


class Schedule:
    """alpha(u), sigma(u) for continuous u in [0, 1000], from a trained DDPM's alpha_bars."""

    def __init__(self, ddpm):
        ab = torch.cat([torch.ones(1, device=ddpm.device), ddpm.alpha_bars])   # index u: 0 = clean, 1000 = t 999
        self.log_ab = ab.log()

    def alpha_sigma(self, u):
        """u: tensor of times. Returns alpha, sigma with shape (B, 1, 1, 1). log(alpha_bar) is interpolated."""
        u = u.clamp(0, U_MAX)
        lo = u.floor().long().clamp(max=int(U_MAX) - 1)
        w = u - lo
        ab = ((1 - w) * self.log_ab[lo] + w * self.log_ab[lo + 1]).exp().view(-1, 1, 1, 1)
        return ab.sqrt(), (1 - ab).sqrt()


def net_time(u):
    return u - 1                    # u = 1000 <-> training step t = 999


def grid(n):
    """n DDIM steps: n+1 times from 1000 down to 0, evenly spaced."""
    return [U_MAX * k / n for k in range(n, -1, -1)]


def from_u(n, u0):
    """Times for an n-step model when we start at a (noised) image at time u0: u0, then its grid below u0."""
    return [u0] + [u for u in grid(n) if u < u0 - 1e-6]


def _full(x, u):
    return torch.full((x.shape[0],), float(u), device=x.device)


def ddim_step(model, sched, z, u, u_next):
    """One deterministic DDIM step from time u to u_next with a v-prediction model."""
    a, s = sched.alpha_sigma(u)
    a_n, s_n = sched.alpha_sigma(u_next)
    v = model(z, net_time(u))
    x0 = a * z - s * v
    eps = s * z + a * v
    return a_n * x0 + s_n * eps


@torch.no_grad()
def ddim_v(model, sched, z, us):
    """Sample along the times us (descending, last one = 0)."""
    for u, u_next in zip(us[:-1], us[1:]):
        z = ddim_step(model, sched, z, _full(z, u), _full(z, u_next))
    return z


# ---------------------------------------------------------------------
# Round 0: turn the eps-model into a v-model (normal diffusion training, new target)
# ---------------------------------------------------------------------
def v_loss(model, sched, x0):
    u = torch.randint(1, int(U_MAX) + 1, (x0.shape[0],), device=x0.device).float()   # = t + 1
    a, s = sched.alpha_sigma(u)
    eps = torch.randn_like(x0)
    z = a * x0 + s * eps
    return F.mse_loss(model(z, net_time(u)), a * eps - s * x0)


# ---------------------------------------------------------------------
# One distillation round: 2 teacher steps -> 1 student step
# ---------------------------------------------------------------------
def distill_loss(student, teacher, sched, x0, n):
    """n = number of steps the STUDENT will use. The teacher uses 2n."""
    B = x0.shape[0]
    k = torch.randint(1, n + 1, (B,), device=x0.device).float()
    u = U_MAX * k / n                       # student step:  u -> u - 1000/n
    u_mid = u - U_MAX / (2 * n)             # teacher: u -> u_mid -> u_next
    u_next = u - U_MAX / n
    a, s = sched.alpha_sigma(u)
    z = a * x0 + s * torch.randn_like(x0)

    with torch.no_grad():
        z_next = ddim_step(teacher, sched, ddim_step(teacher, sched, z, u, u_mid), u_mid, u_next)
        # Which x0 must the student predict so that ONE DDIM step z -> z_next is exact?
        #   z_next = a_n * x0 + s_n * (z - a * x0) / s     ->  solve for x0
        a_n, s_n = sched.alpha_sigma(u_next)
        x0_target = (z_next - (s_n / s) * z) / (a_n - (s_n / s) * a)

    x0_student = a * z - s * student(z, net_time(u))
    weight = (a ** 2 / s ** 2).clamp(min=1)                  # "truncated SNR" weighting from the paper
    return (weight * (x0_student - x0_target) ** 2).mean()
