# File: utils.py
# Utility functions for cifar10_pcn experiments.
# Extracted from utils_cifar_imagenet.py — only the functions actually used here.

import os
import time
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from absl import flags, logging

FLAGS = flags.FLAGS

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


##############################################################################
# Directory creation
##############################################################################
def create_timestamped_dir(base_output_dir, model_name):
    """
    Creates a directory named like:
       base_output_dir / [model_name]_YYYYMMDD_HH
    If that directory exists, it appends _verX.
    Uses atomic try/except to avoid race conditions with concurrent processes.
    """
    ts = time.strftime('%Y%m%d_%H')  # e.g. 20250214_15
    base_name = f"{model_name}_{ts}"
    path_candidate = os.path.join(base_output_dir, base_name)

    # Try creating the base path first, then ver1, ver2, etc.
    # os.makedirs with exist_ok=False is atomic — no race condition.
    ver_idx = 0
    while True:
        try:
            os.makedirs(path_candidate, exist_ok=False)
            return path_candidate
        except FileExistsError:
            ver_idx += 1
            path_candidate = os.path.join(
                base_output_dir, f"{base_name}_ver{ver_idx}"
            )


##############################################################################
# Training helpers
##############################################################################
def flow_weight(t, cutoff=0.8):
    """
    Flow weighting function (FM training only, not used by EqM):
    - w_flow = 1 for t < cutoff
    - linearly from 1 down to 0 as t goes from cutoff..1
    - 0 for t >= 1

    Purpose: In Phase 2 of Energy Matching, contrastive divergence (CD) refines
    the energy landscape near data (t ≈ 1). This weight ramps down the FM loss
    near t=1 so CD can take over, avoiding conflicting gradients.
    EqM uses uniform weight=1 everywhere (no phase separation needed).
    """
    w = torch.ones_like(t)
    decay_region = (t >= cutoff) & (t < 1.0)
    w[decay_region] = 1.0 - (t[decay_region] - cutoff) / (1.0 - cutoff)
    w[t >= 1.0] = 0.0
    return w


def ema(source, target, decay):
    """
    Exponential Moving Average update.
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for batch in iter(dataloader):
            yield batch[0] if isinstance(batch, (list, tuple)) else batch


##############################################################################
# Visualization
##############################################################################
def save_pos_neg_grids(
    pos_samples: torch.Tensor,
    neg_samples: torch.Tensor,
    savedir: str,
    step: int
):
    """
    Create a single figure with 2 subplots side by side:
      - Left: a grid (8x8) of 'positive' real samples
      - Right: a grid (8x8) of 'negative' MCMC samples
    Then save it as pos_neg_grid_step_<step>.png
    """
    # 1) Take up to 64 from each set
    pos_samples = pos_samples[:64].detach().cpu()
    neg_samples = neg_samples[:64].detach().cpu()

    # 2) Scale from [-1,1] => [0,1] if needed
    pos_samples = (pos_samples + 1) / 2
    neg_samples = (neg_samples + 1) / 2

    # 3) Create torchvision grids
    grid_pos = make_grid(pos_samples, nrow=8)  # shape: [C,H,W]
    grid_neg = make_grid(neg_samples, nrow=8)

    # 4) Plot them side-by-side with matplotlib
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    for ax, grid, title in zip(
        axes,
        [grid_pos, grid_neg],
        ["Positive (Real Data)", "Negative (MCMC Chain)"]
    ):
        # permute C,H,W => H,W,C, then convert to numpy
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    outpath = os.path.join(savedir, f"pos_neg_grid_step_{step}.png")
    plt.savefig(outpath, dpi=100)
    plt.close(fig)

    print(f"[save_pos_neg_grids] Saved {outpath}")


##############################################################################
# Time-dependent noise schedule for the SDE
##############################################################################
def plot_epsilon(t, at_data=False):
    """
    A piecewise function for epsilon(t) in the SDE:
      - 0 for t < FLAGS.time_cutoff
      - linearly from 0..epsilon_max as t goes from FLAGS.time_cutoff..1.0
      - constant epsilon_max for t >= 1.0

    If at_data is True, always return epsilon_max (ignore time).
    """
    eps_max = FLAGS.epsilon_max
    cutoff = FLAGS.time_cutoff

    # If at_data is True, always return eps_max
    if at_data:
        return eps_max

    if t < cutoff:
        return 0.0
    elif t < 1.0:
        frac = (t - cutoff) / (1.0 - cutoff)  # goes from 0..1
        return frac * eps_max
    else:
        return eps_max


##############################################################################
# SDE integration (used during training for periodic sample generation)
##############################################################################
def sde_euler_maruyama(model, x0, t0, t1, dt=0.01, steps_to_save=None):
    """
    Euler–Maruyama SDE integration from t=t0 to t1 with step dt.

    Used during training (not generation) to periodically visualize sample
    quality. Follows the learned velocity field v(x) plus optional Langevin
    noise scaled by plot_epsilon(t).

    When epsilon_max=0 (Phase 1, no CD), noise vanishes and this reduces to
    deterministic Euler ODE integration. For standalone generation from a
    checkpoint, use generate_from_checkpoint.py instead (supports Heun).

    Can optionally save intermediate trajectory snapshots via steps_to_save.
    Clamps output to [-1, 1] once at the end.
    """
    model.eval()
    times = torch.arange(t0, t1+1e-6, dt, device=device)

    x = x0.clone().to(device)
    trajectory = []

    for step_idx, t_val in enumerate(times):
        # Optionally store a copy BEFORE the update
        if steps_to_save is None or (step_idx in steps_to_save):
            trajectory.append(x.clone().detach())

        with torch.no_grad():
            # 1) Evaluate drift v(t, x)
            v = model(t_val.unsqueeze(0), x)

            # 2) Time-dependent noise scale e(t)
            e = plot_epsilon(float(t_val))
            e_tensor = torch.tensor(e, device=x.device, dtype=x.dtype)
            dt_tensor = torch.tensor(dt, device=x.device, dtype=x.dtype)
            
            # 3) Euler–Maruyama step
            noise = torch.randn_like(x)
            sigma = torch.sqrt(2.0 * e_tensor * dt_tensor)
            x = x + v * dt_tensor + sigma * noise

    # After the final step, clamp once at the very end
    x = x.clamp(-1, 1)

    # Append final state
    if steps_to_save is None:
        trajectory.append(x.clone().detach())
    else:
        last_step_idx = len(times)  # "one past" the last loop index
        if last_step_idx in steps_to_save:
            trajectory[-1] = x.clone().detach()
        else:
            trajectory.append(x.clone().detach())

    # Return shape: (num_snapshots, batch, channels, height, width)
    return torch.stack(trajectory, dim=0)


##############################################################################
# Gibbs / MALA sampling (Phase 2 CD only — requires lambda_cd > 0)
##############################################################################
def gibbs_sampling_n_steps_fast(x_init, model, t, n_steps, dt, epsilon):
    """
    Fixed-time Langevin sampling: run n_steps of MALA at a SINGLE time t
    with constant noise epsilon.

      x_{k+1} = x_k - dt·∇_x V(x_k, t) + √(2·dt·ε)·N(0,I)

    Unlike gibbs_sampling_time_sweep (which sweeps t from 0→1 with
    time-dependent noise), this keeps t and epsilon constant throughout.
    Simpler but less refined — useful for quick Langevin diagnostics.

    NOTE: Currently unused in cifar10_pcn. Kept for potential future use.
    """
    samples = x_init.clone().detach().to(device)
    noise_std = (2.0 * dt * epsilon) ** 0.5

    for _ in range(n_steps):
        samples.requires_grad_(True)
        V = model.potential(samples, t)
        grad_V = torch.autograd.grad(
            outputs=V,
            inputs=samples,
            grad_outputs=torch.ones_like(V),
            create_graph=False
        )[0]
        with torch.no_grad():
            samples = samples - dt * grad_V
            samples += noise_std * torch.randn_like(samples)
            # Removed the per-step clamp; clamp once at the end.

    samples = samples.clamp(-1.0, 1.0)
    return samples.detach()


def gibbs_sampling_time_sweep(
    x_init: torch.Tensor,
    model,
    at_data_mask: torch.Tensor,
    n_steps: int = 150,
    dt: float = 0.01
):
    """
    Time-sweeping Langevin sampling for CD negative chain generation.

    Runs n_steps of MALA while sweeping t from 0 to n_steps*dt, using
    time-dependent noise epsilon(t) from plot_epsilon():

      x_{k+1} = x_k - dt·∇_x V(x_k, t_k) + √(2·dt·ε(t_k))·N(0,I)

    This is the main CD negative sampler used in forward_all() during
    Phase 2 training (lambda_cd > 0). It generates negative samples
    for the contrastive divergence loss and for pos/neg diagnostic grids.

    at_data_mask controls per-sample noise scheduling:
      - at_data_mask[i] = True  → always uses epsilon=epsilon_max (data-initialized)
      - at_data_mask[i] = False → follows piecewise schedule from plot_epsilon()

    Returns the final samples (clamped to [-1, 1]).
    """
    device = x_init.device
    samples = x_init.clone().detach().to(device)

    for step in range(n_steps):
        t_val = step * dt  # e.g., 0.0, 0.01, 0.02, ...
        samples.requires_grad_(True)

        # Time tensor
        t_tensor = torch.full(
            (samples.size(0),),
            t_val,
            device=device,
            dtype=samples.dtype
        )

        # Potential and gradient
        V = model.potential(samples, t_tensor)
        grad_V = torch.autograd.grad(
            V,
            samples,
            grad_outputs=torch.ones_like(V),
            create_graph=False
        )[0]

        # MALA update
        with torch.no_grad():
            samples = samples - dt * grad_V

            # For each sample in the batch, compute epsilon(t)
            # and thus the noise_std for that sample
            e_vals = []
            for i in range(samples.size(0)):
                e_vals.append(plot_epsilon(t_val, at_data=bool(at_data_mask[i].item())))
            e_vals = torch.tensor(e_vals, device=device, dtype=samples.dtype)

            noise_std = (2.0 * dt * e_vals).sqrt()
            noise = torch.randn_like(samples)
            samples += noise * noise_std.view(-1, 1, 1, 1)

    # Final clamp to valid image range (not for each step though see)
    samples = samples.clamp(-1.0, 1.0)
    return samples.detach()
