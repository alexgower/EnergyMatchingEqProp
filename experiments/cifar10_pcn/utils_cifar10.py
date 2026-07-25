# File: utils_cifar10.py
# CIFAR-10-specific helper functions for the cifar10_pcn experiment.
# Separated from utils.py (which contains functions copied from utils_cifar_imagenet.py).
# (Ported from experiments/mnist_pcn/utils_mnist.py — everything here is
# shape-agnostic; only the module name changed.)

import os
import torch
import torch.nn.functional as F
from absl import flags, logging

from utils import sde_euler_maruyama

# Optional: Weights & Biases (imported at module level so _generate_and_save can use it)
try:
    import wandb
except ImportError:
    wandb = None

FLAGS = flags.FLAGS


##############################################################################
# Helper: count_parameters
##############################################################################
def count_parameters(module: torch.nn.Module):
    """Count the total trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


##############################################################################
# Helper: velocity cosine similarity (PCN diagnostic)
##############################################################################
def velocity_cosine_similarity(v1, v2):
    """
    Batch-averaged cosine similarity between two velocity fields.

    Used to compare PCN velocity (from energy relaxation) against feedforward
    velocity (from backprop). At small γ, this should be close to 1.0.

    Args:
        v1, v2: velocity tensors of shape (B, C, H, W).

    Returns: scalar float in [-1, 1].
    """
    v1_flat = v1.view(v1.size(0), -1)
    v2_flat = v2.view(v2.size(0), -1)
    cos = F.cosine_similarity(v1_flat, v2_flat, dim=1)
    return cos.mean().item()




def log_pcn_step_diagnostics(model, step, log_dict, data_batch=None,
                              v_cos_every=500, wandb_module=None):
    """
    Per-step PCN diagnostics — called from the training loop every log step.

    Logs cheap diagnostics every step, expensive ones (v_cos) periodically.

    Cheap (every step, from cached _last_diagnostics):
      - pcn/max_error: max|e_k| across layers (convergence indicator)
      - pcn/max_eq_residual: max|∂E/∂e| (IFT assumption validity)

    Periodic (every v_cos_every steps):
      - pcn/v_cos: velocity cosine similarity to feedforward

    Args:
        model: raw PCNVelocityWrapper.
        step: current training step.
        log_dict: mutable dict to add metrics to (for W&B logging).
        data_batch: training batch (subset used for v_cos, can be None).
        v_cos_every: compute v_cos every N steps (0 to disable).
        wandb_module: wandb module or None.
    """
    diag = getattr(model, '_last_diagnostics', None)
    if diag is None:
        return

    pcn_msg_parts = []

    # Error-param diagnostics (only present when error_param=True)
    if "max_error" in diag:
        log_dict["pcn/max_error"] = diag["max_error"]
        log_dict["pcn/max_eq_residual"] = diag["max_eq_residual"]
        pcn_msg_parts.append(f"max|e|={diag['max_error']:.4f}")
        pcn_msg_parts.append(f"max|dE/de|={diag['max_eq_residual']:.4f}")

        # max|∂E/∂h| — directly validates the adiabatic assumption
        if "max_eq_residual_h" in diag:
            log_dict["pcn/max_eq_residual_h"] = diag["max_eq_residual_h"]
            pcn_msg_parts.append(f"max|dE/dh|={diag['max_eq_residual_h']:.4f}")

        # Per-layer error norms: e_k* ∝ γ · ∏ J_j — monitors Jacobian health
        # Growth from L→0 signals exploding Jacobians, shrinkage is healthy.
        if "per_layer_error_norm" in diag:
            e_norms = diag["per_layer_error_norm"]
            for k, en in enumerate(e_norms):
                log_dict[f"pcn/error_norm_L{k}"] = en
            norms_str = " ".join(f"L{k}={en:.4f}" for k, en in enumerate(e_norms))
            pcn_msg_parts.append(f"e_norms=[{norms_str}]")

    # Energy convergence (always available)
    if "energy" in diag and len(diag["energy"]) >= 2:
        energies = diag["energy"]
        e_drop = energies[0] - energies[-1]
        log_dict["pcn/energy_drop"] = e_drop
        pcn_msg_parts.append(f"E_drop={e_drop:.4f}")

    if pcn_msg_parts:
        logging.info(f"  PCN: {', '.join(pcn_msg_parts)}")

    # Periodic: velocity cosine similarity (requires extra forward pass)
    if (v_cos_every > 0 and step % v_cos_every == 0
            and data_batch is not None
            and hasattr(model, 'feedforward_velocity')):
        with torch.no_grad():
            diag_x = data_batch[:16]
            v_pcn = model.velocity(diag_x, t=None)
            v_ff = model.feedforward_velocity(diag_x)
        v_cos = velocity_cosine_similarity(v_pcn, v_ff)
        log_dict["pcn/v_cos"] = v_cos
        logging.info(f"  PCN v_cos={v_cos:.6f}")

##############################################################################
# EqM c(γ) schedule
##############################################################################
def eqm_c_schedule(gamma, c_type, a=0.8, b=1.0):
    """Compute the EqM c(γ) target scaling. c(1)=0 enforces equilibrium at data."""
    if c_type == "linear":
        return 1.0 - gamma
    elif c_type == "truncated":
        return torch.where(gamma <= a, torch.ones_like(gamma), (1.0 - gamma) / (1.0 - a))
    elif c_type == "piecewise":
        return torch.where(gamma <= a, b - (b - 1.0) / a * gamma, (1.0 - gamma) / (1.0 - a))
    else:
        raise ValueError(f"Unknown eqm_c_type: {c_type}")


##############################################################################
# Periodic sample generation during training
##############################################################################
def generate_and_save(model, tag, img_shape, step, savedir, device):
    """
    Generate sample images during training and save to disk + optional W&B.

    Uses convergence-based sampling for EqM (iterate until velocity is small)
    or fixed-t1 SDE integration for FM.
    """
    from torchvision.utils import save_image as _save_img, make_grid as _make_grid

    model.eval()
    with torch.no_grad():
        init = torch.randn(64, *img_shape, device=device)
        dt_gen = FLAGS.gen_dt

        use_convergence = (FLAGS.training_objective == "eqm"
                           and FLAGS.gen_converge_threshold > 0)

        if use_convergence:
            x = init.clone()
            max_gen_steps = 5000
            for gen_i in range(max_gen_steps):
                t_gen = torch.full((x.size(0),), gen_i * dt_gen, device=device)
                v = model(t_gen, x)
                x = x + v * dt_gen
                v_norm = v.view(v.size(0), -1).norm(dim=1).mean().item()
                if v_norm < FLAGS.gen_converge_threshold:
                    logging.info(f"  [{tag}] Converged at step {gen_i+1} (v_norm={v_norm:.4f})")
                    break
            else:
                logging.info(f"  [{tag}] Hit max {max_gen_steps} steps (v_norm={v_norm:.4f})")
            final = x.clamp(-1, 1)
            n_steps_used = min(gen_i + 1, max_gen_steps)
        else:
            traj = sde_euler_maruyama(model, init, t0=0.0, t1=FLAGS.gen_t1, dt=dt_gen)
            final = traj[-1].clamp(-1, 1)
            n_steps_used = int(FLAGS.gen_t1 / dt_gen)

    mode_tag = "converged" if use_convergence else f"t1_{FLAGS.gen_t1:.1f}"
    fname = f"{tag}_generated_images_step_{step}_{mode_tag}_nsteps{n_steps_used}.png"
    img_grid = final / 2.0 + 0.5
    _save_img(img_grid, os.path.join(savedir, fname), nrow=8)

    # Log to W&B if active
    if wandb is not None and wandb.run is not None:
        grid_tensor = _make_grid(img_grid, nrow=8)
        wandb.log({
            f"generated_samples/{tag}": wandb.Image(
                grid_tensor.permute(1, 2, 0).cpu().numpy(),
                caption=f"{tag} step={step} {mode_tag} nsteps={n_steps_used}"
            )
        }, step=step)

    model.train()
