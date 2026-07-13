#!/usr/bin/env python3
"""
Generate images from a saved checkpoint with configurable generation parameters.

Supports two generation modes:
  - Fixed-time integration: ODE from t=0 to t=t1 (sweep dt and t1).
    Extending t1 > 1.0 compensates for the systematic v_mag undershoot
    caused by MSE-optimal magnitude reduction.
  - Convergence-based: Iterate until velocity norm < threshold (for EqM models).

Integrators:
  - euler: First-order Euler step. Fast (1 model eval/step).
  - heun:  Second-order Heun's method. More accurate (2 model evals/step).

Usage examples:
  # Default generation (fixed t1=1.0, dt=0.01, euler)
  python generate_from_checkpoint.py \
      --ckpt=results_mnist_pcn/EM_mnist_pcn_.../checkpoint_50000.pt

  # Heun integrator with sweep
  python generate_from_checkpoint.py \
      --ckpt=path/to/checkpoint.pt \
      --integrator=heun \
      --gen_t1_sweep=1.0,1.1 --gen_dt_sweep=0.01,0.005

  # Convergence-based (EqM) with Heun
  python generate_from_checkpoint.py \
      --ckpt=path/to/checkpoint.pt \
      --integrator=heun --gen_converge_threshold=1.0 --use_ema
"""

import os
import sys
import torch
import math
from datetime import datetime

from absl import app, flags, logging
import config_multigpu as config

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

config.define_flags()
FLAGS = flags.FLAGS

# Generation-specific flags (supplements shared config flags)
flags.DEFINE_string("ckpt", "", "Path to checkpoint file (required)")
flags.DEFINE_string("gen_output_dir", "", "Output directory (default: same as checkpoint's training run folder)")
flags.DEFINE_string("gen_t1_sweep", "1.0",
                    "Comma-separated integration endpoints (e.g. '1.0,1.5,2.0'). "
                    "Ignored when gen_converge_threshold > 0 (convergence finds its own endpoint).")
flags.DEFINE_string("gen_dt_sweep", "0.01",
                    "Comma-separated dt values (e.g. '0.01,0.005'). Used in both fixed-t1 and convergence modes.")
flags.DEFINE_integer("n_samples", 64, "Number of samples to generate per config")
flags.DEFINE_string("integrator", "heun",
                    "ODE integrator: 'euler' (1st order, 1 eval/step) or "
                    "'heun' (2nd order, 2 evals/step, more accurate). Default: heun.")
flags.DEFINE_bool("use_ema", False, "Also generate with EMA model weights")
flags.DEFINE_bool("use_normal", True, "Use non-EMA model weights (default)")

# Import models
from network_cnn import EBCNNModelWrapper
from network_transformer_vit import EBViTModelWrapper
from network_pcn import PCNVelocityWrapper


def _step(model, x, t_val, dt, integrator):
    """Single integration step: Euler or Heun."""
    v1 = model(t_val, x)
    if integrator == "heun":
        x_pred = x + v1 * dt
        v2 = model(t_val, x_pred)  # model is time-independent so same t is fine
        return x + dt * (v1 + v2) / 2, v1
    else:  # euler
        return x + v1 * dt, v1


def ode_integrate(model, x0, t0, t1, dt=0.01, integrator="euler"):
    """
    ODE integration from t0 to t1 (deterministic, no noise).
    Supports 'euler' (1st order) and 'heun' (2nd order) integrators.
    Returns only the final sample for efficiency.
    """
    device = x0.device
    times = torch.arange(t0, t1 + 1e-6, dt, device=device)
    x = x0.clone()

    with torch.no_grad():
        for t_val in times:
            x, _ = _step(model, x, t_val.unsqueeze(0), dt, integrator)

    return x.clamp(-1, 1)


def generate_converged(model, x0, dt=0.01, threshold=0.5, max_steps=5000,
                       integrator="euler"):
    """
    Adaptive compute generation: iterate until mean velocity norm < threshold.
    Returns (final_samples, n_steps_used).
    """
    x = x0.clone()
    with torch.no_grad():
        for i in range(max_steps):
            t_val = torch.full((x.size(0),), i * dt, device=x.device)
            x, v = _step(model, x, t_val, dt, integrator)
            v_norm = v.view(v.size(0), -1).norm(dim=1).mean().item()
            if v_norm < threshold:
                logging.info(f"  Converged at step {i+1} (v_norm={v_norm:.4f})")
                return x.clamp(-1, 1), i + 1
    logging.info(f"  Hit max {max_steps} steps (v_norm={v_norm:.4f})")
    return x.clamp(-1, 1), max_steps


def build_model(device):
    """Build model from FLAGS (same logic as train script)."""
    img_shape = (1, 28, 28)

    if FLAGS.model_type == "pcn":
        model = PCNVelocityWrapper(
            gamma=FLAGS.pcn_gamma,
            T_free=FLAGS.T_free,
            dt_relax=FLAGS.pcn_dt,
            async_mode=FLAGS.pcn_async,
            init_mode=FLAGS.pcn_init_mode,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None,
            pool_type=FLAGS.pool_type,
            activation=FLAGS.activation,
        ).to(device)
    elif FLAGS.model_type in ("historical", "vgg5"):
        version = "vgg5" if FLAGS.model_type == "vgg5" else "historical"
        model = EBCNNModelWrapper(
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None,
            version=version,
            pool_type=FLAGS.pool_type,
        ).to(device)
    else:
        # Default: UNet + ViT head (paper architecture)
        ch_mult = config.parse_channel_mult(FLAGS)
        model = EBViTModelWrapper(
            dim=img_shape,
            num_channels=FLAGS.num_channels,
            num_res_blocks=FLAGS.num_res_blocks,
            channel_mult=ch_mult,
            attention_resolutions=FLAGS.attention_resolutions,
            num_heads=FLAGS.num_heads,
            num_head_channels=FLAGS.num_head_channels,
            dropout=FLAGS.dropout,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp,
            patch_size=7,
            embed_dim=FLAGS.embed_dim,
            transformer_nheads=FLAGS.transformer_nheads,
            transformer_nlayers=FLAGS.transformer_nlayers,
        ).to(device)

    return model, img_shape


def load_checkpoint(model, ckpt_path, device, use_ema=True):
    """Load checkpoint and return the model with weights loaded."""
    ckpt = torch.load(ckpt_path, map_location=device)

    key = "ema_model" if use_ema else "net_model"
    state_dict = ckpt[key]
    # Strip 'module.' prefix if saved from DDP
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    step = ckpt.get("step", "unknown")
    logging.info(f"Loaded {key} from {ckpt_path} (step={step})")
    return model, step


def generate_grid(model, img_shape, device, n_samples, t1, dt, integrator="euler"):
    """Generate a grid of samples with given parameters."""
    model.eval()

    init = torch.randn(n_samples, *img_shape, device=device)
    final = ode_integrate(model, init, t0=0.0, t1=t1, dt=dt, integrator=integrator)
    final_01 = final / 2.0 + 0.5  # [-1,1] -> [0,1]

    return final_01


def main(_):
    if not FLAGS.ckpt:
        raise ValueError("--ckpt is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Parse sweep parameters
    t1_values = [float(x.strip()) for x in FLAGS.gen_t1_sweep.split(",")]
    dt_values = [float(x.strip()) for x in FLAGS.gen_dt_sweep.split(",")]

    # Build model
    model, img_shape = build_model(device)

    # Determine which weight variants to use
    weight_variants = []
    if FLAGS.use_ema:
        weight_variants.append(("ema", True))
    if FLAGS.use_normal:
        weight_variants.append(("normal", False))
    if not weight_variants:
        weight_variants.append(("normal", True))  # default

    # Output directory: same as checkpoint's training run folder (or override)
    if FLAGS.gen_output_dir:
        out_dir = FLAGS.gen_output_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(FLAGS.ckpt))
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Saving generated images to: {out_dir}")

    from torchvision.utils import save_image

    total_configs = len(weight_variants) * len(t1_values) * len(dt_values)
    config_idx = 0

    for weight_tag, use_ema in weight_variants:
        # Load weights
        model, step = load_checkpoint(model, FLAGS.ckpt, device, use_ema=use_ema)

        if FLAGS.gen_converge_threshold > 0:
            # Convergence mode: sweep dt (affects step size / trajectory quality)
            for dt in dt_values:
                config_idx += 1
                logging.info(
                    f"[{config_idx}] {weight_tag} | convergence mode | "
                    f"dt={dt} | threshold={FLAGS.gen_converge_threshold} | n={FLAGS.n_samples}"
                )
                init = torch.randn(FLAGS.n_samples, *img_shape, device=device)
                samples, n_steps = generate_converged(
                    model, init, dt=dt,
                    threshold=FLAGS.gen_converge_threshold,
                    integrator=FLAGS.integrator
                )
                grid = samples / 2.0 + 0.5
                nrow = int(math.sqrt(FLAGS.n_samples))
                fname = (
                    f"GEN_{weight_tag}_{FLAGS.integrator}_step{step}"
                    f"_converged_dt{dt}_thresh{FLAGS.gen_converge_threshold}"
                    f"_nsteps{n_steps}.png"
                )
                fpath = os.path.join(out_dir, fname)
                save_image(grid, fpath, nrow=nrow)
                logging.info(f"  Saved {fpath}")
        else:
            # Standard sweep over t1 × dt
            for t1 in t1_values:
                for dt in dt_values:
                    config_idx += 1
                    n_steps = int(round(t1 / dt))

                    logging.info(
                        f"[{config_idx}/{total_configs}] "
                        f"{weight_tag} | t1={t1:.2f} | dt={dt} | "
                        f"steps={n_steps} | n={FLAGS.n_samples}"
                    )

                    grid = generate_grid(
                        model, img_shape, device,
                        FLAGS.n_samples, t1, dt,
                        integrator=FLAGS.integrator
                    )

                    nrow = int(math.sqrt(FLAGS.n_samples))
                    fname = (
                        f"GEN_{weight_tag}_{FLAGS.integrator}_step{step}"
                        f"_t1{t1:.2f}_dt{dt}_nsteps{n_steps}.png"
                    )
                    fpath = os.path.join(out_dir, fname)
                    save_image(grid, fpath, nrow=nrow)
                    logging.info(f"  Saved {fpath}")

    logging.info(f"\nDone! Generated samples saved to {out_dir}/")


if __name__ == "__main__":
    app.run(main)