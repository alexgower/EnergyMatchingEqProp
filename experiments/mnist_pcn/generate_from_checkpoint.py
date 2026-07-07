#!/usr/bin/env python3
"""
Generate images from a saved checkpoint with configurable generation parameters.

Sweep dt and t1 (integration endpoint) without retraining.
Extending t1 > 1.0 is equivalent to velocity rescaling (compensates for
the systematic v_mag undershoot caused by MSE-optimal magnitude reduction).

Usage examples:
  # Default generation
  python generate_from_checkpoint.py \
      --ckpt=results_mnist_pcn/EBMTime_20260405_16/checkpoint_50000.pt \
      --output_dir=./gen_sweep

  # Sweep t1 values
  python generate_from_checkpoint.py \
      --ckpt=results_mnist_pcn/EBMTime_20260405_16/checkpoint_50000.pt \
      --output_dir=./gen_sweep \
      --gen_t1=1.0,1.05,1.1,1.15,1.2 \
      --gen_dt=0.01,0.005 \
      --n_samples=64

  # Single config
  python generate_from_checkpoint.py \
      --ckpt=path/to/checkpoint.pt \
      --gen_t1=1.1 --gen_dt=0.005 --use_ema
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
flags.DEFINE_string("gen_t1_sweep", "1.0", "Comma-separated list of integration endpoints (e.g. '1.0,1.5,2.0')")
flags.DEFINE_string("gen_dt_sweep", "0.01", "Comma-separated list of dt values (e.g. '0.01,0.005')")
flags.DEFINE_integer("n_samples", 64, "Number of samples to generate per config")
flags.DEFINE_bool("use_ema", False, "Also generate with EMA model weights")
flags.DEFINE_bool("use_normal", True, "Use non-EMA model weights (default)")

# Import models
from network_cnn import EBCNNModelWrapper
from network_transformer_vit import EBViTModelWrapper


def sde_euler_maruyama_gen(model, x0, t0, t1, dt=0.01):
    """
    Euler-Maruyama integration from t0 to t1 (deterministic, no noise).
    Returns only the final sample for efficiency.
    """
    device = x0.device
    times = torch.arange(t0, t1 + 1e-6, dt, device=device)
    x = x0.clone()

    with torch.no_grad():
        for t_val in times:
            v = model(t_val.unsqueeze(0), x)
            dt_tensor = torch.tensor(dt, device=device, dtype=x.dtype)
            x = x + v * dt_tensor

    return x.clamp(-1, 1)


def generate_converged(model, x0, dt=0.01, threshold=0.5, max_steps=5000):
    """
    Adaptive compute generation: iterate until mean velocity norm < threshold.
    Returns (final_samples, n_steps_used).
    """
    x = x0.clone()
    with torch.no_grad():
        for i in range(max_steps):
            t_val = torch.full((x.size(0),), i * dt, device=x.device)
            v = model(t_val, x)
            x = x + v * dt
            v_norm = v.view(v.size(0), -1).norm(dim=1).mean().item()
            if v_norm < threshold:
                logging.info(f"  Converged at step {i+1} (v_norm={v_norm:.4f})")
                return x.clamp(-1, 1), i + 1
    logging.info(f"  Hit max {max_steps} steps (v_norm={v_norm:.4f})")
    return x.clamp(-1, 1), max_steps


def build_model(device):
    """Build model from FLAGS (same logic as train script)."""
    img_shape = (1, 28, 28)

    if FLAGS.model_type in ("historical", "vgg5"):
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


def generate_grid(model, img_shape, device, n_samples, t1, dt):
    """Generate a grid of samples with given parameters."""
    model.eval()

    init = torch.randn(n_samples, *img_shape, device=device)
    final = sde_euler_maruyama_gen(model, init, t0=0.0, t1=t1, dt=dt)
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
        weight_variants.append(("ema", True))  # default

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
            # Convergence mode: single run until velocity norm < threshold
            config_idx += 1
            logging.info(
                f"[{config_idx}] {weight_tag} | convergence mode | "
                f"threshold={FLAGS.gen_converge_threshold} | n={FLAGS.n_samples}"
            )
            init = torch.randn(FLAGS.n_samples, *img_shape, device=device)
            samples, n_steps = generate_converged(
                model, init, dt=dt_values[0],
                threshold=FLAGS.gen_converge_threshold
            )
            grid = samples / 2.0 + 0.5
            nrow = int(math.sqrt(FLAGS.n_samples))
            fname = (
                f"GEN_{weight_tag}_step{step}"
                f"_converged_thresh{FLAGS.gen_converge_threshold}"
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
                        FLAGS.n_samples, t1, dt
                    )

                    nrow = int(math.sqrt(FLAGS.n_samples))
                    fname = (
                        f"GEN_{weight_tag}_step{step}"
                        f"_t1{t1:.2f}_dt{dt}_nsteps{n_steps}.png"
                    )
                    fpath = os.path.join(out_dir, fname)
                    save_image(grid, fpath, nrow=nrow)
                    logging.info(f"  Saved {fpath}")

    logging.info(f"\nDone! Generated samples saved to {out_dir}/")


if __name__ == "__main__":
    app.run(main)