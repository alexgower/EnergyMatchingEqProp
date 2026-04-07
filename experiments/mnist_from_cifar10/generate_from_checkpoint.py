#!/usr/bin/env python3
"""
Generate images from a saved checkpoint with configurable generation parameters.

Sweep dt, t1 (integration endpoint), and gen_mode without retraining.
Extending t1 > 1.0 is equivalent to velocity rescaling (compensates for
the systematic v_mag undershoot caused by MSE-optimal magnitude reduction).

Usage examples:
  # Default spring generation
  python generate_from_checkpoint.py \
      --ckpt=results_mnist_from_cifar10/EBMTime_20260405_16/checkpoint_2200.pt \
      --output_dir=./gen_sweep

  # Sweep t1 values for spring and EGD
  python generate_from_checkpoint.py \
      --ckpt=results_mnist_from_cifar10/EBMTime_20260405_16/checkpoint_2200.pt \
      --output_dir=./gen_sweep \
      --t1=1.0,1.05,1.1,1.15,1.2 \
      --dt=0.01,0.005 \
      --gen_mode=spring,energy_gd \
      --n_samples=64

  # Single config
  python generate_from_checkpoint.py \
      --ckpt=path/to/checkpoint.pt \
      --t1=1.1 --dt=0.005 --gen_mode=energy_gd --use_ema
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

# Generation-specific flags (override defaults from config)
flags.DEFINE_string("ckpt", "", "Path to checkpoint file (required)")
flags.DEFINE_string("gen_output_dir", "./gen_outputs", "Output directory for generated images")
flags.DEFINE_string("gen_t1", "1.0", "Comma-separated list of integration endpoints (e.g. '1.0,1.1,1.2')")
flags.DEFINE_string("gen_dt", "0.01", "Comma-separated list of dt values (e.g. '0.01,0.005')")
flags.DEFINE_string("gen_modes", "spring", "Comma-separated gen modes: spring,energy_gd")
flags.DEFINE_integer("n_samples", 64, "Number of samples to generate per config")
flags.DEFINE_bool("use_ema", False, "Also generate with EMA model weights")
flags.DEFINE_bool("use_normal", True, "Use non-EMA model weights (default)")

# Import models
from network_ep import EBEPModelWrapper
from network_ep_mlp import EBEPMLPModelWrapper
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


def build_model(device):
    """Build model from FLAGS (same logic as train script)."""
    if FLAGS.model_type == "ep_cnn":
        model = EBEPModelWrapper(
            T=FLAGS.ep_T,
            epsilon_ep=FLAGS.ep_epsilon,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp,
            init_gain=FLAGS.ep_init_gain,
            activation=FLAGS.ep_act,
            act_s4=FLAGS.ep_act_s4 if FLAGS.ep_act_s4 else FLAGS.ep_act,
            skip_s4=FLAGS.ep_skip_s4,
            spectral_norm_enabled=FLAGS.ep_spectral_norm,
            learning_mode=FLAGS.ep_learning_mode,
            neumann_K=FLAGS.ep_K,
            spectral_scale=FLAGS.ep_spectral_scale,
            x_intra_weights=FLAGS.x_intra_weights,
            lambda_spring=FLAGS.lambda_spring,
            cnn_channels=[int(c) for c in FLAGS.cnn_channels],
        ).to(device)
        img_shape = (1, 28, 28)
    elif FLAGS.model_type == "ep_mlp":
        archi = [int(x) for x in FLAGS.ep_archi]
        model = EBEPMLPModelWrapper(
            archi=archi,
            T=FLAGS.ep_T,
            epsilon_ep=FLAGS.ep_epsilon,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None,
            activation=FLAGS.ep_act,
            init_gain=FLAGS.ep_init_gain,
            spectral_norm_enabled=FLAGS.ep_spectral_norm,
            learning_mode=FLAGS.ep_learning_mode,
            neumann_K=FLAGS.ep_K,
            spectral_scale=FLAGS.ep_spectral_scale,
            x_intra_weights=FLAGS.x_intra_weights,
            lambda_spring=FLAGS.lambda_spring,
        ).to(device)
        if FLAGS.dataset == "sklearn_digits" or FLAGS.dataset == "mnist_8x8":
            img_shape = (1, 8, 8)
        else:
            img_shape = (1, 28, 28)
    else:
        raise ValueError(f"Unsupported model_type for generation: {FLAGS.model_type}")
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
    ss = ckpt.get("spectral_scale", "unknown")

    if "spectral_scale" in ckpt and hasattr(model, 'update_spectral_scale'):
        model.update_spectral_scale(ckpt["spectral_scale"])

    logging.info(f"Loaded {key} from {ckpt_path} (step={step}, ss={ss})")
    return model, step


def generate_grid(model, img_shape, device, n_samples, t1, dt, gen_mode):
    """Generate a grid of samples with given parameters."""
    model.eval()

    # Set generation mode
    if gen_mode == 'energy_gd' and hasattr(model, '_gen_energy_gd'):
        model._gen_energy_gd = True

    init = torch.randn(n_samples, *img_shape, device=device)
    final = sde_euler_maruyama_gen(model, init, t0=0.0, t1=t1, dt=dt)
    final_01 = final / 2.0 + 0.5  # [-1,1] -> [0,1]

    # Reset generation mode
    if gen_mode == 'energy_gd' and hasattr(model, '_gen_energy_gd'):
        model._gen_energy_gd = False

    return final_01


def main(_):
    if not FLAGS.ckpt:
        raise ValueError("--ckpt is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Parse sweep parameters
    t1_values = [float(x.strip()) for x in FLAGS.gen_t1.split(",")]
    dt_values = [float(x.strip()) for x in FLAGS.gen_dt.split(",")]
    gen_modes = [x.strip() for x in FLAGS.gen_modes.split(",")]

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

    # Create output directory
    os.makedirs(FLAGS.gen_output_dir, exist_ok=True)

    # Get checkpoint basename for labeling
    ckpt_name = os.path.splitext(os.path.basename(FLAGS.ckpt))[0]

    from torchvision.utils import save_image

    total_configs = len(weight_variants) * len(gen_modes) * len(t1_values) * len(dt_values)
    config_idx = 0

    for weight_tag, use_ema in weight_variants:
        # Load weights
        model, step = load_checkpoint(model, FLAGS.ckpt, device, use_ema=use_ema)

        for gen_mode in gen_modes:
            for t1 in t1_values:
                for dt in dt_values:
                    config_idx += 1
                    n_steps = int(round(t1 / dt))

                    logging.info(
                        f"[{config_idx}/{total_configs}] "
                        f"{weight_tag} | {gen_mode} | t1={t1:.2f} | dt={dt} | "
                        f"steps={n_steps} | n={FLAGS.n_samples}"
                    )

                    grid = generate_grid(
                        model, img_shape, device,
                        FLAGS.n_samples, t1, dt, gen_mode
                    )

                    # Filename encodes all parameters
                    nrow = int(math.sqrt(FLAGS.n_samples))
                    fname = (
                        f"{ckpt_name}_step{step}_{weight_tag}_{gen_mode}_"
                        f"t1={t1:.2f}_dt={dt}.png"
                    )
                    fpath = os.path.join(FLAGS.gen_output_dir, fname)
                    save_image(grid, fpath, nrow=nrow)
                    logging.info(f"  Saved {fpath}")

    logging.info(f"\nDone! {total_configs} configs saved to {FLAGS.gen_output_dir}/")


if __name__ == "__main__":
    app.run(main)