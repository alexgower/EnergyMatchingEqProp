#!/usr/bin/env python3
"""
Compute flow_bins from an existing checkpoint without training.
Loads model, runs forward passes over the dataset, reports time-binned MSE.

Usage:
  python eval_flow_bins.py \
      --ckpt=results_mnist_from_cifar10/digits/GOOD-no-beta-anneal-no-spectral-scale/checkpoint_2000.pt \
      --n_evals=20
"""

import os, sys, torch, math
import numpy as np
from absl import app, flags, logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import config_multigpu as config
config.define_flags()
FLAGS = flags.FLAGS

flags.DEFINE_string("ckpt", "", "Path to checkpoint file (required)")
flags.DEFINE_integer("n_evals", 20, "Number of forward passes to average over")

from network_ep_mlp import EBEPMLPModelWrapper
from network_ep_cnn import EBEPModelWrapper


def build_model(device):
    if FLAGS.model_type == "ep_mlp":
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
    elif FLAGS.model_type == "ep_cnn":
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
    else:
        raise ValueError(f"Unsupported model_type: {FLAGS.model_type}")
    return model


def main(_):
    if not FLAGS.ckpt:
        raise ValueError("--ckpt is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Build model and load checkpoint
    model = build_model(device)
    ckpt = torch.load(FLAGS.ckpt, map_location=device)
    state_dict = ckpt.get("net_model", ckpt.get("model", None))
    if state_dict is None:
        raise ValueError("Checkpoint has no 'net_model' or 'model' key")
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if "spectral_scale" in ckpt and hasattr(model, 'update_spectral_scale'):
        model.update_spectral_scale(ckpt["spectral_scale"])
    step = ckpt.get("step", "?")
    logging.info(f"Loaded checkpoint step={step}")

    # Load dataset
    if FLAGS.dataset == "sklearn_digits":
        from sklearn.datasets import load_digits
        digits = load_digits()
        x_data = torch.tensor(digits.data, dtype=torch.float32)
        x_data = (x_data / 8.0) * 2.0 - 1.0
        img_shape = (1, 8, 8)
    else:
        raise ValueError(f"Only sklearn_digits supported for now, got {FLAGS.dataset}")

    x_data = x_data.to(device)
    N = len(x_data)
    logging.info(f"Data: {N} samples")

    # OT flow matcher
    from torchdyn.core import NeuralODE
    try:
        from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    except ImportError:
        from torchdiffeq import odeint
        logging.warning("torchcfm not found, using simple OT approximation")
        flow_matcher = None

    model.eval()
    all_bins = []

    with torch.no_grad():
        for eval_idx in range(FLAGS.n_evals):
            # Sample x1 from data
            x1 = x_data.view(N, *img_shape)

            # Sample x0 ~ N(0, I)
            x0 = torch.randn_like(x1)

            if flow_matcher is not None:
                t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
            else:
                # Simple linear interpolation with uniform t
                t = torch.rand(N, device=device)
                t_view = t.view(-1, 1, 1, 1)
                xt = (1 - t_view) * x0 + t_view * x1
                ut = x1 - x0

            # Get model velocity via spring clamping
            B = xt.size(0)
            x_flat = xt.view(B, -1)
            lam = model.lambda_spring

            with torch.enable_grad():
                x_star, h_star = model._converge_ep_spring_free(x_flat, model.T, lam)

            v = model.output_scale * lam * (x_star - x_flat)
            v = v.view_as(ut)

            # Per-sample MSE
            per_sample_mse = (v - ut).pow(2).flatten(1).mean(dim=1)

            # Bin
            bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            bins = []
            for i in range(len(bin_edges) - 1):
                mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
                if mask.sum() > 0:
                    bins.append(per_sample_mse[mask].mean().item())
                else:
                    bins.append(float('nan'))
            all_bins.append(bins)

            if eval_idx % 5 == 0:
                flow = per_sample_mse.mean().item()
                bins_str = ','.join(f'{b:.3f}' for b in bins)
                logging.info(f"[Eval {eval_idx}/{FLAGS.n_evals}] flow={flow:.5f}, bins=[{bins_str}]")

    all_bins = np.array(all_bins)
    mean_bins = np.nanmean(all_bins, axis=0)
    std_bins = np.nanstd(all_bins, axis=0)

    logging.info(f"\n=== Final flow_bins (averaged over {FLAGS.n_evals} evals) ===")
    logging.info(f"Mean: [{','.join(f'{b:.3f}' for b in mean_bins)}]")
    logging.info(f"Std:  [{','.join(f'{b:.3f}' for b in std_bins)}]")
    logging.info(f"Ratio max/min: {np.nanmax(mean_bins)/np.nanmin(mean_bins):.2f}x")

    # Save CSV
    out_dir = os.path.dirname(FLAGS.ckpt)
    csv_path = os.path.join(out_dir, 'flow_bins_eval.csv')
    with open(csv_path, 'w') as f:
        f.write("bin_start,bin_end,mean,std\n")
        for i, (lo, hi) in enumerate(zip([0.0,0.2,0.4,0.6,0.8], [0.2,0.4,0.6,0.8,1.0])):
            f.write(f"{lo},{hi},{mean_bins[i]:.6f},{std_bins[i]:.6f}\n")
    logging.info(f"Saved to {csv_path}")


if __name__ == "__main__":
    app.run(main)