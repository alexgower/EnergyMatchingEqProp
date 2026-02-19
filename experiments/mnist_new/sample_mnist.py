# File: sample_mnist.py
# Single-GPU sampling script for MNIST Energy Matching.
# Adapted from experiments/cifar10/sample_cifar_heun_1gpu.py.

import os
import sys
import math
from datetime import datetime

import torch
from torchvision.utils import save_image, make_grid

# ── absl config ─────────────────────────────────────────────────────────
from absl import app, flags, logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import experiments.mnist_new.config as config
config.define_flags()
FLAGS = flags.FLAGS

# ── Model & shared utils ────────────────────────────────────────────────
from experiments.mnist_new.network import build_model
from utils_cifar_imagenet import plot_epsilon

# ── Optional: Euler-Heun SDE integrator via torchsde ────────────────────
try:
    import torchsde

    def solve_sde_heun(model, x, t_start, t_end, dt=0.025):
        """Integrate x from t_start to t_end with Stratonovich Euler-Heun."""
        if t_end <= t_start:
            return x
        orig_shape, B = x.shape, x.size(0)
        x_flat = x.view(B, -1)

        class _FlattenSDE(torchsde.SDEStratonovich):
            def __init__(self, net):
                super().__init__(noise_type="diagonal")
                self.net = net

            def f(self, t, y):
                y_unflat = y.view(*orig_shape)
                return self.net(t.expand(B).to(y.device), y_unflat).view(B, -1)

            def g(self, t, y):
                e_val = plot_epsilon(float(t))
                if e_val <= 0:
                    return torch.zeros_like(y)
                e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
                scale = torch.sqrt(2.0 * e_tensor)
                return scale.expand_as(y)

        sde = _FlattenSDE(model)
        ts = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)
        with torch.no_grad():
            x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
        return x_sol[-1].view(*orig_shape).clamp(-1, 1)

    HAS_TORCHSDE = True
except ImportError:
    HAS_TORCHSDE = False


def solve_sde_euler_maruyama(model, x, t_start, t_end, dt=0.025):
    """Fallback Euler-Maruyama integrator (no torchsde needed)."""
    model.eval()
    times = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)

    with torch.no_grad():
        for t_val in times:
            v = model(t_val.unsqueeze(0), x)
            e = plot_epsilon(float(t_val))
            e_tensor = torch.tensor(e, device=x.device, dtype=x.dtype)
            dt_tensor = torch.tensor(dt, device=x.device, dtype=x.dtype)
            noise = torch.randn_like(x)
            sigma = torch.sqrt(2.0 * e_tensor * dt_tensor)
            x = x + v * dt_tensor + sigma * noise

    return x.clamp(-1, 1)


##############################################################################
# Main
##############################################################################
def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(
        FLAGS.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + "_samples"
    )
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Samples will be saved in: {save_dir}")

    # Build model
    model = build_model(FLAGS).to(device).eval()

    # Load checkpoint
    if not FLAGS.resume_ckpt or not os.path.isfile(FLAGS.resume_ckpt):
        raise FileNotFoundError("--resume_ckpt is missing or invalid.")
    ckpt = torch.load(FLAGS.resume_ckpt, map_location=device)
    key = "ema_model" if FLAGS.use_ema else "net_model"
    model.load_state_dict(ckpt[key], strict=True)
    logging.info(f"Loaded {key} from {FLAGS.resume_ckpt}")

    # Generate: noise → image via SDE
    x = torch.randn(FLAGS.batch_size, 1, 28, 28, device=device)

    if HAS_TORCHSDE:
        logging.info("Using Euler-Heun SDE integrator (torchsde)")
        x = solve_sde_heun(model, x, 0.0, FLAGS.t_end, dt=FLAGS.dt_gibbs)
    else:
        logging.info("torchsde not available, using Euler-Maruyama fallback")
        x = solve_sde_euler_maruyama(model, x, 0.0, FLAGS.t_end, dt=FLAGS.dt_gibbs)

    x_01 = (x + 1.0) / 2.0  # => [0, 1]

    # Save grid
    nrow = int(math.sqrt(FLAGS.batch_size))
    grid = make_grid(x_01, nrow=nrow, padding=2)
    grid_path = os.path.join(save_dir, "samples_grid.png")
    save_image(grid, grid_path)
    logging.info(f"Saved grid ({FLAGS.batch_size} images) to {grid_path}")


if __name__ == "__main__":
    app.run(main)
