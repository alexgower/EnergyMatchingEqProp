###############################################################################
# File: sample_cifar_heun_1gpu.py
# Adapted from CIFAR-10 for MNIST sampling.
#
# Purpose:
#   Generate a batch of MNIST-sized images with an energy-based ViT model
#   using Euler–Heun SDE integration, then save samples on a single grid.
#
# Example:
#   python experiments/mnist_from_cifar10/sample_cifar_heun_1gpu.py \
#       --resume_ckpt=checkpoint_50000.pt \
#       --batch_size=128 \
#       --t_end=2.0 \
#       --dt_gibbs=0.025 \
#       --use_ema=True \
#       --epsilon_max=0.1 \
#       --time_cutoff=1.0
###############################################################################
import os, sys, math
from datetime import datetime
import torch
from torchvision.utils import save_image, make_grid

# ───────────────────────────────────────── Flags ─────────────────────────────────────────
from absl import app, flags, logging

import config_multigpu as config
config.define_flags()                                # model‑architecture flags
FLAGS = flags.FLAGS

flags.DEFINE_float("t_end", 2.0,
                   "Final SDE time (CIFAR-10: 3.25, MNIST: 2.0).")
flags.DEFINE_bool("use_ema", True,
                  "If True, load EMA weights; else raw weights.")

# ───────────────────────────────────– Model & Utils ──────────────────────────────────────
from network_transformer_vit import EBViTModelWrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils_cifar_imagenet import plot_epsilon

# ------------------------ Euler–Heun SDE integrator (modified) ---------------------------
import torchsde
def solve_sde_heun(model, x, t_start, t_end, dt=0.01):
    """Integrate `x` from t_start to t_end with Stratonovich Euler–Heun."""
    if t_end <= t_start:
        return x
    orig_shape, B = x.shape, x.size(0)
    x_flat = x.view(B, -1)

    class _FlattenSDE(torchsde.SDEStratonovich):
        def __init__(self, net):
            super().__init__(noise_type="diagonal")
            self.net = net

        # drift
        def f(self, t, y):
            y_unflat = y.view(*orig_shape)
            return self.net(t.expand(B).to(y.device), y_unflat).view(B, -1)

        def g(self, t, y):
            # Diffusion
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
            scale = torch.sqrt(2.0 * e_tensor)
            return scale.expand_as(y)

    sde  = _FlattenSDE(model)
    ts   = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)
    with torch.no_grad():
        x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
    return x_sol[-1].view(*orig_shape).clamp(-1, 1)

# ────────────────────────────────────────── Main ─────────────────────────────────────────
def main(_):
    # 1) Device & directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(FLAGS.output_dir,
                            datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Samples will be saved in: {save_dir}")

    # 2) Build model
    ch_mult = config.parse_channel_mult(FLAGS)
    model = EBViTModelWrapper(
        dim=(1, 28, 28),
        num_channels       = FLAGS.num_channels,
        num_res_blocks     = FLAGS.num_res_blocks,
        channel_mult       = ch_mult,
        attention_resolutions = FLAGS.attention_resolutions,
        num_heads          = FLAGS.num_heads,
        num_head_channels  = FLAGS.num_head_channels,
        dropout            = FLAGS.dropout,
        output_scale       = FLAGS.output_scale,
        energy_clamp       = FLAGS.energy_clamp,
        patch_size         = 7,  # NOTE: not specified in paper for MNIST. Unsure.
        embed_dim          = FLAGS.embed_dim,
        transformer_nheads = FLAGS.transformer_nheads,
        transformer_nlayers= FLAGS.transformer_nlayers,
    ).to(device).eval()

    # 3) Load checkpoint
    if not FLAGS.resume_ckpt or not os.path.isfile(FLAGS.resume_ckpt):
        raise FileNotFoundError("--resume_ckpt is missing or invalid.")
    ckpt = torch.load(FLAGS.resume_ckpt, map_location=device)
    key  = "ema_model" if FLAGS.use_ema else "net_model"
    model.load_state_dict(ckpt[key], strict=True)
    logging.info(f"Loaded {key} from {FLAGS.resume_ckpt}")

    # 4) Noise → image
    x = torch.randn(FLAGS.batch_size, 1, 28, 28, device=device)
    x = solve_sde_heun(model, x, 0.0, FLAGS.t_end, dt=FLAGS.dt_gibbs)  # in [‑1,1]
    x_01 = (x + 1.0) / 2.0                                             # → [0,1]

    # 5) Single‑grid save
    nrow = int(math.sqrt(FLAGS.batch_size))
    grid = make_grid(x_01, nrow=nrow, padding=2)
    grid_path = os.path.join(save_dir, "samples_grid.png")
    save_image(grid, grid_path)
    logging.info(f"Saved grid ({FLAGS.batch_size} images) to {grid_path}")

if __name__ == "__main__":
    app.run(main)
