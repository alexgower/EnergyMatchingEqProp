# in_paper_diag_sample_grid.py — paper sample grids, regenerated from a seeded FID run.
#
# WHY THIS EXISTS: the FID runners stream generated images straight into the
# torchmetrics FID state and never store them, so the figure's samples cannot be
# recovered from job 34743306's output. They are, however, exactly reproducible:
# the runner seeds torch.manual_seed(fid_seed*1000 + rank) and then draws its
# first batch as randn(batch_size, 3, 32, 32), and the 1-GPU runner is written so
# that world_size=1 reproduces rank 0's stream (fid_cifar_heun_1gpu.py:327-331).
# This script replays exactly that first batch through the same integrator and
# the same flags, and saves the first N images as uint8.
#
# The batch size MATTERS: torchsde's Brownian path is drawn for the whole (B, D)
# flattened state, so generating 16 images directly would NOT give the same
# trajectories as the first 16 of a 128-batch. Always regenerate the full batch
# and slice afterwards.
#
# Usage (see submit_scripts/in_paper/in_paper_submit_imagenet32_sample_grid.sh):
#   uv run python3 in_paper_diag_sample_grid.py --model_type=... --resume_ckpt=...
#       --use_ema --fid_seed=1 --fid_times=2.5 --batch_size=128 --grid_out=<file.npy>
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from absl import app, flags, logging

import fid_cifar_heun_1gpu as fid1          # defines every model + sampler flag

FLAGS = flags.FLAGS
flags.DEFINE_string("grid_out", "samples.npy", "output .npy (uint8, N x 3 x 32 x 32)")
flags.DEFINE_integer("grid_n", 16, "how many of the first batch to keep")


def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_model, img_shape = fid1.build_model(device)
    net_model.eval()

    # --- checkpoint (same block as the FID runner) ---
    ckpt = torch.load(FLAGS.resume_ckpt, map_location=device)
    key = "ema_model" if FLAGS.use_ema else "net_model"
    sd = {k.replace("module.", ""): v for k, v in ckpt[key].items()}
    if FLAGS.ffn_checkpoint_into_pcn:
        net_model.pcn.load_from_ebvit(sd, strict=True)
        logging.info(f"loaded {key} via EBViT->PCN key remap")
    else:
        net_model.load_state_dict(sd, strict=True)
        logging.info(f"loaded {key}")
    net_model.eval()

    # --- the seeded first batch, exactly as the FID run drew it ---
    torch.manual_seed(FLAGS.fid_seed * 1000)
    times = [float(x.strip()) for x in FLAGS.fid_times.split(",")]
    x = torch.randn(FLAGS.batch_size, *img_shape, device=device)
    logging.info(f"seed={FLAGS.fid_seed * 1000} batch={FLAGS.batch_size} "
                 f"times={times} model={FLAGS.model_type}")

    t_prev = 0.0
    for t_end in times:
        x = fid1.solve_sde_heun(net_model, x, t_prev, t_end, dt=FLAGS.dt_gibbs)
        t_prev = t_end

    imgs = (((x + 1.0) / 2.0) * 255).clamp(0, 255).to(torch.uint8)[:FLAGS.grid_n]
    np.save(FLAGS.grid_out, imgs.cpu().numpy())
    logging.info(f"wrote {FLAGS.grid_out} shape={tuple(imgs.shape)} "
                 f"(first {FLAGS.grid_n} of the seeded batch, no selection by eye)")


if __name__ == "__main__":
    app.run(main)
