#######################################################################
# File: fid_cifar_heun_multigpu.py
#
# Multi-GPU port of fid_cifar_heun_1gpu.py, modeled on the authors' own
# experiments/imagenet/fid_imagenet_heun_multigpu.py. Generation (the ~22h
# bottleneck of the 1-GPU script) is sharded across ranks: each rank
# generates n_samples/world_size fakes and processes its DistributedSampler
# shard of the real set; torchmetrics' FrechetInceptionDistance merges the
# feature statistics across ranks inside .compute(). Wall-clock is ~1/N of
# the 1-GPU script at identical GPU-hours and IDENTICAL numbers.
#
# Conventions deliberately copied from fid_cifar_heun_1gpu.py so results
# overlay exactly:
#   - real images: plain ToTensor() -> [0,1] -> x255 uint8 (no Normalize)
#   - fake images: clamp(-1,1) -> (x+1)/2 -> x255 uint8
#   - same default 17-point tau_s sweep, same solve_sde_heun (torchsde
#     Stratonovich "heun", eps(t) schedule via plot_epsilon)
#
# NOTE no --cache_real_features here (unlike the patched 1gpu script): each
# rank only sees ITS shard of the real set, so a monolithic cache would be
# wrong under a different world size. The real pass is only ~6min/world_size.
#
# Usage:
#   torchrun --standalone --nproc_per_node=4 \
#       experiments/cifar10/fid_cifar_heun_multigpu.py \
#       --resume_ckpt=/path/to/checkpoint.pt \
#       --use_ema --batch_size=128 --dt_gibbs=0.01 \
#       --epsilon_max=0.01 --time_cutoff=1.0
#######################################################################

import datetime
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T
from absl import app, flags, logging

import config_multigpu as config
config.define_flags()
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_ema", True,
                  "If True, load the EMA model from the checkpoint (default True).")
flags.DEFINE_integer("n_samples", 50000,
                     "TOTAL fake samples across all ranks. Reduce for a quick read "
                     "(NB FID biases upward with fewer samples -- rank, don't report).")
flags.DEFINE_string("fid_times", "1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0",
                    "Comma-separated tau_s evaluation points. Cost is set by the max.")

flags.DEFINE_integer("fid_seed", -1,
                     "Sampling seed for the fake-sample noise (per rank: "
                     "fid_seed*1000+rank). -1 = unseeded (legacy). Mirrors the "
                     "flag in the cifar10_pcn port, for matched seed replicates.")

from torchmetrics.image.fid import FrechetInceptionDistance
from network_transformer_vit import EBViTModelWrapper
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from utils_cifar_imagenet import (
    create_timestamped_dir,
    plot_epsilon
)

import torchsde
from tqdm import tqdm


def get_cifar10_train_loader(batch_size, num_workers, rank, world_size, root=None):
    """CIFAR-10 train loader sharded by DistributedSampler; images in [0,1]
    (plain ToTensor, matching fid_cifar_heun_1gpu.py -- NOT [-1,1])."""
    if root is None:
        root = os.environ.get("CIFAR10_PATH", "./data")
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=T.ToTensor()
    )
    # 50000 % {1,2,4} == 0 -> no DistributedSampler duplicate-padding of the
    # real set at the world sizes we use (padding would double-count samples).
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=sampler, drop_last=False,
    )


def solve_sde_heun(model, x, t_start, t_end, dt=0.01):
    """Integrates x from t_start..t_end with Stratonovich Euler-Heun
    (identical to fid_cifar_heun_1gpu.py)."""
    if t_end <= t_start:
        return x

    orig_shape = x.shape
    B = x.size(0)
    x_flat = x.view(B, -1)

    class FlattenSDE(torchsde.SDEStratonovich):
        def __init__(self, net):
            super().__init__(noise_type="diagonal")
            self.net = net

        def f(self, t, y):
            y_unflat = y.view(*orig_shape)
            t_batch = t.expand(B).to(y.device)
            v = self.net(t_batch, y_unflat)
            return v.view(B, -1)

        def g(self, t, y):
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
            scale = torch.sqrt(2.0 * e_tensor)
            return scale.expand_as(y)

    sde = FlattenSDE(model)
    ts = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)

    with torch.no_grad():
        x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
        x_final = x_sol[-1].view(*orig_shape).clamp(-1, 1)

    return x_final


def main(argv):
    # A) distributed init (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK).
    # Long timeout: generation runs for hours with NO collectives, so rank skew
    # at the final all-gather can exceed the NCCL default
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=datetime.timedelta(hours=6))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # B) logging (rank 0 only)
    if rank == 0:
        if not FLAGS.output_dir:
            FLAGS.output_dir = "./sampling_results"
        savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
        logging.get_absl_handler().use_absl_log_file(
            program_name="fid_cifar10", log_dir=savedir)
        logging.set_verbosity(logging.INFO)
        logging.info(f"[Rank 0] Saving logs to: {savedir}  world_size={world_size}")
    else:
        logging.set_verbosity(logging.ERROR)
    dist.barrier()

    # C) model + checkpoint (same construction as fid_cifar_heun_1gpu.py)
    ch_mult = config.parse_channel_mult(FLAGS)
    net_model = EBViTModelWrapper(
        dim=(3, 32, 32),
        num_channels=FLAGS.num_channels,
        num_res_blocks=FLAGS.num_res_blocks,
        channel_mult=ch_mult,
        attention_resolutions=FLAGS.attention_resolutions,
        num_heads=FLAGS.num_heads,
        num_head_channels=FLAGS.num_head_channels,
        dropout=FLAGS.dropout,
        output_scale=FLAGS.output_scale,
        energy_clamp=FLAGS.energy_clamp,
        patch_size=4,
        embed_dim=FLAGS.embed_dim,
        transformer_nheads=FLAGS.transformer_nheads,
        transformer_nlayers=FLAGS.transformer_nlayers,
    ).to(device)
    net_model.eval()

    if not FLAGS.resume_ckpt or not os.path.exists(FLAGS.resume_ckpt):
        raise ValueError(f"[Rank {rank}] --resume_ckpt not found: {FLAGS.resume_ckpt}")
    if rank == 0:
        logging.info(f"[Rank 0] Loading checkpoint: {FLAGS.resume_ckpt}")
    ckpt_data = torch.load(FLAGS.resume_ckpt, map_location=device)
    key = "ema_model" if FLAGS.use_ema else "net_model"
    net_model.load_state_dict(ckpt_data[key], strict=True)
    net_model.eval()
    dist.barrier()

    if FLAGS.fid_seed >= 0:
        torch.manual_seed(FLAGS.fid_seed * 1000 + rank)
        if rank == 0:
            logging.info(f"[Rank 0] fid_seed={FLAGS.fid_seed}")

    # D) FID accumulators (torchmetrics merges across ranks at .compute())
    times_to_sample = [float(t) for t in FLAGS.fid_times.split(",") if t.strip()]
    fid_dict = {t_val: FrechetInceptionDistance(feature=2048).to(device)
                for t_val in times_to_sample}

    # E) real shard (each rank does 50000/world_size images)
    train_loader = get_cifar10_train_loader(
        batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
        rank=rank, world_size=world_size,
    )
    if rank == 0:
        logging.info("[Rank 0] Updating FID with real images (sharded)...")
    for real_imgs, _ in tqdm(train_loader, desc=f"Rank {rank} Real Data",
                             disable=(rank != 0)):
        real_uint8 = (real_imgs.to(device) * 255).clamp(0, 255).to(torch.uint8)
        for t_val in times_to_sample:
            fid_dict[t_val].update(real_uint8, real=True)
    dist.barrier()

    # F) fake shard: each rank generates ~n_samples/world_size
    total_gen_local = FLAGS.n_samples // world_size
    if rank < (FLAGS.n_samples % world_size):
        total_gen_local += 1
    if rank == 0:
        logging.info(f"[Rank 0] Each rank generates ~{total_gen_local} of "
                     f"{FLAGS.n_samples} fakes (tau_s: {times_to_sample})...")

    n_batches = (total_gen_local + FLAGS.batch_size - 1) // FLAGS.batch_size
    for _ in tqdm(range(n_batches), desc=f"Rank {rank} Generating",
                  disable=(rank != 0)):
        curr_bsz = min(FLAGS.batch_size, total_gen_local)
        total_gen_local -= curr_bsz
        if curr_bsz <= 0:
            break
        x = torch.randn(curr_bsz, 3, 32, 32, device=device)
        t_prev = 0.0
        for t_end in times_to_sample:
            x = solve_sde_heun(net_model, x, t_prev, t_end, dt=FLAGS.dt_gibbs)
            x_01 = (x + 1.0) / 2.0
            x_uint8 = (x_01 * 255).clamp(0, 255).to(torch.uint8)
            fid_dict[t_end].update(x_uint8, real=False)
            t_prev = t_end
    dist.barrier()

    # G) compute (merged across ranks internally)
    if rank == 0:
        logging.info("[Rank 0] Computing final FIDs...")
    for t_val in times_to_sample:
        fid_val = fid_dict[t_val].compute()
        if rank == 0:
            logging.info(f"FID at t={t_val:.2f} => {fid_val:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
