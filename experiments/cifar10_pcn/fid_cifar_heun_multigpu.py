#######################################################################
# File: fid_cifar_heun_multigpu.py  (cifar10_pcn tree, 2026-08-20)
#
# Multi-GPU port of the PCN-aware fid_cifar_heun_1gpu.py, modeled line-for-
# line on experiments/cifar10/fid_cifar_heun_multigpu.py (the FFN port).
# Motivation: PCN drift = one full relaxation per evaluation, so report-
# protocol runs are ~27h (tau=1) to ~88h (tau=3.25) on 1 GPU — the latter
# exceeds any walltime. Sharding generation over N ranks divides wall-clock
# by N at identical GPU-hours and IDENTICAL numbers (torchmetrics merges
# feature statistics across ranks inside .compute()).
#
# Reuses the 1-GPU script's build_model (full pcn/ffn MODEL_REGISTRY,
# --ffn_checkpoint_into_pcn remap) and solve_sde_heun by import, so the
# numerics match the single-GPU results exactly.
#
# NOTE --cache_real_features is IGNORED here (each rank sees only its shard
# of the real set; a monolithic cache would be wrong under a different world
# size). The sharded real pass is only ~6min/world_size.
#
# Usage (report-protocol post-CD example):
#   torchrun --standalone --nproc_per_node=4 \
#       experiments/cifar10_pcn/fid_cifar_heun_multigpu.py \
#       --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
#       --ffn_checkpoint_into_pcn --pcn_gamma=0.1 --K_h=1 --T_free=14 \
#       --resume_ckpt=/path/to/checkpoint.pt --use_ema \
#       --fid_n_samples=50000 --fid_times=3.25 --batch_size=128 \
#       --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0
#######################################################################

import datetime
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T
from absl import app, flags, logging

# Importing the 1-GPU script defines ALL flags (config.define_flags() plus
# use_ema / ffn_checkpoint_into_pcn / fid_n_samples / fid_times /
# cache_real_features) and gives us build_model + solve_sde_heun. Do NOT
# call config.define_flags() again here.
import fid_cifar_heun_1gpu as base

FLAGS = flags.FLAGS

# NB --real_dataset and --fid_seed are defined in fid_cifar_heun_1gpu (imported
# below) so the two runners share one definition; defining them again here would
# raise absl DuplicateFlagError at import.

from torchmetrics.image.fid import FrechetInceptionDistance
from utils import create_timestamped_dir
from tqdm import tqdm


def get_cifar10_train_loader_sharded(batch_size, num_workers, rank, world_size,
                                     root=None):
    """CIFAR-10 train loader sharded by DistributedSampler; images in [0,1]
    (plain ToTensor, matching fid_cifar_heun_1gpu.py — NOT [-1,1])."""
    if FLAGS.real_dataset == "imagenet32":
        # Official ImageNet32 train batches (1,281,167 imgs), same [0,1] ToTensor
        # convention as the CIFAR path. NB 1281167 % world_size != 0 -> the
        # DistributedSampler pads a handful of duplicates (<=7 of 1.28M; negligible).
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imagenet"))
        from dataset_imagenet32 import ImageNet32Dataset
        dataset = ImageNet32Dataset(split="train", root=os.environ.get("IMAGENET32_PATH"),
                                    transform=T.ToTensor())
    else:
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


def main(argv):
    # A) distributed init (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK).
    # Long timeout: generation runs for hours with NO collectives, so rank
    # skew at the final merge can exceed the NCCL default. PCN relaxation
    # makes runs even longer than the FFN port -> 12h.
    dist.init_process_group(backend="nccl", init_method="env://",
                            timeout=datetime.timedelta(hours=12))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # B) logging (rank 0 only); same fid_evals default as the 1-GPU script.
    if rank == 0:
        if not FLAGS.output_dir or FLAGS.output_dir.rstrip("/") == "./results_cifar10_pcn":
            FLAGS.output_dir = "./results_cifar10_pcn/fid_evals/"
        savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
        logging.get_absl_handler().use_absl_log_file(
            program_name="fid_cifar10", log_dir=savedir)
        logging.set_verbosity(logging.INFO)
        logging.info(f"[Rank 0] Saving logs to: {savedir}  world_size={world_size}")
    else:
        logging.set_verbosity(logging.ERROR)
    dist.barrier()

    # C) model + checkpoint — same construction/loading as the 1-GPU script,
    # including the FFN->PCN key remap branch.
    net_model, img_shape = base.build_model(device)

    if not FLAGS.resume_ckpt or not os.path.exists(FLAGS.resume_ckpt):
        raise ValueError(f"[Rank {rank}] --resume_ckpt not found: {FLAGS.resume_ckpt}")
    if rank == 0:
        logging.info(f"[Rank 0] Loading checkpoint: {FLAGS.resume_ckpt}")
    ckpt_data = torch.load(FLAGS.resume_ckpt, map_location=device)
    key = "ema_model" if FLAGS.use_ema else "net_model"
    state_dict = {k.replace('module.', ''): v for k, v in ckpt_data[key].items()}
    if FLAGS.ffn_checkpoint_into_pcn:
        if not hasattr(net_model, "pcn"):
            raise ValueError("--ffn_checkpoint_into_pcn requires a pcn --model_type")
        net_model.pcn.load_from_ebvit(state_dict, strict=True)
        if rank == 0:
            logging.info(f"[Rank 0] Loaded {key} via EBViT->PCN key remap.")
    else:
        net_model.load_state_dict(state_dict, strict=True)
        if rank == 0:
            logging.info(f"[Rank 0] Loaded {key}.")
    net_model.eval()
    dist.barrier()

    if FLAGS.fid_seed >= 0:
        torch.manual_seed(FLAGS.fid_seed * 1000 + rank)
        if rank == 0:
            logging.info(f"[Rank 0] fid_seed={FLAGS.fid_seed} "
                         f"(per-rank seed = fid_seed*1000+rank)")

    # D) FID accumulators (torchmetrics merges across ranks at .compute())
    times_to_sample = [float(t) for t in FLAGS.fid_times.split(",") if t.strip()]
    fid_dict = {t_val: FrechetInceptionDistance(feature=2048).to(device)
                for t_val in times_to_sample}

    # E) real shard (each rank does 50000/world_size images)
    train_loader = get_cifar10_train_loader_sharded(
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

    # F) fake shard: each rank generates ~fid_n_samples/world_size
    total_gen_local = FLAGS.fid_n_samples // world_size
    if rank < (FLAGS.fid_n_samples % world_size):
        total_gen_local += 1
    if rank == 0:
        logging.info(f"[Rank 0] Each rank generates ~{total_gen_local} of "
                     f"{FLAGS.fid_n_samples} fakes (tau_s: {times_to_sample})...")

    n_batches = (total_gen_local + FLAGS.batch_size - 1) // FLAGS.batch_size
    for _ in tqdm(range(n_batches), desc=f"Rank {rank} Generating",
                  disable=(rank != 0)):
        curr_bsz = min(FLAGS.batch_size, total_gen_local)
        total_gen_local -= curr_bsz
        if curr_bsz <= 0:
            break
        x = torch.randn(curr_bsz, *img_shape, device=device)
        t_prev = 0.0
        for t_end in times_to_sample:
            x = base.solve_sde_heun(net_model, x, t_prev, t_end,
                                    dt=FLAGS.dt_gibbs)
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
            logging.info(f"FID at t={t_val:.2f} => {fid_val:.4f} "
                         f"(total fakes: {FLAGS.fid_n_samples}, real: {len(train_loader.dataset)})")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
