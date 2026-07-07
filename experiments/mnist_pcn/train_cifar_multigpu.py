# File: train_cifar_multigpu.py
# Energy Matching training for MNIST.
# Supports Phase 1 (flow matching) and Phase 2 (flow matching + contrastive divergence).
# Adapted from CIFAR-10 training script.
import os
import sys
import time
import copy
import datetime
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 1) Import absl + config
from absl import app, flags, logging
import config_multigpu as config  # your config file

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

config.define_flags()  # register all the flags
FLAGS = flags.FLAGS

# 2) Import your usual goodies
from torchvision import datasets, transforms

from utils_cifar_imagenet import (
    create_timestamped_dir,
    flow_weight,
    gibbs_sampling_time_sweep,
    warmup_lr,
    ema,
    infiniteloop,
    save_pos_neg_grids,
    sde_euler_maruyama
)
# NOTE: generate_samples from utils is NOT imported because it hardcodes
# CIFAR-10 dimensions (3, 32, 32). We use inline MNIST-correct generation below.


# 3) Import EBM models
from network_transformer_vit import EBViTModelWrapper
from network_cnn import EBCNNModelWrapper

# TorchCFM flow classes
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher


##############################################################################
# Helper: count_parameters
##############################################################################
def count_parameters(module: torch.nn.Module):
    """Count the total trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


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
# Single forward function that computes flow_loss + cd_loss in one go,
# but now uses separate mini-batches: x_real_flow for flow, x_real_cd for CD.
##############################################################################
def forward_all(model,
                flow_matcher,
                x_real_flow,
                x_real_cd,       # separate CD batch
                lambda_cd,
                cd_neg_clamp,
                cd_trim_fraction,
                n_gibbs,
                dt_gibbs,
                epsilon_max,
                time_cutoff):
    """
    Do the entire forward pass (flow + optional CD) using the
    *DDP-wrapped* model. We have two mini-batches: one for flow,
    one for CD.

    Returns: ``total_loss, flow_loss, cd_loss, pos_energy, neg_energy`` so
    that the caller can log energy statistics similarly to the ImageNet
    training script. Optionally discards a fraction of highest negative
    energies (``cd_trim_fraction``) when computing the CD gradient.
    """
    device = x_real_flow.device

    # ----------------------------------------------------------
    # 1) Flow matching (using x_real_flow)
    # ----------------------------------------------------------
    x0_flow = torch.randn_like(x_real_flow)
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0_flow, x_real_flow)

    # EqM: scale target by λ·c(γ) so velocity vanishes at data (equilibrium)
    if FLAGS.training_objective == "eqm":
        c_gamma = eqm_c_schedule(t, FLAGS.eqm_c_type, FLAGS.eqm_a, FLAGS.eqm_b)
        ut = ut * (FLAGS.eqm_lambda * c_gamma).view(-1, 1, 1, 1)

    vt = model(t, xt)  # calls forward() in EBViTModelWrapper
    flow_mse = (vt - ut).square()
    # EqM uses uniform weighting (c(γ) already handles near-data suppression);
    # FM uses flow_weight to ramp down near data for Phase 2 CD handoff.
    w_flow = torch.ones_like(t) if FLAGS.training_objective == "eqm" else flow_weight(t, cutoff=time_cutoff)
    flow_loss = torch.mean(w_flow * flow_mse.mean(dim=[1, 2, 3]))

    # For magnitude, we want the Frobenius norm over the entire 3D tensor (C, H, W) per item in batch.
    vt_mag = vt.view(vt.size(0), -1).norm(dim=1).mean()
    ut_mag = ut.view(ut.size(0), -1).norm(dim=1).mean()

    # ----------------------------------------------------------
    # 2) Optional CD loss (using x_real_cd)
    # ----------------------------------------------------------
    cd_loss = torch.tensor(0.0, device=device)
    pos_energy = torch.tensor(0.0, device=device)
    neg_energy = torch.tensor(0.0, device=device)
    raw_model = model.module if hasattr(model, 'module') else model
    if lambda_cd > 0.0:
        pos_energy = model(torch.ones_like(t), x_real_cd, return_potential=True)

        ### Conditionally split negative samples based on flag.
        if FLAGS.split_negative:
            # 50/50 split: half from x_real_cd, half from noise
            B = x_real_cd.size(0)
            half_b = B // 2
            x_neg_init = torch.empty_like(x_real_cd)

            x_neg_init[:half_b] = x_real_cd[:half_b]
            x_neg_init[half_b:] = torch.randn_like(x_neg_init[half_b:])
            at_data_mask = torch.zeros(B, dtype=torch.bool, device=device)
            at_data_mask[:half_b] = True
        else:
            # Original approach: all negative samples from noise
            x_neg_init = torch.randn_like(x_real_cd)
            at_data_mask = torch.zeros(x_real_cd.size(0), dtype=torch.bool, device=device)

        if FLAGS.same_temperature_scheduler:
            at_data_mask = torch.zeros_like(at_data_mask)

        x_neg = gibbs_sampling_time_sweep(
            x_init=x_neg_init,
            model=raw_model,
            at_data_mask=at_data_mask,
            n_steps=n_gibbs,
            dt=dt_gibbs
        )

        neg_energy = model(torch.ones_like(t), x_neg, return_potential=True)

        # Optionally use a trimmed mean for the negative energies
        if cd_trim_fraction > 0.0:
            B = neg_energy.size(0)
            k = int(cd_trim_fraction * B)
            if k > 0:
                neg_sorted, _ = neg_energy.sort()
                neg_trimmed = neg_sorted[: B - k]
                neg_stat = neg_trimmed.mean()
            else:
                neg_stat = neg_energy.mean()
        else:
            neg_stat = neg_energy.mean()

        cd_val = pos_energy.mean() - neg_stat

        cd_val_scaled = lambda_cd * cd_val
        if cd_neg_clamp > 0:
            cd_val_scaled = torch.maximum(
                cd_val_scaled,
                torch.tensor(-cd_neg_clamp, device=device)
            )
        cd_loss = cd_val_scaled

    total_loss = flow_loss + cd_loss
    return total_loss, flow_loss, cd_loss, pos_energy, neg_energy, vt_mag, ut_mag


##############################################################################
# Training loop
##############################################################################
def train_loop(rank, world_size, argv):
    # -----------------------------------------------------------------------
    # 0) Init distributed (auto-detect CPU/GPU)
    # -----------------------------------------------------------------------
    use_cuda = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "0") != ""
    if use_cuda:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        device = torch.device("cpu")
        logging.info("[CPU mode] Using gloo backend. Training will be slow.")

    # -----------------------------------------------------------------------
    # 1) Create output dir on rank=0
    # -----------------------------------------------------------------------
    savedir = None
    if rank == 0:
        savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
        if not FLAGS.my_log_dir:
            FLAGS.my_log_dir = savedir

        logging.get_absl_handler().use_absl_log_file(
            program_name="train",
            log_dir=FLAGS.my_log_dir
        )
        logging.set_verbosity(logging.INFO)
        logging.info(f"[Rank 0] Using output directory: {savedir}\n")
        logging.info("========== Hyperparameters (FLAGS) ==========")
        for key, val in FLAGS.flag_values_dict().items():
            logging.info(f"{key} = {val}")
        logging.info("=============================================\n")

    # -----------------------------------------------------------------------
    # 2) Dataset with distributed sampler
    # -----------------------------------------------------------------------
    # NOTE: RandomHorizontalFlip removed — flipping digits is not a valid
    # augmentation for MNIST (e.g. flipped '7' is not a valid digit).
    data_root = os.environ.get("MNIST_PATH", "./data")
    if rank == 0:
        dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ])
        )
        dist.barrier()  # allow other ranks to see the downloaded data
    else:
        dist.barrier()  # wait for rank 0 to download
        dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ])
        )
    img_shape = (1, 28, 28)

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True
    )
    datalooper = infiniteloop(dataloader)

    # -----------------------------------------------------------------------
    # 3) Model + DDP
    # -----------------------------------------------------------------------
    if FLAGS.model_type in ("historical", "vgg5"):
        version = "vgg5" if FLAGS.model_type == "vgg5" else "historical"
        net_model = EBCNNModelWrapper(
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None,
            version=version,
            pool_type=FLAGS.pool_type,
        ).to(device)
    else:
        # Default: UNet + ViT head (paper architecture)
        ch_mult = config.parse_channel_mult(FLAGS)
        net_model = EBViTModelWrapper(
            dim=(1, 28, 28),
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

    # If we include the CD loss (lambda_cd > 0) then every parameter is used
    # in the backward pass and find_unused_parameters should be False. When the
    # CD loss is disabled some parameters are skipped and we set it to True to
    # avoid DDP errors.
    find_unused = False if FLAGS.lambda_cd > 0.0 else True
    if world_size > 1:
        if use_cuda:
            net_model = DDP(net_model, device_ids=[rank], output_device=rank,
                            find_unused_parameters=find_unused)
        else:
            net_model = DDP(net_model, find_unused_parameters=find_unused)

    # EMA model (not DDP)
    raw_model = net_model.module if hasattr(net_model, 'module') else net_model
    ema_model = copy.deepcopy(raw_model).to(device)

    # Log params count on rank=0
    if rank == 0:
        total_params = count_parameters(raw_model)
        logging.info(f"Total trainable params: {total_params}")

    # -----------------------------------------------------------------------
    # 4) Optimizer, scheduler
    # -----------------------------------------------------------------------
    optim = torch.optim.Adam(
        net_model.parameters(),
        lr=FLAGS.lr,
        betas=(0.9, 0.95)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # -----------------------------------------------------------------------
    # 5) Optional checkpoint resume
    # -----------------------------------------------------------------------
    start_step = 0
    checkpoint_data = None
    if rank == 0 and FLAGS.resume_ckpt and os.path.exists(FLAGS.resume_ckpt):
        logging.info(f"[Rank 0] Resuming from {FLAGS.resume_ckpt}")
        checkpoint_data = torch.load(FLAGS.resume_ckpt, map_location=device)

    dist.barrier()
    checkpoint_data = [checkpoint_data]
    dist.broadcast_object_list(checkpoint_data, src=0)
    checkpoint_data = checkpoint_data[0]

    if checkpoint_data is not None:
        # Strip 'module.' prefix if the checkpoint was saved from a DDP wrapper
        net_state = {k.replace('module.', ''): v for k, v in checkpoint_data["net_model"].items()}
        raw_model.load_state_dict(net_state)

        ema_model.load_state_dict(checkpoint_data["ema_model"])

        sched.load_state_dict(checkpoint_data["sched"])
        optim.load_state_dict(checkpoint_data["optim"])
        # Ensure optimizer state tensors are on the correct device
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_step = checkpoint_data["step"]

        if rank == 0:
            logging.info(f"[Rank 0] Resumed at step={start_step}")

        # ---- Override saved hyperparameters with CLI flags ----
        # The optimizer state dict restores lr from the checkpoint, silently
        # ignoring the --lr flag. Force the CLI value into all param groups.
        for pg in optim.param_groups:
            pg['lr'] = FLAGS.lr
            pg['initial_lr'] = FLAGS.lr  # LambdaLR uses setdefault('initial_lr'), so must set explicitly
        # Reset the scheduler so it applies warmup based on the new lr.
        # Without this, the scheduler's internal state still references the
        # old lr and warmup behaves incorrectly.
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

        if rank == 0:
            logging.info(f"[Rank 0] Applied CLI overrides: lr={FLAGS.lr}")

    # -----------------------------------------------------------------------
    # 6) Setup flow matcher, etc.
    # -----------------------------------------------------------------------
    sigma = 0.0
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    steps_per_log = 10
    last_log_time = time.time()

    # -----------------------------------------------------------------------
    # 7) Actual Training Loop
    # -----------------------------------------------------------------------
    sdp_ctx = (torch.backends.cuda.sdp_kernel(
        enable_math=True, enable_flash=False, enable_mem_efficient=False
    ) if use_cuda else open("/dev/null"))
    with sdp_ctx:
        for step in range(start_step, FLAGS.total_steps + 1):
            train_sampler.set_epoch(step)  # shuffle each epoch in distributed

            optim.zero_grad()

            # Grab next batch for flow
            x_real_flow = next(datalooper).to(device)
            # Grab another batch for CD (independent from flow)
            x_real_cd = next(datalooper).to(device)

            # ------------------------------------------------------------------
            # Forward + backward pass
            # ------------------------------------------------------------------
            is_save_step = False
            if step > 0:
                if step <= 500:
                    is_save_step = (step % 100 == 0)
                else:
                    is_save_step = (FLAGS.save_step > 0 and step % FLAGS.save_step == 0)

            if step == start_step:
                print(f"[DEBUG] Step {step} | Calling forward pass...", flush=True)

            total_loss, flow_loss, cd_loss, pos_energy, neg_energy, vt_mag, ut_mag = forward_all(
                model=net_model,
                flow_matcher=flow_matcher,
                x_real_flow=x_real_flow,
                x_real_cd=x_real_cd,
                lambda_cd=FLAGS.lambda_cd,
                cd_neg_clamp=FLAGS.cd_neg_clamp,
                cd_trim_fraction=FLAGS.cd_trim_fraction,
                n_gibbs=FLAGS.n_gibbs,
                dt_gibbs=FLAGS.dt_gibbs,
                epsilon_max=FLAGS.epsilon_max,
                time_cutoff=FLAGS.time_cutoff
            )
            total_loss.backward()

            if step == start_step:
                print(f"[DEBUG] Step {step} | backward() complete! Updating weights...", flush=True)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)

            optim.step()
            sched.step()

            # Update EMA
            ema(raw_model, ema_model, FLAGS.ema_decay)

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------
            if rank == 0 and step % steps_per_log == 0:
                now = time.time()
                elapsed = now - last_log_time
                sps = steps_per_log / elapsed if elapsed > 1e-9 else 0.0
                last_log_time = now
                curr_lr = sched.get_last_lr()[0]
                logging.info(
                    f"[Step {step}] "
                    f"flow={flow_loss.item():.5f}, cd={cd_loss.item():.5f}, "
                    f"v_mag={vt_mag.item():.5f}, u_mag={ut_mag.item():.5f}, "
                    f"pos_std={pos_energy.std().item():.5f}, "
                    f"pos_min={pos_energy.min().item():.5f}, pos_max={pos_energy.max().item():.5f}, "
                    f"neg_std={neg_energy.std().item():.5f}, "
                    f"neg_min={neg_energy.min().item():.5f}, neg_max={neg_energy.max().item():.5f}, "
                    f"grad_norm={pre_clip_norm:.4f}, clipped={'Y' if pre_clip_norm > FLAGS.grad_clip else 'N'}, "
                    f"LR={curr_lr:.6f}, {sps:.2f} it/s"
                )

            # -------------------------------------------------
            # Save checkpoint occasionally (rank=0)
            # -------------------------------------------------
            if rank == 0 and is_save_step:
                # Generate SDE samples inline (can't use generate_samples from
                # utils — it hardcodes CIFAR-10 dims (3,32,32))
                # for tag, mdl in [("normal", raw_model), ("ema", ema_model)]:
                for tag, mdl in [("normal", raw_model)]:
                    mdl.eval()
                    with torch.no_grad():
                        init = torch.randn(64, *img_shape, device=device)
                        dt_gen = 0.01

                        # EqM: use convergence-based sampling (iterate until velocity is small)
                        # FM: use fixed integration from t=0 to t=gen_t1
                        use_convergence = (FLAGS.training_objective == "eqm"
                                           and FLAGS.gen_converge_threshold > 0)

                        if use_convergence:
                            # Adaptive compute: iterate until velocity norm is small
                            x = init.clone()
                            max_gen_steps = 5000  # safety cap
                            for gen_i in range(max_gen_steps):
                                t_gen = torch.full((x.size(0),), gen_i * dt_gen, device=device)
                                v = mdl(t_gen, x)
                                x = x + v * dt_gen
                                # Per-sample L2 norm (over all pixels), averaged over batch
                                v_norm = v.view(v.size(0), -1).norm(dim=1).mean().item()
                                if v_norm < FLAGS.gen_converge_threshold:
                                    logging.info(f"  [{tag}] Converged at step {gen_i+1} (v_norm={v_norm:.4f})")
                                    break
                            else:
                                logging.info(f"  [{tag}] Hit max {max_gen_steps} steps (v_norm={v_norm:.4f})")
                            final = x.clamp(-1, 1)
                            n_steps_used = min(gen_i + 1, max_gen_steps)
                        else:
                            # Fixed endpoint integration (default for FM)
                            traj = sde_euler_maruyama(mdl, init, t0=0.0, t1=FLAGS.gen_t1, dt=dt_gen)
                            final = traj[-1].clamp(-1, 1)
                            n_steps_used = int(FLAGS.gen_t1 / dt_gen)

                    from torchvision.utils import save_image as _save_img
                    mode_tag = "converged" if use_convergence else f"t1_{FLAGS.gen_t1:.1f}"
                    fname = f"{tag}_generated_images_step_{step}_{mode_tag}_nsteps{n_steps_used}.png"
                    _save_img(final / 2.0 + 0.5, os.path.join(savedir, fname), nrow=8)
                    mdl.train()

                # (a) create real data batch
                real_batch = next(datalooper).to(device)[:64]  # up to 64 for an 8x8 grid
                # (b) negative samples via MCMC (time sweep)
                x_neg_init = torch.randn_like(real_batch)
                at_data_mask = torch.zeros(real_batch.size(0), dtype=torch.bool, device=device)
                x_neg = gibbs_sampling_time_sweep(
                    x_init=x_neg_init,
                    model=raw_model,
                    at_data_mask=at_data_mask,
                    n_steps=FLAGS.n_gibbs,
                    dt=FLAGS.dt_gibbs
                )
                # (c) Save side-by-side grids
                save_pos_neg_grids(real_batch, x_neg, savedir, step)

                ckpt_latest = os.path.join(savedir,
                                          f"{FLAGS.model}_mnist_weights_step_latest.pt")
                ckpt_numbered = os.path.join(savedir, f"checkpoint_{step}.pt")

                checkpoint_data = {
                    "net_model": raw_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                }

                torch.save(checkpoint_data, ckpt_latest)
                torch.save(checkpoint_data, ckpt_numbered)

                logging.info(f"[Rank 0] Saved checkpoint => {ckpt_latest}")
                logging.info(f"[Rank 0] Saved checkpoint => {ckpt_numbered}")

    dist.barrier()
    dist.destroy_process_group()


def main(argv):
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "":
        default_ws = torch.cuda.device_count()
    else:
        default_ws = 1
    world_size = int(os.environ.get("WORLD_SIZE", default_ws))
    rank = int(os.environ.get("RANK", 0))
    train_loop(rank, world_size, argv)


if __name__ == "__main__":
    app.run(main)
