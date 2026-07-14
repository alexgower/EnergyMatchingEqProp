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


config.define_flags()  # register all the flags
FLAGS = flags.FLAGS

# 2) Import your usual goodies
from torchvision import datasets, transforms

from utils import (
    create_timestamped_dir,
    flow_weight,
    gibbs_sampling_time_sweep,
    ema,
    infiniteloop,
    save_pos_neg_grids,
    sde_euler_maruyama
)
from utils_mnist import (count_parameters, eqm_c_schedule, generate_and_save,
                         velocity_cosine_similarity,
                         log_pcn_step_diagnostics)


# 3) Import EBM models
from network_transformer_vit import EBViTModelWrapper
from network_cnn import EBCNNModelWrapper

# TorchCFM flow classes
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

# Optional: Weights & Biases
try:
    import wandb
except ImportError:
    wandb = None



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
    return total_loss, flow_loss, cd_loss, pos_energy, neg_energy, vt_mag, ut_mag, ut, t, vt


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

        # ---- Weights & Biases ----
        if FLAGS.wandb_project and wandb is not None:
            run_name = FLAGS.wandb_run_name
            if not run_name:
                # Auto-generate: base identity + any flags changed from defaults
                base = f"{FLAGS.model_type}_{FLAGS.pool_type}_{FLAGS.training_objective}"
                # Flags that are always in the base (skip in diff)
                skip = {"model_type", "pool_type", "training_objective",
                        "wandb_project", "wandb_run_name", "output_dir",
                        "my_log_dir", "resume_ckpt", "gen_output_dir"}
                changed = []
                for name, flag in FLAGS.__flags.items():
                    if name in skip:
                        continue
                    if flag.value != flag.default:
                        # Shorten the value for readability
                        v = flag.value
                        if isinstance(v, bool):
                            changed.append(f"{name}" if v else f"no{name}")
                        elif isinstance(v, float) and v == int(v):
                            changed.append(f"{name}{int(v)}")
                        else:
                            changed.append(f"{name}{v}")
                diff = "_".join(changed)
                run_name = f"{base}_{diff}" if diff else base
            wandb.init(
                project=FLAGS.wandb_project,
                name=run_name,
                config=FLAGS.flag_values_dict(),
                dir=savedir,
            )
            logging.info(f"[Rank 0] W&B run: {wandb.run.url}")
        elif FLAGS.wandb_project and wandb is None:
            logging.warning("--wandb_project set but `wandb` not installed. Skipping.")

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
                transforms.Normalize((0.5,),(0.5,))  # [0,1] → [-1,1]: (x - 0.5) / 0.5
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
                transforms.Normalize((0.5,),(0.5,))  # [0,1] → [-1,1]: (x - 0.5) / 0.5
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
    if FLAGS.model_type == "pcn":
        # Stage 2: PCN energy relaxation velocity
        from network_pcn import PCNVelocityWrapper
        net_model = PCNVelocityWrapper(
            gamma=FLAGS.pcn_gamma,
            T_free=FLAGS.T_free,
            dt_relax=FLAGS.pcn_dt,
            async_mode=FLAGS.pcn_async,
            init_mode=FLAGS.pcn_init_mode,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None,
            n_cg_steps=FLAGS.pcn_cg_steps,
            pool_type=FLAGS.pool_type,
            activation=FLAGS.activation,
            error_param=FLAGS.pcn_error_param,
            # EP parameters (Stage 3)
            param_grad_mode=FLAGS.param_grad_mode,
            lambda_spring=FLAGS.lambda_spring,
            beta=FLAGS.beta,
            T_nudge=FLAGS.T_nudge,
            thirdphase=FLAGS.thirdphase,
            K_h=FLAGS.K_h,
        ).to(device)
        if FLAGS.pcn_float64:
            net_model = net_model.to(torch.float64)
            logging.info("[PCN] Using float64 for exact gradient correspondence")
        if FLAGS.pcn_error_param:
            logging.info("[PCN] Using error-parameterized dynamics (H ≈ I)")
        logging.info(f"[PCN] K_h={FLAGS.K_h} (h-equilibration steps)")
        if FLAGS.param_grad_mode == "ep":
            logging.info(f"[PCN] EP mode: λ_spring={FLAGS.lambda_spring}, β={FLAGS.beta}, "
                         f"T_free={FLAGS.T_free}, T_nudge={FLAGS.T_nudge}, "
                         f"thirdphase={FLAGS.thirdphase}")
    elif FLAGS.model_type in ("historical", "vgg5", "mlp"):
        version = FLAGS.model_type
        net_model = EBCNNModelWrapper(
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None,
            version=version,
            pool_type=FLAGS.pool_type,
            activation=FLAGS.activation,
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
        betas=(0.9, 0.95)  # From Energy Matching paper (lower β₂ than default 0.999)
    )
    def lr_schedule(step):
        """Warmup then optional cosine decay."""
        if step < FLAGS.warmup:
            return step / FLAGS.warmup
        if FLAGS.lr_decay == "cosine":
            progress = (step - FLAGS.warmup) / max(1, FLAGS.total_steps - FLAGS.warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0  # constant after warmup

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_schedule)

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

        # ---- Optionally override lr if CLI value differs from checkpoint ----
        # optim.load_state_dict() restores the checkpoint's lr, which may differ
        # from --lr if the user wants to change it on resume.
        ckpt_lr = optim.param_groups[0]['lr']
        if abs(FLAGS.lr - ckpt_lr) > 1e-12:
            for pg in optim.param_groups:
                pg['lr'] = FLAGS.lr
                pg['initial_lr'] = FLAGS.lr
            # Recreate scheduler with correct step count so warmup/cosine don't restart
            sched = torch.optim.lr_scheduler.LambdaLR(
                optim, lr_lambda=lr_schedule, last_epoch=start_step
            )
            if rank == 0:
                logging.info(f"[Rank 0] Overriding lr: {ckpt_lr} → {FLAGS.lr} (at step {start_step})")

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
            # Cast to float64 if PCN float64 mode
            if FLAGS.model_type == "pcn" and FLAGS.pcn_float64:
                x_real_flow = x_real_flow.to(torch.float64)
                x_real_cd = x_real_cd.to(torch.float64)

            # ------------------------------------------------------------------
            # Forward + backward pass
            # ------------------------------------------------------------------
            is_save_step = False
            if step > 0:
                if step <= 500:
                    is_save_step = (step % 100 == 0)
                else:
                    is_save_step = (FLAGS.save_step > 0 and step % FLAGS.save_step == 0)

            is_ep_mode = (FLAGS.model_type == "pcn" and FLAGS.param_grad_mode == "ep")

            # Both EP and IFT use forward_all for velocity, flow loss, CD, EqM.
            # In EP mode, velocity() returns detached spring displacement (no graph).
            # In IFT mode, velocity() returns grad-tracked output (create_graph=True).
            total_loss, flow_loss, cd_loss, pos_energy, neg_energy, vt_mag, ut_mag, ut, t, vt = forward_all(
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

            if is_ep_mode:
                # EP: parameter gradients via nudge phases + energy difference.
                # total_loss has no grad (velocity was detached), so we skip .backward()
                # and instead compute EP gradients from the cached free-phase equilibrium.
                ep_diag = raw_model.compute_ep_gradients(ut)
            else:
                # IFT / feedforward: standard backprop through total_loss
                total_loss.backward()

            if step == start_step:
                print(f"[DEBUG] Step {step} | backward() complete! Updating weights...", flush=True)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)

            # Skip weight update if gradient is an outlier (IFT spike)
            grad_skipped = False
            if FLAGS.grad_skip_threshold > 0 and pre_clip_norm > FLAGS.grad_skip_threshold:
                grad_skipped = True
                optim.zero_grad()  # discard the bad gradient
            else:
                optim.step()
            sched.step()

            # ---- Diagnostic dump for high-gradient batches ----
            # Triggers at half the skip threshold (or gnorm>30 if no threshold)
            warn_thr = FLAGS.grad_skip_threshold / 2 if FLAGS.grad_skip_threshold > 0 else 30.0
            if pre_clip_norm > warn_thr and rank == 0:
                with torch.no_grad():
                    # t distribution of this batch
                    t_vals = t.detach().cpu()
                    per_sample_mse = (vt - ut).square().mean(dim=[1, 2, 3]).detach().cpu()
                    per_sample_vmag = vt.view(vt.size(0), -1).norm(dim=1).detach().cpu()
                    per_sample_umag = ut.view(ut.size(0), -1).norm(dim=1).detach().cpu()
                    worst_idx = per_sample_mse.argmax().item()

                    # Per-layer gradient norms
                    layer_gnorms = []
                    for name, p in raw_model.named_parameters():
                        if p.grad is not None:
                            layer_gnorms.append((name, p.grad.norm().item()))
                    layer_gnorms.sort(key=lambda x: -x[1])

                logging.warning(
                    f"  [GRAD_SPIKE] step={step} gnorm={pre_clip_norm:.1f} "
                    f"skipped={grad_skipped} "
                    f"t_range=[{t_vals.min():.3f},{t_vals.max():.3f}] "
                    f"t_mean={t_vals.mean():.3f} "
                    f"worst_sample: idx={worst_idx} t={t_vals[worst_idx]:.4f} "
                    f"mse={per_sample_mse[worst_idx]:.4f} "
                    f"v_mag={per_sample_vmag[worst_idx]:.1f} "
                    f"u_mag={per_sample_umag[worst_idx]:.1f}"
                )
                # Top 3 layers by gradient norm
                top_layers = " | ".join(
                    f"{n.split('.')[-2]}.{n.split('.')[-1]}={g:.1f}"
                    for n, g in layer_gnorms[:3]
                )
                logging.warning(f"  [GRAD_SPIKE] top_layers: {top_layers}")

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

                log_dict = {
                    "flow_loss": flow_loss.item(),
                    "cd_loss": cd_loss.item(),
                    "v_mag": vt_mag.item(),
                    "u_mag": ut_mag.item(),
                    "pos_energy_std": pos_energy.std().item(),
                    "pos_energy_min": pos_energy.min().item(),
                    "pos_energy_max": pos_energy.max().item(),
                    "neg_energy_std": neg_energy.std().item(),
                    "neg_energy_min": neg_energy.min().item(),
                    "neg_energy_max": neg_energy.max().item(),
                    "grad_norm": float(pre_clip_norm),
                    "lr": curr_lr,
                    "steps_per_sec": sps,
                }

                clip_status = 'SKIP' if grad_skipped else ('Y' if pre_clip_norm > FLAGS.grad_clip else 'N')
                logging.info(
                    f"[Step {step}] "
                    f"flow={log_dict['flow_loss']:.5f}, cd={log_dict['cd_loss']:.5f}, "
                    f"v_mag={log_dict['v_mag']:.5f}, u_mag={log_dict['u_mag']:.5f}, "
                    f"pos_std={log_dict['pos_energy_std']:.5f}, "
                    f"pos_min={log_dict['pos_energy_min']:.5f}, pos_max={log_dict['pos_energy_max']:.5f}, "
                    f"neg_std={log_dict['neg_energy_std']:.5f}, "
                    f"neg_min={log_dict['neg_energy_min']:.5f}, neg_max={log_dict['neg_energy_max']:.5f}, "
                    f"grad_norm={log_dict['grad_norm']:.4f}, clipped={clip_status}, "
                    f"LR={curr_lr:.6f}, {sps:.2f} it/s"
                )

                # PCN per-step diagnostics (max|e|, max|dE/de|, v_cos)
                if FLAGS.model_type == "pcn":
                    log_pcn_step_diagnostics(
                        raw_model, step, log_dict,
                        data_batch=x_real_flow, v_cos_every=500,
                    )

                # EP-specific diagnostics
                if is_ep_mode:
                    log_dict["ep_loss"] = ep_diag["ep_loss"]
                    log_dict["nudge_disp"] = ep_diag["nudge_disp"]
                    log_dict["free_x_disp"] = ep_diag["free_x_disp"]
                    neg_tag = ""
                    if "nudge_disp_neg" in ep_diag:
                        log_dict["nudge_disp_neg"] = ep_diag["nudge_disp_neg"]
                        neg_tag = f", nudge_disp_neg={ep_diag['nudge_disp_neg']:.6f}"
                    logging.info(
                        f"  [EP] ep_loss={ep_diag['ep_loss']:.6f}, "
                        f"nudge_disp={ep_diag['nudge_disp']:.6f}, "
                        f"{neg_tag}"
                        f"free_x_disp={ep_diag['free_x_disp']:.6f}"
                    )

                if wandb is not None and wandb.run is not None:
                    wandb.log(log_dict, step=step)

            # -------------------------------------------------
            # Save checkpoint occasionally (rank=0)
            # -------------------------------------------------
            if rank == 0 and is_save_step:
                # Generate sample images for visual quality check
                gen_models = [("normal", raw_model)]
                if FLAGS.gen_ema:
                    gen_models.append(("ema", ema_model))
                for tag, mdl in gen_models:
                    generate_and_save(mdl, tag, img_shape, step, savedir, device)



                # CD diagnostics: MCMC negative samples + pos/neg comparison grid
                # Only run when CD is active (lambda_cd > 0), otherwise these are useless
                if FLAGS.lambda_cd > 0:
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

    # ---- Finalize W&B ----
    if rank == 0 and wandb is not None and wandb.run is not None:
        wandb.finish()

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
