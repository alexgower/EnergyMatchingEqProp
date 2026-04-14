# File: train_cifar_multigpu.py
# Adapted from CIFAR-10 for MNIST training.
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
from network_ep_cnn import EBEPModelWrapper
from network_ep_mlp import EBEPMLPModelWrapper
from network_ep_cet import EBEPCETModelWrapper

# TorchCFM flow classes
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from sklearn.datasets import load_digits


##############################################################################
# Helper: count_parameters
##############################################################################
def count_parameters(module: torch.nn.Module):
    """Count the total trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


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

    vt = model(t, xt)  # calls forward() in EBViTModelWrapper
    flow_mse = (vt - ut).square()
    w_flow = flow_weight(t, cutoff=time_cutoff)
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
        # pos_energy = raw_model.potential(x_real_cd, torch.ones_like(t)) # TODO this was original code see if should change back
        pos_energy = model(torch.ones_like(t), x_real_cd, return_potential=True) 

        ### NEW/CHANGED: Conditionally split negative samples based on flag.
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

        # neg_energy = raw_model.potential(x_neg, torch.ones_like(t)) # TODO this was original code see if should change back
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


def forward_all_ep_spring(raw_model, flow_matcher, x_real_flow, beta, T1, T2,
                          lambda_spring, time_cutoff, thirdphase=False,
                          record_trace=False):
    """
    Spring-clamped EP forward pass for flow matching.

    x is a dynamic variable springing back to x_t. Velocity is read from
    equilibrium displacement: v = output_scale * lambda_spring * (x* - x_t).
    No autograd for velocity, no create_graph anywhere.

    If thirdphase=True, uses three-phase EP: positive (+β) and negative (-β)
    nudge phases from the same free-phase equilibrium, giving O(β²) gradient
    estimates instead of O(β).

    If record_trace=True, neuron-value traces are stored on raw_model:
      raw_model._last_spring_free_final  — final free-phase state (reference for displacement)
      raw_model._last_nudge_pos_trace    — per-step positive-nudge trace
      raw_model._last_nudge_neg_trace    — per-step negative-nudge trace (or None)

    Returns (flow_loss_log, vt_mag, ut_mag, nudge_disp, free_disp) for logging.
    NOTE: ep_spring_gradient_step() calls .backward() internally.
    """
    device = x_real_flow.device

    x0 = torch.randn_like(x_real_flow)
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x_real_flow)

    # --- Resample high-t entries if time_cutoff < 1.0 ---
    if time_cutoff < 1.0:
        mask = t > time_cutoff  # (B,) boolean
        n_resample = mask.sum().item()
        if n_resample > 0:
            # Resample t uniformly in [0, time_cutoff]
            t[mask] = torch.rand(n_resample, device=device) * time_cutoff
            # Recompute interpolation: xt = (1-t)*x0 + t*x1
            # x0 is noise, x_real_flow is x1
            t_view = t[mask].view(-1, 1, 1, 1)  # (n_resample, 1, 1, 1)
            xt[mask] = (1 - t_view) * x0[mask] + t_view * x_real_flow[mask]
            # ut = x1 - x0 (OT linear path: independent of t, no recomputation needed)

    if hasattr(raw_model, 'archi'):  # MLP
        x_input = xt.view(xt.size(0), -1)   # (B, 784)
        ut_input = ut.view(ut.size(0), -1)  # (B, 784)
    else:  # CNN
        x_input = xt
        ut_input = ut

    # 1. Free phase — O(1) memory, no create_graph
    x_star, h_star = raw_model._converge_ep_spring_free(x_input, T1, lambda_spring)

    if record_trace:
        # Store the final free-phase state as the displacement reference.
        # Kept separate from _last_convergence_trace (set by potential()) to
        # avoid overwrite when potential() is called later in the save block.
        if hasattr(raw_model, 'archi'):  # MLP: h_star is [h1, h2, ...]
            free_final = {"x_neurons": raw_model._sample_neurons(x_star[0])}
            for idx, h in enumerate(h_star):
                free_final[f"h{idx+1}_neurons"] = raw_model._sample_neurons(h[0])
        elif hasattr(raw_model, 'n_patches'):  # CET: h_star is [z] where z is (B, N_P, D_T)
            free_final = {
                "x_neurons": raw_model._sample_neurons(x_star[0]),
                "z_neurons": raw_model._sample_neurons(h_star[0][0]),
            }
        else:  # CNN: h_star is [s1, s2, s3, s4] with shape (B, C, H, W)
            free_final = {
                "x_neurons":  raw_model._sample_neurons(x_star[0]),
                "s1_neurons": raw_model._sample_neurons(h_star[0][0]),
                "s2_neurons": raw_model._sample_neurons(h_star[1][0]),
                "s3_neurons": raw_model._sample_neurons(h_star[2][0]),
                "s4_neurons": raw_model._sample_neurons(h_star[3][0]),
            }
        raw_model._last_spring_free_final = free_final

    # Free displacement: ||x* - x_t||² + ||h*||² (x starts at x_t, h starts at 0)
    free_disp = (
        (x_star - x_input).pow(2).sum()
        + sum(h.pow(2).sum() for h in h_star)
    ).sqrt().item()

    # 2. Velocity — simple subtraction, no autograd
    v_log = raw_model.output_scale * lambda_spring * (x_star - x_input)

    # 3. Flow loss for logging only
    w_flow = flow_weight(t, cutoff=time_cutoff)
    flow_loss_log = torch.mean(w_flow * (v_log.detach() - ut_input).pow(2).flatten(1).mean(dim=1))
    vt_mag = v_log.detach().view(v_log.size(0), -1).norm(dim=1).mean()
    ut_mag = ut.view(ut.size(0), -1).norm(dim=1).mean()

    # Time-binned flow loss diagnostic (unweighted MSE per bin)
    per_sample_mse = (v_log.detach() - ut_input).pow(2).flatten(1).mean(dim=1)  # (B,)
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    flow_bins = []
    for i in range(len(bin_edges) - 1):
        bin_mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
        if bin_mask.sum() > 0:
            flow_bins.append(per_sample_mse[bin_mask].mean().item())
        else:
            flow_bins.append(float('nan'))

    # 4. Positive nudged phase: β = +beta — O(1) memory, no create_graph
    x_plus, h_plus, trace_pos = raw_model._converge_ep_spring_nudged(
        x_input, x_star, h_star, ut_input, beta, T2, lambda_spring,
        record_trace=record_trace)

    if thirdphase:
        # 4b. Negative nudged phase: β = -beta, starting from SAME (x_star, h_star)
        x_minus, h_minus, trace_neg = raw_model._converge_ep_spring_nudged(
            x_input, x_star, h_star, ut_input, -beta, T2, lambda_spring,
            record_trace=record_trace)

        # 5. Three-phase EP gradient: (E_plus - E_minus) / (2β)
        # ep_spring_gradient_step computes (E_second - E_first) / beta_arg,
        # so pass (minus, plus, 2*beta) to get (E_plus - E_minus) / (2β).
        neurons_plus = [x_plus.detach()] + [h.detach() for h in h_plus]
        neurons_minus = [x_minus.detach()] + [h.detach() for h in h_minus]
        raw_model.ep_spring_gradient_step(neurons_minus, neurons_plus, 2 * beta)

        if record_trace:
            raw_model._last_nudge_pos_trace = trace_pos
            raw_model._last_nudge_neg_trace = trace_neg

        # 6. Nudge displacement: use positive phase displacement (for diagnostics)
        nudge_disp = (
            (x_plus - x_star).pow(2).sum()
            + sum((hp - hs).pow(2).sum() for hp, hs in zip(h_plus, h_star))
        ).sqrt().item()
    else:
        # 5. Original two-phase EP gradient: (E_beta - E_star) / β
        neurons_star = [x_star.detach()] + [h.detach() for h in h_star]
        neurons_beta = [x_plus.detach()] + [h.detach() for h in h_plus]
        raw_model.ep_spring_gradient_step(neurons_star, neurons_beta, beta)

        if record_trace:
            raw_model._last_nudge_pos_trace = trace_pos
            raw_model._last_nudge_neg_trace = None

        # 6. Nudge displacement: ||x_beta - x_star|| + ||h_beta - h_star||
        nudge_disp = (
            (x_plus - x_star).pow(2).sum()
            + sum((hp - hs).pow(2).sum() for hp, hs in zip(h_plus, h_star))
        ).sqrt().item()

    return flow_loss_log, vt_mag, ut_mag, nudge_disp, free_disp, flow_bins



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
    # -----------------------------------------------------------------------
    # 2) Dataset with distributed sampler
    # -----------------------------------------------------------------------
    if FLAGS.dataset == "sklearn_digits":
        # sklearn digits: 8x8 grayscale, values 0-16, 1797 samples
        digits = load_digits()
        images = torch.tensor(digits.data, dtype=torch.float32)  # (1797, 64)
        images = images / 16.0          # normalise to [0, 1]
        images = images * 2.0 - 1.0    # scale to [-1, 1]
        images = images.view(-1, 1, 8, 8)  # (N, 1, 8, 8)
        dataset = torch.utils.data.TensorDataset(images)
        img_shape = (1, 8, 8)
    elif FLAGS.dataset == "mnist_8x8":
        # MNIST downsampled to 8×8 — 60K samples at the same resolution as sklearn digits
        data_root = os.environ.get("MNIST_PATH", "./data")
        _transform_8x8 = transforms.Compose([
            transforms.Resize(8),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if rank == 0:
            dataset = datasets.MNIST(root=data_root, train=True, download=True,
                                     transform=_transform_8x8)
            dist.barrier()
        else:
            dist.barrier()
            dataset = datasets.MNIST(root=data_root, train=True, download=False,
                                     transform=_transform_8x8)
        img_shape = (1, 8, 8)
    else:
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
    if FLAGS.model_type in ("cnn", "cnn_v2"):
        version = "v2" if FLAGS.model_type == "cnn_v2" else "v1"
        net_model = EBCNNModelWrapper(
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None,
            version=version,
        ).to(device)
    elif FLAGS.model_type == "ep_cnn":
        net_model = EBEPModelWrapper(
            T=FLAGS.ep_T,
            epsilon_ep=FLAGS.ep_epsilon,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp,
            init_gain=FLAGS.ep_init_gain,
            activation=FLAGS.ep_act,
            act_s4=FLAGS.ep_act_s4 if FLAGS.ep_act_s4 else FLAGS.ep_act,
            skip_s4=FLAGS.ep_skip_s4,
            spectral_norm_enabled=FLAGS.ep_spectral_norm,
            spectral_scale=FLAGS.ep_spectral_scale,
            x_intra_weights=FLAGS.x_intra_weights,
            lambda_spring=FLAGS.lambda_spring,
            cnn_channels=[int(c) for c in FLAGS.cnn_channels],
        ).to(device)
    elif FLAGS.model_type == "ep_mlp":
        archi = [int(x) for x in FLAGS.ep_archi]
        net_model = EBEPMLPModelWrapper(
            archi=archi,
            T=FLAGS.ep_T,
            epsilon_ep=FLAGS.ep_epsilon,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None,
            activation=FLAGS.ep_act,
            init_gain=FLAGS.ep_init_gain,
            spectral_norm_enabled=FLAGS.ep_spectral_norm,
            spectral_scale=FLAGS.ep_spectral_scale,
            x_intra_weights=FLAGS.x_intra_weights,
            lambda_spring=FLAGS.lambda_spring,
        ).to(device)
    elif FLAGS.model_type == "ep_cet":
        net_model = EBEPCETModelWrapper(
            img_channels=1 if FLAGS.dataset != "cifar10" else 3,
            img_size=8 if FLAGS.dataset == "sklearn_digits" else 28,
            patch_size=int(FLAGS.cet_patch_size),
            stride=FLAGS.cet_stride if FLAGS.cet_stride > 0 else None,
            token_dim=int(FLAGS.cet_token_dim),
            n_heads=int(FLAGS.cet_n_heads),
            head_dim=int(FLAGS.cet_head_dim),
            n_memories=int(FLAGS.cet_n_memories),
            inv_temp=float(FLAGS.cet_inv_temp),
            T=FLAGS.ep_T,
            epsilon_ep=FLAGS.ep_epsilon,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp,
            init_gain=FLAGS.ep_init_gain,
            spectral_norm_enabled=FLAGS.ep_spectral_norm,
            spectral_scale=FLAGS.ep_spectral_scale,
            lambda_spring=FLAGS.lambda_spring,
            normalize_tokens=FLAGS.cet_normalize_tokens,
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
        
        # EMA model never receives the DDP wrapper, so we load its keys unaltered # TODO check this fix though
        ema_model.load_state_dict(checkpoint_data["ema_model"])
        
        sched.load_state_dict(checkpoint_data["sched"])
        optim.load_state_dict(checkpoint_data["optim"])
        # Ensure optimizer state tensors are on the correct device
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_step = checkpoint_data["step"]
        if "spectral_scale" in checkpoint_data:
            raw_model.update_spectral_scale(checkpoint_data["spectral_scale"])
            
        if rank == 0:
            logging.info(f"[Rank 0] Resumed at step={start_step} (spectral_scale={raw_model.spectral_scale:.4f})")

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

        # Similarly, spectral_scale was loaded from checkpoint above.
        # Override with the CLI flag value so the user can change it on resume.
        # Guard with hasattr so this is a no-op for non-EP models (ViT, CNN).
        if hasattr(raw_model, 'update_spectral_scale'):
            raw_model.update_spectral_scale(FLAGS.ep_spectral_scale)

        if rank == 0:
            logging.info(f"[Rank 0] Applied CLI overrides: lr={FLAGS.lr}, spectral_scale={FLAGS.ep_spectral_scale}")

    # -----------------------------------------------------------------------
    # 6) Setup flow matcher, etc.
    # -----------------------------------------------------------------------
    sigma = 0.0
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    steps_per_log = 10
    last_log_time = time.time()
    skipped_steps = 0

    # -----------------------------------------------------------------------
    # 7) Actual Training Loop (with the difficulty-update logic + 2 data batches)
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
            # Spring EP: forward_all_ep_spring (ep_spring_gradient_step does its own .backward())
            # Fallback: forward_all + total_loss.backward() (BPTT/DEQ/ViT/CNN)
            # ------------------------------------------------------------------
            is_save_step = False
            if step > 0:
                if step <= 500:
                    is_save_step = (step % 100 == 0)
                else:
                    is_save_step = (FLAGS.save_step > 0 and step % FLAGS.save_step == 0)

            is_ep_spring_mode = (FLAGS.ep_learning_mode == 'spring' and
                                 FLAGS.model_type in ('ep_cnn', 'ep_mlp', 'ep_cet'))

            if step == start_step:
                print(f"[DEBUG] Step {step} | mode={'EP-spring' if is_ep_spring_mode else 'BPTT/DEQ'} | "
                      f"Calling forward pass...", flush=True)

            # Compute effective beta (with optional exponential annealing)
            eff_beta = (FLAGS.ep_beta * (0.5 ** (step / FLAGS.beta_anneal_halflife))
                        if FLAGS.beta_anneal_halflife > 0 else FLAGS.ep_beta)

            if is_ep_spring_mode:
                # Spring-clamped EP: no create_graph, velocity via x displacement
                flow_loss, vt_mag, ut_mag, nudge_disp, free_disp, flow_bins = forward_all_ep_spring(
                    raw_model=raw_model,
                    flow_matcher=flow_matcher,
                    x_real_flow=x_real_flow,
                    beta=eff_beta,
                    T1=FLAGS.ep_T1,
                    T2=FLAGS.ep_T2,
                    lambda_spring=FLAGS.lambda_spring,
                    time_cutoff=FLAGS.time_cutoff,
                    thirdphase=FLAGS.ep_thirdphase,
                    record_trace=(is_save_step and rank == 0),
                )
                cd_loss = torch.tensor(0.0, device=device)
                pos_energy = torch.zeros(1, device=device)
                neg_energy = torch.zeros(1, device=device)
            else:
                nudge_disp = 0.0  # not applicable outside EP mode
                free_disp = 0.0
                flow_bins = [float('nan')] * 5  # not computed for BPTT/DEQ mode
                # Standard BPTT / DEQ / ViT / CNN path
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

            # Compute & log Jacobian spectral radius (after backward, using separate detached pass)
            should_compute_spectral = (
                rank == 0
                and is_save_step
                and FLAGS.model_type in ('ep_cnn', 'ep_mlp', 'ep_cet')
            )
            if should_compute_spectral and hasattr(raw_model, 'compute_jacobian_spectral_radius'):
                x_sample = x_real_flow[:1].detach()
                if FLAGS.model_type == 'ep_mlp':
                    x_sample = x_sample.view(1, -1)
                h_star = raw_model._converge_detached(x_sample)
                rho, rho_hist = raw_model.compute_jacobian_spectral_radius(x_sample, h_star, n_iters=30)
                rho_T = rho ** raw_model.T
                if 0 < rho < 1:
                    steps_converge = int(math.log(1e-3) / math.log(rho))
                else:
                    steps_converge = -1
                pi_converged = abs(rho_hist[-1] - rho_hist[-2]) < 1e-4 if len(rho_hist) >= 2 else 'N/A'
                logging.info(f"[Jacobian] step={step}, rho={rho:.4f}, T={raw_model.T}, "
                      f"rho^T={rho_T:.6f}, steps_to_converge(eps=1e-3)="
                      f"{'inf' if steps_converge < 0 else steps_converge}, "
                      f"power_iter_converged={pi_converged}")


            if step == start_step:
                print(f"[DEBUG-BPTT] Step {step} | backward() complete! Updaing weights...", flush=True)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)


            # Per-module gradient diagnostics (every 100 steps)
            if rank == 0 and step % 100 == 0 and is_ep_spring_mode:
                grad_diag = []
                for name, p in raw_model.named_parameters():
                    if p.grad is not None:
                        g = p.grad.norm().item()
                        w = p.data.norm().item()
                        ratio = g / w if w > 1e-12 else float('inf')
                        grad_diag.append((name, g, w, ratio))
                grad_diag.sort(key=lambda x: x[1], reverse=True)
                logging.info(f"[GradDiag step={step}] per-module grad_norm / weight_norm / ratio:")
                for name, g, w, r in grad_diag:
                    logging.info(f"  {name:40s}  grad={g:.6e}  weight={w:.6e}  g/w={r:.6e}")



            # ------------------------------------------------------------------
            # Adaptive Spectral Scale Controller
            # ------------------------------------------------------------------
            if (is_ep_spring_mode) and FLAGS.adaptive_ss_rho_target > 0.0:
                # Measure ρ using a single sample from the current batch
                x_sample_ss = x_real_flow[:1].detach()
                if FLAGS.model_type == 'ep_mlp':
                    x_sample_ss = x_sample_ss.view(1, -1)
                h_star_ss = raw_model._converge_detached(x_sample_ss)
                rho_current, _ = raw_model.compute_jacobian_spectral_radius(x_sample_ss, h_star_ss, n_iters=30)
                
                current_ss = raw_model.spectral_scale
                new_ss = current_ss
                
                # Ratio-based shrink if exceeding max
                if rho_current > FLAGS.adaptive_ss_rho_target:
                    new_ss = current_ss * (FLAGS.adaptive_ss_rho_target / rho_current)
                # Gentle 0.2% grow if safely below max
                elif rho_current < FLAGS.adaptive_ss_rho_target:
                    new_ss = current_ss * 1.002
                
                # Clamp to reasonable bounds to prevent extreme divergence or vanishing
                new_ss = max(0.2, min(FLAGS.adaptive_ss_max, new_ss))
                
                if new_ss != current_ss:
                    raw_model.update_spectral_scale(new_ss)
                    if getattr(FLAGS, 'debug', False) or step % steps_per_log == 0:
                        logging.info(f"[AdaptiveSS] step={step}, rho={rho_current:.4f}, ss_old={current_ss:.4f} -> ss_new={new_ss:.4f}")

            # ------------------------------------------------------------------
            # Outlier batch skip: discard gradient entirely if nudge_disp or
            # grad norm signals the EP linear-regime assumption was violated.
            # Applying even a clipped wrong-direction gradient corrupts Adam's
            # second-moment estimates and compounds over subsequent steps.
            # ------------------------------------------------------------------
            skip_this_step = False
            skip_reason = ""
            if (is_ep_spring_mode):
                if FLAGS.skip_nudge_disp_threshold > 0 and nudge_disp > FLAGS.skip_nudge_disp_threshold:
                    skip_this_step = True
                    skip_reason = f"nudge_disp={nudge_disp:.2f} > {FLAGS.skip_nudge_disp_threshold}"
            if FLAGS.skip_grad_norm_multiplier > 0 and pre_clip_norm > FLAGS.skip_grad_norm_multiplier * FLAGS.grad_clip:
                skip_this_step = True
                skip_reason += f"{' AND ' if skip_reason else ''}grad_norm={pre_clip_norm:.2f} > {FLAGS.skip_grad_norm_multiplier}*grad_clip={FLAGS.skip_grad_norm_multiplier * FLAGS.grad_clip:.2f}"

            if skip_this_step:
                optim.zero_grad()
                skipped_steps += 1
                if rank == 0:
                    logging.info(f"[SKIP step {step}] {skip_reason} — gradient discarded (total skips: {skipped_steps})")
            else:
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
                if is_ep_spring_mode:
                    ep_ratio = nudge_disp / free_disp if free_disp > 1e-12 else float('inf')
                    curr_ss = raw_model.spectral_scale
                    ep_str = f", ss={curr_ss:.4f}, beta={eff_beta:.6f}, nudge_disp={nudge_disp:.4e}, free_disp={free_disp:.4e}, n/f_ratio={ep_ratio:.4e}"
                else:
                    ep_str = ""
                logging.info(
                    f"[Step {step}] "
                    f"flow={flow_loss.item():.5f}, cd={cd_loss.item():.5f}, "
                    f"v_mag={vt_mag.item():.5f}, u_mag={ut_mag.item():.5f}, "
                    f"pos_std={pos_energy.std().item():.5f}, "
                    f"pos_min={pos_energy.min().item():.5f}, pos_max={pos_energy.max().item():.5f}, "
                    f"neg_std={neg_energy.std().item():.5f}, "
                    f"neg_min={neg_energy.min().item():.5f}, neg_max={neg_energy.max().item():.5f}, "
                    f"grad_norm={pre_clip_norm:.4f}, clipped={'Y' if pre_clip_norm > FLAGS.grad_clip else 'N'}, "
                    f"LR={curr_lr:.6f}{ep_str}, {sps:.2f} it/s"
                    f"{f', skipped={skipped_steps}' if skipped_steps > 0 else ''}"
                    f"{f', flow_bins=[{flow_bins[0]:.3f},{flow_bins[1]:.3f},{flow_bins[2]:.3f},{flow_bins[3]:.3f},{flow_bins[4]:.3f}]' if step % 100 == 0 and not all(math.isnan(x) for x in flow_bins) else ''}"
                )

            # -------------------------------------------------
            # Save checkpoint occasionally (rank=0)
            # -------------------------------------------------
            if rank == 0 and is_save_step:
                # Generate SDE samples inline (can't use generate_samples from
                # utils — it hardcodes CIFAR-10 dims (3,32,32))
                for tag, mdl in [("normal", raw_model), ("ema", ema_model)]:
                    mdl.eval()
                    with torch.no_grad():
                        init = torch.randn(64, *img_shape, device=device)
                        # Standard (spring) generation
                        traj = sde_euler_maruyama(mdl, init, t0=0.0, t1=1.0, dt=0.01)
                        final = traj[-1].clamp(-1, 1)
                        final_01 = final / 2.0 + 0.5
                    from torchvision.utils import save_image as _save_img
                    _save_img(final_01, os.path.join(savedir, f"{tag}_generated_FM_images_step_{step}.png"), nrow=8)

                    # Energy gradient descent generation (neuromorphic inference mode)
                    if FLAGS.gen_mode == 'energy_gd' and FLAGS.model_type in ('ep_cnn', 'ep_mlp', 'ep_cet'):
                        mdl._gen_energy_gd = True
                        with torch.no_grad():
                            init_egd = torch.randn(64, *img_shape, device=device)
                            traj_egd = sde_euler_maruyama(mdl, init_egd, t0=0.0, t1=1.0, dt=0.01)
                            final_egd = traj_egd[-1].clamp(-1, 1)
                            final_egd_01 = final_egd / 2.0 + 0.5
                        _save_img(final_egd_01, os.path.join(savedir, f"{tag}_generated_EGD_images_step_{step}.png"), nrow=8)
                        mdl._gen_energy_gd = False

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
                    "spectral_scale": raw_model.spectral_scale,
                }

                torch.save(checkpoint_data, ckpt_latest)
                torch.save(checkpoint_data, ckpt_numbered)

                logging.info(f"[Rank 0] Saved checkpoint => {ckpt_latest}")
                logging.info(f"[Rank 0] Saved checkpoint => {ckpt_numbered}")

                # EP convergence diagnostic plot
                if FLAGS.model_type in ("ep_cnn", "ep_mlp", "ep_cet"):
                    sample_x = real_batch[:1].to(device)  # single sample
                    raw_model = net_model.module if hasattr(net_model, 'module') else net_model
                    raw_model.potential(sample_x, torch.tensor(0.0, device=device), record_trace=True)
                    raw_model.save_convergence_plot(savedir, step)
                    raw_model.save_layer_activations_plot(savedir, step)
                    raw_model.save_nudge_traces_plot(savedir, step)

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
