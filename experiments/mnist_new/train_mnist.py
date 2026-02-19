# File: train_mnist.py
# Single-GPU training script for MNIST Energy Matching.
# Adapted from the official CIFAR-10 multi-GPU code
# (experiments/cifar10/train_cifar_multigpu.py).
#
# Faithfully implements the paper's two-phase training:
#   Phase 1 (warm-up): LOT only,      50k iters, EMA=0.999
#   Phase 2 (main):    LOT + LCD,   3.3k iters, EMA=0.99

import os
import sys
import time
import copy

import torch
from torchvision import datasets, transforms

# ── absl config ─────────────────────────────────────────────────────────
from absl import app, flags, logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import experiments.mnist_new.config as config
config.define_flags()
FLAGS = flags.FLAGS

# ── Model & shared utils ────────────────────────────────────────────────
from experiments.mnist_new.network import build_model

from utils_cifar_imagenet import (
    create_timestamped_dir,
    sde_euler_maruyama,
    flow_weight,
    gibbs_sampling_time_sweep,
    gibbs_sampling_n_steps_fast,
    warmup_lr,
    ema,
    infiniteloop,
    save_pos_neg_grids,
)
from torchvision.utils import save_image

# OT flow matching from torchcfm
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)


##############################################################################
# Helper
##############################################################################
def count_parameters(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def cd_weight(t, cutoff=1.0):
    """
    Contrastive Divergence weighting function (from utils_cifar_imagenet):
      w_cd = 0 for t < cutoff
      linearly from 0..1 as t goes from cutoff..1
      1 for t > 1.0
    """
    w = torch.zeros_like(t)
    region = (t >= cutoff) & (t <= 1.0)
    w[region] = (t[region] - cutoff) / (1.0 - cutoff + 1e-9)
    w[t > 1.0] = 1.0
    return w


##############################################################################
# Forward pass: flow + optional CD  (same logic as official CIFAR-10 code)
##############################################################################
def forward_all(
    model,
    flow_matcher,
    x_real_flow,
    x_real_cd,
    lambda_cd,
    cd_neg_clamp,
    cd_trim_fraction,
    n_gibbs,
    dt_gibbs,
    epsilon_max,
    time_cutoff,
):
    """
    Full forward pass (flow + optional CD) using separate mini-batches.
    Returns: total_loss, flow_loss, cd_loss, pos_energy, neg_energy
    """
    device = x_real_flow.device

    # ── 1) Flow matching (using x_real_flow) ────────────────────────────
    x0_flow = torch.randn_like(x_real_flow)
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
        x0_flow, x_real_flow
    )

    vt = model(t, xt)   # calls velocity via forward()
    flow_mse = (vt - ut).square()
    w_flow = flow_weight(t, cutoff=time_cutoff)
    flow_loss = torch.mean(w_flow * flow_mse.mean(dim=[1, 2, 3]))

    # ── 2) Optional CD loss (using x_real_cd) ───────────────────────────
    cd_loss = torch.tensor(0.0, device=device)
    pos_energy = torch.tensor(0.0, device=device)
    neg_energy = torch.tensor(0.0, device=device)

    if lambda_cd > 0.0:
        pos_energy = model.potential(
            x_real_cd, torch.ones(x_real_cd.size(0), device=device)
        )

        # Negative sample initialization
        B = x_real_cd.size(0)
        if FLAGS.split_negative:
            # 50/50: half from real data, half from noise
            half_b = B // 2
            x_neg_init = torch.empty_like(x_real_cd)
            x_neg_init[:half_b] = x_real_cd[:half_b]
            x_neg_init[half_b:] = torch.randn_like(x_neg_init[half_b:])
            at_data_mask = torch.zeros(B, dtype=torch.bool, device=device)
            at_data_mask[:half_b] = True
        else:
            x_neg_init = torch.randn_like(x_real_cd)
            at_data_mask = torch.zeros(B, dtype=torch.bool, device=device)

        if FLAGS.same_temperature_scheduler:
            at_data_mask = torch.zeros_like(at_data_mask)

        # Gibbs sampling with time-dependent temperature sweep
        x_neg = gibbs_sampling_time_sweep(
            x_init=x_neg_init,
            model=model,
            at_data_mask=at_data_mask,
            n_steps=n_gibbs,
            dt=dt_gibbs,
        )

        neg_energy = model.potential(
            x_neg, torch.ones(x_neg.size(0), device=device)
        )

        # Trimmed mean for negative energies
        if cd_trim_fraction > 0.0:
            k = int(cd_trim_fraction * B)
            if k > 0:
                neg_sorted, _ = neg_energy.sort()
                neg_stat = neg_sorted[: B - k].mean()
            else:
                neg_stat = neg_energy.mean()
        else:
            neg_stat = neg_energy.mean()

        cd_val = pos_energy.mean() - neg_stat
        cd_val_scaled = lambda_cd * cd_val

        # Clamp: LCD >= -β
        if cd_neg_clamp > 0:
            cd_val_scaled = torch.maximum(
                cd_val_scaled,
                torch.tensor(-cd_neg_clamp, device=device),
            )
        cd_loss = cd_val_scaled

    total_loss = flow_loss + cd_loss
    return total_loss, flow_loss, cd_loss, pos_energy, neg_energy


##############################################################################
# Training loop
##############################################################################
def train(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ── Output directory ────────────────────────────────────────────────
    savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
    if not FLAGS.my_log_dir:
        FLAGS.my_log_dir = savedir
    logging.get_absl_handler().use_absl_log_file(
        program_name="train", log_dir=FLAGS.my_log_dir
    )
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)  # Also print to terminal

    logging.info("========== Hyperparameters ==========")
    for key, val in FLAGS.flag_values_dict().items():
        logging.info(f"{key} = {val}")
    logging.info("=====================================\n")

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),   # => [-1, 1]
        ]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    datalooper = infiniteloop(dataloader)

    # ── Model ───────────────────────────────────────────────────────────
    net_model = build_model(FLAGS).to(device)

    ema_model = copy.deepcopy(net_model).to(device)

    total_params = count_parameters(net_model)
    logging.info(f"Total trainable params: {total_params:,}")

    # ── Optimizer & scheduler ───────────────────────────────────────────
    optim = torch.optim.Adam(
        net_model.parameters(), lr=FLAGS.lr, betas=(0.9, 0.95)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # ── Optional checkpoint resume ──────────────────────────────────────
    start_step = 0
    if FLAGS.resume_ckpt and os.path.exists(FLAGS.resume_ckpt):
        logging.info(f"Resuming from {FLAGS.resume_ckpt}")
        ckpt = torch.load(FLAGS.resume_ckpt, map_location=device)
        net_model.load_state_dict(ckpt["net_model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        sched.load_state_dict(ckpt["sched"])
        optim.load_state_dict(ckpt["optim"])
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_step = ckpt["step"]
        logging.info(f"Resumed at step={start_step}")

    # ── Flow matcher ────────────────────────────────────────────────────
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    # ====================================================================
    # PHASE 1: LOT only (warm-up)
    # ====================================================================
    total_phase1 = FLAGS.total_steps_phase1
    steps_per_log = 10
    last_log_time = time.time()

    logging.info(f"\n{'='*60}")
    logging.info(f"PHASE 1: LOT only for {total_phase1} steps (EMA={FLAGS.ema_decay_phase1})")
    logging.info(f"{'='*60}\n")

    for step in range(start_step, total_phase1 + 1):
        optim.zero_grad()

        x_real_flow = next(datalooper).to(device)
        x_real_cd = next(datalooper).to(device)

        total_loss, flow_loss, cd_loss, pos_energy, neg_energy = forward_all(
            model=net_model,
            flow_matcher=flow_matcher,
            x_real_flow=x_real_flow,
            x_real_cd=x_real_cd,
            lambda_cd=0.0,   # Phase 1: no CD
            cd_neg_clamp=FLAGS.cd_neg_clamp,
            cd_trim_fraction=FLAGS.cd_trim_fraction,
            n_gibbs=FLAGS.n_gibbs,
            dt_gibbs=FLAGS.dt_gibbs,
            epsilon_max=FLAGS.epsilon_max,
            time_cutoff=FLAGS.time_cutoff,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
        optim.step()
        sched.step()

        ema(net_model, ema_model, FLAGS.ema_decay_phase1)

        # Logging
        if step % steps_per_log == 0:
            now = time.time()
            elapsed = now - last_log_time
            sps = steps_per_log / elapsed if elapsed > 1e-9 else 0.0
            last_log_time = now
            curr_lr = sched.get_last_lr()[0]
            logging.info(
                f"[Phase1 Step {step}] "
                f"flow={flow_loss.item():.5f}, "
                f"LR={curr_lr:.6f}, {sps:.2f} it/s"
            )

        # Checkpoint + samples
        if FLAGS.save_step > 0 and step % FLAGS.save_step == 0 and step > 0:
            _save_checkpoint_and_samples(
                net_model, ema_model, optim, sched, step, savedir,
                datalooper, device, phase="phase1"
            )

    # ====================================================================
    # PHASE 2: LOT + LCD
    # ====================================================================
    total_phase2 = FLAGS.total_steps_phase2
    if total_phase2 <= 0:
        logging.info("Phase 2 skipped (total_steps_phase2 <= 0).")
        return

    logging.info(f"\n{'='*60}")
    logging.info(f"PHASE 2: LOT + LCD for {total_phase2} steps (EMA={FLAGS.ema_decay_phase2})")
    logging.info(f"  lambda_cd={FLAGS.lambda_cd}, n_gibbs={FLAGS.n_gibbs}, "
                 f"dt_gibbs={FLAGS.dt_gibbs}, epsilon_max={FLAGS.epsilon_max}")
    logging.info(f"{'='*60}\n")

    last_log_time = time.time()

    for step in range(total_phase2 + 1):
        optim.zero_grad()

        x_real_flow = next(datalooper).to(device)
        x_real_cd = next(datalooper).to(device)

        total_loss, flow_loss, cd_loss, pos_energy, neg_energy = forward_all(
            model=net_model,
            flow_matcher=flow_matcher,
            x_real_flow=x_real_flow,
            x_real_cd=x_real_cd,
            lambda_cd=FLAGS.lambda_cd,   # Phase 2: CD enabled
            cd_neg_clamp=FLAGS.cd_neg_clamp,
            cd_trim_fraction=FLAGS.cd_trim_fraction,
            n_gibbs=FLAGS.n_gibbs,
            dt_gibbs=FLAGS.dt_gibbs,
            epsilon_max=FLAGS.epsilon_max,
            time_cutoff=FLAGS.time_cutoff,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
        optim.step()
        sched.step()

        ema(net_model, ema_model, FLAGS.ema_decay_phase2)

        # Logging
        if step % steps_per_log == 0:
            now = time.time()
            elapsed = now - last_log_time
            sps = steps_per_log / elapsed if elapsed > 1e-9 else 0.0
            last_log_time = now
            curr_lr = sched.get_last_lr()[0]

            pos_e_str = ""
            neg_e_str = ""
            if isinstance(pos_energy, torch.Tensor) and pos_energy.dim() > 0:
                pos_e_str = (
                    f"pos_std={pos_energy.std().item():.3f}, "
                    f"pos_min={pos_energy.min().item():.3f}, "
                    f"pos_max={pos_energy.max().item():.3f}, "
                )
            if isinstance(neg_energy, torch.Tensor) and neg_energy.dim() > 0:
                neg_e_str = (
                    f"neg_std={neg_energy.std().item():.3f}, "
                    f"neg_min={neg_energy.min().item():.3f}, "
                    f"neg_max={neg_energy.max().item():.3f}, "
                )

            logging.info(
                f"[Phase2 Step {step}] "
                f"flow={flow_loss.item():.5f}, cd={cd_loss.item():.5f}, "
                f"{pos_e_str}{neg_e_str}"
                f"LR={curr_lr:.6f}, {sps:.2f} it/s"
            )

        # Checkpoint + samples
        if FLAGS.save_step > 0 and step % FLAGS.save_step == 0 and step > 0:
            _save_checkpoint_and_samples(
                net_model, ema_model, optim, sched,
                total_phase1 + step, savedir, datalooper, device,
                phase="phase2"
            )

    logging.info("Training complete.")


##############################################################################
# Checkpoint + sample helper
##############################################################################
def _generate_sde_samples(model, device, num=64):
    """Generate samples via SDE Euler-Maruyama with MNIST-correct (1,28,28) noise."""
    model_clone = copy.deepcopy(model).to(device)
    model_clone.eval()
    with torch.no_grad():
        init = torch.randn(num, 1, 28, 28, device=device)
        traj = sde_euler_maruyama(model_clone, init, t0=0.0, t1=1.0, dt=0.01)
        final = traj[-1].clamp(-1, 1)
        final = final / 2.0 + 0.5  # => [0,1]
    return final


def _save_checkpoint_and_samples(
    net_model, ema_model, optim, sched, step, savedir,
    datalooper, device, phase="phase1"
):
    """Save checkpoint, generate samples, and save pos/neg grids."""
    # Generate SDE samples (MNIST-correct dimensions)
    samples_normal = _generate_sde_samples(net_model, device, num=64)
    save_image(
        samples_normal,
        os.path.join(savedir, f"normal_generated_FM_images_step_{step}.png"),
        nrow=8,
    )
    samples_ema = _generate_sde_samples(ema_model, device, num=64)
    save_image(
        samples_ema,
        os.path.join(savedir, f"ema_generated_FM_images_step_{step}.png"),
        nrow=8,
    )
    logging.info(f"Saved sample grids for step {step}")

    # Positive / negative comparison grids
    real_batch = next(datalooper).to(device)[:64]
    x_neg_init = torch.randn_like(real_batch)
    at_data_mask = torch.zeros(
        real_batch.size(0), dtype=torch.bool, device=device
    )
    x_neg = gibbs_sampling_time_sweep(
        x_init=x_neg_init,
        model=net_model,
        at_data_mask=at_data_mask,
        n_steps=FLAGS.n_gibbs,
        dt=FLAGS.dt_gibbs,
    )
    save_pos_neg_grids(real_batch, x_neg, savedir, step)

    # Save checkpoint
    ckpt_latest = os.path.join(
        savedir, f"{FLAGS.model}_mnist_weights_step_latest.pt"
    )
    ckpt_numbered = os.path.join(savedir, f"checkpoint_{step}.pt")

    checkpoint_data = {
        "net_model": net_model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "sched": sched.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "phase": phase,
    }

    torch.save(checkpoint_data, ckpt_latest)
    torch.save(checkpoint_data, ckpt_numbered)
    logging.info(f"Saved checkpoint => {ckpt_latest}")
    logging.info(f"Saved checkpoint => {ckpt_numbered}")


##############################################################################
# Entry point
##############################################################################
if __name__ == "__main__":
    app.run(train)
