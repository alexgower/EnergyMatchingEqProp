# Sample grids for the CIFAR-10 arms (App. samples)

Figures for the appendix's generated-samples section: the three main arms of Sec. 5
(backpropagation, implicit solve, EP) and the norm- and attention-free architecture of
Sec. \ref{subsec:hardware}. Produced 2026-09-20.

## What is in here

| file | what |
|---|---|
| `samples_<arm>.npy` | (128, 3, 32, 32) uint8 — the seeded batch for that arm, as sampled |
| `samples_<arm>.png` | the paper figure: ALL 128 samples of that arm, 16 per row (542 x 270 px) |
| `make_figures.py` | rebuilds the PNGs from the .npy (no GPU needed) |
| `generation.log` | the box's own log of the four sampling runs |

`uv run python3 make_figures.py [ncol]` regenerates the four figures (default 16 per row). The figure
is the WHOLE batch in the order the sampler produced it — no block is chosen, no image is selected, so
there is nothing to disclose about selection beyond the seed. Because panel i is the same x0 in every
arm, the four figures can be read image by image as well as as a whole.

## How the samples were drawn

Each arm is sampled from its **phase-2 checkpoint** under **its own native inference**, with EMA
weights, Euler--Heun via torchsde at dt = 0.01, epsilon_max = 0.01, time_cutoff = 1.0 — i.e. the FID
protocol of `in_paper_fid_vs_steps_main_arms/`, at each arm's own optimal sampling time.

| arm | checkpoint | inference | tau_s | FID of these weights |
|---|---|---|---|---|
| backpropagation | `in_paper/equivalence_at_inference/in_paper_fid_calculation_train_backprop_infer_pcn/checkpoint_postcd_147000.pt` | `--model_type=ffn_unet_vit` (autograd velocity) | 3.25 | 3.5370 |
| implicit solve | `main/stage2_ift_main_g01_drop01_p2cd_twin_fix/EM_cifar10_pcn_20260916_13/checkpoint_147000.pt` | `--model_type=pcn_unet_vit --pcn_error_param --pcn_gamma=0.1 --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0` | 3.25 | 3.6351 |
| EP | `main/stage3_ep_main_g01_drop01_p2cd_EP100K_V2/EM_cifar10_pcn_20260918_01/checkpoint_102000.pt` | as the implicit-solve row but `--pcn_gamma=0.003` (this arm's phase-2 solver gamma, and the gamma its FID used) | 3.25 | 3.6401 |
| norm/attention-free | `strip/ablate_ws192_175k_p2cd/EM_cifar10_pcn_20260909_06/checkpoint_177000.pt` | `--model_type=ffn_unet_mlp --unet_no_attention --unet_no_norm --unet_ws --mlp_head=flatten --residual_alpha=-1 --num_channels=192` | **5.0** | 3.5089 |

**Matched noise.** Every run drew its batch as the FID runner does: `torch.manual_seed(fid_seed*1000)`
with `fid_seed=1`, then `randn(128, 3, 32, 32)`. All four therefore start from the SAME x0, so panel i
of one grid and panel i of another are the same noise vector pushed through different models. The
caption may say "matched noise" but NOT "matched sampling time": the ws192 arm is integrated to
tau_s = 5.0 because that is where it is best; the three main arms are at 3.25.

**Why the whole batch is integrated, and shown.** torchsde draws the Brownian path over the entire
flattened (B, D) state, so generating fewer images directly would give different trajectories from a
prefix of the 128-batch: the batch size is part of the protocol. The figures show all 128, which also
removes the question of which subset was chosen.

## Reproduce

On a 4-GPU box with the repo, CIFAR-10 data and the four checkpoints in place (`/root/run_sample_grids.sh`
of 2026-09-20; one arm per GPU, ~16 min wall clock on A100s — the PCN arms dominate because every SDE
step runs a relaxation):

    cd experiments/cifar10_pcn
    COMMON="--use_ema --fid_seed=1 --batch_size=128 --dt_gibbs=0.01 --epsilon_max=0.01 \
            --time_cutoff=1.0 --grid_n=128"
    uv run python3 in_paper_diag_sample_grid.py --model_type=ffn_unet_vit \
      --resume_ckpt=<backprop postcd 147k> --fid_times=3.25 --grid_out=samples_backprop.npy $COMMON
    uv run python3 in_paper_diag_sample_grid.py --model_type=pcn_unet_vit --pcn_error_param \
      --param_grad_mode=ift --pcn_gamma=0.1 --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
      --resume_ckpt=<ift postcd 147k> --fid_times=3.25 --grid_out=samples_ift.npy $COMMON
    uv run python3 in_paper_diag_sample_grid.py --model_type=pcn_unet_vit --pcn_error_param \
      --param_grad_mode=ift --pcn_gamma=0.003 --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
      --resume_ckpt=<ep postcd 102k> --fid_times=3.25 --grid_out=samples_ep.npy $COMMON
    uv run python3 in_paper_diag_sample_grid.py --model_type=ffn_unet_mlp --unet_no_attention \
      --unet_no_norm --unet_ws --mlp_head=flatten --residual_alpha=-1 --num_channels=192 \
      --resume_ckpt=<ws192 postcd 177k> --fid_times=5.0 --grid_out=samples_ws192.npy $COMMON

then `uv run python3 make_figures.py` here. `param_grad_mode` is inert at inference (no
gradients are taken w.r.t. parameters), so the EP arm is sampled with the same code path as the
implicit-solve arm; what differs is the checkpoint and the relaxation gamma.

## Caveats worth keeping

- These are the phase-2 (post-CD) models. The training-time grids inside each run directory are raw
  (non-EMA) weights at tau_s = 1.0 and are NOT this protocol; do not mix them into the figure.
- A grid of 128 is not evidence about sample quality — the FIDs in
  `in_paper_fid_vs_steps_main_arms/` are. The figure's job is to show that the EP arm produces
  ordinary CIFAR-10 images, not to rank the arms by eye.
