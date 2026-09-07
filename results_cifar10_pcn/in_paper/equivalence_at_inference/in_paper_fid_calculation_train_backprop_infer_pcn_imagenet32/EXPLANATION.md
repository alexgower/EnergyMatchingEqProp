# Train with backprop, infer on the PCN — ImageNet-32 (appendix table rows)

## 1. Claim

Loading the Energy Matching authors' released ImageNet-32 checkpoints into the
PCN unchanged and sampling by relaxation alone reproduces the feedforward FID.

## 2. What the paper quotes

| checkpoint | τ_s | backprop | PCN relaxation | source |
|---|---|---|---|---|
| warm-up (Algorithm 1, 640k) | 1.0 | 8.33 | 8.33 | `fid_backprop_warmup640k_tau1_EMA_seed1_8.3271/`, `fid_pcn_warmup640k_tau1_EMA_seed1_8.3265/` |
| main (+ Algorithm 2 CD, 641k) | 2.5 | 6.75 | 6.78 | `fid_backprop_main641k_tau2.5_EMA_seed1_6.7470/`, `fid_pcn_main641k_tau2.5_EMA_seed1_6.7826/` |

Two decimals, as in the CIFAR-10 table. Paired differences −0.0006 and +0.036.
The authors report ≈ 6.6 at T = 2.5 for the main checkpoint.

## 3. Figures (NOT cited in the paper)

`figure_E_imagenet32_pcn_samples.png` (4 × 8, all 32 saved samples) and
`figure_F_imagenet32_ffn_vs_pcn_samples.png` (two 2 × 8 blocks, matched seeds),
built by `uv run python3 make_figures.py` in this folder from
`samples_main641k_tau2.5_seed1_{backprop,pcn}.npy` (job 34931831: main
checkpoint, τ_s = 2.5, seed 1, EMA, clamp 1e4, γ_V = 1e-6, batch 128, first 32
of the seeded batch in order).

**Superseded for the paper 2026-09-05**: the sample figures moved to CIFAR-10
(`../in_paper_fid_calculation_train_backprop_infer_pcn/`, §3 there) because
ImageNet-32 samples read poorly at 32×32 — as does the real data, which was
checked. Kept here as the ImageNet-32 record; still usable as an appendix panel
if the transfer needs a visual. Matched-seed agreement over the 16 paired:
mean |Δpixel| 10.0/255, per-image correlation 0.949–0.992 (median 0.969) versus
0.084 for unrelated samples.

## 4. Definitions

- **Protocol** (= the authors' `fid_imagenet_heun_multigpu.py`): 50k samples vs the
  full 1,281,167-image ImageNet-32 train set, EMA weights, Heun dt 0.01, ε_max 0.01,
  time cutoff 1.0; TF32 kernels; matched sampling seed `--fid_seed=1` (per rank
  `1000 + rank`) so the two samplers see the same initial noise.
- **Samplers**: backprop = `--model_type=ffn_unet_vit` autograd velocity; PCN =
  `--model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift --ffn_checkpoint_into_pcn
  --pcn_gamma=0.001 --K_h=1 --T_free=14`, i.e. every drift evaluation by one relaxation
  sweep at γ_V = 10⁻⁶. Both pass `--energy_clamp=10000`; the PCN absorbs it into its
  top prediction function (see `../in_paper_imagenet32_velocity_correspondence/`).
- **Checkpoints**: `checkpoints_authors/imagenet32x32_warm_up_640000.pt`,
  `checkpoints_authors/imagenet32x32_main_training_641000.pt` (HF m1balcerak/energy_matching).
- **Jobs**: 34743303 / 34743304 (warm-up backprop / PCN), 34743305 / 34743306 (main),
  each 1 node x 4 GPUs. Originals:
  `results_cifar10_pcn/fid_evals/EM_cifar10_pcn_20260903_{04,04_ver1,05,07}/`.

## 5. Reproduce

From the repo root, with `IMAGENET32_PATH` pointing at the official downsampled
train batches. `--real_dataset=imagenet32` swaps the FID reference set; the PCN
and feedforward arms differ only in `MODELARGS`.

    export IMAGENET32_PATH=$PWD/experiments/imagenet/data
    CKPT=checkpoints_authors/imagenet32x32_main_training_641000.pt   # TAU=2.5
    # warm-up row: imagenet32x32_warm_up_640000.pt with TAU=1.0
    MODELARGS="--model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
               --ffn_checkpoint_into_pcn --pcn_gamma=0.001 --K_h=1 --pcn_dt=1.0"
    # feedforward arm: MODELARGS="--model_type=ffn_unet_vit"
    uv run torchrun --standalone --nproc_per_node=4 \
      experiments/cifar10_pcn/fid_cifar_heun_multigpu.py \
      $MODELARGS --energy_clamp=10000 --real_dataset=imagenet32 \
      --resume_ckpt=$CKPT --use_ema --fid_times=2.5 --fid_n_samples=50000 --fid_seed=1 \
      --batch_size=128 --num_workers=4 --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0

The PCN arm at τ=2.5 takes ~17 h on 4 GPUs (the feedforward arm ~2.5 h); split
across 2 nodes with `--nnodes=2` and a c10d rendezvous to halve that.

## 6. Bonus findings not in the paper

- The main-checkpoint pair differs by +0.036 (0.5%) against −0.0006 at warm-up: a
  small real sampler difference at 2.5× the integration length on the CD-sharpened
  landscape → `../in_paper_imagenet32_velocity_correspondence/CHANGELOG.md` 2026-09-04 (t).
