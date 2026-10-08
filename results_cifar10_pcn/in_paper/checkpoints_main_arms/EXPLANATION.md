# Trained checkpoints of the CIFAR-10 arms (Sec. 5, App. samples)

The final weights behind the paper's CIFAR-10 FIDs and sample figures.


For personal use:
- **In this folder:** checkpoints from the implicit-solve, EP and norm/attention-free models, each after its contrastive phase (i.e. Energy Matching Phase 2).
- **Elsewhere:** the backpropagation model (replicating usual Energy Matching) lives in the Sec. 4.1 folder,
  `equivalence_at_inference/in_paper_fid_calculation_train_backprop_infer_pcn/`.


For replication:
- **Not in git:** note the `.pt` checkpoint files are too large so download them from HuggingFace instead (will add link soon)

## The files

| file | arm | trained for | size | md5 |
|---|---|---|---|---|
| `ift_postcd_147000.pt` | implicit solve | 145k phase-1 + 2k contrastive steps | 764 MB | `0c40f9c102b7b8a2ef3b8df834e66f33` |
| `ep_postcd_102000.pt` | EP | 100k phase-1 + 2k contrastive steps | 764 MB | `6dc09bad8db4445fd2ee2cc7f732acd4` |
| `ws192_postcd_177000.pt` | norm/attention-free | 175k phase-1 + 2k contrastive steps | 1.2 GB | `baa7c164035499592cbafbca43f4b3e8` |

- Each file is a training checkpoint dict: `net_model`, `ema_model`, `sched`, `optim`, `step`.
- **Every paper number uses `ema_model`**, so always pass `--use_ema`.
- The two PCN arms (implicit solve, EP) are in PCN layout and load into `PCNVelocityWrapper`.
- The norm/attention-free arm is a feedforward model and needs the architecture flags in the command below.

## What each one produced

| arm | FID (50k samples, EMA, seed 1) | sampling time τ_s | used in |
|---|---|---|---|
| implicit solve | 3.6351 | 3.25 | Table 5, App. samples |
| EP | 3.6401 | 3.25 | Table 5, App. samples |
| norm/attention-free | 3.5089 | 5.0 | Table 5, App. samples |
| backpropagation (Sec. 4.1 folder) | 3.5370 | 3.25 | Table 5, Sec. 4.1, App. samples |

- The evaluation logs behind these FIDs are in `fid_vs_steps_main_arms/`.
- The sample grids drawn from these weights are in `sample_grids_main_arms/`.

## Reproducing an FID

All arms use the same protocol: 50k samples, EMA weights, `fid_seed=1`, Euler–Heun with
`dt_gibbs=0.01`, `epsilon_max=0.01`, `time_cutoff=1.0`, batch 128, 4 GPUs.

- **PCN arms** sample by relaxation with `experiments/cifar10_pcn/fid_cifar_heun_multigpu.py`
  (about 14 h each on 4 A100s).
- **Feedforward arms** sample by autograd with `experiments/cifar10/fid_cifar_heun_multigpu.py`
  (about 2 h).

```bash
COMMON="--use_ema --fid_seed=1 --batch_size=128 --num_workers=4 --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0"
D=results_cifar10_pcn/in_paper/checkpoints_main_arms

# implicit solve -> 3.6351
uv run torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/fid_cifar_heun_multigpu.py \
  --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
  --pcn_gamma=0.1 --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
  --resume_ckpt=$D/ift_postcd_147000.pt --fid_n_samples=50000 --fid_times=3.25 $COMMON

# EP -> 3.6401 (same command, this arm's gamma)
uv run torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/fid_cifar_heun_multigpu.py \
  --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
  --pcn_gamma=0.003 --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
  --resume_ckpt=$D/ep_postcd_102000.pt --fid_n_samples=50000 --fid_times=3.25 $COMMON

# norm/attention-free -> 3.5089
uv run torchrun --standalone --nproc_per_node=4 experiments/cifar10/fid_cifar_heun_multigpu.py \
  --model_type=ffn_unet_mlp --unet_no_attention --unet_no_norm --unet_ws --mlp_head=flatten \
  --residual_alpha=-1 --num_channels=192 \
  --resume_ckpt=$D/ws192_postcd_177000.pt --n_samples=50000 --fid_times=5.0 $COMMON
```

Things to know when running these:

- **γ is in code units.** `--pcn_gamma=0.1` is the paper's γ = 10⁻⁴, and `0.003` is 3×10⁻⁶
  (paper γ = code γ / 1000 on CIFAR-10).
- **`param_grad_mode` does nothing at inference**, so both PCN arms run the same code path. Only the
  checkpoint and γ differ.
- **Expect small differences between draws.** Repeat evaluations of one checkpoint vary by a few
  hundredths of an FID point; the backpropagation weights have four filed draws spanning 6.39–6.54 at τ_s = 1.0.

## Full release manifest

Seven checkpoints in total (5.6 GB): each arm's final model, plus the phase-1 checkpoint it was
polished from. The two phase-1 files are not stored in this folder; the release host (HuggingFace) carries them
under the names below.

| arm | phase | file | md5 | FID (50k, EMA) |
|---|---|---|---|---|
| backpropagation | 1 | `checkpoint_warmup_145000.pt` (Sec. 4.1 folder) | `e26c05d49e1ca18e25729fcee73c60c6` | 6.4242 (τ_s 1.0) |
| backpropagation | 2 | `checkpoint_postcd_147000.pt` (Sec. 4.1 folder) | `ff66ba4e68c8fd166a7e06541a920955` | 3.5370 (τ_s 3.25) |
| implicit solve | 1 | `ift_phase1_145000.pt` | `2281fd8461c7e20d46bda41202bad063` | 6.7040 (τ_s 1.0) |
| implicit solve | 2 | `ift_postcd_147000.pt` | `0c40f9c102b7b8a2ef3b8df834e66f33` | 3.6351 (τ_s 3.25) |
| EP | 1 | `ep_phase1_100000.pt` | `ed9305abc0a21ddd34fa170604836283` | 6.2336 (τ_s 1.0) |
| EP | 2 | `ep_postcd_102000.pt` | `6dc09bad8db4445fd2ee2cc7f732acd4` | 3.6401 (τ_s 3.25) |
| norm/attention-free | 2 | `ws192_postcd_177000.pt` | `baa7c164035499592cbafbca43f4b3e8` | 3.5089 (τ_s 5.0) |

- **EP's phase-1 file is its 100k checkpoint**, because that is where its contrastive phase starts
  (Sec. 5). The same run's 145k checkpoint, discussed in App. experimental-implementation, is also
  released as `ep_phase1_145000.pt`.
- **The norm/attention-free arm has no phase-1 row.** Its phase 1 ran a different schedule, to 175k
  steps (see `fid_vs_steps_main_arms/EXPLANATION.md`), so it is not comparable with the others.

## Caveats

- **Phase-1 and phase-2 FIDs are not comparable.** Phase-1 numbers are at τ_s = 1.0; phase-2 numbers are
  at each arm's best τ_s (3.25, or 5.0 for the norm/attention-free arm; see
  `fid_vs_steps_main_arms/tau_sweep/`).
- **The EP model's sampling γ doesn't matter.** It was trained and evaluated with γ = 3×10⁻⁶, but
  sampling it at the phase-1 γ = 10⁻⁴ gives the same FID (5.72 vs 5.70 on a 10k-sample check).
