# Phase-2 checkpoints of the CIFAR-10 arms

The weights behind the paper's CIFAR-10 numbers and figures, copied here (2026-09-20) so every
in-paper claim resolves inside `in_paper/` rather than into the live training trees. These are
**copies, verified by md5 against their sources**; the originals stay where the runs wrote them,
because the evaluation logs record those paths and that provenance should not be rewritten.

| file | md5 | size | original (still in place) | step |
|---|---|---|---|---|
| `ift_postcd_147000.pt` | `0c40f9c102b7b8a2ef3b8df834e66f33` | 764 MB | `results_cifar10_pcn/main/stage2_ift_main_g01_drop01_p2cd_twin_fix/EM_cifar10_pcn_20260916_13/checkpoint_147000.pt` | 145k + 2k CD |
| `ep_postcd_102000.pt` | `6dc09bad8db4445fd2ee2cc7f732acd4` | 764 MB | `results_cifar10_pcn/main/stage3_ep_main_g01_drop01_p2cd_EP100K_V2/EM_cifar10_pcn_20260918_01/checkpoint_102000.pt` | 100k + 2k CD |
| `ws192_postcd_177000.pt` | `baa7c164035499592cbafbca43f4b3e8` | 1.2 GB | `results_cifar10_pcn/strip/ablate_ws192_175k_p2cd/EM_cifar10_pcn_20260909_06/checkpoint_177000.pt` | 175k + 2k CD |

The fourth arm's phase-2 weights are **not** duplicated here: backpropagation's
`checkpoint_postcd_147000.pt` already lives inside `in_paper/` at
`equivalence_at_inference/in_paper_fid_calculation_train_backprop_infer_pcn/`, which is the claim that
owns it (that folder also holds its phase-1 `checkpoint_warmup_145000.pt`).

Each file is a training checkpoint dict: `net_model`, `ema_model`, `sched`, `optim`, `step`. Every
paper number uses `ema_model`.

## What uses them

- `in_paper/in_paper_fid_vs_steps_main_arms/` — the report-protocol FIDs (`3.6351`, `3.6401`,
  `3.5089` respectively) and `tables_section5.tex`. Note `make_tables.py` resolves each table cell by
  the checkpoint path **recorded in the evaluation log**, i.e. the original path, so copying here
  changes nothing in the table build.
- `in_paper/sample_grids_main_arms/` — the appendix sample figures.

## Reproduce a number from one of these

    uv run torchrun --standalone --nproc_per_node=4 \
      experiments/cifar10_pcn/fid_cifar_heun_multigpu.py \
      --resume_ckpt=<file above> --use_ema --fid_seed=1 --fid_n_samples=50000 \
      --batch_size=128 --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0 \
      <arm's inference flags and --fid_times, from sample_grids_main_arms/EXPLANATION.md>

The per-arm inference flags (model type, relaxation gamma, K_h, T_free, n_CG, and the optimal
tau_s: 3.25 for the implicit-solve and EP arms, 5.0 for ws192) are tabulated in
`in_paper/sample_grids_main_arms/EXPLANATION.md`.
