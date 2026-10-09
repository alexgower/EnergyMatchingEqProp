# CIFAR-10 FID of the main arms (Table 5, phase-1 FID table)

Every 50k-sample FID behind the paper's CIFAR-10 comparison: the final (post-contrastive-divergence) models
of each arm, and the phase-1 models at 50k, 100k and 145k training steps.

- **Claim.** The three full-architecture arms (backpropagation, implicit solve, EP) track each other at every
  phase-1 checkpoint and finish within 0.1 FID of each other after the contrastive phase. The norm- and
  attention-free architecture reaches the same quality with more training.
- **Built by** `make_tables.py`, which reads the evaluation logs and writes the two LaTeX tables. No FID is
  typed in by hand except the previous published Energy Matching 3.34, which the builder flags as not measured here.

## What is in here

| file | what |
|---|---|
| `tables_section5.tex` | our rows of Table 5; the published reference rows above them are maintained in the paper |
| `tables_phase1_fid_vs_steps.tex` | the phase-1 table (FID at 50k, 100k, 145k steps) |
| `params_cache.json` | parameter counts summed from the checkpoints (49.99M and 78.38M); the builder needs it |
| `fid_<arm>_<checkpoint>_<weights>_<value>/` | one folder per evaluation: the evaluation log, with `fid_cifar10.INFO` linking to it |
| `tau_sweep/` | the sampling-time sweep that set τ_s = 3.25 for the post-CD models |

`uv run python3 make_tables.py` rebuilds both tables, reading only the evaluation logs filed in this folder. It finds each table cell by the
**checkpoint path recorded in the evaluation log**, then reads the FID from that log, so a cell can only come
from the run it names. A cell with no matching evaluation prints as `\todo{}`.

## Table 5: the final models

All 50k samples, EMA weights, seed 1, at each model's best sampling time τ_s.

| arm | params | training | FID | τ_s | record |
|---|---|---|---|---|---|
| backpropagation (our replication) | 50M | 145k + 2k CD | 3.54 | 3.25 | `fid_ffn_postcd147k_ema_tau3.25_seed1_3.5370/` (copy of the Sec. 4.1 record) |
| implicit solve | 50M | 145k + 2k CD | 3.64 | 3.25 | `fid_ift_postcd147k_ema_tau3.25_3.6351/` |
| EP | 50M | 100k + 2k CD | 3.64 | 3.25 | `fid_ep_postcd102k_ema_tau3.25_3.6401/` |
| norm/attention-free | 78M | 175k + 2k CD | 3.51 | 5.0 | `fid_ws192_postcd177k_ema_tau5.0_3.5089/` |

- **The norm/attention-free row has a longer budget on purpose.** The claim is that a hardware-plausible
  architecture *can* match. Its phase 1 also ran a different schedule: a cosine decay to zero at 145k, then a
  restart at constant 6×10⁻⁴ to 175k. At the matched 147k budget the same architecture reads 3.78 at τ_s = 4.5
  (`fid_ws192_postcd147k_ema_tau4.5_3.7755/`).
- **EP's contrastive phase starts from 100k, not 145k.** The EP run's 145k checkpoint does not survive the first
  contrastive updates (`appendix_ep145k_mechanism/`). The 100k checkpoint is no weaker than the other arms'
  145k ones (phase-1 table below), so the comparison stays like-for-like.
- **EP's phase-2 model is sampled with γ_code = 0.003**, the γ of its contrastive phase. Sampling it at the
  phase-1 γ_code = 0.1 gives the same FID (5.72 vs 5.70 on a 10k-sample check), so this is a solver setting,
  not a model change. Why phase 2 used different relaxation settings: `appendix_contrastive_phase/`.
- **3.54 is not "corrected" to 3.48.** The same backpropagation checkpoint, same τ_s and seed, reads 3.48 on a
  different batch geometry (4 ranks × batch 64 instead of 8 × 128). That is draw noise (see Resolution), and
  3.54 is the draw Sec. 4.1 quotes.

## Phase-1 table: FID against training steps

50k samples, EMA weights, τ_s = 1.0 (phase-1 models have had no contrastive phase, so the longer sampling
times do not apply).

| arm | 50k | 100k | 145k |
|---|---|---|---|
| backpropagation | 11.22 | 6.13 | 6.42 |
| implicit solve | 12.35 | 6.45 | 6.70 |
| EP | 12.60 | 6.23 | 6.87 |

- **All three arms improve by 5–6 FID from 50k to 100k, then get slightly worse by 145k.** The rise is small
  (0.25–0.65 FID) but consistent across arms and about 10× the seed spread. "No better, or slightly worse" is the
  right strength of claim.
- **Raw (non-EMA) FIDs** are filed alongside each EMA record and listed below. They are not comparable between
  arms at a fixed step, because EMA lags by its 10k-step horizon.

Every measured phase-1 cell, EMA / raw:

| step | backpropagation | implicit solve | EP | norm/attention-free |
|---|---|---|---|---|
| 50k | 11.22 / 12.97 | 12.35 / 13.24 | 12.60 / 11.03 | — |
| 100k | 6.13 / 10.27 | 6.45 / 13.69 | 6.23 / 11.20 | — |
| 145k | 6.42 / 11.99 | 6.70 / 11.95 | 6.87 / 12.37 | 14.25 / 14.45 |
| 175k | — | — | — | 13.10 / 14.74 |

The norm/attention-free architecture has no 50k or 100k evaluation, because its claim is about the endpoint
(training-schedule appendix).

The backpropagation 145k weights have four draws (6.39, 6.42, 6.46, 6.54, seeds 1–3 and unseeded); the table
quotes seed 1, the draw filed here. All four are filed in the Sec. 4.1 folder. The same weights sampled through the PCN, for Sec. 4.1, give values within 0.002 FID of each
draw; those belong to the Sec. 4.1 folder and are not used here.

## Protocol

Every number here, and every number the builder will accept, uses:

- 50,000 generated samples against the CIFAR-10 training set, torchmetrics `FrechetInceptionDistance(feature=2048)`
- EMA weights
- Euler–Heun SDE sampling (torchsde) with `dt_gibbs = 0.01`, `epsilon_max = 0.01`, `time_cutoff = 1.0`
- each arm under its **own** inference: autograd velocity for the feedforward arms, PCN relaxation for the
  implicit-solve and EP arms

10k- or 5k-sample screens are never mixed in. Commands for each arm are in `checkpoints_main_arms/EXPLANATION.md`.

## Resolution: about 0.1 FID

Repeat evaluations of the same checkpoint at the same settings:

| checkpoint | τ_s | draws | spread |
|---|---|---|---|
| backpropagation, post-CD 147k | 3.25 | 3.537 (8 ranks × 128) vs 3.481 (4 × 64), both seed 1 | 0.06 |
| norm/attention-free, post-CD 147k | 3.25 | 4.402 (unseeded) vs 4.280 (seed 1) | 0.12 |
| backpropagation, 145k | 1.0 | eight draws from 6.39 to 6.54 | 0.15 |

- **A fixed seed does not fix the draw.** `fid_seed` sets `torch.manual_seed(fid_seed·1000 + rank)`, but the
  noise is drawn per batch, so changing the batch size or GPU count changes the samples.
- **So differences below about 0.2 FID are not differences.** Two decimal places is the honest precision.

## Our scoring vs the published 3.34

The authors' own post-CD checkpoint, scored in our pipeline:

| scoring | FID at τ_s = 3.25 |
|---|---|
| Energy Matching, as published | 3.34 |
| their checkpoint, our pipeline, one 0→3.25 integration | 3.44 (`fid_authors_postcd147k_ema_tau3.25_3.4404/`) |
| their checkpoint, our pipeline, their chained τ ladder | 3.46 (`fid_authors_postcd147k_ema_ladder_tau1-3.25_3.4611/`) |
| our replication, our pipeline | 3.54 / 3.48 (two draws) |

- **Our pipeline matches theirs** in integrator, step sizes, sample count, EMA, real-image features, FID
  implementation and image conversion. It differs in batch size and GPU count, and their ladder integrates through
  ten restarts (1.0, 1.25, … 3.25) where ours integrates once.
- **The ladder does not explain the 0.1 gap.** Scoring their checkpoint through their exact ladder gives 3.46,
  against 3.44 for one segment, the same within noise. The remaining gap is the evaluation environment
  (library versions), not our replication.
- **Their phase-1 checkpoint** scores 6.67 at τ_s = 1.0 in our pipeline (`fid_authors_step145000_ema_6.6671/`).


## Where the training runs live

| arm | phase 1 | phase 2 |
|---|---|---|
| backpropagation | `results_cifar10_ffn/main/results_cifar10_stage0_4gpu_accum1024/` | `results_cifar10_ffn/main/results_cifar10_accum1024_p2cd/` |
| implicit solve | `results_cifar10_pcn/main/stage2_ift_main_g01_drop01/` | `.../stage2_ift_main_g01_drop01_p2cd_twin_fix/` |
| EP | `results_cifar10_pcn/main/stage3_ep_main_g01_drop01/` | `.../stage3_ep_main_g01_drop01_p2cd_EP100K_V2/` |
| norm/attention-free | `strip/ablate_ws_fullstrip192/` to 145k, `strip/ablate_ws192_warmrestart/` to 175k | `strip/ablate_ws192_175k_p2cd/` (177k), `strip/ablate_ws192_p2cd/` (147k) |

The final models are released as checkpoints; see `checkpoints_main_arms/EXPLANATION.md` for files, hashes and
reproduction commands.

## Notes for maintaining this folder

- **Only evaluations of runs the paper discusses are filed here.** Others are kept outside the release
  (`results_cifar10_pcn/investigations/`).
- **Filing a new evaluation:** copy its log folder in as `fid_<arm>_<checkpoint>_<weights>_<value>/`, keeping
  the per-host log and the `fid_cifar10.INFO` link. The builder scans only this folder, so every record a table
  uses must be filed here; no builder edit is needed.
- **γ is in code units** in every log: γ_code = 1000 × γ_V on CIFAR-10 (`--pcn_gamma=0.1` is the paper's 10⁻⁴).
- **The 50k evaluations of the backpropagation and implicit-solve 50k-step checkpoints** were copied here from
  `results_cifar10_pcn/fid_main/`, where they were first filed; `fid_ffn_step145000_ema_seed1_6.4242/` and
  `fid_ffn_postcd147k_ema_tau3.25_seed1_3.5370/` are copies of Sec. 4.1 records. Duplicates count once.
