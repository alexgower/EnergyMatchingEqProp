# Velocity correspondence on CIFAR-10 (Sec 4.1, figures A/B/C)

History, superseded numbers, spreads and forensics: `CHANGELOG.md` (this folder).
Rule: a result lands here only when it changes a paper sentence or a figure.

## 1. Claim

The PCN relaxation velocity matches the feedforward autograd velocity with error
∝ γ down to an arithmetic floor; at the operating γ the bias sits an order of
magnitude below the floor of the production arithmetic, and the law holds to the
top of the measured range, so inference imposes no upper limit on γ.

## 2. What the paper quotes

| # | number (2 s.f.) | supports | source (file:line, column) |
|---|---|---|---|
| 1 | ε ≈ 0.13 γ_V, linear over γ_V 1e-6 … 1e-2 | error ∝ γ | `sweep_output.log:74-82` (f64), `sweep_upper_gamma.log` f64 rows, col `rel_L2_err` |
| 1 | cos(v_FFN, v_PCN) = 1.000000000 at γ_V = 1e-4; direction the larger error component | error ∝ γ, mostly in direction | `sweep_output.log:80`, cols `v_cos`, `nw_perp` > `nw_par` |
| 2 | fp32 floor 4.1e-7 (below γ_V ≈ 3e-7) | arithmetic floor; smaller γ buys nothing | `sweep_output.log:9` |
| 2 | TF32 floor 2.0e-4, flat over γ_V 1e-7 … 1e-3 | floor of the production arithmetic | `sweep_output.log:95` … `:103` |
| 2 | float64: no floor down to γ_V = 1e-7 (1.7e-8) | the floor is arithmetic, not algorithmic | `sweep_output.log:74` |
| 2 | curves coincide above γ_V ≈ 1e-2 | (TF32 = fp32 at γ_V 1e-2, 1.4e-3 vs 1.3e-3; not at 1e-3) | `sweep_upper_gamma.log:12`, `sweep_upper_gamma_tf32.log:12` (and `:8` each) |
| 3 | ε = 1.3e-5 at γ_V = 1e-4, 15× below the TF32 floor | operating-point bias an order of magnitude under the production floor | `sweep_output.log:14` (fp32) / `:80` (f64) vs `:101` |
| 4 | law holds to γ_V = 0.1: ε = 1.6e-2, cos 0.9999 | no inference upper limit; the ceiling comes from training | `sweep_upper_gamma.log:15` (fp32), `:43` (f64), `sweep_upper_gamma_tf32.log:15` |
| S | K_h-independent to within 1% over K_h ∈ {1,2,4,8,14} | a single sweep reaches equilibrium | `sweep_output.log:5-49`, col `rel_L2_err` |
| S | random init: K_h = 1 fails (cos 0.30 / 0.005), K_h = 2 converges, K_h ≥ 5 = feedforward-init values | not an initialisation artifact | `sweep_random_init.log:8-19` |
| D | 512 interpolants, single draw; the prefactor varies by up to ~2× across draws | disclosure sentence | `sweep_slope_seeds.log`, `sweep_unseeded_check.log` |

(#1-4 = the four claims of the section; S = supporting fact; D = disclosure.)

## 3. Figures

Regenerate all three: `cd <this folder> && python3 make_figures.py`
(numpy + matplotlib only; V-units, no titles, no secondary axis).

| figure | log(s) and section | columns |
|---|---|---|
| `figure_A_gamma_floors.png` | `sweep_output.log` main (fp32, K_h=1) / FLOAT64 / TF32 sections; `sweep_upper_gamma.log` (fp32, f64); `sweep_upper_gamma_tf32.log` | `rel_L2_err` |
| `figure_B_kh_flatness.png` | `sweep_output.log` main section, K_h ∈ {1,2,4,8,14}, γ_code 1.0 / 0.1 / 0.01 | `rel_L2_err` |
| `figure_C_direction_vs_magnitude.png` | `sweep_output.log` FLOAT64 section + `sweep_upper_gamma.log` f64 rows | `nw_rel`, `nw_par`, `nw_perp` (`FIGC_WEIGHTING=ps` → `ps_*`) |

## 4. Definitions

- **ε** = norm-weighted per-sample RMS of ‖v_i − v_ff,i‖/‖v_ff,i‖ (weights ‖v_ff,i‖²)
  ≡ ‖Δv‖/‖v_ff‖ over the pooled batch (log column `nw_rel`; the batch column
  `rel_L2_err` is the mean of four per-batch ratios and agrees to 1.7%).
  ε∥ = |‖v_i‖/‖v_ff,i‖ − 1|, ε⊥ = sqrt(ε_i² − ε∥,i²), same weighting; ε² = ε∥² + ε⊥² exactly.
- **γ units**: logs are in code units; γ_V = γ_code / 1000 on CIFAR-10 (α = output_scale = 1000;
  100 on MNIST). Recipe point γ_code 0.1 = γ_V 1e-4.
- **Interpolants**: 512 = 128 × 4 CIFAR-10 training images paired with Gaussian noise by the
  OT flow-matcher at random t; permutation, noise, t and OT pairing seeded (seed 0); the
  same construction in every pass and in the ImageNet-32 folder. K_h = 1 unless stated.
- **Checkpoint**: `../in_paper_fid_calculation_train_backprop_infer_pcn/checkpoint_warmup_145000.pt`
  (EMA; the backprop replication used in the FID table). Reference velocity = exact autograd
  through the same PCN graph. Float64 with the dtype-honest GroupNorm patch, chunked evaluation.
- **Instrument**: `experiments/cifar10_pcn/in_paper_diag_gamma_grid.py` (jobs 34758499
  main grid, 34424419/34450607 upper-γ, 34428065 random-init, 34871302 math mode).

## 5. Reproduce

Run from `experiments/cifar10_pcn/`; each command appends one table to stdout,
which is what the `sweep_*.log` files in this folder hold.

    CK=$PWD/../../results_cifar10_pcn/in_paper/equivalence_at_inference/\
    in_paper_fid_calculation_train_backprop_infer_pcn/checkpoint_warmup_145000.pt
    COMMON="DIAG_CKPT=$CK DIAG_B=128 DIAG_NBATCH=4"

    # sweep_output.log — the three passes of figures A/B/C
    env $COMMON DIAG_KH=1,2,4,8,14 uv run python3 -u in_paper_diag_gamma_grid.py
    env $COMMON DIAG_KH=1 DIAG_F64=1 DIAG_CHUNK=32 uv run python3 -u in_paper_diag_gamma_grid.py
    env $COMMON DIAG_KH=1 DIAG_TF32=1 uv run python3 -u in_paper_diag_gamma_grid.py

    # sweep_upper_gamma.log / sweep_upper_gamma_tf32.log — γ_code 1..100
    UP="DIAG_GAMMAS=1.0,2.0,3.0,5.0,10.0,20.0,30.0,100.0"
    env $COMMON DIAG_KH=1 $UP uv run python3 -u in_paper_diag_gamma_grid.py
    env $COMMON DIAG_KH=1 $UP DIAG_F64=1 DIAG_CHUNK=32 uv run python3 -u in_paper_diag_gamma_grid.py
    env $COMMON DIAG_KH=1 $UP DIAG_TF32=1 uv run python3 -u in_paper_diag_gamma_grid.py

    # sweep_random_init.log — the attractor check (footnote in the appendix)
    env $COMMON DIAG_INIT=random DIAG_KH=1,2,5,10,20,60 DIAG_GAMMAS=0.01,0.1 \
      uv run python3 -u in_paper_diag_gamma_grid.py

    # sweep_fid_mathmode.log — the FID samplers' own math mode (matmul fp32, cuDNN TF32)
    env $COMMON DIAG_KH=1 DIAG_TF32=cudnn uv run python3 -u in_paper_diag_gamma_grid.py

Each needs one A100 (80 GB); the float64 passes need `DIAG_CHUNK=32` to fit.
Other knobs: `DIAG_GAMMAS`, `DIAG_SEED` (interpolant draw), `DIAG_CLAMP`,
`DIAG_DATASET=imagenet32`, `DIAG_PERSAMPLE=0` for the batch-level pooling.

## 6. Bonus findings not in the paper

- **The physical (first-layer, local) readout equals the FFN velocity too**: reading dE/dx at fixed hiddens with the stored first-layer error — what a substrate computes, no Jacobian chain — matches v_FFN to 2.7e-7 (fp32 floor) at K_h = 1 for every γ, and to 4e-10 in float64. So "no backward pass" is a measured property of the one-sweep state, and velocity()'s chain readout is an equal-at-equilibrium convenience, not a hidden backprop. Caveat worth one appendix sentence: the readout must use the *stored* errors; re-forming e_0 from O(1) activations in float32 loses it to cancellation (0.6% at the operating γ) → CHANGELOG 2026-09-06 (j).
- The O(γ) prefactor is heavy-tailed across interpolant draws (bulk near 1.3e-4 per γ_code, tail to 2.2e-4) → CHANGELOG 2026-09-02 (e).
- On medians the prefactor is ~2× larger on CIFAR-10 than on ImageNet-32 → CHANGELOG 2026-09-02 (e); ImageNet folder.
- The direction/magnitude split is draw-sensitive (67-84% direction across draws) → CHANGELOG 2026-09-02 (e).
- The one-sweep (deadbeat) convergence behind figure B is specific to dt = 1 → CHANGELOG 2026-09-02 (d); SUMMARY_CHANGELOG §5.
