# Velocity correspondence transferred to ImageNet-32 (appendix: imagenet-transfer, figure D)

History, superseded numbers, spreads and forensics: `CHANGELOG.md` (this folder).
Rule: a result lands here only when it changes a paper sentence or a figure.

**STATUS: ALL ROWS FINAL** (folded-clamp grid 34769635, upper-gamma extension
34799850, FID 2x2 complete 2026-09-04).

## 1. Claim

The law and its floors transfer to the authors' ImageNet-32 checkpoint with the
same one-sweep budget; the recipe's output clamp is absorbed into the top
prediction function, so no separate clamp term exists; at the operating point
the O(γ) bias is 5.2e-8 (float64), four orders of magnitude below the sampling
runs' TF32 floor — in float32 the same point reads 3.9e-7, which is that
arithmetic's floor rather than the bias; PCN inference reproduces the
feedforward FID.

## 2. What the paper quotes

| # | number (2 s.f.) | supports | source (file:line, column `rel_L2_err` unless stated) | status |
|---|---|---|---|---|
| L | ε ∝ γ over γ_V 1e-5 … 1e-3 (unclamped) | law transfers | `sweep_output.log:22-27` | final |
| L | fp32 floor 3.8e-7; TF32 floor 2.0e-4; TF32 merges onto the fp32 line at γ_V ≈ 1e-2 (7.1e-4 vs 6.8e-4 at γ_code 10; equal by 30), as on CIFAR-10 | floors transfer | `sweep_output.log:19`, `:53`; `sweep_upper_gamma.log:23` vs `:54` (γ_code 10), `:25` vs `:56` (30) | final |
| L | law holds to γ_V = 0.1 (ε = 7.0e-3, cos 0.99998) | no inference upper limit on ImageNet-32 either | `sweep_upper_gamma.log:26` (fp32), `:57` (TF32) | final |
| C | clamp c·tanh(V/c), c = 1e4, absorbed into the top prediction f_L' = c·tanh(αf_L/c)/α; the clamped curve follows the same law and shares the fp32 floor | one appendix sentence | `experiments/imagenet/config_multigpu_imagenet32.py` (`energy_clamp`); `sweep_output.log:85` (folded fp32, γ_code 1e-3 = 3.9e-7 = floor), `:89` (γ_code 0.1: 5.1e-6), `:130` (float64) | final |
| O | operating point γ_V = 1e-6, **O(γ) bias itself: ε = 5.2e-8** (float64, no floor in the way) — four orders of magnitude below the sampling runs' TF32 floor of 2.0e-4 | the number the paper quotes for the operating point | `sweep_output.log:126` (float64 clamped, γ_code 1e-3) | final |
| O | same operating point **measured in float32: ε = 3.9e-7**, which is the fp32 floor, not the O(γ) bias — float32 cannot resolve 5.2e-8; ~500× below the TF32 floor; cos = 1.000000000 | why the two numbers differ | `sweep_output.log:85` (fp32 clamped) vs `:53` (TF32); `:126` col `v_cos` | final |
| F | FID, backprop vs PCN, matched seed (quote to two decimals like the CIFAR table): warm-up τ 1.0 **8.33 vs 8.33** (8.3271 vs 8.3265, −0.0006); main τ 2.5 **6.75 vs 6.78** (6.7470 vs 6.7826, +0.036; authors report ≈ 6.6) | FID matches | `../in_paper_fid_calculation_train_backprop_infer_pcn_imagenet32/` (four named subfolders with the absl logs; originals in `results_cifar10_pcn/fid_evals/EM_cifar10_pcn_20260903_{04,04_ver1,05,07}/`) | final |
| D | 512 interpolants, single draw | disclosure (shared with CIFAR-10) | `sweep_slope_seeds.log`, `sweep_unseeded_check.log` | final |

(L = law/floors transfer; C = clamp; O = operating point; F = FID; D = disclosure.)

## 3. Figure

`figure_D_imagenet_transfer.png`: `cd <this folder> && uv run python3 make_figures.py`
(numpy + matplotlib; V-units, no title, no secondary axis). **Use `uv run`** — the
login node's system python3 carries a 2018-era matplotlib that rejects the
`text(c=...)` kwarg. Source `sweep_output.log`, four sections (unclamped fp32
`:1` | TF32 `:33` | clamped fp32 `:65` | clamped fp32+float64 `:97`), plus
`sweep_upper_gamma.log` (the same four passes at γ_code 1..100, job 34799850)
spliced in so the figure spans γ_V 1e-7..1e-1 like figure A, column `rel_L2_err`.

Four curves, each labelled with both its arithmetic and its clamp state
(relabelled 2026-09-05):

| curve | label | source |
|---|---|---|
| red | `float32 + TF32 kernels, clamped (sampling-run arithmetic)` | **spliced**, see below |
| blue | `float32, unclamped` | `sweep_output.log:1` + upper-γ |
| green | `float32, clamped` | `sweep_output.log:65` + upper-γ |
| purple | `float64, clamped` | `sweep_output.log:97` + upper-γ |

**The red curve is spliced, deliberately.** The FID samplers run PyTorch's
default math mode — matmul fp32, cuDNN TF32 — *with* the clamp, which is not the
both-flags unclamped pass in `sweep_output.log:33`. So for γ_V 1e-7..1e-3 the
curve is `sweep_fid_mathmode.log` (clamped, sampler mode, job 34871302): the
sampling-run arithmetic exactly. Above γ_V 1e-3 only the both-flags **unclamped**
sweep exists (`sweep_upper_gamma.log`) and it is spliced on to carry the merge
onto the fp32 line; the label's "clamped" is therefore exact over the floor and
the operating point, and the tail is unclamped. Two measurements justify this:
(i) the matmul flag changes nothing — the both-flags *clamped* sweep
(`sweep_tf32_clamped.log`) matches the sampler-mode clamped sweep in
`rel_L2_err` at all nine of its γ, so the cuDNN convolutions set the floor
alone; (ii) at the floor the clamp changes nothing (1.92e-4 vs 1.93e-4), while
above it the clamp only rescales the prefactor (0.8×), which is the region the
tail covers. `make_figures.py` prints the overlap comparison at every shared γ
(0.1–0.5% apart) each time it runs.

Guide line, operating-γ marker, the fp32/TF32 floor annotations and the axis
limits are all computed from the data. The clamped curves share the fp32 floor
and run parallel at 0.8× the unclamped prefactor; float64 has no floor and
reaches 1e-8. The CIFAR-10 reference curve was removed on 2026-09-03 (see the L
row: the prefactor comparison is now a number in the text).

## 4. Definitions

- **ε, ε∥, ε⊥, weighting, γ units**: exactly as in the CIFAR-10 folder's EXPLANATION
  (norm-weighted per-sample RMS ≡ batch ‖Δv‖/‖v_ff‖; γ_V = γ_code/1000, α = 1000).
- **Clamp**: the FFN's potential is V = c·tanh(α f_L/c) and the sampler differentiates
  it (`network_unet.py` EBViTModelWrapper `potential` 420-438, `velocity` 441-455). The
  PCN's top node predicts f_L' = c·tanh(α f_L/c)/α (`network_pcn.PCNEnergyModelBase._fold_top`,
  set from `energy_clamp` by the wrapper), so E = ½Σ‖r‖² + γ o' tilts the same quantity
  and v = −(1/γ)∇_x E → −∇_x V by the existing theorem. No factor is applied anywhere else.
- **Interpolants**: 512 = 128 × 4 real ImageNet-32 training images (official
  downsampled batches, 1,281,167 loaded) + Gaussian noise via the OT flow-matcher,
  seeded (seed 0); K_h = 1.
- **Checkpoint**: `checkpoints_authors/imagenet32x32_warm_up_640000.pt` (EMA, step
  640000; HF m1balcerak/energy_matching). Reference = exact autograd through the same
  PCN graph (the predict-scan already returns o').
- **Instrument**: `in_paper_diag_gamma_grid.py` with `DIAG_DATASET=imagenet32`
  (jobs 34769635 main grid, 34799850 upper-γ, 34871302 math mode).
- **FID jobs** (`--pcn_gamma=0.001`, `--energy_clamp=10000` on both samplers, 50k samples
  vs the full 1,281,167-image train set, EMA, matched `fid_seed=1`): 34743303 warm-up/backprop;
  34743304 warm-up/PCN; 34743305 main/backprop; 34743306 main/PCN. Commands in
  `../in_paper_fid_calculation_train_backprop_infer_pcn_imagenet32/EXPLANATION.md`.

## 5. Reproduce

Run from `experiments/cifar10_pcn/` with `IMAGENET32_PATH` pointing at the
official downsampled train batches (`$IMAGENET32_PATH/Imagenet32_train/`).

    export IMAGENET32_PATH=$PWD/../imagenet/data
    CK=$PWD/../../checkpoints_authors/imagenet32x32_warm_up_640000.pt
    COMMON="DIAG_CKPT=$CK DIAG_DATASET=imagenet32 DIAG_KH=1 DIAG_B=128 DIAG_NBATCH=4"

    # sweep_output.log — the figure's four passes
    env $COMMON uv run python3 -u in_paper_diag_gamma_grid.py                      # float32
    env $COMMON DIAG_TF32=1 uv run python3 -u in_paper_diag_gamma_grid.py          # TF32
    env $COMMON DIAG_CLAMP=10000 uv run python3 -u in_paper_diag_gamma_grid.py     # clamped
    env $COMMON DIAG_CLAMP=10000 DIAG_F64=1 DIAG_CHUNK=32 \
      uv run python3 -u in_paper_diag_gamma_grid.py                                # clamped f64

    # sweep_upper_gamma.log — the same four with
    UP="DIAG_GAMMAS=1.0,2.0,3.0,5.0,10.0,20.0,30.0,100.0"

    # sweep_tf32_clamped.log — both TF32 flags AND the clamp
    env $COMMON DIAG_TF32=1 DIAG_CLAMP=10000 uv run python3 -u in_paper_diag_gamma_grid.py

    # sweep_fid_mathmode.log — the samplers' own mode (matmul fp32, cuDNN TF32)
    env $COMMON DIAG_TF32=cudnn uv run python3 -u in_paper_diag_gamma_grid.py
    env $COMMON DIAG_TF32=cudnn DIAG_CLAMP=10000 uv run python3 -u in_paper_diag_gamma_grid.py

One A100 (80 GB) each; the float64 pass needs `DIAG_CHUNK=32` to fit.

## 6. Bonus findings not in the paper

- The ImageNet-32 prefactor is about half CIFAR-10's on matched draws — not in the paper (draw-dependent at the 2× level either way) → CHANGELOG 2026-09-02 (e), 2026-09-03 (q).
- Above the floor the clamped curve's prefactor is 0.75× the unclamped one at c = 1e4 and returns to 1.00× as c → ∞ (0.955× at 3e4, 1.000× at 1e6): the clamped potential is a genuinely different function, not a fold artefact → CHANGELOG 2026-09-03 (l).
- The clamped prefactor is stable across interpolant draws while the unclamped one is heavy-tailed → CHANGELOG 2026-09-02 (d), (e).
- The direction/magnitude split on ImageNet-32 is not robustly either way across draws → CHANGELOG 2026-09-02 (e).
- On CUDA the ViT head's `nn.TransformerEncoderLayer` returns a value under `no_grad` (its fused inference fast path, used by the relaxation) that differs by a per-sample ~3e-4 (o-units) from its value under autograd (the velocity readout), even in float64; on the CPU the two agree to 1e-16. This is the whole e_L = γ + δ "offset" and was the cause of the pre-fold "shelf". Velocities and parameter updates are blind to it (envelope theorem; parents identical across modes; head update cosine 0.9999) → CHANGELOG 2026-09-03 (h), (k), (l), 2026-09-04 (s), (v), (w).
- The FID samplers run PyTorch's default math mode (matmul fp32, cuDNN TF32); the velocity floor in that mode is 2.0e-4 on both datasets, flat in γ, so figure D's TF32 curve and label stand → CHANGELOG 2026-09-04 (u), (w).
