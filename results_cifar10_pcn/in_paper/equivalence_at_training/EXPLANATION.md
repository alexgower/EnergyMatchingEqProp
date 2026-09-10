# Equivalence at training — parameter updates vs true backprop (Sec 4.2)

Paper objects: **figures H and I**, **Table 4** (γ×β), **the budget-axes table**.
Everything here is `cos(estimator update, true FFN backprop gradient)` — never
estimator against estimator.

`CHANGELOG.md` (this folder, not pushed) holds the history: superseded numbers,
retired claims and how each was found. This file holds only what is currently
true and where it came from.

## 1. Claim

For the same weights and the same batch, the EP update and the IFT update on the
PCN both point where the backprop gradient of the Energy Matching loss points
(cos ≈ 1). IFT is exact wherever it was measured; EP is exact inside a window in
(γ, β) whose lower edge is float32 cancellation in the finite difference and
whose upper edge belongs to the nudge. EP additionally needs a converged free
phase. The nudged-phase budget and K_h cost nothing, and the linearised nudge
widens the usable β range by three decades over the quadratic one.

**Measurement.** CIFAR-10 replication warm-up checkpoint
`../equivalence_at_inference/in_paper_fid_calculation_train_backprop_infer_pcn/checkpoint_warmup_145000.pt`,
**live `net_model` weights** (the ones training takes gradients at). Reference =
exact autograd of the Energy Matching loss through `EBViTModelWrapper`, compared
to the PCN update via the FFN→PCN key remap. Operating budget K_h = 1,
T_free = 14, T_nudge = 6, β = 30, linear three-phase nudge, dt = 1; IFT K_h = 2,
n_CG = 10. **512 seeded interpolants** unless a row says otherwise: float32
128 × 4, float64 32 × 16 (float64 does not fit at B = 128).

## 2. What the paper quotes

γ below is γ_V; γ_code = 1000·γ_V on CIFAR-10, so the operating γ_code 0.1 = γ_V 1e-4.

| number | supports | source |
|---|---|---|
| IFT = 1.0000 at every γ from 1e-7 to 1e-2, 0.9976 at 1e-1 | "the correspondence is not the constraint"; Table 4 control row | `sweep_512.log` ("figure G" block), col `cos(ift,ffn)` |
| IFT = 1.0000 at n_CG ∈ {1,2,3,5,10,20,50}, at γ 1e-5 / 1e-4 / 3e-4 | budget-axes table, left column | `sweep_appendix_512.log` |
| **T_free shape**: in every cell that reaches full fidelity the cosine is ≤ −0.94 at T_free ≤ 6, ≥ 0.9 at 8, and ≥ 0.99 by 10. Tightest cells: −0.9487, **0.9141**, and 0.99 first reached at exactly **10** (both at β = 3) | the whole T_free paragraph, main text and appendix | `tfree_collation.txt` (every T_free sweep in one table, with per-row sources) |
| the three cells that plateau below 0.99 (γ 1e-2 at β ≥ 30; γ 1e-1) rise from the same transition, reading 0.77 / 0.70 / 0.25 at T_free = 8 | the appendix's exception clause | `tfree_collation.txt` |
| at γ 1e-5 the cosine turns positive but stalls at **0.544**; at 5e-6 and 1e-6 it never turns positive out to T_free = 30 | "no transition to full fidelity to locate" | `tfree_collation.txt` |
| **γ×β window** — the 6×6 grid, IFT-float32 control, EP-float64 control | Table 4 in full | `sweep_gamma_beta_table.log` (grid), `sweep_512.log` (IFT row), `sweep_512_f64.log` (float64 row) |
| float64 plateau height across γ: 1.0000 at 1e-6…1e-5, 0.9999 at 5e-5–1e-4, 0.9993 at 3e-4, 0.9982 at 1e-3, then **0.942 at 1e-2, 0.658 at 1e-1** | float64's role in the section — a same-draw comparison across γ | `sweep_tfree_f64.log` + `_rest` (job 35100884) |
| linear nudge, float32: −0.004 at β 0.01, 0.9789 at 1, ≥ 0.99 from β = 2 (0.9923) to 1e3 (0.9970), 0.9687 at 1e4 | figure I; "three decades" | `sweep_figure_I_beta.log`, `sweep_beta_high.log` |
| linear nudge, float64: 1.0000 from β 0.01 to 10, 0.9994 at 100, 0.9914 at 1e3, 0.8871 at 1e4 | the low-β floor is arithmetic; the high-β roll-off is not | `sweep_figure_I_f64_full.log` (job 35141419) |
| quadratic nudge: 0.9962 at β 1, 0.9494 at 2 in float32; **0.9997 and 0.9120 in float64** — the collapse is at the same β in both arithmetics | "usable at β = 1 and nowhere above"; the ceiling is not arithmetic | `sweep_figure_I_beta.log`, `sweep_figure_I_f64_full.log` |
| backprop reference norm ‖g_BP‖ = **0.3232**; quadratic update norm reaches **849** at β = 5 | the appendix's divergence sentence | `sweep_512.log` line `FFN ref: \|g_ff\| mean`; `sweep_figure_I_beta.log`. `make_tables.py` prints the reference norm on every run |
| shrinking the relaxation step makes the quadratic collapse **worse**: at β = 3 the norm goes 38.99 (dt 1) → 523.6 (0.5) → 14931 (0.1), a factor 383 | "a property of the nudge form, not the solver" | `sweep_quad_beta_dt.log` (job 35032779) — **32 interpolants**, so its β = 5 norm is 869, not the 512 draw's 849; do not mix the two in one sentence |
| **float64 does not rescue the upper edge**, seed-replicated on paired draws: across four seeds γ 1e-2 gives f32 0.983–0.988 vs f64 0.939–0.964, and γ 1e-1 gives f32 0.829–0.852 vs f64 0.658–0.737 | "float64 does not rescue that edge" in the main text — Table 4's control row is the quoted number, this is its replication | `sweep_upper_edge_seeds.log` (job 35149424). The *sign* of the gap is real within this instrument but unexplained; do not build an argument on it |
| T_nudge flat from 1 (0.9998 → 0.9999); K_h flat 1 → 5 (0.9999) | budget-axes table, centre and right | `sweep_appendix_512.log` |

**Two warnings that prevent a specific misreading of the published objects:**

- **Never quote a crossing budget.** The sign at T_free = 7 flips with the
  interpolant draw (−0.611 in the 512 draw, +0.439 in the 128 draw, same γ, β
  and instrument). Quote the shape above. → CHANGELOG (k), (m).
- **Figures H and I: the float64 arms are a different draw** (32 × 16 against
  float32's 128 × 4, because float64 does not fit at B = 128). Compare within a
  panel, never across one. → CHANGELOG (m).

## 3. Figures and tables

Two figures and two tables; `make_figures.py` has one code path per figure and
`make_tables.py` builds both tables. Nothing else here is a paper object.

- `figure_H_gamma_x_tfree.png` — cos vs T_free, one line per γ, float32 | float64
  panels, operating T_free = 14 marked. Six of the nine measured γ are plotted;
  the other three lie on top of neighbours (reason in the builder). Sources:
  `sweep_512.log` ("figure H" block), `sweep_tfree_extra_gammas.log`,
  `sweep_tfree_high_gamma.log` (float32); `sweep_tfree_f64.log` + `_rest` (float64).
- `figure_I_beta.png` — cos vs β at γ 1e-4, four curves (linear and quadratic,
  each in both arithmetics), β from 0.01 to 1e4. Read off it: the low-β float32
  floor, and the quadratic collapsing at the same β in both arithmetics.
  Sources: `sweep_figure_I_beta.log`, `sweep_beta_high.log`,
  `sweep_figure_I_f64_full.log`.
- `tables_section42.tex` — both tables as **caption-free** `tabular` source with
  `\cellcolor{gray!15}` below 0.99 (needs `colortbl`); captions are hand-written
  in the paper so the file can be regenerated and diffed without prose churn.
  Table 4 = the 6×6 γ×β grid plus two control rows; budget axes = n_CG, T_nudge,
  K_h at γ 1e-4, rows 1/2/3/5/10/20/50.
- `tfree_collation.txt` — every (γ, β) pair with a T_free sweep in one table,
  current and superseded draws separated, source log per row. Built by
  `collate_tfree.py`.

**Build:** `uv run python3 make_figures.py`, `uv run python3 make_tables.py`,
`uv run python3 collate_tfree.py > tfree_collation.txt` (from this folder).

**Measured, filed, deliberately not published.** Logs stay here; the builders do
not emit them. Rebuild from the logs if a reviewer asks. Reasons in CHANGELOG (j).
- β sweep with update norms (`sweep_figure_I_beta.log`, `sweep_beta_high.log`) —
  its cosines duplicate figure I; the appendix quotes only its 849 and 0.3232.
- γ 1e-6 and 1e-7 columns for Table 4 (`sweep_table4_lowgamma.log`, job 35142571) —
  below the floor the β-dependence reverses, which would read as an inconsistency
  in a table that stops at 1e-5.
- **Do not reuse the letter G.** A figure G existed and was deleted 2026-09-09;
  CHANGELOG (n) records what it plotted and why it went.

## 4. Definitions that must survive editing

- **Instrument and pooling.** Every object must name its instrument; no figure or
  table may mix them.

  | object | instrument | pooling | batch |
  |---|---|---|---|
  | figure H, budget-axes table, γ sweeps | `in_paper_diag_ep_vs_ift_vs_ffn.py` | ONE cosine over the concatenated shared tensors (FFN→PCN remap) | f32 128×4, f64 32×16 |
  | figure I, Table 4 | `in_paper_diag_sweep_relax_budget.py` | same pooled cosine, same remap | f32 128×4, f64 32×16 |

  Both share `remap_cos`, so their pooling agrees — but they build their
  interpolants differently (`in_paper_diag_ep_vs_ift_vs_ffn` draws in float64 and
  casts, so its two arithmetics are paired; `in_paper_diag_sweep_relax_budget`
  draws in float32), so at the same nominal settings they measure **different
  draws**. **Batch geometry is part of the instrument**: an arm at a different
  batch is a different draw even inside the same script. A third instrument,
  `diag_pcn_unet_fidelity.py`, pools differently again (mean over batches of a
  cosine over ALL parameters) and is not part of this section — nothing here is
  built from it, and its numbers must never be quoted beside these two.
- **Interpolants**: seeded OT flow-matching pairs (torch seed 0 *and* the numpy
  global RNG torchcfm's pairing draws from).
- **EP and IFT are different estimators of the same quantity** and fail in
  correlated ways; agreeing with each other proves nothing. `cos(ep,ift)` is
  logged and never quoted.
- **Units**: γ_code = 1000·γ_V on CIFAR-10 (α is dataset-dependent). Operating
  point γ_code 0.1 = γ_V 1e-4.
- **The float32 floor is not a budget problem.** Neither more free-phase sweeps
  (T_free to 120, `sweep_floor_tfree_probe.log`) nor larger β (to 3000,
  `sweep_floor_beta_probe.log`) lift it.
- **Hardware framing (open, not resolved).** A GPU forms the finite difference of
  two nudged states in IEEE float32; the operating point sits one decade above
  that floor. An analog substrate would measure two equilibria and subtract, so
  its limit is its measurement noise floor, not IEEE arithmetic. Whether
  differential equilibrium measurement reaches comparable relative precision is
  an open question this paper does not settle.
- **The training ceiling is not a per-step effect.** The storm arm (γ 3e-4, IFT)
  trained cleanly to 60k and exploded in ~40 steps at 60,670, yet the per-step
  cosine at that γ is 0.9998 (EP) and 1.0000 (IFT), and on healthy weights the
  IFT Hessian is the identity to 4e-4 even at γ 1e-3 (zero negative curvature).
  Indefiniteness (λ_min = −2.0 at γ 1e-4) appears only on the post-storm
  checkpoint. So state the upper limit as *observed in training*, never as a
  per-step fidelity failure. Its T_free = 5 / K_h = 1 / n_CG = 3 are all
  measured-exact for IFT, so those weights were not shaped by an under-converged
  estimator. `sweep_gamma_grid.log`, `sweep_ncg.log`, `sweep_lambda_min_vs_gamma.log`.

## 5. Standing rule: single draws do not support point claims near a sign change

Where the estimator is accurate the interpolant draw is invisible — the clean
region of Table 4 reproduces to four decimals across draws *and* instruments.
Where the estimator is near a sign change or below its arithmetic floor, a single
draw supports no point claim: not a value, not an ordering, not a threshold
budget, not a sign. In those regimes quote a shape, a bound, or a range over
draws, and name the draw and the instrument.

Five contradictions in this section came from ignoring this; all five sat at the
floor, in its transition, at the upper edge, or at the zero crossing, and none
in the clean region (CHANGELOG (k), (m)). One of them — two instruments reading
0.850 and 0.337 at the same γ 1e-1 cell — is **not** explained by the rule
(four draws through one instrument were stable at 0.846 ± 0.023) and remains
unattributed, so a draw difference is the first thing to check and not the last.

## 6. Reproduce

From `experiments/cifar10_pcn/`, `CK` = the replication warm-up checkpoint above:

    R="uv run python3 -u in_paper_diag_ep_vs_ift_vs_ffn.py $CK"
    P="DIAG_WEIGHTS=net_model DIAG_KH_EP=1 DIAG_TFREE=14 DIAG_TNUDGE=6 DIAG_KH_IFT=2 DIAG_NCG=10"
    # gamma sweep -> Table 4's two control rows. Paired arithmetics.
    for F in 0 1; do env $P DIAG_FP64=$F DIAG_B=128 DIAG_NBATCH=4 \
      DIAG_GAMMAS=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100 $R; done
    # figure H: gamma x T_free (float64 needs DIAG_B=32 DIAG_NBATCH=16)
    for G in 0.001 0.01 0.1 0.3 1.0 10.0 100.0; do for TF in 5 6 7 8 9 10 12 14 20 30; do
      env $P DIAG_GAMMAS=$G DIAG_TFREE=$TF DIAG_B=128 DIAG_NBATCH=4 $R; done; done
    # budget axes (T_nudge, K_h at gamma_code 0.1) and n_CG (DIAG_NCG); figure I / Table 4:
    env DIAG_CKPT=$CK DIAG_WEIGHTS=net_model DIAG_GAMMA=0.1 DIAG_NUDGES=linear,quadratic \
      DIAG_BUDGETS="1:14:6" DIAG_BETAS=0.01,0.03,0.1,0.3,1,2,3,5,10,30,100 \
      uv run python3 -u in_paper_diag_sweep_relax_budget.py $CK
    # Hessian conditioning vs gamma on two checkpoints:
    DIAG_CK_HEALTHY=$CK DIAG_CK_BROKEN=<storm checkpoint_65000.pt> \
      uv run python3 -u diag_lambda_min_vs_gamma.py

One A100 each; every run is single-batch gradient computation, minutes not hours.
