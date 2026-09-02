# Inference correspondence: O(gamma) bias vs precision floors  (Sec 4.1 figure)

**Claim.** The PCN relaxation velocity equals the feedforward v = -scale*dF/dx
up to an O(gamma) magnitude bias; the achievable error is set by arithmetic
precision, not by relaxation budget.

**Units.** Unless a line says `gamma_V`, every gamma in this file is in CODE
units (the units of the sweep logs here): gamma_V = gamma_code / 1000 on
CIFAR-10. Recipe point = gamma_code 0.1 = gamma_V 1e-4. Full table below.

**Results** (replication checkpoint, 4x128 CFM batches, K_h=1):
- rel L2 error linear in gamma over 2.5 decades: 1.6e-5 @0.1, 4.8e-5 @0.3, 1.6e-4 @1.0.
- Floors: TF32 ~2e-4 (flat in gamma; the regime ALL production runs used),
  float32 ~4e-7 (bites below gamma~0.003), float64: NO floor observed — the smallest measured point (1.8e-8 @ gamma=1e-4) still lies on the O(gamma) line.
- At the recipe gamma=0.1 the O(gamma) bias sits BELOW the TF32 floor.
- cos = 1 to 9 decimals everywhere incl. gamma=1. NB this does NOT mean the bias
  is purely in ||v|| (as this file claimed before 2026-09-02): 1-cos ~ theta^2/2,
  so a 1.4e-5 rad rotation still reads cos=1.000000000. The float64
  decomposition (figure C) puts ~90% of eps^2 in DIRECTION, ~10% in magnitude.
- K_h in {1,2,4,8,14} agree to within 1% (figure_B, a standalone PNG — not a
  panel of A): inference relaxation is deadbeat; production K_h=1 loses nothing.

**Produced by.** `experiments/cifar10_pcn/in_paper_diag_gamma_grid.py`
(submit: `submit_scripts/in_paper/in_paper_submit_gamma_grid.sh`, job 34391508) on
`../in_paper_fid_calculation_train_backprop_infer_pcn/checkpoint_warmup_145000.pt`
(EMA). Reference = exact autograd through the same PCN graph (predict-scan).
TF32 toggled via DIAG_TF32; float64 needs the dtype-honest GroupNorm patch
(torchcfm's GroupNorm32 hard-casts to fp32; patched in-script).

**Axis convention.** Figure axes plot the relative velocity error
epsilon = ||v_PCN - v_FFN|| / ||v_FFN|| (define epsilon once in the caption;
axes carry only the epsilon label to avoid norm-bar clutter).

## Extensions (same diag, same checkpoint)

**Upper gamma** (`sweep_upper_gamma.log` job 34424419 for fp32/fp64,
`sweep_upper_gamma_tf32.log` job 34450607 for the TF32 pass): O(gamma)
continues **perfectly linear to gamma=100** (rel err 1.4e-2; cos still 0.9999)
in all three arithmetics — no expansion breakdown, no solver instability.
Above gamma~1 the TF32 curve MERGES onto the fp32/fp64 line (1.6e-2 vs 1.4e-2
at gamma=100): once the O(gamma) bias exceeds the 2e-4 hardware floor, the
arithmetic stops mattering and all three curves coincide. The floors only
matter at small gamma. The
inference correspondence therefore has NO measurable upper limit within four
decades; the operating ceiling on gamma comes entirely from TRAINING
(stability + gradient fidelity), which bites near gamma~0.3. Inference-only
ports have orders of magnitude more headroom than trained ones.

**Random-init attractor** (`sweep_random_init.log`, job 34428065,
DIAG_INIT=random): from a RANDOM initial state, K_h=1 gives cos 0.35
(gamma=0.01) / 0.003 (gamma=0.1) — i.e. nothing — but **K_h=2 already reaches
cos 1.0000 and rel err ~1e-6, with only a ~1.5x step to the K_h>=5 plateau,
which is then flat out to K_h=60** (exact values in the extracted-numbers
section below). The equilibrium is a
genuine attractor of the relaxation reached in two sweeps from anywhere, not
an artifact of feedforward initialisation. (With feedforward init, K_h=1
suffices because the init is already the untilted equilibrium — see figure B.)

**Reproduce.**
    cd experiments/cifar10_pcn
    DIAG_CKPT=<ckpt> DIAG_KH=1,2,4,8,14 uv run python in_paper_diag_gamma_grid.py
    DIAG_CKPT=<ckpt> DIAG_F64=1  uv run python ...   # float64 pass
    DIAG_CKPT=<ckpt> DIAG_TF32=1 uv run python ...   # hardware-floor pass
    DIAG_CKPT=<ckpt> DIAG_GAMMAS=1,2,3,5,10,20,30,100 uv run python ...  # upper range
    DIAG_CKPT=<ckpt> DIAG_INIT=random DIAG_KH=1,2,5,10,20,60 \
      DIAG_GAMMAS=0.01,0.1 uv run python ...                             # attractor check

**Figures.** `make_figures.py` (in this directory) regenerates all three PNGs
(figure_A_gamma_floors, figure_B_kh_flatness, figure_C_direction_vs_magnitude)
from the logs here — standalone, needs only numpy+matplotlib.

## Clamp-strength convention (added 2026-09-01)

The paper states the theory in V-units: the clamp acts on the potential V
with strength gamma_V, and v = -(1/gamma_V) grad_x E -> -grad_x V. The CODE
(and every number in these result folders) parametrizes the clamp on the
pre-scale scalar o = V/alpha (alpha = output_scale = 1000): code gamma =
alpha x gamma_V. The two are numerically identical (same floats up to one
extra rounding at machine epsilon; verified reasoning in PAPER_RESULTS_SUMMARY).
Conversion of every operating point used anywhere in this project:

NB alpha is DATASET-DEPENDENT: alpha=1000 on CIFAR-10, alpha=100 on MNIST
(config output_scale). Convert with the run's own alpha.

| code gamma (as run) | gamma_V on CIFAR (alpha=1000) | gamma_V on MNIST (alpha=100) |
|---|---|---|
| 1.0   | 1e-3 | 1e-2 |
| 0.3   | 3e-4 (storm ceiling) | — |
| 0.1   | 1e-4 (CIFAR MAIN OPERATING POINT) | 1e-3 (MNIST VGG5) |
| 0.01  | 1e-5 (CIFAR float32-floor measurement) | 1e-4 (MNIST UNet DAG arms) |

## Figures regenerated in V-units (2026-09-01)

All three figures plot gamma_V = gamma_code / alpha (alpha = 1000, CIFAR-10),
matching the paper's convention (clamp stated on the potential V). The sweep
logs in this folder remain in code units, unmodified; convert with the table
above. (A secondary top axis in code units existed briefly on 2026-09-01 and
was REMOVED the same day, along with all figure titles, so the paper can
caption the figures itself.) Nothing about the data changed — this is the
exact relabelling documented in the Clamp-strength convention section above.
Recipe marker moved 0.1 -> 1e-4; the proportional guide line's constant
rescaled accordingly (0.155*gamma_V == 1.55e-4*gamma_code). Figure B legend
gammas likewise converted. The measured window spans gamma_V 1e-7..1e-1
(six decades; both walls on-plot).

## Numbers for the paper text (extracted 2026-09-02 from the logs in this folder)

All gamma below in CODE units (gamma_V = gamma_code/1000). Checkpoint = the
backprop replication warm-up 145k (checkpoint_warmup_145000.pt), the same one
in the ffn2pcn FID table.

Interpolant counts PER PASS (caption must say this): float32 main pass and
TF32 pass = 128 x 4 = 512 interpolants; **float64 pass = 64 x 2 = 128**;
random-init pass = 128 x 2 = 256. Panel B (K_h flatness) comes from the
512-interpolant float32 main pass, same checkpoint.

float64, K_h=1, at gamma_code=0.1 (gamma_V=1e-4):
  eps = 1.495e-5, eps_par = 4.72e-6, eps_perp = 1.419e-5
  eps_perp^2/eps^2 = 0.900  -> direction carries 90% of the squared error
  theta = eps_perp = 1.42e-5 rad; cos = 1.000000000 (nine decimals)
  (float32 at the same gamma: eps = 1.594e-5.)
eps/gamma_code = 1.495e-4 flat from 1e-4 to 1 (six decades in gamma_V:
1e-7..1e-3); it drifts to 1.25e-4 (gamma_code 2..30) and 1.30e-4 (100), and
the direction fraction drifts 0.90 -> 0.84 over the top two decades.

Floors (K_h=1, small gamma): float32 4.17e-7 (at 1e-4); TF32 1.99e-4 flat
from 1e-4 up to gamma_code 1.0 (2.2e-4 there). The O(gamma) bias exceeds
the TF32 floor by 3x only from gamma_code ~ 4 (gamma_V ~ 4e-3): say
"above gamma ~ a few x 10^-3" for where all three curves coincide.

Agreement across K_h in {1,2,4,8,14} (float32), for the three gammas figure B
now plots: see the 2026-09-02 section at the end of this file (within 1%).
[Superseded: this line used to quote gamma_code 0.001 (4.604-4.616e-7, "within
0.3%") — that gamma is floor-limited and is no longer plotted.]

Random-init attractor pass (float32, 256 interpolants, init = unit Gaussian
randn_like hidden states, gammas 0.01/0.1, K_h {1,2,5,10,20,60}):
  K_h=1 does NOT converge from random init (eps ~1.2, cos 0.35 / 0.003);
  K_h=2: eps 9.6e-7 (0.01), 7.4e-6 (0.1); K_h>=5: 1.40e-6 / 1.35e-5,
  unchanged out to K_h=60 (feedforward-init values: 4.6e-7 / 1.59e-5).
Figure C legend now reads eps (total) / eps_perp (direction) /
eps_par (magnitude); y-axis "relative velocity error".

## Figure B third curve changed to gamma_V=1e-5 (2026-09-02)

Figure B previously plotted gamma_V = 1e-3 / 1e-4 / 1e-6 (gamma_code 1 / 0.1 /
0.001). The 1e-6 point sits AT the float32 floor (eps 4.6e-7 vs floor 4.2e-7),
so its flatness in K_h demonstrated floor-limited arithmetic, not
K_h-independence of the O(gamma) bias -- a different claim from the other two
curves, in one figure. Replaced by gamma_V = 1e-5 (gamma_code 0.01), where
eps = 1.65e-6 is ~4x the floor and therefore bias-dominated. All three curves
now make the same claim, they are exactly one decade apart, and the recipe
point 1e-4 is the middle one. NO NEW COMPUTE: gamma_code=0.01 was already in
the swept grid at every K_h; only make_figures.py changed.

K_h spread (max-min over K_h in {1,2,4,8,14}, float32 main pass, 512 interpolants):
  gamma_V=1e-3   eps 1.591-1.595e-4   spread 0.25%
  gamma_V=1e-4   eps 1.594-1.596e-5   spread 0.13%
  gamma_V=1e-5   eps 1.647-1.663e-6   spread 0.97%
Quotable claim: eps is independent of the relaxation budget to within 1% over
K_h = 1..14 at every gamma tested.
