# in_paper_diag_gamma_grid.py  — PAPER FIGURE DATA (Sec 4.1, equivalence at inference)
#
# Dense gamma grid of the INFERENCE correspondence on the backprop replication
# checkpoint (the same weights as every other certification and as the
# FFN->PCN report FIDs 6.52 / 3.47):
#
#     v_PCN(x,t)  vs  v_FFN = -output_scale * dF/dx      (the exact reference)
#
# Reports, per (gamma, K_h, dtype): velocity cosine, ||v_PCN||/||v_FFN||, and
# the relative L2 error. This is the chart the paper asks for: the O(gamma) bias
# on one side, the float precision floor on the other, and the relaxation budget
# K_h as a second axis (figure B: the deviation is K_h-independent at dt=1).
#
# NB inference only — no EP, no nudge phases, no parameter gradients (those are
# the separate Sec 4.2 certification sweeps).
#
# Env knobs:
#   DIAG_CKPT   FFN replication checkpoint (remapped into the PCN)
#   DIAG_KH     comma list of K_h values (the INFERENCE budget: relax_errors
#               takes K_h sweeps; T_free is EP-only and inert at inference)
#   DIAG_F64    "1" -> also run a float64 pass (the gamma-floor table)
#   DIAG_INIT   relaxation init: feedforward (default) | random | zeros
#   DIAG_GRAN   PC node granularity: block (default, 27 nodes) | weight (74)
#   DIAG_B      batch size (default 128)
#   DIAG_NBATCH number of (x_t, u_t) batches to average (default 4)
import os, sys
import torch
import torchvision
import torchvision.transforms as T

# Precision control. GPU convolutions default to TF32 (~10-bit mantissa),
# which pins the velocity error at ~2e-4 relative and MASKS the O(gamma) law
# DIAG_TF32=1 reproduces that hardware
# floor deliberately (it is the regime all training/FID runs used); the
# default here is TF32 OFF so the correspondence itself is measurable.
# DIAG_TF32: unset -> true float32 everywhere (matmul + cuDNN TF32 both OFF);
#            "1"   -> both ON (the TRAINING scripts' mode: train_cifar_multigpu sets
#                     matmul TF32 on; cuDNN TF32 is PyTorch's default);
#            "cudnn" -> matmul OFF, cuDNN ON = PyTorch's DEFAULT = the mode the FID
#                     SAMPLERS actually run in (none of the fid_*.py scripts touch
#                     these switches). Added so figure D's "our runs" floor
#                     is measured in the samplers' real mode, not assumed.
_tf = os.environ.get("DIAG_TF32", "")
torch.backends.cuda.matmul.allow_tf32 = (_tf == "1")
torch.backends.cudnn.allow_tf32 = (_tf in ("1", "cudnn"))

# torchcfm's GroupNorm32 hard-casts activations to fp32, which breaks a true
# float64 pass ("mixed dtype"); make it dtype-honest for this diagnostic.
import torch.nn.functional as _Fnn
from torchcfm.models.unet import nn as _tcfm_nn
_tcfm_nn.GroupNorm32.forward = lambda self, x: _Fnn.group_norm(
    x, self.num_groups, self.weight, self.bias, self.eps)

sys.argv = ["in_paper_diag_gamma_grid"]
from absl import flags
import config_multigpu as config
config.define_flags()
FLAGS = flags.FLAGS
FLAGS(["x", "--model_type=pcn_unet_vit", "--pcn_error_param", "--pcn_dt=1.0"])

from network_pcn import PCNVelocityWrapper
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

CKPT   = os.environ["DIAG_CKPT"]
KHS    = [int(v) for v in os.environ.get("DIAG_KH", "1").split(",")]
GAMMAS = [float(v) for v in os.environ.get(
    "DIAG_GAMMAS", "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0").split(",")]
B      = int(os.environ.get("DIAG_B", "128"))
NBATCH = int(os.environ.get("DIAG_NBATCH", "4"))
DTYPES = [torch.float32] + ([torch.float64] if os.environ.get("DIAG_F64") else [])
# DIAG_CLAMP: energy clamp c (V-units; ImageNet recipe c=10000). Since 2026-09-03 it
# is FOLDED into the PCN's top prediction f_L' = c tanh(output_scale f_L/c)/output_scale
# (PCNEnergyModelBase._fold_top), so the reference (predict-scan + autograd) and the
# relaxation see one and the same clamped function. 0/unset = none.
CLAMP  = float(os.environ.get("DIAG_CLAMP", "0")) or None
# DIAG_CHUNK=n: evaluate each interpolant batch in sub-batches of n. The
# interpolants and every metric are UNCHANGED (data batching, seeds and the
# batch-level cos/ratio/rel are all still over the full batch); only the
# forward/relaxation memory shrinks. Needed for float64 at B=128 (OOM at 78 GB).
CHUNK  = int(os.environ.get("DIAG_CHUNK", "0"))
def _chunked(fn, t, xt):
    if not CHUNK or xt.shape[0] <= CHUNK:
        return fn(t, xt)
    return torch.cat([fn(t[i:i + CHUNK], xt[i:i + CHUNK])
                      for i in range(0, xt.shape[0], CHUNK)], 0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
key = "ema_model" if "ema_model" in ck else "net_model"
SD = {k.replace("module.", ""): v for k, v in ck[key].items()}
print(f"ckpt={CKPT} ({key}) step={ck.get('step','?')}  B={B} x {NBATCH} batches  clamp={CLAMP}  seed={os.environ.get('DIAG_SEED', '0')}")

# --- training-realistic inputs: the CFM (x_t, u_t) pairs the model sees ---
if os.environ.get("DIAG_DATASET", "cifar10") == "imagenet32":
    # ImageNet-32 transfer check: same diag on the authors' ImageNet32 checkpoint
    # with real ImageNet32 training batches (official downsampled release).
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imagenet"))
    from dataset_imagenet32 import ImageNet32Dataset
    ds = ImageNet32Dataset(split="train", root=os.environ.get("IMAGENET32_PATH"),
                           transform=torchvision.transforms.ToTensor())
else:
    ds = torchvision.datasets.CIFAR10(root=os.environ.get("CIFAR10_PATH", "./data"),
                                  train=True, download=False, transform=T.ToTensor())
fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
g = torch.Generator().manual_seed(int(os.environ.get("DIAG_SEED", "0")))
idx = torch.randperm(len(ds), generator=g)[:B * NBATCH]
torch.manual_seed(int(os.environ.get("DIAG_SEED", "0")))
# torchcfm's OT pairing samples the plan with numpy's GLOBAL RNG (np.random.choice
# in optimal_transport.py), which torch.manual_seed does not seed. Unseeded, every
# process drew different (x0,x1) pairs -> different interpolants per pass (measured
# 2026-09-02: 19% slope drift between two ImageNet-32 runs). Seeded since then, so
# every pass with the same B*NBATCH shares identical interpolants.
import numpy as _np, random as _random
# DIAG_SEED (default 0) seeds the interpolant draw: image permutation, x0 noise,
# t, and the OT pairing. Different seeds = independent interpolant sets, used to
# put a draw-to-draw error bar on the O(gamma) prefactor.
SEED = int(os.environ.get("DIAG_SEED", "0"))
# DIAG_NOSEED=1 leaves numpy UNSEEDED (torch seeds still applied): reproduces the
# pre-2026-09-02 condition -- same images/noise/t, only the OT pairing random --
# to test whether those old runs are statistically consistent with seeded draws.
if os.environ.get("DIAG_NOSEED") != "1":
    _np.random.seed(SEED); _random.seed(SEED)
else:
    print("DIAG_NOSEED=1: numpy left unseeded (OT pairing random)")
BATCHES = []
for b in range(NBATCH):
    x1 = torch.stack([ds[i][0] for i in idx[b*B:(b+1)*B]]).to(device) * 2 - 1
    x0 = torch.randn_like(x1)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
    BATCHES.append((t.to(device), xt.to(device)))

def build(gamma, k_h, dtype):
    m = PCNVelocityWrapper(
        # T_free is a constructor argument of the shared wrapper and is INERT
        # here: it is read only by _velocity_ep_spring (EP training). At
        # inference velocity() dispatches to _velocity_error_param ->
        # relax_errors(x, gamma, K_h, dt_relax). Verified numerically (job
        # 34939353): T_free in {1, 14, 100} moves the sampler velocity by less
        # than rebuilding the model at identical flags. Same for pcn_cg_steps.
        gamma=gamma, T_free=1, dt_relax=1.0, async_mode=True,
        init_mode=os.environ.get("DIAG_INIT", "feedforward"),
        output_scale=FLAGS.output_scale, energy_clamp=CLAMP,
        error_param=True, param_grad_mode="ift", K_h=k_h,
        n_cg_steps=FLAGS.pcn_cg_steps, pcn_arch="unet",
        unet_kwargs=dict(
            granularity=os.environ.get("DIAG_GRAN", "block"),
            dim=(3, 32, 32), num_channels=FLAGS.num_channels,
            num_res_blocks=FLAGS.num_res_blocks,
            channel_mult=config.parse_channel_mult(FLAGS),
            attention_resolutions=FLAGS.attention_resolutions,
            num_heads=FLAGS.num_heads, num_head_channels=FLAGS.num_head_channels,
            patch_size=FLAGS.patch_size, embed_dim=FLAGS.embed_dim,
            transformer_nheads=FLAGS.transformer_nheads,
            transformer_nlayers=FLAGS.transformer_nlayers))
    m.pcn.load_from_ebvit({k: v.to(dtype) for k, v in SD.items()}, strict=True)
    return m.to(device).to(dtype).eval()

def ffn_velocity(m, t, xt):
    """Exact feedforward reference v = -output_scale * dF/dx (no relaxation)."""
    x_req = xt.detach().clone().requires_grad_(True)
    hs = []
    for k in range(m.pcn.L):
        hs.append(m.pcn.predict(k, x_req, hs))
    F_ps = hs[-1].reshape(hs[-1].shape[0], -1).sum(1)      # per-sample F (o-units)
    v = -FLAGS.output_scale * torch.autograd.grad(F_ps.sum(), x_req)[0]
    # (energy clamp: folded into the top prediction function since 2026-09-03, so
    #  hs[-1] is already o' and autograd carries sech^2 — nothing to apply here.)
    return v.detach()

# PER-SAMPLE columns (default ON; DIAG_PERSAMPLE=0 disables). Per sample:
#   r_i    = ||v_i|| / ||v_ff,i||          (magnitude ratio)
#   rel_i  = ||v_i - v_ff,i|| / ||v_ff,i|| (total relative error)
#   par_i  = |r_i - 1|                     (magnitude component)
#   perp_i = sqrt(rel_i^2 - par_i^2)       (direction component)
# so rel_i^2 = par_i^2 + perp_i^2 EXACTLY, per sample.
# These are aggregated two ways, and the identity survives BOTH because any
# weighted quadratic mean of the components preserves it:
#   ps_*  = unweighted RMS over samples          (every interpolant counts equally)
#   nw_*  = norm-weighted RMS, weights ||v_ff,i||^2
# nw_rel is ALGEBRAICALLY IDENTICAL to the batch-pooled ratio: with w_i=||v_ff,i||^2,
# sum_i w_i rel_i^2 = sum_i ||dv_i||^2 and sum_i w_i = ||v_ff||^2, so the weighted RMS
# is ||dv||/||v_ff|| over the flattened batch -- the weights exactly undo the
# per-sample normalisation. Hence nw_par/nw_perp decompose precisely the quantity
# figure A plots: the aggregate the sampler experiences, and the conventional way to
# report a vector field's relative error. Report one and say which; nw_*
# keeps figures A and C on one definition. The batch-level columns
# treat the whole batch as ONE vector, so a per-sample scalar rescaling that
# varies across samples (e.g. the energy clamp's sech^2(V_i/c) factor) reads
# as a DIRECTION error there even though it is a pure magnitude error per sample.
PERSAMPLE = os.environ.get("DIAG_PERSAMPLE", "1") != "0"   # default ON
print(f"\n{'dtype':>7} {'K_h':>6} {'gamma':>8} | {'v_cos':>10} "
      f"{'norm_ratio':>10} {'rel_L2_err':>10}"
      + (f" | {'ps_cos':>12} {'ps_ratio':>12} {'ps_rel_rms':>10} {'ps_par_rms':>10} {'ps_perp_rms':>11}"
         f" | {'nw_rel':>9} {'nw_par':>9} {'nw_perp':>10}" if PERSAMPLE else ""))
print("-" * (62 + (95 if PERSAMPLE else 0)))
for dtype in DTYPES:
    ref = None
    for k_h in KHS:
        for gamma in GAMMAS:
            m = build(gamma, k_h, dtype)
            if ref is None:   # FFN reference is gamma/T-independent
                ref = [_chunked(lambda tt, xx: ffn_velocity(m, tt, xx), t, xt.to(dtype))
                       for t, xt in BATCHES]
            cs, nr, re = [], [], []
            ps = []   # per-sample (cos_i, r_i, rel_i, par_i, perp_i), float64
            for (t, xt), v_ff in zip(BATCHES, ref):
                v = _chunked(lambda tt, xx: m(tt, xx).detach(), t.to(dtype), xt.to(dtype))
                cs.append(float((v * v_ff).sum() / (v.norm() * v_ff.norm() + 1e-30)))
                nr.append(float(v.norm() / (v_ff.norm() + 1e-30)))
                re.append(float((v - v_ff).norm() / (v_ff.norm() + 1e-30)))
                if PERSAMPLE:
                    a = v.double().flatten(1); b = v_ff.double().flatten(1)
                    na = a.norm(dim=1); nb = b.norm(dim=1) + 1e-300
                    cos_i = (a * b).sum(1) / (na * nb + 1e-300)
                    r_i = na / nb
                    rel_i = (a - b).norm(dim=1) / nb
                    par_i = (r_i - 1.0).abs()
                    perp_i = (rel_i ** 2 - par_i ** 2).clamp_min(0).sqrt()
                    ps.append(torch.stack([cos_i, r_i, rel_i, par_i, perp_i, nb ** 2], 1))
            n = len(cs)
            extra = ""
            if PERSAMPLE:
                # cos/ratio: means. rel,par,perp: quadratic means, unweighted (ps_) and
                # norm-weighted by ||v_ff,i||^2 (nw_). Both preserve rel^2=par^2+perp^2;
                # nw_rel is the batch-pooled ratio.
                A = torch.cat(ps, 0)
                w = A[:, 5]; W = w.sum()
                rms = lambda c: A[:, c].pow(2).mean().sqrt().item()
                nw  = lambda c: (A[:, c].pow(2).mul(w).sum() / W).sqrt().item()
                P = [A[:, 0].mean().item(), A[:, 1].mean().item(),
                     rms(2), rms(3), rms(4), nw(2), nw(3), nw(4)]
                extra = (f" | {P[0]:12.9f} {P[1]:12.9f} {P[2]:10.3e} {P[3]:10.3e} {P[4]:11.3e}"
                         f" | {P[5]:9.3e} {P[6]:9.3e} {P[7]:10.3e}")
            print(f"{str(dtype).replace('torch.',''):>7} {k_h:>6} {gamma:>8} | "
                  f"{sum(cs)/n:12.9f} {sum(nr)/n:12.9f} {sum(re)/n:12.3e}" + extra, flush=True)
        ref = None if len(KHS) > 1 else ref
print("\nRead: cos -> 1 and ratio -> 1 as gamma -> 0 is the O(gamma) correspondence;")
print("degradation at the SMALLEST gammas in float32 (absent in float64) is the")
print("precision floor. K_h rows show the relaxation-budget dependence")
print("(K_h = sweeps actually taken at inference; production FID runs used K_h=1).")
