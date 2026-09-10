"""
Compare EP and IFT parameter gradients to the EXACT feedforward (FFN) backprop
gradient, as gamma -> 0 (the small-gamma Energy-Matching correspondence) — the
gamma-window tuner: pick the training --pcn_gamma as the largest gamma with
cos(.,g_ff) ~ 1.

  - Default checkpoint: the authors' released warm-up (FFN weights, loaded into
    the PCN via load_from_ebvit). Stage-2/3 PCN checkpoints (keys 'pcn.*')
    load directly when they exist.
  - math-SDPA forced: every loss is a double-backprop through ViT attention.

Usage: uv run python experiments/cifar10_pcn/in_paper_diag_ep_vs_ift_vs_ffn.py [ckpt]
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchvision import datasets, transforms
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from network_pcn import PCNVelocityWrapper
from network_unet import EBViTModelWrapper

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_authors/cifar10_warm_up_145000.pt"
WEIGHTS = os.environ.get("DIAG_WEIGHTS", "net_model")
OUTPUT_SCALE, LAM = 1000.0, 1.0
K_H = int(os.environ.get("DIAG_KH_EP", 2))       # EP h-equilibration budget
T_NUDGE = int(os.environ.get("DIAG_TNUDGE", 20))  # EP nudged-phase sweeps
T_FREE = int(os.environ.get("DIAG_TFREE", 15))
K_H_IFT = int(os.environ.get("DIAG_KH_IFT", 5))   # IFT fixed-point sweeps
N_CG_STEPS = int(os.environ.get("DIAG_NCG", 20))  # CG iters for the IFT solve
BETA = float(os.environ.get("DIAG_BETA", 30.0))   # UNet+ViT DAG needs large beta (curvature scaling; cf. MNIST beta=30)
NUDGE_TYPE = "linear"
GAMMAS = [float(x) for x in os.environ.get(
    "DIAG_GAMMAS", "0.003,0.01,0.03,0.1,0.3,1.0").split(",")]
N_BATCH = int(os.environ.get("DIAG_NBATCH", 2))
B = int(os.environ.get("DIAG_B", 16))   # 50M-param model + double-backward: keep batches small
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# float64 answers 'is the small-gamma degradation a precision floor?'
DTYPE = torch.float64 if os.environ.get("DIAG_FP64", "0") == "1" else torch.float32
# torchcfm's GroupNorm32 hard-casts activations to fp32, which breaks a true
# float64 pass ("mixed dtype"); make it dtype-honest, as in_paper_diag_gamma_grid
# does for the inference sweeps.
import torch.nn.functional as _Fnn
from torchcfm.models.unet import nn as _tcfm_nn
_tcfm_nn.GroupNorm32.forward = lambda self, x: _Fnn.group_norm(
    x, self.num_groups, self.weight, self.bias, self.eps)

if device == 'cuda':   # double-backprop through attention: math kernel only
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

sd = torch.load(CKPT, map_location='cpu', weights_only=False)[WEIGHTS]
sd = {k.replace("module.", ""): v for k, v in sd.items()}
IS_PCN_CKPT = any(k.startswith("pcn.") for k in sd)
print(f"ckpt={CKPT} [{WEIGHTS}] format={'pcn' if IS_PCN_CKPT else 'ebvit'}")
print(f"device={device} dtype={DTYPE} batch={B} N_batch={N_BATCH}")
print(f"EP budget: K_h={K_H} T_free={T_FREE} T_nudge={T_NUDGE} beta={BETA} "
      f"nudge={NUDGE_TYPE} | IFT: K_h={K_H_IFT} n_cg={N_CG_STEPS}\n")

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ds = datasets.CIFAR10(os.environ.get("CIFAR10_PATH", "./data"), train=True,
                      download=True, transform=tf)
loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=True, drop_last=True)
fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
SEED = int(os.environ.get("DIAG_SEED", 0))   # interpolant draw; 0 = every sweep so far
torch.manual_seed(SEED)
import numpy as _np, random as _random   # torchcfm's OT pairing draws from
_np.random.seed(SEED); _random.seed(SEED)     # the global numpy RNG: seed it too
_it = iter(loader)
def next_ot():
    global _it
    try: x1 = next(_it)[0]
    except StopIteration: _it = iter(loader); x1 = next(_it)[0]
    # PAIRED ACROSS DTYPES: draw in float64 and cast at the end, so the float32
    # and float64 passes see the SAME interpolants (a float32 draw consumes the
    # RNG stream differently, which left the previous f64/f32 comparison
    # unpaired: gradient norms 0.619 vs 0.480 on nominally identical batches).
    x1 = x1.to(device, torch.float64); x0 = torch.randn_like(x1)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
    return t.to(device, DTYPE), xt.to(device, DTYPE), ut.to(device, DTYPE)  # cast last
BATCHES = [next_ot() for _ in range(N_BATCH)]

def grads_of(module):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in module.named_parameters()}
def cosine(ga, gb):
    dot = sum((ga[n]*gb[n]).sum() for n in ga if n in gb).item()
    na = (sum((ga[n]**2).sum() for n in ga if n in gb).item())**0.5
    nb = (sum((gb[n]**2).sum() for n in gb if n in ga).item())**0.5
    return dot/(na*nb+1e-30)
def gnorm(g): return (sum((x**2).sum() for x in g.values()).item())**0.5
def mean(a): return sum(a)/len(a)

def build(mode, gamma):
    m = PCNVelocityWrapper(gamma=gamma, T_free=T_FREE, error_param=True,
        param_grad_mode=mode, n_cg_steps=N_CG_STEPS,
        K_h=(K_H_IFT if mode == 'ift' else K_H),
        output_scale=OUTPUT_SCALE, lambda_spring=LAM, beta=BETA,
        T_nudge=T_NUDGE, thirdphase=True, nudge_type=NUDGE_TYPE,
        pcn_arch="unet")
    if IS_PCN_CKPT:
        m.load_state_dict(sd)
    else:
        res = m.pcn.load_from_ebvit(sd, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
    return m.to(device).to(DTYPE)

def remap_cos(g_ffn, g_pcn, pcn):
    """cos over the concatenated shared tensors, FFN names -> PCN names.

    THE POOLING RULE for Sec 4.2, and it is a choice: every matched tensor is
    flattened and concatenated into ONE vector per estimator, and we take a
    single cosine. Not a mean of per-tensor or per-group cosines -- the optimiser
    rescales the update as a whole, so the question is whether the whole vector
    points where backprop points, and a per-tensor mean would give a 512-element
    bias the same vote as a multi-million-element conv weight.
    Consequence to remember when reading a per-group breakdown next to this
    number: at gamma_code 0.01 the three groups read 0.90 / 0.82 / 0.93 while
    THIS pooled cosine is 0.538 -- the groups' errors are correlated in a way
    that damages the summed direction more than any group alone. Because the
    vectors carry their true magnitudes, the value is dominated by the largest-
    norm groups (ViT head ~0.37, UNet ~0.13); time_embed (~9e-4) is invisible.
    Unmatched keys are skipped silently; that is safe only because build() loads
    the FFN checkpoint with strict=True and asserts no missing/unexpected keys,
    so the remap is known total before we get here.
    """
    ga, gb = [], []
    for n, g in g_ffn.items():
        pn = pcn._remap_ebvit_key(n)
        if pn is None or pn not in g_pcn: continue
        ga.append(g.flatten()); gb.append(g_pcn[pn].flatten())
    a, b = torch.cat(ga), torch.cat(gb)
    return (a @ b / (a.norm()*b.norm() + 1e-30)).item()

# ---- FFN reference (gamma-free): the real EBViTModelWrapper ----
# PCN-format checkpoints (a live EP/IFT run's own weights, keys "layers.<i>.*")
# have no EBViT twin to load, so the two cos(*,ffn) columns come out nan and only
# cos(ep,ift) is available -- the recurring trajectory-certification mode used by
# the EP main chain's per-leg monitor (OPS note 12: budgets validated at fixed
# weights can break down along the run; re-certify every ~20k steps).
#
# READ THAT MODE NARROWLY. cos(ep,ift) is a NECESSARY-NOT-SUFFICIENT health check:
#   - it DOES catch EP's estimator breaking, because IFT is exact wherever it has
#     been measured, so EP drifting away from IFT means the finite difference has
#     gone bad (an under-converged free phase, or a displacement below the float32
#     floor). That is exactly the OPS-note-12 failure it is there to watch for.
#   - it does NOT show that either estimator points where backprop points. Both
#     share the same weights, the same gamma clamp and the same free-phase
#     relaxation, so they fail in CORRELATED ways: if the relaxation itself
#     degrades along the trajectory, both updates move together and this cosine
#     stays ~1 while both drift from the true gradient.
# So no paper number may come from this mode, and none does -- every Sec 4.2
# measurement is against the true FFN backprop gradient on the fixed replication
# checkpoint. In principle the remap is invertible and a PCN checkpoint could be
# pushed back into an EBViT to recover a real reference; nothing does that yet.
ffn = None
if not IS_PCN_CKPT:
    ffn = EBViTModelWrapper(dropout=0.0, output_scale=OUTPUT_SCALE).to(device).to(DTYPE)
    ffn.load_state_dict(sd, strict=True)
def ffn_grad(t, xt, ut):
    ffn.zero_grad(set_to_none=True)
    v = ffn.velocity(xt, t)
    ((v - ut) ** 2).mean().backward()
    return grads_of(ffn)
g_ff = None
if ffn is not None:
    g_ff = [ffn_grad(*BATCHES[b]) for b in range(N_BATCH)]
    print(f"FFN ref: |g_ff| mean = {mean([gnorm(g) for g in g_ff]):.4f}\n")
else:
    print("PCN checkpoint: cos(ep,ift)-only mode (no FFN reference)\n")

print(f"{'gamma':>7} | {'|g_ift|':>9} {'|g_ep|':>9} | {'cos(ift,ffn)':>12} "
      f"{'cos(ep,ffn)':>11} {'cos(ep,ift)':>11}")
print("-"*74)
for gamma in GAMMAS:
    gi, ge, cif, cef, cei = [], [], [], [], []
    for b in range(N_BATCH):
        t, xt, ut = BATCHES[b]
        mi = build('ift', gamma); mi.zero_grad()
        v = mi(t, xt); ((v-ut)**2).mean().backward()
        g_i = grads_of(mi.pcn)
        me = build('ep', gamma); me.zero_grad()
        me(t, xt); me.compute_ep_gradients(ut)
        g_e = grads_of(me.pcn)
        gi.append(gnorm(g_i)); ge.append(gnorm(g_e))
        cif.append(remap_cos(g_ff[b], g_i, mi.pcn) if g_ff else float('nan'))
        cef.append(remap_cos(g_ff[b], g_e, me.pcn) if g_ff else float('nan'))
        cei.append(cosine(g_e, g_i))
    print(f"{gamma:>7} | {mean(gi):9.4f} {mean(ge):9.4f} | {mean(cif):12.4f} "
          f"{mean(cef):11.4f} {mean(cei):11.4f}")

print("\nIf cos(ep,ffn) and cos(ift,ffn) -> 1 as gamma->0: small-gamma correspondence")
print("holds and EP is faithful. If cos(ep,ift) stays ~1 but both drift from ffn:")
print("EP tracks IFT but the small-gamma limit isn't reached. If cos(ep,ift)<1:")
print("EP relaxation under-converged (raise budget).")
