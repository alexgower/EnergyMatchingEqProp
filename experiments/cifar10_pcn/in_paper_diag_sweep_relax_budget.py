"""
EP relaxation-budget tuner at fixed gamma: find the CHEAPEST
(nudge_type, lambda_spring, K_h, T_free, T_nudge, beta) with
cos(EP grad, exact FFN backprop grad) > 0.99 — these are the flags the
final training runs copy.


Fidelity is measured OFF the loss minimum in the true training regime:
real CIFAR OT batches, frozen once, shared by every config (paired comparison).

Usage: uv run python experiments/cifar10_pcn/in_paper_diag_sweep_relax_budget.py [ckpt]
"""
import sys, os, itertools, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchvision import datasets, transforms
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from network_pcn import PCNVelocityWrapper
from network_unet import EBViTModelWrapper

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_authors/cifar10_warm_up_145000.pt"
WEIGHTS = os.environ.get("DIAG_WEIGHTS", "net_model")
GAMMA = float(os.environ.get("DIAG_GAMMA", 0.3))   # CIFAR e-param window is LARGE gamma (see in_paper_diag_ep_vs_ift_vs_ffn run 2026-08-06)
OUTPUT_SCALE = 1000.0
N_CG_STEPS = 20
# -------- sweep axes (env-overridable for follow-up slices) --------
NUDGE_TYPES = os.environ.get("DIAG_NUDGES", "quadratic,linear").split(",")
LAMBDAS = [float(x) for x in os.environ.get("DIAG_LAMBDAS", "1.0,2.0").split(",")]
# (K_h, T_free, T_nudge) triples as K:Tf:Tn;K:Tf:Tn;...
BUDGETS = [tuple(int(v) for v in b.split(":")) for b in
           os.environ.get("DIAG_BUDGETS", "1:10:10;2:15:15;3:25:30").split(";")]
BETAS = [float(x) for x in os.environ.get("DIAG_BETAS", "0.5,5.0,30.0").split(",")]
DTS = [float(x) for x in os.environ.get("DIAG_DTS", "0.5").split(",")]
# thirdphase: 1 = three-phase (+beta/-beta, cancels O(beta^2) quad bias, 2x
# T_nudge cost), 0 = two-phase (single +beta nudge, 1x T_nudge) — the linear
# tilt has no curvature bias, so two-phase may be free of the penalty
# three-phase exists to fix.
THIRDS = [bool(int(x)) for x in os.environ.get("DIAG_THIRDS", "1").split(",")]
# -------------------------------------------------------------------
N_BATCH = int(os.environ.get("DIAG_NBATCH", 2))
B = int(os.environ.get("DIAG_B", 16))
COS_TARGET = 0.99
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# float64 pass (DIAG_FP64=1): needed to separate the nudges' ARITHMETIC floor from
# their estimator bias. torchcfm's GroupNorm32 hard-casts to fp32, which breaks a
# true float64 run ("mixed dtype"), so make it dtype-honest as the other diags do.
DTYPE = torch.float64 if os.environ.get("DIAG_FP64", "0") == "1" else torch.float32
if DTYPE is torch.float64:
    import torch.nn.functional as _Fnn
    from torchcfm.models.unet import nn as _tcfm_nn
    _tcfm_nn.GroupNorm32.forward = lambda self, x: _Fnn.group_norm(
        x, self.num_groups, self.weight, self.bias, self.eps)

if os.environ.get("DIAG_TF32", "0") == "1":
    # TF32 matmul for the ViT-heavy sweeps (convs are TF32 via cuDNN default
    # already). Fidelity probe for a ~1.2-1.4x throughput lever.
    torch.backends.cuda.matmul.allow_tf32 = True
    print("TF32 matmul: ON")

if device == 'cuda':
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

sd = torch.load(CKPT, map_location='cpu', weights_only=False)[WEIGHTS]
sd = {k.replace("module.", ""): v for k, v in sd.items()}
print(f"ckpt={CKPT} [{WEIGHTS}]  gamma={GAMMA}  device={device} batch={B} "
      f"N_batch={N_BATCH}\n")

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ds = datasets.CIFAR10(os.environ.get("CIFAR10_PATH", "./data"), train=True,
                      download=True, transform=tf)
loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=True, drop_last=True)
fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
SEED = int(os.environ.get("DIAG_SEED", 0))   # interpolant draw
torch.manual_seed(SEED)
import numpy as _np, random as _random   # torchcfm's OT pairing draws from
_np.random.seed(SEED); _random.seed(SEED)     # the global numpy RNG: seed it too
_it = iter(loader)
def next_ot():
    global _it
    try: x1 = next(_it)[0]
    except StopIteration: _it = iter(loader); x1 = next(_it)[0]
    x1 = x1.to(device, DTYPE); x0 = torch.randn_like(x1)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
    return t.to(device, DTYPE), xt.to(device, DTYPE), ut.to(device, DTYPE)
BATCHES = [next_ot() for _ in range(N_BATCH)]

def grads_of(module):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in module.named_parameters()}
def gnorm(g): return (sum((x**2).sum() for x in g.values()).item())**0.5
def mean(a): return sum(a)/len(a)
def remap_cos(g_ffn, g_pcn, pcn):
    ga, gb = [], []
    for n, g in g_ffn.items():
        pn = pcn._remap_ebvit_key(n)
        if pn is None or pn not in g_pcn: continue
        ga.append(g.flatten()); gb.append(g_pcn[pn].flatten())
    a, b = torch.cat(ga), torch.cat(gb)
    return (a @ b / (a.norm()*b.norm() + 1e-30)).item()

# ---- FFN reference gradient (exact backprop; gamma-free) ----
ffn = EBViTModelWrapper(dropout=0.0, output_scale=OUTPUT_SCALE).to(device).to(DTYPE)
ffn.load_state_dict(sd, strict=True)
def ffn_grad(t, xt, ut):
    ffn.zero_grad(set_to_none=True)
    v = ffn.velocity(xt, t)
    ((v - ut) ** 2).mean().backward()
    return grads_of(ffn)
g_ff = [ffn_grad(*BATCHES[b]) for b in range(N_BATCH)]
print(f"FFN ref |g_ff| = {mean([gnorm(g) for g in g_ff]):.4f}\n")

def cost(K_h, T_free, T_nudge, third):
    return (T_free + (2 if third else 1)*T_nudge) * (K_h+1)

rows = []
print(f"{'type':>9} {'lam':>4} {'K_h':>4} {'T_f':>4} {'T_n':>4} {'beta':>5} "
      f"{'dt':>5} {'3ph':>4} | {'|g_ep|':>8} {'cos_ffn':>8} {'cost':>5}")
print("-"*80)
for nudge, lam, (K_h, T_free, T_nudge), beta, dt_r, third in itertools.product(
        NUDGE_TYPES, LAMBDAS, BUDGETS, BETAS, DTS, THIRDS):
    cs, gs = [], []
    for b in range(N_BATCH):
        t, xt, ut = BATCHES[b]
        m = PCNVelocityWrapper(gamma=GAMMA, T_free=T_free, error_param=True,
            param_grad_mode='ep', K_h=K_h, dt_relax=dt_r,
            output_scale=OUTPUT_SCALE, lambda_spring=lam, beta=beta,
            T_nudge=T_nudge, thirdphase=third, nudge_type=nudge,
            pcn_arch="unet")
        res = m.pcn.load_from_ebvit(sd, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        m = m.to(device).to(DTYPE); m.zero_grad()
        m(t, xt); m.compute_ep_gradients(ut)
        g = grads_of(m.pcn)
        cs.append(remap_cos(g_ff[b], g, m.pcn)); gs.append(gnorm(g))
    c = cost(K_h, T_free, T_nudge, third)
    rows.append((nudge, lam, K_h, T_free, T_nudge, beta, mean(gs), mean(cs), c,
                 dt_r, third))
    print(f"{nudge:>9} {lam:>4} {K_h:>4} {T_free:>4} {T_nudge:>4} {beta:>5} "
          f"{dt_r:>5} {int(third):>4} | {mean(gs):8.4f} {mean(cs):8.4f} {c:>5}")

print(f"\n=== cheapest config with cos_ffn > {COS_TARGET}, per nudge type ===")
for nt in NUDGE_TYPES:
    ok = sorted([r for r in rows if r[0] == nt and r[7] > COS_TARGET],
                key=lambda r: (r[8], -r[7]))
    if ok:
        r = ok[0]
        print(f"  {nt:>9}: lam={r[1]} K_h={r[2]} T_free={r[3]} T_nudge={r[4]} "
              f"beta={r[5]} dt={r[9]} 3ph={int(r[10])} -> cos={r[7]:.4f}, cost={r[8]}")
    else:
        best = max([r for r in rows if r[0] == nt], key=lambda r: r[7])
        print(f"  {nt:>9}: NONE reached {COS_TARGET}; best cos={best[7]:.4f} at "
              f"lam={best[1]} K_h={best[2]} T_free={best[3]} T_nudge={best[4]} "
              f"beta={best[5]} dt={best[9]} 3ph={int(best[10])} "
              f"(raise budget or revisit gamma)")
