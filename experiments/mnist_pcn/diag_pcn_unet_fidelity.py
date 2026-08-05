# File: diag_pcn_unet_fidelity.py
"""
PCN-UNet fidelity diagnostic (pattern of diag_ep_vs_ffn.py, for the DAG UNet).

Loads the trained stage-0 FFN checkpoint into BOTH the feedforward
EBViTModelWrapper and the PCN-UNet (same parameter tensors via load_from_ebvit),
then checks, on real FM training batches (OT pairs, random t):

  A. Potential:  V_ffn(x) vs V_pcn = output_scale·(1/γ)·E_int at equilibrium
     (should agree to O(γ) — the small-γ Energy Matching correspondence).
  B. Velocity:   v_ffn = -∇V_ffn  vs  PCN e-param IFT velocity (eval path)
     and the EP spring velocity (train path). Target cos ≳ 0.999 at γ=0.1.
  C. EP gradient: parameter gradients from spring-clamped EP (linear nudge)
     vs FFN backprop gradients of ½‖v-u‖² — cosine per shared tensor
     (CNN/VGG5 reference at γ=0.1: cos ≈ 0.9985, see ep-gamma-window).

Run (single GPU):
    uv run python3 experiments/mnist_pcn/diag_pcn_unet_fidelity.py
"""
import os
import os as _os_early
import sys

import torch
from torchvision import datasets, transforms
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network_unet import EBViTModelWrapper
from network_pcn import PCNVelocityWrapper

# Env-overridable: the arch flags below MUST match how the checkpoint was trained.
# Pre-2026-07-21 checkpoints are patch_size=7 / num_head_channels=64 (1 attn head);
# newer ones are 4/32 (2 heads). patch_size mismatch fails loudly on load; a
# num_head_channels mismatch loads SILENTLY and quietly changes the attention.
CKPT = _os_early.environ.get(
    "DIAG_CKPT",
    "results_mnist_pcn/ffn_unet_vit/EM_mnist_pcn_20260715_11/checkpoint_50000.pt")
PATCH_SIZE = int(_os_early.environ.get("DIAG_PATCH_SIZE", 7))
NUM_HEAD_CHANNELS = int(_os_early.environ.get("DIAG_HEAD_CH", 64))
WEIGHTS = "net_model"          # training-fidelity comparison → non-EMA weights
GAMMA = float(_os_early.environ.get("DIAG_GAMMA", 0.1))   # EP sweet spot (ep-gamma-window)
OUTPUT_SCALE = 100.0
B = 16
N_BATCHES = 4  # default; env-overridden below
import os as _os
T_FREE = int(_os.environ.get("DIAG_T_FREE", 10))
K_H = int(_os.environ.get("DIAG_K_H", 1))
DT_RELAX = 0.5
LAMBDA_SPRING = 1.0
BETA = float(_os_early.environ.get("DIAG_BETA", 10.0))
FLOAT64 = _os_early.environ.get("DIAG_FLOAT64", "0") == "1"
T_NUDGE = int(_os.environ.get("DIAG_T_NUDGE", 10))
X_GRAD_MODE = _os.environ.get("DIAG_XGRAD", "adiabatic")
N_BATCHES = int(_os.environ.get("DIAG_N_BATCHES", 4))
SEED = int(_os.environ.get("DIAG_SEED", 0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}  ckpt={CKPT}  weights={WEIGHTS}  gamma={GAMMA}")
print(f"budget: T_free={T_FREE} T_nudge={T_NUDGE} K_h={K_H}  x_grad_mode={X_GRAD_MODE}  seed={SEED} n_batches={N_BATCHES}")
print(f"gamma={GAMMA} beta={BETA} float64={FLOAT64}")
print(f"arch: patch_size={PATCH_SIZE} num_head_channels={NUM_HEAD_CHANNELS}")

# Force the math SDP backend: velocity is already a gradient, so every loss
# here is a DOUBLE backprop through the ViT attention, and the efficient/flash
# kernels have no second derivative (same context as train_cifar_multigpu:465).
if device.type == "cuda":
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    # TF32 matmul toggle (convs are already TF32 via cuDNN default)
    _tf32 = _os.environ.get("DIAG_TF32", "0") == "1"
    torch.backends.cuda.matmul.allow_tf32 = _tf32
    print(f"TF32 matmul: {_tf32}")

sd = torch.load(CKPT, map_location="cpu", weights_only=False)[WEIGHTS]
sd = {k.replace("module.", ""): v for k, v in sd.items()}

# ---- models (identical weights; dropout=0 in BOTH for determinism) ----
DTYPE = torch.float64 if FLOAT64 else torch.float32
ffn = EBViTModelWrapper(dropout=0.0, output_scale=OUTPUT_SCALE,
                        patch_size=PATCH_SIZE,
                        num_head_channels=NUM_HEAD_CHANNELS).to(device).to(DTYPE)
missing, unexpected = ffn.load_state_dict(sd, strict=True), None
ffn.eval()

pcn_w = PCNVelocityWrapper(
    gamma=GAMMA, T_free=T_FREE, dt_relax=DT_RELAX, output_scale=OUTPUT_SCALE,
    error_param=True, param_grad_mode="ep", K_h=K_H,
    lambda_spring=LAMBDA_SPRING, beta=BETA, T_nudge=T_NUDGE,
    thirdphase=True, nudge_type="linear",
    pcn_arch="unet", ep_x_grad_mode=X_GRAD_MODE,
    unet_kwargs={"patch_size": PATCH_SIZE,
                 "num_head_channels": NUM_HEAD_CHANNELS},
).to(device).to(DTYPE)
res = pcn_w.pcn.load_from_ebvit(sd, strict=True)
assert not res.missing_keys and not res.unexpected_keys
print(f"checkpoint loaded into both models (L={pcn_w.pcn.L} PC nodes)")

# ---- data: real FM batches ----
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
ds = datasets.MNIST(os.environ.get("MNIST_PATH", "./data"), train=True,
                    download=True, transform=tf)
loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=True, drop_last=True)
fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
# Seed EVERYTHING batch-related so different configs see IDENTICAL batches
# (paired comparison; unseeded runs varied by +-0.08 in EP cos on batch
# draw alone). DataLoader has num_workers=0 -> shuffle + randn + FM
# sampling all consume the global RNG deterministically.
torch.manual_seed(SEED)
_it = iter(loader)

def next_batch():
    global _it
    try:
        x1 = next(_it)[0]
    except StopIteration:
        _it = iter(loader)
        x1 = next(_it)[0]
    x0 = torch.randn_like(x1)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
    return (t.to(device).to(DTYPE), xt.to(device).to(DTYPE),
            ut.to(device).to(DTYPE))

def cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

def grad_cos_report(g_ffn, g_pcn):
    """Cosine over the concatenation of shared parameter-gradient tensors."""
    ga, gb = [], []
    per = []
    for name, g in g_ffn.items():
        pname = pcn_w.pcn._remap_ebvit_key(name)
        if pname is None or pname not in g_pcn:
            continue
        ga.append(g.flatten()); gb.append(g_pcn[pname].flatten())
        per.append((name, cos(g, g_pcn[pname])))
    total = cos(torch.cat(ga), torch.cat(gb))
    worst = sorted(per, key=lambda p: p[1])[:5]
    # Grouped cos: localize WHERE EP-vs-backprop disagreement lives.
    groups = {"time_embed": [], "unet": [], "vit": []}
    for (name, _), a, b in zip(
            [(n, None) for n, g in g_ffn.items()
             if pcn_w.pcn._remap_ebvit_key(n) is not None
             and pcn_w.pcn._remap_ebvit_key(n) in g_pcn], ga, gb):
        if name.startswith("time_embed"):
            groups["time_embed"].append((a, b))
        elif name.startswith(("patch_embed", "transformer_encoder", "final_linear")):
            groups["vit"].append((a, b))
        else:
            groups["unet"].append((a, b))
    gcos = {}
    for gname, pairs in groups.items():
        if pairs:
            A = torch.cat([a for a, _ in pairs]); Bv = torch.cat([b for _, b in pairs])
            gcos[gname] = (cos(A, Bv), A.norm().item(), Bv.norm().item())
    return total, worst, gcos

import time as _time
_t0 = _time.time()
resA, resB_ift, resB_spring, resC = [], [], [], []
for bi in range(N_BATCHES):
    t, xt, ut = next_batch()

    # ---- A: potential ----
    with torch.no_grad():
        V_ffn = ffn.potential(xt, t)                # (B,)
    pcn_w.eval()
    V_pcn = pcn_w.potential(xt, t)                  # (B,) per-sample now
    resA.append((V_ffn.mean().item(), V_pcn.mean().item()))

    # ---- B: velocity ----
    v_ffn = ffn.velocity(xt, t).detach()
    pcn_w.eval()
    v_ift = pcn_w.velocity(xt, t).detach()          # e-param IFT path
    pcn_w.train()
    v_spring = pcn_w.velocity(xt, t).detach()       # EP spring path (+ fills cache)
    resB_ift.append((cos(v_ffn, v_ift), (v_ift.norm() / v_ffn.norm()).item()))
    resB_spring.append((cos(v_ffn, v_spring), (v_spring.norm() / v_ffn.norm()).item()))

    # ---- C: EP param gradients vs FFN backprop ----
    # FFN: ∇θ (1/B)·½‖v(xt)-ut‖²
    ffn.zero_grad(set_to_none=True)
    v = ffn.velocity(xt, t)
    # Same convention as training flow_loss (pixel-MEAN over dims, batch mean,
    # no 1/2): EP's ep_gradient_step normalizes to exactly this, so the
    # |g_ffn| vs |g_ep| norm ratio is now meaningful (was off ~392x with the
    # old 0.5*sum/B convention -- a diag artifact, cosines were unaffected).
    loss = ((v - ut) ** 2).mean(dim=[1, 2, 3]).mean()
    loss.backward()
    g_ffn = {n: p.grad.detach().clone() for n, p in ffn.named_parameters()
             if p.grad is not None}

    # EP: spring cache was just filled by the train-mode velocity above
    for p in pcn_w.pcn.parameters():
        p.grad = None
    diag = pcn_w.compute_ep_gradients(ut)
    g_pcn = {n: p.grad.detach().clone() for n, p in pcn_w.pcn.named_parameters()
             if p.grad is not None}
    total, worst, gcos = grad_cos_report(g_ffn, g_pcn)
    resC.append(total)
    if bi == 0:
        print("  [EP grad by group] " + "  ".join(
            f"{g}: cos={c:.4f} |g_ffn|={na:.2e} |g_ep|={nb:.2e}"
            for g, (c, na, nb) in gcos.items()))
        print(f"  [EP diag] {diag if not isinstance(diag, dict) else {k: v for k, v in diag.items() if 'constraint' in k or 'mismatch' in k}}")
        print(f"  [EP grad] worst-5 per-tensor cos: {[(n.split('.')[-2] + '.' + n.split('.')[-1], round(c, 4)) for n, c in worst]}")

mean = lambda xs: sum(xs) / len(xs)
print(f"elapsed: {_time.time()-_t0:.1f}s for {N_BATCHES} batches")
print("\n================ RESULTS ================")
print(f"A. potential:  mean V_ffn = {mean([a for a, _ in resA]):+.3f}   "
      f"V_pcn = {mean([b for _, b in resA]):+.3f}   "
      f"(agree to O(gamma) expected)")
print(f"B. velocity cos(FFN, PCN-IFT):    {mean([c for c, _ in resB_ift]):.5f}   "
      f"norm ratio {mean([r for _, r in resB_ift]):.4f}")
print(f"B. velocity cos(FFN, EP-spring):  {mean([c for c, _ in resB_spring]):.5f}   "
      f"norm ratio {mean([r for _, r in resB_spring]):.4f}")
print(f"C. EP-vs-backprop param-grad cos: {mean(resC):.5f}   "
      f"(CNN/VGG5 reference at gamma=0.1: 0.9985)")
ok = mean([c for c, _ in resB_ift]) > 0.999 and mean(resC) > 0.99
print("VERDICT:", "PASS" if ok else "BELOW TARGET — inspect before training")
