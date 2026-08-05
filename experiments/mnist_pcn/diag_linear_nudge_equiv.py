"""
Diagnostic: linearized-tilt nudge gradient equivalence (on a trained checkpoint).

Validates the core claim of the linear-tilt nudge (nudge_type='linear'):
the linear tilt targets the SAME parameter gradient as the quadratic nudge and
as IFT at small beta, and stays accurate as beta grows — whereas the quadratic
nudge's O((beta/lambda)^2) finite-difference bias degrades its gradient.

Runs in float64 for a clean finite difference. Loads a trained Stage-2 IFT
checkpoint so gradients are in a meaningful (non-noise-dominated) regime.

Usage:
    python diag_linear_nudge_equiv.py [ckpt_path]
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_pcn import PCNVelocityWrapper

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "results_mnist_pcn/stage_2_pcn_eparam_K10_good/checkpoint_40000.pt")

# Trained-run config (handoff: Stage 2 IFT, error_param, K_h=10, gamma=1, scale=100).
GAMMA = 1.0
OUTPUT_SCALE = 100.0
LAM = 1.0            # spring stiffness for EP (IFT training didn't use it)

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64
B = 32 if device == 'cuda' else 4
print(f"device={device}, batch={B}, dtype={DTYPE}, ckpt={CKPT}")

# Generation-regime input: start from noise (what the model sees at t=0).
x = torch.randn(B, 1, 28, 28, device=device, dtype=DTYPE)
ut = torch.randn(B, 1, 28, 28, device=device, dtype=DTYPE)

state = torch.load(CKPT, map_location='cpu', weights_only=False)['net_model']

def grads_of(model):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in model.named_parameters()}

def cosine(ga, gb):
    dot = sum((ga[n] * gb[n]).sum() for n in ga).item()
    na = (sum((ga[n]**2).sum() for n in ga).item()) ** 0.5
    nb = (sum((gb[n]**2).sum() for n in gb).item()) ** 0.5
    return dot / (na * nb + 1e-30)

def gnorm(g):
    return (sum((gg**2).sum() for gg in g.values()).item()) ** 0.5

def make(mode, beta, nudge_type='quadratic', T_nudge=20):
    m = PCNVelocityWrapper(
        gamma=GAMMA, T_free=15, error_param=True, param_grad_mode=mode,
        n_cg_steps=20, K_h=(10 if mode == 'ift' else 3),
        output_scale=OUTPUT_SCALE, lambda_spring=LAM,
        beta=beta, T_nudge=T_nudge, thirdphase=True, nudge_type=nudge_type)
    m.load_state_dict(state)
    return m.to(device).to(DTYPE)

# ---- IFT reference gradient (implicit function theorem = the "truth") ----
m_ift = make('ift', beta=0.1)
m_ift.zero_grad()
v = m_ift(torch.rand(B, device=device, dtype=DTYPE), x)
((v - ut) ** 2).mean().backward()
g_ift = grads_of(m_ift)
print(f"IFT reference: flow={((v-ut)**2).mean().item():.4f}, "
      f"|v|={v.norm().item():.2f}, |g_ift|={gnorm(g_ift):.5f}\n")

hdr = (f"{'beta':>7s} {'beta/lam':>8s} | {'cos(quad,IFT)':>13s} {'cos(lin,IFT)':>13s} "
       f"{'cos(lin,quad)':>13s} | {'|g_quad|':>10s} {'|g_lin|':>10s}")
print(hdr)
print("-" * len(hdr))
for beta in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    m_q = make('ep', beta=beta, nudge_type='quadratic'); m_q.zero_grad()
    m_q(torch.rand(B, device=device, dtype=DTYPE), x); m_q.compute_ep_gradients(ut)
    g_q = grads_of(m_q)

    m_l = make('ep', beta=beta, nudge_type='linear'); m_l.zero_grad()
    m_l(torch.rand(B, device=device, dtype=DTYPE), x); m_l.compute_ep_gradients(ut)
    g_l = grads_of(m_l)

    print(f"{beta:7.2f} {beta/LAM:8.2f} | {cosine(g_q, g_ift):13.4f} "
          f"{cosine(g_l, g_ift):13.4f} {cosine(g_l, g_q):13.4f} | "
          f"{gnorm(g_q):10.4f} {gnorm(g_l):10.4f}")

print("\nExpected: small beta -> all cosines ~1.0 (linear == quadratic == IFT).")
print("Large beta -> cos(quad,IFT) degrades (O((beta/lambda)^2) bias) and |g_quad|")
print("may blow up; cos(lin,IFT) stays high, |g_lin| stays bounded (zero curvature).")