"""
Compare EP and IFT parameter gradients to the EXACT feedforward (FFN) backprop
gradient, as gamma -> 0 (the small-gamma Energy-Matching correspondence).

Rationale (user): the small-gamma limit is where the PCN velocity equals the
feedforward v = -output_scale*grad_x F(x). The FFN backprop gradient of the flow
loss is EXACT (no relaxation, no CG), so it's an unambiguous gold standard --
unlike IFT, whose CG solve may itself be under-converged on stiff weights.

Test: for a sweep of gamma, compute
  g_ff  = d/dtheta 1/2||v_ff - u||^2   via double-backprop through v_ff (gamma-free)
  g_ift = PCN IFT gradient at gamma
  g_ep  = PCN EP gradient at gamma (generous relax budget so it's converged)
and report cos(g_ift,g_ff), cos(g_ep,g_ff), cos(g_ep,g_ift), plus |g| for each.
Expectation: as gamma->0, cos(.,g_ff) -> 1 (correspondence holds) if relaxation
is converged. float64 to isolate from float32 cancellation. Avg over batches.

Relax budget is deliberately generous (K_h=5, T_free=30, T_nudge=40).
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchvision import datasets, transforms
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from network_pcn import PCNVelocityWrapper

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "results_mnist_pcn/stage_2_pcn_eparam_K10_good/checkpoint_5000.pt")
OUTPUT_SCALE, LAM = 100.0, 1.0
K_H, T_FREE, T_NUDGE, BETA = 3, 20, 25, 0.1   # moderate but converged budget (EP)
K_H_IFT = 10        # IFT fixed-point sweeps (more generous than EP's K_H)
N_CG_STEPS = 20     # CG iterations for the IFT implicit solve
GAMMAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]   # small-gamma first (most important)
N_BATCH = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32   # ~10x faster than float64; user showed float64 doesn't change the picture
B = 128 if device == 'cuda' else 8
state = torch.load(CKPT, map_location='cpu', weights_only=False)['net_model']
print(f"ckpt={CKPT}\ndevice={device} dtype={DTYPE} batch={B} N_batch={N_BATCH}")
print(f"EP budget: K_h={K_H} T_free={T_FREE} T_nudge={T_NUDGE} beta={BETA} | "
      f"IFT: K_h={K_H_IFT} n_cg={N_CG_STEPS}\n")

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
ds = datasets.MNIST(os.environ.get("MNIST_PATH", "./data"), train=True, download=True, transform=tf)
loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=True, drop_last=True)
fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
_it = iter(loader)
def next_ot():
    global _it
    try: x1 = next(_it)[0]
    except StopIteration: _it = iter(loader); x1 = next(_it)[0]
    x1 = x1.to(device, DTYPE); x0 = torch.randn_like(x1)
    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
    return t.to(DTYPE), xt.to(DTYPE), ut.to(DTYPE)
BATCHES = [next_ot() for _ in range(N_BATCH)]

def grads_of(m):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in m.named_parameters()}
def cosine(ga, gb):
    dot = sum((ga[n]*gb[n]).sum() for n in ga).item()
    na = (sum((ga[n]**2).sum() for n in ga).item())**0.5
    nb = (sum((gb[n]**2).sum() for n in gb).item())**0.5
    return dot/(na*nb+1e-30)
def gnorm(g): return (sum((x**2).sum() for x in g.values()).item())**0.5
def mean(a): return sum(a)/len(a)

def build(mode, gamma):
    m = PCNVelocityWrapper(gamma=gamma, T_free=T_FREE, error_param=True,
        param_grad_mode=mode, n_cg_steps=N_CG_STEPS,
        K_h=(K_H_IFT if mode == 'ift' else K_H),
        output_scale=OUTPUT_SCALE, lambda_spring=LAM, beta=BETA,
        T_nudge=T_NUDGE, thirdphase=True)
    m.load_state_dict(state); return m.to(device).to(DTYPE)

def ffn_grad(t, xt, ut):
    """Exact FFN backprop gradient: d/dtheta 1/2||v_ff - u||^2, v_ff=-scale*grad_x F."""
    m = build('ift', 1.0)  # gamma irrelevant for the FFN path
    m.zero_grad()
    x_req = xt.detach().requires_grad_(True)
    h = x_req
    for k, layer in enumerate(m.pcn.layers):
        if k == m.pcn.L - 1:
            h = h.view(h.size(0), -1)
        h = layer(h)
    F = h.sum()
    dFdx = torch.autograd.grad(F, x_req, create_graph=True)[0]
    v_ff = -m.output_scale * dFdx
    ((v_ff - ut) ** 2).mean().backward()
    return grads_of(m)

# FFN reference (gamma-independent)
g_ff = [ffn_grad(*BATCHES[b]) for b in range(N_BATCH)]
print(f"FFN ref: |g_ff| mean = {mean([gnorm(g) for g in g_ff]):.4f}\n")

print(f"{'gamma':>7} | {'|g_ift|':>9} {'|g_ep|':>9} | {'cos(ift,ffn)':>12} "
      f"{'cos(ep,ffn)':>11} {'cos(ep,ift)':>11}")
print("-"*74)
for gamma in GAMMAS:
    gi, ge, cif, cef, cei = [], [], [], [], []
    for b in range(N_BATCH):
        t, xt, ut = BATCHES[b]
        mi = build('ift', gamma); mi.zero_grad()
        v = mi(t, xt); ((v-ut)**2).mean().backward(); g_i = grads_of(mi)
        me = build('ep', gamma); me.zero_grad()
        me(t, xt); me.compute_ep_gradients(ut); g_e = grads_of(me)
        gi.append(gnorm(g_i)); ge.append(gnorm(g_e))
        cif.append(cosine(g_i, g_ff[b])); cef.append(cosine(g_e, g_ff[b]))
        cei.append(cosine(g_e, g_i))
    print(f"{gamma:>7} | {mean(gi):9.4f} {mean(ge):9.4f} | {mean(cif):12.4f} "
          f"{mean(cef):11.4f} {mean(cei):11.4f}")

print("\nIf cos(ep,ffn) and cos(ift,ffn) -> 1 as gamma->0: small-gamma correspondence")
print("holds and EP is faithful. If cos(ep,ift) stays ~1 but both drift from ffn:")
print("EP tracks IFT but the small-gamma limit isn't reached. If cos(ep,ift)<1:")
print("EP relaxation under-converged (raise budget).")
