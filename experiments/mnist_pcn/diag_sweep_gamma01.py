"""
Hyperparameter sweep at gamma=0.1 to choose the gradEP MNIST training config.

Fidelity = cos(EP gradient, EXACT FFN backprop gradient) on an off-minimum
checkpoint with real MNIST OT inputs (the regime that matters), averaged over
batches. This is the pre-flight done right (earlier ones were confounded by
gamma=1 / at-minimum measurement).

Axes: nudge_type (quad vs linear), lambda_spring, relaxation budget
(K_h, T_free, T_nudge), beta. Goal: cheapest config with cos(ep,ffn) > ~0.99,
separately for quad and linear, and see how large a beta each tolerates.

float32 (gamma=0.1 is well-conditioned in float32 per diag_ep_vs_ffn).
"""
import sys, os, itertools, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchvision import datasets, transforms
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from network_pcn import PCNVelocityWrapper

CKPT = "results_mnist_pcn/stage_2_pcn_eparam_K10_good/checkpoint_5000.pt"
GAMMA, OUTPUT_SCALE = 0.1, 100.0
N_BATCH = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32
B = 128 if device == 'cuda' else 8
state = torch.load(CKPT, map_location='cpu', weights_only=False)['net_model']
print(f"ckpt={CKPT}  gamma={GAMMA}  dtype={DTYPE}  batch={B}  N_batch={N_BATCH}\n")

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

def ffn_grad(t, xt, ut):
    m = PCNVelocityWrapper(gamma=1.0, error_param=True, param_grad_mode='ift',
        output_scale=OUTPUT_SCALE); m.load_state_dict(state); m = m.to(device).to(DTYPE)
    m.zero_grad(); x_req = xt.detach().requires_grad_(True); h = x_req
    for k, layer in enumerate(m.pcn.layers):
        if k == m.pcn.L - 1: h = h.view(h.size(0), -1)
        h = layer(h)
    dFdx = torch.autograd.grad(h.sum(), x_req, create_graph=True)[0]
    ((-m.output_scale*dFdx - ut)**2).mean().backward()
    return grads_of(m)
g_ff = [ffn_grad(*BATCHES[b]) for b in range(N_BATCH)]
print(f"FFN ref |g_ff| = {mean([gnorm(g) for g in g_ff]):.4f}\n")

def cost(K_h, T_free, T_nudge): return T_free*(K_h+1) + 2*T_nudge*(K_h+1)

rows = []
BUDGETS = [(1, 10, 10), (2, 15, 15), (3, 25, 30), (5, 40, 50)]
print(f"{'type':>5} {'lam':>4} {'K_h':>4} {'T_f':>4} {'T_n':>4} {'beta':>5} | "
      f"{'|g_ep|':>8} {'cos_ffn':>8} {'cost':>5}")
print("-"*66)
for nudge, lam, (K_h, T_free, T_nudge), beta in itertools.product(
        ['quadratic', 'linear'], [1.0, 2.0], BUDGETS, [0.1, 0.5, 2.0]):
    cs, gs = [], []
    for b in range(N_BATCH):
        t, xt, ut = BATCHES[b]
        m = PCNVelocityWrapper(gamma=GAMMA, T_free=T_free, error_param=True,
            param_grad_mode='ep', K_h=K_h, dt_relax=0.5, output_scale=OUTPUT_SCALE,
            lambda_spring=lam, beta=beta, T_nudge=T_nudge, thirdphase=True,
            nudge_type=nudge)
        m.load_state_dict(state); m = m.to(device).to(DTYPE); m.zero_grad()
        m(t, xt); m.compute_ep_gradients(ut)
        g = grads_of(m); cs.append(cosine(g, g_ff[b])); gs.append(gnorm(g))
    c = cost(K_h, T_free, T_nudge)
    rows.append((nudge, lam, K_h, T_free, T_nudge, beta, mean(gs), mean(cs), c))
    print(f"{nudge:>5} {lam:>4} {K_h:>4} {T_free:>4} {T_nudge:>4} {beta:>5} | "
          f"{mean(gs):8.4f} {mean(cs):8.4f} {c:>5}")

print("\n=== cheapest config with cos_ffn > 0.99, per nudge type ===")
for nt in ['quadratic', 'linear']:
    ok = sorted([r for r in rows if r[0] == nt and r[7] > 0.99], key=lambda r: (r[8], -r[7]))
    if ok:
        r = ok[0]
        print(f"  {nt:>9}: lam={r[1]} K_h={r[2]} T_free={r[3]} T_nudge={r[4]} beta={r[5]} "
              f"-> cos={r[7]:.4f}, cost={r[8]}")
    else:
        best = max([r for r in rows if r[0] == nt], key=lambda r: r[7])
        print(f"  {nt:>9}: none >0.99; best cos={best[7]:.4f} at {best[1:6]}")
print("\nAlso note the max beta each nudge tolerates while staying >0.99 (linear")
print("should tolerate larger beta -> better gradient SNR headroom).")
