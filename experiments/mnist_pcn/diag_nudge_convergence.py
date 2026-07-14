"""Diagnostic: check nudge phase convergence at various T_nudge / beta."""
import sys, torch, json
sys.path.insert(0, '.')
from network_pcn import (PCNVelocityWrapper, PCNEnergyModel,
                          _errors_to_hiddens, _compute_energy_from_errors)

def measure_nudge_residuals(model, x, ut, T_nudge_val, beta_val):
    """Run EP with given T_nudge/beta, return convergence metrics."""
    pcn = model.pcn
    gamma = model.gamma
    dt = model.dt_relax
    lam = model.lambda_spring
    
    # Free phase (always use model defaults)
    x_star, errors_star, _ = pcn.relax_spring_free_errors(
        x, gamma, model.T_free, dt, lam,
        K_h=model.K_h, async_mode=model.async_mode,
        init_mode=model.init_mode
    )
    
    vel_scale = model.output_scale * model.alpha * lam
    beta_eff = beta_val / (vel_scale ** 2)
    target_x = (x.detach() + ut.detach() / vel_scale).detach()
    nudge_coeff = beta_eff * vel_scale ** 2 / 2.0
    x_t_det = x.detach()
    
    # Run nudge with EXTRA steps, logging residuals at each step
    x_n = x_star.detach().clone()
    errors = [e.detach().clone() for e in errors_star]
    
    step_metrics = []
    for step in range(max(T_nudge_val, 20)):  # always run 20 to see convergence
        # Inner: equilibrate errors
        for _ in range(model.K_h):
            even = list(range(0, pcn.L, 2))
            odd = list(range(1, pcn.L, 2))
            for k in even:
                errors[k] = pcn._update_error_gd(x_n, errors, k, gamma, dt)
            for k in odd:
                errors[k] = pcn._update_error_gd(x_n, errors, k, gamma, dt)
        
        # Compute x gradient (residual = how far from x-equilibrium)
        hiddens_fixed = _errors_to_hiddens(errors, pcn, x_n.detach())
        hiddens_fixed = [h.detach() for h in hiddens_fixed]
        with torch.enable_grad():
            xg = x_n.detach().requires_grad_(True)
            E = pcn.compute_energy(xg, hiddens_fixed, gamma)
            spring = (lam / 2.0) * (xg - x_t_det).pow(2).sum()
            nudge = nudge_coeff * (xg - target_x).pow(2).sum()
            grad_x = torch.autograd.grad(E + spring + nudge, xg)[0]
        
        # Error residual
        with torch.enable_grad():
            e_check = [e.detach().requires_grad_(True) for e in errors]
            E_check = _compute_energy_from_errors(e_check, pcn, x_n.detach(), gamma)
            e_grads = torch.autograd.grad(E_check, e_check)
            max_de = max(g.abs().max().item() for g in e_grads)
        
        step_metrics.append({
            'step': step + 1,
            'x_grad_norm': grad_x.norm().item(),
            'x_grad_max': grad_x.abs().max().item(),
            'max_dE_de': max_de,
            'x_disp': (x_n - x_star).pow(2).mean().sqrt().item(),
        })
        
        # Outer: update x
        x_n = (xg - dt * grad_x).detach()
    
    # Compute EP gradient at T_nudge_val
    # Positive nudge
    x_plus, eq_plus = pcn.relax_spring_nudged_errors(
        x.detach(), x_star, errors_star, ut,
        beta_eff, T_nudge_val, gamma, dt, lam,
        model.output_scale, model.alpha,
        K_h=model.K_h, async_mode=model.async_mode
    )
    # Negative nudge
    x_minus, eq_minus = pcn.relax_spring_nudged_errors(
        x.detach(), x_star, errors_star, ut,
        -beta_eff, T_nudge_val, gamma, dt, lam,
        model.output_scale, model.alpha,
        K_h=model.K_h, async_mode=model.async_mode
    )
    
    # EP gradient
    model.zero_grad()
    ep_loss = pcn.ep_gradient_step_errors(
        x_minus, eq_minus, x_plus, eq_plus,
        2 * beta_eff, gamma
    )
    ep_gnorm = sum(p.grad.pow(2).sum() for p in model.parameters()).sqrt().item()
    
    return step_metrics, ep_loss, ep_gnorm


# Load checkpoint and run
device = 'cuda'
CKPTS = [
    ('/workspace/EnergyMatchingEqProp/results_mnist_pcn/stage_3_ep/EM_mnist_pcn_ep_20260713_15/checkpoint_5000.pt', 'step5k'),
]

# Check if checkpoint exists
import os
for path, label in CKPTS:
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        # Try alternatives
        alt = path.replace('_15/', '_10_ver2/')
        if os.path.exists(alt):
            CKPTS = [(alt, 'ver2_5k')]
            print(f"Using alternative: {alt}")

torch.manual_seed(42)
x = torch.randn(64, 1, 28, 28, device=device)
ut = torch.randn(64, 1, 28, 28, device=device)

for ckpt_path, label in CKPTS:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    print(f"\n{'='*70}")
    print(f"Checkpoint: {label}")
    print(f"{'='*70}")
    
    # Also get IFT reference gradient
    m_ref = PCNVelocityWrapper(T_free=10, error_param=True, param_grad_mode='ift',
        n_cg_steps=10, K_h=10, output_scale=100.0).to(device)
    m_ref.load_state_dict(ckpt['net_model'], strict=False)
    m_ref.zero_grad()
    v_ref = m_ref(torch.rand(64, device=device), x)
    ((v_ref - ut)**2).mean().backward()
    ift_gnorm = sum(p.grad.pow(2).sum() for p in m_ref.parameters()).sqrt().item()
    ift_flow = ((v_ref - ut)**2).mean().item()
    print(f"IFT reference: flow={ift_flow:.4f}, grad_norm={ift_gnorm:.4f}")
    
    print(f"\n--- Nudge convergence profile (T_nudge steps) ---")
    for beta_val in [0.1, 0.05, 0.01]:
        m = PCNVelocityWrapper(T_free=10, error_param=True, param_grad_mode='ep',
            lambda_spring=1.0, beta=beta_val, K_h=1, output_scale=100.0).to(device)
        m.load_state_dict(ckpt['net_model'], strict=False)
        
        metrics, ep_loss_val, ep_gnorm = measure_nudge_residuals(m, x, ut, 5, beta_val)
        
        print(f"\n  beta={beta_val}:")
        print(f"  {'step':>4s}  {'|grad_x|':>10s}  {'max|dE/de|':>10s}  {'nudge_disp':>10s}")
        for s in [0, 1, 2, 4, 9, 14, 19]:
            if s < len(metrics):
                m_ = metrics[s]
                print(f"  {m_['step']:4d}  {m_['x_grad_max']:10.6f}  {m_['max_dE_de']:10.6f}  {m_['x_disp']:10.6f}")
        
        # EP vs IFT at various T_nudge
        for T_val in [5, 10, 15, 20]:
            m2 = PCNVelocityWrapper(T_free=10, error_param=True, param_grad_mode='ep',
                lambda_spring=1.0, beta=beta_val, T_nudge=T_val, K_h=1,
                thirdphase=True, output_scale=100.0).to(device)
            m2.load_state_dict(ckpt['net_model'], strict=False)
            m2.zero_grad()
            m2(torch.rand(64, device=device), x)
            ep_diag = m2.compute_ep_gradients(ut)
            ep_gn = sum(p.grad.pow(2).sum() for p in m2.parameters()).sqrt().item()
            
            # Cosine with IFT
            cos = sum((p.grad * m_ref.state_dict().get(n, torch.zeros_like(p.grad))).sum()
                      for n, p in m2.named_parameters())
            # Actually compute properly
            ep_grads = {n: p.grad.clone() for n, p in m2.named_parameters()}
            ref_grads = {n: p.grad.clone() for n, p in m_ref.named_parameters()}
            dot = sum((ep_grads[n] * ref_grads[n]).sum() for n in ep_grads if n in ref_grads).item()
            cos_val = dot / (ep_gn * ift_gnorm + 1e-12)
            
            print(f"    T_nudge={T_val:2d}: ep_loss={ep_diag['ep_loss']:+.4f}, gnorm={ep_gn:.2f}, cos_IFT={cos_val:.4f}, nudge_disp={ep_diag['nudge_disp']:.6f}")

print("\nDone.")
