"""Stress test: EP gradient quality across checkpoints and many batches."""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from network_pcn import PCNVelocityWrapper

device = 'cuda'
CKPTS = [
    ('/workspace/EnergyMatchingEqProp/results_cifar10_pcn/stage_3_ep/CHANGE_ME/checkpoint_5000.pt', 'step5k (healthy)'),
    ('/workspace/EnergyMatchingEqProp/results_cifar10_pcn/stage_3_ep/CHANGE_ME/checkpoint_10000.pt', 'step10k (post-blowup)'),
]

N_BATCHES = 50  # test 50 batches per config

for ckpt_path, label in CKPTS:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"\n{'='*70}")
    print(f"Checkpoint: {label}")
    print(f"{'='*70}")
    
    # IFT reference (averaged over batches)
    ift_gnorms = []
    for seed in range(N_BATCHES):
        torch.manual_seed(seed)
        x = torch.randn(128,3,32,32, device=device)
        ut = torch.randn(128,3,32,32, device=device)
        t = torch.rand(128, device=device)
        m = PCNVelocityWrapper(T_free=10, error_param=True, param_grad_mode='ift',
            n_cg_steps=10, K_h=10, output_scale=1000.0).to(device)
        m.load_state_dict(ckpt['net_model'], strict=False)
        m.zero_grad()
        v = m(t, x); ((v-ut)**2).mean().backward()
        ift_gnorms.append(sum(p.grad.pow(2).sum() for p in m.parameters()).sqrt().item())
    
    ift_arr = np.array(ift_gnorms)
    print(f"IFT: gnorm mean={ift_arr.mean():.2f} std={ift_arr.std():.2f} max={ift_arr.max():.2f} min={ift_arr.min():.2f}")
    
    # EP with various settings
    for T_n in [5, 10]:
        ep_gnorms = []
        ep_losses = []
        ep_cos = []
        
        for seed in range(N_BATCHES):
            torch.manual_seed(seed)
            x = torch.randn(128,3,32,32, device=device)
            ut = torch.randn(128,3,32,32, device=device)
            t = torch.rand(128, device=device)
            
            m_ep = PCNVelocityWrapper(T_free=10, error_param=True, param_grad_mode='ep',
                lambda_spring=1.0, beta=0.1, T_nudge=T_n, K_h=1,
                thirdphase=True, output_scale=1000.0).to(device)
            m_ep.load_state_dict(ckpt['net_model'], strict=False)
            m_ep.zero_grad()
            m_ep(t, x)
            diag = m_ep.compute_ep_gradients(ut)
            gn = sum(p.grad.pow(2).sum() for p in m_ep.parameters()).sqrt().item()
            ep_gnorms.append(gn)
            ep_losses.append(diag['ep_loss'])
            
            # Cosine with IFT (recompute IFT for this batch)
            m_ift = PCNVelocityWrapper(T_free=10, error_param=True, param_grad_mode='ift',
                n_cg_steps=10, K_h=10, output_scale=1000.0).to(device)
            m_ift.load_state_dict(ckpt['net_model'], strict=False)
            m_ift.zero_grad()
            v = m_ift(t, x); ((v-ut)**2).mean().backward()
            ift_gn = sum(p.grad.pow(2).sum() for p in m_ift.parameters()).sqrt().item()
            dot = sum((p1.grad * p2.grad).sum() for (_, p1), (_, p2) 
                      in zip(m_ep.named_parameters(), m_ift.named_parameters())).item()
            cos = dot / (gn * ift_gn + 1e-12)
            ep_cos.append(cos)
        
        ep_arr = np.array(ep_gnorms)
        cos_arr = np.array(ep_cos)
        loss_arr = np.array(ep_losses)
        
        print(f"\n  EP T_nudge={T_n}, beta=0.1:")
        print(f"    gnorm: mean={ep_arr.mean():.2f} std={ep_arr.std():.2f} max={ep_arr.max():.2f} min={ep_arr.min():.2f}")
        print(f"    cos_IFT: mean={cos_arr.mean():.4f} std={cos_arr.std():.4f} min={cos_arr.min():.4f}")
        print(f"    ep_loss: mean={loss_arr.mean():.4f} std={loss_arr.std():.4f}")
        
        # Would-be-skipped at various thresholds
        for thr in [30, 50, 100]:
            n_skip = (ep_arr > thr).sum()
            print(f"    gnorm>{thr}: {n_skip}/{N_BATCHES} ({100*n_skip/N_BATCHES:.1f}%)")

print("\nDone.")
