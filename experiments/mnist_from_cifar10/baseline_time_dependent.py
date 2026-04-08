#!/usr/bin/env python3
"""
Time-dependent flow matching baseline on sklearn digits.
Standard backprop FFN with time conditioning — no EP, no energy model.
Logs flow_bins every 100 steps for comparison with EP model.

Usage:
  python baseline_time_dependent.py [--steps=2000] [--lr=1e-3] [--hidden=128]

Runs in ~2-3 minutes on GPU, ~10 minutes on CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
from sklearn.datasets import load_digits

# ---- Config ----
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', type=str, default='./baseline_results')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---- Data ----
digits = load_digits()
x_data = torch.tensor(digits.data, dtype=torch.float32)
# Scale to [-1, 1]
x_data = (x_data / 8.0) * 2.0 - 1.0
x_data = x_data.to(device)
N, D = x_data.shape  # 1797, 64
print(f"Data: {N} samples, {D} dimensions")

# ---- OT coupling (same as EP codebase) ----
def ot_sample(x_data, batch_size=None):
    """Sample (x0, x1) pairs with OT coupling via batch-wise assignment."""
    if batch_size is None:
        batch_size = len(x_data)
    # Sample x1 from data
    idx = torch.randint(0, len(x_data), (batch_size,))
    x1 = x_data[idx]
    # Sample x0 from N(0, I)
    x0 = torch.randn_like(x1)
    
    # Mini-batch OT: sort by projection onto random direction
    # (cheap approximation to OT, same as many flow matching codebases)
    proj = torch.randn(D, device=device)
    proj = proj / proj.norm()
    
    x0_proj = x0 @ proj
    x1_proj = x1 @ proj
    
    x0_order = x0_proj.argsort()
    x1_order = x1_proj.argsort()
    
    x0_sorted = x0[x0_order]
    x1_sorted = x1[x1_order]
    
    return x0_sorted, x1_sorted

# ---- Model: simple time-conditioned MLP ----
class TimeConditionedMLP(nn.Module):
    """v_theta(x, t) — standard time-dependent velocity network."""
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        # t is concatenated as an extra input dimension
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    
    def forward(self, x, t):
        """x: (B, D), t: (B,) -> v: (B, D)"""
        t_expanded = t.unsqueeze(1)  # (B, 1)
        xt = torch.cat([x, t_expanded], dim=1)  # (B, D+1)
        return self.net(xt)

model = TimeConditionedMLP(dim=D, hidden=args.hidden).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {n_params} trainable parameters")

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---- Training loop ----
os.makedirs(args.output_dir, exist_ok=True)
log_path = os.path.join(args.output_dir, 'train.log')
log_file = open(log_path, 'w')

def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log(f"Training time-dependent baseline for {args.steps} steps")
log(f"hidden={args.hidden}, lr={args.lr}, params={n_params}")

for step in range(args.steps + 1):
    model.train()
    
    # Sample OT pairs
    x0, x1 = ot_sample(x_data)
    
    # Sample t ~ U(0, 1)
    t = torch.rand(len(x0), device=device)
    
    # Interpolate: x_t = (1-t)*x0 + t*x1
    t_view = t.unsqueeze(1)
    x_t = (1 - t_view) * x0 + t_view * x1
    
    # Target velocity: u_t = x1 - x0 (for OT linear paths)
    u_t = x1 - x0
    
    # Predict velocity
    v_pred = model(x_t, t)
    
    # MSE loss
    per_sample_mse = (v_pred - u_t).pow(2).mean(dim=1)  # (B,)
    loss = per_sample_mse.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Logging
    if step % 10 == 0:
        v_mag = v_pred.detach().norm(dim=1).mean().item()
        u_mag = u_t.norm(dim=1).mean().item()
        msg = f"[Step {step}] flow={loss.item():.5f}, v_mag={v_mag:.3f}, u_mag={u_mag:.3f}"
        
        # Flow bins every 100 steps
        if step % 100 == 0:
            bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            bins = []
            for i in range(len(bin_edges) - 1):
                mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
                if mask.sum() > 0:
                    bins.append(per_sample_mse[mask].detach().mean().item())
                else:
                    bins.append(float('nan'))
            bins_str = ','.join(f'{b:.3f}' for b in bins)
            msg += f", flow_bins=[{bins_str}]"
        
        log(msg)

# ---- Save final checkpoint ----
ckpt_path = os.path.join(args.output_dir, 'checkpoint_final.pt')
torch.save({
    'model': model.state_dict(),
    'step': args.steps,
    'args': vars(args),
}, ckpt_path)
log(f"Saved checkpoint to {ckpt_path}")

# ---- Final eval: careful flow_bins with more samples ----
log("\n--- Final flow_bins evaluation (10 repeats averaged) ---")
model.eval()
all_bins = []
with torch.no_grad():
    for _ in range(10):
        x0, x1 = ot_sample(x_data)
        t = torch.rand(len(x0), device=device)
        t_view = t.unsqueeze(1)
        x_t = (1 - t_view) * x0 + t_view * x1
        u_t = x1 - x0
        v_pred = model(x_t, t)
        per_sample_mse = (v_pred - u_t).pow(2).mean(dim=1)
        
        bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bins = []
        for i in range(len(bin_edges) - 1):
            mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
            if mask.sum() > 0:
                bins.append(per_sample_mse[mask].mean().item())
            else:
                bins.append(float('nan'))
        all_bins.append(bins)

all_bins = np.array(all_bins)
mean_bins = np.nanmean(all_bins, axis=0)
std_bins = np.nanstd(all_bins, axis=0)

log(f"Mean flow_bins: [{','.join(f'{b:.3f}' for b in mean_bins)}]")
log(f"Std  flow_bins: [{','.join(f'{b:.3f}' for b in std_bins)}]")
log(f"Ratio max/min: {np.nanmax(mean_bins)/np.nanmin(mean_bins):.2f}x")

# Also save bins as CSV for plotting
bins_csv = os.path.join(args.output_dir, 'flow_bins.csv')
with open(bins_csv, 'w') as f:
    f.write("bin_start,bin_end,mean,std\n")
    for i, (lo, hi) in enumerate(zip([0.0,0.2,0.4,0.6,0.8], [0.2,0.4,0.6,0.8,1.0])):
        f.write(f"{lo},{hi},{mean_bins[i]:.6f},{std_bins[i]:.6f}\n")
log(f"Saved bins to {bins_csv}")

log_file.close()
print("Done!")