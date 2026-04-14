# File: network_ep_cnn.py
# EP-compatible recurrent CNN energy model for MNIST (1×28×28).
#
# Energy function with bilinear couplings between adjacent hidden state layers.
# Hidden states evolve to equilibrium via gradient descent on E, then
# ∇_x E at equilibrium gives the velocity field for flow matching.
#
# Same interface as EBViTModelWrapper / EBCNNModelWrapper:
#   potential(x, t), velocity(x, t), forward(t, x)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def soft_clamp(x, clamp_val):
    """Tanh-based clamp: output in [-clamp_val, clamp_val]."""
    return clamp_val * torch.tanh(x / clamp_val)


class EBEPModelWrapper(nn.Module):
    """
    EP-compatible recurrent CNN energy model (~1.9M params).

    Architecture:
      s⁰ = x            (1, 28, 28)    fixed input
      s¹                 (32, 14, 14)   hidden, evolves
      s²                 (64, 7, 7)    hidden, evolves
      s³                 (64, 7, 7)    hidden, evolves
      s⁴                 (256,)         hidden, evolves

    Coupling weights (single conv/linear — bilinear form for EP):
      w₁: Conv(1→32, k=3, s=2, p=1)     s⁰ ↔ s¹
      w₂: Conv(32→64, k=3, s=2, p=1)    s¹ ↔ s²
      w₃: Conv(64→64, k=3, s=1, p=1)    s² ↔ s³
      w₄: Linear(64*7*7→256)             flatten(s³) ↔ s⁴

    Energy:
      E = ½Σ||sⁿ||² - ε·Φ(x, s)
      where Φ = b_x·s⁰ + act(s¹)•w₁(s⁰) + act(s²)•w₂(act(s¹)) + act(s³)•w₃(act(s²)) + act(s⁴)·w₄(flat(act(s³)))

    Convergence: sⁿ ← sⁿ - ε·∇ₛₙ E  for T steps (no grad on s).
    Output: V(x) = E(x, s*) · output_scale
    """

    def __init__(self, T=50, epsilon_ep=0.5, output_scale=1.0, energy_clamp=None, init_gain=1.0, activation='identity', act_s4='', skip_s4=False, spectral_norm_enabled=False, spectral_scale=1.0, x_intra_weights=False, lambda_spring=10.0, cnn_channels=None):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 64, 256]
        self.cnn_channels = cnn_channels
        self.T = T
        self.epsilon_ep = epsilon_ep
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp
        self.spectral_scale = spectral_scale
        self.x_intra_weights = x_intra_weights
        self.lambda_spring = lambda_spring

        # Activation for coupling terms
        if activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'identity':
            self.act = lambda x: x
        elif activation == 'soft_clamp':
            self.act = lambda x: soft_clamp(x, 10.0)
        elif activation == 'silu':
            self.act = lambda x: soft_clamp(F.silu(x), 10.0)
        elif activation == 'softsign':
            self.act = lambda x: x / (1 + x.abs())
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Separate activation for s4 (flat layer — no spatial collective amplification risk).
        # Empty string or same as activation => falls back to self.act.
        _act_s4 = act_s4 if act_s4 else activation
        if _act_s4 == 'tanh':
            self.act_s4 = torch.tanh
        elif _act_s4 == 'identity':
            self.act_s4 = lambda x: x
        elif _act_s4 == 'soft_clamp':
            self.act_s4 = lambda x: soft_clamp(x, 10.0)
        elif _act_s4 == 'silu':
            self.act_s4 = lambda x: soft_clamp(F.silu(x), 10.0)
        elif _act_s4 == 'softsign':
            self.act_s4 = lambda x: x / (1 + x.abs())
        else:
            raise ValueError(f"Unknown act_s4: {_act_s4}")

        # Visible layer bias (learnable external field on x)
        self.visible_bias = nn.Parameter(torch.zeros(1, 1, 28, 28))

        # Learnable quadratic weights for x^T W x term (if enabled)
        # Use Linear layer from flattened x to x (784 -> 784)
        if x_intra_weights:
            self.x_intra_weights_layer = nn.Linear(784, 784, bias=False)
            if init_gain != 1.0:
                nn.init.xavier_normal_(self.x_intra_weights_layer.weight, gain=init_gain)
            if spectral_norm_enabled:
                self.x_intra_weights_layer = spectral_norm(self.x_intra_weights_layer)
        else:
            self.x_intra_weights_layer = None

        # Coupling weights - pyramid architecture: 1→c0→c1→c2→c3
        self.w1 = nn.Conv2d(1, cnn_channels[0], kernel_size=3, stride=2, padding=1, bias=True)
        self.w2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, stride=2, padding=1, bias=True)
        self.w3 = nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=1, padding=1, bias=True)
        self.w4 = nn.Linear(cnn_channels[2] * 7 * 7, cnn_channels[3], bias=True)

        # Optional direct x→s4 skip coupling (global velocity pathway, ~200K params)
        # bias=False: visible_bias already provides the x intercept term.
        self.skip_s4 = skip_s4
        if skip_s4:
            self.w_skip_s4 = nn.Linear(784, cnn_channels[3], bias=False)

        if init_gain != 1.0:
            nn.init.xavier_normal_(self.w1.weight, gain=init_gain)
            nn.init.xavier_normal_(self.w2.weight, gain=init_gain)
            nn.init.xavier_normal_(self.w3.weight, gain=init_gain)
            nn.init.xavier_normal_(self.w4.weight, gain=init_gain)
            if skip_s4:
                nn.init.xavier_normal_(self.w_skip_s4.weight, gain=init_gain)

        if spectral_norm_enabled:
            self.w1 = spectral_norm(self.w1)
            self.w2 = spectral_norm(self.w2)
            self.w3 = spectral_norm(self.w3)
            self.w4 = spectral_norm(self.w4)
            if skip_s4:
                self.w_skip_s4 = spectral_norm(self.w_skip_s4)

        # Store last convergence trace for plotting (set during potential())
        self._last_convergence_trace = None

    def update_spectral_scale(self, new_scale):
        """Update the spectral scale used in spectral normalisation."""
        self.spectral_scale = new_scale

    def _sample_neurons(self, tensor):
        """
        Sample 16 representative neuron values for trace logging (CNN).

        For flat tensors (s4, shape (D,)): stride sampling across 16 elements.
        For spatial tensors (s1-s3, x, shape (C,H,W)): 8 evenly-spaced channel
        groups × 2 spatial positions (center + corner), interleaved so each
        consecutive pair is the same channel at center then corner:
          [(ch0,center), (ch0,corner), (ch1,center), (ch1,corner), ...]
        This directly shows the center-vs-corner activation gap per channel.
        """
        t = tensor.detach()
        if t.dim() == 1:
            # Flat layer (e.g. s4): stride sampling across 16 elements
            return t[::max(1, t.numel() // 16)][:16].cpu().tolist()
        C, H, W = t.shape
        # 8 evenly-spaced channel indices
        ch_groups = [
            0,
            max(0, C // 8),
            max(0, C // 4),
            max(0, 3 * C // 8),
            max(0, C // 2),
            max(0, 5 * C // 8),
            max(0, 3 * C // 4),
            min(C - 1, 7 * C // 8),
        ]
        # Interleave center and corner for each channel group
        indices = []
        for ch in ch_groups:
            indices.append((ch, H // 2, W // 2))  # center
            indices.append((ch, 0,      0))         # corner
        return [t[c, h, w].item() for c, h, w in indices]

    def _coupling(self, s0, s1, s2, s3, s4):
        """
        Compute Φ_coupling(x, h) per sample => shape (B,).
        This is the interaction/coupling part of the energy.
        Weight outputs are scaled by self.spectral_scale to control ρ.
        """
        B = s0.size(0)
        act = self.act
        c = self.spectral_scale

        # Visible bias term: b_x · x
        # s0 is (B, 1, 28, 28), visible_bias is (1, 1, 28, 28)
        # Note: No activation is applied to s0 (x) to avoid distorting the input signal.
        phi = (self.visible_bias * s0).view(B, -1).sum(dim=1)

        # Add quadratic x^T W x term if enabled
        # Flatten x, apply linear transformation, then compute x^T W x
        if self.x_intra_weights_layer is not None:
            x_flat = s0.view(B, -1)  # (B, 784)
            Wx = c * self.x_intra_weights_layer(x_flat)  # (B, 784), with spectral scaling
            phi = phi + (x_flat * Wx).sum(dim=1)  # x^T W x

        # s0 → s1 coupling (conv with stride-2 downsampling)
        phi = phi + (act(s1) * (c * self.w1(s0))).view(B, -1).sum(dim=1)

        # s1 → s2 coupling
        phi = phi + (act(s2) * (c * self.w2(act(s1)))).view(B, -1).sum(dim=1)

        # s2 → s3 coupling (same resolution)
        phi = phi + (act(s3) * (c * self.w3(act(s2)))).view(B, -1).sum(dim=1)

        # s3 → s4 coupling (flatten + linear)
        s3_act = act(s3).view(B, -1)  # (B, 6272)
        phi = phi + (self.act_s4(s4) * (c * self.w4(s3_act))).sum(dim=1)

        # Optional x → s4 skip coupling (global velocity pathway)
        if self.skip_s4:
            x_flat = s0.view(B, -1)  # (B, 784)
            phi = phi + (self.act_s4(s4) * (c * self.w_skip_s4(x_flat))).sum(dim=1)

        return phi  # (B,)

    def _energy(self, s0, s1, s2, s3, s4):
        """
        Compute E = ½||s0||² + ½Σ||sⁿ||² - Φ_coupling per sample => shape (B,).
        Includes ½||x||² so that velocity_energy_gd and ep_spring_gradient_step
        are fully consistent: the x quad cancels in (E_β - E*) / β (since x is
        detached), and provides the correct restoring force in velocity_energy_gd.
        """
        B = s0.size(0)

        quad = 0.5 * (s0.view(B, -1).pow(2).sum(dim=1)
                      + s1.view(B, -1).pow(2).sum(dim=1)
                      + s2.view(B, -1).pow(2).sum(dim=1)
                      + s3.view(B, -1).pow(2).sum(dim=1)
                      + s4.pow(2).sum(dim=1))

        return quad - self._coupling(s0, s1, s2, s3, s4)  # (B,)

    def _primitive(self, s0, s1, s2, s3, s4):
        """
        Compute Φ(h) = (1/2)||h||² - ε·E per sample => shape (B,).
        Expanded: Φ = (1-ε)·(1/2)||h||² + ε·Φ_coupling
        """
        B = s0.size(0)
        eps = self.epsilon_ep

        # (1-ε) · (1/2)||h||²
        quad = 0.5 * (s1.view(B, -1).pow(2).sum(dim=1)
                      + s2.view(B, -1).pow(2).sum(dim=1)
                      + s3.view(B, -1).pow(2).sum(dim=1)
                      + s4.pow(2).sum(dim=1))

        coupling = self._coupling(s0, s1, s2, s3, s4)

        return (1.0 - eps) * quad + eps * coupling  # (B,)

    # ------------------------------------------------------------------
    # Spring-clamped EP — no create_graph anywhere, O(1) memory
    # ------------------------------------------------------------------

    def _converge_ep_spring_free(self, x_t, T1, lambda_spring, record_trace=False):
        """
        Free phase for spring-clamped EP (CNN variant).

        Energy minimised: E_spring = E_int(x, h) + (λ/2)||x - x_t||²
        Primitive used:   Φ_spring = Φ_int(x, h) + (1-ε)½||x||² - ε(λ/2)||x - x_t||²

        x (B,1,28,28) becomes a dynamic variable springing back to x_t.
        Fully detached — O(1) memory, no create_graph.

        Returns:
            x_star  (B, 1, 28, 28) detached
            h_star  list [s1, s2, s3, s4] detached
        """
        eps = self.epsilon_ep
        B = x_t.size(0)
        device = x_t.device

        x = x_t.detach().clone()
        s1 = torch.zeros(B, self.cnn_channels[0], 14, 14, device=device)
        s2 = torch.zeros(B, self.cnn_channels[1], 7, 7, device=device)
        s3 = torch.zeros(B, self.cnn_channels[2], 7, 7, device=device)
        s4 = torch.zeros(B, self.cnn_channels[3], device=device)

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T1):
                if record_trace:
                    trace.append({
                        "x": x[0].norm().item(),
                        "s1": s1[0].norm().item(),
                        "s2": s2[0].norm().item(),
                        "s3": s3[0].norm().item(),
                        "s4": s4[0].norm().item(),
                        "x_neurons": self._sample_neurons(x[0]),
                        "s1_neurons": self._sample_neurons(s1[0]),
                        "s2_neurons": self._sample_neurons(s2[0]),
                        "s3_neurons": self._sample_neurons(s3[0]),
                        "s4_neurons": self._sample_neurons(s4[0]),
                    })

                x_grad = x.detach().requires_grad_(True)
                s1 = s1.detach().requires_grad_(True)
                s2 = s2.detach().requires_grad_(True)
                s3 = s3.detach().requires_grad_(True)
                s4 = s4.detach().requires_grad_(True)

                Phi_int = self._primitive(x_grad, s1, s2, s3, s4).sum()
                x_kinetic = (1.0 - eps) * 0.5 * x_grad.pow(2).view(B, -1).sum()  # (1-ε)||x||²/2 — matches hidden primitive
                spring = eps * (lambda_spring / 2.0) * (x_grad - x_t.detach()).pow(2).view(B, -1).sum()
                Phi_spring = Phi_int + x_kinetic - spring

                grads = torch.autograd.grad(Phi_spring, [x_grad, s1, s2, s3, s4])
                x = grads[0].detach()
                s1, s2, s3, s4 = [g.detach() for g in grads[1:]]

        if record_trace:
            trace.append({
                "x": x[0].norm().item(),
                "s1": s1[0].norm().item(),
                "s2": s2[0].norm().item(),
                "s3": s3[0].norm().item(),
                "s4": s4[0].norm().item(),
                "x_neurons": self._sample_neurons(x[0]),
                "s1_neurons": self._sample_neurons(s1[0]),
                "s2_neurons": self._sample_neurons(s2[0]),
                "s3_neurons": self._sample_neurons(s3[0]),
                "s4_neurons": self._sample_neurons(s4[0]),
            })
            self._last_convergence_trace = trace

        return x, [s1, s2, s3, s4]  # x_star, h_star — fully detached

    def _converge_ep_spring_nudged(self, x_t, x_star, h_star, ut, beta, T2, lambda_spring,
                                    record_trace=False):
        """
        Nudged phase for spring-clamped EP (CNN variant).

        target_x = x_t + ut / (output_scale * lambda_spring)

        Nudge loss:    L = (output_scale * λ)² / 2 * ||x - target_x||²
        Φ_nudged = Φ_int - ε(λ/2)||x - x_t||² - ε*β*(output_scale*λ)²/2 * ||x - target_x||²

        No create_graph — purely quadratic nudge in x.

        Returns:
            x_beta  (B, 1, 28, 28) detached
            h_beta  list [s1_β, s2_β, s3_β, s4_β] detached
            trace   list of per-step neuron dicts, or None if record_trace=False
        """
        eps = self.epsilon_ep
        scale = self.output_scale
        B = x_t.size(0)

        x_t_det = x_t.detach()
        target_x = (x_t_det + ut.detach() / (scale * lambda_spring)).detach()
        nudge_coeff = eps * beta * (scale * lambda_spring) ** 2 / 2.0

        x = x_star.detach().clone()
        s1, s2, s3, s4 = [h.detach().clone() for h in h_star]

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T2):
                if record_trace:
                    trace.append({
                        "x":  x[0].norm().item(),
                        "s1": s1[0].norm().item(),
                        "s2": s2[0].norm().item(),
                        "s3": s3[0].norm().item(),
                        "s4": s4[0].norm().item(),
                        "x_neurons":  self._sample_neurons(x[0]),
                        "s1_neurons": self._sample_neurons(s1[0]),
                        "s2_neurons": self._sample_neurons(s2[0]),
                        "s3_neurons": self._sample_neurons(s3[0]),
                        "s4_neurons": self._sample_neurons(s4[0]),
                    })

                x_grad = x.detach().requires_grad_(True)
                s1 = s1.detach().requires_grad_(True)
                s2 = s2.detach().requires_grad_(True)
                s3 = s3.detach().requires_grad_(True)
                s4 = s4.detach().requires_grad_(True)

                Phi_int = self._primitive(x_grad, s1, s2, s3, s4).sum()
                x_kinetic = (1.0 - eps) * 0.5 * x_grad.pow(2).view(B, -1).sum()  # (1-ε)||x||²/2 — matches hidden primitive
                spring = eps * (lambda_spring / 2.0) * (x_grad - x_t_det).pow(2).view(B, -1).sum()
                nudge = nudge_coeff * (x_grad - target_x).pow(2).view(B, -1).sum()
                Phi_nudged = Phi_int + x_kinetic - spring - nudge

                grads = torch.autograd.grad(Phi_nudged, [x_grad, s1, s2, s3, s4])
                x = grads[0].detach()
                s1, s2, s3, s4 = [g.detach() for g in grads[1:]]

        if record_trace:
            trace.append({
                "x":  x[0].norm().item(),
                "s1": s1[0].norm().item(),
                "s2": s2[0].norm().item(),
                "s3": s3[0].norm().item(),
                "s4": s4[0].norm().item(),
                "x_neurons":  self._sample_neurons(x[0]),
                "s1_neurons": self._sample_neurons(s1[0]),
                "s2_neurons": self._sample_neurons(s2[0]),
                "s3_neurons": self._sample_neurons(s3[0]),
                "s4_neurons": self._sample_neurons(s4[0]),
            })

        return x, [s1, s2, s3, s4], trace  # x_beta, h_beta, trace — fully detached

    def ep_spring_gradient_step(self, neurons_star, neurons_beta, beta):
        """
        EP parameter gradient for spring-clamped EP (CNN variant).

        ep_loss = (E_beta - E_star) / beta

        neurons_star: [x*.detach(), s1*.detach(), s2*.detach(), s3*.detach(), s4*.detach()]
        neurons_beta: [x_β.detach(), s1_β.detach(), ...]
        All neuron tensors detached — only θ is live.
        """
        E_star = self._energy(*neurons_star)   # (B,)  live θ
        E_beta = self._energy(*neurons_beta)   # (B,)  live θ
        ep_loss = (E_beta - E_star).mean() / beta
        ep_loss.backward()

    def velocity_at_h(self, x, h_list):
        """
        Compute velocity -output_scale * ∇_x E(x, h)  for given hidden states,
        without re-running convergence. Useful for logging after the free phase.

        x:      (B, 1, 28, 28) image tensor
        h_list: list of detached hidden state tensors [s1, s2, s3, s4]
        Returns: velocity (B, 1, 28, 28)
        """
        s1, s2, s3, s4 = [h.detach() for h in h_list]
        with torch.enable_grad():
            x_req = x.clone().detach().requires_grad_(True)
            E = self._energy(x_req, s1, s2, s3, s4)  # (B,)
            v = -self.output_scale * torch.autograd.grad(E.sum(), x_req)[0]
        return v.detach()  # (B, 1, 28, 28)

    def velocity_energy_gd(self, x, t, h_steps=None):
        """
        Compute velocity via energy gradient descent (neuromorphic inference mode).

        x is FIXED, only hidden states h evolve to equilibrium via the
        primitive dynamics. Velocity uses full _energy (which now includes
        ½||x||²), providing the same restoring force as spring-clamped training.

        Args:
            x: (B, 1, 28, 28) input tensor
            t: time (unused, kept for interface compatibility)
            h_steps: number of h convergence steps (default: self.T)
        Returns:
            velocity (B, 1, 28, 28), detached
        """
        if h_steps is None:
            h_steps = self.T
        x_det = x.detach()
        h_star = self._converge_ep_free(x_det, h_steps)

        s1, s2, s3, s4 = [h.detach() for h in h_star]
        with torch.enable_grad():
            x_req = x_det.clone().requires_grad_(True)
            B = x_req.size(0)
            v = -self.output_scale * torch.autograd.grad(self._energy(x_req, s1, s2, s3, s4).sum(), x_req)[0]
        return v.detach()

    def potential(self, x, t, record_trace=False):
        """
        Compute V(x) = E(x, s*) · output_scale.
        s* found via spring-clamped free phase.
        """
        x_req = x if x.requires_grad else x.clone().requires_grad_(True)
        _, h_list = self._converge_ep_spring_free(
            x_req, self.T, self.lambda_spring, record_trace=record_trace)
        s1, s2, s3, s4 = h_list
        E = self._energy(x_req, s1, s2, s3, s4)
        V = E * self.output_scale
        if self.energy_clamp is not None and self.energy_clamp > 0:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """Spring-mode velocity: v = output_scale * lambda_spring * (x* - x_t)."""
        x_det = x.detach()
        x_star, _ = self._converge_ep_spring_free(x_det, self.T, self.lambda_spring)
        return self.output_scale * self.lambda_spring * (x_star - x_det)

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        """Same signature as EBViTModelWrapper."""
        if return_potential:
            return self.potential(x, t)
        elif getattr(self, '_gen_energy_gd', False):
            return self.velocity_energy_gd(x, t)
        else:
            return self.velocity(x, t)

    def save_convergence_plot(self, save_dir, step):
        """
        Save a plot of hidden state norms vs convergence timestep
        for a single sample (batch[0]). Must call potential() with
        record_trace=True first.
        """
        trace = self._last_convergence_trace
        if trace is None:
            return

        timesteps = list(range(len(trace)))
        fig, ax = plt.subplots(figsize=(8, 5))

        for key in ["s1", "s2", "s3", "s4"]:
            values = [t[key] for t in trace]
            ax.plot(timesteps, values, label=f"||{key}||", linewidth=1.5)

        ax.set_xlabel("Convergence step t")
        ax.set_ylabel("||sⁿ|| (single sample)")
        ax.set_title(f"Hidden state convergence (training step {step})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(save_dir, f"convergence_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[convergence_plot] Saved {path}")

    def save_layer_activations_plot(self, save_dir, step):
        """
        Save a plot showing how 8 sample neurons from each layer evolve
        over convergence steps. Must call potential() with record_trace=True first.
        """
        trace = self._last_convergence_trace
        if trace is None or "s1_neurons" not in trace[0]:
            return

        timesteps = list(range(len(trace)))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        layer_keys = [("s1_neurons", "s\u00b9"), ("s2_neurons", "s\u00b2"),
                      ("s3_neurons", "s\u00b3"), ("s4_neurons", "s\u2074")]

        for idx, (key, label) in enumerate(layer_keys):
            ax = axes[idx // 2, idx % 2]
            n_neurons = len(trace[0][key])
            for n in range(n_neurons):
                values = [t[key][n] for t in trace]
                ax.plot(timesteps, values, linewidth=0.8, alpha=0.75, label=f'n{n}')
            ax.set_xlabel("Convergence step t")
            ax.set_ylabel(f"{label} value")
            ax.set_title(f"{label} sample neurons")
            ax.legend(fontsize=5, ncol=4)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Per-neuron convergence traces (training step {step})', fontsize=13)
        fig.tight_layout()

        path = os.path.join(save_dir, f"neuron_traces_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[neuron_traces] Saved {path}")

    def save_nudge_traces_plot(self, save_dir, step):
        """
        Save a plot of per-neuron displacement from h* during the nudge phase(s).

        Reads:
          self._last_spring_free_final  — dict of final free-phase neuron values
          self._last_nudge_pos_trace    — list of per-step dicts, positive nudge
          self._last_nudge_neg_trace    — list of per-step dicts, negative nudge (or None)

        Layout: 2×2 grid (s1, s2, s3, s4).
        Each panel: 8 neuron displacement lines, positive=solid, negative=dashed.
        """
        free_final = getattr(self, '_last_spring_free_final', None)
        trace_pos  = getattr(self, '_last_nudge_pos_trace', None)
        if free_final is None or trace_pos is None:
            return
        trace_neg = getattr(self, '_last_nudge_neg_trace', None)

        layer_keys = [
            ("s1_neurons", "s¹"),
            ("s2_neurons", "s²"),
            ("s3_neurons", "s³"),
            ("s4_neurons", "s⁴"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx, (key, label) in enumerate(layer_keys):
            ax = axes[idx // 2, idx % 2]
            ref = free_final.get(key, [0.0] * 16)
            n_neurons = len(ref)

            timesteps_pos = list(range(len(trace_pos)))
            for n in range(n_neurons):
                col = colors[n % len(colors)]
                disp_pos = [entry[key][n] - ref[n] for entry in trace_pos]
                ax.plot(timesteps_pos, disp_pos, color=col, linewidth=0.8,
                        alpha=0.85, linestyle='-',
                        label=f'n{n}' if idx == 0 else None)

            if trace_neg is not None:
                timesteps_neg = list(range(len(trace_neg)))
                for n in range(n_neurons):
                    col = colors[n % len(colors)]
                    disp_neg = [entry[key][n] - ref[n] for entry in trace_neg]
                    ax.plot(timesteps_neg, disp_neg, color=col, linewidth=0.8,
                            alpha=0.55, linestyle='--')

            ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
            ax.set_xlabel("Nudge step t")
            ax.set_ylabel(f"Δ{label} (displacement from h*)")
            ax.set_title(f"{label} nudge displacement")
            ax.grid(True, alpha=0.3)

        # Single legend on first panel
        axes[0, 0].legend(fontsize=5, ncol=4, title="neuron (solid=+β, dash=−β)")
        fig.suptitle(f'Nudge displacement from h* (training step {step})', fontsize=13)
        fig.tight_layout()

        path = os.path.join(save_dir, f"nudge_displacement_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[nudge_displacement] Saved {path}")



    def _converge_detached(self, x):
        """
        Run dynamics to convergence WITHOUT building the BPTT graph.
        Each step is detached, so this is cheap and uses minimal memory.
        Used for Jacobian spectral radius computation.
        """
        B = x.size(0)
        device = x.device
        s1 = torch.zeros(B, self.cnn_channels[0], 14, 14, device=device)
        s2 = torch.zeros(B, self.cnn_channels[1], 7, 7, device=device)
        s3 = torch.zeros(B, self.cnn_channels[2], 7, 7, device=device)
        s4 = torch.zeros(B, self.cnn_channels[3], device=device)

        with torch.enable_grad():
            for t in range(self.T):
                s1 = s1.detach().requires_grad_(True)
                s2 = s2.detach().requires_grad_(True)
                s3 = s3.detach().requires_grad_(True)
                s4 = s4.detach().requires_grad_(True)
                Phi = self._primitive(x.detach(), s1, s2, s3, s4).sum()
                grads = torch.autograd.grad(Phi, [s1, s2, s3, s4])
                s1, s2, s3, s4 = [g.detach() for g in grads]

        return [s1, s2, s3, s4]

    def compute_jacobian_spectral_radius(self, x_single, hidden_star, n_iters=20):
        """
        Estimate spectral radius ρ of J = ∇²_h Φ at converged state h*.
        Uses power iteration. J is symmetric (Hessian), so ρ = ||J||_op.

        ρ is the per-step BPTT gradient decay factor:
          - ρ^T = total gradient decay over T unrolling steps
          - ρ < 1 required for convergence, but causes vanishing gradients

        Returns: (rho, rho_history)
        """
        device = x_single.device

        # Random unit vector (same shapes as hidden states)
        v = [torch.randn_like(hs[:1]) for hs in hidden_star]
        v_norm = sum(vi.pow(2).sum() for vi in v).sqrt()
        v = [vi / v_norm for vi in v]

        rho_history = []
        for _ in range(n_iters):
            h = [hs[:1].detach().clone().requires_grad_(True) for hs in hidden_star]
            Phi = self._primitive(x_single.detach(), h[0], h[1], h[2], h[3]).sum()
            h_new = list(torch.autograd.grad(Phi, h, create_graph=True))

            # J·v (= J^T·v since J is symmetric Hessian)
            Jv = torch.autograd.grad(outputs=h_new, inputs=h, grad_outputs=v)

            Jv_norm = sum(jvi.pow(2).sum() for jvi in Jv).sqrt().item()
            rho_history.append(Jv_norm)

            if Jv_norm > 1e-12:
                v = [jvi.detach() / Jv_norm for jvi in Jv]
            else:
                break

        return rho_history[-1] if rho_history else 0.0, rho_history

