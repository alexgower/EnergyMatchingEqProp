# File: network_ep.py
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
      s¹                 (64, 14, 14)   hidden, evolves
      s²                 (128, 7, 7)    hidden, evolves
      s³                 (128, 7, 7)    hidden, evolves
      s⁴                 (256,)         hidden, evolves

    Coupling weights (single conv/linear — bilinear form for EP):
      w₁: Conv(1→64, k=4, s=2, p=1)     s⁰ ↔ s¹
      w₂: Conv(64→128, k=4, s=2, p=1)   s¹ ↔ s²
      w₃: Conv(128→128, k=3, s=1, p=1)  s² ↔ s³
      w₄: Linear(6272→256)              flatten(s³) ↔ s⁴

    Energy:
      E = ½Σ||sⁿ||² - ε·Φ(x, s)
      where Φ = b_x·s⁰ + act(s¹)•w₁(s⁰) + act(s²)•w₂(act(s¹)) + act(s³)•w₃(act(s²)) + act(s⁴)·w₄(flat(act(s³)))

    Convergence: sⁿ ← sⁿ - ε·∇ₛₙ E  for T steps (no grad on s).
    Output: V(x) = E(x, s*) · output_scale
    """

    def __init__(self, T=50, epsilon_ep=0.5, output_scale=1.0, energy_clamp=None, init_gain=1.0, activation='identity', spectral_norm_enabled=False, learning_mode='bptt', neumann_K=10, spectral_scale=1.0, x_intra_weights=False, lambda_spring=10.0):
        super().__init__()
        self.T = T
        self.epsilon_ep = epsilon_ep
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp
        self.learning_mode = learning_mode
        self.neumann_K = neumann_K
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
        else:
            raise ValueError(f"Unknown activation: {activation}")

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

        # Coupling weights - High channel count in w1 for expressive x gradients
        self.w1 = nn.Conv2d(1, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.w2 = nn.Conv2d(256, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.w3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.w4 = nn.Linear(64 * 7 * 7, 256, bias=True)

        if init_gain != 1.0:
            nn.init.xavier_normal_(self.w1.weight, gain=init_gain)
            nn.init.xavier_normal_(self.w2.weight, gain=init_gain)
            nn.init.xavier_normal_(self.w3.weight, gain=init_gain)
            nn.init.xavier_normal_(self.w4.weight, gain=init_gain)

        if spectral_norm_enabled:
            self.w1 = spectral_norm(self.w1)
            self.w2 = spectral_norm(self.w2)
            self.w3 = spectral_norm(self.w3)
            self.w4 = spectral_norm(self.w4)

        # Store last convergence trace for plotting (set during potential())
        self._last_convergence_trace = None

    def update_spectral_scale(self, new_scale):
        """Update the spectral scale used in spectral normalisation."""
        self.spectral_scale = new_scale

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
        phi = phi + (act(s4) * (c * self.w4(s3_act))).sum(dim=1)

        return phi  # (B,)

    def _energy(self, s0, s1, s2, s3, s4):
        """
        Compute E = ½Σ||sⁿ||² - Φ_coupling per sample => shape (B,).
        Used for potential V(x) = E(x, h*) · output_scale.
        """
        B = s0.size(0)

        quad = 0.5 * (s1.view(B, -1).pow(2).sum(dim=1)
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

    def _converge(self, x, record_trace=False):
        """
        Run T steps of gradient descent on hidden states.
        Returns equilibrium states.
        Maintains autograd graph through the entire unrolling (BPTT).
        """
        B = x.size(0)
        device = x.device

        # Initialize hidden states to zeros
        s1 = torch.zeros(B, 256, 14, 14, device=device, requires_grad=True)
        s2 = torch.zeros(B, 64, 7, 7, device=device, requires_grad=True)
        s3 = torch.zeros(B, 64, 7, 7, device=device, requires_grad=True)
        s4 = torch.zeros(B, 256, device=device, requires_grad=True)

        trace = [] if record_trace else None

        for t in range(self.T):
            # Record norms for batch[0] before update
            if record_trace:
                trace.append({
                    "s1": s1[0].norm().item(),
                    "s2": s2[0].norm().item(),
                    "s3": s3[0].norm().item(),
                    "s4": s4[0].norm().item(),
                    "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                    "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                    "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                    "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
                })

            Phi = self._primitive(x, s1, s2, s3, s4).sum()

            grads = torch.autograd.grad(
                Phi, [s1, s2, s3, s4],
                create_graph=True,
            )

            # Primitive dynamics: s = ∇_s Φ
            s1, s2, s3, s4 = grads

        if record_trace:
            trace.append({
                "s1": s1[0].norm().item(),
                "s2": s2[0].norm().item(),
                "s3": s3[0].norm().item(),
                "s4": s4[0].norm().item(),
                "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
            })
            self._last_convergence_trace = trace

        return s1, s2, s3, s4

    def _converge_deq(self, x, record_trace=False):
        """
        DEQ convergence: T detached forward steps to h*, then K steps with graph.
        Each graph-enabled step = 1 Neumann iteration = 1 EP nudged step.
        K=1: simplest phantom grad. K=T: equivalent to full BPTT.
        """
        B = x.size(0)
        device = x.device

        # Phase 1: T detached forward steps to reach h*
        # x.detach() is used here because we don't want to build a graph
        # through T steps (that's the whole point of DEQ). The fixed point
        # equation s* = ∇_s Φ(x, s*) has the same solution whether x is
        # detached or not — detach only affects 2nd-order gradients (∂²Φ/∂s∂x),
        # not the forward dynamics.
        s1 = torch.zeros(B, 256, 14, 14, device=device)
        s2 = torch.zeros(B, 64, 7, 7, device=device)
        s3 = torch.zeros(B, 64, 7, 7, device=device)
        s4 = torch.zeros(B, 256, device=device)

        trace = [] if record_trace else None

        with torch.enable_grad():
            for t in range(self.T):
                if record_trace:
                    trace.append({
                        "s1": s1[0].norm().item(),
                        "s2": s2[0].norm().item(),
                        "s3": s3[0].norm().item(),
                        "s4": s4[0].norm().item(),
                        "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
                    })

                s1 = s1.detach().requires_grad_(True)
                s2 = s2.detach().requires_grad_(True)
                s3 = s3.detach().requires_grad_(True)
                s4 = s4.detach().requires_grad_(True)
                Phi = self._primitive(x.detach(), s1, s2, s3, s4).sum()
                grads = torch.autograd.grad(Phi, [s1, s2, s3, s4])
                s1, s2, s3, s4 = [g.detach() for g in grads]

        # Phase 2: K steps with create_graph=True (Neumann backward)
        # Live x (not detached) is needed here so the Neumann iterations
        # build a graph connecting s* back to x. This enables computation of
        # ds*/dx (the implicit gradient) when velocity() later calls
        # ∂V/∂x = ∂E/∂x + (∂E/∂s*)·(ds*/dx). With x.detach(), ds*/dx = 0
        # and the implicit gradient contribution would be lost entirely.
        s1 = s1.requires_grad_(True)
        s2 = s2.requires_grad_(True)
        s3 = s3.requires_grad_(True)
        s4 = s4.requires_grad_(True)

        for k in range(self.neumann_K):
            if record_trace:
                trace.append({
                    "s1": s1[0].norm().item(),
                    "s2": s2[0].norm().item(),
                    "s3": s3[0].norm().item(),
                    "s4": s4[0].norm().item(),
                    "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                    "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                    "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                    "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
                })
            Phi = self._primitive(x, s1, s2, s3, s4).sum()
            s1, s2, s3, s4 = torch.autograd.grad(
                Phi, [s1, s2, s3, s4], create_graph=True
            )

        if record_trace:
            trace.append({
                "s1": s1[0].norm().item(),
                "s2": s2[0].norm().item(),
                "s3": s3[0].norm().item(),
                "s4": s4[0].norm().item(),
                "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
            })
            self._last_convergence_trace = trace

        return s1, s2, s3, s4

    # ------------------------------------------------------------------
    # EP training mode — O(1) memory for both phases
    # ------------------------------------------------------------------

    def _converge_ep_free(self, x, T1, record_trace=False):
        """
        Free phase: run T1 steps of detached primitive dynamics.
        No autograd graph is built — O(1) memory regardless of T1.
        Returns [s1, s2, s3, s4] as plain (detached) tensors.
        """
        B = x.size(0)
        device = x.device
        s1 = torch.zeros(B, 256, 14, 14, device=device)
        s2 = torch.zeros(B, 64, 7, 7, device=device)
        s3 = torch.zeros(B, 64, 7, 7, device=device)
        s4 = torch.zeros(B, 256, device=device)

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T1):
                if record_trace:
                    trace.append({
                        "s1": s1[0].norm().item(),
                        "s2": s2[0].norm().item(),
                        "s3": s3[0].norm().item(),
                        "s4": s4[0].norm().item(),
                        "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
                    })

                s1 = s1.detach().requires_grad_(True)
                s2 = s2.detach().requires_grad_(True)
                s3 = s3.detach().requires_grad_(True)
                s4 = s4.detach().requires_grad_(True)
                Phi = self._primitive(x.detach(), s1, s2, s3, s4).sum()
                grads = torch.autograd.grad(Phi, [s1, s2, s3, s4])
                s1, s2, s3, s4 = [g.detach() for g in grads]

        if record_trace:
            trace.append({
                "s1": s1[0].norm().item(),
                "s2": s2[0].norm().item(),
                "s3": s3[0].norm().item(),
                "s4": s4[0].norm().item(),
                "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
            })
            self._last_convergence_trace = trace

        return [s1, s2, s3, s4]  # fully detached


    def _converge_ep_nudged(self, x, h_star, beta, ut, T2):
        """
        Nudged phase: run T2 steps of primitive ascent on
            Phi_nudged = (1-eps)*(1/2)||h||^2 + eps*(Phi_coupling - beta*L(h))
        where L(h) = ||v(x,h) - ut||^2 and v = -output_scale * grad_x E.

        Each step requires create_graph=True for the inner grad_x E so that
        dL/ds flows back through the velocity (mixed Hessian d2E/dx ds).
        The graph is discarded after each step — O(1) memory still holds.

        h_star: list [s1*, s2*, s3*, s4*]  (detached)
        Returns [s1_beta, s2_beta, s3_beta, s4_beta]  (detached)
        """
        eps = self.epsilon_ep
        scale = self.output_scale

        s1, s2, s3, s4 = [h.detach().clone() for h in h_star]

        with torch.enable_grad():
            for _ in range(T2):
                s1 = s1.detach().requires_grad_(True)
                s2 = s2.detach().requires_grad_(True)
                s3 = s3.detach().requires_grad_(True)
                s4 = s4.detach().requires_grad_(True)

                # --- compute L(h) ---
                # L depends on h via v = -output_scale * grad_x E(x, h).
                # To get dL/dh, we need h_grad IN the E_for_v call so that
                # the autograd graph connects: L -> v -> grad_x E -> E -> h_grad.
                # create_graph=True is required so backward can compute d(grad_x E)/dh.
                x_req = x.detach().requires_grad_(True)
                E_for_v = self._energy(x_req,
                                       s1, s2,
                                       s3, s4).sum()  # h_grad tensors IN graph
                # ∂L/∂s = 2(v-u)·∂v/∂s = 2(v-u)·(-output_scale·∂²E/∂x∂s)
                grad_x_E = torch.autograd.grad(E_for_v, x_req, create_graph=True)[0]
                v = -scale * grad_x_E  # (B, 1, 28, 28)
                L = (v - ut).pow(2).mean()  # scalar, depends on s via grad_x_E

                # Nudged primitive: Phi_nudged = Phi_free - eps*beta*L
                Phi_free = self._primitive(x.detach(), s1, s2, s3, s4).sum()
                Phi_nudged = Phi_free - eps * beta * L

                grads = torch.autograd.grad(Phi_nudged, [s1, s2, s3, s4])
                s1, s2, s3, s4 = [g.detach() for g in grads]
                # Graph discarded here — O(1) memory across steps

        return [s1, s2, s3, s4]  # fully detached

    def ep_gradient_step(self, x, h_star, h_beta, beta, ut, explicit_grad=False):
        """
        Compute the full EP gradient estimator and accumulate into param.grad:

            ep_loss = (1/beta) * [E(h_beta) - E(h*)]  +  L(h_beta)

        When explicit_grad=True, L(h_beta) contributes ∂L/∂θ via create_graph=True.
        When explicit_grad=False (default), L(h_beta) is detached from θ,
        giving implicit-only EP. The implicit gradient dominates by 1/(1-ρ).

        Accumulates into param.grad (replaces flow_loss.backward()).
        """
        scale = self.output_scale
        s1_b, s2_b, s3_b, s4_b = [h.detach() for h in h_beta]
        s1_s, s2_s, s3_s, s4_s = [h.detach() for h in h_star]
        x_det = x.detach()

        # 1. Implicit gradient: (E_beta - E_star) / beta  (live theta, x & h detached)
        E_beta = self._energy(x_det, s1_b, s2_b, s3_b, s4_b)  # (B,)
        E_star = self._energy(x_det, s1_s, s2_s, s3_s, s4_s)  # (B,)

        # 2. Explicit gradient: L(h_beta)
        #    create_graph controls whether ∂L/∂θ flows through the velocity.
        x_req = x_det.requires_grad_(False).detach().requires_grad_(True)
        E_for_v = self._energy(x_req, s1_b, s2_b, s3_b, s4_b)  # (B,)  live theta
        grad_x_E = torch.autograd.grad(E_for_v.sum(), x_req, create_graph=explicit_grad)[0]
        if not explicit_grad:
            grad_x_E = grad_x_E.detach()
        v_beta = -scale * grad_x_E
        L_beta = (v_beta - ut).pow(2).mean()

        # Full EP estimator
        ep_loss = (E_beta - E_star).mean() / beta + L_beta
        ep_loss.backward()

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
        s1 = torch.zeros(B, 256, 14, 14, device=device)
        s2 = torch.zeros(B, 64, 7, 7, device=device)
        s3 = torch.zeros(B, 64, 7, 7, device=device)
        s4 = torch.zeros(B, 256, device=device)

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
                        "x_neurons": x[0].view(-1)[::max(1, x[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                        "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
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
                "x_neurons": x[0].view(-1)[::max(1, x[0].numel()//8)][:8].detach().cpu().tolist(),
                "s1_neurons": s1[0].view(-1)[::max(1, s1[0].numel()//8)][:8].detach().cpu().tolist(),
                "s2_neurons": s2[0].view(-1)[::max(1, s2[0].numel()//8)][:8].detach().cpu().tolist(),
                "s3_neurons": s3[0].view(-1)[::max(1, s3[0].numel()//8)][:8].detach().cpu().tolist(),
                "s4_neurons": s4[0].view(-1)[::max(1, s4[0].numel()//8)][:8].detach().cpu().tolist(),
            })
            self._last_convergence_trace = trace

        return x, [s1, s2, s3, s4]  # x_star, h_star — fully detached

    def _converge_ep_spring_nudged(self, x_t, x_star, h_star, ut, beta, T2, lambda_spring):
        """
        Nudged phase for spring-clamped EP (CNN variant).

        target_x = x_t + ut / (output_scale * lambda_spring)

        Nudge loss:    L = (output_scale * λ)² / 2 * ||x - target_x||²
        Φ_nudged = Φ_int - ε(λ/2)||x - x_t||² - ε*β*(output_scale*λ)²/2 * ||x - target_x||²

        No create_graph — purely quadratic nudge in x.

        Returns:
            x_beta  (B, 1, 28, 28) detached
            h_beta  list [s1_β, s2_β, s3_β, s4_β] detached
        """
        eps = self.epsilon_ep
        scale = self.output_scale
        B = x_t.size(0)

        x_t_det = x_t.detach()
        target_x = (x_t_det + ut.detach() / (scale * lambda_spring)).detach()
        nudge_coeff = eps * beta * (scale * lambda_spring) ** 2 / 2.0

        x = x_star.detach().clone()
        s1, s2, s3, s4 = [h.detach().clone() for h in h_star]

        with torch.enable_grad():
            for _ in range(T2):
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

        return x, [s1, s2, s3, s4]  # x_beta, h_beta — fully detached

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

    def potential(self, x, t, record_trace=False):
        """
        Compute V(x) = E(x, s*) · output_scale.
        s* found by convergence; method depends on self.learning_mode.
        """
        x_req = x if x.requires_grad else x.clone().requires_grad_(True)

        if self.learning_mode == 'deq':
            s1, s2, s3, s4 = self._converge_deq(x_req, record_trace=record_trace)
        elif self.learning_mode == 'spring':
            # Spring EP: x is dynamic, trace includes x norms/neurons alongside s1-s4
            # TODO - see if need to do something with x now dynamic variable
            _, h_list = self._converge_ep_spring_free(
                x_req, self.T, self.lambda_spring, record_trace=record_trace)
            s1, s2, s3, s4 = h_list
        elif self.learning_mode == 'ep':
            s1, s2, s3, s4 = self._converge_ep_free(x_req, self.T, record_trace=record_trace)
        else:  # 'bptt'
            s1, s2, s3, s4 = self._converge(x_req, record_trace=record_trace)

        E = self._energy(x_req, s1, s2, s3, s4)

        V = E * self.output_scale
        if self.energy_clamp is not None and self.energy_clamp > 0:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """Compute velocity field => shape (B, C, H, W).

        Spring mode: v = output_scale * lambda_spring * (x* - x_t).
          Matches training exactly — no autograd needed.
        All other modes: v = -∇_x V(x) via autograd through potential().
        """
        if self.learning_mode == 'spring':
            x_det = x.detach()
            x_star, _ = self._converge_ep_spring_free(x_det, self.T, self.lambda_spring)
            return self.output_scale * self.lambda_spring * (x_star - x_det)
        else:
            with torch.enable_grad():
                x = x.clone().detach().requires_grad_(True)
                V = self.potential(x, t)
                dVdx = torch.autograd.grad(
                    outputs=V,
                    inputs=x,
                    grad_outputs=torch.ones_like(V),
                    create_graph=True,
                )[0]
                return -dVdx

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        """Same signature as EBViTModelWrapper."""
        if return_potential:
            return self.potential(x, t)
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
            for n in range(8):
                values = [t[key][n] for t in trace]
                ax.plot(timesteps, values, linewidth=1.0, alpha=0.8, label=f'n{n}')
            ax.set_xlabel("Convergence step t")
            ax.set_ylabel(f"{label} value")
            ax.set_title(f"{label} sample neurons")
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Per-neuron convergence traces (training step {step})', fontsize=13)
        fig.tight_layout()

        path = os.path.join(save_dir, f"neuron_traces_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[neuron_traces] Saved {path}")

    def _converge_detached(self, x):
        """
        Run dynamics to convergence WITHOUT building the BPTT graph.
        Each step is detached, so this is cheap and uses minimal memory.
        Used for Jacobian spectral radius computation.
        """
        B = x.size(0)
        device = x.device
        s1 = torch.zeros(B, 256, 14, 14, device=device)
        s2 = torch.zeros(B, 64, 7, 7, device=device)
        s3 = torch.zeros(B, 64, 7, 7, device=device)
        s4 = torch.zeros(B, 256, device=device)

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

