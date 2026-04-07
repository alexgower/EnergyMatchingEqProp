# File: network_ep_mlp.py
# EP-style recurrent MLP energy model for MNIST (1×28×28 → 784-flat).
#
# Matches the architecture of the rough-and-ready FlowEqProp codebase:
#   archi = [784, 512, 512], with visible_bias and bilinear couplings.
#
# Exposes the same interface as other model wrappers:
#   potential(x, t), velocity(x, t), forward(t, x)
#
# **Primitive formulation (EP convention):**
#   Coupling: Φ_coupling = b_x · x + act(h_1) · W_0(x) + Σ_{l>0} act(h_{l+1}) · W_l(act(h_l))
#   Energy:   E(x, h) = (1/2)||h||² - Φ_coupling    (for potential V)
#   Primitive: Φ(h) = (1/2)||h||² - ε·E = (1-ε)·(1/2)||h||² + ε·Φ_coupling
#
# Dynamics: h_{t+1} = ∇_h Φ  (direct assignment, equivalent to h - ε·∇E)
# Potential: V(x) = E(x, h*) · output_scale

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as apply_spectral_norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def soft_clamp(x, clamp_val):
    """Tanh-based clamp: output in [-clamp_val, clamp_val]."""
    return clamp_val * torch.tanh(x / clamp_val)


class EBEPMLPModelWrapper(nn.Module):
    """
    EP-compatible recurrent MLP energy model for MNIST.

    Architecture: x (784) ↔ h1 (512) ↔ h2 (512)
    with visible_bias on x layer.

    Primitive: Φ(h) = (1-ε)·(1/2)||h||² + ε·Φ_coupling
    Dynamics:  h_{t+1} = ∇_h Φ(h_t)  [direct assignment]
    Energy:    E = (1/2)||h||² - Φ_coupling  [for potential V(x)]
    Output:    V(x) = E(x, h*) · output_scale
    """

    def __init__(self, archi=None, T=100, epsilon_ep=0.5, output_scale=1.0,
                 energy_clamp=None, activation='identity', init_gain=1.0, spectral_norm_enabled=False,
                 learning_mode='bptt', neumann_K=10, spectral_scale=1.0, x_intra_weights=False,
                 lambda_spring=10.0):
        super().__init__()
        if archi is None:
            archi = [784, 512, 512]

        self.archi = archi
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
        elif activation == 'silu' or activation == 'swish':
            self.act = lambda x: soft_clamp(torch.nn.functional.silu(x), 10.0)
        elif activation == 'softsign':
            self.act = lambda x: x / (1 + x.abs())
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Visible layer bias (learnable external field on x)
        self.visible_bias = nn.Parameter(torch.zeros(archi[0]))

        # Learnable quadratic weights for x^T W x term (if enabled)
        if x_intra_weights:
            # Create a linear layer from x to x for the quadratic term
            self.x_intra_weights_layer = nn.Linear(archi[0], archi[0], bias=False)
            if init_gain != 1.0:
                nn.init.xavier_normal_(self.x_intra_weights_layer.weight, gain=init_gain)
            if spectral_norm_enabled:
                self.x_intra_weights_layer = apply_spectral_norm(self.x_intra_weights_layer)
        else:
            self.x_intra_weights_layer = None

        # Inter-layer coupling weights
        self.synapses = nn.ModuleList()
        for idx in range(len(archi) - 1):
            layer = nn.Linear(archi[idx], archi[idx + 1], bias=True)
            if init_gain != 1.0:
                nn.init.xavier_normal_(layer.weight, gain=init_gain)
            if spectral_norm_enabled:
                layer = apply_spectral_norm(layer)
            self.synapses.append(layer)

        # Convergence trace for diagnostics
        self._last_convergence_trace = None

    def update_spectral_scale(self, new_scale):
        """Update the spectral scale used in spectral normalisation."""
        self.spectral_scale = new_scale

    def _sample_neurons(self, tensor):
        """
        Sample up to 8 representative neuron values for trace logging.
        For MLP all tensors are flat (1D), so stride-sampling is used.
        The tensor should already be a 1D slice (single sample, e.g. h[0]).
        """
        t = tensor.detach()
        if t.dim() > 1:
            t = t.view(-1)  # safety flatten
        return t[::max(1, t.numel() // 8)][:8].cpu().tolist()

    def _coupling(self, neurons):
        """
        Compute Φ_coupling(x, h) per sample => shape (B,).
        This is the interaction/coupling part of the energy.
        Weight outputs are scaled by self.spectral_scale to control ρ.

        neurons: list of [x_flat, h1, h2, ...] tensors
        Φ_coupling = b_x · x + (x^T W x if enabled) + act(h1) · W_0(x) + ...
        """
        act = self.act
        c = self.spectral_scale

        # Visible bias term: b_x · x
        # Note: No activation is applied to x (neurons[0]) to avoid distorting the input signal.
        phi = torch.sum(self.visible_bias * neurons[0], dim=1)

        # Add quadratic x^T W x term if enabled
        if self.x_intra_weights_layer is not None:
            # x^T W x = sum_i x_i * (W x)_i
            Wx = c * self.x_intra_weights_layer(neurons[0])  # Apply spectral scaling
            phi = phi + torch.sum(neurons[0] * Wx, dim=1)

        # Inter-layer couplings
        for idx in range(len(self.synapses)):
            if idx == 0:
                phi = phi + torch.sum(
                    act(neurons[1]) * (c * self.synapses[0](neurons[0])),
                    dim=1
                )
            else:
                phi = phi + torch.sum(
                    act(neurons[idx + 1]) * (c * self.synapses[idx](act(neurons[idx]))),
                    dim=1
                )

        return phi  # (B,)

    def _energy(self, neurons):
        """
        Compute E = (1/2)||h||² - Φ_coupling per sample => shape (B,).
        Used for potential V(x) = E(x, h*) · output_scale.
        No epsilon here — epsilon only appears in the primitive.
        """
        quad = torch.zeros(neurons[0].size(0), device=neurons[0].device)
        for idx in range(1, len(neurons)):
            quad = quad + 0.5 * neurons[idx].pow(2).sum(dim=1)

        return quad - self._coupling(neurons)  # (B,)

    def _primitive(self, neurons):
        """
        Compute Φ(h) = (1/2)||h||² - ε·E per sample => shape (B,).

        Expanded: Φ = (1-ε)·(1/2)||h||² + ε·Φ_coupling

        The dynamics h_{t+1} = ∇_h Φ give ε-step gradient descent on E.
        """
        eps = self.epsilon_ep

        # (1-ε) · (1/2)||h||²
        quad = torch.zeros(neurons[0].size(0), device=neurons[0].device)
        for idx in range(1, len(neurons)):
            quad = quad + 0.5 * neurons[idx].pow(2).sum(dim=1)

        coupling = self._coupling(neurons)

        return (1.0 - eps) * quad + eps * coupling  # (B,)

    def _converge(self, x_flat, record_trace=False):
        """
        Run T steps of primitive dynamics: h_{t+1} = ∇_h Φ(h_t).
        Equivalent to ε-step gradient descent on E.
        Returns equilibrium hidden states.
        Maintains autograd graph through the entire unrolling (BPTT).
        """
        B = x_flat.size(0)
        device = x_flat.device

        # Initialize hidden states to zeros
        hidden = []
        for idx in range(1, len(self.archi)):
            hidden.append(torch.zeros(B, self.archi[idx], device=device, requires_grad=True))

        trace = [] if record_trace else None

        for t in range(self.T):
            neurons = [x_flat] + hidden

            if record_trace:
                entry = {}
                for idx, h in enumerate(hidden):
                    entry[f"h{idx+1}"] = h[0].norm().item()
                    entry[f"h{idx+1}_neurons"] = h[0].view(-1).detach().cpu().tolist()
                trace.append(entry)

            # Primitive dynamics: h = ∇_h Φ
            Phi = self._primitive(neurons).sum()
            grads = torch.autograd.grad(Phi, hidden, create_graph=True)

            # Full BPTT: keep the graph connected across all T steps
            hidden = list(grads)

        if record_trace:
            entry = {}
            for idx, h in enumerate(hidden):
                entry[f"h{idx+1}"] = h[0].norm().item()
                entry[f"h{idx+1}_neurons"] = h[0].view(-1).detach().cpu().tolist()
            trace.append(entry)
            self._last_convergence_trace = trace

        return hidden

    def _converge_deq(self, x_flat, record_trace=False):
        """
        DEQ convergence: T detached forward steps to h*, then K steps with graph.
        Each graph-enabled step = 1 Neumann iteration = 1 EP nudged step.
        """
        B = x_flat.size(0)
        device = x_flat.device

        # Phase 1: T detached forward steps to reach h*
        # x_flat.detach() is used here because we don't want to build a graph
        # through T steps (that's the whole point of DEQ). The fixed point
        # h* = ∇_h Φ(x, h*) has the same solution whether x is detached or
        # not — detach only affects 2nd-order gradients, not forward dynamics.
        hidden = [torch.zeros(B, self.archi[idx+1], device=device)
                  for idx in range(len(self.archi)-1)]

        trace = [] if record_trace else None

        with torch.enable_grad():
            for t in range(self.T):
                if record_trace:
                    entry = {}
                    for idx, h in enumerate(hidden):
                        entry[f"h{idx+1}"] = h[0].norm().item()
                        entry[f"h{idx+1}_neurons"] = h[0].view(-1).detach().cpu().tolist()
                    trace.append(entry)

                h_grad = [h.detach().requires_grad_(True) for h in hidden]
                neurons = [x_flat.detach()] + h_grad
                Phi = self._primitive(neurons).sum()
                grads = torch.autograd.grad(Phi, h_grad)
                hidden = [g.detach() for g in grads]

        # Phase 2: K steps with create_graph=True (Neumann backward)
        # Live x_flat (not detached) so the Neumann iterations build a graph
        # connecting h* back to x. This enables ds*/dx (the implicit gradient)
        # for computing ∂V/∂x = ∂E/∂x + (∂E/∂s*)·(ds*/dx) in velocity().
        hidden = [h.requires_grad_(True) for h in hidden]

        for k in range(self.neumann_K):
            if record_trace:
                entry = {}
                for idx, h in enumerate(hidden):
                    entry[f"h{idx+1}"] = h[0].norm().item()
                    entry[f"h{idx+1}_neurons"] = h[0].view(-1).detach().cpu().tolist()
                trace.append(entry)
            neurons = [x_flat] + hidden
            Phi = self._primitive(neurons).sum()
            grads = torch.autograd.grad(Phi, hidden, create_graph=True)
            hidden = list(grads)

        if record_trace:
            entry = {}
            for idx, h in enumerate(hidden):
                entry[f"h{idx+1}"] = h[0].norm().item()
                entry[f"h{idx+1}_neurons"] = h[0].view(-1).detach().cpu().tolist()
            trace.append(entry)
            self._last_convergence_trace = trace

        return hidden

    # ------------------------------------------------------------------
    # EP training mode — O(1) memory for both phases
    # ------------------------------------------------------------------

    def _converge_ep_free(self, x_flat, T1, record_trace=False):
        """
        Free phase: run T1 steps of detached primitive dynamics.
        No autograd graph is built — O(1) memory regardless of T1.
        Returns list of plain (detached) tensors [h1, h2, ...].
        """
        B = x_flat.size(0)
        device = x_flat.device
        hidden = [torch.zeros(B, self.archi[idx + 1], device=device)
                  for idx in range(len(self.archi) - 1)]

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T1):
                if record_trace:
                    entry = {}
                    for idx, h in enumerate(hidden):
                        entry[f"h{idx+1}"] = h[0].norm().item()
                        entry[f"h{idx+1}_neurons"] = h[0].detach().cpu().tolist()
                    trace.append(entry)

                h_grad = [h.detach().requires_grad_(True) for h in hidden]
                neurons = [x_flat.detach()] + h_grad
                Phi = self._primitive(neurons).sum()
                grads = torch.autograd.grad(Phi, h_grad)
                hidden = [g.detach() for g in grads]

        if record_trace:
            entry = {}
            for idx, h in enumerate(hidden):
                entry[f"h{idx+1}"] = h[0].norm().item()
                entry[f"h{idx+1}_neurons"] = h[0].detach().cpu().tolist()
            trace.append(entry)
            self._last_convergence_trace = trace

        return hidden  # fully detached


    def _converge_ep_nudged(self, x_flat, h_star, beta, ut, T2):
        """
        Nudged phase: run T2 steps of primitive ascent on
            Phi_nudged = (1-eps)*(1/2)||h||^2 + eps*(Phi_coupling - beta*L(h))
        where L(h) = ||v(x,h) - ut||^2 and v = -output_scale * grad_x E.

        Each step requires create_graph=True for the inner grad_x E computation
        so that dL/dh can flow back through the velocity. The graph is discarded
        immediately after each step, so memory is still O(1).

        Returns list of plain (detached) tensors [h1_beta, h2_beta, ...].
        """
        eps = self.epsilon_ep
        scale = self.output_scale

        hidden = [h.detach().clone() for h in h_star]

        with torch.enable_grad():
            for _ in range(T2):
                h_grad = [h.detach().requires_grad_(True) for h in hidden]

                # --- compute L(h) so we can include it in the nudged primitive ---
                # L depends on h via v = -output_scale * grad_x E(x, h).
                # To get dL/dh, h_grad must be IN the E_for_v call so that
                # the graph connects: L -> v -> grad_x E -> E_for_v -> h_grad.
                # create_graph=True is required so backward can compute d(grad_x E)/dh.
                x_req = x_flat.detach().requires_grad_(True)
                neurons_for_v = [x_req] + h_grad  # h_grad tensors IN graph (not .detach())
                E_for_v = self._energy(neurons_for_v).sum()
                # ∂L/∂h = 2(v-u)·∂v/∂h = 2(v-u)·(-output_scale·∂²E/∂x∂h) (mixed Hessian)
                grad_x_E = torch.autograd.grad(E_for_v, x_req, create_graph=True)[0]
                v = -scale * grad_x_E  # (B, 784)
                L = (v - ut).pow(2).mean()  # scalar, depends on h_grad via grad_x_E

                # Nudged primitive: Phi_nudged = (1-eps)*(1/2)||h||^2 + eps*(Phi_coupling - beta*L)
                # Equivalently: Phi_nudged = Phi_free - eps*beta*L
                neurons_free = [x_flat.detach()] + h_grad
                Phi_free = self._primitive(neurons_free).sum()
                Phi_nudged = Phi_free - eps * beta * L

                grads = torch.autograd.grad(Phi_nudged, h_grad)
                hidden = [g.detach() for g in grads]
                # Graph is discarded here — O(1) memory across steps

        return hidden  # fully detached

    def ep_gradient_step(self, x_flat, h_star, h_beta, beta, ut, explicit_grad=False):
        """
        Compute the full EP gradient estimator and accumulate into param.grad:

            ep_loss = (1/beta) * [E(h_beta) - E(h*)]  +  L(h_beta)

        where
            E terms: x detached, h detached, live theta  (implicit gradient)
            L(h_beta) = ||v(x, h_beta; theta) - ut||^2

        When explicit_grad=True, L(h_beta) contributes ∂L/∂θ via create_graph=True
        (full EP estimator including the explicit gradient through the velocity).
        When explicit_grad=False (default), L(h_beta) is detached from θ,
        giving implicit-only EP: just (E_β - E*)/β. The implicit gradient
        dominates by factor 1/(1-ρ) at high ρ, so dropping the explicit
        O(1) term is a small bias for much stabler gradients.

        Accumulates into param.grad (replaces flow_loss.backward()).
        """
        scale = self.output_scale

        # 1. Implicit gradient part: (E_beta - E_star) / beta
        x_det = x_flat.detach()
        neurons_beta = [x_det] + [h.detach() for h in h_beta]
        neurons_star = [x_det] + [h.detach() for h in h_star]
        E_beta = self._energy(neurons_beta)   # (B,)  live theta
        E_star = self._energy(neurons_star)   # (B,)  live theta

        # 2. Explicit gradient part: L(h_beta)
        #    create_graph controls whether ∂L/∂θ flows through the velocity.
        x_req = x_flat.detach().requires_grad_(True)
        neurons_for_v = [x_req] + [h.detach() for h in h_beta]
        E_for_v = self._energy(neurons_for_v)  # (B,)  live theta
        grad_x_E = torch.autograd.grad(E_for_v.sum(), x_req, create_graph=explicit_grad)[0]
        if not explicit_grad:
            grad_x_E = grad_x_E.detach()
        v_beta = -scale * grad_x_E  # velocity at h_beta
        L_beta = (v_beta - ut).pow(2).mean()

        # Full EP estimator
        ep_loss = (E_beta - E_star).mean() / beta + L_beta
        ep_loss.backward()

    # ------------------------------------------------------------------
    # Spring-clamped EP — no create_graph anywhere, O(1) memory
    # ------------------------------------------------------------------

    def _converge_ep_spring_free(self, x_t, T1, lambda_spring, record_trace=False):
        """
        Free phase for spring-clamped EP.

        Energy minimised: E_spring = E_int(x, h) + (λ/2)||x - x_t||²
        Primitive used:   Φ_spring = Φ_int(x, h) + (1-ε)½||x||² - ε(λ/2)||x - x_t||²

        x becomes a dynamic variable (springs back to x_t).
        Fully detached — O(1) memory, no create_graph.

        Returns:
            x_star  (B, D) detached
            h_star  list of (B, archi[i]) detached tensors
        """
        eps = self.epsilon_ep
        B = x_t.size(0)
        device = x_t.device

        x = x_t.detach().clone()
        hidden = [torch.zeros(B, self.archi[idx + 1], device=device)
                  for idx in range(len(self.archi) - 1)]

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T1):
                if record_trace:
                    entry = {"x": x[0].norm().item(),
                             "x_neurons": x[0].detach().cpu().tolist()}
                    for idx, h in enumerate(hidden):
                        entry[f"h{idx+1}"] = h[0].norm().item()
                        entry[f"h{idx+1}_neurons"] = h[0].detach().cpu().tolist()
                    trace.append(entry)

                x_grad = x.detach().requires_grad_(True)
                h_grad = [h.detach().requires_grad_(True) for h in hidden]
                neurons = [x_grad] + h_grad

                Phi_int = self._primitive(neurons).sum()
                x_kinetic = (1.0 - eps) * 0.5 * x_grad.pow(2).sum()  # (1-ε)||x||²/2 — matches hidden primitive
                spring = eps * (lambda_spring / 2.0) * (x_grad - x_t.detach()).pow(2).sum()
                Phi_spring = Phi_int + x_kinetic - spring

                all_vars = [x_grad] + h_grad
                grads = torch.autograd.grad(Phi_spring, all_vars)
                x = grads[0].detach()
                hidden = [g.detach() for g in grads[1:]]

        if record_trace:
            entry = {"x": x[0].norm().item(),
                     "x_neurons": x[0].detach().cpu().tolist()}
            for idx, h in enumerate(hidden):
                entry[f"h{idx+1}"] = h[0].norm().item()
                entry[f"h{idx+1}_neurons"] = h[0].detach().cpu().tolist()
            trace.append(entry)
            self._last_convergence_trace = trace

        return x, hidden  # x_star, h_star — fully detached

    def _converge_ep_spring_nudged(self, x_t, x_star, h_star, ut, beta, T2, lambda_spring,
                                    record_trace=False):
        """
        Nudged phase for spring-clamped EP.

        target_x = x_t + ut / (output_scale * lambda_spring)
        At equilibrium: v = output_scale * λ * (x* - x_t) = ut  ✓

        Nudge loss:    L = (output_scale * λ)² / 2 * ||x - target_x||²
        Φ_nudged = Φ_int - ε(λ/2)||x - x_t||² - ε*β*(output_scale*λ)²/2 * ||x - target_x||²

        No create_graph — the nudge term is purely quadratic in x.
        ∂L/∂h = 0, so hidden states feel nudge only through coupling.

        Returns:
            x_beta  (B, D) detached
            h_beta  list of detached tensors
            trace   list of per-step neuron dicts, or None if record_trace=False
        """
        eps = self.epsilon_ep
        scale = self.output_scale
        B = x_t.size(0)

        x_t_det = x_t.detach()
        target_x = (x_t_det + ut.detach() / (scale * lambda_spring)).detach()
        nudge_coeff = eps * beta * (scale * lambda_spring) ** 2 / 2.0

        x = x_star.detach().clone()
        hidden = [h.detach().clone() for h in h_star]

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T2):
                if record_trace:
                    entry = {"x": x[0].norm().item(),
                             "x_neurons": self._sample_neurons(x[0])}
                    for idx, h in enumerate(hidden):
                        entry[f"h{idx+1}"] = h[0].norm().item()
                        entry[f"h{idx+1}_neurons"] = self._sample_neurons(h[0])
                    trace.append(entry)

                x_grad = x.detach().requires_grad_(True)
                h_grad = [h.detach().requires_grad_(True) for h in hidden]
                neurons = [x_grad] + h_grad

                Phi_int = self._primitive(neurons).sum()
                x_kinetic = (1.0 - eps) * 0.5 * x_grad.pow(2).sum()  # (1-ε)||x||²/2 — matches hidden primitive
                spring = eps * (lambda_spring / 2.0) * (x_grad - x_t_det).pow(2).sum()
                nudge = nudge_coeff * (x_grad - target_x).pow(2).sum()
                Phi_nudged = Phi_int + x_kinetic - spring - nudge

                all_vars = [x_grad] + h_grad
                grads = torch.autograd.grad(Phi_nudged, all_vars)
                x = grads[0].detach()
                hidden = [g.detach() for g in grads[1:]]

        if record_trace:
            entry = {"x": x[0].norm().item(),
                     "x_neurons": self._sample_neurons(x[0])}
            for idx, h in enumerate(hidden):
                entry[f"h{idx+1}"] = h[0].norm().item()
                entry[f"h{idx+1}_neurons"] = self._sample_neurons(h[0])
            trace.append(entry)

        return x, hidden, trace  # x_beta, h_beta, trace — fully detached

    def ep_spring_gradient_step(self, neurons_star, neurons_beta, beta):
        """
        EP parameter gradient for spring-clamped EP.

        ep_loss = (E_beta - E_star) / beta

        x and h are detached (they are neuron values, not parameters).
        Only θ (model weights) is live — so ep_loss.backward() gives ∂ep_loss/∂θ.
        ∂L/∂θ = 0 because L depends only on x, not on θ.

        neurons_star: [x*.detach(), h1*.detach(), ...]  — live θ only
        neurons_beta: [x_β.detach(), h1_β.detach(), ...]  — live θ only
        """
        E_star = self._energy(neurons_star)   # (B,)  live θ
        E_beta = self._energy(neurons_beta)   # (B,)  live θ
        ep_loss = (E_beta - E_star).mean() / beta
        ep_loss.backward()


    def velocity_at_h(self, x, hidden):
        """
        Compute velocity -output_scale * ∇_x E(x, h)  for given hidden states h,
        without re-running convergence. Useful for logging after the free phase.

        x:      (B, 1, 28, 28) image tensor (or flat — will be flattened internally)
        hidden: list of detached hidden state tensors [h1, h2, ...]
        Returns: velocity (B, 1, 28, 28)
        """
        B = x.size(0)
        with torch.enable_grad():
            x_flat_req = x.view(B, -1).detach().requires_grad_(True)
            neurons = [x_flat_req] + [h.detach() for h in hidden]
            E = self._energy(neurons)  # (B,)
            v_flat = -self.output_scale * torch.autograd.grad(E.sum(), x_flat_req)[0]
        return v_flat.view_as(x).detach()  # (B, 1, 28, 28)

    def velocity_energy_gd(self, x, t, h_steps=None):
        """
        Compute velocity via energy gradient descent (neuromorphic inference mode).

        x is FIXED, only hidden states h evolve to equilibrium via the
        primitive dynamics. Velocity includes x's quadratic self-regularisation
        to match what x experiences during spring-clamped training:

            v = -output_scale * grad_x [E(x, h*) + (1/2)||x||^2]
              = output_scale * (grad_x coupling - x)

        Without the (1/2)||x||^2 term, x has no restoring force and pixels
        with large coupling gradients diverge (causing speckle artifacts).

        Args:
            x: (B, 1, 28, 28) input tensor (will be flattened internally)
            t: time (unused, kept for interface compatibility)
            h_steps: number of h convergence steps (default: self.T)
        Returns:
            velocity (B, 1, 28, 28), detached
        """
        if h_steps is None:
            h_steps = self.T
        B = x.size(0)
        x_flat = x.detach().view(B, -1)
        h_star = self._converge_ep_free(x_flat, h_steps)

        with torch.enable_grad():
            x_flat_req = x_flat.clone().requires_grad_(True)
            neurons = [x_flat_req] + [h.detach() for h in h_star]
            # E_int includes x quadratic, matching spring training's x_kinetic
            E_int = self._energy(neurons) + 0.5 * x_flat_req.pow(2).sum(dim=1)
            v_flat = -self.output_scale * torch.autograd.grad(E_int.sum(), x_flat_req)[0]
        return v_flat.view_as(x).detach()

    def potential(self, x, t, record_trace=False):
        """
        Compute V(x) = E(x, s*) · output_scale.
        x: (B, 1, 28, 28) image tensor
        """
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, 784)

        x_req = x_flat if x_flat.requires_grad else x_flat.clone().requires_grad_(True)

        if self.learning_mode == 'deq':
            hidden = self._converge_deq(x_req, record_trace=record_trace)
        elif self.learning_mode == 'spring':
            # Spring EP: x is dynamic, trace includes x norms/neurons alongside h
            # TODO - check if we should be doing something with returned x in this case - maybe like in generations things will be very different?
            _, hidden = self._converge_ep_spring_free(
                x_req, self.T, self.lambda_spring, record_trace=record_trace)
        elif self.learning_mode == 'ep':
            hidden = self._converge_ep_free(x_req, self.T, record_trace=record_trace)
        else:  # 'bptt'
            hidden = self._converge(x_req, record_trace=record_trace)

        neurons = [x_req] + hidden
        E = self._energy(neurons)

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
            B = x.size(0)
            x_flat = x.detach().view(B, -1)
            x_star, _ = self._converge_ep_spring_free(x_flat, self.T, self.lambda_spring)
            v_flat = self.output_scale * self.lambda_spring * (x_star - x_flat)
            return v_flat.view_as(x)
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
        """Same signature as other model wrappers."""
        if return_potential:
            return self.potential(x, t)
        elif getattr(self, '_gen_energy_gd', False):
            return self.velocity_energy_gd(x, t)
        else:
            return self.velocity(x, t)

    def save_convergence_plot(self, save_dir, step):
        """Save norm convergence plot."""
        trace = self._last_convergence_trace
        if trace is None:
            return

        timesteps = list(range(len(trace)))
        fig, ax = plt.subplots(figsize=(8, 5))

        # Get hidden layer names from first trace entry
        hidden_keys = [k for k in trace[0].keys() if not k.endswith('_neurons')]

        for key in hidden_keys:
            values = [t[key] for t in trace]
            ax.plot(timesteps, values, label=f"||{key}||", linewidth=1.5)

        ax.set_xlabel("Convergence step t")
        ax.set_ylabel("||hⁿ|| (single sample)")
        ax.set_title(f"Hidden state convergence (training step {step})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(save_dir, f"convergence_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[convergence_plot] Saved {path}")

    def save_layer_activations_plot(self, save_dir, step):
        """Save per-neuron convergence traces."""
        trace = self._last_convergence_trace
        if trace is None:
            return

        # Get neuron trace keys
        neuron_keys = [k for k in trace[0].keys() if k.endswith('_neurons')]
        if not neuron_keys:
            return

        timesteps = list(range(len(trace)))
        n_layers = len(neuron_keys)
        cols = min(n_layers, 3)
        rows = (n_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

        for idx, key in enumerate(neuron_keys):
            ax = axes[idx // cols, idx % cols]
            label = key.replace('_neurons', '')
            num_neurons = len(trace[0][key])
            for n in range(num_neurons):
                values = [t[key][n] for t in trace]
                ax.plot(timesteps, values, linewidth=0.5, alpha=0.3)
            ax.set_xlabel("Convergence step t")
            ax.set_ylabel(f"{label} value")
            ax.set_title(f"{label} all neurons")
            # ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(neuron_keys), rows * cols):
            axes[idx // cols, idx % cols].set_visible(False)

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
          self._last_spring_free_final  — dict of final free-phase h-neuron values
          self._last_nudge_pos_trace    — list of per-step dicts, positive nudge
          self._last_nudge_neg_trace    — list of per-step dicts, negative nudge (or None)

        Layout: one subplot per hidden layer (h1, h2, ...).
        Each panel: 8 neuron displacement lines, positive=solid, negative=dashed.
        """
        free_final = getattr(self, '_last_spring_free_final', None)
        trace_pos  = getattr(self, '_last_nudge_pos_trace', None)
        if free_final is None or trace_pos is None:
            return
        trace_neg = getattr(self, '_last_nudge_neg_trace', None)

        # Discover which hidden layers are present
        neuron_keys = sorted(
            [k for k in trace_pos[0].keys() if k.endswith('_neurons') and k != 'x_neurons'],
            key=lambda k: k
        )
        if not neuron_keys:
            return

        n_layers = len(neuron_keys)
        cols = min(n_layers, 3)
        rows = (n_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx, key in enumerate(neuron_keys):
            ax = axes[idx // cols, idx % cols]
            label = key.replace('_neurons', '')
            ref = free_final.get(key, [0.0] * 8)
            n_neurons = len(ref)

            timesteps_pos = list(range(len(trace_pos)))
            for n in range(n_neurons):
                col = colors[n % len(colors)]
                disp_pos = [entry[key][n] - ref[n] for entry in trace_pos]
                ax.plot(timesteps_pos, disp_pos, color=col, linewidth=1.0,
                        alpha=0.85, linestyle='-',
                        label=f'n{n}' if idx == 0 else None)

            if trace_neg is not None:
                timesteps_neg = list(range(len(trace_neg)))
                for n in range(n_neurons):
                    col = colors[n % len(colors)]
                    disp_neg = [entry[key][n] - ref[n] for entry in trace_neg]
                    ax.plot(timesteps_neg, disp_neg, color=col, linewidth=1.0,
                            alpha=0.55, linestyle='--')

            ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
            ax.set_xlabel("Nudge step t")
            ax.set_ylabel(f"\u0394{label} (displacement from h*)")
            ax.set_title(f"{label} nudge displacement")
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_layers, rows * cols):
            axes[idx // cols, idx % cols].set_visible(False)

        # Single legend on first panel
        axes[0, 0].legend(fontsize=6, ncol=2, title="neuron (solid=+β, dash=−β)")
        fig.suptitle(f'Nudge displacement from h* (training step {step})', fontsize=13)
        fig.tight_layout()

        path = os.path.join(save_dir, f"nudge_displacement_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[nudge_displacement] Saved {path}")

    def _converge_detached(self, x_flat):
        """
        Run dynamics to convergence WITHOUT building the BPTT graph.
        Each step is detached, so this is cheap and uses minimal memory.
        Used for Jacobian spectral radius computation.
        """
        B = x_flat.size(0)
        device = x_flat.device
        hidden = [torch.zeros(B, self.archi[idx+1], device=device)
                  for idx in range(len(self.archi)-1)]

        with torch.enable_grad():
            for t in range(self.T):
                h_grad = [h.detach().requires_grad_(True) for h in hidden]
                neurons = [x_flat.detach()] + h_grad
                Phi = self._primitive(neurons).sum()
                grads = torch.autograd.grad(Phi, h_grad)
                hidden = [g.detach() for g in grads]

        return hidden

    def compute_jacobian_spectral_radius(self, x_flat_single, hidden_star, n_iters=20):
        """
        Estimate spectral radius ρ of J = ∇²_h Φ at converged state h*.
        Uses power iteration. J is symmetric (Hessian), so ρ = ||J||_op.

        ρ is the per-step BPTT gradient decay factor:
          - ρ^T = total gradient decay over T unrolling steps
          - ρ < 1 required for convergence, but causes vanishing gradients

        Returns: (rho, rho_history)
        """
        device = x_flat_single.device
        n_hidden = len(self.archi) - 1

        # Random unit vector (same shapes as hidden states)
        v = [torch.randn(1, self.archi[i+1], device=device) for i in range(n_hidden)]
        v_norm = sum(vi.pow(2).sum() for vi in v).sqrt()
        v = [vi / v_norm for vi in v]

        rho_history = []
        for _ in range(n_iters):
            h = [hs.detach().clone().requires_grad_(True) for hs in hidden_star]
            neurons = [x_flat_single.detach()] + h
            Phi = self._primitive(neurons).sum()
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

