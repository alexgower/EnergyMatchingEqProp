# File: network_pcn.py
# PCN (Predictive Coding Network) energy dynamics for CIFAR-10 (3×32×32).
#
# Stage 2 of the FlowEqProp project: replaces feedforward backprop with
# PCN-style energy relaxation to compute the velocity v(x).
#
# Architecture: VGG5 decomposed into L=6 PCN layers, each f_k = conv + act + pool.
# Energy: E(x, h, o) = Σ_k ½||f_k(h_{k-1}) - h_k||² + γ·o
# Velocity: v(x) = -output_scale · (1/γ) · ∂_x E(x, h*, o*)
#
# In the small-γ limit: v → -output_scale · ∇_x F(x), exactly recovering
# Energy Matching with scalar potential V(x) = output_scale · F(x).
#
# Two parameterizations of the relaxation dynamics:
#   1. h-param (fallback): optimize hidden states h_k, Hessian ill-conditioned,
#      requires float64 and many CG steps.
#   2. e-param (default): optimize prediction errors e_k = f_k(h_{k-1}) - h_k,
#      Hessian ≈ I, works in float32 with CG=1-3.
#
# Relaxation uses async even/odd updates (Scellier et al., 2026, App A.1)
# with SiLU + gradient descent (not PGD, which is ReLU-specific).
#
# Possible TODO items:
#   - ReLU + PGD/mod-PGD (currently SiLU + gradient descent) dynamics
#
# Backward pass for parameter gradients uses custom IFT (implicit function
# theorem) at the equilibrium point. In e-space where ∂E/∂e = 0:
#   ∂e*/∂θ = -(∂²E/∂e²)⁻¹ · ∂²E/(∂e∂θ)
# Solved via conjugate gradient with autograd Hessian-vector products.
# NOTE: The HVP computation passes through second derivatives of activation
# functions and Jacobian products, which can spike on outlier batches.
# EP (Stage 3) computes the *same* gradient but via a finite-difference of two
# first-order evaluations, avoiding the numerically fragile Hessian entirely.

import torch
import torch.nn as nn
import logging
from torch.autograd import Function


def soft_clamp(x, clamp_val):
    """Tanh-based clamp: output in [-clamp_val, clamp_val]."""
    return clamp_val * torch.tanh(x / clamp_val)


##############################################################################
# Utilities: flatten/unflatten and error ↔ hidden conversion
##############################################################################

def _flatten_hiddens(hiddens):
    """Flatten list of L tensors (different shapes) into one vector per sample."""
    return torch.cat([h.reshape(h.size(0), -1) for h in hiddens], dim=1)


def _unflatten_hiddens(h_flat, shapes):
    """Inverse of _flatten_hiddens."""
    hiddens = []
    offset = 0
    for shape in shapes:
        # shape = (B, C, H, W) or (B, D); product of shape[1:] gives
        # the per-sample element count (excluding batch dimension)
        numel = 1
        for s in shape[1:]:
            numel *= s
        h = h_flat[:, offset:offset + numel].reshape(shape)
        hiddens.append(h)
        offset += numel
    return hiddens


def _errors_to_hiddens(errors, pcn, x):
    """
    Reconstruct hidden states from prediction errors via forward scan.

    Given errors e_k = f_k(parents_k) - h_k, reconstruct h_k = f_k(parents_k) - e_k.
    This is a sequential scan in topological order: parents of node k always
    precede it, so they are already reconstructed when node k is visited
    (chain AND DAG topologies — the graph is read via pcn.predict).

    Args:
        errors: list of L error tensors [e_1, ..., e_L].
        pcn: PCNEnergyModelBase (provides layers, parents and L).
        x: input images (B, C, H, W).

    Returns: list of L hidden state tensors [h_1, ..., h_L].
    """
    hiddens = []
    for k in range(pcn.L):
        pred = pcn.predict(k, x, hiddens)
        hiddens.append(pred - errors[k])
    return hiddens


def _hiddens_to_errors(hiddens, pcn, x):
    """
    Compute prediction errors from hidden states.

    e_k = f_k(h_{k-1}) - h_k for each layer k.

    Args:
        hiddens: list of L hidden state tensors.
        pcn: PCNEnergyModelBase.
        x: input images.

    Returns: list of L error tensors [e_1, ..., e_L].
    """
    errors = []
    for k in range(pcn.L):
        # Uses the actual dynamical parent values (not a fresh feedforward)
        pred = pcn.predict(k, x, hiddens)
        errors.append(pred - hiddens[k])
    return errors


def _compute_energy_from_errors(errors, pcn, x, gamma):
    """
    Compute PCN energy from errors: E = Σ_k ½||e_k||² + γ·o(e).

    The output o depends on errors through the forward scan
    h_k = f_k(h_{k-1}) - e_k, so o = h_L.

    Args:
        errors: list of L error tensors (with grad tracking if needed).
        pcn: PCNEnergyModelBase.
        x: input images.
        gamma: clamp strength.

    Returns: scalar energy (summed over batch).
    """
    # Sum of squared errors: Σ_k ½||e_k||²
    energy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    for e_k in errors:
        energy = energy + 0.5 * (e_k ** 2).sum()

    # Linear clamp γ·o where o = h_L (reconstructed from errors)
    hiddens = _errors_to_hiddens(errors, pcn, x)
    output = hiddens[-1]  # (B, 1)
    energy = energy + gamma * output.sum()
    return energy


##############################################################################
# IFT backward: correct parameter gradients at equilibrium
##############################################################################

def _cg_solve(hvp_fn, b, n_steps=10, tol=1e-5):
    """
    Conjugate gradient solver: find w such that A·w = b.

    CG iteratively improves w by stepping along conjugate directions {p_i}
    that are A-orthogonal (p_i^T A p_j = 0). Each step:
      1. Compute A·p via hvp_fn (the only access to A we need)
      2. Step size α = r^T r / p^T A p (exact line minimisation of ||Aw-b||²)
      3. Update solution w ← w + α·p, residual r ← r - α·A·p
      4. New direction p ← r + β·p_old (β chosen for A-conjugacy)
    For A = I (our e-param Hessian), converges in 1 step exactly.

    Args:
        hvp_fn: callable, computes A·v (Hessian-vector product)
        b: target vector (B, D)
        n_steps: max CG iterations
        tol: convergence threshold

    Returns: w ≈ A⁻¹·b
    """
    w = torch.zeros_like(b)
    r = b.clone()  # residual = b - A·w = b (since w=0)
    p = r.clone()  # search direction
    rr = (r * r).sum()

    for i in range(n_steps):
        Ap = hvp_fn(p)
        pAp = (p * Ap).sum()
        if pAp.abs() < 1e-12:
            break
        alpha = rr / pAp
        w = w + alpha * p
        r = r - alpha * Ap
        rr_new = (r * r).sum()
        if rr_new.sqrt() < tol:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return w


class IFTEquilibriumHiddens(Function):
    """
    Autograd function that attaches IFT-correct gradients to equilibrium states.

    Forward: passes through the detached h* unchanged.
    Backward: uses the implicit function theorem to compute ∂h*/∂θ.

    At equilibrium, g(h*, θ) = ∂E/∂h = 0. The IFT gives:
        ∂h*/∂θ = -(∂g/∂h)⁻¹ · ∂g/∂θ
    i.e. how h* shifts when θ changes = (inverse Hessian) × (how the energy
    gradient changes with θ). We never form this matrix — see below.

    In the backward pass, we receive upstream gradient v = ∂L/∂h*.
    We need to compute: ∂L/∂θ = v^T · ∂h*/∂θ = -[(∂g/∂h)⁻ᵀ · v]^T · ∂g/∂θ

    Steps:
        1. Solve (∂g/∂h)^T · w = v  via CG (using Hessian-vector products).
           This IS computing w = (∂g/∂h)⁻ᵀ · v, but via fast iterative HVPs
           instead of forming the full D×D Hessian matrix.
        2. Compute ∂L/∂θ = -w^T · ∂g/∂θ  via autograd
    """

    @staticmethod
    def forward(ctx, h_flat, pcn, x, gamma, n_cg_steps, shapes):
        """
        Args:
            h_flat: flattened equilibrium states (B, D), requires_grad=True
            pcn: PCNEnergyModelBase (not saved — we access it via ctx)
            x: input images (B, C, H, W)
            gamma: clamp strength
            n_cg_steps: CG iterations for the linear solve
            shapes: list of hidden state shapes for unflattening
        """
        # ctx ("context") is a mailbox between forward and backward.
        # Store everything backward() will need to compute the IFT gradients.
        # Python objects go on ctx directly; tensors use save_for_backward
        # (which lets PyTorch manage their memory lifecycle).
        ctx.pcn = pcn
        ctx.gamma = gamma
        ctx.n_cg_steps = n_cg_steps
        ctx.shapes = shapes
        ctx.save_for_backward(x, h_flat)  # retrievable later as ctx.saved_tensors
        # Forward value is just h_flat unchanged — the custom part is backward only
        return h_flat.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: ∂L/∂h_flat (B, D) — upstream gradient from the velocity
        computation flowing back through E(x, h*, θ) → ∂E/∂x → v → loss.

        We need to propagate this through the implicit h*(θ) dependence.

        Precision note (h-param only): The Hessian ∂²E/∂h² has O(1) blocks
        from both ||f_k(h_{k-1}) - h_k||² and ||f_{k+1}(h_k) - h_{k+1}||²
        that partially cancel. In float32, this cancellation destroys
        significant digits, especially for early layers (0-1) where the
        signal passes through many such cancellations. Float64 is required.
        (The e-param path avoids this entirely: ∂²E/∂e² ≈ I, no cancellation.)
        """
        x, h_flat = ctx.saved_tensors
        pcn = ctx.pcn
        gamma = ctx.gamma
        shapes = ctx.shapes

        with torch.enable_grad():
            # 1. Reconstruct h* with grad tracking for the Hessian
            h_leaf = h_flat.detach().requires_grad_(True)
            hiddens = _unflatten_hiddens(h_leaf, shapes)

            # 2. Compute g(h*, θ) = ∂E/∂h at equilibrium, with create_graph=True
            #    so we can differentiate g w.r.t. h (for Hessian) and θ
            E = pcn.compute_energy(x.detach(), hiddens, gamma)
            g = torch.autograd.grad(E, h_leaf, create_graph=True)[0]

            # 3. Solve H^T · w = grad_output via CG, where H = ∂g/∂h = ∂²E/∂h².
            #    hvp_fn computes H^T · v without forming H, using double autograd:
            #      g = ∂E/∂h (vector, has a computational graph from create_graph=True)
            #      gv = g^T · v = Σ_i g_i · v_i  (scalar)
            #      ∂(gv)/∂h = Σ_i v_i · ∂g_i/∂h = (∂g/∂h)^T · v = H^T · v
            #    So one autograd call gives us H^T · v — exactly what CG needs.
            def hvp_fn(v):
                gv = (g * v).sum()
                hvp = torch.autograd.grad(gv, h_leaf, retain_graph=True)[0]
                return hvp

            w = _cg_solve(hvp_fn, grad_output.detach(), n_steps=ctx.n_cg_steps)

            # 4. Compute ∂L/∂θ = -w^T · ∂g/∂θ for all parameters
            all_params = list(pcn.parameters())
            ift_grads = torch.autograd.grad(
                (g * (-w)).sum(), all_params,
                retain_graph=False, allow_unused=True
            )
            # Manually write IFT gradients into each param's .grad field.
            # We can't return them via autograd's normal mechanism (returning
            # from backward()) because θ wasn't a forward() input — it lives
            # inside ctx.pcn. So we accumulate directly into .grad, using
            # add_ (not replace) in case other loss terms already contributed.
            for p, ift_g in zip(all_params, ift_grads):
                if ift_g is not None:
                    if p.grad is None:
                        p.grad = ift_g
                    else:
                        p.grad.add_(ift_g)

        # Return None for all forward args
        return None, None, None, None, None, None


class IFTEquilibriumErrors(Function):
    """
    IFT backward in error-parameterized space.

    Same role as IFTEquilibriumHiddens but operates on prediction errors
    e_k = f_k(h_{k-1}) - h_k instead of hidden states h_k.

    Advantage: the Hessian ∂²E/∂e² ≈ I + γ·(small perturbation), so CG
    converges in 2-5 steps (vs 10+ in h-space) and float32 is sufficient
    (no catastrophic cancellation from subtracting O(1) quantities).
    """

    @staticmethod
    def forward(ctx, e_flat, pcn, x, gamma, n_cg_steps, shapes):
        # ctx ("context") is a mailbox between forward and backward.
        # Store everything backward() will need to compute the IFT gradients.
        # Python objects go on ctx directly; tensors use save_for_backward
        # (which lets PyTorch manage their memory lifecycle).
        ctx.pcn = pcn
        ctx.gamma = gamma
        ctx.n_cg_steps = n_cg_steps
        ctx.shapes = shapes
        ctx.save_for_backward(x, e_flat)  # retrievable later as ctx.saved_tensors
        # Forward value is just e_flat unchanged — the custom part is backward only
        return e_flat.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        IFT backward in e-space.

        At equilibrium, g(e*, θ) = ∂E/∂e = 0. The IFT gives:
            ∂e*/∂θ = -(∂g/∂e)⁻¹ · ∂g/∂θ

        The Hessian ∂g/∂e = ∂²E/∂e² ≈ I (from Σ ½||e_k||²) plus
        γ · ∂²o/∂e² (bounded perturbation from the clamp term).
        """
        x, e_flat = ctx.saved_tensors
        pcn = ctx.pcn
        gamma = ctx.gamma
        shapes = ctx.shapes

        with torch.enable_grad():
            # Step 1: Re-create errors as leaf tensors so we can differentiate E w.r.t. e
            e_leaf = e_flat.detach().requires_grad_(True)
            errors = _unflatten_hiddens(e_leaf, shapes)

            # Step 2: Compute g(e) = ∂E/∂e. At equilibrium g=0, but we need
            # the computational graph of g for the Hessian and ∂g/∂θ below.
            E = _compute_energy_from_errors(errors, pcn, x.detach(), gamma)
            g = torch.autograd.grad(E, e_leaf, create_graph=True)[0]

            # Step 3: Solve H^T · w = grad_output via CG, where H = ∂²E/∂e² ≈ I.
            #    hvp_fn computes H^T · v without forming H, using double autograd:
            #      g = ∂E/∂e (vector, has computational graph from create_graph=True)
            #      gv = g^T · v = Σ_i g_i · v_i  (scalar)
            #      ∂(gv)/∂e = Σ_i v_i · ∂g_i/∂e = (∂g/∂e)^T · v = H^T · v
            #    So one autograd call gives us H^T · v — exactly what CG needs.
            #    Since H ≈ I here, CG converges in ~1 step.
            def hvp_fn(v):
                gv = (g * v).sum()
                hvp = torch.autograd.grad(gv, e_leaf, retain_graph=True)[0]
                return hvp

            w = _cg_solve(hvp_fn, grad_output.detach(), n_steps=ctx.n_cg_steps)

            # Step 4: Compute ∂L/∂θ = -w^T · ∂g/∂θ.
            # (g * (-w)).sum() is the scalar w^T·g; autograd differentiates
            # this w.r.t. θ (pcn.parameters), giving the IFT parameter gradients.
            all_params = list(pcn.parameters())
            ift_grads = torch.autograd.grad(
                (g * (-w)).sum(), all_params,
                retain_graph=False, allow_unused=True
            )
            # Manually write IFT gradients into each param's .grad field.
            # We can't return them via autograd's normal mechanism (returning
            # from backward()) because θ wasn't a forward() input — it lives
            # inside ctx.pcn. So we accumulate directly into .grad, using
            # add_ (not replace) in case other loss terms already contributed.
            for p, ift_g in zip(all_params, ift_grads):
                if ift_g is not None:
                    if p.grad is None:
                        p.grad = ift_g
                    else:
                        p.grad.add_(ift_g)

        # Return None for all inputs (grads go to params via .grad, not through autograd)
        return None, None, None, None, None, None


##############################################################################
# PCN energy + dynamics engine (topology-agnostic base class)
##############################################################################

class PCNEnergyModelBase(nn.Module):
    """
    Topology-agnostic PCN energy + dynamics engine.

    Holds NO architecture of its own: a subclass sets self.layers (ModuleList of
    prediction functions f_k), self.L, and self.parents (parents[k] = node
    indices feeding f_k; -1 denotes the input x), then inherits the full engine
    here -- energy, relaxation (h- and error-parameterized), spring/EP phases and
    IFT gradients -- all routed through self.predict(k, x, hiddens).

    Concrete architectures:
      PCNCNNEnergyModel (network_pcn_cnn.py)   -- VGG5 as a linear-chain PCN.
      PCNUNetEnergyModel  (network_pcn_unet.py) -- DAG over UNet+ViT blocks.

    Energy:  E_int(x, h, o) = sum_k 0.5*||f_k(parents_k) - h_k||^2 + gamma*o,
    with h_0 = x (input) and the final node's scalar output = o.
    """

    def feedforward_init(self, x):
        """
        Pure forward pass through all layers → list of L hidden states.

        At these values, all residuals r_k = f_k(h_{k-1}) - h_k = 0 exactly.
        This is the free-phase equilibrium (Scellier Eq. 9).

        Returns: list of L tensors [h_1, h_2, ..., h_L], each detached.
        """
        hiddens = []
        for k in range(self.L):
            # Topological order: parents of node k are already in hiddens.
            h = self.predict(k, x, hiddens)
            hiddens.append(h.detach().clone())
        return hiddens

    def _prepare_input(self, k, parent_vals):
        """Combine gathered parent states into the input of f_k.

        Chain default: exactly one parent, flattened before the final Linear
        (k == L-1). DAG subclasses override this or predict() (e.g. the UNet
        decoder nodes channel-concat [prev, skip]).
        """
        h_prev = parent_vals[0]
        if k == self.L - 1:
            h_prev = h_prev.view(h_prev.size(0), -1)
        return h_prev

    def predict(self, k, x, hiddens):
        """Topology-aware prediction f_k(parents_k) — the single hook through
        which every energy/relaxation/scan routine reads the graph structure.

        Gathering is generic (it just reads self.parents, so any number of
        parents works), but the default COMBINING step (_prepare_input) is the
        CHAIN case: exactly one parent. So a DAG subclass overrides either
        _prepare_input (if only the combining differs) or predict itself —
        PCNUNetEnergyModel overrides predict, because it must both channel-concat
        [prev, skip] and inject the constant time embedding into each block.
        PCNCNNEnergyModel overrides neither: the defaults here ARE the chain.

        hiddens need only be populated up to k-1 (nodes are in topological
        order, so all parents of node k precede it).
        """
        vals = [x if p < 0 else hiddens[p] for p in self.parents[k]]
        return self.layers[k](self._prepare_input(k, vals))

    def compute_energy(self, x, hiddens, gamma):
        """
        Compute PCN energy: E_int = Σ_k ½||r_k||² + γ·o

        The sum-of-squared-residuals measures prediction error across all layers.
        The linear clamp γ·o injects a constant "current" at the output node,
        whose backward propagation through residuals implements backprop of F(x).

        Args:
            x: input images (B, C, H, W), should have requires_grad if
               computing ∂E/∂x for velocity.
            hiddens: list of L hidden state tensors.
            gamma: linear clamp strength.

        Returns: scalar energy (summed over batch).
        """
        energy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for k in range(self.L):
            pred = self.predict(k, x, hiddens)
            r_k = pred - hiddens[k]
            energy = energy + 0.5 * (r_k ** 2).sum()
        # Linear clamp: γ · o (sum over batch)
        output = hiddens[-1]  # (B, 1)
        energy = energy + gamma * output.sum()
        return energy

    def compute_energy_per_sample(self, x, hiddens, gamma):
        """Per-sample PCN energy => (B,). Same quantity as compute_energy but
        reduced over feature dims only, keeping the batch dim.

        compute_energy sums over the batch, which is correct for the relaxation
        dynamics (∂/∂x_i Σ_j E_j = ∂E_i/∂x_i, so the batch sum is harmless for
        those gradients) but destroys per-sample values. Per-sample scoring —
        the CD energy term and the negative sampler's per-sample force ∇_x V —
        needs this reduction instead. Routed through predict() so it is
        topology-agnostic (chain AND DAG)."""
        energy = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for k in range(self.L):
            pred = self.predict(k, x, hiddens)
            r_k = pred - hiddens[k]
            energy = energy + 0.5 * (r_k ** 2).flatten(1).sum(1)
        output = hiddens[-1]  # (B, 1)
        energy = energy + gamma * output.flatten(1).sum(1)
        return energy

    def compute_residuals(self, x, hiddens):
        """
        Compute per-layer residual norms ||r_k|| for diagnostics.

        Returns: list of L scalar values (batch-averaged L2 norms).
        Useful for detecting vanishing/exploding backward signal.
        At equilibrium: r_k = J_{k+1}^T · r_{k+1}, so residuals naturally
        decay from output toward input, scaled by Jacobian transposes:
          - ||r_L|| = γ (constant, by equilibrium of output layer)
          - ||r_k|| ∝ Π_{j=k+1}^{L} ||J_j^T|| · γ (decreases toward input)
          - Moderate decay = healthy (Jacobians have spectral norm ≲ 1)
          - Extreme decay = vanishing (early layers get no gradient signal)
          - Growth toward input = exploding (Jacobians have spectral norm > 1)
        """
        norms = []
        for k in range(self.L):
            with torch.no_grad():
                pred = self.predict(k, x, hiddens)
                r_k = pred - hiddens[k]
                # Per-sample L2 norm, then batch average
                norm = r_k.view(r_k.size(0), -1).norm(dim=1).mean().item()
                norms.append(norm)
        return norms

    def relax_hiddens(self, x, gamma, K_h, dt, async_mode=True,
                      init_mode="feedforward"):
        """
        Relax hidden states to equilibrium via gradient descent on E_int.

        Uses async even/odd traversal (Scellier App A.1): each iteration
        updates even-indexed layers first, then odd-indexed layers.

        All hidden states are FULLY DETACHED — no computation graph through
        the relaxation. Parameter gradients are handled separately by the
        IFT backward (IFTEquilibriumHiddens) in the velocity computation.

        Args:
            x: input images (B, C, H, W).
            gamma: linear clamp strength.
            T_free: number of relaxation iterations.
            dt: step size for gradient descent.
            async_mode: if True, even/odd async updates.
            init_mode: "feedforward" (recommended), "zeros", or "random".

        Returns:
            hiddens: list of L equilibrium hidden state tensors (detached).
            diagnostics: dict with "energy" and "residual_norms".
        """
        x_detached = x.detach()

        # ---- Initialize hidden states ----
        if init_mode == "feedforward":
            hiddens = self.feedforward_init(x_detached)
        elif init_mode == "zeros":
            hiddens = self._make_zero_hiddens(x_detached)
        elif init_mode == "random":
            hiddens = self._make_random_hiddens(x_detached)
        else:
            raise ValueError(f"Unknown pcn_init_mode: {init_mode}")

        diagnostics = {"energy": [], "residual_norms": []}

        # Record initial state
        with torch.no_grad():
            diagnostics["energy"].append(
                self.compute_energy(x_detached, hiddens, gamma).item()
            )
            diagnostics["residual_norms"].append(
                self.compute_residuals(x_detached, hiddens)
            )

        # ---- Relaxation loop (fully detached) ----
        for step in range(K_h):
            self._relax_sweep_h(x_detached, hiddens, gamma, dt, async_mode)

            # Record diagnostics
            with torch.no_grad():
                diagnostics["energy"].append(
                    self.compute_energy(x_detached, hiddens, gamma).item()
                )
                diagnostics["residual_norms"].append(
                    self.compute_residuals(x_detached, hiddens)
                )

        return hiddens, diagnostics

    def _update_layer_gd(self, x, hiddens, k, gamma, dt):
        """
        Single gradient descent step for layer k:
            h_k ← h_k - dt · ∂E_int/∂h_k

        Fully detached — parameter gradients are handled by the IFT backward
        (IFTEquilibriumHiddens) after relaxation converges.

        Args:
            x: input (detached).
            hiddens: current hidden states (list of L tensors).
            k: layer index to update.
            gamma: linear clamp strength.
            dt: step size.

        Returns: updated h_k (detached).
        """
        with torch.enable_grad():
            # We want ∂E/∂h_k treating all OTHER h_j as constants.
            # detach() cuts h_k from any prior graph, requires_grad_(True)
            # makes it a fresh leaf that autograd can differentiate w.r.t.
            # We then splice it into a shallow copy of the list so that
            # compute_energy sees the grad-tracked h_k at position k.
            h_k = hiddens[k].detach().requires_grad_(True)
            hiddens_with_grad = list(hiddens)
            hiddens_with_grad[k] = h_k
            E = self.compute_energy(x, hiddens_with_grad, gamma)
            grad = torch.autograd.grad(E, h_k)[0]
        return (h_k - dt * grad).detach()

    def _update_nodes_gd(self, x, hiddens, ks, gamma, dt):
        """
        Batched GD step for a SET of nodes: ONE energy evaluation yields
        ∂E/∂h_k for every k in ks (vs. one full-graph evaluation per node in
        _update_layer_gd — an ~L/2× per-sweep saving; decisive for the
        21-node UNet DAG).

        All nodes in ks update from the same pre-pass state (Jacobi within
        the set). On the chain this is numerically identical to the
        sequential per-node loop, because same-parity nodes share no energy
        term. On a DAG with same-parity couplings (UNet skip pairs) the
        relaxation trajectory differs but the fixed point is the same —
        and simultaneous updates are the physically-parallel dynamics the
        EP story assumes. Mutates `hiddens` in place.
        """
        if not ks:
            return
        with torch.enable_grad():
            leaves = [hiddens[k].detach().requires_grad_(True) for k in ks]
            tmp = list(hiddens)
            for k, leaf in zip(ks, leaves):
                tmp[k] = leaf
            E = self.compute_energy(x, tmp, gamma)
            grads = torch.autograd.grad(E, leaves)
        for k, leaf, g in zip(ks, leaves, grads):
            hiddens[k] = (leaf - dt * g).detach()

    def _relax_sweep_h(self, x, hiddens, gamma, dt, async_mode):
        """One relaxation sweep over all nodes (h-param): 2 energy
        evaluations in async even/odd mode, 1 in synchronous (full Jacobi)."""
        if async_mode:
            for parity in (0, 1):
                self._update_nodes_gd(
                    x, hiddens, [k for k in range(self.L) if k % 2 == parity],
                    gamma, dt)
        else:
            self._update_nodes_gd(x, hiddens, list(range(self.L)), gamma, dt)

    def _make_zero_hiddens(self, x):
        """Create zero-initialized hidden states with correct shapes."""
        shapes_hiddens = self.feedforward_init(x)
        return [torch.zeros_like(h) for h in shapes_hiddens]

    def _make_random_hiddens(self, x):
        """Create random-initialized hidden states (N(0,1)) with correct shapes."""
        shapes_hiddens = self.feedforward_init(x)
        return [torch.randn_like(h) for h in shapes_hiddens]

    # ------------------------------------------------------------------
    # Error-parameterized relaxation
    # ------------------------------------------------------------------

    def relax_errors(self, x, gamma, K_h, dt, async_mode=True,
                     init_mode="feedforward"):
        """
        Relax prediction errors to equilibrium via gradient descent on E.

        Tracks e_k = f_k(h_{k-1}) - h_k instead of h_k directly.
        Advantage: the Hessian ∂²E/∂e² ≈ I, enabling float32 + few CG steps.

        Uses async even/odd traversal (Scellier App A.1).

        Args:
            x: input images (B, C, H, W).
            gamma: linear clamp strength.
            T_free: number of relaxation iterations.
            dt: step size for gradient descent.
            async_mode: if True, even/odd async updates.
            init_mode: "feedforward" (e_k=0), "zeros" (h_k=0), or "random".

        Returns:
            errors: list of L equilibrium error tensors (detached).
            diagnostics: dict with "energy" and "residual_norms".
        """
        x_detached = x.detach()

        # ---- Initialize errors ----
        if init_mode == "feedforward":
            # Feedforward init: h_k = f_k(h_{k-1}) exactly → e_k = 0
            ff_hiddens = self.feedforward_init(x_detached)
            errors = [torch.zeros_like(h) for h in ff_hiddens]
        elif init_mode == "zeros":
            # All h_k = 0 → compute corresponding errors
            zero_hiddens = self._make_zero_hiddens(x_detached)
            errors = _hiddens_to_errors(zero_hiddens, self, x_detached)
            errors = [e.detach() for e in errors]
        elif init_mode == "random":
            # Random h_k → compute corresponding errors
            rand_hiddens = self._make_random_hiddens(x_detached)
            errors = _hiddens_to_errors(rand_hiddens, self, x_detached)
            errors = [e.detach() for e in errors]
        else:
            raise ValueError(f"Unknown pcn_init_mode: {init_mode}")

        diagnostics = {"energy": [], "residual_norms": []}

        # Record initial state
        with torch.no_grad():
            hiddens_init = _errors_to_hiddens(errors, self, x_detached)
            diagnostics["energy"].append(
                self.compute_energy(x_detached, hiddens_init, gamma).item()
            )
            diagnostics["residual_norms"].append(
                self.compute_residuals(x_detached, hiddens_init)
            )

        # ---- Relaxation loop (fully detached) ----
        for step in range(K_h):
            self._relax_sweep_e(x_detached, errors, gamma, dt, async_mode)

            # Record diagnostics
            with torch.no_grad():
                hiddens_diag = _errors_to_hiddens(errors, self, x_detached)
                diagnostics["energy"].append(
                    self.compute_energy(x_detached, hiddens_diag, gamma).item()
                )
                diagnostics["residual_norms"].append(
                    self.compute_residuals(x_detached, hiddens_diag)
                )

        # ---- Equilibrium quality diagnostics ----
        # (h-param gets similar info from residual_norms above; here we
        # compute directly from the error tensors which is cheaper)
        with torch.no_grad():
            diagnostics["max_error"] = max(e.abs().max().item() for e in errors)
            diagnostics["per_layer_error_norm"] = [
                e.norm().item() / (e.numel() ** 0.5) for e in errors
            ]

        # Equilibrium residual: max|∂E/∂e| (requires one grad computation)
        with torch.enable_grad():
            e_check = [e.detach().requires_grad_(True) for e in errors]
            E_check = _compute_energy_from_errors(e_check, self, x_detached, gamma)
            grads_check = torch.autograd.grad(E_check, e_check)
            diagnostics["max_eq_residual"] = max(
                g.abs().max().item() for g in grads_check
            )

        return errors, diagnostics

    def _update_error_gd(self, x, errors, k, gamma, dt):
        """
        Single gradient descent step for error k:
            e_k ← e_k - dt · ∂E/∂e_k

        The energy gradient in e-space: ∂E/∂e_k = e_k + γ · ∂o/∂e_k.
        The first term (e_k) dominates, making the Hessian ≈ I.

        Args:
            x: input (detached).
            errors: current error states (list of L tensors).
            k: layer index to update.
            gamma: linear clamp strength.
            dt: step size.

        Returns: updated e_k (detached).
        """
        with torch.enable_grad():
            # We want ∂E/∂e_k treating all OTHER e_j as constants.
            # detach() cuts e_k from any prior graph, requires_grad_(True)
            # makes it a fresh leaf that autograd can differentiate w.r.t.
            # We then splice it into a shallow copy of the list so that
            # _compute_energy_from_errors sees the grad-tracked e_k at position k.
            e_k = errors[k].detach().requires_grad_(True)
            errors_with_grad = list(errors)
            errors_with_grad[k] = e_k
            E = _compute_energy_from_errors(errors_with_grad, self, x, gamma)
            grad = torch.autograd.grad(E, e_k)[0]
        return (e_k - dt * grad).detach()

    def _update_nodes_error_gd(self, x, errors, ks, gamma, dt):
        """Batched twin of _update_error_gd: one energy evaluation gives
        ∂E/∂e_k for every k in ks (see _update_nodes_gd for the Jacobi-vs-
        sequential note). Mutates `errors` in place."""
        if not ks:
            return
        with torch.enable_grad():
            leaves = [errors[k].detach().requires_grad_(True) for k in ks]
            tmp = list(errors)
            for k, leaf in zip(ks, leaves):
                tmp[k] = leaf
            E = _compute_energy_from_errors(tmp, self, x, gamma)
            grads = torch.autograd.grad(E, leaves)
        for k, leaf, g in zip(ks, leaves, grads):
            errors[k] = (leaf - dt * g).detach()

    def _relax_sweep_e(self, x, errors, gamma, dt, async_mode):
        """One relaxation sweep over all error nodes: 2 energy evaluations
        in async even/odd mode, 1 in synchronous (full Jacobi)."""
        if async_mode:
            for parity in (0, 1):
                self._update_nodes_error_gd(
                    x, errors, [k for k in range(self.L) if k % 2 == parity],
                    gamma, dt)
        else:
            self._update_nodes_error_gd(x, errors, list(range(self.L)), gamma, dt)

    # ------------------------------------------------------------------
    # Spring-clamped EP: free phase, nudge phase, gradient step
    # ------------------------------------------------------------------
    #
    # In spring-clamped EP, x becomes a dynamic variable with a spring
    # pulling it back to the input x_t:
    #   E_spring = E_int(x, h, γ) + (λ/2)||x - x_t||²
    #
    # At equilibrium, the spring force balances the energy gradient:
    #   λ(x* - x_t) = -∂E_int/∂x
    # so the velocity can be read off from the displacement:
    #   v = output_scale · (1/γ) · λ · (x* - x_t)
    # This matches the IFT velocity: v = -output_scale · (1/γ) · ∂E_int/∂x
    #
    # For EP parameter gradients, a nudge phase adds a loss that pushes
    # x toward a target displacement:
    #   E_nudge = E_spring + β · L_nudge(x)
    # and the gradient is estimated as (E_β - E*) / β.


    def relax_spring_free(self, x_t, gamma, T_free, dt, lambda_spring,
                          K_h=1, async_mode=True, init_mode="feedforward"):
        """
        Free phase for spring-clamped EP: relax both x and h to equilibrium.

        Energy minimised: E_spring = E_int(x, h, γ) + (λ/2)||x - x_t||²

        x starts at x_t and is pulled back by the spring. Hidden states
        start from feedforward init (or zeros/random) and relax as usual.

        Args:
            x_t: input images (B, C, H, W) — the spring anchor.
            gamma: γ, linear clamp strength.
            T_free: number of relaxation iterations.
            dt: step size for gradient descent.
            lambda_spring: spring constant λ.
            async_mode: even/odd async updates for h.
            init_mode: "feedforward", "zeros", or "random" for h init.

        Returns:
            x_star: equilibrium x (B, C, H, W), detached.
            h_star: list of L equilibrium hidden states, detached.
            diagnostics: dict with energy trajectory etc.
        """
        x_detached = x_t.detach()
        x = x_detached.clone()  # x starts at x_t

        # Initialise hidden states
        if init_mode == "feedforward":
            hiddens = self.feedforward_init(x_detached)
        elif init_mode == "zeros":
            hiddens = self._make_zero_hiddens(x_detached)
        else:
            hiddens = self._make_random_hiddens(x_detached)
        hiddens = [h.detach() for h in hiddens]

        diagnostics = {"energy": [], "x_disp": []}

        for step in range(T_free):
            # Inner loop: equilibrate h at current x (τ_h << τ_x)
            for _ in range(K_h):
                self._relax_sweep_h(x, hiddens, gamma, dt, async_mode)

            # Outer step: update x via ∂E_spring/∂x|_{h fixed}.
            # This computes the PARTIAL derivative ∂E/∂x with h treated as
            # constants — the same adiabatic assumption as the error-param
            # version. In h-param this happens naturally because hiddens are
            # independent GD variables (detached tensors), so no explicit
            # detach-and-reconstruct step is needed (unlike error-param,
            # where h = f(x, e) creates a dependency requiring x to be
            # detached to avoid the total derivative dE/dx path).
            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                E = self.compute_energy(x_grad, hiddens, gamma)
                spring = (lambda_spring / 2.0) * (x_grad - x_detached).pow(2).sum()
                E_spring = E + spring
                grad_x = torch.autograd.grad(E_spring, x_grad)[0]
            x = (x_grad - dt * grad_x).detach()

            # Diagnostics
            with torch.no_grad():
                diagnostics["energy"].append(
                    self.compute_energy(x, hiddens, gamma).item()
                )
                diagnostics["x_disp"].append(
                    (x - x_detached).pow(2).mean().sqrt().item()
                )

        # ---- Equilibrium quality diagnostics (matching relax_hiddens) ----
        with torch.no_grad():
            diagnostics["residual_norms"] = self.compute_residuals(x, hiddens)


        # max|∂E/∂h|: validates the adiabatic assumption (∂E/∂h ≈ 0).
        # In h-param, hiddens are independent GD variables, so ∂E/∂h is
        # already the correct partial derivative (no detach subtlety).
        with torch.enable_grad():
            h_check = [h.detach().requires_grad_(True) for h in hiddens]
            E_h_check = self.compute_energy(x.detach(), h_check, gamma)
            h_grads = torch.autograd.grad(E_h_check, h_check)
            diagnostics["max_eq_residual_h"] = max(
                g.abs().max().item() for g in h_grads
            )

        return x, hiddens, diagnostics


    def relax_spring_nudged(self, x_t, x_star, h_star, u_target,
                            beta, T_nudge, gamma, dt, lambda_spring,
                            output_scale, alpha,
                            K_h=1, async_mode=True, nudge_type="quadratic"):
        """
        Nudge phase for spring-clamped EP: re-relax with a velocity nudge loss.

        Energy minimised:
            E_nudge = E_int(x, h, γ) + (λ/2)||x - x_t||²
                    + β · (output_scale·α·λ)² / 2 · ||x - target_x||²

        where target_x = x_t + u_target / (output_scale · α · λ),
        chosen so that at nudge equilibrium the velocity is pushed toward u_target.

        Starts from the free-phase equilibrium (x*, h*).

        Args:
            x_t: input images — spring anchor.
            x_star: free-phase equilibrium x (detached).
            h_star: free-phase equilibrium hiddens (list, detached).
            u_target: target velocity u_t from flow matching (B, C, H, W).
            beta: nudge strength β.
            T_nudge: number of nudge relaxation steps.
            gamma: γ, linear clamp strength.
            dt: step size.
            lambda_spring: spring constant λ.
            output_scale: velocity scale factor.
            alpha: 1/γ factor.
            async_mode: even/odd async updates.

        Returns:
            x_beta: nudge equilibrium x (detached).
            h_beta: nudge equilibrium hiddens (list, detached).
        """
        x_t_det = x_t.detach()
        vel_scale = output_scale * alpha * lambda_spring
        
        # Quadratic nudge target (unused when nudge_type='linear').
        target_x = (x_t_det + u_target.detach() / vel_scale).detach()
        nudge_coeff = beta * vel_scale ** 2 / 2.0
        
        # Linear tilt: frozen velocity mismatch g = v_free - v_hat; tilt coeff
        # beta*vel_scale matches the quadratic nudge's first-order displacement.
        g_frozen = (vel_scale * (x_star.detach() - x_t_det)
                    - u_target.detach()).detach()
        tilt_coeff = beta * vel_scale



        x = x_star.detach().clone()
        hiddens = [h.detach().clone() for h in h_star]

        for step in range(T_nudge):
            # Inner loop: equilibrate h at current x (τ_h << τ_x)
            for _ in range(K_h):
                self._relax_sweep_h(x, hiddens, gamma, dt, async_mode)

            # Outer step: update x with spring + nudge.
            # Same partial-derivative logic as relax_spring_free (see comment
            # there): hiddens are independent GD variables, so no detach-
            # and-reconstruct needed unlike the error-param version.
            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                E = self.compute_energy(x_grad, hiddens, gamma)
                spring = (lambda_spring / 2.0) * (x_grad - x_t_det).pow(2).sum()

                if nudge_type == "linear":
                    # Linear tilt: constant force tilt_coeff*g (zero curvature).
                    nudge = tilt_coeff * (g_frozen * x_grad).sum()
                else:
                    nudge = nudge_coeff * (x_grad - target_x).pow(2).sum()
                    
                E_nudged = E + spring + nudge
                grad_x = torch.autograd.grad(E_nudged, x_grad)[0]
            x = (x_grad - dt * grad_x).detach()

        return x, hiddens


    def relax_spring_free_errors(self, x_t, gamma, T_free, dt, lambda_spring,
                                 K_h=1, async_mode=True, init_mode="feedforward",
                                 x_grad_mode="adiabatic"):
        """
        Free phase for spring-clamped EP in error parameterization.

        Same as relax_spring_free but tracks errors e_k = f_k(h_{k-1}) - h_k
        instead of hidden states. Convergence is faster because H ≈ I.

        Returns:
            x_star: equilibrium x (detached).
            errors: list of L equilibrium error tensors (detached).
            diagnostics: dict.
        """
        x_detached = x_t.detach()
        x = x_detached.clone()

        # Initialise errors (must correspond to actual h values, not just shapes)
        if init_mode == "feedforward":
            # Feedforward init: h_k = f_k(h_{k-1}) exactly → e_k = 0
            errors = [torch.zeros_like(h) for h in self.feedforward_init(x_detached)]
        elif init_mode == "zeros":
            # All h_k = 0 → compute corresponding errors e_k = f_k(h_{k-1}) - 0
            zero_hiddens = self._make_zero_hiddens(x_detached)
            errors = _hiddens_to_errors(zero_hiddens, self, x_detached)
            errors = [e.detach() for e in errors]
        elif init_mode == "random":
            # Random h_k → compute corresponding errors e_k = f_k(h_{k-1}) - h_k
            rand_hiddens = self._make_random_hiddens(x_detached)
            errors = _hiddens_to_errors(rand_hiddens, self, x_detached)
            errors = [e.detach() for e in errors]
        else:
            raise ValueError(f"Unknown pcn_init_mode: {init_mode}")

        diagnostics = {"energy": [], "x_disp": []}

        for step in range(T_free):
            # Inner loop: equilibrate errors at current x (τ_h << τ_x)
            for _ in range(K_h):
                self._relax_sweep_e(x, errors, gamma, dt, async_mode)

            # Outer step: update x via gradient descent on E_spring.
            # x_grad_mode selects the x-gradient of the e-parameterized energy:
            #  - "adiabatic": reconstruct h from errors with x DETACHED, take
            #    ∂E/∂x|_{h fixed}. Matches the h-param update but is only exact
            #    when ∂E/∂h ≈ 0 (h fully equilibrated) — a real bias on deep
            #    DAGs where K_h inner sweeps cannot equilibrate all nodes.
            #  - "total": dE/dx|_{e fixed} through the reconstruction scan —
            #    the exact gradient of E(x, e) at frozen errors; requires NO
            #    h-equilibrium assumption.
            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                if x_grad_mode == "total":
                    E = _compute_energy_from_errors(
                        [e.detach() for e in errors], self, x_grad, gamma)
                else:
                    hiddens_fixed = _errors_to_hiddens(errors, self, x.detach())
                    hiddens_fixed = [h.detach() for h in hiddens_fixed]
                    E = self.compute_energy(x_grad, hiddens_fixed, gamma)
                spring = (lambda_spring / 2.0) * (x_grad - x_detached).pow(2).sum()
                E_spring = E + spring
                grad_x = torch.autograd.grad(E_spring, x_grad)[0]
            x = (x_grad - dt * grad_x).detach()

            # Diagnostics
            with torch.no_grad():
                hiddens_diag = _errors_to_hiddens(errors, self, x)
                diagnostics["energy"].append(
                    self.compute_energy(x, hiddens_diag, gamma).item()
                )
                diagnostics["x_disp"].append(
                    (x - x_detached).pow(2).mean().sqrt().item()
                )

        # ---- Equilibrium quality diagnostics (matching relax_errors) ----
        with torch.no_grad():
            diagnostics["max_error"] = max(e.abs().max().item() for e in errors)
            diagnostics["per_layer_error_norm"] = [
                e.norm().item() / (e.numel() ** 0.5) for e in errors
            ]

        # Equilibrium residual in e-space: max|∂E/∂e|
        with torch.enable_grad():
            e_check = [e.detach().requires_grad_(True) for e in errors]
            E_check = _compute_energy_from_errors(e_check, self, x.detach(), gamma)
            grads_check = torch.autograd.grad(E_check, e_check)
            diagnostics["max_eq_residual"] = max(
                g.abs().max().item() for g in grads_check
            )

        # Equilibrium residual in h-space: max|∂E/∂h|
        # This directly validates the adiabatic assumption (∂E/∂h ≈ 0)
        # used by the detached x-gradient. At exact convergence ∂E/∂e ≈ 0
        # implies ∂E/∂h ≈ 0, but Jacobian amplification could cause them
        # to differ at finite K_h.
        with torch.enable_grad():
            hiddens_eq = _errors_to_hiddens(errors, self, x.detach())
            h_check = [h.detach().requires_grad_(True) for h in hiddens_eq]
            E_h_check = self.compute_energy(x.detach(), h_check, gamma)
            h_grads = torch.autograd.grad(E_h_check, h_check)
            diagnostics["max_eq_residual_h"] = max(
                g.abs().max().item() for g in h_grads
            )

        return x, errors, diagnostics

    def relax_spring_nudged_errors(self, x_t, x_star, errors_star, u_target,
                                   beta, T_nudge, gamma, dt, lambda_spring,
                                   output_scale, alpha,
                                   K_h=1, async_mode=True, nudge_type="quadratic",
                                   x_grad_mode="adiabatic"):
        """
        Nudge phase for spring-clamped EP in error parameterization.

        Same as relax_spring_nudged but tracks errors instead of hiddens.

        Returns:
            x_beta: nudge equilibrium x (detached).
            errors_beta: nudge equilibrium errors (list, detached).
        """
        x_t_det = x_t.detach()
        vel_scale = output_scale * alpha * lambda_spring
        
        # Quadratic nudge target (unused when nudge_type='linear').
        target_x = (x_t_det + u_target.detach() / vel_scale).detach()
        nudge_coeff = beta * vel_scale ** 2 / 2.0
        
        # Linear tilt: frozen velocity mismatch g = v_free - v_hat; tilt coeff
        # beta*vel_scale matches the quadratic nudge's first-order displacement.
        g_frozen = (vel_scale * (x_star.detach() - x_t_det)
                    - u_target.detach()).detach()
        tilt_coeff = beta * vel_scale

        x = x_star.detach().clone()
        errors = [e.detach().clone() for e in errors_star]

        for step in range(T_nudge):
            # Inner loop: equilibrate errors at current x (τ_h << τ_x)
            for _ in range(K_h):
                self._relax_sweep_e(x, errors, gamma, dt, async_mode)

            # Outer step: update x with spring + nudge.
            # x_grad_mode: "adiabatic" ∂E/∂x|_h vs "total" dE/dx|_e — see the
            # free phase for the trade-off (total needs no h-equilibrium).
            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                if x_grad_mode == "total":
                    E = _compute_energy_from_errors(
                        [e.detach() for e in errors], self, x_grad, gamma)
                else:
                    hiddens_fixed = _errors_to_hiddens(errors, self, x.detach())
                    hiddens_fixed = [h.detach() for h in hiddens_fixed]
                    E = self.compute_energy(x_grad, hiddens_fixed, gamma)
                spring = (lambda_spring / 2.0) * (x_grad - x_t_det).pow(2).sum()
                
                if nudge_type == "linear":
                    # Linear tilt: constant force tilt_coeff*g (zero curvature).
                    nudge = tilt_coeff * (g_frozen * x_grad).sum()
                else:
                    nudge = nudge_coeff * (x_grad - target_x).pow(2).sum()
                    
                E_nudged = E + spring + nudge
                grad_x = torch.autograd.grad(E_nudged, x_grad)[0]
            x = (x_grad - dt * grad_x).detach()

        return x, errors

    def ep_gradient_step(self, x_first, h_first, x_second, h_second,
                         beta, gamma):
        """
        EP parameter gradient: ∂L/∂θ ≈ (E_second - E_first) / β.

        For two-phase EP: first=star, second=beta → (E_β - E*) / β.
        For three-phase: first=minus, second=plus → (E_+ - E_-) / (2β).

        All neuron states (x, h) are DETACHED — only θ is live.
        Calls .backward() to accumulate gradients into param.grad.

        Args:
            x_first, h_first: first-phase equilibrium (detached).
            x_second, h_second: second-phase equilibrium (detached).
            beta: effective β (use 2β for three-phase).
            gamma: γ for energy computation.

        Returns:
            ep_loss: scalar (E_second - E_first) / β for logging.
        """
        # Compute energies with θ live (no detach on model weights)
        E_first = self.compute_energy(x_first.detach(), 
                                       [h.detach() for h in h_first], gamma)
        E_second = self.compute_energy(x_second.detach(),
                                        [h.detach() for h in h_second], gamma)


        # EP loss: normalized to match flow_loss = (1/N)·Σ||v-u||² exactly.
        # Note our nudged phase EP pre-factors implementation estimates ∂C/∂θ 
        # where C = (1/2)·Σ||v-u||² (energy has 1/2 factor).
        # flow_loss has no 1/2, so ∂flow_loss/∂θ = (2/N)·∂C/∂θ.
        # The factor of 2 accounts for d/dx(x²) = 2x in flow_loss.
        # Also note 'ep_loss' is the scalar whose .backward() deriative gives the loss gradient
        # it is not a loss in itself.
        N = x_first.numel()  # B * C * H * W
        ep_loss = 2.0 * (E_second - E_first) / (beta * N)
        ep_loss.backward()
        return ep_loss.item()

    def ep_gradient_step_errors(self, x_first, errors_first, x_second, errors_second,
                                beta, gamma):
        """
        EP parameter gradient in error parameterization.

        Same as ep_gradient_step but reconstructs hiddens from errors
        before computing energy (with θ live).

        Returns:
            ep_loss: scalar for logging.
        """
        # Reconstruct hiddens from errors WITH live θ
        h_first = _errors_to_hiddens([e.detach() for e in errors_first],
                                      self, x_first.detach())
        h_second = _errors_to_hiddens([e.detach() for e in errors_second],
                                       self, x_second.detach())

        E_first = self.compute_energy(x_first.detach(), h_first, gamma)
        E_second = self.compute_energy(x_second.detach(), h_second, gamma)

        # EP loss: normalized to match flow_loss = (1/N)·Σ||v-u||² exactly.
        # Note our nudged phase EP pre-factors implementation estimates ∂C/∂θ 
        # where C = (1/2)·Σ||v-u||² (energy has 1/2 factor).
        # flow_loss has no 1/2, so ∂flow_loss/∂θ = (2/N)·∂C/∂θ.
        # The factor of 2 accounts for d/dx(x²) = 2x in flow_loss.
        # Also note 'ep_loss' is the scalar whose .backward() deriative gives the loss gradient
        # it is not a loss in itself.
        N = x_first.numel()  # B * C * H * W
        ep_loss = 2.0 * (E_second - E_first) / (beta * N)
        ep_loss.backward()
        return ep_loss.item()


class PCNVelocityWrapper(nn.Module):
    """
    PCN-based velocity model, exposing the same interface as EBCNNModelWrapper:
        forward(t, x), potential(x, t), velocity(x, t)

    Computes velocity via PCN energy relaxation + envelope theorem:
        1. Initialize h_k from feedforward pass (or zeros/random)
        2. Relax h to equilibrium via async gradient descent on E_int (assumes τ_x >> τ_h)
        3. At equilibrium, compute v = -output_scale · (1/γ) · ∂_x E_int

    The envelope theorem justifies detaching h* before computing ∂_x E:
    at equilibrium ∂E/∂h = 0, so dE_vis/dx = ∂E_int/∂x|_{h fixed}.

    In the small-γ limit with α = 1/γ:
        v → -output_scale · ∇_x F(x)
    matching the feedforward EBCNNModelWrapper exactly.

    Args:
        gamma: γ, linear clamp strength. α = 1/γ set automatically.
        T_free: number of relaxation iterations.
        dt_relax: step size for hidden state relaxation.
        async_mode: even/odd async updates (Scellier App A.1).
        init_mode: "feedforward", "zeros", or "random".
        output_scale: velocity magnitude multiplier (same as feedforward model).
        energy_clamp: optional tanh-based energy clamp (same as feedforward).
        n_cg_steps: CG iterations for IFT backward linear solve.
        pool_type: VGG5 pooling type.
        activation: activation function for layers.
        param_grad_mode: "ift" (Stage 2) or "ep" (Stage 3).
        lambda_spring: spring constant for EP x-clamping.
        beta: EP nudge strength.
        T_nudge: nudge phase relaxation steps.
        thirdphase: use three-phase EP for O(β²) accuracy.
    """

    def __init__(self, gamma=0.01, T_free=10, dt_relax=0.5, async_mode=True,
                 init_mode="feedforward", output_scale=100.0, energy_clamp=None,
                 n_cg_steps=10, pool_type="avgpool", activation="silu",
                 error_param=False, param_grad_mode="ift",
                 lambda_spring=10.0, beta=0.1, T_nudge=4, thirdphase=True,
                 K_h=2, nudge_type="quadratic",
                 pcn_arch="cnn", unet_kwargs=None,
                 ep_x_grad_mode="adiabatic"):
        super().__init__()
        # Lazy imports: the architecture modules import PCNEnergyModelBase from
        # THIS module, so importing them at top level would be circular.
        if pcn_arch in ("unet", "unet_vit"):
            # "unet" kept as alias for the ViT version (in-flight runs/resume
            # legs use it).
            from network_pcn_unet import PCNUNetEnergyModel
            self.pcn = PCNUNetEnergyModel(**(unet_kwargs or {}))
        elif pcn_arch == "cnn":
            from network_pcn_cnn import PCNCNNEnergyModel
            self.pcn = PCNCNNEnergyModel(pool_type=pool_type, activation=activation)
        else:
            raise ValueError(f"Unknown pcn_arch: {pcn_arch}")
        self.pcn_arch = pcn_arch
        self.gamma = gamma
        self.alpha = 1.0 / gamma  # v = -output_scale · α · ∂E/∂x; total scale = output_scale/γ
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp
        self.T_free = T_free
        self.dt_relax = dt_relax
        self.async_mode = async_mode
        self.init_mode = init_mode
        self.n_cg_steps = n_cg_steps
        self.error_param = error_param

        # EP-specific parameters (Stage 3)
        self.param_grad_mode = param_grad_mode
        self.lambda_spring = lambda_spring
        self.beta = beta
        self.T_nudge = T_nudge
        self.thirdphase = thirdphase
        self.K_h = K_h  # inner h-equilibration steps per x-step (τ_h << τ_x)
        self.nudge_type = nudge_type
        # x-gradient mode for the e-param spring phases: "adiabatic" ∂E/∂x|_h
        # (assumes h equilibrated) or "total" dE/dx|_e (exact at frozen errors;
        # no h-equilibrium assumption — relevant for deep DAGs).
        self.ep_x_grad_mode = ep_x_grad_mode

        # Store latest diagnostics for external access (e.g. W&B logging)
        self._last_diagnostics = None

    def potential(self, x, t):
        """
        Per-sample potential V(x) => (B,), matching the FFN wrappers' contract.

        V(x) = output_scale · α · E_int(x, h*)  (per sample). At equilibrium
        E_int ≈ γ·F(x), so V ≈ output_scale · F(x).

        Grad behaviour: the hidden relaxation is always detached (relax_hiddens
        returns a detached h*), but the energy is evaluated in the AMBIENT grad
        context — so this is differentiable in θ and x when the caller has grad
        enabled (CD energy term; the Langevin negative sampler's ∇_x V), and a
        plain value under `torch.no_grad()` (logging/generation). By the
        envelope theorem (∂E/∂h*=0 at equilibrium) the θ- and x-gradients of the
        detached-h* energy equal the true dV/dθ and dV/dx, so both the CD
        gradient and the sampler force are correct.
        """
        hiddens, _ = self.pcn.relax_hiddens(
            x, self.gamma, self.K_h, self.dt_relax,
            self.async_mode, self.init_mode
        )
        E = self.pcn.compute_energy_per_sample(x, hiddens, self.gamma)  # (B,)
        V = self.output_scale * self.alpha * E
        if self.energy_clamp is not None and self.energy_clamp > 0:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t, velocity_mode=None):
        """
        Compute velocity v(x).

        Dispatches based on param_grad_mode:
          - "ift": v = -output_scale·α·∂E/∂x with IFT backward (Stage 2)
          - "ep":  v = output_scale·α·λ·(x*-x_t) from spring displacement (Stage 3)
                   Caches free-phase equilibrium for later compute_ep_gradients() call.

        velocity_mode (optional, for inference/generation):
          - None (default): spring during EP training, internal energy otherwise.
          - "spring": force spring-clamped velocity (useful to test spring dynamics
            at inference time on an EP-trained model).
          - "internal": force internal-energy velocity v = -c·∂E_int/∂x (useful to
            test whether the raw energy landscape produces good velocities without
            the spring).

        """
        training = torch.is_grad_enabled()

        # Explicit mode override (for generation comparisons)
        if velocity_mode == "spring":
            return self._velocity_ep_spring(x)
        elif velocity_mode == "internal":
            if self.error_param:
                return self._velocity_error_param(x, training=False)
            else:
                return self._velocity_h_param(x, training=False)

        # Default dispatch: spring during EP training, internal energy otherwise
        if training and self.param_grad_mode == "ep":
            return self._velocity_ep_spring(x)

        if self.error_param:
            return self._velocity_error_param(x, training)
        else:
            return self._velocity_h_param(x, training)

    def _velocity_ep_spring(self, x):
        """
        EP spring-clamped velocity: v = output_scale · α · λ · (x* - x_t).

        Runs the free phase to get x*, then reads velocity from spring
        displacement. No autograd, no create_graph. Caches the free-phase
        equilibrium state on self for use by compute_ep_gradients().

        Returns: velocity (B, C, H, W), detached.
        """
        gamma = self.gamma
        T_free = self.T_free
        dt = self.dt_relax
        lam = self.lambda_spring

        if self.error_param:
            x_star, eq_state, diag = self.pcn.relax_spring_free_errors(
                x, gamma, T_free, dt, lam,
                K_h=self.K_h, async_mode=self.async_mode,
                init_mode=self.init_mode,
                x_grad_mode=self.ep_x_grad_mode
            )
        else:
            x_star, eq_state, diag = self.pcn.relax_spring_free(
                x, gamma, T_free, dt, lam,
                K_h=self.K_h, async_mode=self.async_mode,
                init_mode=self.init_mode
            )

        self._last_diagnostics = diag
        # Cache for compute_ep_gradients()
        self._ep_cache = {
            "x_t": x.detach(),
            "x_star": x_star,
            "eq_state": eq_state,  # hiddens or errors depending on error_param
        }

        v = (self.output_scale * self.alpha * lam
             * (x_star - x.detach())).detach()
        return v


    def compute_ep_gradients(self, u_target):
        """
        EP parameter gradient step: nudge phase(s) + energy difference → .backward().

        Must be called AFTER velocity() in EP mode (which caches the free-phase
        equilibrium). This is the EP equivalent of loss.backward() — it
        accumulates gradients into param.grad via (E_β - E*) / β.

        Args:
            u_target: target velocity from flow matching (B, C, H, W).

        Returns:
            ep_diagnostics: dict with ep_loss, nudge_disp, free_x_disp.
        """
        cache = self._ep_cache
        x_t = cache["x_t"]
        x_star = cache["x_star"]
        eq_state = cache["eq_state"]

        gamma = self.gamma
        dt = self.dt_relax
        lam = self.lambda_spring
        T_nudge = self.T_nudge

        # Compensate β so nudge strength is independent of λ and output_scale.
        # The nudge formula uses nudge_coeff = β·vel_scale²/2. By dividing β by
        # vel_scale², the effective nudge_coeff becomes β_user/2 — the user tunes
        # a single β that works regardless of λ or output_scale.
        vel_scale = self.output_scale * self.alpha * lam
        beta = self.beta / (vel_scale ** 2)

        if self.error_param:
            # Positive nudge
            x_plus, eq_plus = self.pcn.relax_spring_nudged_errors(
                x_t, x_star, eq_state, u_target,
                beta, T_nudge, gamma, dt, lam,
                self.output_scale, self.alpha,
                K_h=self.K_h, async_mode=self.async_mode,
                nudge_type=self.nudge_type,
                x_grad_mode=self.ep_x_grad_mode
            )
            if self.thirdphase:
                # Negative nudge (same free eq, β → -β)
                x_minus, eq_minus = self.pcn.relax_spring_nudged_errors(
                    x_t, x_star, eq_state, u_target,
                    -beta, T_nudge, gamma, dt, lam,
                    self.output_scale, self.alpha,
                    K_h=self.K_h, async_mode=self.async_mode,
                    nudge_type=self.nudge_type,
                    x_grad_mode=self.ep_x_grad_mode
                )
                ep_loss = self.pcn.ep_gradient_step_errors(
                    x_minus, eq_minus, x_plus, eq_plus,
                    2 * beta, gamma
                )
            else:
                ep_loss = self.pcn.ep_gradient_step_errors(
                    x_star, eq_state, x_plus, eq_plus,
                    beta, gamma
                )
        else:
            x_plus, eq_plus = self.pcn.relax_spring_nudged(
                x_t, x_star, eq_state, u_target,
                beta, T_nudge, gamma, dt, lam,
                self.output_scale, self.alpha,
                K_h=self.K_h, async_mode=self.async_mode,
                nudge_type=self.nudge_type
            )
            if self.thirdphase:
                x_minus, eq_minus = self.pcn.relax_spring_nudged(
                    x_t, x_star, eq_state, u_target,
                    -beta, T_nudge, gamma, dt, lam,
                    self.output_scale, self.alpha,
                    K_h=self.K_h, async_mode=self.async_mode,
                    nudge_type=self.nudge_type
                )
                ep_loss = self.pcn.ep_gradient_step(
                    x_minus, eq_minus, x_plus, eq_plus,
                    2 * beta, gamma
                )
            else:
                ep_loss = self.pcn.ep_gradient_step(
                    x_star, eq_state, x_plus, eq_plus,
                    beta, gamma
                )

        nudge_disp = (x_plus - x_star).pow(2).mean().sqrt().item()
        diag = self._last_diagnostics or {}

        # Free-phase velocity mismatch g = v_free - v_hat (drives the nudge). Its
        # RMS is the per-element velocity error; its max sets the binding case
        # for the linear-tilt constraint beta*|g| << gamma (largest at high t).
        with torch.no_grad():
            g_mismatch = vel_scale * (x_star - x_t) - u_target
            vel_mismatch = g_mismatch.pow(2).mean().sqrt().item()

        result = {
            "ep_loss": ep_loss,
            "nudge_disp": nudge_disp,
            "free_x_disp": diag.get("x_disp", [0.0])[-1],
            "vel_mismatch": vel_mismatch,
        }
        if self.nudge_type == "linear":
            # Linear-tilt finite-difference constraint (appendix "β|g| ≪ γ"): this
            # is the worst-case probe displacement |Δx|_max, which IS the appendix's
            # β|g|/γ in the code's normalization — Δx = (β/vel_scale)·g/λ — and must
            # stay << the O(1) length scale over which ∇E varies. 
            with torch.no_grad():
                g_max = g_mismatch.abs().max().item()
            result["linear_nudge_constraint"] = (self.beta / vel_scale) * g_max / self.lambda_spring
        if self.thirdphase:
            result["nudge_disp_neg"] = (x_minus - x_star).pow(2).mean().sqrt().item()
        return result

    def _velocity_h_param(self, x, training):
        """Original h-parameterized velocity computation."""
        # 1. Relax to equilibrium (fully detached)
        hiddens, diag = self.pcn.relax_hiddens(
            x, self.gamma, self.K_h, self.dt_relax,
            self.async_mode, self.init_mode
        )
        self._last_diagnostics = diag

        # 2. Attach IFT backward for training
        if training:
            h_flat = _flatten_hiddens(hiddens).requires_grad_(True)
            shapes = [h.shape for h in hiddens]
            h_flat_ift = IFTEquilibriumHiddens.apply(
                h_flat, self.pcn, x, self.gamma, self.n_cg_steps, shapes
            )
            hiddens = _unflatten_hiddens(h_flat_ift, shapes)

        # 3. Compute ∂E/∂x at equilibrium
        with torch.enable_grad():
            x_grad = x.clone().detach().requires_grad_(True)
            E = self.pcn.compute_energy(x_grad, hiddens, self.gamma)
            dEdx = torch.autograd.grad(
                outputs=E,
                inputs=x_grad,
                grad_outputs=torch.ones_like(E),
                create_graph=training,
            )[0]

        # 4. Velocity: v = -output_scale · α · ∂E/∂x
        v = -self.output_scale * self.alpha * dEdx
        return v

    def _velocity_error_param(self, x, training):
        """
        Error-parameterized velocity computation.

        Same physics as _velocity_h_param but tracks prediction errors
        e_k = f_k(h_{k-1}) - h_k instead of h_k directly.

        Advantage: Hessian ∂²E/∂e² ≈ I, so CG converges in 2-5 steps
        and float32 is sufficient (no catastrophic cancellation).
        """
        # 1. Relax errors to equilibrium (fully detached)
        errors, diag = self.pcn.relax_errors(
            x, self.gamma, self.K_h, self.dt_relax,
            self.async_mode, self.init_mode
        )
        self._last_diagnostics = diag

        # 2. Attach IFT backward for training (in e-space)
        if training:
            e_flat = _flatten_hiddens(errors).requires_grad_(True)
            shapes = [e.shape for e in errors]
            e_flat_ift = IFTEquilibriumErrors.apply(
                e_flat, self.pcn, x, self.gamma, self.n_cg_steps, shapes
            )
            errors = _unflatten_hiddens(e_flat_ift, shapes)

        # 3. Reconstruct hiddens from errors and compute ∂E/∂x
        with torch.enable_grad():
            x_grad = x.clone().detach().requires_grad_(True)
            hiddens = _errors_to_hiddens(errors, self.pcn, x_grad)
            E = self.pcn.compute_energy(x_grad, hiddens, self.gamma)
            dEdx = torch.autograd.grad(
                outputs=E,
                inputs=x_grad,
                grad_outputs=torch.ones_like(E),
                create_graph=training,
            )[0]

        # 4. Velocity: v = -output_scale · α · ∂E/∂x
        v = -self.output_scale * self.alpha * dEdx
        return v

    def forward(self, t, x, return_potential=False, velocity_mode=None, *args, **kwargs):
        """Same signature as EBCNNModelWrapper.forward()."""
        if return_potential:
            return self.potential(x, t)
        else:
            return self.velocity(x, t, velocity_mode=velocity_mode)


    def feedforward_velocity(self, x):
        """
        Compute velocity via pure feedforward + backprop (no relaxation).

        Used as a reference for cosine similarity diagnostics. This is what
        the feedforward EBCNNModelWrapper would compute: v = -output_scale · ∇_x F(x).

        Only valid for diagnostic comparison — uses the same weights but
        bypasses PCN dynamics entirely.
        """
        with torch.enable_grad():
            x_grad = x.clone().detach().requires_grad_(True)
            # Pure forward pass through the graph (topology-aware, keeps grad)
            h_list = []
            for k in range(self.pcn.L):
                h_list.append(self.pcn.predict(k, x_grad, h_list))
            # Final node is the raw scalar output F(x), shape (B, 1)
            F_x = h_list[-1].sum()  # sum over batch for scalar grad target (its only 1 element anyway so this is a trick)
            dFdx = torch.autograd.grad(
                outputs=F_x,
                inputs=x_grad,
                create_graph=False,
            )[0]
        return -self.output_scale * dFdx
