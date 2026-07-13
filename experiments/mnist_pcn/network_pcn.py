# File: network_pcn.py
# PCN (Predictive Coding Network) energy dynamics for MNIST (1×28×28).
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

    Given errors e_k = f_k(h_{k-1}) - h_k, reconstruct h_k = f_k(h_{k-1}) - e_k.
    This is a sequential scan from k=1 to L, since each h_k depends on h_{k-1}.

    Args:
        errors: list of L error tensors [e_1, ..., e_L].
        pcn: PCNEnergyModel (provides layers and L).
        x: input images (B, C, H, W).

    Returns: list of L hidden state tensors [h_1, ..., h_L].
    """
    hiddens = []
    h_prev = x
    for k, (layer, e_k) in enumerate(zip(pcn.layers, errors)):
        # Last layer is FC: flatten (B, C, H, W) → (B, C*H*W) before linear
        if k == pcn.L - 1:
            h_prev = h_prev.view(h_prev.size(0), -1)

        pred = layer(h_prev)
        h_k = pred - e_k
        hiddens.append(h_k)
        h_prev = h_k
    return hiddens


def _hiddens_to_errors(hiddens, pcn, x):
    """
    Compute prediction errors from hidden states.

    e_k = f_k(h_{k-1}) - h_k for each layer k.

    Args:
        hiddens: list of L hidden state tensors.
        pcn: PCNEnergyModel.
        x: input images.

    Returns: list of L error tensors [e_1, ..., e_L].
    """
    errors = []
    for k in range(pcn.L):
        h_prev = pcn._get_h_prev(x, hiddens, k) # i.e. actual dynamical h_prev values, not calculated from just feedforwards
        pred = pcn.layers[k](h_prev)
        errors.append(pred - hiddens[k])
    return errors


def _compute_energy_from_errors(errors, pcn, x, gamma):
    """
    Compute PCN energy from errors: E = Σ_k ½||e_k||² + γ·o(e).

    The output o depends on errors through the forward scan
    h_k = f_k(h_{k-1}) - e_k, so o = h_L.

    Args:
        errors: list of L error tensors (with grad tracking if needed).
        pcn: PCNEnergyModel.
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
            pcn: PCNEnergyModel (not saved — we access it via ctx)
            x: input images (B, 1, 28, 28)
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
# PCN Layer: one prediction function f_k
##############################################################################

class PCNLayer(nn.Module):
    """
    One PCN prediction layer: f_k(h_{k-1}) → prediction of h_k.

    Each layer applies conv (or linear) + activation + optional pooling,
    matching the standard VGG ordering (conv → act → pool).

    In the PCN energy, the residual r_k = f_k(h_{k-1}) - h_k measures the
    prediction error at layer k.
    """
    def __init__(self, transform, activation_fn=None, pool=None):
        """
        Args:
            transform: nn.Conv2d or nn.Linear — the learnable transform.
            activation_fn: nn.Module activation (e.g. nn.SiLU()). None for output layer.
            pool: nn.Module pooling (e.g. nn.AvgPool2d(2,2)). None if no pooling.
        """
        super().__init__()
        self.transform = transform
        self.activation_fn = activation_fn
        self.pool = pool

    def forward(self, h_prev):
        """Compute f_k(h_{k-1}): transform → activation → pool."""
        out = self.transform(h_prev)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


##############################################################################
# PCN Energy Model: VGG5 decomposed into L=6 layers
##############################################################################

class PCNEnergyModel(nn.Module):
    """
    VGG5 decomposed into L=6 PCN prediction layers for energy-based dynamics.

    Energy function:
        E_int(x, h, o) = Σ_{k=1}^{L} ½||f_k(h_{k-1}) - h_k||² + γ·o

    where h_0 = x (visible input), h_L = o (scalar output), and f_k is the
    k-th prediction layer (conv + act + optional pool).

    Layer decomposition (matching network_cnn.py VGG5Energy ordering):
        f_1: Conv(1, 128, 3, pad=1) + SiLU              → h_1: (B, 128, 28, 28)
        f_2: Conv(128, 256, 3, pad=1) + SiLU + Pool(2)  → h_2: (B, 256, 14, 14)
        f_3: Conv(256, 512, 3, pad=1) + SiLU + Pool(2)  → h_3: (B, 512, 7, 7)
        f_4: Conv(512, 512, 3, pad=1) + SiLU + Pool(2)  → h_4: (B, 512, 3, 3)
        f_5: Conv(512, 512, 3, pad=1) + SiLU + Pool(2)  → h_5: (B, 512, 1, 1)
        f_6: Linear(512, 1)  [no activation]             → o:   (B, 1)
    """

    def __init__(self, pool_type="avgpool", activation="silu"):
        super().__init__()

        act_cls = nn.SiLU if activation == "silu" else nn.ReLU

        def make_pool(channels):
            if pool_type == "maxpool":
                return nn.MaxPool2d(2, 2)
            elif pool_type == "avgpool":
                return nn.AvgPool2d(2, 2)
            elif pool_type == "stride_conv":
                return nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
            else:
                raise ValueError(f"Unknown pool_type: {pool_type}")

        self.layers = nn.ModuleList([
            # Layer 1: (B,1,28,28) → (B,128,28,28)
            PCNLayer(
                nn.Conv2d(1, 128, kernel_size=3, padding=1),
                activation_fn=act_cls(),
            ),
            # Layer 2: (B,128,28,28) → (B,256,14,14)
            PCNLayer(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                activation_fn=act_cls(),
                pool=make_pool(256),
            ),
            # Layer 3: (B,256,14,14) → (B,512,7,7)
            PCNLayer(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                activation_fn=act_cls(),
                pool=make_pool(512),
            ),
            # Layer 4: (B,512,7,7) → (B,512,3,3)
            PCNLayer(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                activation_fn=act_cls(),
                pool=make_pool(512),
            ),
            # Layer 5: (B,512,3,3) → (B,512,1,1)
            PCNLayer(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                activation_fn=act_cls(),
                pool=make_pool(512),
            ),
            # Layer 6 (output): (B,512,1,1) → (B,1) scalar
            # Flatten is handled inside forward; Linear has no activation.
            PCNLayer(
                nn.Linear(512, 1),
                activation_fn=None,
            ),
        ])
        self.L = len(self.layers)
        self._pool_type = pool_type
        self._activation = activation

    def feedforward_init(self, x):
        """
        Pure forward pass through all layers → list of L hidden states.

        At these values, all residuals r_k = f_k(h_{k-1}) - h_k = 0 exactly.
        This is the free-phase equilibrium (Scellier Eq. 9).

        Returns: list of L tensors [h_1, h_2, ..., h_L], each detached.
        """
        hiddens = []
        h = x
        for k, layer in enumerate(self.layers):
            if k == self.L - 1:
                # Output layer: flatten before linear
                h = h.view(h.size(0), -1)
            h = layer(h)
            hiddens.append(h.detach().clone())
        return hiddens

    def _get_h_prev(self, x, hiddens, k):
        """Get h_{k-1} for layer k. k=0 uses x; k>0 uses hiddens[k-1]."""
        if k == 0:
            return x
        h_prev = hiddens[k - 1]
        if k == self.L - 1:
            # Output layer expects flattened input
            h_prev = h_prev.view(h_prev.size(0), -1)
        return h_prev

    def compute_energy(self, x, hiddens, gamma):
        """
        Compute PCN energy: E_int = Σ_k ½||r_k||² + γ·o

        The sum-of-squared-residuals measures prediction error across all layers.
        The linear clamp γ·o injects a constant "current" at the output node,
        whose backward propagation through residuals implements backprop of F(x).

        Args:
            x: input images (B, 1, 28, 28), should have requires_grad if
               computing ∂E/∂x for velocity.
            hiddens: list of L hidden state tensors.
            gamma: linear clamp strength.

        Returns: scalar energy (summed over batch).
        """
        energy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for k in range(self.L):
            h_prev = self._get_h_prev(x, hiddens, k)
            pred = self.layers[k](h_prev)
            r_k = pred - hiddens[k]
            energy = energy + 0.5 * (r_k ** 2).sum()
        # Linear clamp: γ · o (sum over batch)
        output = hiddens[-1]  # (B, 1)
        energy = energy + gamma * output.sum()
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
            h_prev = self._get_h_prev(x, hiddens, k)
            with torch.no_grad():
                pred = self.layers[k](h_prev)
                r_k = pred - hiddens[k]
                # Per-sample L2 norm, then batch average
                norm = r_k.view(r_k.size(0), -1).norm(dim=1).mean().item()
                norms.append(norm)
        return norms

    def relax_hiddens(self, x, gamma, K, dt, async_mode=True,
                      init_mode="feedforward"):
        """
        Relax hidden states to equilibrium via gradient descent on E_int.

        Uses async even/odd traversal (Scellier App A.1): each iteration
        updates even-indexed layers first, then odd-indexed layers.

        All hidden states are FULLY DETACHED — no computation graph through
        the relaxation. Parameter gradients are handled separately by the
        IFT backward (IFTEquilibriumHiddens) in the velocity computation.

        Args:
            x: input images (B, 1, 28, 28).
            gamma: linear clamp strength.
            K: number of relaxation iterations.
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
        for step in range(K):
            if async_mode:
                for parity in [0, 1]:
                    for k in range(self.L):
                        if k % 2 != parity:
                            continue
                        hiddens[k] = self._update_layer_gd(
                            x_detached, hiddens, k, gamma, dt
                        )
            else:
                new_hiddens = [
                    self._update_layer_gd(x_detached, hiddens, k, gamma, dt)
                    for k in range(self.L)
                ]
                hiddens = new_hiddens

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

    def relax_errors(self, x, gamma, K, dt, async_mode=True,
                     init_mode="feedforward"):
        """
        Relax prediction errors to equilibrium via gradient descent on E.

        Tracks e_k = f_k(h_{k-1}) - h_k instead of h_k directly.
        Advantage: the Hessian ∂²E/∂e² ≈ I, enabling float32 + few CG steps.

        Uses async even/odd traversal (Scellier App A.1).

        Args:
            x: input images (B, 1, 28, 28).
            gamma: linear clamp strength.
            K: number of relaxation iterations.
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
        for step in range(K):
            if async_mode:
                for parity in [0, 1]:
                    for k in range(self.L):
                        if k % 2 != parity:
                            continue
                        errors[k] = self._update_error_gd(
                            x_detached, errors, k, gamma, dt
                        )
            else:
                new_errors = [
                    self._update_error_gd(x_detached, errors, k, gamma, dt)
                    for k in range(self.L)
                ]
                errors = new_errors

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


##############################################################################
# Velocity Wrapper: drop-in replacement for EBCNNModelWrapper
##############################################################################

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
        K: number of relaxation iterations.
        dt_relax: step size for hidden state relaxation.
        async_mode: even/odd async updates (Scellier App A.1).
        init_mode: "feedforward", "zeros", or "random".
        output_scale: velocity magnitude multiplier (same as feedforward model).
        energy_clamp: optional tanh-based energy clamp (same as feedforward).
        n_cg_steps: CG iterations for IFT backward linear solve.
        pool_type: VGG5 pooling type.
        activation: activation function for layers.
    """

    def __init__(self, gamma=0.01, K=10, dt_relax=0.5, async_mode=True,
                 init_mode="feedforward", output_scale=100.0, energy_clamp=None,
                 n_cg_steps=10, pool_type="avgpool", activation="silu",
                 error_param=False):
        super().__init__()
        self.pcn = PCNEnergyModel(pool_type=pool_type, activation=activation)
        self.gamma = gamma
        self.alpha = 1.0 / gamma  # v = -output_scale · α · ∂E/∂x; total scale = output_scale/γ
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp
        self.K = K
        self.dt_relax = dt_relax
        self.async_mode = async_mode
        self.init_mode = init_mode
        self.n_cg_steps = n_cg_steps
        self.error_param = error_param

        # Store latest diagnostics for external access (e.g. W&B logging)
        self._last_diagnostics = None

    def potential(self, x, t):
        """
        Compute scalar potential V(x) ≈ output_scale · F(x) at small γ.

        Specifically: V(x) = output_scale · α · E_int(x, h*, o*)
                            = output_scale · (1/γ) · E_int
        At equilibrium, E_int ≈ γ·F(x), so V ≈ output_scale · F(x).

        Shape: (B,).
        """
        hiddens, _ = self.pcn.relax_hiddens(
            x, self.gamma, self.K, self.dt_relax,
            self.async_mode, self.init_mode
        )
        with torch.no_grad():
            E = self.pcn.compute_energy(x, hiddens, self.gamma)
            V = self.output_scale * self.alpha * E
            if self.energy_clamp is not None and self.energy_clamp > 0:
                V = soft_clamp(V, self.energy_clamp)
            # Normalize by batch size (energy was summed over batch)
            return V / x.size(0)

    def velocity(self, x, t):
        """
        Compute velocity v(x) = -output_scale · (1/γ) · ∂_x E_int(x, h*, o*).

        Uses implicit function theorem (IFT) for parameter gradients:
            1. Relax h to equilibrium (fully detached — fast)
            2. Attach IFT backward to h* via IFTEquilibriumHiddens
            3. Compute ∂E/∂x at equilibrium with create_graph=True
            4. loss.backward() flows through ∂E/∂x → h* → IFT → θ

        The IFT backward solves (∂²E/∂h²)·w = upstream_grad via CG,
        giving the exact ∂h*/∂θ = -(∂²E/∂h²)⁻¹·∂²E/(∂h∂θ).
        This matches feedforward ∂v/∂θ by the envelope theorem proof.

        During inference: everything detached, no IFT overhead.

        Shape: (B, 1, 28, 28) — same as input.
        """
        training = torch.is_grad_enabled()

        if self.error_param:
            return self._velocity_error_param(x, training)
        else:
            return self._velocity_h_param(x, training)

    def _velocity_h_param(self, x, training):
        """Original h-parameterized velocity computation."""
        # 1. Relax to equilibrium (fully detached)
        hiddens, diag = self.pcn.relax_hiddens(
            x, self.gamma, self.K, self.dt_relax,
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
            x, self.gamma, self.K, self.dt_relax,
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

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        """Same signature as EBCNNModelWrapper.forward()."""
        if return_potential:
            return self.potential(x, t)
        else:
            return self.velocity(x, t)

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
            # Pure forward pass through all layers
            h = x_grad
            for k, layer in enumerate(self.pcn.layers):
                if k == self.pcn.L - 1:
                    h = h.view(h.size(0), -1)
                h = layer(h)
            # h is now the raw scalar output F(x), shape (B, 1)
            F_x = h.sum()  # sum over batch for scalar grad target (its only 1 element anyway so this is a trick)
            dFdx = torch.autograd.grad(
                outputs=F_x,
                inputs=x_grad,
                create_graph=False,
            )[0]
        return -self.output_scale * dFdx
