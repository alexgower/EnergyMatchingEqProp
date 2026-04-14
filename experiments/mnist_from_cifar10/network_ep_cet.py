# File: network_ep_cet.py
# Convergent Energy Transformer (CET) for EP-based flow matching.
#
# Based on Høier et al. (2026) "Training a Convergent Energy Transformer
# with Equilibrium Propagation", adapted for unconditional generation
# with spring-clamped GradEP (no decoder, no inpainting mask).
#
# Architecture:
#   x (image, B×C×H×W) ↔ z (tokens, B×N_P×D_T)
#   Energy: E(x,z) = ½||z||² - Φ_enc(x,z) - Φ_pos(z) - Φ_mem(z) - Φ_att(z)
#
# Coupling terms:
#   Φ_enc = Σ F(x, W_enc)·z + z·b_enc        (encoder: patches → tokens)
#   Φ_pos = z · b_pos                          (positional bias)
#   Φ_mem = Σ_j Σ_k [ReLU(w_k · z_j)]²       (Hopfield memory bank)
#   Φ_att = (1/γ) Σ_lm log Σ_n exp(γ A_lmn)   (Modern Hopfield attention)
#
# Dynamics: gradient descent on E with optional token normalisation.
#   z ← z - ε·∇_z E  (optionally followed by projection to zero-mean, unit-std)
#   x ← x - ε·∇_x E_spring  (spring-clamped to anchor x_t)
#
# This is equivalent to the primitive formulation:
#   Φ = (1-ε)½||z||² + ε·Φ_coupling  ⟹  z_new = ∇_z Φ = z - ε·∇_z E
# but uses explicit GD steps for clarity with the non-bilinear CET energy.
#
# Velocity: v = -α ∇_x E  (read from spring displacement during training).
#
# Interface matches EBEPMLPModelWrapper / EBEPModelWrapper for drop-in use
# with train_cifar_multigpu.py.  Does NOT define self.archi so the training
# loop treats x as (B, C, H, W) images (like we did for CNN but not MLP before), not flattened vectors.

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class EBEPCETModelWrapper(nn.Module):
    """
    Convergent Energy Transformer for EP-based flow matching.

    x (visible): image tensor (B, C, H, W)
    z (hidden):  token tensor (B, N_P, D_T)

    Energy: E = ½||z||² - Φ_coupling(x, z)
    Dynamics: z ← z - ε·∇_z E,  optionally projected to zero-mean unit-std.

    Default MNIST config (~107K params):
        patch_size=7, token_dim=128, n_heads=4, head_dim=32, n_memories=512

    Larger config (~410K params):
        token_dim=256, n_heads=8, head_dim=32, n_memories=1024
    """

    def __init__(
        self,
        img_channels=1,
        img_size=28,
        patch_size=7,
        stride=None,  # defaults to patch_size (non-overlapping)
        token_dim=128,
        n_heads=4,
        head_dim=32,
        n_memories=512,
        inv_temp=0.25,
        T=300,
        epsilon_ep=0.1,
        output_scale=2.0,
        energy_clamp=None,
        init_gain=0.5,
        spectral_norm_enabled=False,
        spectral_scale=1.0,
        lambda_spring=15.0,
        normalize_tokens=False,
        enc_act="none",
        dense_encoder=False,
    ):
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_memories = n_memories
        self.inv_temp = inv_temp
        self.T = T
        self.epsilon_ep = epsilon_ep
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp
        self.spectral_scale = spectral_scale
        self.lambda_spring = lambda_spring
        self.normalize_tokens = normalize_tokens

        # Activation for z in encoder coupling
        if enc_act == "none" or enc_act == "identity":
            self.enc_act = None
        elif enc_act == "silu":
            self.enc_act = lambda z: torch.nn.functional.silu(z)
        elif enc_act == "relu2":
            self.enc_act = lambda z: torch.nn.functional.relu(z).pow(2)
        else:
            raise ValueError(f"Unknown enc_act: {enc_act}")

        # -- derived --
        self.stride = stride if stride is not None else patch_size
        self.grid_h = (img_size - patch_size) // self.stride + 1
        self.grid_w = self.grid_h  # square images
        self.n_patches = self.grid_h * self.grid_w

        # ----------------------------------------------------------------
        # Learnable parameters
        # ----------------------------------------------------------------

        # Encoder weights
        self.dense_encoder = dense_encoder
        if dense_encoder:
            # Dense linear: flattened image → all tokens
            img_flat_dim = img_channels * img_size * img_size
            self.encoder_linear = nn.Linear(img_flat_dim, self.n_patches * token_dim, bias=False)
            nn.init.xavier_normal_(self.encoder_linear.weight, gain=init_gain)
            self.encoder_weight = None  # not used
        else:
            # Conv2d kernel: each patch → token vector
            # Shape: (D_T, C, P_H, P_W)
            self.encoder_weight = nn.Parameter(
                torch.empty(token_dim, img_channels, patch_size, patch_size)
            )
            nn.init.xavier_normal_(self.encoder_weight, gain=init_gain)
            self.encoder_linear = None  # not used

        # Encoder bias: shared across patches, shape (D_T,)
        self.encoder_bias = nn.Parameter(torch.zeros(token_dim))

        # Visible bias: learnable per-pixel offset on x (matches MLP's visible_bias)
        self.visible_bias = nn.Parameter(torch.zeros(img_channels, img_size, img_size))

        # Positional bias: per-token per-feature, shape (N_P, D_T)
        self.pos_bias = nn.Parameter(torch.zeros(self.n_patches, token_dim))

        # Memory module: (D_T, N_M)
        self.memory_weight = nn.Parameter(
            torch.empty(token_dim, n_memories)
        )
        nn.init.xavier_normal_(self.memory_weight, gain=init_gain)

        # Attention key/query weights: (N_H, D_H, D_T)
        self.W_K = nn.Parameter(
            torch.empty(n_heads, head_dim, token_dim)
        )
        self.W_Q = nn.Parameter(
            torch.empty(n_heads, head_dim, token_dim)
        )
        # Init each head's weight matrix separately for proper fan computation
        for h in range(n_heads):
            nn.init.xavier_normal_(self.W_K[h], gain=init_gain)
            nn.init.xavier_normal_(self.W_Q[h], gain=init_gain)

        # ----------------------------------------------------------------
        # Diagnostic state
        # ----------------------------------------------------------------
        self._last_convergence_trace = None

    # ==================================================================
    # Utility
    # ==================================================================

    def update_spectral_scale(self, new_scale):
        self.spectral_scale = new_scale

    def _sample_neurons(self, tensor):
        """Sample up to 8 representative values for trace logging."""
        t = tensor.detach().flatten()
        return t[:: max(1, t.numel() // 8)][:8].cpu().tolist()

    def _project_tokens(self, z):
        """
        Project tokens to zero mean and unit std per token.
        z: (B, N_P, D_T) — normalisation over D_T dimension.
        """
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (z - mean) / std

    # ==================================================================
    # Energy components
    # ==================================================================

    def _encode(self, x):
        """
        Encode image → token-space vectors.
        Dense mode: Linear(flatten(x)) → (B, N_P, D_T)
        Conv mode:  Conv2d(x) → (B, N_P, D_T)
        """
        c = self.spectral_scale
        B = x.shape[0]
        if self.dense_encoder:
            enc = c * self.encoder_linear(x.reshape(B, -1))   # (B, N_P * D_T)
            return enc.reshape(B, self.n_patches, self.token_dim)  # (B, N_P, D_T)
        else:
            enc = F.conv2d(x, c * self.encoder_weight, bias=None,
                           stride=self.stride)                # (B, D_T, G_H, G_W)
            enc = enc.reshape(B, self.token_dim, -1)          # (B, D_T, N_P)
            return enc.permute(0, 2, 1)                       # (B, N_P, D_T)

    def _coupling(self, x, z):
        """
        Φ_coupling(x, z) per sample → (B,).

        Sum of encoder, positional bias, memory, and attention couplings.
        """
        # TODO go through this carefully including einsum to sure is correct

        c = self.spectral_scale

        # Visible bias: b · x  (per-pixel learnable offset)
        B = x.shape[0]
        phi_vis = (self.visible_bias.unsqueeze(0) * x).reshape(B, -1).sum(dim=1)  # (B,)

        # 1) Encoder: Σ_ij F(x,W)_ij · z_ij  +  Σ_ij z_ij · b_enc_i
        encoded = self._encode(x)                                   # (B, N_P, D_T)
        z_act = self.enc_act(z) if self.enc_act is not None else z
        phi_enc = (encoded * z_act).sum(dim=[1, 2])                 # (B,)
        phi_bias = (z * self.encoder_bias).sum(dim=[1, 2])          # (B,)

        # 2) Positional bias: Σ_ij z_ij · b_pos_ij
        phi_pos = (z * self.pos_bias.unsqueeze(0)).sum(dim=[1, 2])  # (B,)

        # 3) Memory: Σ_j Σ_k [ReLU(w_k · z_j)]²
        #    z: (B, N_P, D_T), memory_weight: (D_T, N_M)
        mem_proj = torch.matmul(z, c * self.memory_weight)          # (B, N_P, N_M)
        phi_mem = F.relu(mem_proj).pow(2).sum(dim=[1, 2])           # (B,)

        # 4) Attention: (1/γ) Σ_{h,m} log Σ_n exp(γ · A_{h,m,n})
        #    Q_{h,d,p} = Σ_k W^Q_{h,d,k} z_{p,k}
        Q = torch.einsum('bpd,hrd->bhrp', z, c * self.W_Q)         # (B, H, D_H, N_P)
        K = torch.einsum('bpd,hrd->bhrp', z, c * self.W_K)         # (B, H, D_H, N_P)
        # A_{h,m,n} = Σ_d Q_{h,d,m} K_{h,d,n}
        A = torch.einsum('bhdm,bhdn->bhmn', Q, K)                  # (B, H, N_P, N_P)
        gamma = self.inv_temp
        phi_att = (1.0 / gamma) * torch.logsumexp(
            gamma * A, dim=-1).sum(dim=[1, 2])                      # (B,)

        return phi_vis + phi_enc + phi_bias + phi_pos + phi_mem + phi_att

    def _energy(self, neurons):
        """
        E = ½||x||² + ½||z||² - Φ_coupling(x, z)  per sample → (B,).

        neurons: [x, z] where
            x: (B, C, H, W)
            z: (B, N_P, D_T)
        """
        x, z = neurons[0], neurons[1]
        quad_x = 0.5 * x.pow(2).flatten(1).sum(dim=1)   # (B,)
        quad_z = 0.5 * z.pow(2).sum(dim=[1, 2])          # (B,)
        return quad_x + quad_z - self._coupling(x, z)

    # ==================================================================
    # Spring-clamped EP convergence
    # ==================================================================

    def _converge_ep_spring_free(self, x_t, T1, lambda_spring,
                                  record_trace=False):
        """
        Free phase: gradient descent on
            E_spring = E_int(x,z) + (λ/2)||x - x_t||²

        Both x and z evolve.  z is optionally projected to zero-mean unit-std.
        Fully detached — O(1) memory.

        Returns (x_star, [z_star]) matching the MLP/CNN interface.
        """
        eps = self.epsilon_ep
        B = x_t.size(0)
        device = x_t.device
        x_t_det = x_t.detach()

        x = x_t_det.clone()
        z = torch.zeros(B, self.n_patches, self.token_dim, device=device)

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T1):
                if record_trace:
                    trace.append({
                        "x": x[0].norm().item(),
                        "x_neurons": self._sample_neurons(x[0]),
                        "z": z[0].norm().item(),
                        "z_neurons": self._sample_neurons(z[0]),
                    })

                x_g = x.detach().requires_grad_(True)
                z_g = z.detach().requires_grad_(True)

                E_int = self._energy([x_g, z_g])
                spring = (lambda_spring / 2.0) * (x_g - x_t_det).pow(2).sum()
                E_total = E_int.sum() + spring

                gx, gz = torch.autograd.grad(E_total, [x_g, z_g])

                x = (x_g - eps * gx).detach()
                z = (z_g - eps * gz).detach()
                if self.normalize_tokens:
                    z = self._project_tokens(z)

        if record_trace:
            trace.append({
                "x": x[0].norm().item(),
                "x_neurons": self._sample_neurons(x[0]),
                "z": z[0].norm().item(),
                "z_neurons": self._sample_neurons(z[0]),
            })
            self._last_convergence_trace = trace

        return x, [z]

    def _converge_ep_spring_nudged(self, x_t, x_star, h_star, ut, beta,
                                    T2, lambda_spring, record_trace=False):
        """
        Nudged phase for spring-clamped EP.

        target_x = x_t + ut / (α·λ)
        Nudge loss: L = (α·λ)²/2 · ||x - target_x||²

        Returns (x_beta, [z_beta], trace).
        """
        eps = self.epsilon_ep
        scale = self.output_scale
        x_t_det = x_t.detach()
        target_x = (x_t_det + ut.detach() / (scale * lambda_spring)).detach()
        nudge_coeff = beta * (scale * lambda_spring) ** 2 / 2.0

        x = x_star.detach().clone()
        z = h_star[0].detach().clone()

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T2):
                if record_trace:
                    trace.append({
                        "x": x[0].norm().item(),
                        "x_neurons": self._sample_neurons(x[0]),
                        "z": z[0].norm().item(),
                        "z_neurons": self._sample_neurons(z[0]),
                    })

                x_g = x.detach().requires_grad_(True)
                z_g = z.detach().requires_grad_(True)

                E_int = self._energy([x_g, z_g])
                spring = (lambda_spring / 2.0) * (x_g - x_t_det).pow(2).sum()
                nudge = nudge_coeff * (x_g - target_x).pow(2).sum()
                E_total = E_int.sum() + spring + nudge

                gx, gz = torch.autograd.grad(E_total, [x_g, z_g])

                x = (x_g - eps * gx).detach()
                z = (z_g - eps * gz).detach()
                if self.normalize_tokens:
                    z = self._project_tokens(z)

        if record_trace:
            trace.append({
                "x": x[0].norm().item(),
                "x_neurons": self._sample_neurons(x[0]),
                "z": z[0].norm().item(),
                "z_neurons": self._sample_neurons(z[0]),
            })

        return x, [z], trace

    def ep_spring_gradient_step(self, neurons_star, neurons_beta, beta):
        """
        EP parameter gradient: (E_beta - E_star) / beta.
        x and z are detached; only θ is live.
        """
        E_star = self._energy(neurons_star)   # (B,) live θ
        E_beta = self._energy(neurons_beta)   # (B,) live θ
        ep_loss = (E_beta - E_star).mean() / beta
        ep_loss.backward()

    # ==================================================================
    # Non-spring convergence (x fixed) — for diagnostics & Jacobian
    # ==================================================================

    def _converge_ep_free(self, x, T1, record_trace=False):
        """Free phase with x clamped (no spring). Returns [z_star]."""
        eps = self.epsilon_ep
        B = x.size(0)
        device = x.device

        z = torch.zeros(B, self.n_patches, self.token_dim, device=device)

        trace = [] if record_trace else None

        with torch.enable_grad():
            for _ in range(T1):
                if record_trace:
                    trace.append({
                        "z": z[0].norm().item(),
                        "z_neurons": self._sample_neurons(z[0]),
                    })
                z_g = z.detach().requires_grad_(True)
                E = self._energy([x.detach(), z_g]).sum()
                gz = torch.autograd.grad(E, z_g)[0]
                z = (z_g - eps * gz).detach()
                if self.normalize_tokens:
                    z = self._project_tokens(z)

        if record_trace:
            trace.append({
                "z": z[0].norm().item(),
                "z_neurons": self._sample_neurons(z[0]),
            })
            self._last_convergence_trace = trace

        return [z]

    def _converge_detached(self, x):
        """For Jacobian spectral radius computation."""
        return self._converge_ep_free(x, self.T)

    # ==================================================================
    # Velocity
    # ==================================================================

    def velocity_at_h(self, x, hidden):
        """
        Compute v = -α ∇_x E at given z, without re-convergence.
        Used for logging after free phase.
        """
        z = hidden[0]
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            E = self._energy([x_req, z.detach()])
            v = -self.output_scale * torch.autograd.grad(E.sum(), x_req)[0]
        return v.detach()

    def velocity_energy_gd(self, x, t, h_steps=None):
        """
        Generation velocity via energy gradient descent.
        x is FIXED, z evolves to equilibrium, then
            v = -α ∇_x [E(x,z*)]
        """
        if h_steps is None:
            h_steps = self.T

        B = x.size(0)
        device = x.device
        eps = self.epsilon_ep

        # Converge z with x fixed
        z = torch.zeros(B, self.n_patches, self.token_dim, device=device)
        with torch.enable_grad():
            for _ in range(h_steps):
                z_g = z.detach().requires_grad_(True)
                E = self._energy([x.detach(), z_g]).sum()
                gz = torch.autograd.grad(E, z_g)[0]
                z = (z_g - eps * gz).detach()
                if self.normalize_tokens:
                    z = self._project_tokens(z)

        # Compute velocity from full _energy (now includes ½||x||²)
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            E_full = self._energy([x_req, z.detach()])
            v = -self.output_scale * torch.autograd.grad(E_full.sum(), x_req)[0]
        return v.detach()

    def velocity(self, x, t):
        """Spring-mode velocity: v = α·λ·(x* - x_t)."""
        x_star, _ = self._converge_ep_spring_free(
            x.detach(), self.T, self.lambda_spring)
        return self.output_scale * self.lambda_spring * (x_star - x.detach())

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        if return_potential:
            return self.potential(x, t)
        elif getattr(self, '_gen_energy_gd', False):
            return self.velocity_energy_gd(x, t)
        else:
            return self.velocity(x, t)

    def potential(self, x, t, record_trace=False):
        """V(x) = E(x, z*) · output_scale."""
        _, h_star = self._converge_ep_spring_free(
            x, self.T, self.lambda_spring, record_trace=record_trace)
        E = self._energy([x, h_star[0]])
        V = E * self.output_scale
        if self.energy_clamp is not None and self.energy_clamp > 0:
            V = self.energy_clamp * torch.tanh(V / self.energy_clamp)
        return V

    # ==================================================================
    # Jacobian diagnostics
    # ==================================================================

    def compute_jacobian_spectral_radius(self, x, hidden_star, n_iters=20):
        z_star = hidden_star[0]
        eps = self.epsilon_ep

        v = torch.randn_like(z_star)
        v = v / v.norm()

        rho_history = []
        for _ in range(n_iters):
            z = z_star.detach().clone().requires_grad_(True)
            E = self._energy([x.detach(), z]).sum()
            gz = torch.autograd.grad(E, z, create_graph=True)[0]
            z_new = z - eps * gz
            if self.normalize_tokens:
                z_new = self._project_tokens(z_new)

            Jv = torch.autograd.grad(z_new, z, grad_outputs=v)[0]
            Jv_norm = Jv.norm().item()
            rho_history.append(Jv_norm)

            if Jv_norm > 1e-12:
                v = Jv.detach() / Jv_norm
            else:
                break

        return rho_history[-1] if rho_history else 0.0, rho_history

    # ==================================================================
    # Diagnostic plots
    # ==================================================================

    def save_convergence_plot(self, save_dir, step):
        trace = self._last_convergence_trace
        if trace is None:
            return
        timesteps = list(range(len(trace)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(timesteps, [t.get("x", 0) for t in trace], linewidth=1.5)
        axes[0].set_xlabel("Step"); axes[0].set_ylabel("||x||")
        axes[0].set_title("x convergence"); axes[0].grid(True, alpha=0.3)

        axes[1].plot(timesteps, [t.get("z", 0) for t in trace], linewidth=1.5)
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("||z||")
        axes[1].set_title("z (tokens) convergence"); axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"CET convergence (step {step})")
        fig.tight_layout()
        path = os.path.join(save_dir, f"convergence_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def save_layer_activations_plot(self, save_dir, step):
        """Save per-neuron convergence traces for x and z."""
        trace = self._last_convergence_trace
        if trace is None:
            return

        timesteps = list(range(len(trace)))

        # Two panels: x neurons, z neurons
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # x neurons
        if "x_neurons" in trace[0] and len(trace[0]["x_neurons"]) > 0:
            n_neurons = len(trace[0]["x_neurons"])
            for n in range(n_neurons):
                values = [t["x_neurons"][n] for t in trace]
                axes[0].plot(timesteps, values, linewidth=0.5, alpha=0.6)
            axes[0].set_xlabel("Convergence step")
            axes[0].set_ylabel("x neuron value")
            axes[0].set_title("x (visible) neuron traces")
            axes[0].grid(True, alpha=0.3)

        # z neurons
        if "z_neurons" in trace[0] and len(trace[0]["z_neurons"]) > 0:
            n_neurons = len(trace[0]["z_neurons"])
            for n in range(n_neurons):
                values = [t["z_neurons"][n] for t in trace]
                axes[1].plot(timesteps, values, linewidth=0.5, alpha=0.6)
            axes[1].set_xlabel("Convergence step")
            axes[1].set_ylabel("z neuron value")
            axes[1].set_title("z (token) neuron traces")
            axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"CET neuron traces (step {step})")
        fig.tight_layout()
        path = os.path.join(save_dir, f"neuron_traces_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[neuron_traces] Saved {path}")

    def save_nudge_traces_plot(self, save_dir, step):
        """
        Plot per-neuron displacement from h* during nudge phase(s).
        Two panels: x displacement, z displacement.
        Solid = positive nudge, dashed = negative nudge.
        """
        free_final = getattr(self, '_last_spring_free_final', None)
        trace_pos = getattr(self, '_last_nudge_pos_trace', None)
        if free_final is None or trace_pos is None:
            return
        trace_neg = getattr(self, '_last_nudge_neg_trace', None)

        neuron_keys = [k for k in trace_pos[0].keys()
                    if k.endswith('_neurons')]
        if not neuron_keys:
            return

        n_panels = len(neuron_keys)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), squeeze=False)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx, key in enumerate(neuron_keys):
            ax = axes[0, idx]
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
            ax.set_xlabel("Nudge step")
            ax.set_ylabel(f"Δ{label}")
            ax.set_title(f"{label} nudge displacement")
            ax.grid(True, alpha=0.3)

        axes[0, 0].legend(fontsize=6, ncol=2, title="neuron (solid=+β, dash=−β)")
        fig.suptitle(f"CET nudge displacement from equilibrium (step {step})")
        fig.tight_layout()
        path = os.path.join(save_dir, f"nudge_displacement_step_{step}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[nudge_displacement] Saved {path}")