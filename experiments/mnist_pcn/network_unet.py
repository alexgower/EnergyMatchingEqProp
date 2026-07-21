# File: network_unet.py
# UNet-based Energy-Matching potential V(x) models for MNIST (1×28×28).
# Contains three feedforward architectures sharing the potential/velocity/forward
# interface:
#   EBViTModelWrapper   torchcfm UNet + patch-ViT scalar head  (model_type=unet_vit)
#   EBMLPModelWrapper   torchcfm UNet + avgpool-MLP head        (model_type=unet_mlp)
#   EBConvUNetWrapper   from-scratch attention/norm-free conv UNet (model_type=conv_unet)
# (Renamed from network_transformer_vit.py; EBConvUNetWrapper merged in from the
# former network_unet_connectivity.py.)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcfm.models.unet.unet import UNetModelWrapper

##############################################################################
# Simple Patch Embedding (like in ViT)
##############################################################################
class PatchEmbed(nn.Module):
    """
    Splits the (B, C, H, W) feature map into non-overlapping patches and
    embeds each patch to `embed_dim`.
    """
    def __init__(
        self,
        in_channels=1,
        patch_size=4,  # Paper-inferred. Appendix D never states MNIST's patch size,
                       # but Figure 7 gives CIFAR patch_size=4 and the MNIST paragraph
                       # enumerates exactly what the downscale changed (channels,
                       # channel_mult, UNet heads, embed_dim, layers, ViT heads,
                       # output_scale) — patch size is NOT among them, so it stayed 4.
                       # 28/4 = 7 -> 49 tokens. (Runs before 2026-07-21 used 7 -> 16
                       # tokens. NB param count is IDENTICAL either way, 2,028,290: the
                       # patch conv and pos_embed trade 6272<->2048 — so "matches the
                       # paper's 2M" can NOT disambiguate patch size.)
        embed_dim=128,
        image_size=(28, 28),
        include_pos_embed=True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding via Conv2d
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Optional learnable positional embeddings
        self.include_pos_embed = include_pos_embed
        if include_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim)
            )
        else:
            self.pos_embed = None

        # Initialize patch embedding weights
        nn.init.xavier_uniform_(self.patch_embed.weight)

    def forward(self, x):
        # x: (B, C, H, W)
        # => (B, E, H', W') after patch_embed
        x = self.patch_embed(x)  # shape: (B, embed_dim, H', W')
        B, E, Hp, Wp = x.shape
        # Flatten
        x = x.view(B, E, Hp * Wp).transpose(1, 2)  # => (B, N, E)
        # Add positional embedding if needed
        if self.pos_embed is not None:
            x = x + self.pos_embed  # (1, N, E) broadcast
        return x


##############################################################################
# Soft clamp
##############################################################################
def soft_clamp(x, clamp_val):
    """Tanh-based clamp: output in [-clamp_val, clamp_val]."""
    return clamp_val * torch.tanh(x / clamp_val)


##############################################################################
# Helper to create a dummy time tensor
##############################################################################
def dummy_time(x, value=0.5):
    """
    Create a (B,)-shaped tensor of `value`, matching x's device and dtype.
    """
    return torch.full(
        (x.shape[0],),
        value,
        device=x.device,
        dtype=x.dtype
    )


##############################################################################
# 1) EBM with a patch-based ViT head
##############################################################################
class EBViTModelWrapper(UNetModelWrapper):
    """
    Energy-Based Model with a patch-based ViT on top of the UNet output.
    Ignores the input time; always feeds a fixed dummy time to the UNet.
    
    Note: potential() and velocity() now accept (x, t) where t is ignored.
    # TODO make sure all these parameters are equivalent to EnergyMatching paper where we can
    """

    def __init__(
        self,
        # --- UNet backbone parameters ---
        dim=(1, 28, 28),           # Input image shape (C, H, W). CIFAR-10: (3, 32, 32)
        num_channels=32,           # Base channel count in UNet. All stages scale from this. CIFAR-10: 128
        num_res_blocks=2,          # Residual blocks per UNet resolution stage
        channel_mult=[1, 2, 2],    # Per-stage channel multiplier → channels are [32, 64, 64]. CIFAR-10: [1,2,2,2]
        attention_resolutions="14", # Self-attention at these resolutions (csv). Converted to a
                                    # DOWNSAMPLE FACTOR: ds = image_size // res. "14" -> 28//14 = 2,
                                    # matching CIFAR's 32//16 = 2 (attention at the half-res stage).
                                    # Inheriting "16" literally would give 28//16 = 1, i.e. attention
                                    # at full 28x28 — structurally different and far costlier.
        num_heads=2,               # Number of attention heads in UNet self-attention layers. CIFAR-10: 4
        num_head_channels=32,      # Channels per attention head. NB torchcfm lets this
                                   # OVERRIDE num_heads: heads = channels // num_head_channels.
                                   # CIFAR is self-consistent (attention blocks have 256 ch,
                                   # 256//64 = 4 = its stated "4 heads"). MNIST's blocks have
                                   # only 64 ch, so inheriting 64 would give 1 head — silently
                                   # contradicting Appendix D's "attention heads in the UNet
                                   # to 2". 32 gives 64//32 = 2 heads as stated. Param count
                                   # unchanged (qkv/proj depend on channels, not head count).
        dropout=0.1,               # Dropout rate in UNet residual and attention layers
        # UNet architectural flags (rarely changed)
        class_cond=False,          # Class-conditional generation (unused — no labels)
        learn_sigma=False,         # Learn noise variance (diffusion-specific, unused here)
        use_checkpoint=False,      # Gradient checkpointing to save memory
        use_fp16=False,            # Half-precision forward pass
        resblock_updown=False,     # Use residual blocks for up/downsampling instead of conv
        use_scale_shift_norm=False, # AdaGN: scale+shift norm by time embedding
        use_new_attention_order=False, # Alternate attention implementation
        # --- ViT head parameters (applied to UNet output) ---
        patch_size=4,              # Split UNet output into patches → 28/4 = 7×7 = 49 tokens. Paper-inferred (Figure 7 CIFAR=4, not in MNIST downscale list).
        embed_dim=128,             # Dimension of each patch token embedding. CIFAR-10: 128
        transformer_nheads=2,      # Attention heads in the ViT transformer encoder. CIFAR-10: 4
        transformer_nlayers=2,     # Number of transformer encoder layers (depth). CIFAR-10: 2
        include_pos_embed=True,    # Add learnable positional embeddings to patch tokens
        # --- Energy model parameters ---
        output_scale=100.0,        # Multiplier for scalar energy V(x). CIFAR-10: 1000.0 (more pixels → larger energies)
        energy_clamp=None,         # Tanh-based soft clamp on V(x). None = no clamping
        **kwargs
    ):
        super().__init__(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            class_cond=class_cond,
            learn_sigma=learn_sigma,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            resblock_updown=resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            use_new_attention_order=use_new_attention_order,
            **kwargs
        )

        self.out_channels = dim[0]
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

        # 1) PatchEmbed for the UNet output
        self.patch_embed = PatchEmbed(
            in_channels=self.out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=dim[1:],  # (H, W)
            include_pos_embed=include_pos_embed
        )

        # 2) A small Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_nheads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_nlayers
        )

        # 3) Final linear => scalar
        self.final_linear = nn.Linear(embed_dim, 1)

    def potential(self, x, t):
        """
        Computes scalar potential V(x,t) => shape (B,).
        Ignores the provided time and always uses a fixed dummy time.
        """
        t_dummy = dummy_time(x, value=0.5)
        # UNet forward: shape (B, C, H, W)
        unet_out = super().forward(t_dummy, x)
        # Patch-embed: (B, N, embed_dim)
        tokens = self.patch_embed(unet_out)
        # Transformer: (B, N, embed_dim)
        encoded = self.transformer_encoder(tokens)
        # Mean-pool across tokens: (B, embed_dim)
        pooled = encoded.mean(dim=1)
        # Final linear to scalar: (B, 1) -> (B,)
        V = self.final_linear(pooled).view(-1)
        V = V * self.output_scale
        if self.energy_clamp is not None:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """
        Computes -∂V/∂x => shape (B, C, H, W).
        Ignores the provided time.
        """
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            V = self.potential(x, t)
            dVdx = torch.autograd.grad(
                outputs=V,
                inputs=x,
                grad_outputs=torch.ones_like(V),
                create_graph=True
            )[0]
            return -dVdx

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        """
        Forward pass accepts a time tensor and an input tensor.
        The provided time is ignored (dummy time is used internally).
        If return_potential=True, returns V(x,t); otherwise returns velocity.
        """
        if return_potential:
            return self.potential(x, t)
        else:
            return self.velocity(x, t)


##############################################################################
# 2) EBM with a simple MLP head (SiLU in the hidden layer)
##############################################################################
class EBMLPModelWrapper(UNetModelWrapper):
    """
    Energy-Based Model that extends the UNet code but uses a simple MLP head.
    Ignores the provided time; always feeds a fixed dummy time to the UNet.
    
    MLP architecture:
        Global average pool -> Linear -> SiLU -> Linear -> scalar V(x,t).
    """

    def __init__(
        self,
        # --- UNet backbone parameters (same as EBViTModelWrapper) ---
        dim=(1, 28, 28),           # Input image shape (C, H, W)
        num_channels=32,           # Base channel count in UNet
        num_res_blocks=2,          # Residual blocks per resolution stage
        channel_mult=[1, 2, 2],    # Per-stage channel multiplier
        attention_resolutions="14", # Self-attention at these spatial resolutions
        num_heads=2,               # Attention heads in UNet
        num_head_channels=32,      # Channels per attention head (see EBViT: 32 -> 2 heads, per Appendix D)
        dropout=0.1,               # Dropout rate
        # UNet architectural flags (rarely changed)
        class_cond=False,
        learn_sigma=False,
        use_checkpoint=False,
        use_fp16=False,
        resblock_updown=False,
        use_scale_shift_norm=False,
        use_new_attention_order=False,
        # --- Energy model parameters ---
        output_scale=100.0,        # Multiplier for scalar energy V(x)
        energy_clamp=None,         # Tanh-based soft clamp on V(x). None = no clamping
        **kwargs
    ):
        super().__init__(
            dim=dim,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            class_cond=class_cond,
            learn_sigma=learn_sigma,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            resblock_updown=resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            use_new_attention_order=use_new_attention_order,
            **kwargs
        )

        self.out_channels = dim[0]
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Simple MLP: [Flatten -> Linear -> SiLU -> Linear -> scalar]
        self.mlp = nn.Sequential(
            nn.Flatten(),                     # => (B, C)
            nn.Linear(self.out_channels, self.out_channels),
            nn.SiLU(),
            nn.Linear(self.out_channels, 1)   # => scalar output
        )

    def potential(self, x, t):
        """
        Computes scalar potential V(x,t) => shape (B,).
        Ignores the provided time.
        """
        t_dummy = dummy_time(x, value=0.5)
        # UNet forward: (B, C, H, W)
        unet_out = super().forward(t_dummy, x)
        # Global average pool: (B, C, 1, 1)
        pooled = self.pool(unet_out)
        # MLP: (B, 1) -> (B,)
        V = self.mlp(pooled).view(-1)
        V = V * self.output_scale
        if self.energy_clamp is not None:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """
        Computes -∂V/∂x => shape (B, C, H, W).
        Ignores the provided time.
        """
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            V = self.potential(x, t)
            dVdx = torch.autograd.grad(
                outputs=V,
                inputs=x,
                grad_outputs=torch.ones_like(V),
                create_graph=True
            )[0]
            return -dVdx

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        """
        Forward pass accepts a time tensor and an input tensor.
        The provided time is ignored (dummy time is used internally).
        If return_potential=True, returns V(x,t); otherwise returns velocity.
        """
        if return_potential:
            return self.potential(x, t)
        else:
            return self.velocity(x, t)


##############################################################################
# 3) Attention-free, NORM-free conv UNet (crossbar-native)
#    NOTE: unlike EBViT/EBMLP above (which wrap the torchcfm UNet), this is a
#    from-scratch conv UNet — no torchcfm dependency, no attention, no norm
#    (unless --conv_unet_norm). Every op maps to analog matrix-vector mult.
#    Reuses soft_clamp defined above.
##############################################################################
def _norm(channels, use_norm):
    """GroupNorm with the largest power-of-2 group count ≤32 that divides
    `channels` (matches torchcfm's 32-group convention where possible).
    Identity when use_norm is False (keeps the crossbar-native variant)."""
    if not use_norm:
        return nn.Identity()
    g = 32
    while channels % g:
        g //= 2
    return nn.GroupNorm(g, channels)


class ConvBlock(nn.Module):
    """conv-[GN]-SiLU-conv-[GN]-SiLU with additive residual when channels match."""

    def __init__(self, c_in, c_out, use_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm1 = _norm(c_out, use_norm)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = _norm(c_out, use_norm)
        self.act = nn.SiLU()
        self.residual = (c_in == c_out)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        return x + h if self.residual else h


def make_down(channels, pool_type):
    if pool_type == "avgpool":
        return nn.AvgPool2d(2, 2)
    elif pool_type == "stride_conv":
        return nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
    else:
        raise ValueError(f"conv_unet supports 'avgpool' or 'stride_conv', got {pool_type}")


class ConvUNetEnergy(nn.Module):
    """Attention-free, norm-free UNet-connectivity scalar energy V(x)."""

    def __init__(self, in_channels=1, num_channels=32, channel_mult=(1, 2, 2),
                 pool_type="avgpool", use_norm=False):
        super().__init__()
        assert len(channel_mult) == 3, "3 resolution levels (28/14/7) expected"
        c1, c2, c3 = [num_channels * m for m in channel_mult]
        def CB(a, b): return ConvBlock(a, b, use_norm=use_norm)

        self.enc28 = nn.Sequential(CB(in_channels, c1), CB(c1, c1))
        self.down1 = make_down(c1, pool_type)
        self.enc14 = nn.Sequential(CB(c1, c2), CB(c2, c2))
        self.down2 = make_down(c2, pool_type)
        self.mid7 = nn.Sequential(CB(c2, c3), CB(c3, c3))
        self.dec14 = nn.Sequential(CB(c3 + c2, c2), CB(c2, c2))
        self.dec28 = nn.Sequential(CB(c2 + c1, c1), CB(c1, c1))
        self.head_conv = nn.Conv2d(c1, c1, 3, padding=1)
        self.head_fc1 = nn.Linear(c1, c1)
        self.head_fc2 = nn.Linear(c1, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        s1 = self.enc28(x)                                   # (B,c1,28,28)
        h = self.down1(s1)
        s2 = self.enc14(h)                                   # (B,c2,14,14)
        h = self.down2(s2)
        h = self.mid7(h)                                     # (B,c3,7,7)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.dec14(torch.cat([h, s2], dim=1))            # (B,c2,14,14)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.dec28(torch.cat([h, s1], dim=1))            # (B,c1,28,28)
        h = self.act(self.head_conv(h))
        h = h.mean(dim=[2, 3])                               # global avgpool (B,c1)
        h = self.act(self.head_fc1(h))
        return self.head_fc2(h)                              # (B,1) raw scalar


class EBConvUNetWrapper(nn.Module):
    """Same interface as EBCNNModelWrapper: potential/velocity/forward."""

    def __init__(self, output_scale=100.0, energy_clamp=None,
                 num_channels=32, channel_mult=(1, 2, 2), pool_type="avgpool",
                 use_norm=False):
        super().__init__()
        self.net = ConvUNetEnergy(num_channels=num_channels,
                                  channel_mult=channel_mult,
                                  pool_type=pool_type, use_norm=use_norm)
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

    def potential(self, x, t):
        """Scalar potential V(x) => shape (B,). Time is ignored."""
        V = self.net(x).view(-1)
        V = V * self.output_scale
        if self.energy_clamp is not None and self.energy_clamp > 0:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """v = -dV/dx => shape (B, C, H, W)."""
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            V = self.potential(x, t)
            dVdx = torch.autograd.grad(
                outputs=V, inputs=x,
                grad_outputs=torch.ones_like(V),
                create_graph=True,
            )[0]
            return -dVdx

    def forward(self, t, x, return_potential=False, *args, **kwargs):
        if return_potential:
            return self.potential(x, t)
        return self.velocity(x, t)
