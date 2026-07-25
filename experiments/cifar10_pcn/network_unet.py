# File: network_unet.py
# UNet-based Energy-Matching potential V(x) models for CIFAR-10 (3×32×32).
# Contains three feedforward architectures sharing the potential/velocity/forward
# interface:
#   EBViTModelWrapper   torchcfm UNet + patch-ViT scalar head  (model_type=unet_vit)
#   EBMLPModelWrapper   torchcfm UNet + avgpool-MLP head        (model_type=unet_mlp)
#   EBRonnebergerConvUNetWrapper
#                       from-scratch conv UNet, RONNEBERGER-style connectivity
#                       (model_type=ffn_ronneberger_conv_unet)
#
# NOTE the two UNet lineages here are NOT the same architecture. The torchcfm
# backbone above is the DDPM/diffusion UNet (Ho et al.): every encoder block
# pushes a skip onto a stack and EVERY decoder block pops one, pre-activation
# ResBlocks, zero-init second conv, GroupNorm throughout, learned conv
# down/up-sampling. The conv UNet below is the classic Ronneberger (2015)
# U-Net: ONE skip per RESOLUTION LEVEL. Renamed 2026-07-21 (in mnist_pcn) to
# stop the two being conflated.
#
# To ablate attention/normalization, prefer stripping them from the torchcfm
# backbone (--unet_no_attention / --unet_no_norm) over comparing against the
# Ronneberger net: stripping varies ONE axis on the SAME code path, whereas a
# cross-family comparison varies many at once.
# (Ported from experiments/mnist_pcn/network_unet.py with CIFAR-10 defaults;
# the Ronneberger net is generalized to arbitrary in_channels / level counts.)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcfm.models.unet.unet import UNetModelWrapper, AttentionBlock

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
        in_channels=3,
        patch_size=4,  # Paper value for CIFAR-10 (Figure 7). 32/4 = 8 -> 64 tokens.
        embed_dim=384,
        image_size=(32, 32),
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
# Crossbar-native ablations: strip attention / normalization in place
##############################################################################
def _replace_all(root, predicate, factory):
    """Swap every submodule satisfying `predicate` for `factory()`. Returns count.

    Operates on the ALREADY-BUILT module tree rather than on constructor args,
    which is what makes the guarantee airtight: whatever torchcfm chose to
    build, these ops are gone afterwards (their parameters go with them).
    """
    n = 0
    for parent in root.modules():
        for name, child in list(parent.named_children()):
            if predicate(child):
                setattr(parent, name, factory())
                n += 1
    return n


def strip_attention(model):
    """Remove EVERY AttentionBlock, leaving convs/skips only.

    NB this is NOT achievable with --attention_resolutions alone: that flag
    controls the encoder/decoder stages, but UNetModel hardcodes the middle
    block as ResBlock -> AttentionBlock -> ResBlock, so one attention block
    always survives. AttentionBlock is not a TimestepBlock, so
    TimestepEmbedSequential calls the replacement Identity as layer(x) -- the
    substitution is shape- and signature-safe.
    """
    return _replace_all(model, lambda m: isinstance(m, AttentionBlock), nn.Identity)


def strip_norm(model):
    """Remove EVERY GroupNorm.

    Leaves an architecture whose every op is an MVM (conv / linear), a fixed
    fan-out (nearest upsample), a wiring op (concat / residual add) or an
    elementwise nonlinearity -- i.e. analog-crossbar-native. The constant
    t=0.5 time embedding stays, but it is a FIXED per-channel bias and folds
    into the preceding conv's bias at inference, so it costs no hardware.

    Stability caveat: a deep pre-activation ResNet with no normalization can
    diverge. torchcfm's zero-init second conv + always-on residual make each
    block start as the identity, which is what makes this survivable -- but it
    is not guaranteed, so smoke-test before spending a full run.
    """
    return _replace_all(model, lambda m: isinstance(m, nn.GroupNorm), nn.Identity)


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
        dim=(3, 32, 32),           # Input image shape (C, H, W). MNIST: (1, 28, 28)
        num_channels=128,          # Base channel count in UNet. All stages scale from this. MNIST: 32
        num_res_blocks=2,          # Residual blocks per UNet resolution stage
        channel_mult=[1, 2, 2, 2], # Per-stage channel multiplier → channels are [128, 256, 256, 256]. MNIST: [1,2,2]
        attention_resolutions="16", # Self-attention at these resolutions (csv). Converted to a
                                    # DOWNSAMPLE FACTOR: ds = image_size // res. "16" -> 32//16 = 2,
                                    # i.e. attention at the half-res 16x16 stage (paper value).
        num_heads=4,               # Number of attention heads in UNet self-attention layers. MNIST: 2
        num_head_channels=64,      # Channels per attention head. NB torchcfm lets this
                                   # OVERRIDE num_heads: heads = channels // num_head_channels.
                                   # CIFAR is self-consistent (attention blocks have 256 ch,
                                   # 256//64 = 4 = the paper's stated "4 heads"). MNIST's blocks
                                   # only have 64 ch, so it needed 32 to get 2 heads.
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
        patch_size=4,              # Split UNet output into patches → 32/4 = 8×8 = 64 tokens. Paper value (Figure 7).
        embed_dim=384,             # Dimension of each patch token embedding. MNIST: 128
        transformer_nheads=4,      # Attention heads in the ViT transformer encoder. MNIST: 2
        transformer_nlayers=8,     # Number of transformer encoder layers (depth). MNIST: 2
        include_pos_embed=True,    # Add learnable positional embeddings to patch tokens
        # --- Energy model parameters ---
        output_scale=1000.0,       # Multiplier for scalar energy V(x). MNIST: 100.0 (fewer pixels → smaller energies)
        energy_clamp=None,         # Tanh-based soft clamp on V(x). None = no clamping
        # --- crossbar-native ablations (see strip_attention / strip_norm) ---
        no_attention=False,        # Drop ALL attention, incl. the hardcoded middle block
        no_norm=False,             # Drop ALL GroupNorm -> every op is an MVM/wiring op
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

        # Applied to the BUILT tree, so the guarantee holds regardless of how
        # attention_resolutions was interpreted (see strip_attention docstring).
        if no_attention:
            strip_attention(self)
        if no_norm:
            strip_norm(self)

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
        dim=(3, 32, 32),           # Input image shape (C, H, W)
        num_channels=128,          # Base channel count in UNet
        num_res_blocks=2,          # Residual blocks per resolution stage
        channel_mult=[1, 2, 2, 2], # Per-stage channel multiplier
        attention_resolutions="16", # Self-attention at these spatial resolutions
        num_heads=4,               # Attention heads in UNet
        num_head_channels=64,      # Channels per attention head (see EBViT: 256 ch // 64 -> 4 heads)
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
        output_scale=1000.0,       # Multiplier for scalar energy V(x)
        energy_clamp=None,         # Tanh-based soft clamp on V(x). None = no clamping
        no_attention=False,        # Drop ALL attention, incl. the hardcoded middle block
        no_norm=False,             # Drop ALL GroupNorm -> every op is an MVM/wiring op
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

        # Applied to the BUILT tree, so the guarantee holds regardless of how
        # attention_resolutions was interpreted (see strip_attention docstring).
        if no_attention:
            strip_attention(self)
        if no_norm:
            strip_norm(self)

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
# 3) Attention-free, NORM-free RONNEBERGER conv UNet (crossbar-native)
#    NOTE: unlike EBViT/EBMLP above (which wrap the torchcfm/DDPM UNet), this is
#    a from-scratch conv UNet — no torchcfm dependency, no attention, no norm
#    (unless --conv_unet_norm). Every op maps to analog matrix-vector mult.
#    Its connectivity is Ronneberger-style (1 skip per resolution level), NOT
#    the DDPM stack-based scheme — hence the name.
#    Generalized (vs the mnist_pcn version) to arbitrary in_channels and
#    level counts: len(channel_mult) levels → resolutions H, H/2, H/4, ...
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
        raise ValueError(f"ronneberger_conv_unet supports 'avgpool' or 'stride_conv', got {pool_type}")


class RonnebergerConvUNetEnergy(nn.Module):
    """Attention-free, norm-free RONNEBERGER-style U-Net scalar energy V(x).

    Connectivity is the classic Ronneberger (2015) U-Net: ONE skip per
    resolution level. This is deliberately NOT the torchcfm/DDPM backbone
    above, which carries one skip per BLOCK, pre-activation ResBlocks with
    zero-init second convs, GroupNorm throughout, and learned conv
    down/up-sampling. Other differences: this net is post-activation, drops
    the residual entirely when channel count changes (rather than inserting a
    1x1 conv), has no time-embedding bias, and upsamples with bare
    nearest-neighbour interpolation (no conv after).

    len(channel_mult) sets the number of resolution levels: for CIFAR-10 the
    default (1, 2, 2, 2) gives 4 levels at 32/16/8/4 with 3 skips. (The
    mnist_pcn original hardcoded 3 levels at 28/14/7; this version builds the
    encoder/decoder stacks generically, so its state_dict keys differ.)

    Consequence: it is an INDEPENDENT architecture family, not "the UNet
    minus attention". A cross-family comparison therefore varies many axes
    at once -- for a single-axis attention/norm ablation use the torchcfm
    backbone with --unet_no_attention / --unet_no_norm instead.
    """

    def __init__(self, in_channels=3, num_channels=32, channel_mult=(1, 2, 2, 2),
                 pool_type="avgpool", use_norm=False):
        super().__init__()
        assert len(channel_mult) >= 2, "at least 2 resolution levels expected"
        chans = [num_channels * m for m in channel_mult]
        def CB(a, b): return ConvBlock(a, b, use_norm=use_norm)

        # Encoder: one 2-ConvBlock stage + downsample per level except the last
        self.encs = nn.ModuleList()
        self.downs = nn.ModuleList()
        c_prev = in_channels
        for c in chans[:-1]:
            self.encs.append(nn.Sequential(CB(c_prev, c), CB(c, c)))
            self.downs.append(make_down(c, pool_type))
            c_prev = c

        # Bottleneck at the coarsest resolution
        self.mid = nn.Sequential(CB(c_prev, chans[-1]), CB(chans[-1], chans[-1]))

        # Decoder: upsample + concat the level's skip + 2-ConvBlock stage
        self.decs = nn.ModuleList()
        c_prev = chans[-1]
        for c in reversed(chans[:-1]):
            self.decs.append(nn.Sequential(CB(c_prev + c, c), CB(c, c)))
            c_prev = c

        c1 = chans[0]
        self.head_conv = nn.Conv2d(c1, c1, 3, padding=1)
        self.head_fc1 = nn.Linear(c1, c1)
        self.head_fc2 = nn.Linear(c1, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        skips = []
        h = x
        for enc, down in zip(self.encs, self.downs):
            h = enc(h)
            skips.append(h)                                  # one skip per level
            h = down(h)
        h = self.mid(h)                                      # coarsest resolution
        for dec in self.decs:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = dec(torch.cat([h, skips.pop()], dim=1))
        h = self.act(self.head_conv(h))
        h = h.mean(dim=[2, 3])                               # global avgpool (B,c1)
        h = self.act(self.head_fc1(h))
        return self.head_fc2(h)                              # (B,1) raw scalar


class EBRonnebergerConvUNetWrapper(nn.Module):
    """Same interface as EBCNNModelWrapper: potential/velocity/forward."""

    def __init__(self, output_scale=1000.0, energy_clamp=None,
                 in_channels=3, num_channels=32, channel_mult=(1, 2, 2, 2),
                 pool_type="avgpool", use_norm=False):
        super().__init__()
        self.net = RonnebergerConvUNetEnergy(in_channels=in_channels,
                                  num_channels=num_channels,
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
