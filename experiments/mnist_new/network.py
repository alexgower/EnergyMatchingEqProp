# File: network.py
# EBM models for MNIST (1×28×28).
#
# Two model options:
#   1) "unet_vit" — UNet + ViT head (paper's described architecture)
#   2) "cnn"      — Simple CNN + MLP (from the authors' earlier code)
#
# Both expose the same interface: potential(x, t), velocity(x, t), forward(t, x).

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
        patch_size=7,
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
# 1) EBM with UNet + ViT head  (paper's described architecture)
##############################################################################
class EBViTModelWrapper(UNetModelWrapper):
    """
    Energy-Based Model with a patch-based ViT on top of the UNet output.
    Ignores the input time; always feeds a fixed dummy time to the UNet.

    Adapted for MNIST: dim=(1, 28, 28), patch_size=7 => 4×4 = 16 patches.
    Paper spec: embed_dim=128, 2 transformer layers, 2 heads, output_scale=100.
    """

    def __init__(
        self,
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=2,
        channel_mult=[1, 2, 2],
        attention_resolutions="14",
        num_heads=2,
        num_head_channels=32,
        dropout=0.1,
        # UNet flags
        class_cond=False,
        learn_sigma=False,
        use_checkpoint=False,
        use_fp16=False,
        resblock_updown=False,
        use_scale_shift_norm=False,
        use_new_attention_order=False,
        # ViT-specific
        patch_size=7,
        embed_dim=128,
        transformer_nheads=2,
        transformer_nlayers=2,
        include_pos_embed=True,
        # EBM extras
        output_scale=100.0,
        energy_clamp=None,
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
        if self.energy_clamp is not None and self.energy_clamp > 0:
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
# 2) Simple CNN + MLP  (from the authors' earlier MNIST code)
##############################################################################

class DoubleConv(nn.Module):
    """Two consecutive conv(3×3) + GELU. No normalization layers."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x


class Downsample(nn.Module):
    """Downsample by 2× using stride-2 conv (kernel_size=4, stride=2, pad=1)."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=True
        )

    def forward(self, x):
        return self.conv(x)


class Network_2M_MNIST28x28(nn.Module):
    """
    Simple CNN energy model for MNIST (~2M params).
    - Input:  (B, 1, 28, 28)
    - Output: (B, 1) scalar energy
    - 2 stages: (1->64)->down->(64->128), skip dim = 192
    - MLP: 192 -> 1024 -> 1024 -> 1
    """
    def __init__(self):
        super().__init__()
        # Stage 1: (1 -> 64), 28x28
        self.doubleconv1 = DoubleConv(1, 64)
        self.down1 = Downsample(64)        # 28->14

        # Stage 2: (64 -> 128), 14x14
        self.doubleconv2 = DoubleConv(64, 128)
        self.down2 = Downsample(128)       # 14->7

        # skip dimension = 64 + 128 = 192
        # MLP: 192 -> 1024 -> 1024 -> 1
        self.fc0 = nn.Linear(192, 1024, bias=True)
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.out = nn.Linear(1024, 1, bias=True)

    def forward(self, x):
        s1 = self.doubleconv1(x)   # (B,64,28,28)
        x = self.down1(s1)         # (B,64,14,14)

        s2 = self.doubleconv2(x)   # (B,128,14,14)
        x = self.down2(s2)         # (B,128,7,7)

        # Global average pool skip features
        skip1 = F.adaptive_avg_pool2d(s1, (1, 1)).view(s1.size(0), -1)  # (B,64)
        skip2 = F.adaptive_avg_pool2d(s2, (1, 1)).view(s2.size(0), -1)  # (B,128)

        concat_skips = torch.cat([skip1, skip2], dim=1)  # (B,192)

        out = F.gelu(self.fc0(concat_skips))
        out = F.gelu(self.fc1(out))
        energy = self.out(out)  # (B,1)
        return energy


class EBCNNModelWrapper(nn.Module):
    """
    Wrapper around Network_2M_MNIST28x28 that provides the same interface as
    EBViTModelWrapper: potential(x, t), velocity(x, t), forward(t, x).

    This allows the CNN model to be used interchangeably with the UNet+ViT
    model in the training and sampling scripts.
    """

    def __init__(self, output_scale=100.0, energy_clamp=None):
        super().__init__()
        self.cnn = Network_2M_MNIST28x28()
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

    def potential(self, x, t):
        """
        Computes scalar potential V(x,t) => shape (B,).
        Time is ignored (CNN is not time-conditioned).
        """
        V = self.cnn(x).view(-1)  # (B,)
        V = V * self.output_scale
        if self.energy_clamp is not None and self.energy_clamp > 0:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """
        Computes -∂V/∂x => shape (B, C, H, W).
        Time is ignored.
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
        Forward pass: same signature as EBViTModelWrapper.
        If return_potential=True, returns V(x,t); otherwise returns velocity.
        """
        if return_potential:
            return self.potential(x, t)
        else:
            return self.velocity(x, t)


##############################################################################
# Factory function: build model from FLAGS
##############################################################################
def build_model(FLAGS):
    """
    Instantiate the model based on FLAGS.model_type.
      - "unet_vit": UNet + ViT head (paper's architecture)
      - "cnn":      Simple CNN + MLP (authors' earlier code)
    """
    from experiments.mnist_new import config

    energy_clamp = FLAGS.energy_clamp if FLAGS.energy_clamp > 0 else None

    if FLAGS.model_type == "unet_vit":
        ch_mult = config.parse_channel_mult(FLAGS)
        return EBViTModelWrapper(
            dim=(1, 28, 28),
            num_channels=FLAGS.num_channels,
            num_res_blocks=FLAGS.num_res_blocks,
            channel_mult=ch_mult,
            attention_resolutions=FLAGS.attention_resolutions,
            num_heads=FLAGS.num_heads,
            num_head_channels=FLAGS.num_head_channels,
            dropout=FLAGS.dropout,
            output_scale=FLAGS.output_scale,
            energy_clamp=energy_clamp,
            patch_size=7,
            embed_dim=FLAGS.embed_dim,
            transformer_nheads=FLAGS.transformer_nheads,
            transformer_nlayers=FLAGS.transformer_nlayers,
        )
    elif FLAGS.model_type == "cnn":
        return EBCNNModelWrapper(
            output_scale=FLAGS.output_scale,
            energy_clamp=energy_clamp,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{FLAGS.model_type}'. "
            f"Choose 'unet_vit' or 'cnn'."
        )
