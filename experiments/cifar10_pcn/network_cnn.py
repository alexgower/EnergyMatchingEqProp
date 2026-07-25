# File: network_cnn.py
# CNN energy models for CIFAR-10 (3×32×32).
#
# Exposes the same interface as EBViTModelWrapper:
#   potential(x, t), velocity(x, t), forward(t, x)
#
# Architectures:
#   - historical:  3-stage CNN with flattened spatial features → MLP → scalar V(x) (~2.5M params)
#   - vgg5:        VGG5 from Scellier et al., adapted for energy output (~6.2M params)
#   - mlp:         Pure MLP baseline, no convolutions (~4.7M params default)
#
# Uses SiLU activations per the Energy Matching paper (Section D):
#   "We recommend using SiLU activation functions wherever possible,
#    as they smooth out the energy landscape and improve the numerical
#    stability of the ∇_x V(x) computation."

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_clamp(x, clamp_val):
    """Tanh-based clamp: output in [-clamp_val, clamp_val]."""
    return clamp_val * torch.tanh(x / clamp_val)


##############################################################################
# CNN building blocks
##############################################################################

class DoubleConv(nn.Module):
    """Two consecutive conv(3×3) + activation."""
    def __init__(self, in_channels, out_channels, activation="silu"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True # Classic padding = (kernel_size - 1)/2 to preserve 32x32 size
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.act = nn.SiLU() if activation == "silu" else nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class Downsample(nn.Module):
    """Downsample by 2× using stride-2 conv."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=True # Kernel size 3 might be ore conventional now, but this halves spatial dimension is the point
        )

    def forward(self, x):
        return self.conv(x)


##############################################################################
# Historical: 3 stages + flattened spatial features (~2.5M params)
##############################################################################

class HistoricalCNNEnergy(nn.Module):
    """
    CNN energy model for CIFAR-10 (~2.5M params).
    Keeps spatial features (no global avg pool) — important for sharp velocity fields.
    - Input:  (B, 3, 32, 32)
    - Output: (B, 1) scalar energy
    - 3 stages: (3->64)->down->(64->128)->down->(128->128) at 8x8
    - Flatten 128*8*8=8192 -> 192 -> 1
    """
    def __init__(self, activation="silu"):
        super().__init__()
        self.act = nn.SiLU() if activation == "silu" else nn.ReLU()

        # Stage 1: (3 -> 64), 32x32
        self.doubleconv1 = DoubleConv(3, 64, activation=activation)
        self.down1 = Downsample(64)        # 32->16

        # Stage 2: (64 -> 128), 16x16
        self.doubleconv2 = DoubleConv(64, 128, activation=activation)
        self.down2 = Downsample(128)       # 16->8

        # Stage 3: (128 -> 128), 8x8
        self.doubleconv3 = DoubleConv(128, 128, activation=activation)

        # Flatten spatial features: 128 * 8 * 8 = 8192
        # MLP: 8192 -> 192 -> 1
        self.fc0 = nn.Linear(8192, 192, bias=True)
        self.out = nn.Linear(192, 1, bias=True)

    def forward(self, x):
        x = self.doubleconv1(x)    # (B,64,32,32)
        x = self.down1(x)          # (B,64,16,16)

        x = self.doubleconv2(x)    # (B,128,16,16)
        x = self.down2(x)          # (B,128,8,8)

        x = self.doubleconv3(x)    # (B,128,8,8)

        # Flatten — keep all spatial info
        x = x.view(x.size(0), -1)  # (B, 8192)

        out = self.act(self.fc0(x))
        energy = self.out(out)      # (B,1)
        return energy


##############################################################################
# VGG5: Scellier et al. architecture, adapted for energy output (~6.2M params)
##############################################################################

class VGG5Energy(nn.Module):
    """
    VGG5 from Scellier et al. (Table 5) adapted for scalar energy output.
    Modifications from the original:
      - 3 input channels (CIFAR-10)
      - SiLU activations (smoother ∂V/∂x than ReLU)
      - Scalar output (energy) instead of class logits
      - Configurable downsampling: maxpool / avgpool / stride_conv

    Architecture: 5 conv layers (3×3, padding=1) with 4 downsampling ops.
    Spatial progression: 32 → 32 → [down] 16 → 16 → [down] 8 → [down] 4 → [down] 2

    Parameter counts vary significantly by pool_type:
      - maxpool / avgpool:  ~6.2M params (no learnable downsampling)
      - stride_conv:       ~19.8M params (4×4 learned kernels at 256–512 ch)

    pool_type options:
      - "maxpool":    MaxPool 2×2 — matches Scellier for EP comparability.
                      Sparse gradients (1/4 pixels per window). Bad for generation.
      - "avgpool":    AvgPool 2×2 — dense uniform gradients, all pixels contribute.
                      Best for generation (smooth ∂V/∂x). Recommended default.
      - "stride_conv": Learned stride-2 conv (4×4 kernel) — dense gradients,
                      learnable weights, but ~3× the params of avgpool/maxpool.
    """
    def __init__(self, pool_type="avgpool", activation="silu"):
        super().__init__()

        def act():
            """Return the activation module."""
            if activation == "silu":
                return nn.SiLU()
            elif activation == "relu":
                return nn.ReLU()
            else:
                raise ValueError(f"Unknown activation: {activation}. Use 'silu' or 'relu'.")

        def down(channels):
            """Return the downsampling module for the given channel count."""
            if pool_type == "maxpool":
                return nn.MaxPool2d(2, 2) # i.e. kernel_size=2, stride=2, i.e. non-overlapping patches to half spatial dimension
            elif pool_type == "avgpool":
                return nn.AvgPool2d(2, 2)
            elif pool_type == "stride_conv":
                return nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
            else:
                raise ValueError(f"Unknown pool_type: {pool_type}. Use 'maxpool', 'avgpool', 'stride_conv'.")

        self.features = nn.Sequential(
            # Block 1: 32×32
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            act(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            act(),
            down(256),                # → 16×16

            # Block 2: 16×16
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            act(),
            down(512),                # → 8×8

            # Block 3: 8×8
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            act(),
            down(512),                # → 4×4

            # Block 4: 4×4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            act(),
            down(512),                # → 2×2
        )
        self.head = nn.Linear(512 * 2 * 2, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B, 2048)
        return self.head(x)          # (B, 1)


##############################################################################
# MLP: Pure feedforward baseline, no convolutions (~4.7M params default)
##############################################################################

class MLPEnergy(nn.Module):
    """
    Pure MLP energy model for CIFAR-10 (no convolutions).
    Flattens 3×32×32 = 3072 inputs and maps through hidden layers to scalar.

    Default hidden sizes [1024, 1024, 512] give ~4.7M params. SiLU activations
    for smooth ∂V/∂x gradients.

    This serves as a "how bad can it get without spatial inductive bias?"
    baseline for the generation quality ablation.
    """
    def __init__(self, hidden_sizes=None, activation="silu"):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [1024, 1024, 512]

        act_cls = nn.SiLU if activation == "silu" else nn.ReLU


        layers = []
        in_dim = 3072  # 3 * 32 * 32
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 3072)
        return self.net(x)           # (B, 1)


##############################################################################
# Wrapper with standard EBM interface
##############################################################################

class EBCNNModelWrapper(nn.Module):
    """
    Wrapper around CNN/MLP architectures providing the same interface as
    EBViTModelWrapper: potential(x, t), velocity(x, t), forward(t, x).

    This allows any architecture to be used interchangeably with the UNet+ViT
    in the training and sampling scripts.

    Supported versions:
      - "historical": HistoricalCNNEnergy (~2.5M params)
      - "vgg5":       VGG5Energy (~6.2M params)
      - "mlp":        MLPEnergy (~4.7M params)
    """

    def __init__(self, output_scale=1000.0, energy_clamp=None, version="historical", pool_type="avgpool", activation="silu"):
        super().__init__()
        if version == "vgg5":
            self.net = VGG5Energy(pool_type=pool_type, activation=activation)
        elif version == "historical":
            self.net = HistoricalCNNEnergy(activation=activation)
        elif version == "mlp":
            self.net = MLPEnergy(activation=activation)
        else:
            raise ValueError(f"Unknown CNN version: {version}. Use 'historical', 'vgg5', or 'mlp'.")

        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

    def potential(self, x, t):
        """Computes scalar potential V(x) => shape (B,). Time is ignored."""
        V = self.net(x).view(-1)  # (B,)
        V = V * self.output_scale
        if self.energy_clamp is not None and self.energy_clamp > 0:
            V = soft_clamp(V, self.energy_clamp)
        return V

    def velocity(self, x, t):
        """Computes -∂V/∂x => shape (B, C, H, W). Time is ignored."""
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
        """Same signature as EBViTModelWrapper."""
        if return_potential:
            return self.potential(x, t)
        else:
            return self.velocity(x, t)
