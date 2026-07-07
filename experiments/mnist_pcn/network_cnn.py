# File: network_cnn.py
# CNN energy models for MNIST (1×28×28).
#
# Exposes the same interface as EBViTModelWrapper:
#   potential(x, t), velocity(x, t), forward(t, x)
#
# Architectures:
#   - cnn_v2:  3-stage CNN with flattened spatial features → MLP → scalar V(x) (~2.1M params)
#   - vgg5:    VGG5 from Scellier et al., adapted for energy output (~6.2M params)
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
    """Two consecutive conv(3×3) + SiLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        return x


class Downsample(nn.Module):
    """Downsample by 2× using stride-2 conv."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=True
        )

    def forward(self, x):
        return self.conv(x)


##############################################################################
# CNN v2: 3 stages + flattened spatial features (~2.1M params)
##############################################################################

class Network_2M_v2_MNIST28x28(nn.Module):
    """
    CNN energy model for MNIST (~2.1M params).
    Keeps spatial features (no global avg pool) — important for sharp velocity fields.
    - Input:  (B, 1, 28, 28)
    - Output: (B, 1) scalar energy
    - 3 stages: (1->64)->down->(64->128)->down->(128->128) at 7x7
    - Flatten 128*7*7=6272 -> 192 -> 1
    """
    def __init__(self):
        super().__init__()
        # Stage 1: (1 -> 64), 28x28
        self.doubleconv1 = DoubleConv(1, 64)
        self.down1 = Downsample(64)        # 28->14

        # Stage 2: (64 -> 128), 14x14
        self.doubleconv2 = DoubleConv(64, 128)
        self.down2 = Downsample(128)       # 14->7

        # Stage 3: (128 -> 128), 7x7 (new!)
        self.doubleconv3 = DoubleConv(128, 128)

        # Flatten spatial features: 128 * 7 * 7 = 6272
        # MLP: 6272 -> 192 -> 1
        self.fc0 = nn.Linear(6272, 192, bias=True)
        self.out = nn.Linear(192, 1, bias=True)

    def forward(self, x):
        x = self.doubleconv1(x)    # (B,64,28,28)
        x = self.down1(x)          # (B,64,14,14)

        x = self.doubleconv2(x)    # (B,128,14,14)
        x = self.down2(x)          # (B,128,7,7)

        x = self.doubleconv3(x)    # (B,128,7,7)

        # Flatten — keep all spatial info
        x = x.view(x.size(0), -1)  # (B, 6272)

        out = F.silu(self.fc0(x))
        energy = self.out(out)      # (B,1)
        return energy


##############################################################################
# VGG5: Scellier et al. architecture, adapted for energy output (~6.2M params)
##############################################################################

class VGG5Energy_MNIST28x28(nn.Module):
    """
    VGG5 from Scellier et al. (Table 5) adapted for scalar energy output.
    Modifications from the original:
      - 1 input channel (MNIST)
      - SiLU activations (smoother ∂V/∂x than ReLU) # TODO ablate against ReLU
      - Scalar output (energy) instead of class logits
      - Configurable downsampling: maxpool / avgpool / stride_conv

    Architecture: 5 conv layers (3×3, padding=1) with 4 downsampling ops.
    Spatial progression: 28 → 28 → [down] 14 → 14 → [down] 7 → [down] 3 → [down] 1
    ~6.2M params (slightly more with stride_conv due to learned downsample kernels).

    pool_type options:
      - "maxpool":    MaxPool 2×2 — matches Scellier for EP comparability.
                      Sparse gradients (1/4 pixels per window).
      - "avgpool":    AvgPool 2×2 — dense gradients, all pixels contribute equally.
      - "stride_conv": Learned stride-2 conv (4×4 kernel) — dense gradients,
                      learnable weights. Best for generation.
    """
    def __init__(self, pool_type="stride_conv"):
        super().__init__()

        def down(channels):
            """Return the downsampling module for the given channel count."""
            if pool_type == "maxpool":
                return nn.MaxPool2d(2, 2)
            elif pool_type == "avgpool":
                return nn.AvgPool2d(2, 2)
            elif pool_type == "stride_conv":
                return nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
            else:
                raise ValueError(f"Unknown pool_type: {pool_type}. Use 'maxpool', 'avgpool', 'stride_conv'.")

        self.features = nn.Sequential(
            # Block 1: 28×28
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            down(256),                # → 14×14

            # Block 2: 14×14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            down(512),                # → 7×7

            # Block 3: 7×7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            down(512),                # → 3×3

            # Block 4: 3×3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            down(512),                # → 1×1
        )
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (B, 512)
        return self.head(x)          # (B, 1)


##############################################################################
# Wrapper with standard EBM interface
##############################################################################

class EBCNNModelWrapper(nn.Module):
    """
    Wrapper around CNN architectures providing the same interface as
    EBViTModelWrapper: potential(x, t), velocity(x, t), forward(t, x).

    This allows any CNN to be used interchangeably with the UNet+ViT in
    the training and sampling scripts.

    Supported versions:
      - "historical": Network_2M_v2_MNIST28x28 (~2.1M params)
      - "vgg5":       VGG5Energy_MNIST28x28 (~6.2M params)
    """

    def __init__(self, output_scale=100.0, energy_clamp=None, version="historical", pool_type="stride_conv"):
        super().__init__()
        if version == "vgg5":
            self.cnn = VGG5Energy_MNIST28x28(pool_type=pool_type)
        elif version == "historical":
            self.cnn = Network_2M_v2_MNIST28x28()
        else:
            raise ValueError(f"Unknown CNN version: {version}. Use 'historical' or 'vgg5'.")
        self.output_scale = output_scale
        self.energy_clamp = energy_clamp

    def potential(self, x, t):
        """Computes scalar potential V(x) => shape (B,). Time is ignored."""
        V = self.cnn(x).view(-1)  # (B,)
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
