# File: network_cnn.py
# Simple CNN energy model for MNIST (1×28×28).
#
# Exposes the same interface as EBViTModelWrapper:
#   potential(x, t), velocity(x, t), forward(t, x)
#
# Architecture: 2-stage CNN with skip connections → MLP → scalar V(x).
# Uses SiLU activations per the paper's recommendation (Section D):
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
# CNN Energy Model (~1.8M params)
##############################################################################

class Network_2M_MNIST28x28(nn.Module):
    """
    Simple CNN energy model for MNIST.
    - Input:  (B, 1, 28, 28)
    - Output: (B, 1) scalar energy
    - 2 stages: (1->64)->down->(64->128)->down, skip dim = 192
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

        out = F.silu(self.fc0(concat_skips))
        out = F.silu(self.fc1(out))
        energy = self.out(out)  # (B,1)
        return energy


##############################################################################
# CNN v2: 3 stages + flattened spatial features (~2.1M params)
##############################################################################

class Network_2M_v2_MNIST28x28(nn.Module):
    """
    Improved CNN energy model for MNIST (~2.1M params).
    Key difference from v1: keeps spatial features (no global avg pool).
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
# Wrapper with standard EBM interface
##############################################################################

class EBCNNModelWrapper(nn.Module):
    """
    Wrapper around Network_2M_MNIST28x28 providing the same interface as
    EBViTModelWrapper: potential(x, t), velocity(x, t), forward(t, x).

    This allows the CNN to be used interchangeably with the UNet+ViT in
    the training and sampling scripts.
    """

    def __init__(self, output_scale=100.0, energy_clamp=None, version="v1"):
        super().__init__()
        if version == "v2":
            self.cnn = Network_2M_v2_MNIST28x28()
        else:
            self.cnn = Network_2M_MNIST28x28()
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
