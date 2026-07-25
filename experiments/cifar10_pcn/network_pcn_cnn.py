# File: network_pcn_cnn.py
# CNN (VGG5-decomposed, linear-chain) Predictive Coding architecture for the engine.
# Mirrors network_cnn.py on the FFN side: this is the CNN-family architecture,
# here expressed as PC prediction functions. All dynamics live in the engine
# base class PCNEnergyModelBase (network_pcn.py); this file only builds layers.

import torch.nn as nn

from network_pcn import PCNEnergyModelBase


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


class PCNCNNEnergyModel(PCNEnergyModelBase):
    """
    VGG5 decomposed into L=6 PCN prediction layers for energy-based dynamics.

    Energy function:
        E_int(x, h, o) = Σ_{k=1}^{L} ½||f_k(h_{k-1}) - h_k||² + γ·o

    where h_0 = x (visible input), h_L = o (scalar output), and f_k is the
    k-th prediction layer (conv + act + optional pool).

    Layer decomposition (matching network_cnn.py VGG5Energy ordering):
        f_1: Conv(3, 128, 3, pad=1) + SiLU              → h_1: (B, 128, 32, 32)
        f_2: Conv(128, 256, 3, pad=1) + SiLU + Pool(2)  → h_2: (B, 256, 16, 16)
        f_3: Conv(256, 512, 3, pad=1) + SiLU + Pool(2)  → h_3: (B, 512, 8, 8)
        f_4: Conv(512, 512, 3, pad=1) + SiLU + Pool(2)  → h_4: (B, 512, 4, 4)
        f_5: Conv(512, 512, 3, pad=1) + SiLU + Pool(2)  → h_5: (B, 512, 2, 2)
        f_6: Linear(2048, 1)  [no activation]            → o:   (B, 1)
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
            # Layer 1: (B,3,32,32) → (B,128,32,32)
            PCNLayer(
                nn.Conv2d(3, 128, kernel_size=3, padding=1),
                activation_fn=act_cls(),
            ),
            # Layer 2: (B,128,32,32) → (B,256,16,16)
            PCNLayer(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                activation_fn=act_cls(),
                pool=make_pool(256),
            ),
            # Layer 3: (B,256,16,16) → (B,512,8,8)
            PCNLayer(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                activation_fn=act_cls(),
                pool=make_pool(512),
            ),
            # Layer 4: (B,512,8,8) → (B,512,4,4)
            PCNLayer(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                activation_fn=act_cls(),
                pool=make_pool(512),
            ),
            # Layer 5: (B,512,4,4) → (B,512,2,2)
            PCNLayer(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                activation_fn=act_cls(),
                pool=make_pool(512),
            ),
            # Layer 6 (output): (B,512,2,2) → (B,1) scalar
            # Flatten is handled inside forward (base _prepare_input flattens
            # the parent state before the final Linear); no activation.
            PCNLayer(
                nn.Linear(512 * 2 * 2, 1),
                activation_fn=None,
            ),
        ])
        self.L = len(self.layers)
        self._pool_type = pool_type
        self._activation = activation
        # Topology: parents[k] = list of node indices whose states feed f_k
        # (-1 denotes the input x). The chain is [[-1], [0], [1], ...]; DAG
        # subclasses (e.g. the PCN-UNet with skip connections) override this.
        self.parents = [[k - 1] for k in range(self.L)]
