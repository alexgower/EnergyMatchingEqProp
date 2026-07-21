# File: network_pcn_unet.py
"""
PCN-UNet: the stage-0 UNet+ViT energy model as a DAG Predictive Coding Network.

Adopts the torchcfm UNet blocks + the ViT head of EBViTModelWrapper byte-for-byte
as PCN prediction functions f_k, one PC state node per UNet block (21 nodes for
the MNIST config), with skip connections expressed as extra DAG parents:

    nodes 0..E-1 : input_blocks[i]      parents [i-1]          (node 0: [x])
    node  E      : middle_block         parents [E-1]
    nodes E+1..  : output_blocks[j]     parents [prev, skip]   skip = E-1-j
    node  -2     : out (GN→SiLU→Conv)   parents [prev]         → (B,1,28,28)
    node  -1     : ViT head → scalar o  parents [prev]         → (B,1)

Energy is the inherited PCN energy  E_int = Σ_k ½‖f_k(parents_k) − h_k‖² + γ·o.
Fan-out (an encoder state feeding both its successor and a decoder concat) needs
no special handling: relaxation takes autograd grads of the total energy, which
sums both residual contributions in ∂E/∂h_k automatically.

Because the modules are adopted unchanged (same parameter tensors, same
structure), the trained stage-0 FFN checkpoint loads directly via
load_from_ebvit(), and at feedforward equilibrium (all e_k = 0) the PCN scan
reproduces the FFN forward pass exactly — the basis of the fidelity diagnostic.

The (constant, t=0.5) time embedding is kept as a real module so its trained
weights load from the checkpoint; each ResBlock receives emb = time_embed(temb)
recomputed per prediction (two tiny Linears on a (1, C) tensor — negligible).
"""

import torch
import torch.nn as nn

from torchcfm.models.unet.nn import timestep_embedding

from network_pcn import PCNEnergyModelBase
from network_unet import EBViTModelWrapper


class ViTHead(nn.Module):
    """Node f_L: UNet output image → raw scalar o (B, 1).

    Wraps the adopted PatchEmbed / TransformerEncoder / final Linear of
    EBViTModelWrapper. Returns the RAW scalar (no output_scale, no clamp) —
    the same convention as the CNN PCN's final Linear node, so the wrapper's
    V = output_scale·α·E_int matches the FFN's V = output_scale·F(x) at small γ.
    """

    def __init__(self, patch_embed, transformer_encoder, final_linear):
        super().__init__()
        self.patch_embed = patch_embed
        # NB the WHOLE stack, not one layer: nn.TransformerEncoder clones its
        # prototype layer transformer_nlayers times (2 for MNIST, 8 for CIFAR),
        # each layer = multi-head self-attention + FFN(E->4E->E) + LN/residuals.
        # It is all inside THIS prediction function, so attention lives on a
        # single graph edge rather than being exposed as extra PC state nodes.
        self.transformer_encoder = transformer_encoder
        self.final_linear = final_linear

    def forward(self, x):
        # x = the UNet output IMAGE (B, 1, 28, 28) — i.e. the previous node's state.
        # Conv(k=4,s=4) cuts it into non-overlapping 4x4 patches (28/4=7 per side
        # -> N=49 tokens) and embeds each to width E, + learnable pos-embed.
        tokens = self.patch_embed(x)               # (B, N, E)   e.g. (B, 49, 128)
        # Patches attend to each other; shape unchanged (a refinement, not a reduction).
        encoded = self.transformer_encoder(tokens)  # (B, N, E)
        # Average over the TOKEN axis -> one summary vector per image.
        pooled = encoded.mean(dim=1)                # (B, E)
        # Down to the scalar. RAW: output_scale/clamp are applied by the wrapper's
        # potential(); this o is what the energy's linear clamp term gamma*o acts on.
        return self.final_linear(pooled)            # (B, 1) raw scalar o


class PCNUNetEnergyModel(PCNEnergyModelBase):
    """DAG PCN over the adopted stage-0 UNet+ViT blocks.

    Subclasses PCNEnergyModel for all dynamics (relaxation, spring/EP phases,
    energy, error parameterization — everything routes through predict()),
    but builds its own layers/parents and overrides predict() to (a) concat
    multi-parent inputs (decoder skips) and (b) inject the constant time
    embedding into TimestepEmbedSequential blocks.
    """

    def __init__(self, num_channels=32, num_res_blocks=2, channel_mult=(1, 2, 2),
                 attention_resolutions="14", num_heads=2, num_head_channels=32,
                 embed_dim=128, transformer_nheads=2, transformer_nlayers=2,
                 patch_size=4, dim=(1, 28, 28)):
        # Skip the engine base's setup (there is none) and build UNet blocks.
        nn.Module.__init__(self)

        # Build the exact stage-0 architecture, then adopt its submodules.
        # dropout is hard-zeroed: relaxation/EP evaluate the energy many times
        # per step and free/nudged phases must see the SAME function — active
        # dropout would make the energy stochastic between evaluations.
        # (Dropout has no parameters, so checkpoint compatibility is unaffected;
        # note this deviates from the stage-0 FFN recipe, which trained at 0.1.)
        ebvit = EBViTModelWrapper(
            dim=dim, num_channels=num_channels, num_res_blocks=num_res_blocks,
            channel_mult=list(channel_mult),
            attention_resolutions=attention_resolutions,
            num_heads=num_heads, num_head_channels=num_head_channels,
            dropout=0.0,
            patch_size=patch_size, embed_dim=embed_dim,
            transformer_nheads=transformer_nheads,
            transformer_nlayers=transformer_nlayers,
            output_scale=1.0, energy_clamp=None,
        )

        # ---- adopt modules as prediction functions (topological order) ----
        E = len(ebvit.input_blocks)
        D = len(ebvit.output_blocks)
        layers = list(ebvit.input_blocks) + [ebvit.middle_block] \
            + list(ebvit.output_blocks) + [ebvit.out]
        vit_head = ViTHead(ebvit.patch_embed, ebvit.transformer_encoder,
                           ebvit.final_linear)
        layers.append(vit_head)
        self.layers = nn.ModuleList(layers)
        self.L = len(self.layers)          # E + 1 + D + 2 (21 for MNIST config)

        # Which nodes are TimestepEmbedSequential (take (h, emb))
        self._needs_emb = [True] * (E + 1 + D) + [False, False]

        # ---- DAG topology (mirrors UNetModel.forward's skip stack) ----
        parents = [[k - 1] for k in range(E)]     # encoder chain; node 0 → [-1]=x
        stack = list(range(E))                    # hs: every encoder output pushed
        parents.append([E - 1])                   # middle
        node = E
        for _ in range(D):                        # decoder: cat([h, hs.pop()])
            parents.append([node, stack.pop()])   # [prev, skip] — concat order!
            node += 1
        parents.append([node])                    # out projection
        parents.append([node + 1])                # ViT head
        self.parents = parents
        assert not stack, "skip stack not fully consumed"

        self._mid_idx = E
        self._out_idx = E + 1 + D
        self._vit_idx = E + 1 + D + 1
        # Which top-level state_dict prefixes of the SOURCE wrapper belong to the
        # head. The FFN names params flat (patch_embed.*, transformer_encoder.*,
        # final_linear.*) while we store every block in one ModuleList (layers.k.*),
        # so _remap_ebvit_key needs to know which keys route to the FINAL node
        # (layers.<_vit_idx>.*) rather than to a UNet block. Kept as an attribute
        # so a variant with a different head (e.g. the avgpool-MLP head, whose
        # params live under "pool."/"mlp.") can just reassign this tuple and
        # inherit the remapping logic unchanged.
        self._head_prefixes = ("patch_embed.", "transformer_encoder.", "final_linear.")

        # ---- constant time embedding (t = 0.5, as in dummy_time) ----
        # The torchcfm UNet is a DIFFUSION backbone: every ResBlock expects
        # (h, emb) with emb a timestep embedding. Energy Matching's potential is
        # time-INdependent, which the FFN achieves by always feeding t=0.5 (a
        # constant, so it acts as a fixed bias). We adopt the same trick, and must
        # use the same constant to reproduce the FFN exactly.
        #   emb = time_embed( timestep_embedding(t, model_channels) )
        # time_embed is ADOPTED (a small MLP with learnable weights), so the FFN
        # checkpoint's "time_embed.*" keys load straight in (hence _remap_ebvit_key
        # passes them through unchanged).
        self.time_embed = ebvit.time_embed
        self.model_channels = ebvit.model_channels
        # Precompute only the SINUSOID (the MLP's input): t is fixed at 0.5 so it
        # never changes. We deliberately do NOT cache the final emb — time_embed's
        # weights are trained, so a cached output would go stale each step.
        # persistent=False keeps this buffer OUT of state_dict: it is deterministic
        # (rebuilt every __init__), and if it were persistent, loading an EBViT
        # checkpoint with strict=True would fail on a missing "_temb_in" key.
        self.register_buffer(
            "_temb_in",
            timestep_embedding(torch.full((1,), 0.5), self.model_channels),
            persistent=False,
        )

        # Vestigial: mirrors the metadata PCNCNNEnergyModel records about how the
        # VGG5 chain was built (pooling / activation choice). Meaningless here —
        # the UNet has its own down/up-sampling and hardcoded SiLU. Nothing in the
        # engine currently reads these; kept only so any code expecting a PCN model
        # to expose them does not break. Safe to delete from both classes.
        self._pool_type = "n/a"
        self._activation = "n/a"

    def predict(self, k, x, hiddens):
        """Topology-aware f_k: gather parents (channel-concat for decoder
        skips, in [prev, skip] order matching unet.py's cat([h, hs.pop()])),
        inject the constant time embedding where the block expects one."""
        vals = [x if p < 0 else hiddens[p] for p in self.parents[k]]
        h_in = vals[0] if len(vals) == 1 else torch.cat(vals, dim=1)
        if self._needs_emb[k]:
            emb = self.time_embed(self._temb_in)  # (1, 4·C), broadcasts over batch
            return self.layers[k](h_in, emb)
        return self.layers[k](h_in)

    # ------------------------------------------------------------------
    # Stage-0 checkpoint loading
    # ------------------------------------------------------------------
    def load_from_ebvit(self, state_dict, strict=True):
        """Load an EBViTModelWrapper state_dict (the stage-0 FFN checkpoint,
        'module.' prefix already stripped) by remapping its top-level keys onto
        the adopted-module layout. Returns (missing, unexpected)."""
        remapped = {}
        skipped = []
        for key, val in state_dict.items():
            new = self._remap_ebvit_key(key)
            if new is None:
                skipped.append(key)
            else:
                remapped[new] = val
        if skipped and strict:
            raise KeyError(f"unmapped EBViT keys: {skipped[:8]}"
                           f"{'...' if len(skipped) > 8 else ''}")
        result = self.load_state_dict(remapped, strict=strict)
        return result

    def _remap_ebvit_key(self, key):
        if key.startswith("time_embed."):
            return key
        if key.startswith("input_blocks."):
            idx, rest = key[len("input_blocks."):].split(".", 1)
            return f"layers.{int(idx)}.{rest}"
        if key.startswith("middle_block."):
            return f"layers.{self._mid_idx}." + key[len("middle_block."):]
        if key.startswith("output_blocks."):
            idx, rest = key[len("output_blocks."):].split(".", 1)
            return f"layers.{self._mid_idx + 1 + int(idx)}.{rest}"
        if key.startswith("out."):
            return f"layers.{self._out_idx}." + key[len("out."):]
        for prefix in self._head_prefixes:
            if key.startswith(prefix):
                return f"layers.{self._vit_idx}.{key}"
        return None
