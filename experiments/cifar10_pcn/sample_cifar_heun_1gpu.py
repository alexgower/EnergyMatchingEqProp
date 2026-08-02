###############################################################################
# File: sample_cifar_heun_1gpu.py
#
# Purpose:
#   Generate a batch of CIFAR-10-sized images with a trained energy-based
#   model using Euler–Heun SDE integration, then save samples on a single grid.
#
#   Restored from experiments/cifar10/sample_cifar_heun_1gpu.py, with the
#   model construction upgraded to the cifar10_pcn MODEL_REGISTRY so any
#   --model_type checkpoint (FFN or PCN) can be sampled. Pass the SAME
#   architecture flags used at training time. For richer sweeps (Euler/Heun
#   ODE, convergence mode, velocity modes, Langevin τs) prefer
#   generate_from_checkpoint.py; this script is the paper-parity Heun SDE
#   sampler (Figure 6 style).
#
# Example:
#   python experiments/cifar10_pcn/sample_cifar_heun_1gpu.py \
#       --resume_ckpt=path/to/checkpoint_147000.pt \
#       --model_type=ffn_unet_vit \
#       --batch_size=128 \
#       --t_end=3.25 \
#       --dt_gibbs=0.01 \
#       --use_ema=True \
#       --epsilon_max=0.01 \
#       --time_cutoff=1.0
###############################################################################
import os, sys, math
from datetime import datetime
import torch
from torchvision.utils import save_image, make_grid

# ───────────────────────────────────────── Flags ─────────────────────────────────────────
from absl import app, flags, logging

import config_multigpu as config
config.define_flags()                                # model-architecture flags
FLAGS = flags.FLAGS

flags.DEFINE_float("t_end", 3.25,
                   "Final SDE time (t_start is fixed to 0).")
flags.DEFINE_bool("use_ema", True,
                  "If True, load EMA weights; else raw weights.")

# ───────────────────────────────────– Model & Utils ──────────────────────────────────────
from network_cnn import EBCNNModelWrapper
from network_unet import (EBViTModelWrapper, EBMLPModelWrapper,
                          EBRonnebergerConvUNetWrapper)
from network_pcn import PCNVelocityWrapper
from utils import plot_epsilon


def build_model(device):
    """Build model from FLAGS (same registry dispatch as the train script)."""
    img_shape = (3, 32, 32)

    paradigm, arch, pcn_topology = config.resolve_model_type(FLAGS.model_type)
    _clamp = FLAGS.energy_clamp if FLAGS.energy_clamp and FLAGS.energy_clamp > 0 else None
    if paradigm == "pcn":
        model = PCNVelocityWrapper(
            gamma=FLAGS.pcn_gamma,
            T_free=FLAGS.T_free,
            dt_relax=FLAGS.pcn_dt,
            async_mode=FLAGS.pcn_async,
            init_mode=FLAGS.pcn_init_mode,
            output_scale=FLAGS.output_scale,
            energy_clamp=_clamp,
            n_cg_steps=FLAGS.pcn_cg_steps,
            pool_type=FLAGS.pool_type,
            activation=FLAGS.activation,
            error_param=FLAGS.pcn_error_param,
            param_grad_mode=FLAGS.param_grad_mode,
            lambda_spring=FLAGS.lambda_spring,
            beta=FLAGS.beta,
            T_nudge=FLAGS.T_nudge,
            thirdphase=FLAGS.thirdphase,
            K_h=FLAGS.K_h,
            nudge_type=FLAGS.nudge_type,
            ep_x_grad_mode=FLAGS.ep_x_grad_mode,
            pcn_arch=pcn_topology,
            unet_kwargs=dict(
                dim=img_shape,
                num_channels=FLAGS.num_channels,
                num_res_blocks=FLAGS.num_res_blocks,
                channel_mult=config.parse_channel_mult(FLAGS),
                attention_resolutions=FLAGS.attention_resolutions,
                num_heads=FLAGS.num_heads,
                num_head_channels=FLAGS.num_head_channels,
                patch_size=FLAGS.patch_size,
                no_attention=FLAGS.unet_no_attention,
                no_norm=FLAGS.unet_no_norm,
                embed_dim=FLAGS.embed_dim,
                transformer_nheads=FLAGS.transformer_nheads,
                transformer_nlayers=FLAGS.transformer_nlayers,
            ) if arch == "unet_vit" else None,
        ).to(device)
    elif arch == "ronneberger_conv_unet":
        model = EBRonnebergerConvUNetWrapper(
            output_scale=FLAGS.output_scale,
            energy_clamp=_clamp,
            in_channels=img_shape[0],
            num_channels=FLAGS.num_channels,
            channel_mult=config.parse_channel_mult(FLAGS),
            pool_type=FLAGS.pool_type,
            use_norm=FLAGS.conv_unet_norm,
        ).to(device)
    elif arch == "unet_mlp":
        model = EBMLPModelWrapper(
            dim=img_shape,
            num_channels=FLAGS.num_channels,
            num_res_blocks=FLAGS.num_res_blocks,
            channel_mult=config.parse_channel_mult(FLAGS),
            attention_resolutions=FLAGS.attention_resolutions,
            num_heads=FLAGS.num_heads,
            num_head_channels=FLAGS.num_head_channels,
            dropout=FLAGS.dropout,
            no_attention=FLAGS.unet_no_attention,
            no_norm=FLAGS.unet_no_norm,
            output_scale=FLAGS.output_scale,
            energy_clamp=_clamp,
        ).to(device)
    elif arch in ("historical", "vgg5", "mlp"):
        model = EBCNNModelWrapper(
            output_scale=FLAGS.output_scale,
            energy_clamp=_clamp,
            version=arch,
            pool_type=FLAGS.pool_type,
            activation=FLAGS.activation,
        ).to(device)
    else:  # unet_vit — UNet + ViT head (paper architecture)
        model = EBViTModelWrapper(
            dim=img_shape,
            num_channels=FLAGS.num_channels,
            num_res_blocks=FLAGS.num_res_blocks,
            channel_mult=config.parse_channel_mult(FLAGS),
            attention_resolutions=FLAGS.attention_resolutions,
            num_heads=FLAGS.num_heads,
            num_head_channels=FLAGS.num_head_channels,
            dropout=FLAGS.dropout,
            output_scale=FLAGS.output_scale,
            energy_clamp=FLAGS.energy_clamp,
            patch_size=FLAGS.patch_size,
            no_attention=FLAGS.unet_no_attention,
            no_norm=FLAGS.unet_no_norm,
            embed_dim=FLAGS.embed_dim,
            transformer_nheads=FLAGS.transformer_nheads,
            transformer_nlayers=FLAGS.transformer_nlayers,
        ).to(device)

    return model, img_shape


# ------------------------ Euler–Heun SDE integrator (modified) ---------------------------
import torchsde
def solve_sde_heun(model, x, t_start, t_end, dt=0.01):
    """Integrate `x` from t_start to t_end with Stratonovich Euler–Heun."""
    if t_end <= t_start:
        return x
    orig_shape, B = x.shape, x.size(0)
    x_flat = x.view(B, -1)

    class _FlattenSDE(torchsde.SDEStratonovich):
        def __init__(self, net):
            super().__init__(noise_type="diagonal")
            self.net = net

        # drift
        def f(self, t, y):
            y_unflat = y.view(*orig_shape)
            return self.net(t.expand(B).to(y.device), y_unflat).view(B, -1)

        def g(self, t, y):
            # Diffusion
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
            scale = torch.sqrt(2.0 * e_tensor)
            return scale.expand_as(y)

    sde  = _FlattenSDE(model)
    ts   = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)
    with torch.no_grad():
        x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
    return x_sol[-1].view(*orig_shape).clamp(-1, 1)

# ────────────────────────────────────────── Main ─────────────────────────────────────────
def main(_):
    # 1) Device & directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(FLAGS.output_dir,
                            datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Samples will be saved in: {save_dir}")

    # 2) Build model
    model, img_shape = build_model(device)
    model.eval()

    # 3) Load checkpoint
    if not FLAGS.resume_ckpt or not os.path.isfile(FLAGS.resume_ckpt):
        raise FileNotFoundError("--resume_ckpt is missing or invalid.")
    ckpt = torch.load(FLAGS.resume_ckpt, map_location=device)
    key  = "ema_model" if FLAGS.use_ema else "net_model"
    state_dict = ckpt[key]
    # Strip 'module.' prefix if the checkpoint was saved from a DDP wrapper
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    logging.info(f"Loaded {key} from {FLAGS.resume_ckpt}")

    # 4) Noise → image
    x = torch.randn(FLAGS.batch_size, *img_shape, device=device)
    x = solve_sde_heun(model, x, 0.0, FLAGS.t_end, dt=FLAGS.dt_gibbs)  # in [-1,1]
    x_01 = (x + 1.0) / 2.0                                             # → [0,1]

    # 5) Single-grid save
    nrow = int(math.sqrt(FLAGS.batch_size))
    grid = make_grid(x_01, nrow=nrow, padding=2)
    grid_path = os.path.join(save_dir, "samples_grid.png")
    save_image(grid, grid_path)
    logging.info(f"Saved grid ({FLAGS.batch_size} images) to {grid_path}")

if __name__ == "__main__":
    app.run(main)
