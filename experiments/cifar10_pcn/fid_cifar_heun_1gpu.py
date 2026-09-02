#######################################################################
# File: fid_cifar_heun_1gpu.py
#
# Description:
#   Computes the Frechet Inception Distance (FID) for a trained
#   energy-based model on CIFAR-10 using a single GPU. It generates
#   samples via an SDE solver (Heun's method) and compares them
#   against the real training data.
#
#   Restored from experiments/cifar10/fid_cifar_heun_1gpu.py, with the
#   model construction upgraded to the cifar10_pcn MODEL_REGISTRY so any
#   --model_type checkpoint (FFN or PCN) can be evaluated. Pass the SAME
#   architecture flags used at training time. NB: PCN models evaluate one
#   full relaxation per drift evaluation, so 50k samples is slow — reduce
#   --fid_n_samples for quick PCN checks.
#
# Usage example:
#   python fid_cifar_heun_1gpu.py \
#       --resume_ckpt=/path/to/checkpoint.pt \
#       --model_type=ffn_unet_vit \
#       --batch_size=128 \
#       --dt_gibbs=0.01 \
#       --use_ema=True
#######################################################################

import os
import sys
import torch

# absl flags
from absl import app, flags, logging

# Single-GPU config
import config_multigpu as config
config.define_flags()
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_ema", True,
                  "If True, load the EMA model from the checkpoint (default True).")
flags.DEFINE_bool("ffn_checkpoint_into_pcn", False,
                  "If True, --resume_ckpt is a stage-0 FFN (EBViT) checkpoint to "
                  "be remapped into a PCN model via load_from_ebvit (requires a "
                  "pcn --model_type): train-with-backprop, infer-on-PCN eval.")
flags.DEFINE_integer("fid_n_samples", 50000,
                     "Number of fake samples to generate for FID (paper: 50000).")
flags.DEFINE_string("fid_times", "1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0",
                    "Comma-separated sampling times τs at which FID is computed "
                    "(one FID accumulator each; the trajectory is integrated "
                    "incrementally through them).")
flags.DEFINE_string("real_dataset", "cifar10",
                    "Real image set for the FID statistics: 'cifar10' (default, "
                    "torchvision train split) or 'imagenet32' (the official "
                    "downsampled-ImageNet train batches via IMAGENET32_PATH, as in "
                    "the Energy Matching ImageNet32 protocol).")
flags.DEFINE_integer("fid_seed", -1,
                     "Sampling seed for the fake-sample noise. Single-GPU here; "
                     "the multi-GPU runner offsets it per rank (fid_seed*1000+rank). "
                     "-1 = unseeded (legacy). Used for sampling-seed error bars.")
flags.DEFINE_string("cache_real_features", "",
                    "Path to cache the real-set Inception feature statistics "
                    "(e.g. ./data/fid_real_features_cifar10_50k.pt — shared with "
                    "the cifar10 tree). Loaded if it exists (skips the ~40min "
                    "real pass), created otherwise. Empty = no caching.")

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# TorchMetrics FID
from torchmetrics.image.fid import FrechetInceptionDistance

# Our EBM models + utilities (local copies, same as the training script)
from network_cnn import EBCNNModelWrapper
from network_unet import (EBViTModelWrapper, EBMLPModelWrapper,
                          EBRonnebergerConvUNetWrapper)
from network_pcn import PCNVelocityWrapper
from utils import create_timestamped_dir, plot_epsilon

# Progress bar
from tqdm import tqdm

##############################################################################
# 1) CIFAR-10 Data (single GPU)
##############################################################################

def get_cifar10_train_loader(batch_size, num_workers, root=None):
    """Real-set DataLoader in [0,1] (plain ToTensor). CIFAR-10 train by default;
    --real_dataset=imagenet32 switches to the official downsampled-ImageNet
    train batches, matching get_cifar10_train_loader_sharded in the multi-GPU
    runner so both entry points score against identical reals."""
    if FLAGS.real_dataset == "imagenet32":
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imagenet"))
        from dataset_imagenet32 import ImageNet32Dataset
        dataset = ImageNet32Dataset(split="train", root=os.environ.get("IMAGENET32_PATH"),
                                    transform=T.ToTensor())
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, drop_last=False)
    if root is None:
        root = os.environ.get("CIFAR10_PATH", "./data")

    transform = T.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Keep order consistent for evaluation
        drop_last=False,
    )
    return loader

##############################################################################
# 2) Model construction (same registry dispatch as the train script)
##############################################################################

def build_model(device):
    """Build model from FLAGS (same logic as generate_from_checkpoint.py)."""
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
                ws=FLAGS.unet_ws,
                frozen_dropout=FLAGS.pcn_frozen_dropout,
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
            mlp_head=FLAGS.mlp_head,
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
                ws=FLAGS.unet_ws,
            residual_alpha=FLAGS.residual_alpha,
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
                ws=FLAGS.unet_ws,
            residual_alpha=FLAGS.residual_alpha,
            embed_dim=FLAGS.embed_dim,
            transformer_nheads=FLAGS.transformer_nheads,
            transformer_nlayers=FLAGS.transformer_nlayers,
        ).to(device)

    return model, img_shape

##############################################################################
# 3) SDE solver: Euler–Heun
##############################################################################
import torchsde

def solve_sde_heun(model, x, t_start, t_end, dt=0.01):
    """
    Integrates x from t_start..t_end using Stratonovich Euler–Heun
    (via torchsde) with no storing of entire trajectory in memory.
    Returns the final state at t_end, in [-1,1].
    """
    if t_end <= t_start:
        return x

    # Flatten for torchsde
    orig_shape = x.shape
    B = x.size(0)
    x_flat = x.view(B, -1)

    class FlattenSDE(torchsde.SDEStratonovich):
        def __init__(self, net):
            super().__init__(noise_type="diagonal")
            self.net = net

        def f(self, t, y):
            # Drift
            y_unflat = y.view(*orig_shape)
            # Ensure time tensor is on the same device and has a batch dimension
            t_batch = t.expand(B).to(y.device)
            v = self.net(t_batch, y_unflat)
            return v.view(B, -1)

        def g(self, t, y):
            # Diffusion
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
            scale = torch.sqrt(2.0 * e_tensor)
            return scale.expand_as(y) # Use expand_as for robustness

    sde = FlattenSDE(model)
    ts = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)

    with torch.no_grad():
        # "heun" is the name in older torchsde for the Stratonovich Heun method
        x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
        x_final = x_sol[-1].view(*orig_shape).clamp(-1, 1)

    return x_final


##############################################################################
# 4) Main FID computation
##############################################################################
def main(argv):
    # ------------------------------------------------------------
    # A) Initialize Device and Logging
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not FLAGS.output_dir:
        FLAGS.output_dir = "./sampling_results"
    # FID evals are logs, not runs: when running with the config's default
    # output_dir, drop eval dirs into fid_evals/ (the organized bucket) instead
    # of littering the results_cifar10_pcn/ top level. An explicit
    # --output_dir is always respected unchanged.
    if FLAGS.output_dir.rstrip("/") == "./results_cifar10_pcn":
        FLAGS.output_dir = "./results_cifar10_pcn/fid_evals/"
    savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
    logging.get_absl_handler().use_absl_log_file(program_name="fid_cifar10", log_dir=savedir)
    logging.set_verbosity(logging.INFO)
    logging.info(f"Saving logs to: {savedir}")
    logging.info(f"Using device: {device}")

    # ------------------------------------------------------------
    # B) Build Model & Load Checkpoint
    # ------------------------------------------------------------
    net_model, img_shape = build_model(device)
    net_model.eval()

    if not FLAGS.resume_ckpt or not os.path.exists(FLAGS.resume_ckpt):
        raise ValueError(f"--resume_ckpt not found: {FLAGS.resume_ckpt}")

    logging.info(f"Loading checkpoint: {FLAGS.resume_ckpt}")
    ckpt_data = torch.load(FLAGS.resume_ckpt, map_location=device)
    key = "ema_model" if FLAGS.use_ema else "net_model"
    state_dict = ckpt_data[key]
    # Strip 'module.' prefix if the checkpoint was saved from a DDP wrapper
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if FLAGS.ffn_checkpoint_into_pcn:
        if not hasattr(net_model, "pcn"):
            raise ValueError("--ffn_checkpoint_into_pcn requires a pcn --model_type")
        net_model.pcn.load_from_ebvit(state_dict, strict=True)
        logging.info(f"Loaded {key} via EBViT->PCN key remap (train-FFN, infer-PCN).")
    else:
        net_model.load_state_dict(state_dict, strict=True)
        logging.info(f"Loaded {key}.")
    net_model.eval()

    # Sampling seed for the fake-sample noise. World size is 1 here, so this is
    # the multi-GPU runner's rank-0 seed (fid_seed*1000+0) -- a 1-GPU run and a
    # rank of a multi-GPU run at the same fid_seed draw the same initial noise.
    if FLAGS.fid_seed >= 0:
        torch.manual_seed(FLAGS.fid_seed * 1000)
        logging.info(f"fid_seed={FLAGS.fid_seed} (seed = fid_seed*1000)")

    # ------------------------------------------------------------
    # C) Process Real CIFAR Data for FID
    # ------------------------------------------------------------
    times_to_sample = [float(x.strip()) for x in FLAGS.fid_times.split(",")]

    # Create a separate FID calculator for each sampling time
    fid_dict = {t_val: FrechetInceptionDistance(feature=2048).to(device) for t_val in times_to_sample}

    # --- real Inception features: compute ONCE (all tau_s share the same real
    #     set), optionally cached to disk (ported from cifar10; the shared cache
    #     ./data/fid_real_features_cifar10_50k.pt saves ~40min per quick-FID).
    #     The original loop ran Inception on the real set len(times)x redundantly.
    def _grab_real(m):
        return {"sum": m.real_features_sum.detach().cpu().clone(),
                "cov": m.real_features_cov_sum.detach().cpu().clone(),
                "num": m.real_features_num_samples.detach().cpu().clone()}
    def _set_real(m, st):
        m.real_features_sum.copy_(st["sum"].to(device))
        m.real_features_cov_sum.copy_(st["cov"].to(device))
        m.real_features_num_samples.copy_(st["num"].to(device))

    cache = FLAGS.cache_real_features
    if cache and os.path.exists(cache):
        logging.info(f"Loading cached real features from {cache} (skipping real pass)")
        st = torch.load(cache, map_location="cpu")
    else:
        logging.info("Computing real Inception features once...")
        ref = FrechetInceptionDistance(feature=2048).to(device)
        train_loader = get_cifar10_train_loader(
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
        )
        for real_imgs, _ in tqdm(train_loader, desc="Processing Real Data"):
            real_imgs = real_imgs.to(device)  # in [0,1]
            real_uint8 = (real_imgs * 255).clamp(0, 255).to(torch.uint8)
            ref.update(real_uint8, real=True)
        st = _grab_real(ref)
        if cache:
            torch.save(st, cache)
            logging.info(f"Saved real features to {cache}")
    for _m in fid_dict.values():
        _set_real(_m, st)

    # ------------------------------------------------------------
    # D) Generate and Process Fake Images
    # ------------------------------------------------------------
    total_samples_to_gen = FLAGS.fid_n_samples
    logging.info(f"Generating {total_samples_to_gen} fake samples for FID...")

    n_batches = (total_samples_to_gen + FLAGS.batch_size - 1) // FLAGS.batch_size
    num_generated = 0

    for _ in tqdm(range(n_batches), desc="Generating Fake Data"):
        remaining_to_gen = total_samples_to_gen - num_generated
        curr_bsz = min(FLAGS.batch_size, remaining_to_gen)

        if curr_bsz <= 0:
            break

        # Start from standard normal in [B, 3, 32, 32]
        x = torch.randn(curr_bsz, *img_shape, device=device)

        t_prev = 0.0
        # Sequentially generate samples for each time point
        for t_end in times_to_sample:
            # Integrate from t_prev to t_end
            x = solve_sde_heun(net_model, x, t_prev, t_end, dt=FLAGS.dt_gibbs)

            # Convert to [0,1] range uint8 for FID
            x_01 = (x + 1.0) / 2.0
            x_uint8 = (x_01 * 255).clamp(0, 255).to(torch.uint8)

            # Update the corresponding FID metric with fake images
            fid_dict[t_end].update(x_uint8, real=False)

            # The end time of this step becomes the start time for the next
            t_prev = t_end

        num_generated += curr_bsz

    # ------------------------------------------------------------
    # E) Compute and Print Final FIDs
    # ------------------------------------------------------------
    logging.info("Computing final FID scores...")
    # NB train_loader does not exist on a cache-real-features hit -- report the
    # real count from the metric state instead (cifar10 job 32125261 lost 3.9h
    # of generation to exactly this reference).
    _real_n = int(next(iter(fid_dict.values())).real_features_num_samples)
    logging.info(f"Comparison is based on {_real_n} real vs {num_generated} fake samples.")

    for t_val in times_to_sample:
        fid_val = fid_dict[t_val].compute()
        # Also log the sample counts to verify
        real_count = fid_dict[t_val].real_features_num_samples
        fake_count = fid_dict[t_val].fake_features_num_samples
        logging.info(f"FID at t={t_val:.2f} => {fid_val:.4f} (real: {real_count}, fake: {fake_count})")


if __name__ == "__main__":
    app.run(main)
