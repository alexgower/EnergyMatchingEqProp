# File: config_multigpu.py
# Configuration for MNIST Energy Matching training.
# Adapted from CIFAR-10 config, hyperparameters from paper Section D (Table 4).
#
# Phase 2 CLI overrides (after Phase 1 completes):
#   --lambda_cd=1e-3 --n_gibbs=75 --epsilon_max=0.1 --ema_decay=0.99 --total_steps=3300

from absl import flags

def define_flags():
    FLAGS = flags.FLAGS

    # Model + output
    flags.DEFINE_string("model", "EBMTime", "Flow matching model type")
    flags.DEFINE_string("output_dir", "./results_mnist_from_cifar10/", "Directory for results")

    # Flow/EBM Model parameters (MNIST: downscaled to ~2M params)
    flags.DEFINE_integer("num_channels", 32, "Base channels (CIFAR-10: 128)")
    flags.DEFINE_integer("num_res_blocks", 2, "Number of resblocks per stage")

    flags.DEFINE_float("energy_clamp", None,
                       "Energy clamp (tanh-based). If None, no clamp is applied.")

    # UNet + attention
    flags.DEFINE_integer("num_heads", 2, "Number of attention heads for UNet's internal self-attention. (CIFAR-10: 4)")
    # NOTE: num_head_channels is not specified in the paper for MNIST. Keeping
    # at 64 (same as CIFAR-10) — may need tuning.
    flags.DEFINE_integer("num_head_channels", 64, "Channels per UNet attention head (unsure for MNIST, CIFAR-10: 64).")
    flags.DEFINE_float("dropout", 0.1, "Dropout rate in UNet + Transformer layers.")
    # NOTE: attention_resolutions not specified for MNIST in the paper. Using 14
    # (= 28 / 2) as the MNIST equivalent of CIFAR-10's 16 (= 32 / 2). Unsure.
    flags.DEFINE_string("attention_resolutions", "14", "Attention at these resolution(s). (CIFAR-10: '16', unsure for MNIST)")

    # Patch-based ViT parameters (MNIST: simplified head)
    flags.DEFINE_integer("embed_dim", 128, "Embedding dimension for patch-based ViT head. (CIFAR-10: 384)")
    flags.DEFINE_integer("transformer_nheads", 2, "Number of heads in the ViT encoder. (CIFAR-10: 4)")
    flags.DEFINE_integer("transformer_nlayers", 2, "Number of layers (blocks) in the ViT encoder. (CIFAR-10: 8)")
    flags.DEFINE_float("output_scale", 100.0, "Multiplier for final potential output. (CIFAR-10: 1000.0)")

    flags.DEFINE_list(
        "channel_mult", ["1", "2", "2"],
        "Channel multipliers for each UNet resolution block. (CIFAR-10: [1,2,2,2])"
    )

    flags.DEFINE_bool("debug", False, "Debug mode")

    # Training (MNIST: 50k Phase 1 steps, single A100)
    flags.DEFINE_float("lr", 1e-4, "Learning rate (CIFAR-10: 1.2e-3)")
    flags.DEFINE_float("grad_clip", 1.0, "Gradient norm clipping")
    flags.DEFINE_integer("total_steps", 50000, "Total training steps for Phase 1 (CIFAR-10: 145k)")
    # NOTE: warmup not specified in paper for MNIST. Proportionally scaled from
    # CIFAR-10's 10000/145000 ≈ 7% → 5000/50000 = 10%.
    flags.DEFINE_integer("warmup", 5000, "Learning rate warmup steps (proportionally scaled from CIFAR-10's 10000)")
    flags.DEFINE_integer("batch_size", 128, "Batch size")
    flags.DEFINE_integer("num_workers", 4, "Dataloader workers")
    flags.DEFINE_float("ema_decay", 0.999, "EMA decay for Phase 1 (CIFAR-10: 0.9999). Phase 2 uses 0.99.")

    # Evaluation / Saving
    flags.DEFINE_integer("save_step", 5000, "Checkpoint save frequency (0=disable)")
    flags.DEFINE_string("resume_ckpt", "", "Path to checkpoint for resuming training")

    # EBM + CD (Phase 1 defaults: CD disabled. See file header for Phase 2 CLI overrides.)
    flags.DEFINE_float("epsilon_max", 0.0, "Max step size in Gibbs sampling (Phase 2: 0.1)")
    flags.DEFINE_float("dt_gibbs", 0.025, "Step size for Gibbs sampling (CIFAR-10: 0.01)")
    flags.DEFINE_integer("n_gibbs", 0, "Number of Gibbs steps (Phase 2: 75)")
    flags.DEFINE_float("lambda_cd", 0., "Coefficient for contrastive divergence loss (Phase 2: 1e-3)")
    flags.DEFINE_float("time_cutoff", 1.0, "Flow loss decays to zero beyond t>=time_cutoff")
    flags.DEFINE_float("cd_neg_clamp", 0.05,
                       "Clamp negative total CD below -cd_neg_clamp. 0=disable clamp. (CIFAR-10: 0.02)")
    flags.DEFINE_float(
        "cd_trim_fraction",
        0.0,
        "Fraction of highest negative energies discarded for CD (CIFAR-10: 0.1).",
    )
    flags.DEFINE_bool("split_negative", False, "If True, initialize half of the negative samples from x_real_cd, half from noise")
    flags.DEFINE_bool(
        "same_temperature_scheduler",
        True,
        "If True, ignore at_data_mask and use the same temperature schedule for all samples",
    )


    # Optional log dir
    flags.DEFINE_string("my_log_dir", "", "Directory for Abseil logs.")


def parse_channel_mult(FLAGS):
    return [int(c) for c in FLAGS.channel_mult]
