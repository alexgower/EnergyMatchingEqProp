# File: config.py
# Configuration for MNIST Energy Matching training (single-GPU).
# Hyperparameters from paper Section D (Table 4).

from absl import flags


def define_flags():
    FLAGS = flags.FLAGS

    # ── Model architecture ──────────────────────────────────────────────
    flags.DEFINE_string("model", "EBMTime", "Model identifier for output dirs")
    flags.DEFINE_string("output_dir", "./results_mnist/", "Base output directory")
    flags.DEFINE_string("model_type", "cnn",
                        "Model architecture: 'unet_vit' (paper) or 'cnn' (earlier code)")

    # UNet backbone (downscaled to ~2M params for MNIST)
    flags.DEFINE_integer("num_channels", 32, "UNet base channel width")
    flags.DEFINE_integer("num_res_blocks", 2, "ResBlocks per UNet stage")
    flags.DEFINE_list("channel_mult", ["1", "2", "2"],
                      "Channel multipliers per UNet stage")
    flags.DEFINE_string("attention_resolutions", "14",
                        "Resolution(s) at which to apply UNet attention")
    flags.DEFINE_integer("num_heads", 2, "Attention heads in UNet")
    flags.DEFINE_integer("num_head_channels", 32,
                         "Channels per UNet attention head")
    flags.DEFINE_float("dropout", 0.1, "Dropout in UNet + Transformer")

    # ViT head on top of UNet output
    flags.DEFINE_integer("embed_dim", 128, "ViT patch embedding dimension")
    flags.DEFINE_integer("transformer_nheads", 2, "ViT attention heads")
    flags.DEFINE_integer("transformer_nlayers", 2, "ViT encoder layers")
    flags.DEFINE_float("output_scale", 100.0, "Scalar multiplier for V(x)")
    flags.DEFINE_float("energy_clamp", 0.0,
                       "Tanh-based energy clamp (0 = disabled)")

    # ── Training ────────────────────────────────────────────────────────
    flags.DEFINE_float("lr", 1e-4, "Learning rate")
    flags.DEFINE_float("grad_clip", 1.0, "Gradient norm clipping")
    flags.DEFINE_integer("batch_size", 128, "Batch size")
    flags.DEFINE_integer("num_workers", 4, "DataLoader workers")
    flags.DEFINE_integer("warmup", 5000, "LR warmup steps")

    # Two-phase schedule (paper Section D)
    flags.DEFINE_integer("total_steps_phase1", 50000,
                         "Phase 1 iterations (LOT only)")
    flags.DEFINE_integer("total_steps_phase2", 3300,
                         "Phase 2 iterations (LOT + LCD)")
    flags.DEFINE_float("ema_decay_phase1", 0.999,
                       "EMA decay during Phase 1")
    flags.DEFINE_float("ema_decay_phase2", 0.99,
                       "EMA decay during Phase 2")

    # ── EBM / Contrastive Divergence ────────────────────────────────────
    flags.DEFINE_float("epsilon_max", 0.1,
                       "Max Langevin noise scale ε_max")
    flags.DEFINE_float("dt_gibbs", 0.025,
                       "Langevin step size Δt")
    flags.DEFINE_integer("n_gibbs", 75,
                         "Number of Langevin (MALA) steps M_Langevin")
    flags.DEFINE_float("lambda_cd", 1e-3,
                       "Coefficient for LCD loss in Phase 2")
    flags.DEFINE_float("time_cutoff", 1.0,
                       "τ*: flow loss decays to 0 beyond this time")
    flags.DEFINE_float("cd_neg_clamp", 0.05,
                       "β: clamp LCD ≥ -β (0 = disabled)")
    flags.DEFINE_float("cd_trim_fraction", 0.0,
                       "α: fraction of highest neg energies discarded")
    flags.DEFINE_bool("split_negative", True,
                      "If True, 50/50 init: half from data, half from noise")
    flags.DEFINE_bool("same_temperature_scheduler", True,
                      "If True, use same ε(t) schedule for all negatives")

    # ── Evaluation / saving ─────────────────────────────────────────────
    flags.DEFINE_integer("save_step", 5000, "Checkpoint / sample frequency")
    flags.DEFINE_string("resume_ckpt", "",
                        "Path to checkpoint for resuming training")
    flags.DEFINE_string("my_log_dir", "", "Directory for absl logs")

    # ── Sampling ────────────────────────────────────────────────────────
    flags.DEFINE_float("t_end", 2.0,
                       "τ_s: end time for SDE sampling (paper = 2.0)")
    flags.DEFINE_bool("use_ema", True,
                      "Load EMA weights for sampling")


def parse_channel_mult(FLAGS):
    return [int(c) for c in FLAGS.channel_mult]
