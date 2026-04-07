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
    flags.DEFINE_string("model_type", "unet_vit",
                        "Model architecture: 'unet_vit' (paper), 'cnn'/'cnn_v2' (feedforward CNN), 'ep_cnn' (EP recurrent CNN), or 'ep_mlp' (EP recurrent MLP)")

    # EP recurrent model parameters
    flags.DEFINE_integer("ep_T", 50, "Number of hidden state convergence steps for EP model")
    flags.DEFINE_float("ep_epsilon", 0.5, "Step size for EP hidden state gradient descent")
    flags.DEFINE_float("ep_init_gain", 1.0, "Weight initialization std gain multiplier for EP linear/conv projection layers")
    flags.DEFINE_string("ep_act", "identity", "Activation in EP couplings: 'identity' (bilinear) or 'tanh' (nonlinear, Hopfield-style)")
    flags.DEFINE_string("ep_act_s4", "", "Activation for s4 (deepest flat) layer only. Empty = use ep_act. Options: tanh, silu, softsign, identity.")
    flags.DEFINE_boolean("ep_skip_s4", False, "Add direct x→s4 skip coupling (Linear 784→256, ~200K params). Global velocity pathway.")
    flags.DEFINE_string("dataset", "mnist", "Dataset to train on: 'mnist', 'mnist_8x8' (MNIST downsampled to 8x8, 60K samples), or 'sklearn_digits' (UCI 8x8, 1797 samples).")
    flags.DEFINE_list("ep_archi", ["784", "512", "512"], "Architecture for EP MLP model: [visible, hidden1, hidden2, ...]")
    flags.DEFINE_bool("ep_spectral_norm", True, "Apply spectral normalization to EP coupling weight layers (w1-w4)")
    flags.DEFINE_float("ep_spectral_scale", 1.0, "Scale factor for spectral-normed weights (set <1 to guarantee rho<1, e.g. 0.9)")
    flags.DEFINE_string("ep_learning_mode", "bptt", "Gradient mode: 'bptt' (full unrolling), 'deq' (detached forward + K Neumann backward), 'ep' (x-clamped EP, O(1) memory), or 'spring' (spring-clamped EP: x dynamic, velocity from displacement, no autograd/create_graph).")
    flags.DEFINE_integer("ep_K", 10, "Number of graph-enabled steps for DEQ backward (= Neumann iterations = EP nudged steps)")
    flags.DEFINE_integer("ep_T1", 100, "EP free phase steps (fully detached, O(1) memory). Only used when ep_learning_mode='ep'.")
    flags.DEFINE_integer("ep_T2", 100, "EP nudged phase steps (fully detached, O(1) memory). Only used when ep_learning_mode='ep'.")
    flags.DEFINE_float("ep_beta", 0.01, "EP nudge strength beta (small = linear response regime). Only used when ep_learning_mode='ep'.")
    flags.DEFINE_float("beta_anneal_halflife", 0, "Steps between beta halvings (exponential decay). 0 = no annealing.")
    flags.DEFINE_bool("ep_explicit_grad", False, "Include explicit gradient term ∂L/∂θ in EP estimator. Default False = implicit-only (stabler, drops O(1) term vs O(1/(1-ρ)) implicit).")
    flags.DEFINE_bool("x_intra_weights", False, "Add learnable quadratic weights x^T W x to energy (affects velocity but not convergence)")
    flags.DEFINE_float("lambda_spring", 10.0, "Spring constant for spring-clamped EP (ep_learning_mode='spring'). Controls stiffness: larger = x stays closer to x_t.")
    flags.DEFINE_float("skip_nudge_disp_threshold", 0.0, "Skip optimizer step when EP nudge_disp exceeds this value (0=disabled). Prevents outlier batches from corrupting Adam moments.")
    flags.DEFINE_float("skip_grad_norm_multiplier", 0.0, "Skip optimizer step when pre-clip grad norm exceeds this multiple of grad_clip (0=disabled). E.g. 100 skips when grad_norm > 100*grad_clip.")
    flags.DEFINE_float("adaptive_ss_rho_target", 0.0, "Adaptive spectral scale: target rho (0=disabled). Controller grows ss when rho < target, shrinks when rho > target.")
    flags.DEFINE_float("adaptive_ss_max", 0.99, "Upper clamp for adaptive spectral scale. Raise above 1.0 for silu (which needs ss>1 to push rho past ~0.95).")
    flags.DEFINE_bool("ep_thirdphase", True, "Use three-phase EP (positive and negative nudge) for O(β²) gradient estimates instead of O(β). Costs +50% per step (extra T2 convergence).")
    flags.DEFINE_list("cnn_channels", ["32", "64", "64", "256"],
        "Channel dims for EP CNN layers: [s1_channels, s2_channels, s3_channels, s4_dim]. "
        "Spatial dims are fixed: s1=14x14, s2=7x7, s3=7x7, s4=flat vector.")

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
    flags.DEFINE_string("gen_mode", "spring", "Generation mode: 'spring' (velocity from spring displacement, matches training) or 'energy_gd' (velocity from energy gradient with x fixed and h converged to equilibrium, matches neuromorphic inference)")
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
