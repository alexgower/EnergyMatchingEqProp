# File: config_multigpu.py
# Configuration for CIFAR-10 Energy Matching training (PCN project).
# Adapted from the mnist_pcn config; hyperparameters restored to the paper's
# CIFAR-10 values (Section D / experiments/cifar10) where they differ.
#
# Phase 2 CLI overrides (after Phase 1 completes):
#   --lambda_cd=1e-3 --n_gibbs=200 --epsilon_max=0.01 --ema_decay=0.99 \
#   --split_negative=True --total_steps=147000 --save_step=100

from absl import flags

def define_flags():
    FLAGS = flags.FLAGS

    # Run / dataset / IO
    flags.DEFINE_string("model", "EM_cifar10_pcn", "Tag for output directory and checkpoint naming")
    flags.DEFINE_string("dataset", "cifar10", "Dataset to train on: 'cifar10' (3x32x32)")
    flags.DEFINE_string("output_dir", "./results_cifar10_pcn/", "Directory for results")
    flags.DEFINE_bool("debug", False, "Debug mode")

    ##########################################################################
    # MODEL ARCHITECTURE
    # ------------------------------------------------------------------------
    # --model_type is a single {paradigm}_{arch} name (see MODEL_REGISTRY at the
    # bottom of this file for the full valid list). Paradigm = ffn (feedforward
    # backprop) or pcn (energy-relaxation Predictive Coding). arch = the network
    # shape. Valid combinations (PCN only exists for two architectures):
    #   ffn_vgg5        VGG5 conv stack + FC head
    #   ffn_historical  legacy CNN
    #   ffn_mlp         pure-MLP baseline
    #   ffn_unet_vit    torchcfm UNet + patch-ViT scalar head (paper's CIFAR-10 arch)
    #   ffn_unet_mlp    torchcfm UNet + avgpool-MLP head (no ViT)
    #   ffn_ronneberger_conv_unet
    #                   attention-free, norm-free Ronneberger-style conv UNet
    #                   (crossbar-native; 1 skip per resolution level, NOT the
    #                   torchcfm/DDPM stack-based backbone). "ffn_conv_unet" still works.
    #   pcn_vgg5        ffn_vgg5's architecture, trained as a PCN (VGG5 CNN, linear chain)
    #   pcn_unet_vit    ffn_unet_vit's architecture, trained as a PCN (UNet DAG)
    #
    # The shape flags below are tagged by the ARCHITECTURE they affect; a tag
    # applies to BOTH the ffn_<arch> and pcn_<arch> model_types where each exists
    # (e.g. [unet_vit] covers ffn_unet_vit AND pcn_unet_vit; [vgg5] covers
    # ffn_vgg5 AND pcn_vgg5). Untagged architectures ignore the flag.
    ##########################################################################
    flags.DEFINE_string("model_type", "ffn_unet_vit",
                        "Network as {paradigm}_{arch}. One of: ffn_vgg5, ffn_historical, "
                        "ffn_mlp, ffn_unet_vit, ffn_unet_mlp, ffn_conv_unet, pcn_vgg5, "
                        "pcn_unet_vit (see MODEL_REGISTRY / legend above).")

    # --- Energy head (ALL architectures) ---
    flags.DEFINE_float("output_scale", 1000.0,
                       "[all] Multiplier on the final scalar potential V(x). (MNIST: 100.0)")
    flags.DEFINE_float("energy_clamp", None,
                       "[all] Tanh soft-clamp on V(x). None = no clamp. NOTE: clamping "
                       "throttles the FM velocity v=-∇V — leave off for flow matching.")

    # --- CNN-family shape (pool_type also affects conv_unet) ---
    flags.DEFINE_string("pool_type", "avgpool",
                        "[vgg5, conv_unet] Downsampling: 'avgpool', "
                        "'maxpool' (Scellier original), or 'stride_conv' (learned, best "
                        "for generation). UNet variants use their own down/up-sampling.")
    flags.DEFINE_string("activation", "silu",
                        "[vgg5] Activation: 'silu' (smooth ∂V/∂x, recommended) "
                        "or 'relu' (Scellier original). UNet/conv_unet hardcode SiLU.")

    # --- UNet backbone shape ---
    flags.DEFINE_integer("num_channels", 128,
                         "[unet_vit, unet_mlp, conv_unet] Base channels (MNIST: 32).")
    flags.DEFINE_list("channel_mult", ["1", "2", "2", "2"],
                      "[unet_vit, unet_mlp, conv_unet] Per-stage channel multipliers "
                      "(CIFAR-10 paper: [1,2,2,2] → stages at 32/16/8/4; MNIST: [1,2,2]).")
    flags.DEFINE_integer("num_res_blocks", 2,
                         "[unet_vit, unet_mlp] ResBlocks per stage. "
                         "(conv_unet uses a fixed 2 conv-blocks per stage.)")
    flags.DEFINE_integer("num_heads", 4,
                         "[unet_vit, unet_mlp] UNet self-attention heads (MNIST: 2).")
    flags.DEFINE_integer("num_head_channels", 64,
                         "[unet_vit, unet_mlp] Channels per UNet attention head. This "
                         "OVERRIDES num_heads in torchcfm (heads = channels // this). "
                         "CIFAR's attention blocks have 256 channels, so 64 -> 4 heads, "
                         "self-consistent with the paper's stated 4 heads. (MNIST needed "
                         "32 because its attention blocks only have 64 channels.)")
    flags.DEFINE_string("attention_resolutions", "16",
                        "[unet_vit, unet_mlp] Apply UNet attention at these "
                        "resolution(s). Converted to a downsample factor ds = "
                        "image_size // res: '16' -> 32//16 = 2, i.e. attention at the "
                        "half-res 16x16 stage (paper value). A never-occurring value "
                        "(e.g. '3') disables it.")
    flags.DEFINE_float("dropout", 0.1,
                       "[unet_vit, unet_mlp] UNet/ViT dropout. (pcn_unet_vit forces 0 — a "
                       "stochastic energy would corrupt relaxation; conv_unet has none.)")

    # --- ViT scalar head shape ---
    flags.DEFINE_integer("patch_size", 4,
                         "[unet_vit] ViT patch size on the 32x32 UNet output "
                         "(32/4 = 8x8 = 64 tokens). Paper value (Figure 7).")
    flags.DEFINE_integer("embed_dim", 384,
                         "[unet_vit] Patch-ViT embedding dimension (MNIST: 128).")
    flags.DEFINE_integer("transformer_nheads", 4,
                         "[unet_vit] ViT encoder attention heads (MNIST: 2).")
    flags.DEFINE_integer("transformer_nlayers", 8,
                         "[unet_vit] ViT encoder layers (MNIST: 2).")

    # --- crossbar-native ablations on the torchcfm UNet backbone ---
    flags.DEFINE_bool("unet_no_attention", False,
                      "[unet_vit, unet_mlp] Remove EVERY attention block. NOTE this is "
                      "NOT the same as --attention_resolutions: that only controls the "
                      "encoder/decoder stages, while UNetModel HARDCODES the middle block "
                      "as ResBlock->Attention->ResBlock, so one attention block always "
                      "survives it. Use this flag for a genuinely attention-free net.")
    flags.DEFINE_bool("unet_no_norm", False,
                      "[unet_vit, unet_mlp] Remove EVERY GroupNorm. With "
                      "--unet_no_attention this leaves an architecture whose every op "
                      "is an MVM, a fixed fan-out, a wiring op or an elementwise "
                      "nonlinearity, i.e. analog-crossbar-native -- the single-axis "
                      "version of the ronneberger_conv_unet comparison. "
                      "May destabilize training; smoke-test first.")

    # --- ronneberger_conv_unet-only options ---
    flags.DEFINE_bool("conv_unet_norm", False,
                      "[ronneberger_conv_unet] Insert GroupNorm after each conv. Sibling of "
                      "pool_type. Recovers the small-width quality gap but "
                      "GroupNorm is only partly crossbar-friendly (per-sample statistics).")

    # Training
    flags.DEFINE_float("lr", 1.2e-3, "Learning rate (CIFAR-10 paper; MNIST PCN runs used 1e-4)")
    flags.DEFINE_float("grad_clip", 1.0, "Gradient norm clipping")
    flags.DEFINE_float("grad_skip_threshold", 0.0,
                       "Skip weight update if pre-clip grad norm exceeds this value. "
                       "0 = disabled. NOTE: this is an ABSOLUTE threshold — high-gnorm "
                       "architectures (e.g. scaled UNet ~75) can silently skip ~every "
                       "update at the default; prefer 0 and rely on grad_clip.")
    flags.DEFINE_integer("total_steps", 145000,
                         "Total training steps for Phase 1 (CIFAR-10 paper: 145k; "
                         "Phase 2 continues to 147k). MNIST used 50k.")
    flags.DEFINE_integer("warmup", 10000, "Learning rate warmup steps (CIFAR-10 paper)")
    flags.DEFINE_string("lr_decay", "none",
                        "LR decay after warmup: 'none' (constant after warmup), "
                        "'cosine' (cosine anneal to 0 over remaining steps)")
    flags.DEFINE_integer("batch_size", 128, "Batch size (per GPU)")
    flags.DEFINE_integer("num_workers", 4, "Dataloader workers")
    flags.DEFINE_float("ema_decay", 0.9999, "EMA decay for Phase 1 (CIFAR-10 paper). Phase 2 uses 0.99.")
    flags.DEFINE_bool("gen_ema", False,
                      "If True, also generate samples from the EMA model at checkpoint steps.")
    flags.DEFINE_bool("tf32_matmul", True,
                      "Allow TF32 tensor-core matmuls (attention/Linear layers) on "
                      "Ampere+. Convolutions already run TF32 by cuDNN default, so "
                      "this aligns the matmul path with the regime all conv results "
                      "were computed in. ~1.3-1.5x faster; fp32 accumulation retained.")

    # Equilibrium Matching (EqM) objective — alternative to Flow Matching
    flags.DEFINE_string("training_objective", "fm",
                        "Training objective: 'fm' (flow matching) or 'eqm' (equilibrium matching)")
    flags.DEFINE_string("eqm_c_type", "linear",
                        "EqM c(γ) schedule: 'linear' (1-γ), 'truncated', 'piecewise'")
    flags.DEFINE_float("eqm_a", 0.8,
                       "EqM truncation point for truncated/piecewise c(γ). Gradient constant for γ≤a, decays to 0.")
    flags.DEFINE_float("eqm_b", 1.0,
                       "EqM piecewise starting value. Only used when eqm_c_type='piecewise'.")
    flags.DEFINE_float("eqm_lambda", 1.0,
                       "EqM gradient multiplier λ on top of c(γ). Paper best: 4.0 on ImageNet.")

    # Evaluation / Saving
    flags.DEFINE_integer("save_step", 5000, "Checkpoint save frequency (0=disable)")
    flags.DEFINE_string("resume_ckpt", "", "Path to checkpoint for resuming training")

    # Generation settings
    flags.DEFINE_float("gen_t1", 1.0, "Integration endpoint for sample generation. "
                       "FM uses 1.0; EqM may need 2.0-5.0 to reach equilibrium.")
    flags.DEFINE_float("gen_dt", 0.01, "Step size for sample generation during training.")
    flags.DEFINE_float("gen_converge_threshold", 2.0,
                       "Run generation until per-sample L2 velocity norm (averaged over "
                       "batch) drops below this. Same norm as EqM paper (threshold=10 on "
                       "ImageNet); scaled to CIFAR dims — sqrt(3072/784) ≈ 2x the MNIST "
                       "value of 1.0. Auto-used for EqM, ignored for FM. Set 0 to "
                       "disable and use gen_t1 instead.")

    # Weights & Biases
    flags.DEFINE_string("wandb_project", "", "W&B project name. Empty = disable wandb logging.")
    flags.DEFINE_string("wandb_run_name", "", "W&B run name. Empty = auto-generated from flags.")

    # Optional log dir
    flags.DEFINE_string("my_log_dir", "", "Directory for Abseil logs.")

    # EBM + CD (Phase 1 defaults: CD disabled. See file header for Phase 2 CLI overrides.)
    flags.DEFINE_float("epsilon_max", 0.0, "Max step size in Gibbs sampling (Phase 2: 0.01)")
    flags.DEFINE_float("dt_gibbs", 0.01, "Step size for Gibbs sampling (CIFAR-10 paper; MNIST: 0.025)")
    flags.DEFINE_integer("n_gibbs", 0, "Number of Gibbs steps (Phase 2: 200)")
    flags.DEFINE_float("lambda_cd", 0., "Coefficient for contrastive divergence loss (Phase 2: 1e-3)")
    flags.DEFINE_float("time_cutoff", 1.0, "Flow loss decays to zero beyond t>=time_cutoff")
    flags.DEFINE_float("cd_neg_clamp", 0.02,
                       "Clamp negative total CD below -cd_neg_clamp. 0=disable clamp. (MNIST: 0.05)")
    flags.DEFINE_float(
        "cd_trim_fraction",
        0.1,
        "Fraction of highest negative energies discarded for CD (CIFAR-10 paper; MNIST: 0.0).",
    )
    flags.DEFINE_bool("split_negative", False, "If True, initialize half of the negative samples from x_real_cd, half from noise")
    flags.DEFINE_bool(
        "same_temperature_scheduler",
        True,
        "If True, ignore at_data_mask and use the same temperature schedule for all samples",
    )

    ##########################################################################
    # PCN ENERGY DYNAMICS (only used with the pcn_* model_types)
    # Relaxation + parameter-gradient method. The PCN *topology* is chosen by
    # --model_type: pcn_vgg5 = the VGG5 CNN decomposed into 6 PC layers (linear chain);
    # pcn_unet_vit = DAG over the torchcfm UNet+ViT blocks (one PC node per
    # block, skips as extra parents; loads ffn_unet_vit checkpoints via
    # load_from_ebvit for fidelity checks; dropout hard-zeroed inside the PCN).
    ##########################################################################
    flags.DEFINE_string("ep_x_grad_mode", "total",
                        "x-gradient in the e-param spring phases: 'total' (default; dE/dx "
                        "at e fixed through the reconstruction scan — exact, no "
                        "h-equilibrium assumption, and free/equal-cost) or 'adiabatic' "
                        "(dE/dx at h fixed; assumes h equilibrated — biased on deep DAGs "
                        "at small K_h, kept for ablation).")
    flags.DEFINE_float("pcn_gamma", 0.01,
                       "γ: linear clamp strength on output node. Small γ → exact Energy "
                       "Matching correspondence. α = 1/γ is set automatically.")
    flags.DEFINE_integer("K_h", 2,
                        "Hidden state equilibration steps. Used in both IFT (h-relaxation "
                        "with x fixed) and EP (inner h-steps per x-step, τ_h << τ_x). "
                        "K_h=2 sufficient for error-parameterized PCN (H≈I → fast convergence). "
                        "Increase for h-parameterized mode or harder problems.")
    flags.DEFINE_float("pcn_dt", 0.5,
                       "Step size for hidden state relaxation. dt=1.0 works for ReLU+PGD "
                       "(Scellier), dt<1 recommended for SiLU+gradient descent.")
    flags.DEFINE_bool("pcn_async", True,
                      "Even/odd async layer updates (Scellier App A.1). Critical for "
                      "trainability — synchronous updates cause oscillations.")
    flags.DEFINE_string("pcn_init_mode", "feedforward",
                        "Hidden state initialization: 'feedforward' (recommended, puts h_k "
                        "near correct basin), 'zeros', or 'random' (for ablation).")
    flags.DEFINE_integer("pcn_cg_steps", 10,
                         "CG iterations for IFT backward linear solve. More steps → more "
                         "accurate parameter gradients (10 is usually sufficient).")
    flags.DEFINE_bool("pcn_float64", False,
                      "Run PCN model in float64 for exact gradient correspondence. "
                      "~8x slower but gives cos≈1.0 for all layers vs feedforward.")
    flags.DEFINE_bool("pcn_error_param", False,
                      "Use error-parameterized dynamics (e_k = f_k(h_{k-1}) - h_k) "
                      "instead of h-parameterized. Better conditioned → float32 + fewer CG steps.")

    # --- Stage 3: EP (Equilibrium Propagation) parameter gradients ---
    flags.DEFINE_string("param_grad_mode", "ift",
                        "Parameter gradient method: 'ift' (Stage 2, IFT + CG) or "
                        "'ep' (Stage 3, spring-clamped EP). EP is first-order and "
                        "fully local — no HVPs, no create_graph.")
    flags.DEFINE_float("lambda_spring", 1.0,
                       "Spring constant λ for EP x-clamping. Larger = stiffer spring, "
                       "faster x convergence. v = output_scale·α·λ·(x*-x_t) at equilibrium.")
    flags.DEFINE_float("beta", 0.1,
                       "EP nudge strength β (velocity-space, λ-independent). Internally "
                       "scaled to β_eff = β / vel_scale² in compute_ep_gradients() so "
                       "the user tunes one value regardless of λ or output_scale. "
                       "Larger = faster nudge convergence but O(β) bias; "
                       "smaller = more accurate but float32 noise in (E_β-E*)/β. "
                       "With nudge_type='quadratic' the accuracy/stability constraint is "
                       "β/λ ≪ 1; with nudge_type='linear' it relaxes to β·|g| ≪ γ "
                       "(g = v_free - v̂), letting β be raised for better gradient SNR.")
    flags.DEFINE_string("nudge_type", "quadratic",
                        "EP nudge loss: 'quadratic' (velocity-MSE nudge, quadratic in x, "
                        "binding β/λ ≪ 1 linear-response constraint) or 'linear' "
                        "(frozen-g linear tilt, zero added x-curvature — removes the β/λ "
                        "constraint and the O((β/λ)²) finite-difference bias; targets the "
                        "identical parameter gradient via the Gauss-Newton stop-gradient "
                        "identity ∇_θ½‖v-v̂‖² = g·∇_θv).")
    flags.DEFINE_integer("T_nudge", 4,
                         "Nudge phase relaxation steps. Can be fewer than K (= T_free free phase) "
                         "since starting from an already-converged equilibrium.")
    flags.DEFINE_bool("thirdphase", True,
                      "Three-phase EP: use both +β and -β nudge phases for O(β²) gradient "
                      "accuracy. 1.5× cost of two-phase but significantly less bias.")
    flags.DEFINE_integer("T_free", 10,
                        "Number of free-phase x-relaxation steps in EP mode. In IFT mode "
                        "this is unused (h-relaxation uses K_h directly).")


def parse_channel_mult(FLAGS):
    return [int(c) for c in FLAGS.channel_mult]


# ---------------------------------------------------------------------------
# model_type registry — single source of truth for the {paradigm}_{arch} names.
#   model_type      : (paradigm, arch,       pcn_topology)
# paradigm ∈ {ffn, pcn}; arch = network shape; pcn_topology is the internal
# PCNVelocityWrapper(pcn_arch=...) value ('cnn'|'unet'), None for ffn.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "ffn_vgg5":       ("ffn", "vgg5",       None),
    "ffn_historical": ("ffn", "historical", None),
    "ffn_mlp":        ("ffn", "mlp",        None),
    "ffn_unet_vit":   ("ffn", "unet_vit",   None),
    "ffn_unet_mlp":   ("ffn", "unet_mlp",   None),
    "ffn_ronneberger_conv_unet": ("ffn", "ronneberger_conv_unet", None),
    # Deprecated alias kept working on purpose: submit scripts and in-flight
    # multi-leg jobs still pass the old name, and each leg re-imports this file.
    # TODO delete this soon
    "ffn_conv_unet":  ("ffn", "ronneberger_conv_unet",  None),
    "pcn_vgg5":       ("pcn", "vgg5",       "cnn"),
    "pcn_unet_vit":   ("pcn", "unet_vit",   "unet"),
}


def resolve_model_type(model_type):
    """Return (paradigm, arch, pcn_topology) for a --model_type value.
    Raises with a migration hint on the old flat names."""
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown --model_type={model_type!r}. Valid: {sorted(MODEL_REGISTRY)}.\n"
            "Naming is now {paradigm}_{arch}. Migrate old flags: "
            "unet_vit->ffn_unet_vit, unet_mlp->ffn_unet_mlp, conv_unet->ffn_conv_unet, "
            "vgg5->ffn_vgg5, historical->ffn_historical, mlp->ffn_mlp, "
            "'pcn --pcn_arch=chain'->pcn_vgg5, 'pcn --pcn_arch=unet'->pcn_unet_vit."
        )


def is_pcn(model_type):
    """True if model_type is a PCN paradigm (unknown names -> False, not error)."""
    return MODEL_REGISTRY.get(model_type, (None,))[0] == "pcn"
