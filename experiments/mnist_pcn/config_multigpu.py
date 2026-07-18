# File: config_multigpu.py
# Configuration for MNIST Energy Matching training (PCN project).
# Adapted from CIFAR-10 config, hyperparameters from paper Section D (Table 4).
#
# Phase 2 CLI overrides (after Phase 1 completes):
#   --lambda_cd=1e-3 --n_gibbs=75 --epsilon_max=0.1 --ema_decay=0.99 --total_steps=3300

from absl import flags

def define_flags():
    FLAGS = flags.FLAGS

    # Model + dataset + output
    flags.DEFINE_string("model", "EM_mnist_pcn", "Tag for output directory and checkpoint naming")
    flags.DEFINE_string("dataset", "mnist", "Dataset to train on: 'mnist' (28x28)")
    flags.DEFINE_string("output_dir", "./results_mnist_pcn/", "Directory for results")
    flags.DEFINE_bool("debug", False, "Debug mode")
    flags.DEFINE_string("model_type", "unet_vit",
                        "Model architecture: 'vgg5' (VGG5 + feedforward backprop), "
                        "'pcn' (VGG5 + PCN energy relaxation), 'historical' (legacy CNN), "
                        "'mlp' (pure MLP baseline), 'unet_vit' (UNet + ViT head)")
    flags.DEFINE_string("pool_type", "avgpool",
                        "VGG5 downsampling: 'stride_conv' (learned, best for gen), 'avgpool', 'maxpool' (Scellier original)")
    flags.DEFINE_string("activation", "silu",
                        "Activation function: 'silu' (smooth ∂V/∂x, recommended) or 'relu' (Scellier original)")

    # Energy Based Model
    flags.DEFINE_float("energy_clamp", None,
                       "Energy clamp (tanh-based). If None, no clamp is applied.")
    flags.DEFINE_float("output_scale", 100.0, "Multiplier for final potential output. (CIFAR-10: 1000.0)")


    # Training
    flags.DEFINE_float("lr", 1e-4, "Learning rate (CIFAR-10: 1.2e-3)")
    flags.DEFINE_float("grad_clip", 1.0, "Gradient norm clipping")
    flags.DEFINE_float("grad_skip_threshold", 0.0,
                       "Skip weight update if pre-clip grad norm exceeds this value. "
                       "0 = disabled. Useful for IFT-based PCN where outlier batches "
                       "produce extreme gradients from second-order HVP terms.")
    flags.DEFINE_integer("total_steps", 50000, "Total training steps for Phase 1 (CIFAR-10: 145k)")
    flags.DEFINE_integer("warmup", 5000, "Learning rate warmup steps (proportionally scaled from CIFAR-10's 10000)")     # NOTE: warmup not specified in paper for MNIST. Proportionally scaled from CIFAR-10's 10000/145000 ≈ 7% → 5000/50000 = 10%.
    flags.DEFINE_string("lr_decay", "none",
                        "LR decay after warmup: 'none' (constant after warmup), "
                        "'cosine' (cosine anneal to 0 over remaining steps)")
    flags.DEFINE_integer("batch_size", 128, "Batch size")
    flags.DEFINE_integer("num_workers", 4, "Dataloader workers")
    flags.DEFINE_float("ema_decay", 0.999, "EMA decay for Phase 1 (CIFAR-10: 0.9999). Phase 2 uses 0.99.")
    flags.DEFINE_bool("gen_ema", False,
                      "If True, also generate samples from the EMA model at checkpoint steps.")

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
    flags.DEFINE_float("gen_converge_threshold", 1.0,
                       "Run generation until per-sample L2 velocity norm (averaged over "
                       "batch) drops below this. Same norm as EqM paper (threshold=10 on "
                       "ImageNet, scaled to MNIST dims/λ=1). Auto-used for EqM, ignored "
                       "for FM. Set 0 to disable and use gen_t1 instead.")


    # Weights & Biases
    flags.DEFINE_string("wandb_project", "", "W&B project name. Empty = disable wandb logging.")
    flags.DEFINE_string("wandb_run_name", "", "W&B run name. Empty = auto-generated from flags.")

    # Optional log dir
    flags.DEFINE_string("my_log_dir", "", "Directory for Abseil logs.")

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

    # PCN Energy Dynamics (only used with model_type='pcn')
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


    # UNet + ViT flags (only used with model_type='unet_vit')
    flags.DEFINE_integer("num_channels", 32, "Base channels (CIFAR-10: 128)")
    flags.DEFINE_integer("num_res_blocks", 2, "Number of resblocks per stage")
    flags.DEFINE_integer("num_heads", 2, "Number of attention heads")
    flags.DEFINE_integer("num_head_channels", 64, "Channels per attention head")
    flags.DEFINE_float("dropout", 0.1, "Dropout rate")
    flags.DEFINE_string("attention_resolutions", "14", "Attention at these resolutions")
    flags.DEFINE_integer("embed_dim", 128, "ViT embedding dimension")
    flags.DEFINE_integer("transformer_nheads", 2, "ViT heads")
    flags.DEFINE_integer("transformer_nlayers", 2, "ViT layers")
    flags.DEFINE_list("channel_mult", ["1", "2", "2"], "UNet channel multipliers")



def parse_channel_mult(FLAGS):
    return [int(c) for c in FLAGS.channel_mult]
