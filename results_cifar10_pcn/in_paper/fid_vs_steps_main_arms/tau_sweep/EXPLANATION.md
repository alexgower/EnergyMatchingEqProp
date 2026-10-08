# tau_s sweep behind the post-CD FID protocol

# Check A: optimal sampling time tau_s after phase-2 CD


Command per arm (fid_cifar_heun_1gpu.py):
  --model_type=ffn_unet_vit --pcn_checkpoint_into_ffn --use_ema --fid_seed=1
  --fid_times=2.5,3.0,3.25,3.75,4.5 --fid_n_samples=2048 --batch_size=128
  --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0

FID vs 50000 real, 2048 fake:

| tau_s | IFT phase-2 (ift_postcd_147000) | EP phase-2 (ep_postcd_102000) |
|-------|--------------------------------|-------------------------------|
| 2.50  | 16.7235 | 16.7067 |
| 3.00  | 16.4665 | 16.5530 |
| 3.25  | **16.4064** | **16.5086** |
| 3.75  | 16.4778 | 16.6955 |
| 4.50  | 16.9806 | 17.3748 |

Both arms minimise at tau_s = 3.25, so the protocol sampling time used for the headline
50k-sample FIDs is unchanged by the CD polish, for BOTH the implicit-solve and the EP arm.

Caveats, stated plainly:
- 2048 samples inflates FID in absolute terms (protocol 50k gives 3.6351 IFT / 3.6401 EP).
  These numbers are NOT comparable to the protocol FIDs and must not be quoted as such.
- The comparison ACROSS tau is paired: the tau points are checkpoints of the same
  trajectories (same fid_seed, consecutive integration segments, not restarts), so the
  sampling noise is common-mode and the shape of the curve is the meaningful output.
- Inference is via the FFN twin (--pcn_checkpoint_into_ffn), which is weight-equivalent to
  the PCN to 3e-7 relative velocity.
- The EP curve degrades faster past 3.25 than the IFT curve (+0.87 vs +0.57 at 4.50); the
  two are within 0.1 FID of each other at and below 3.25.

Files: `fidlog_ift_postcd147k.log` and `fidlog_ep_postcd102k.log` hold the FID values (the evaluation logs). `uv run python3 make_tables.py` rebuilds `tables_tau_sweep_not_in_paper.tex` from the two fidlogs.
