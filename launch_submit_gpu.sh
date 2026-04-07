#!/bin/bash

###############################################################################
# launch_submit_gpu.sh — Submit a GPU training job to SLURM
#
# Usage:
#   bash launch_submit_gpu.sh
#
# Modify the variables below to configure your run. EXTRA_FLAGS lets you
# override any absl flag from config_multigpu.py, so each submitted job has
# its own frozen set of hyperparameters (safe to edit this file and resubmit
# immediately — the flags are baked into the SLURM job at submission time).
#
# Examples of EXTRA_FLAGS:
#   EXTRA_FLAGS="--total_steps=100000 --lr=0.001 --batch_size=64"
#   EXTRA_FLAGS="--resume_ckpt=results/EBMTime_20260220_120000/checkpoint_50000.pt"
#   EXTRA_FLAGS="--lambda_cd=1e-3 --n_gibbs=75 --epsilon_max=0.1"
###############################################################################

# ─── Configuration ──────────────────────────────────────────────────────────
EXPERIMENT="mnist_from_cifar10"             # cifar10 | mnist_from_cifar10
RUN_NAME="ep_cnn_v4_scale01_resume"   # Name for this run (shows in squeue / logs)
NGPUS=4                          # Number of GPUs to request
EXTRA_FLAGS="--model_type=ep_cnn --total_steps=50000 --save_step=500 --ep_T=100 --ep_epsilon=0.3 --output_scale=0.01 --ep_act=tanh --resume_ckpt=results_mnist_from_cifar10/EBMTime_20260224_00_ver1/EBMTime_mnist_weights_step_latest.pt"                   # absl flag overrides (see examples above)

# ─── Print summary ──────────────────────────────────────────────────────────
echo "Job configuration:"
echo "  Experiment:   $EXPERIMENT"
echo "  Run name:     $RUN_NAME"
echo "  GPUs:         $NGPUS"
echo "  Extra flags:  ${EXTRA_FLAGS:-'(none)'}"

# ─── Create logs directory ──────────────────────────────────────────────────
mkdir -p logs

# ─── Submit to SLURM ────────────────────────────────────────────────────────
sbatch \
    --job-name="$RUN_NAME" \
    --output="logs/%j_${RUN_NAME}.log" \
    --error="logs/%j_${RUN_NAME}.err" \
    --gres="gpu:${NGPUS}" \
    --time="01:00:00" \
    --export=ALL,EXPERIMENT=$EXPERIMENT,RUN_NAME=$RUN_NAME,NGPUS=$NGPUS,EXTRA_FLAGS="$EXTRA_FLAGS" \
    launch_job_gpu.sh

echo "GPU job submitted!"
echo "Check status with: squeue -u $USER"