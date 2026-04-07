#!/bin/bash

###############################################################################
# launch_submit_cpu.sh — Submit a CPU-only training job to SLURM
#
# Usage:
#   bash launch_submit_cpu.sh
#
# NOTE: The training scripts are designed for GPU (DDP + NCCL). CPU mode uses
# the gloo backend as a fallback. If training fails, the training script may
# need modification for full CPU support. See launch_job_cpu.sh for details.
#
# Modify the variables below to configure your run. EXTRA_FLAGS lets you
# override any absl flag from config_multigpu.py (frozen at submission time).
###############################################################################

# ─── Configuration ──────────────────────────────────────────────────────────
EXPERIMENT="cifar10"              # cifar10 | mnist_from_cifar10
RUN_NAME="EnergyMatching-CPU"    # Name for this run (shows in squeue / logs)
NPROCS=2                          # Number of CPU cores to request
EXTRA_FLAGS=""                    # absl flag overrides

# ─── Print summary ──────────────────────────────────────────────────────────
echo "Job configuration:"
echo "  Experiment:   $EXPERIMENT"
echo "  Run name:     $RUN_NAME"
echo "  CPU procs:    $NPROCS"
echo "  Extra flags:  ${EXTRA_FLAGS:-'(none)'}"

# ─── Create logs directory ──────────────────────────────────────────────────
mkdir -p logs

# ─── Submit to SLURM ────────────────────────────────────────────────────────
sbatch \
    --job-name="$RUN_NAME" \
    --output="logs/%j_${RUN_NAME}.log" \
    --error="logs/%j_${RUN_NAME}.err" \
    --cpus-per-task=$NPROCS \
    --export=ALL,EXPERIMENT=$EXPERIMENT,RUN_NAME=$RUN_NAME,NPROCS=$NPROCS,EXTRA_FLAGS="$EXTRA_FLAGS" \
    launch_job_cpu.sh

echo "CPU job submitted!"
echo "Check status with: squeue -u $USER"