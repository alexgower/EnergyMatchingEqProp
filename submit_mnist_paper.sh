#!/bin/bash

###############################################################################
# submit_mnist_paper.sh — Submit MNIST Phase 1 + Phase 2 as chained SLURM jobs
#
# Reproduces the MNIST experiment from the paper exactly:
#   Phase 1: 50k steps, Algorithm 1 (flow matching only), EMA 0.999
#   Phase 2: 3.3k steps, Algorithm 2 (flow + CD), EMA 0.99
#
# Phase 2 automatically starts after Phase 1 completes successfully.
# Each submission gets a unique RUN_ID so multiple runs don't interfere.
#
# Usage:
#   bash submit_mnist_paper.sh
###############################################################################

PROJECT_ROOT="/home/apg59/rds/hpc-work/EnergyMatchingEqProp"
EXPERIMENT="mnist_from_cifar10"

# Unique ID for this submission — ties Phase 1 and Phase 2 together
RUN_ID="mnist_$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT_DIR="${PROJECT_ROOT}/results_mnist_from_cifar10/${RUN_ID}"

mkdir -p logs

echo "============================================"
echo "  MNIST Paper Reproduction (Phase 1 + 2)"
echo "============================================"
echo "Run ID:     $RUN_ID"
echo "Output dir: $RUN_OUTPUT_DIR"
echo ""

# ─── Phase 1: Flow matching only (50k steps) ───────────────────────────────
# All defaults in config_multigpu.py already match the paper for Phase 1.
# We override output_dir to use our unique RUN_ID directory.
PHASE1_FLAGS="--output_dir=${RUN_OUTPUT_DIR}/phase1"

PHASE1_JOB=$(sbatch --parsable \
    --job-name="${RUN_ID}-P1" \
    --output="logs/%j_${RUN_ID}-Phase1.log" \
    --error="logs/%j_${RUN_ID}-Phase1.err" \
    --gres=gpu:1 \
    --export=ALL,EXPERIMENT=$EXPERIMENT,RUN_NAME=${RUN_ID}-P1,NGPUS=1,EXTRA_FLAGS="$PHASE1_FLAGS" \
    launch_job_gpu.sh)

echo "Phase 1 submitted: Job $PHASE1_JOB"
echo "  50k steps, flow matching only, EMA=0.999"
echo "  Output: ${RUN_OUTPUT_DIR}/phase1/"

# ─── Phase 2: Flow + CD (3.3k steps, depends on Phase 1) ──────────────────
# Creates a wrapper script with the exact output path baked in.

PHASE2_WRAPPER="${PROJECT_ROOT}/logs/${RUN_ID}_phase2.sh"
cat > "$PHASE2_WRAPPER" << PHASE2_EOF
#!/bin/bash
#SBATCH -A CASTELNOVO-SL2-GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p ampere

PROJECT_ROOT="${PROJECT_ROOT}"
EXPERIMENT="${EXPERIMENT}"
PHASE1_DIR="${RUN_OUTPUT_DIR}/phase1"

# Ensure uv is in PATH on compute nodes
export PATH="\$HOME/.local/bin:\$PATH"

# Compute nodes need GCC 11+ libstdc++ for pot/torchcfm C extensions (GLIBCXX_3.4.29)
export LD_LIBRARY_PATH="/usr/local/software/master/gcc/11/lib64:\${LD_LIBRARY_PATH:-}"

# Load modules
module load rhel8/default-amp
module load python

cd "\$PROJECT_ROOT"

# Find the checkpoint from THIS run's Phase 1 (not any other run)
LATEST_CKPT=\$(find "\$PHASE1_DIR" -name "*_weights_step_latest.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)

if [ -z "\$LATEST_CKPT" ]; then
    echo "ERROR: Could not find Phase 1 checkpoint in \$PHASE1_DIR/"
    echo "Contents:"
    find "\$PHASE1_DIR" -type f 2>/dev/null
    exit 1
fi

echo "============================================"
echo "  MNIST Phase 2: Flow + CD (3.3k steps)"
echo "  Run ID: ${RUN_ID}"
echo "============================================"
echo "Resuming from: \$LATEST_CKPT"

# GPU setup
nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \\
           --format=csv -l 30 > "${RUN_OUTPUT_DIR}/gpu_stats_phase2.log" 2>&1 &
MONITOR_PID=\$!

uv run torchrun \\
    --standalone \\
    --nproc_per_node=1 \\
    "experiments/\${EXPERIMENT}/train_cifar_multigpu.py" \\
    --output_dir="${RUN_OUTPUT_DIR}/phase2" \\
    --lambda_cd=1e-3 \\
    --n_gibbs=75 \\
    --epsilon_max=0.1 \\
    --ema_decay=0.99 \\
    --split_negative \\
    --total_steps=3300 \\
    --resume_ckpt="\$LATEST_CKPT"

kill \$MONITOR_PID 2>/dev/null
echo "Phase 2 completed."
PHASE2_EOF

chmod +x "$PHASE2_WRAPPER"

PHASE2_JOB=$(sbatch --parsable \
    --dependency=afterok:$PHASE1_JOB \
    "$PHASE2_WRAPPER")

echo ""
echo "Phase 2 submitted: Job $PHASE2_JOB (depends on $PHASE1_JOB)"
echo "  3.3k steps, flow + CD, EMA=0.99"
echo "  Output: ${RUN_OUTPUT_DIR}/phase2/"
echo ""
echo "Check status: squeue -u $USER"
echo "Phase 2 will start automatically after Phase 1 succeeds."
