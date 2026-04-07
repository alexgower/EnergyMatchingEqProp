#!/bin/bash
#SBATCH -A CASTELNOVO-SL2-CPU
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH -p icelake

#SBATCH --job-name=${RUN_NAME:-"EnergyMatching-CPU-Job"}
#SBATCH --output=logs/%j_${RUN_NAME}.log
#SBATCH --error=logs/%j_${RUN_NAME}.err
# Note: cpus-per-task is set from launch_submit_cpu.sh

###############################################################################
# launch_job_cpu.sh — SLURM job script for CPU-only training (submitted by
#                     launch_submit_cpu.sh)
#
# NOTE: The training scripts are designed for multi-GPU DDP (NCCL backend).
# For CPU-only, we use the gloo backend with a single process. This requires
# setting CUDA_VISIBLE_DEVICES="" to force CPU and overriding the backend.
#
# Environment variables passed from launch_submit_cpu.sh:
#   EXPERIMENT   - which experiment to run (cifar10 | mnist_from_cifar10)
#   RUN_NAME     - name for this run (used in job name / logs)
#   EXTRA_FLAGS  - additional absl flag overrides
#   NPROCS       - number of CPU processes to allocate
###############################################################################

# Ensure uv is in PATH on compute nodes
export PATH="$HOME/.local/bin:$PATH"

# Compute nodes need GCC 11+ libstdc++ for pot/torchcfm C extensions (GLIBCXX_3.4.29)
export LD_LIBRARY_PATH="/usr/local/software/master/gcc/11/lib64:${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="/home/apg59/rds/hpc-work/EnergyMatchingEqProp"
EXPERIMENT="${EXPERIMENT:-cifar10}"
NPROCS="${NPROCS:-2}"
TRAIN_SCRIPT="${PROJECT_ROOT}/experiments/${EXPERIMENT}/train_cifar_multigpu.py"

# ─── Environment setup ─────────────────────────────────────────────────────
echo "Loading modules for Icelake nodes..."
module load rhel8/default-icl
module load python
echo "Modules loaded."

echo "============================================"
echo "  EnergyMatchingEqProp — CPU SLURM Job"
echo "============================================"
echo "Experiment:      $EXPERIMENT"
echo "Training script: $TRAIN_SCRIPT"
echo "Run name:        $RUN_NAME"
echo "CPU processes:   $NPROCS"
echo "Extra flags:     ${EXTRA_FLAGS:-'(none)'}"
echo ""

# ─── Validate ───────────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# CPU info
echo "CPU info:"
lscpu | grep "Model name"
echo "Number of logical CPUs allocated: ${NPROCS}"

# ─── Force CPU-only ─────────────────────────────────────────────────────────
# Hide GPUs so PyTorch uses CPU; override DDP backend to gloo
export CUDA_VISIBLE_DEVICES=""
export DIST_BACKEND="gloo"

echo ""
echo "WARNING: CPU-only mode. The training scripts were designed for GPU (NCCL)."
echo "Using gloo backend with 1 process. Training will be significantly slower."
echo "If this fails, the training script may need modification for CPU support."
echo ""

# ─── Launch training ────────────────────────────────────────────────────────
echo "Launching torchrun on CPU..."

uv run torchrun \
    --standalone \
    --nproc_per_node=1 \
    "$TRAIN_SCRIPT" \
    $EXTRA_FLAGS

echo "Training completed."