#!/bin/bash
#SBATCH -A CASTELNOVO-SL3-GPU
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p ampere

#SBATCH --job-name=${RUN_NAME:-"EnergyMatching-GPU-Job"}
#SBATCH --output=logs/%j_${RUN_NAME}.log
#SBATCH --error=logs/%j_${RUN_NAME}.err

###############################################################################
# launch_job_gpu.sh — SLURM job script for GPU training (submitted by
#                     launch_submit_gpu.sh)
#
# Environment variables passed from launch_submit_gpu.sh:
#   EXPERIMENT   - which experiment to run (cifar10 | mnist_from_cifar10)
#   RUN_NAME     - name for this run (used in job name / logs)
#   EXTRA_FLAGS  - additional absl flag overrides
#   NGPUS        - number of GPUs (default: 1)
###############################################################################

# Ensure uv is in PATH on compute nodes
export PATH="$HOME/.local/bin:$PATH"

# Compute nodes need GCC 11+ libstdc++ for pot/torchcfm C extensions (GLIBCXX_3.4.29)
export LD_LIBRARY_PATH="/usr/local/software/master/gcc/11/lib64:${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="/home/apg59/rds/hpc-work/EnergyMatchingEqProp"
EXPERIMENT="${EXPERIMENT:-cifar10}"
NGPUS="${NGPUS:-1}"
TRAIN_SCRIPT="${PROJECT_ROOT}/experiments/${EXPERIMENT}/train_cifar_multigpu.py"

# ─── Environment setup ─────────────────────────────────────────────────────
echo "Loading modules for Ampere nodes..."
module load rhel8/default-amp
module load python
echo "Modules loaded."

echo "============================================"
echo "  EnergyMatchingEqProp — GPU SLURM Job"
echo "============================================"
echo "Experiment:      $EXPERIMENT"
echo "Training script: $TRAIN_SCRIPT"
echo "Run name:        $RUN_NAME"
echo "GPUs:            $NGPUS"
echo "Extra flags:     ${EXTRA_FLAGS:-'(none)'}"
echo ""

# ─── Validate ───────────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

# ─── GPU checks ─────────────────────────────────────────────────────────────
echo "Checking GPU availability..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERROR: No GPU detected or NVIDIA driver issue!"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Report CUDA devices
uv run python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of CUDA devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  Device {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / (1024**3):.2f} GB)')
"

# GPU optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"

# Start GPU monitoring
echo "Starting GPU monitoring (gpu_stats.log)..."
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 30 > gpu_stats.log 2>&1 &
MONITOR_PID=$!

# ─── Launch training ────────────────────────────────────────────────────────
echo ""
echo "Launching torchrun with $NGPUS GPU(s)..."

uv run torchrun \
    --standalone \
    --nproc_per_node=$NGPUS \
    "$TRAIN_SCRIPT" \
    $EXTRA_FLAGS

# ─── Cleanup ────────────────────────────────────────────────────────────────
kill $MONITOR_PID 2>/dev/null
echo "GPU statistics saved to gpu_stats.log"
echo "Training completed."