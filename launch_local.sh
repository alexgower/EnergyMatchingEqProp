#!/bin/bash

###############################################################################
# launch_local.sh — Run training locally (single node, auto-detect GPUs)
#
# Usage:
#   EXPERIMENT=cifar10 bash launch_local.sh
#   EXPERIMENT=mnist_from_cifar10 bash launch_local.sh
#
# Override any training flag by setting EXTRA_FLAGS, e.g.:
#   EXTRA_FLAGS="--total_steps=100 --batch_size=32" bash launch_local.sh
###############################################################################

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Compute nodes need GCC 11+ libstdc++ for pot/torchcfm C extensions (GLIBCXX_3.4.29)
export LD_LIBRARY_PATH="/usr/local/software/master/gcc/11/lib64:${LD_LIBRARY_PATH:-}"

# ─── Configuration ──────────────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-cifar10}"          # cifar10 | mnist_from_cifar10
NGPUS="${NGPUS:-0}"                         # 0 = auto-detect
EXTRA_FLAGS="${EXTRA_FLAGS:-}"              # additional absl flag overrides

# ─── Validate experiment ────────────────────────────────────────────────────
PROJECT_ROOT="/home/apg59/rds/hpc-work/EnergyMatchingEqProp"
TRAIN_SCRIPT="${PROJECT_ROOT}/experiments/${EXPERIMENT}/train_cifar_multigpu.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    echo "Valid experiments: cifar10, mnist_from_cifar10"
    exit 1
fi

echo "============================================"
echo "  EnergyMatchingEqProp — Local Training"
echo "============================================"
echo "Experiment:     $EXPERIMENT"
echo "Training script: $TRAIN_SCRIPT"
echo "Working dir:    $PROJECT_ROOT"
echo ""

# ─── GPU detection ──────────────────────────────────────────────────────────
echo "Checking GPU availability..."
nvidia-smi 2>/dev/null
if [ $? -ne 0 ]; then
    echo "No GPU detected. Running in CPU-only mode (will be slow)."
    NGPUS=1
    export CUDA_VISIBLE_DEVICES=""
else
    # Auto-detect GPU count if not set
    if [ "$NGPUS" -eq 0 ]; then
        NGPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    fi
    echo "Using $NGPUS GPU(s)"

    # Report CUDA devices
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of CUDA devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  Device {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / (1024**3):.2f} GB)')
"

    # GPU optimizations
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"

    # Start GPU monitoring in the background
    echo "Starting GPU monitoring (gpu_stats.log)..."
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
               --format=csv -l 30 > gpu_stats.log 2>&1 &
    MONITOR_PID=$!
fi

# ─── Launch training ────────────────────────────────────────────────────────
echo ""
echo "Launching torchrun with $NGPUS GPU(s)..."
echo "Extra flags: ${EXTRA_FLAGS:-'(none)'}"
echo ""

cd "$PROJECT_ROOT"

uv run torchrun \
    --standalone \
    --nproc_per_node=$NGPUS \
    "$TRAIN_SCRIPT" \
    $EXTRA_FLAGS

# ─── Cleanup ────────────────────────────────────────────────────────────────
if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null
    echo "GPU statistics saved to gpu_stats.log"
fi

echo "Training completed."