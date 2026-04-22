#!/bin/bash
# ============================================================================
# NanoLFA-Design — Docker Entrypoint
# ============================================================================
# Detects GPU availability, activates the conda environment, and runs
# the provided command.
# ============================================================================

set -e

# GPU detection
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")
    echo "[nanolfa] GPU detected: ${GPU_INFO}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
    echo "[nanolfa] No GPU detected — running in CPU mode"
    echo "[nanolfa] GPU-dependent features (ESMFold, AlphaFold) will be unavailable or slow"
fi

# Ensure conda environment is active
if [ -d "/opt/conda/envs/nanolfa" ]; then
    export PATH=/opt/conda/envs/nanolfa/bin:$PATH
    export CONDA_DEFAULT_ENV=nanolfa
fi

# Create data directories if they don't exist
mkdir -p /data/targets /data/templates /data/results /data/experimental

# Run the provided command
exec "$@"
