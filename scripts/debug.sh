#!/bin/bash
#SBATCH --job-name=cka_rl_metaworld
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --partition=nvidia
#SBATCH --array=0
#SBATCH --output=/scratch/yd2247/cka_rl/logs/%A.out
#SBATCH --error=/scratch/yd2247/cka_rl/logs/%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yd2247@nyu.edu

set -euo pipefail

echo "SLURM_JOBID: ${SLURM_JOBID:-}"

module purge
module load python/3.10
module load cuda/12.2.0          # Best available match (closest to PyTorch's CUDA 12.8)
module load ffmpeg/4.2.2
module load mesa/20.2.1

# Use scratch for caches/tmp (avoid home quota)
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export UV_CACHE_DIR=/scratch/yd2247/.cache/uv
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# Debug: Check conda locations
echo "=== DEBUG: Conda debugging ==="
echo "HOME: $HOME"
echo "CONDA_PREFIX (before): ${CONDA_PREFIX:-not set}"
echo "CONDA_DEFAULT_ENV (before): ${CONDA_DEFAULT_ENV:-not set}"
echo "which conda: $(which conda 2>&1 || echo 'not found')"
echo "CONDA_ROOT (before): ${CONDA_ROOT:-not set}"

# Try to use user's own conda installation
# First, try to find user's conda
USER_CONDA="$HOME/.conda"
if [ -d "$USER_CONDA" ]; then
    echo "Found user conda directory: $USER_CONDA"
    # Try to find conda.sh in common locations
    if [ -f "$USER_CONDA/etc/profile.d/conda.sh" ]; then
        echo "Using: $USER_CONDA/etc/profile.d/conda.sh"
        source "$USER_CONDA/etc/profile.d/conda.sh"
    elif [ -f "$USER_CONDA/condabin/conda.sh" ]; then
        echo "Using: $USER_CONDA/condabin/conda.sh"
        source "$USER_CONDA/condabin/conda.sh"
    else
        echo "Warning: Could not find conda.sh in user conda directory"
        # Fall back to conda module but set MKL_INTERFACE_LAYER to avoid error
        module load conda-gcc/11.2.0
        export MKL_INTERFACE_LAYER=LP64,GNU
        eval "$(conda shell.bash hook)"
    fi
else
    echo "User conda directory not found, using module"
    # Load conda module but set MKL_INTERFACE_LAYER to avoid unbound variable error
    module load conda-gcc/11.2.0
    export MKL_INTERFACE_LAYER=LP64,GNU
    eval "$(conda shell.bash hook)"
fi

echo "CONDA_PREFIX (after init): ${CONDA_PREFIX:-not set}"
echo "which conda (after init): $(which conda 2>&1 || echo 'not found')"

# Activate environment
echo "Activating conda environment: cka-rl-meta"
ENV_PATH="$HOME/.conda/envs/cka-rl-meta"
if [ -d "$ENV_PATH" ]; then
    echo "Found environment at: $ENV_PATH"
    # Try direct activation first
    if [ -f "$ENV_PATH/bin/activate" ]; then
        echo "Trying direct activation via $ENV_PATH/bin/activate"
        source "$ENV_PATH/bin/activate"
    else
        # Use conda activate
        conda activate cka-rl-meta || {
            echo "ERROR: Failed to activate conda environment"
            echo "Available environments:"
            conda env list 2>&1 || echo "Could not list environments"
            exit 1
        }
    fi
else
    echo "Environment not found at $ENV_PATH, trying conda activate"
    conda activate cka-rl-meta || {
        echo "ERROR: Failed to activate conda environment"
        echo "Available environments:"
        conda env list 2>&1 || echo "Could not list environments"
        exit 1
    }
fi

echo "CONDA_PREFIX (after activate): ${CONDA_PREFIX:-not set}"
echo "CONDA_DEFAULT_ENV (after activate): ${CONDA_DEFAULT_ENV:-not set}"

# CRITICAL FIX: Ensure conda environment's Python is first in PATH
# The python/3.10 module may have put system Python first, so we need to fix PATH
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/bin" ]; then
    echo "Fixing PATH to prioritize conda environment..."
    export PATH="${CONDA_PREFIX}/bin:$PATH"
    echo "Updated PATH (first entry): $(echo $PATH | cut -d: -f1)"
fi

# Additional verification
echo "--- Environment Verification ---"
echo "Python executable: $(which python 2>&1)"
echo "Python version: $(python --version 2>&1)"
echo "Python path: $(python -c 'import sys; print(sys.executable)' 2>&1)"
echo "Expected conda Python: ${CONDA_PREFIX:-N/A}/bin/python"
echo "PATH (first 3 entries): $(echo $PATH | tr ':' '\n' | head -3 | tr '\n' ':')"

# Test critical imports
echo "--- Testing Critical Imports ---"
python -c "import numpy; print('✓ numpy:', numpy.__version__)" 2>&1 || echo "✗ numpy: FAILED"
python -c "import torch; print('✓ torch:', torch.__version__)" 2>&1 || echo "✗ torch: FAILED"
python -c "import gymnasium; print('✓ gymnasium:', gymnasium.__version__)" 2>&1 || echo "✗ gymnasium: FAILED"
python -c "import metaworld; print('✓ metaworld: OK')" 2>&1 || echo "✗ metaworld: FAILED"

# Check working directory and experiment files
echo "--- Experiment Files Check ---"
cd /scratch/yd2247/cka_rl/experiments/meta-world
echo "Current directory: $(pwd)"
echo "run_experiments.py exists: $([ -f run_experiments.py ] && echo 'YES' || echo 'NO')"
echo "run_sac.py exists: $([ -f run_sac.py ] && echo 'YES' || echo 'NO')"
echo "tasks.py exists: $([ -f tasks.py ] && echo 'YES' || echo 'NO')"

echo "=== End DEBUG ==="

# Change to experiments/meta-world directory
cd /scratch/yd2247/cka_rl/experiments/meta-world

# Run CKA-RL experiment on Meta-World
# This will run all 20 tasks sequentially as per the paper's experiments
# Note: Low GPU utilization (10-15%) is normal for RL experiments because:
# - Environment interaction is CPU-bound (Meta-World runs on CPU)
# - Most time is spent collecting experience, not training
# - Training happens in small batches, so GPU is used intermittently
# This is expected behavior for single-environment RL experiments
echo "Starting CKA-RL experiment on Meta-World..."
python run_experiments.py \
    --algorithm cka-rl \
    --tag main \
    --seed 42

echo "Experiment completed!"