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

module purge
module load cuda/12.2.0
module load ffmpeg/4.2.2
module load mesa/20.2.1

# Use EGL for headless MuJoCo rendering
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless

# Use scratch for caches/tmp (avoid home quota)
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export UV_CACHE_DIR=/scratch/yd2247/.cache/uv
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# Initialize conda (conda environment provides its own Python, no need for python/3.10 module)
export MKL_INTERFACE_LAYER=LP64,GNU
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate cka-rl-meta

# Ensure conda environment's Python is first in PATH
export PATH="${CONDA_PREFIX}/bin:$PATH"

# Change to experiments directory and run
cd /scratch/yd2247/cka_rl/experiments/meta-world
# python run_experiments_crl.py \
#     --algorithm cka-rl \
#     --tag crl_smoketest \
#     --seed 42 \
#     --start-mode 0 \
#     --capture-video \
#     --video-every-n-episodes 50 \
#     --track \
#     --debug True

# python run_experiments.py \
#     --algorithm cka-rl \
#     --tag main \
#     --seed 42 \

python run_experiments_crl.py \
    --algorithm cka-rl \
    --tag crl_smoketest \
    --seed 41 \
    --track \
    --debug True