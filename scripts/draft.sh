#!/bin/bash
#SBATCH --job-name=cka_rl_metaworld
#SBATCH --time=12:00:00
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
module load conda-gcc/11.2.0
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

# Initialize and activate conda environment
eval "$(conda shell.bash hook)"
conda activate cka-rl-meta

# Change to experiments/meta-world directory
cd /scratch/yd2247/cka_rl/experiments/meta-world

# Run CKA-RL experiment on Meta-World
# This will run all 20 tasks sequentially as per the paper's experiments
echo "Starting CKA-RL experiment on Meta-World..."
python run_experiments.py \
    --algorithm cka-rl \
    --tag main \
    --seed 42

echo "Experiment completed!"