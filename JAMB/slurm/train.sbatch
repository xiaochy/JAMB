#!/bin/bash
#SBATCH --job-name=${RUN_TAG:-gap}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=130G            # dataset decompresses to ~95GB in-process;
                              # use --mem=40G + TRACKS_SHM_DIR on disk-memmap
                              # (see SERVER-SETUP.md) if RAM is scarce
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
# Usage:
#   cd GAP && git checkout <experiment branch>
#   RUN_TAG=<label> sbatch --export=ALL slurm/train.sbatch
# Optional env: TRACKS_SHM_DIR, WANDB_MODE=offline, BATCH_SIZE, EPOCHS

set -euo pipefail
source "${CONDA_PREFIX_1:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null || \
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-gap-train}"

cd "$(dirname "$0")/.."   # GAP root
set -a; source .envrc; set +a

: "${RUN_TAG:?set RUN_TAG=<experiment label>}"
USE_DINO_FEATURES=true \
GPU_ID=0 \
WANDB_MODE="${WANDB_MODE:-offline}" \
bash train.sh handover_block demo_clean_depth 100 0 0 \
  "${BATCH_SIZE:-24}" "${EPOCHS:-300}" 50
