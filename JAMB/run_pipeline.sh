#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GAP_ENV="RoboTwin"
RAW_DATA_ROOT="/data/vxiao/bimanual_manipulation/JAMB/data/raw"
GPU_ID=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

zarr_complete() {
    local zarr_path="$1"
    # episode_ends written last — its presence means processing is done
    [ -f "${zarr_path}/meta/episode_ends/0" ] || \
    [ "$(find "${zarr_path}/meta/" -name "*.0" 2>/dev/null | wc -l)" -gt 0 ]
}

process_task() {
    local task="$1"
    local zarr="./data/${task}-demo_clean-100-pi3-20-5.zarr"
    if zarr_complete "${zarr}"; then
        log "SKIP: ${task} zarr already complete"
        return 0
    fi
    log "START processing: ${task}"
    RAW_DATA_ROOT="${RAW_DATA_ROOT}" conda run -n "${GAP_ENV}" \
        bash process_data.sh "${task}" demo_clean 100 "${GPU_ID}"
    log "DONE processing: ${task}"
}

# ── Step 1: process handover_block_with_bowls ───────────────────────────────
process_task handover_block_with_bowls

# ── Step 2: train handover_block_with_bowls on GPU 0 ────────────────────────
# 100 episodes, checkpoint every 50 epochs, default 300 epochs total
log "START training: handover_block_with_bowls on GPU ${GPU_ID}"
conda run -n "${GAP_ENV}" \
    bash train.sh handover_block_with_bowls demo_clean 100 0 "${GPU_ID}" 32 300 50
log "DONE training"
