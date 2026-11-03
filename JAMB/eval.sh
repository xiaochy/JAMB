#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAMB_ROOT="${SCRIPT_DIR}"
JAMB_PARENT="$(cd "${JAMB_ROOT}/.." && pwd)"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [ -z "${ROBOTWIN_ROOT}" ] && [ -f "${JAMB_ROOT}/../RoboTwin-cvpr-env/script/eval_policy.py" ]; then
    ROBOTWIN_ROOT="$(cd "${JAMB_ROOT}/../RoboTwin-cvpr-env" && pwd)"
fi

policy_name="${POLICY_NAME:-JAMB}"
task_name="${1:-${TASK_NAME:-place_dual_shoes}}"
task_config="${2:-${TASK_CONFIG:-demo_clean}}"
ckpt_setting="${3:-${CKPT_SETTING:-${task_config}}}"
expert_data_num="${4:-${EXPERT_DATA_NUM:-100}}"
checkpoint_num="${5:-${CHECKPOINT_NUM:-300}}"
gpu_id="${6:-${GPU_ID:-0}}"
seed_list="${7:-${SEEDS:-0}}"
test_num="${8:-${TEST_NUM:-100}}"
results_root="${RESULTS_ROOT:-${JAMB_ROOT}/results}"
# Whether to channel-swap the live RGB before feature extraction. Default
# changed to False on 2026-08-01: the /scratch/peilin 26-task batch (the
# active majority of eval traffic through this script) was trained on true
# RGB dino_features. Checkpoints trained via the legacy add_dinov2_to_tracks.py
# pipeline (spurious extra BGR2RGB, net effect BGR into DinoV2) -- e.g. the
# OLD handover_block/stack_blocks_three checkpoints, see memory
# gap_dino_features_color_convention.md -- need SWAP_RGB_CHANNELS=True
# passed explicitly; they will silently eval wrong without it now.
swap_rgb_channels="${SWAP_RGB_CHANNELS:-False}"

if [ ! -f "${ROBOTWIN_ROOT}/script/eval_policy.py" ]; then
    printf 'ROBOTWIN_ROOT must contain script/eval_policy.py: %s\n' "${ROBOTWIN_ROOT}" >&2
    exit 1
fi

if [ -n "${CKPT_PATH:-}" ]; then
    # Explicit path always wins, resolved against a few common roots.
    ckpt_path="${CKPT_PATH}"
    if [[ "${ckpt_path}" = /* ]]; then
        checkpoint_file="${ckpt_path}"
    elif [ -f "${JAMB_ROOT}/${ckpt_path}" ]; then
        checkpoint_file="${JAMB_ROOT}/${ckpt_path}"
    elif [ -f "${ROBOTWIN_ROOT}/${ckpt_path}" ]; then
        checkpoint_file="${ROBOTWIN_ROOT}/${ckpt_path}"
    elif [ -f "${ROBOTWIN_ROOT}/policy/${policy_name}/${ckpt_path}" ]; then
        checkpoint_file="${ROBOTWIN_ROOT}/policy/${policy_name}/${ckpt_path}"
    else
        checkpoint_file="${JAMB_ROOT}/${ckpt_path}"
    fi
else
    # Auto-discover the checkpoint from the Hydra run outputs (or legacy flat layout).
    checkpoint_file="$(python "${JAMB_ROOT}/scripts/find_checkpoint.py" \
        --gap-root "${JAMB_ROOT}" \
        --task-name "${task_name}" \
        --setting "${ckpt_setting}" \
        --expert-data-num "${expert_data_num}" \
        --checkpoint-num "${checkpoint_num}")"
fi

if [ ! -f "${checkpoint_file}" ]; then
    printf 'Checkpoint not found: %s\n' "${checkpoint_file}" >&2
    exit 1
fi

read -r -a seeds <<< "${seed_list}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
# "import JAMB" must resolve to THIS checkout. The parent-dir PYTHONPATH
# trick picks whatever sibling directory is literally named "JAMB" — in a
# worktree (JAMB-<exp>) that loads the main checkout's code and crashes
# state_dict loading with an architecture mismatch (2026-07-05).
_pkg_shim="${JAMB_ROOT}/.eval_pkg"
mkdir -p "${_pkg_shim}"
ln -sfn "${JAMB_ROOT}" "${_pkg_shim}/JAMB"
export PYTHONPATH="${_pkg_shim}:${PYTHONPATH:-}"
export EVAL_RESULT_ROOT="${results_root}"
mkdir -p "${EVAL_RESULT_ROOT}"

cd "${ROBOTWIN_ROOT}"

printf 'Evaluating JAMB policy\n'
printf '  gap_root=%s robotwin_root=%s\n' "${JAMB_ROOT}" "${ROBOTWIN_ROOT}"
printf '  results_root=%s\n' "${results_root}"
printf '  task=%s config=%s checkpoint=%s gpu=%s seeds=%s\n' "${task_name}" "${task_config}" "${checkpoint_file}" "${gpu_id}" "${seed_list}"

for seed in "${seeds[@]}"; do
    PYTHONWARNINGS=ignore::UserWarning \
    python "${JAMB_ROOT}/scripts/eval_policy.py" --config "${JAMB_ROOT}/deploy_policy.yml" \
        --overrides \
        --policy_name "${policy_name}" \
        --task_name "${task_name}" \
        --task_config "${task_config}" \
        --ckpt_setting "${ckpt_setting}" \
        --expert_data_num "${expert_data_num}" \
        --seed "${seed}" \
        --checkpoint_num "${checkpoint_num}" \
        --ckpt_path "${checkpoint_file}" \
        --test_num "${test_num}" \
        --swap_rgb_channels "${swap_rgb_channels}"

done
