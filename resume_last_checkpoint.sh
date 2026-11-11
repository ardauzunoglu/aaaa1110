#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="${SCRIPT_DIR}/vendor/nanotron"
NANOTRON_SRC_DIR="${NANOTRON_DIR}/src"

DEFAULT_RUN_NAME="${RUN_NAME:?Set RUN_NAME to the run to resume}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/artifacts/nanotron_checkpoints}"
RUN_CHECKPOINT_DIR="${RUN_CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${RUN_NAME}}"
CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/generated_configs}"
NUM_LOADING_WORKERS="${NUM_LOADING_WORKERS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun)}"

if [[ ! -d "${RUN_CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${RUN_CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun executable not found: ${TORCHRUN_BIN}" >&2
  exit 1
fi
if ! [[ "${NUM_LOADING_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_LOADING_WORKERS must be a non-negative integer, got: ${NUM_LOADING_WORKERS}" >&2
  exit 1
fi

checkpoint_is_complete() {
  local checkpoint_dir="$1"
  [[ -s "${checkpoint_dir}/config.yaml" ]] || return 1
  [[ -s "${checkpoint_dir}/checkpoint_metadata.json" ]] || return 1
  [[ -s "${checkpoint_dir}/model_config.json" ]] || return 1
  [[ -n "$(find "${checkpoint_dir}/model" -type f -name '*.safetensors' -print -quit 2>/dev/null)" ]] || return 1
  [[ -n "$(find "${checkpoint_dir}/optimizer" -type f -name '*.pt' -print -quit 2>/dev/null)" ]] || return 1
  [[ -n "$(find "${checkpoint_dir}/lr_scheduler" -type f -name '*.pt' -print -quit 2>/dev/null)" ]] || return 1
  [[ -n "$(find "${checkpoint_dir}/random" -type f -name '*.pt' -print -quit 2>/dev/null)" ]] || return 1
}

latest_step=""
if [[ -s "${RUN_CHECKPOINT_DIR}/latest.txt" ]]; then
  candidate_step="$(tr -d '[:space:]' < "${RUN_CHECKPOINT_DIR}/latest.txt")"
  if [[ "${candidate_step}" =~ ^[0-9]+$ ]] && checkpoint_is_complete "${RUN_CHECKPOINT_DIR}/${candidate_step}"; then
    latest_step="${candidate_step}"
  else
    echo "Ignoring invalid or incomplete checkpoint referenced by latest.txt: ${candidate_step}" >&2
  fi
fi

if [[ -z "${latest_step}" ]]; then
  mapfile -t checkpoint_candidates < <(
    find "${RUN_CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | awk '/^[0-9]+$/' \
      | sort -nr
  )
  for candidate_step in "${checkpoint_candidates[@]}"; do
    if checkpoint_is_complete "${RUN_CHECKPOINT_DIR}/${candidate_step}"; then
      latest_step="${candidate_step}"
      break
    fi
  done
fi

if [[ -z "${latest_step}" ]]; then
  echo "No complete numeric checkpoint found under: ${RUN_CHECKPOINT_DIR}" >&2
  exit 1
fi

checkpoint_dir="${RUN_CHECKPOINT_DIR}/${latest_step}"
resume_config="${CONFIG_DIR}/${RUN_NAME}_resume_from_${latest_step}.yaml"
mkdir -p "${CONFIG_DIR}"

"${PYTHON_BIN}" - "${checkpoint_dir}/config.yaml" "${resume_config}" "${checkpoint_dir}" "${NUM_LOADING_WORKERS}" <<'PY'
from pathlib import Path
import sys

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
checkpoint_path = Path(sys.argv[3]).resolve()
num_loading_workers = int(sys.argv[4])

replacements = {
    "resume_checkpoint_path": str(checkpoint_path),
    "load_lr_scheduler": "true",
    "load_optimizer": "true",
}
found = {key: False for key in replacements}
output_lines = []

for line in source_path.read_text().splitlines():
    stripped = line.lstrip()
    indentation = line[: len(line) - len(stripped)]
    key = stripped.split(":", 1)[0] if ":" in stripped else None
    if key in replacements and len(indentation) == 2:
        output_lines.append(f"{indentation}{key}: {replacements[key]}")
        found[key] = True
    elif key == "num_loading_workers":
        output_lines.append(f"{indentation}num_loading_workers: {num_loading_workers}")
    else:
        output_lines.append(line)

missing = [key for key, was_found in found.items() if not was_found]
if missing:
    raise RuntimeError(f"Saved config is missing required checkpoint fields: {missing}")

output_path.write_text("\n".join(output_lines) + "\n")
PY

echo "Run: ${RUN_NAME}"
echo "Resuming from checkpoint: ${checkpoint_dir}"
echo "Resume config: ${resume_config}"
echo "DataLoader workers per rank: ${NUM_LOADING_WORKERS}"
echo "Model, optimizer, LR scheduler, training metadata, and random states will be restored."

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; not launching torchrun."
  exit 0
fi

export PYTHONPATH="${NANOTRON_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "${NANOTRON_DIR}"
exec "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  run_train.py \
  --config-file "${resume_config}" \
  "$@"
