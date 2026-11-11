#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="${SCRIPT_DIR}/vendor/nanotron"
MODELS_ROOT="${1:-${MODELS_ROOT:-${SCRIPT_DIR}/artifacts/nanotron_checkpoints}}"

SAVE_ROOT="${SAVE_ROOT:-${SCRIPT_DIR}/artifacts/hf_checkpoints}"
MODEL_GLOB="${MODEL_GLOB:-*}"

if [[ ! -d "${NANOTRON_DIR}" ]]; then
  echo "Nanotron directory not found: ${NANOTRON_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MODELS_ROOT}" ]]; then
  echo "Models root not found: ${MODELS_ROOT}" >&2
  exit 1
fi

mapfile -t MODEL_DIRS < <(
  find "${MODELS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${MODEL_GLOB}" | sort
)

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
  echo "No model directories found in: ${MODELS_ROOT}" >&2
  exit 1
fi

mkdir -p "${SAVE_ROOT}"

cd "${NANOTRON_DIR}"
export PYTHONPATH="${NANOTRON_DIR}/src:${NANOTRON_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV is set to '${CONDA_ENV}', but conda is not available." >&2
    exit 1
  fi
  LAUNCHER=(conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" -m torch.distributed.run)
else
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
  LAUNCHER=("${PYTHON_BIN}" -m torch.distributed.run)
fi

converted_count=0
skipped_count=0

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  RUN_NAME="$(basename "${MODEL_DIR}")"

  mapfile -t STEPS < <(
    find "${MODEL_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | if command -v rg >/dev/null 2>&1; then
          rg '^[0-9]+$'
        else
          grep -E '^[0-9]+$'
        fi \
      | sort -n
  )

  if [[ ${#STEPS[@]} -eq 0 ]]; then
    echo "Skipping ${RUN_NAME}: no numeric checkpoint directories found." >&2
    skipped_count=$((skipped_count + 1))
    continue
  fi

  LAST_STEP="${STEPS[$((${#STEPS[@]} - 1))]}"
  CKPT_PATH="${MODEL_DIR}/${LAST_STEP}"
  MODEL_SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

  # A completed Hugging Face checkpoint has both its configuration and model
  # weights. Do not treat an existing directory alone as complete, since a
  # failed conversion may leave a partially populated destination behind.
  if [[ -s "${MODEL_SAVE_PATH}/config.json" ]] &&
     { [[ -s "${MODEL_SAVE_PATH}/model.safetensors" ]] ||
       [[ -s "${MODEL_SAVE_PATH}/model.safetensors.index.json" ]] ||
       [[ -s "${MODEL_SAVE_PATH}/pytorch_model.bin" ]] ||
       [[ -s "${MODEL_SAVE_PATH}/pytorch_model.bin.index.json" ]]; }; then
    echo "Skipping ${RUN_NAME}: already converted at ${MODEL_SAVE_PATH}."
    skipped_count=$((skipped_count + 1))
    continue
  fi

  echo "Converting ${RUN_NAME}:"
  echo "  checkpoint_path=${CKPT_PATH}"
  echo "  save_path=${MODEL_SAVE_PATH}"
  echo "  last_step=${LAST_STEP}"

  "${LAUNCHER[@]}" \
    --nproc_per_node="${NPROC_PER_NODE:-1}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29503}" \
    -m examples.llama.convert_nanotron_to_hf \
      --checkpoint_path "${CKPT_PATH}" \
      --save_path "${MODEL_SAVE_PATH}" \
      --tokenizer_name "${TOKENIZER_NAME:-meta-llama/Llama-2-7b-hf}" \
      --config_cls "${CONFIG_CLS:-LlamaConfig}" \
      "${@:2}"

  converted_count=$((converted_count + 1))
done

echo "Finished converting ${converted_count} model(s); skipped ${skipped_count}."
