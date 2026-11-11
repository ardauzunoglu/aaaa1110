#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/train_nanotron_dclm_150m.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base launcher not found at ${BASE_SCRIPT}" >&2
  exit 1
fi

# Each train-step count is twice the corresponding one-pass corpus length.
DATASET_SPECS=(
  #"ardauzunoglu/dclm-1.5b-c4-rewritten-14b-retok:::53250"
  "ardauzunoglu/dclm-1.5b-c4-originals-14b-retok:::53250"
  #"ardauzunoglu/dclm-1.5b-retok:::53250"
  #"ardauzunoglu/dclm-14b-c4-originals-14b-retok:::53250"
  #"ardauzunoglu/dclm-14b-c4-rewritten-14b-retok:::53400"
  #"ardauzunoglu/dclm-28b-retok:::53404"
)

SEEDS=(
  42
  #1000
  #2026
  #1234321
  #123454321
)

DATA_STAGE_REPEATS=4
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llama2_150m_batch_overtrain}"
DRY_RUN="${DRY_RUN:-0}"

if (( ${#DATASET_SPECS[@]} == 0 )); then
  echo "DATASET_SPECS is empty. Edit ${BASH_SOURCE[0]} and add dataset:::train_steps entries." >&2
  exit 1
fi

if (( ${#SEEDS[@]} == 0 )); then
  echo "SEEDS is empty. Edit ${BASH_SOURCE[0]} and add at least one seed." >&2
  exit 1
fi

sanitize_for_run_name() {
  local raw="$1"
  raw="${raw##*/}"
  raw="${raw//[^[:alnum:]]/_}"
  raw="${raw##_}"
  raw="${raw%%_}"
  if [[ -z "${raw}" ]]; then
    raw="dataset"
  fi
  printf '%s' "${raw}"
}

for dataset_spec in "${DATASET_SPECS[@]}"; do
  if [[ "${dataset_spec}" != *":::"* ]]; then
    echo "Invalid DATASET_SPECS entry '${dataset_spec}'. Expected hf_dataset_name:::train_steps" >&2
    exit 1
  fi

  hf_dataset_name="${dataset_spec%%:::*}"
  train_steps="${dataset_spec##*:::}"

  if [[ -z "${hf_dataset_name}" ]]; then
    echo "Invalid DATASET_SPECS entry '${dataset_spec}': dataset name is empty." >&2
    exit 1
  fi

  if ! [[ "${train_steps}" =~ ^[0-9]+$ ]]; then
    echo "Invalid DATASET_SPECS entry '${dataset_spec}': train_steps must be an integer." >&2
    exit 1
  fi

  dataset_slug="$(sanitize_for_run_name "${hf_dataset_name}")"

  for seed in "${SEEDS[@]}"; do
    if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
      echo "Invalid seed '${seed}': seeds must be integers." >&2
      exit 1
    fi

    run_name="${RUN_NAME_PREFIX}_${dataset_slug}_steps${train_steps}_seed${seed}"

    echo "Launching dataset=${hf_dataset_name} train_steps=${train_steps} repeats=${DATA_STAGE_REPEATS} seed=${seed} run_name=${run_name}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    LR_WARMUP_STEPS=512 \
    LR_DECAY_STEPS=52738 \
    CHECKPOINT_INTERVAL=500 \
    DATASET_SHUFFLE=true \
    HF_DATASET_NAME="${hf_dataset_name}" \
    DATA_STAGE_REPEATS="${DATA_STAGE_REPEATS}" \
    TRAIN_STEPS="${train_steps}" \
    SEED="${seed}" \
    RUN_NAME="${run_name}" \
      bash "${BASE_SCRIPT}" "$@"
  done
done
