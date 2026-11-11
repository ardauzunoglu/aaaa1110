#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/train_nanotron_dclm_300m.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base launcher not found at ${BASE_SCRIPT}" >&2
  exit 1
fi

# At the default batch shape, 143051 steps process 149999845376 tokens
# (approximately 150B). The active 1.5B-token dataset supports at most 1434
# steps per pass, so 100 shuffled stages keep every stage within its capacity.
DATASET_SPECS=(
  "ardauzunoglu/dclm-1.5b-c4-originals-14b-retok:::143051"
  #"ardauzunoglu/dclm-1.5b-c4-rewritten-14b-retok:::143051"
  #"ardauzunoglu/dclm-1.5b-retok:::143051"
  #"ardauzunoglu/dclm-14b-c4-originals-14b-retok:::143051"
  #"ardauzunoglu/dclm-14b-c4-rewritten-14b-retok:::143051"
  #"ardauzunoglu/dclm-28b-retok:::143051"
)

SEEDS=(
  42
  #1000
  #2026
  #1234321
  #123454321
)

TARGET_TOKENS=150000000000
DATA_STAGE_REPEATS=10
LR_WARMUP_STEPS=1430
CHECKPOINT_INTERVAL=500
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llama2_300m_batch_overtrain_150b}"
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
    lr_decay_steps=$((train_steps - LR_WARMUP_STEPS))

    echo "Launching dataset=${hf_dataset_name} target_tokens=${TARGET_TOKENS} train_steps=${train_steps} repeats=${DATA_STAGE_REPEATS} seed=${seed} run_name=${run_name}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    TARGET_TOKENS="${TARGET_TOKENS}" \
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS}" \
    LR_DECAY_STEPS="${lr_decay_steps}" \
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
    DATASET_SHUFFLE=true \
    HF_DATASET_NAME="${hf_dataset_name}" \
    DATA_STAGE_REPEATS="${DATA_STAGE_REPEATS}" \
    TRAIN_STEPS="${train_steps}" \
    SEED="${seed}" \
    RUN_NAME="${run_name}" \
      bash "${BASE_SCRIPT}" "$@"
  done
done
