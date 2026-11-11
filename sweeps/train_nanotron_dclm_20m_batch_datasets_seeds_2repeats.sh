#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/train_nanotron_dclm_20m.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base launcher not found at ${BASE_SCRIPT}" >&2
  exit 1
fi

DATASET_SPECS=(
  #"ardauzunoglu/dclm-1.5b-c4-originals-14b-retok"
  #"ardauzunoglu/dclm-1.5b-c4-rewritten-14b-retok"
  "ardauzunoglu/dclm-1.5b-retok"
)

SEEDS=(
  42
  #1000
  #2026
  #1234321
  #123454321
)

MODEL_PARAMS=19236608
TOKENS_PER_PARAMETER=1000
# Must match the one-GPU batch shape in train_nanotron_dclm_20m.sh.
TOKENS_PER_STEP=$((4096 * 16 * 16 * 1))
TARGET_TOKENS=$((MODEL_PARAMS * TOKENS_PER_PARAMETER))
TRAIN_STEPS=$(((TARGET_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))

# The nominal corpus is about 15.5B tokens, below the 19.24B-token target.
# Starting two shuffled data stages makes the second stage repeat the corpus.
DATA_STAGE_REPEATS="${DATA_STAGE_REPEATS:-20}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llama2_20m_batch_overtrain}"
DRY_RUN="${DRY_RUN:-0}"

sanitize_for_run_name() {
  local raw="$1"
  raw="${raw##*/}"
  raw="${raw//[^[:alnum:]]/_}"
  raw="${raw##_}"
  raw="${raw%%_}"
  [[ -n "${raw}" ]] || raw="dataset"
  printf '%s' "${raw}"
}

for hf_dataset_name in "${DATASET_SPECS[@]}"; do
  if [[ -z "${hf_dataset_name}" ]]; then
    echo "DATASET_SPECS contains an empty dataset name." >&2
    exit 1
  fi

  dataset_slug="$(sanitize_for_run_name "${hf_dataset_name}")"

  for seed in "${SEEDS[@]}"; do
    if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
      echo "Invalid seed '${seed}': seeds must be integers." >&2
      exit 1
    fi

    run_name="${RUN_NAME_PREFIX}_${dataset_slug}_${TOKENS_PER_PARAMETER}tpp_steps${TRAIN_STEPS}_seed${seed}"
    echo "Launching dataset=${hf_dataset_name} ratio=${TOKENS_PER_PARAMETER}tpp target_tokens=${TARGET_TOKENS} train_steps=${TRAIN_STEPS} repeats=${DATA_STAGE_REPEATS} seed=${seed} run_name=${run_name}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    LR_WARMUP_STEPS=184 \
    LR_DECAY_STEPS=18162 \
    CHECKPOINT_INTERVAL=500 \
    DATASET_SHUFFLE=true \
    HF_DATASET_NAME="${hf_dataset_name}" \
    DATA_STAGE_REPEATS="${DATA_STAGE_REPEATS}" \
    TARGET_TOKENS="${TARGET_TOKENS}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    SEED="${seed}" \
    RUN_NAME="${run_name}" \
      bash "${BASE_SCRIPT}" "$@"
  done
done
