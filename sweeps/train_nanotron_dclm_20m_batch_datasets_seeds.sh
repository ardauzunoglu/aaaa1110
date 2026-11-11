#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/train_nanotron_dclm_20m.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base launcher not found at ${BASE_SCRIPT}" >&2
  exit 1
fi

# Entries are dataset:::tokens-per-parameter. These mirror the six data
# conditions in the 55M launcher: single-source 0.55B datasets use 10 TPP,
# while the combined or doubled 1.1B datasets use 20 TPP.
DATASET_SPECS=(
  "ardauzunoglu/dclm-0.55b-retok:::10"
  "ardauzunoglu/dclm-0.55b-c4-rewritten-0.55b-retok:::20"
  "ardauzunoglu/dclm-0.55b-c4-originals-0.55b-retok:::20"
  "ardauzunoglu/dclm-1.1b-retok:::20"
  "ardauzunoglu/c4-rewritten-0.55b-retok:::10"
  "ardauzunoglu/c4-originals-0.55b-retok:::10"
)

SEEDS=(
  42
  1000
  2026
  1234321
  123454321
)

MODEL_PARAMS=19236608
# Must match the one-GPU batch shape in train_nanotron_dclm_20m.sh.
TOKENS_PER_STEP=$((4096 * 16 * 16 * 1))
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llama2_20m_batch}"
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

for dataset_spec in "${DATASET_SPECS[@]}"; do
  if [[ "${dataset_spec}" != *":::"* ]]; then
    echo "Invalid DATASET_SPECS entry '${dataset_spec}'. Expected hf_dataset_name:::tokens_per_parameter" >&2
    exit 1
  fi

  hf_dataset_name="${dataset_spec%%:::*}"
  tokens_per_parameter="${dataset_spec##*:::}"
  if [[ -z "${hf_dataset_name}" ]] || ! [[ "${tokens_per_parameter}" =~ ^[0-9]+$ ]] || (( tokens_per_parameter < 1 )); then
    echo "Invalid DATASET_SPECS entry '${dataset_spec}'." >&2
    exit 1
  fi

  target_tokens=$((MODEL_PARAMS * tokens_per_parameter))
  train_steps=$(((target_tokens + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
  dataset_slug="$(sanitize_for_run_name "${hf_dataset_name}")"

  for seed in "${SEEDS[@]}"; do
    if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
      echo "Invalid seed '${seed}': seeds must be integers." >&2
      exit 1
    fi

    run_name="${RUN_NAME_PREFIX}_${dataset_slug}_${tokens_per_parameter}tpp_steps${train_steps}_seed${seed}"
    echo "Launching dataset=${hf_dataset_name} ratio=${tokens_per_parameter}tpp target_tokens=${target_tokens} train_steps=${train_steps} seed=${seed} run_name=${run_name}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    HF_DATASET_NAME="${hf_dataset_name}" \
    TARGET_TOKENS="${target_tokens}" \
    TRAIN_STEPS="${train_steps}" \
    SEED="${seed}" \
    RUN_NAME="${run_name}" \
      bash "${BASE_SCRIPT}" "$@"
  done
done
