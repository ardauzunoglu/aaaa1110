#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${ROOT_DIR}/train_nanotron_dclm.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base launcher not found at ${BASE_SCRIPT}" >&2
  exit 1
fi

# Paper config from Table 2 in 2509.14786.
: "${RUN_NAME:=llama2_1p4b_dclm_papercfg}"
: "${HIDDEN_SIZE:=2048}"
: "${INTERMEDIATE_SIZE:=7168}"
: "${NUM_LAYERS:=24}"
: "${NUM_HEADS:=16}"
: "${NUM_KV_HEADS:=8}"
: "${MAX_POSITION_EMBEDDINGS:=4096}"
: "${INIT_STD:=0.02}"
: "${INITIALIZER_RANGE:=0.02}"

# Paper regularized defaults for this size.
: "${LEARNING_RATE:=0.0003}"
: "${MIN_DECAY_LR:=0.00003}"
: "${WEIGHT_DECAY:=3.2}"
: "${LR_WARMUP_STEPS:=256}"

# Dataset defaults for this model wrapper.
: "${HF_DATASET_NAME:=ardauzunoglu/dclm-14b-retok}"
: "${HF_DATASET_CONFIG:=default}"
: "${HF_DATASET_SPLIT:=train}"
: "${HF_TEXT_COLUMN:=text}"

# Training-length defaults for this model wrapper.
: "${DATA_STAGE_REPEATS:=1}"
: "${TRAIN_STEPS:=13351}"

# Keep tokens/step comparable to the base launcher while being safer on memory.
: "${MICRO_BATCH_SIZE:=2}"
: "${GRAD_ACCUM:=32}"

export RUN_NAME HIDDEN_SIZE INTERMEDIATE_SIZE NUM_LAYERS NUM_HEADS NUM_KV_HEADS
export MAX_POSITION_EMBEDDINGS INIT_STD INITIALIZER_RANGE LEARNING_RATE WEIGHT_DECAY MICRO_BATCH_SIZE GRAD_ACCUM
export HF_DATASET_NAME HF_DATASET_CONFIG HF_DATASET_SPLIT HF_TEXT_COLUMN
export DATA_STAGE_REPEATS TARGET_TOKENS TRAIN_STEPS
export MIN_DECAY_LR LR_WARMUP_STEPS

exec bash "${BASE_SCRIPT}" "$@"
