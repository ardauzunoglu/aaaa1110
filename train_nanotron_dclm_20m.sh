#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${ROOT_DIR}/train_nanotron_dclm.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base launcher not found at ${BASE_SCRIPT}" >&2
  exit 1
fi

# 19,236,608 parameters with the 32k vocabulary and untied embeddings.
: "${RUN_NAME:=llama2_20m_dclm_seed42}"
: "${HIDDEN_SIZE:=256}"
: "${INTERMEDIATE_SIZE:=896}"
: "${NUM_LAYERS:=3}"
: "${NUM_HEADS:=4}"
: "${NUM_KV_HEADS:=4}"
: "${MAX_POSITION_EMBEDDINGS:=4096}"

# Match the initialization, optimizer, and data defaults used by the 55M runs.
: "${INIT_STD:=0.02}"
: "${INITIALIZER_RANGE:=0.02}"
: "${HF_DATASET_NAME:=ardauzunoglu/c4-originals-0.55b-retok}"
: "${HF_DATASET_CONFIG:=default}"
: "${HF_DATASET_SPLIT:=train}"
: "${HF_TEXT_COLUMN:=text}"
: "${LEARNING_RATE:=0.0025}"
: "${MIN_DECAY_LR:=0.0002}"
: "${WEIGHT_DECAY:=1.6}"
: "${LR_WARMUP_STEPS:=64}"

# Keep the same 1,048,576-token global batch as the 55M runs on one GPU:
# 4096 sequence length * 16 micro batch * 16 accumulation * 1 DP rank.
: "${DATA_STAGE_REPEATS:=1}"
: "${TRAIN_STEPS:=184}"
: "${TARGET_TOKENS:=192366080}"
: "${NPROC_PER_NODE:=4}"
: "${NNODES:=1}"
: "${DP:=4}"
: "${MICRO_BATCH_SIZE:=4}"
: "${GRAD_ACCUM:=16}"
: "${CHECKPOINT_INTERVAL:=64}"
: "${SEED:=42}"

export RUN_NAME HIDDEN_SIZE INTERMEDIATE_SIZE NUM_LAYERS NUM_HEADS NUM_KV_HEADS
export MAX_POSITION_EMBEDDINGS INIT_STD INITIALIZER_RANGE
export HF_DATASET_NAME HF_DATASET_CONFIG HF_DATASET_SPLIT HF_TEXT_COLUMN
export LEARNING_RATE MIN_DECAY_LR WEIGHT_DECAY LR_WARMUP_STEPS
export DATA_STAGE_REPEATS TRAIN_STEPS TARGET_TOKENS
export NPROC_PER_NODE NNODES DP MICRO_BATCH_SIZE GRAD_ACCUM CHECKPOINT_INTERVAL
export SEED

exec bash "${BASE_SCRIPT}" "$@"
