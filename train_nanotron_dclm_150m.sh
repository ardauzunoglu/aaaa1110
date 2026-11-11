#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${ROOT_DIR}/train_nanotron_dclm.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base launcher not found at ${BASE_SCRIPT}" >&2
  exit 1
fi

# Keep the 150M architecture, but use the June 2026 regime for all non-architecture knobs.
: "${RUN_NAME:=llama2_55m_dclm_6p7b_c4_originals_6p7b_seed42_0714}"
: "${HIDDEN_SIZE:=512}"
: "${INTERMEDIATE_SIZE:=1792}"
: "${NUM_LAYERS:=6}"
: "${NUM_HEADS:=8}"
: "${NUM_KV_HEADS:=8}"
: "${MAX_POSITION_EMBEDDINGS:=4096}"

# June runs used these init defaults.
: "${INIT_STD:=0.02}"
: "${INITIALIZER_RANGE:=0.02}"

# Dataset defaults mirror the normal 150M wrapper.
: "${HF_DATASET_NAME:=ardauzunoglu/dclm-6.7b-c4-originals-6.7b-retok}"
: "${HF_DATASET_CONFIG:=default}"
: "${HF_DATASET_SPLIT:=train}"
: "${HF_TEXT_COLUMN:=text}"

# June optimizer and schedule defaults.
: "${LEARNING_RATE:=0.0025}"
: "${MIN_DECAY_LR:=0.0002}"
: "${WEIGHT_DECAY:=1.6}"
: "${LR_WARMUP_STEPS:=64}"

# June batch shape and run length defaults.
: "${DATA_STAGE_REPEATS:=1}"
: "${TRAIN_STEPS:=12626}"
: "${TARGET_TOKENS:=549453824}"
: "${NPROC_PER_NODE:=4}"
: "${NNODES:=1}"
: "${DP:=4}"
: "${MICRO_BATCH_SIZE:=16}"
: "${GRAD_ACCUM:=4}"
: "${CHECKPOINT_INTERVAL:=191}"

# Seed
: "${SEED:=42}"

export RUN_NAME HIDDEN_SIZE INTERMEDIATE_SIZE NUM_LAYERS NUM_HEADS NUM_KV_HEADS
export MAX_POSITION_EMBEDDINGS INIT_STD INITIALIZER_RANGE
export HF_DATASET_NAME HF_DATASET_CONFIG HF_DATASET_SPLIT HF_TEXT_COLUMN
export LEARNING_RATE MIN_DECAY_LR WEIGHT_DECAY LR_WARMUP_STEPS
export DATA_STAGE_REPEATS TRAIN_STEPS TARGET_TOKENS
export NPROC_PER_NODE NNODES DP MICRO_BATCH_SIZE GRAD_ACCUM CHECKPOINT_INTERVAL
export SEED

exec bash "${BASE_SCRIPT}" "$@"
