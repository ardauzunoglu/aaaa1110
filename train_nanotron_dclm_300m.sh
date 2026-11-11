#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${ROOT_DIR}/train_nanotron_dclm.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base launcher not found at ${BASE_SCRIPT}" >&2
  exit 1
fi

# Paper config from Table 2 in 2509.14786.
: "${RUN_NAME:=llama2_300m_dclm_3b_0701}"
: "${HIDDEN_SIZE:=768}"
: "${INTERMEDIATE_SIZE:=2688}"
: "${NUM_LAYERS:=12}"
: "${NUM_HEADS:=12}"
: "${NUM_KV_HEADS:=12}"
: "${MAX_POSITION_EMBEDDINGS:=4096}"

# initialization
: "${INIT_STD:=0.02}"
: "${INITIALIZER_RANGE:=0.02}"

# Paper regularized defaults for this size.
: "${LEARNING_RATE:=0.0025}"
: "${MIN_DECAY_LR:=0.0002}"
: "${WEIGHT_DECAY:=1.6}"
: "${LR_WARMUP_STEPS:=64}"

# Dataset defaults for this model wrapper.
: "${HF_DATASET_NAME:=ardauzunoglu/dclm-3b-retok}"
: "${HF_DATASET_CONFIG:=default}"
: "${HF_DATASET_SPLIT:=train}"
: "${HF_TEXT_COLUMN:=text}"

# Training-length defaults for this model wrapper.
: "${DATA_STAGE_REPEATS:=1}"
: "${TRAIN_STEPS:=2840}"
: "${NPROC_PER_NODE:=4}"
: "${NNODES:=1}"
: "${DP:=4}"
: "${MICRO_BATCH_SIZE:=16}"
: "${GRAD_ACCUM:=4}"
: "${CHECKPOINT_INTERVAL:=191}"

export RUN_NAME HIDDEN_SIZE INTERMEDIATE_SIZE NUM_LAYERS NUM_HEADS NUM_KV_HEADS
export MAX_POSITION_EMBEDDINGS INIT_STD INITIALIZER_RANGE LEARNING_RATE WEIGHT_DECAY
export HF_DATASET_NAME HF_DATASET_CONFIG HF_DATASET_SPLIT HF_TEXT_COLUMN
export DATA_STAGE_REPEATS TRAIN_STEPS
export MIN_DECAY_LR LR_WARMUP_STEPS
export NPROC_PER_NODE NNODES DP MICRO_BATCH_SIZE GRAD_ACCUM CHECKPOINT_INTERVAL


exec bash "${BASE_SCRIPT}" "$@"
