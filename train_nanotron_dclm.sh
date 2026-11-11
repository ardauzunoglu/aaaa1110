#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  bash train_nanotron.sh [extra nanotron run_train.py args]

Main knobs (set as env vars before running):
  RUN_NAME, PROJECT_NAME
  NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT
  DP, TP, PP
  TARGET_TOKENS or TRAIN_STEPS
  SEQ_LEN, MICRO_BATCH_SIZE, GRAD_ACCUM
  LEARNING_RATE, MIN_DECAY_LR, LR_WARMUP_STEPS, LR_DECAY_STEPS, WEIGHT_DECAY
  HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS
  HF_DATASET_NAME, HF_DATASET_CONFIG, HF_DATASET_SPLIT, HF_TEXT_COLUMN, DATA_STAGE_REPEATS, DATASET_SHUFFLE
  USE_FLASH_ATTN (1=FlashAttention, 0=PyTorch SDPA)
  NUM_PROC_PER_PROCESS, DATASET_PROCESSING_BATCH_SIZE
  TOKENIZER_NAME_OR_PATH, TOKENIZER_REVISION, VOCAB_SIZE

Examples:
  RUN_NAME=probe60m_test TARGET_TOKENS=200000000 NPROC_PER_NODE=2 bash train_nanotron.sh
  LEARNING_RATE=1e-4 TRAIN_STEPS=2000 MICRO_BATCH_SIZE=16 bash train_nanotron.sh
  TOKENIZER_NAME_OR_PATH=meta-llama/Llama-2-7b-hf VOCAB_SIZE=auto bash train_nanotron.sh
USAGE
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="${ROOT_DIR}/vendor/nanotron"
NANOTRON_SRC_DIR="${NANOTRON_DIR}/src"


if [[ ! -f "${NANOTRON_DIR}/run_train.py" ]]; then
  echo "Nanotron checkout not found at ${NANOTRON_DIR}" >&2
  echo "Expected: ${NANOTRON_DIR}/run_train.py" >&2
  exit 1
fi

# Ensure we import this vendored Nanotron tree (and not an older one from another repo/env).
export PYTHONPATH="${NANOTRON_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Distributed launcher
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

# Parallelism
DP="${DP:-$((NPROC_PER_NODE * NNODES))}"
TP="${TP:-1}"
PP="${PP:-1}"

if (( DP < 1 || TP < 1 || PP < 1 )); then
  echo "DP, TP, and PP must all be >= 1" >&2
  exit 1
fi

# Run + paths
RUN_NAME="${RUN_NAME:-llama2_150m_dclm_5.2b_0625}"
PROJECT_NAME="${PROJECT_NAME:-mlr_nanotron}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT_DIR}/generated_configs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUT_DIR}/nanotron_checkpoints}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${OUT_DIR}/nanotron_benchmarks}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_DIR}/${RUN_NAME}.yaml}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${OUT_DIR}/checkpoint_evals/${RUN_NAME}}"

# Data
HF_DATASET_NAME="${HF_DATASET_NAME:-ardauzunoglu/dclm-14b}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-default}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-train}"
HF_TEXT_COLUMN="${HF_TEXT_COLUMN:-text}"
DATA_STAGE_REPEATS="${DATA_STAGE_REPEATS:-${HF_DATASET_REPEAT_FACTOR:-1}}"
NUM_LOADING_WORKERS="${NUM_LOADING_WORKERS:-16}"
NUM_PROC_PER_PROCESS="${NUM_PROC_PER_PROCESS:-16}"
DATASET_PROCESSING_BATCH_SIZE="${DATASET_PROCESSING_BATCH_SIZE:-1000}"
DATASET_SHUFFLE="${DATASET_SHUFFLE:-false}"
SEED="${SEED:-42}"

# Attention backend. The SDPA path does not require the external flash-attn package.
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
case "${USE_FLASH_ATTN,,}" in
  1|true|yes|on)
    USE_FLASH_ATTN=1
    ATTENTION_IMPLEMENTATION="flash_attention_2"
    ;;
  0|false|no|off)
    USE_FLASH_ATTN=0
    ATTENTION_IMPLEMENTATION="sdpa"
    ;;
  *)
    echo "USE_FLASH_ATTN must be one of: 1, 0, true, false, yes, no, on, off; got '${USE_FLASH_ATTN}'" >&2
    exit 1
    ;;
esac

if ! [[ "${DATA_STAGE_REPEATS}" =~ ^[0-9]+$ ]] || (( DATA_STAGE_REPEATS < 1 )); then
  echo "DATA_STAGE_REPEATS must be a positive integer, got: ${DATA_STAGE_REPEATS}" >&2
  exit 1
fi

# Tokenizer
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-${ROOT_DIR}/tokenizer/llama2-7b-hf}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-}"
TOKENIZER_USE_FAST="${TOKENIZER_USE_FAST:-true}"

# Checkpoint-time local evals (next-token + MCQA)
EVAL_PYTHON="${EVAL_PYTHON:-python}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
EVAL_TRUST_REMOTE_CODE="${EVAL_TRUST_REMOTE_CODE:-false}"
NEXT_TOKEN_EVAL_SCRIPT="${NEXT_TOKEN_EVAL_SCRIPT:-${ROOT_DIR}/evaluation/evaluate_next_token.py}"
NEXT_TOKEN_EVAL_INPUT="${NEXT_TOKEN_EVAL_INPUT:-${ROOT_DIR}/data/targets/dclm_5kdocs_test.jsonl}"
NEXT_TOKEN_EVAL_INPUT_2="${NEXT_TOKEN_EVAL_INPUT_2:-${ROOT_DIR}/data/targets/fineweb_edu_5kdocs_test.jsonl}"
NEXT_TOKEN_EVAL_TEXT_KEY="${NEXT_TOKEN_EVAL_TEXT_KEY:-text}"
NEXT_TOKEN_EVAL_BATCH_SIZE="${NEXT_TOKEN_EVAL_BATCH_SIZE:-1}"
NEXT_TOKEN_EVAL_MAX_DOCS="${NEXT_TOKEN_EVAL_MAX_DOCS:-1000}"
MCQA_EVAL_SCRIPT="${MCQA_EVAL_SCRIPT:-${ROOT_DIR}/evaluation/evaluate_bank.py}"
MCQA_EVAL_INPUT="${MCQA_EVAL_INPUT:-${ROOT_DIR}/data/targets/mmlu_banks/college_biology.jsonl}"
MCQA_EVAL_BATCH_SIZE="${MCQA_EVAL_BATCH_SIZE:-8}"
# Keep legacy LightEval runner effectively disabled; local_jsonl_eval still runs every checkpoint save.
LOCAL_EVAL_INTERVAL="${LOCAL_EVAL_INTERVAL:-999999937}"

# model, match to paper
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2688}"
NUM_LAYERS="${NUM_LAYERS:-12}"
NUM_HEADS="${NUM_HEADS:-12}"
NUM_KV_HEADS="${NUM_KV_HEADS:-12}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-4096}"
INIT_STD="${INIT_STD:-0.02}"
INITIALIZER_RANGE="${INITIALIZER_RANGE:-0.02}"

# Vocab size:
# - If VOCAB_SIZE is set to an integer, use it directly.
# - If VOCAB_SIZE is unset or "auto", derive from tokenizer.
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
if [[ "${VOCAB_SIZE}" == "auto" ]]; then
  echo "Resolving vocab size from tokenizer: ${TOKENIZER_NAME_OR_PATH} (use_fast=${TOKENIZER_USE_FAST})"
  if VOCAB_SIZE="$(python - "$TOKENIZER_NAME_OR_PATH" "$TOKENIZER_REVISION" "$TOKENIZER_USE_FAST" <<'PY'
import sys
from transformers import AutoTokenizer

tokenizer_name_or_path = sys.argv[1]
tokenizer_revision = sys.argv[2] or None
tokenizer_use_fast = sys.argv[3].strip().lower() not in {"0", "false", "no"}

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name_or_path,
    revision=tokenizer_revision,
    use_fast=tokenizer_use_fast,
    trust_remote_code=True,
)
print(len(tokenizer))
PY
  )"; then
    VOCAB_SIZE="${VOCAB_SIZE//$'\n'/}"
    VOCAB_SIZE="${VOCAB_SIZE//$'\r'/}"
    if ! [[ "${VOCAB_SIZE}" =~ ^[0-9]+$ ]]; then
      echo "Auto-derived VOCAB_SIZE is invalid: '${VOCAB_SIZE}'" >&2
      exit 1
    fi
  else
    py_status=$?
    echo "Failed to auto-derive VOCAB_SIZE (python exit code: ${py_status})." >&2
    echo "This failure happens before training starts, so nohup logs can look empty." >&2
    echo "Try one of these:" >&2
    echo "  1) set VOCAB_SIZE explicitly, e.g. VOCAB_SIZE=32000" >&2
    echo "  2) disable fast tokenizer, e.g. TOKENIZER_USE_FAST=false" >&2
    exit "${py_status}"
  fi
fi

# Optimization
LEARNING_RATE="${LEARNING_RATE:-0.0025}" # in the range
MIN_DECAY_LR="${MIN_DECAY_LR:-0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.6}" # in the range
CLIP_GRAD="${CLIP_GRAD:-1.0}" # match
ZERO_STAGE="${ZERO_STAGE:-0}"

# Batch / schedule
SEQ_LEN="${SEQ_LEN:-4096}" # match
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
TARGET_TOKENS="${TARGET_TOKENS:-3200000000}" # match
TRAIN_STEPS="${TRAIN_STEPS:-4961}"

TOKENS_PER_STEP=$((SEQ_LEN * MICRO_BATCH_SIZE * GRAD_ACCUM * DP))
if [[ -z "${TRAIN_STEPS}" ]]; then
  TRAIN_STEPS=$((((TARGET_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP) - 1))
fi

if (( DATA_STAGE_REPEATS > TRAIN_STEPS )); then
  echo "DATA_STAGE_REPEATS (${DATA_STAGE_REPEATS}) cannot exceed TRAIN_STEPS (${TRAIN_STEPS})." >&2
  exit 1
fi

BASE_STAGE_STEPS=$((TRAIN_STEPS / DATA_STAGE_REPEATS))
STAGE_STEPS_REMAINDER=$((TRAIN_STEPS % DATA_STAGE_REPEATS))

if [[ -z "${LR_WARMUP_STEPS}" ]]; then
  LR_WARMUP_STEPS=$((((TRAIN_STEPS + 99) / 100)))
  if (( LR_WARMUP_STEPS < 1 )); then
    LR_WARMUP_STEPS=1
  fi
fi

LR_DECAY_STEPS="${LR_DECAY_STEPS:-${TRAIN_STEPS}}"

CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-191}"
#CHECKPOINT_EVERY_TOKENS="${CHECKPOINT_EVERY_TOKENS:-1000000}"
#if (( CHECKPOINT_EVERY_TOKENS > 0 )); then
#  CHECKPOINT_INTERVAL=$(((CHECKPOINT_EVERY_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
#fi

mkdir -p "${CONFIG_DIR}" "${CHECKPOINT_ROOT}" "${BENCHMARK_DIR}"

cat > "${CONFIG_PATH}" <<YAML
checkpoints:
  checkpoint_interval: ${CHECKPOINT_INTERVAL}
  checkpoints_path: ${CHECKPOINT_ROOT}/${RUN_NAME}
  checkpoints_path_is_shared_file_system: false
  resume_checkpoint_path: null
  save_final_state: true
  save_initial_state: false
data_stages:
YAML

stage_start_step=1
for (( stage_idx=1; stage_idx<=DATA_STAGE_REPEATS; stage_idx++ )); do
  stage_steps=${BASE_STAGE_STEPS}
  stage_seed=$((SEED + stage_idx - 1))
  if (( stage_idx <= STAGE_STEPS_REMAINDER )); then
    stage_steps=$((stage_steps + 1))
  fi

  cat >> "${CONFIG_PATH}" <<YAML
- name: FineWeb Stage (epoch ${stage_idx})
  start_training_step: ${stage_start_step}
  data:
    seed: ${stage_seed}
    num_loading_workers: ${NUM_LOADING_WORKERS}
    dataset:
      hf_dataset_or_datasets: ${HF_DATASET_NAME}
      hf_dataset_config_name: ${HF_DATASET_CONFIG}
      hf_dataset_splits: ${HF_DATASET_SPLIT}
      text_column_name: ${HF_TEXT_COLUMN}
      dataset_processing_num_proc_per_process: ${NUM_PROC_PER_PROCESS}
      dataset_processing_batch_size: ${DATASET_PROCESSING_BATCH_SIZE}
      dataset_overwrite_cache: false
      dataset_shuffle: ${DATASET_SHUFFLE}
YAML
  stage_start_step=$((stage_start_step + stage_steps))
done

cat >> "${CONFIG_PATH}" <<YAML
general:
  benchmark_csv_path: ${BENCHMARK_DIR}/${RUN_NAME}.csv
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: ${PROJECT_NAME}
  run: ${RUN_NAME}
  seed: ${SEED}
  step: null
lighteval:
  eval_interval: ${LOCAL_EVAL_INTERVAL}
  local_jsonl_eval:
    python_executable: ${EVAL_PYTHON}
    tokenizer: ${TOKENIZER_NAME_OR_PATH}
    device: ${EVAL_DEVICE}
    dtype: ${EVAL_DTYPE}
    trust_remote_code: ${EVAL_TRUST_REMOTE_CODE}
    continue_on_error: true
    convert_nanotron_to_hf_before_eval: true
    hf_conversion_output_dir: ${EVAL_OUTPUT_ROOT}/hf_checkpoints
    next_token_script: ${NEXT_TOKEN_EVAL_SCRIPT}
    next_token_input_jsonl: ${NEXT_TOKEN_EVAL_INPUT}
    next_token_text_key: ${NEXT_TOKEN_EVAL_TEXT_KEY}
    next_token_output_dir: ${EVAL_OUTPUT_ROOT}/next_token
    next_token_batch_size: ${NEXT_TOKEN_EVAL_BATCH_SIZE}
    next_token_extra_args:
      - --input
      - "${NEXT_TOKEN_EVAL_INPUT_2}"
      - --max-docs
      - "${NEXT_TOKEN_EVAL_MAX_DOCS}"
    mcqa_script: ${MCQA_EVAL_SCRIPT}
    mcqa_input_jsonl: ${MCQA_EVAL_INPUT}
    mcqa_output_dir: ${EVAL_OUTPUT_ROOT}/mcqa
    mcqa_batch_size: ${MCQA_EVAL_BATCH_SIZE}
    mcqa_save_by_question: false
logging:
  iteration_step_info_interval: 1
  log_level: info
  log_level_replica: info
model:
  ddp_bucket_cap_mb: 25
  dtype: bfloat16
  init_method:
    std: ${INIT_STD}
  make_vocab_size_divisible_by: 1
  model_config:
    bos_token_id: 1
    eos_token_id: 2
    hidden_act: silu
    initializer_range: ${INITIALIZER_RANGE}
    is_llama_config: true
    max_position_embeddings: ${MAX_POSITION_EMBEDDINGS}
    hidden_size: ${HIDDEN_SIZE}
    intermediate_size: ${INTERMEDIATE_SIZE}
    num_hidden_layers: ${NUM_LAYERS}
    num_attention_heads: ${NUM_HEADS}
    num_key_value_heads: ${NUM_KV_HEADS}
    pad_token_id: 0
    pretraining_tp: 1
    rms_norm_eps: 1.0e-05
    rope_scaling: null
    tie_word_embeddings: false
    use_cache: true
    vocab_size: ${VOCAB_SIZE}
    _attn_implementation: ${ATTENTION_IMPLEMENTATION}
optimizer:
  accumulate_grad_in_fp32: true
  clip_grad: ${CLIP_GRAD}
  learning_rate_scheduler:
    learning_rate: ${LEARNING_RATE}
    lr_decay_starting_step: null
    lr_decay_steps: ${LR_DECAY_STEPS}
    lr_decay_style: cosine
    lr_warmup_steps: ${LR_WARMUP_STEPS}
    lr_warmup_style: linear
    min_decay_lr: ${MIN_DECAY_LR}
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    torch_adam_is_fused: true
  weight_decay: ${WEIGHT_DECAY}
  zero_stage: ${ZERO_STAGE}
parallelism:
  dp: ${DP}
  expert_parallel_size: 1
  pp: ${PP}
  pp_engine: 1f1b
  tp: ${TP}
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
profiler:
  profiler_export_path: ${OUT_DIR}/tb_logs/${RUN_NAME}
tokenizer:
  tokenizer_name_or_path: ${TOKENIZER_NAME_OR_PATH}
  tokenizer_revision: ${TOKENIZER_REVISION:-null}
tokens:
  batch_accumulation_per_replica: ${GRAD_ACCUM}
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: ${MICRO_BATCH_SIZE}
  sequence_length: ${SEQ_LEN}
  train_steps: ${TRAIN_STEPS}
  val_check_interval: -1
YAML

echo "Wrote config: ${CONFIG_PATH}"
echo "Run name: ${RUN_NAME}"
echo "Tokens/step: ${TOKENS_PER_STEP}"
echo "Train steps: ${TRAIN_STEPS}"
echo "Checkpoint every: ${CHECKPOINT_INTERVAL} steps"
echo "Checkpoint path: ${CHECKPOINT_ROOT}/${RUN_NAME}"
echo "Dataset: ${HF_DATASET_NAME}/${HF_DATASET_CONFIG}[${HF_DATASET_SPLIT}]"
echo "Data stage repeats (epochs): ${DATA_STAGE_REPEATS}"
echo "Per-stage steps: base=${BASE_STAGE_STEPS}, remainder=${STAGE_STEPS_REMAINDER}"
echo "Tokenizer: ${TOKENIZER_NAME_OR_PATH}"
echo "Tokenizer revision: ${TOKENIZER_REVISION:-<default>}"
echo "Vocab size: ${VOCAB_SIZE}"
echo "Attention implementation: ${ATTENTION_IMPLEMENTATION} (USE_FLASH_ATTN=${USE_FLASH_ATTN})"

cd "${NANOTRON_DIR}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  run_train.py \
  --config-file "${CONFIG_PATH}" \
  "$@"
