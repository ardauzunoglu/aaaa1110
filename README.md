# Pretraining hand-out (nanotron, Llama-2-style models)

Self-contained copy of the pretraining pipeline from `mlr_new/`. Every script resolves
paths relative to this folder, so the folder can be moved anywhere.

## Contents

```
train_nanotron_dclm.sh            # base launcher: writes a nanotron YAML config, then runs torchrun
train_nanotron_dclm_{20m,150m,300m,600m,1p4b}.sh   # size presets (architecture + hparams) -> base launcher
sweeps/*_batch_datasets_seeds*.sh # loop over (dataset, steps) x seeds -> size preset
sweeps/train_670m_megadocs_seeds.sh # 670M on the 6.7B megadocs corpora (section 3b)
resume_last_checkpoint.sh         # resume a run from its latest complete checkpoint
convert_last_checkpoint_to_hf.sh  # convert last nanotron checkpoint of each run to HF format
vendor/nanotron/                  # patched nanotron (use this one, not upstream)
evaluation/                       # next-token + MCQA evals run automatically at each checkpoint
data/targets/                     # eval inputs for those checkpoint evals
requirements.freeze.txt           # pip freeze of the known-good env (Python 3.11)
```

## 1. Environment

Known-good stack: Python 3.11, torch 2.6.0+cu124, transformers 4.57.6, datasets 5.0.1,
numpy 2.4.6, flash-attn 2.6.3.

```bash
uv venv --python 3.11 env && source env/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.freeze.txt     # flash-attn may need --no-build-isolation
```

flash-attn must be built against the same torch version. If it won't install, run with
`USE_FLASH_ATTN=0` (PyTorch SDPA), which does not need flash-attn. Don't install other
packages that change the torch version (e.g. `llm-foundry`); that broke this env before.

You also need:
- `huggingface-cli login` with access to the gated `meta-llama/Llama-2-7b-hf` tokenizer
  and to the `ardauzunoglu/*` training datasets.
- `wandb login`, or set `WANDB_MODE=offline` / `WANDB_MODE=disabled`.

## 2. Launch a single run

The training data is a pre-tokenized HF dataset (`HF_DATASET_NAME`). Each setting is an
environment variable, so to override one, set it before the command:

```bash
RUN_NAME=my_150m_test \
HF_DATASET_NAME=ardauzunoglu/c4-originals-0.55b-retok \
TRAIN_STEPS=523 SEED=42 \
NPROC_PER_NODE=4 DP=4 \
bash train_nanotron_dclm_150m.sh
```

`bash train_nanotron_dclm.sh --help` lists all settings. The most useful are `RUN_NAME`,
`HF_DATASET_NAME`, `TRAIN_STEPS`, `SEED`, `NPROC_PER_NODE`/`DP`, `MICRO_BATCH_SIZE`,
`GRAD_ACCUM`, `LEARNING_RATE`, `CHECKPOINT_INTERVAL`, `DATA_STAGE_REPEATS` (epochs), and
`OUT_DIR`.

Global batch = `SEQ_LEN(4096) × MICRO_BATCH_SIZE × GRAD_ACCUM × DP` tokens. The presets
use about 1M tokens per step. If you use fewer GPUs, keep the global batch the same by
raising `GRAD_ACCUM`. For example, on 1 GPU: `NPROC_PER_NODE=1 DP=1 GRAD_ACCUM=16`.

Size presets (defaults; the file names are historical labels):

| preset | hidden | layers | heads | ffn  | LR     | micro × accum |
|--------|--------|--------|-------|------|--------|---------------|
| 20m    | 256    | 3      | 4     | 896  | 2.5e-3 | 4 × 16        |
| 150m   | 512    | 6      | 8     | 1792 | 2.5e-3 | 16 × 4        |
| 300m   | 768    | 12     | 12    | 2688 | 2.5e-3 | 16 × 4        |
| 600m   | 1024   | 24     | 16    | 7168 | 1e-3   | 8 × 8         |
| 1p4b   | 2048   | 24     | 16    | 7168 | 3e-4   | 2 × 32        |

## 3. Sweeps (datasets × seeds)

Edit the `DATASET_SPECS` list (`hf_dataset:::train_steps`) and the `SEEDS` list at the top
of a sweep script, then:

```bash
DRY_RUN=1 bash sweeps/train_nanotron_dclm_150m_batch_datasets_seeds.sh   # print the plan
bash sweeps/train_nanotron_dclm_150m_batch_datasets_seeds.sh             # run the runs one after another
```

## 3b. 670M models on the 6.7B megadocs corpora

The `600m` preset is the 670M model (hidden 1024, 24 layers, 16 heads / 8 KV heads,
FFN 7168, untied embeddings; LR 1e-3 → 1e-4 cosine, 256 warmup steps; micro 8 × accum
8 × 4 GPUs ≈ 1M tokens/step). Two public corpora:

| dataset | contents | tokens | steps |
|---|---|---|---|
| `ardauzunoglu/c4-megadocsrewritten-6.7b-retok` | 432,079 stitched megadocs (32 rephrasings + original C4 doc per row, joined by `<s>`) | 6.70B | 6353 |
| `ardauzunoglu/dclm-6.7b-c4-megadocsrewritten-6.7b-retok` | the above shuffled with `dclm-6.7b-retok` | 13.40B | 12720 |

The step counts match earlier 670M runs on the other 6.7B corpora, so the runs are
directly comparable. Both fit in one pass over the data: about 6,390 and 12,770 steps
are available. If you set `TRAIN_STEPS` higher, nanotron starts a second pass instead of
stopping.

```bash
DRY_RUN=1 bash sweeps/train_670m_megadocs_seeds.sh   # 2 datasets x seeds 42/1000/2026
bash sweeps/train_670m_megadocs_seeds.sh

# or one run:
RUN_NAME=llama2_670m_megadocs_mix_seed42 \
HF_DATASET_NAME=ardauzunoglu/dclm-6.7b-c4-megadocsrewritten-6.7b-retok \
TRAIN_STEPS=12720 SEED=42 \
bash train_nanotron_dclm_600m.sh
```

Expect the first run to spend a while downloading the dataset (12.7 GB of parquet for
the c4-only corpus and 28.4 GB for the merge) and tokenizing it. Later runs reuse the HF cache,
so put `HF_HOME` on a large shared disk. If you run out of GPU memory at micro-batch 8,
use `MICRO_BATCH_SIZE=4 GRAD_ACCUM=16`, which keeps the same global batch.

## 4. Outputs

All outputs go under `OUT_DIR` (default `./artifacts`):
- `artifacts/nanotron_checkpoints/<RUN_NAME>/<step>/`: checkpoints, saved every
  `CHECKPOINT_INTERVAL` steps and at the end.
- `artifacts/checkpoint_evals/<RUN_NAME>/`: at each checkpoint, the run converts it to HF
  format and runs next-token loss (DCLM + FineWeb-Edu test docs) and MMLU college-biology
  MCQA. These evals are non-fatal: if one fails, training continues.
- `generated_configs/<RUN_NAME>.yaml`: the exact config used for the run.

## 5. Resume and convert

```bash
RUN_NAME=my_150m_test bash resume_last_checkpoint.sh         # DRY_RUN=1 to only write the resume config
bash convert_last_checkpoint_to_hf.sh [MODELS_ROOT]          # -> artifacts/hf_checkpoints/<RUN_NAME>
#   settings: MODEL_GLOB='*150m*' to choose which runs, SAVE_ROOT=..., NPROC_PER_NODE=1
```

## Slurm

The scripts don't submit jobs themselves. Run them inside an allocation, e.g.
`srun --gres=gpu:4 --cpus-per-task=32 bash train_nanotron_dclm_150m.sh`. The first run on
a new dataset downloads it from HF and processes it with `NUM_PROC_PER_PROCESS=16`
workers, so give it enough CPUs and disk for the HF cache (`HF_HOME`).
