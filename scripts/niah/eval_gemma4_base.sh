#!/bin/bash
set -euo pipefail

MODEL_PATH="${GEMMA4_MODEL_PATH:-google/gemma-4-E4B-it}"

WANDB_MODE=disabled uv run run_eval.py \
 --model_name_or_path "${MODEL_PATH}" \
 --datasets ctx_magic_number_32_1024 ctx_magic_number_1024_2048 ctx_magic_number_2048_3072 ctx_magic_number_3072_4096 \
 --split test \
 --eval_batch_size_gen=4 \
 "$@"
