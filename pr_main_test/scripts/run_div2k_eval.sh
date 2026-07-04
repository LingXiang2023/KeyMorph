#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${HIDING_TEST_DATA_DIR:-data/div2k/test}"
CHECKPOINT="${HIDING_CHECKPOINT:-model_zoo/default/checkpoint.pt}"
OUTPUT_DIR="${HIDING_OUTPUT_DIR:-results/div2k_test}"
CUDA_DEVICES="${HIDING_CUDA_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

python test.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --cuda-devices "$CUDA_DEVICES" \
  --batch-size "${HIDING_BATCH_SIZE:-2}" \
  --resize "${HIDING_RESIZE:-512}" \
  --sigma "${HIDING_SIGMA:-20}" \
  --num-workers "${HIDING_NUM_WORKERS:-0}" \
  --no-save-images
