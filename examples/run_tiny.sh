#!/usr/bin/env bash
set -euo pipefail

: "${OPENROUTER_API_KEY:?Please set OPENROUTER_API_KEY before running this example.}"

fair-j run-pipeline \
  --judge-model qwen/qwen3-8b \
  --paraphrase-model qwen/qwen3-8b \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --dataset-path examples/tiny_dataset.jsonl \
  --rubric-path examples/tiny_rubric.json \
  --subset-name tiny_example \
  --workers 1
