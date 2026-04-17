# Examples

This directory contains small runnable examples for `FAIR-J`.

Included files:

- `run_tiny.sh` — an end-to-end example command using the unified pipeline entry point
- `tiny_dataset.csv` — the same tiny fixture as a CSV example for custom-schema or alternate-format testing

The canonical tiny fixture now lives in `data/`.

Default expected dataset columns:

- `id`
- `context`
- `candidate`
- optional `metadata`

Supported input formats:

- `jsonl`
- `csv`
- `parquet`

If your dataset uses different column names, pass them explicitly:

```bash
fair-j run-pipeline \
  --judge-model openrouter/judge-model \
  --paraphrase-model openrouter/paraphrase-model \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --dataset-path path/to/data.parquet \
  --rubric-path data/tiny_rubric.json \
  --id-column example_id \
  --context-column source_text \
  --candidate-column summary_text
```

Minimal end-to-end example:

```bash
./examples/run_tiny.sh
```
