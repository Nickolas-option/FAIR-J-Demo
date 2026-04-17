# Examples

This directory contains small runnable examples for `FAIR-J`.

Included files:

- `tiny_dataset.jsonl` — minimal judge-ready dataset in the default format
- `tiny_dataset.csv` — the same tiny dataset in CSV format
- `tiny_rubric.json` — a minimal 2-criterion rubric
- `run_tiny.sh` — an end-to-end example command using the unified pipeline entry point

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
  --rubric-path examples/tiny_rubric.json \
  --id-column example_id \
  --context-column source_text \
  --candidate-column summary_text
```

Minimal end-to-end example:

```bash
./examples/run_tiny.sh
```
