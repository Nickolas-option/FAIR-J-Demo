# FAIR-J

Minimal `V1` skeleton for a research tool that checks the reliability of `LLM-as-a-Judge`.

## Quick Start With Your Own Data

1. Install dependencies:

```bash
uv sync
```

2. Set the API keys you need:

```bash
export OPENROUTER_API_KEY=your_openrouter_key

# Optional: only needed when the selected model uses the `openai/` prefix.
# export OPENAI_API_KEY=your_openai_key

# Optional: only needed when the selected model uses the `anthropic/` prefix.
# export ANTHROPIC_API_KEY=your_anthropic_key
```

3. Prepare a dataset file and a rubric file.

Supported dataset formats:

- `jsonl`
- `csv`
- `parquet`
- `json` (array of objects)

Default dataset columns:

- `id`
- `context`
- `candidate`

Datasets may contain additional columns. `fair-j` only requires these three mapped fields and keeps any extra non-empty columns as example metadata.

If your dataset uses different column names, pass them explicitly in the CLI:

```bash
uv run fair-j run-pipeline \
  --judge-model qwen/qwen3-8b \
  --paraphrase-model qwen/qwen3-8b \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --dataset-path my_dataset.csv \
  --rubric-path my_rubric.json \
  --id-column example_id \
  --context-column source_text \
  --candidate-column summary_text
```

Example dataset as `JSONL`:

```json
{"id":"ex_001","context":"Source text for example 1.","candidate":"Model output for example 1."}
{"id":"ex_002","context":"Source text for example 2.","candidate":"Model output for example 2."}
```

Example dataset as `JSON`:

```json
[
  {
    "id": "ex_001",
    "context": "Source text for example 1.",
    "candidate": "Model output for example 1."
  },
  {
    "id": "ex_002",
    "context": "Source text for example 2.",
    "candidate": "Model output for example 2."
  }
]
```

Example dataset as `CSV`:

```csv
id,context,candidate
ex_001,"Source text for example 1.","Model output for example 1."
ex_002,"Source text for example 2.","Model output for example 2."
```

You can also pass a `parquet` dataset file as `--dataset-path my_dataset.parquet` as long as it contains the same mapped fields.

Example rubric as `JSON`:

```json
{
  "name": "my_rubric",
  "scale": { "min": 1, "max": 5 },
  "criteria": [
    { "id": "coherence", "text": "The answer should be easy to follow." },
    { "id": "relevance", "text": "The answer should focus on the important information." }
  ]
}
```

4. Run the full pipeline.

Required arguments only:

```bash
uv run fair-j run-pipeline \
  --judge-model qwen/qwen3-8b \
  --paraphrase-model qwen/qwen3-8b \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --dataset-path my_dataset.jsonl \
  --rubric-path my_rubric.json
```

Optional arguments:

```bash
uv run fair-j run-pipeline \
  --judge-model qwen/qwen3-8b \
  --paraphrase-model qwen/qwen3-8b \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --dataset-path my_dataset.jsonl \
  --rubric-path my_rubric.json \
  \
  # Optional: only for `openai/...` judge or paraphrase models.
  # --openai-api-key "$OPENAI_API_KEY" \
  \
  # Optional: only for `anthropic/...` judge or paraphrase models.
  # --anthropic-api-key "$ANTHROPIC_API_KEY" \
  \
  # Optional: defaults to dataset_path.stem
  # --subset-name my_first_run \
  \
  # Optional: defaults to auto-generated path inside runs/
  # --run-dir runs/my_first_run \
  \
  # Optional: defaults to 1
  # --seeds 3 \
  \
  # Optional: defaults to 25
  # --workers 1 \
  \
  # Optional: defaults to 90.0
  # --request-timeout 120 \
  \
  # Optional: only if your dataset uses custom column names
  # --id-column example_id \
  # --context-column source_text \
  # --candidate-column summary_text \
  \
  # Optional: OpenRouter provider controls
  # --provider deepinfra \
  # --provider-quantization fp8 \
  \
  # Optional: render report to a custom file instead of run_dir/report.html
  # --html-output-path runs/my_report.html \
  \
  # Optional: custom HTML page title
  # --html-title "My FAIR-J Report" \
  \
  # Optional: disable adapter progress bar
  # --no-progress
```

This creates a new run in `runs/` and writes the HTML summary to `report.html` inside that run directory.

API keys:

- OpenRouter-routed models use `OPENROUTER_API_KEY`.
- Models with the `openai/` prefix use `OPENAI_API_KEY`.
- Models with the `anthropic/` prefix use `ANTHROPIC_API_KEY`.
- You only need to provide the keys required by the models you actually chose.

Example:

```bash
export OPENROUTER_API_KEY=your_openrouter_key

# Optional, if needed by your model choice:
# export OPENAI_API_KEY=your_openai_key
# export ANTHROPIC_API_KEY=your_anthropic_key
```

How to write model names:

- Use the OpenRouter model id in `--judge-model` and `--paraphrase-model`.
- Format: `<provider>/<model>`
- OpenAI examples: `openai/gpt-4.1-mini`, `openai/gpt-5-nano`
- Anthropic examples: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3.7-sonnet`
- Qwen examples: `qwen/qwen3-8b`, `qwen/qwen3-14b`

Example:

```bash
uv run fair-j run-pipeline \
  --judge-model openai/gpt-4.1-mini \
  --paraphrase-model anthropic/claude-3.5-sonnet \
  --openai-api-key "$OPENAI_API_KEY" \
  --anthropic-api-key "$ANTHROPIC_API_KEY" \
  --dataset-path my_dataset.jsonl \
  --rubric-path my_rubric.json
```

Expected dataset columns:

- `id`
- `context`
- `candidate`

Supported dataset formats:

- `jsonl`
- `csv`
- `parquet`
- `json` (array of objects)

## Commands

Run the full evaluation pipeline in one command:

```bash
fair-j run-pipeline \
  --judge-model openrouter/judge-model \
  --paraphrase-model openrouter/paraphrase-model \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --dataset-path path/to/data.jsonl \
  --rubric-path path/to/rubric.json \
  --workers 25
```

Create rubric variants:

```bash
fair-j make-variants \
  --rubric-path path/to/rubric.json \
  --paraphrase-model openrouter/model \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --output-path path/to/variants.json
```

Prepare a run directory and adapter artifacts:

```bash
fair-j run-adapter \
  --judge-model nvidia/nemotron-3-nano-30b-a3b \
  --paraphrase-model openrouter/paraphrase-model \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --dataset-path path/to/data.jsonl \
  --rubric-path path/to/rubric.json \
  --workers 10 \
  --seeds 5
```

Run core analysis:

```bash
fair-j run-core \
  --run-dir runs/my_run
```

Render a comparison HTML report across completed runs:

```bash
fair-j render-compare-html \
  --run-dir runs/run_a \
  --run-dir runs/run_b \
  --output-path runs/comparison.html
```

## Notes

- `--subset-name` is optional. Default: `dataset_path.stem`
- `--run-dir` is optional for `run-adapter`. Default:
  `runs/<timestamp>__<judge_model_slug>__<subset_name>__<scale_min>-<scale_max>`
- `run-pipeline` is the recommended unified CLI entry point. It runs `run-adapter`, then `run-core`, then renders a single-run HTML report.
- `run-evaluation` is kept as a backwards-compatible alias.
- `run-pipeline` writes the HTML report to `run_dir / "report.html"` by default.
- In `V1`, `--seeds` means the number of repeats for all perturbations.
- `run-adapter` shows a progress bar by default. Use `--no-progress` to disable it.
- `judge_model` values use the OpenRouter-backed judge path through `Instructor` and an OpenAI-compatible client.
- `--workers` is optional for `run-adapter`, `run-evaluation`, and `run-pipeline`. Default: `25`.
- `paraphrase_model` values use a real OpenRouter-backed paraphrase path.
- Supported dataset formats: `jsonl`, `csv`, `parquet`.
- Default dataset columns are `id`, `context`, `candidate`. Use `--id-column`, `--context-column`, and `--candidate-column` for custom schemas.
- Runnable examples live in [`examples/README.md`](examples/README.md).
- The repo keeps only tiny fixture data in `data/` and one example HTML report in `runs/`.
