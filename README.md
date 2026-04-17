# FAIR-J

Minimal `V1` skeleton for a research tool that checks the reliability of `LLM-as-a-Judge`.

## Quick Start With Your Own Data

```bash
uv sync
export OPENROUTER_API_KEY=your_key_here

# 1) Create a dataset file, for example my_dataset.jsonl
cat > my_dataset.jsonl <<'EOF'
{"id":"ex_001","context":"Source text for example 1.","candidate":"Model output for example 1."}
{"id":"ex_002","context":"Source text for example 2.","candidate":"Model output for example 2."}
EOF

# 2) Create a rubric file, for example my_rubric.json
cat > my_rubric.json <<'EOF'
{
  "name": "my_rubric",
  "scale": { "min": 1, "max": 5 },
  "criteria": [
    { "id": "coherence", "text": "The answer should be easy to follow." },
    { "id": "relevance", "text": "The answer should focus on the important information." }
  ]
}
EOF

# 3) Run the full pipeline
uv run fair-j run-pipeline \
  --judge-model qwen/qwen3-8b \
  --paraphrase-model qwen/qwen3-8b \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --dataset-path my_dataset.jsonl \
  --rubric-path my_rubric.json \
  --subset-name my_first_run \
  --workers 1
```

This creates a new run in `runs/` and writes the HTML summary to `report.html` inside that run directory.

API keys:

- This repo currently sends model calls through OpenRouter.
- In practice, that means you should set `OPENROUTER_API_KEY`.
- Even if you want to use an OpenAI model or an Anthropic model, you still pass it through OpenRouter in this repo.
- `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are not used directly by the current CLI.

Example:

```bash
export OPENROUTER_API_KEY=your_openrouter_key
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
  --openrouter-api-key "$OPENROUTER_API_KEY" \
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

If your dataset uses different column names, pass them explicitly:

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
