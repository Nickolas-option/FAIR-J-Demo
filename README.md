# FAIR-J

Minimal `V1` skeleton for a research tool that checks the reliability of `LLM-as-a-Judge`.

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
