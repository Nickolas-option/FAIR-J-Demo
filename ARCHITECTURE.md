## `AdapterInput`

`AdapterInput` is the minimal input contract for any task-specific adapter in `FAIR-J`.

The adapter does not read benchmark-specific formats directly. Instead, it takes:
- OpenRouter model identifiers,
- an OpenRouter API key,
- a path to a judge-ready dataset file,
- and a path to a rubric/checklist file.

This keeps the adapter simple and makes the tool reusable across different tasks.

### Fields

- `judge_model: str`  
  OpenRouter model name used as the judge.

- `paraphrase_model: str`  
  OpenRouter model name used to generate rubric paraphrases.

- `openrouter_api_key: str`  
  API key used for OpenRouter requests.

- `dataset_path: Path`  
  Path to a judge-ready dataset file in `JSONL` format.

- `rubric_path: Path`  
  Path to a rubric/checklist file in `JSON` format.

### Expected format for `dataset_path`

`dataset_path` must point to a `JSONL` file.

Each line must contain one evaluation example with the following minimal fields:

- `id: str`
- `context: str`
- `candidate: str`

Optional field:

- `metadata: object`

`metadata` may contain any additional task-specific fields.

#### Minimal JSONL example row

```json
{"id":"example_001","context":"Input or source content to evaluate against.","candidate":"Model output being judged.","metadata":{"source_id":"doc_17","model_id":"candidate_a"}}
```

### Expected format for `rubric_path`

`rubric_path` must point to a checklist `JSON` file with the following structure:

- `name: str`
- `scale`
  - `min: int | float`
  - `max: int | float`
- `criteria: list[object]`
  - `id: str`
  - `text: str`

#### Minimal checklist JSON example

```json
{
  "name": "rubric_v1",
  "scale": {
    "min": 1,
    "max": 5
  },
  "criteria": [
    {
      "id": "criterion_1",
      "text": "Criterion description."
    }
  ]
}
```

## `Run-level metadata`

Run-level metadata stores values that are constant for the whole run and therefore should not be duplicated in every raw row.

- `judge_model: str`  
  Lives on run-level because one run uses one judge model.

- `paraphrase_model: str`  
  Lives on run-level because one run uses one paraphrase model.

- `dataset_path: Path`  
  Lives on run-level because all raw rows come from the same dataset file.

- `rubric_path: Path`  
  Lives on run-level because all raw rows use the same rubric file.

- `subset_name: str`  
  Lives on run-level because all raw rows belong to the same subset.
  In `V1`, `subset_name` is optional at the CLI level.
  If it is not provided, the default is `dataset_path.stem`.

- `scale_min: int | float`  
  Lives on run-level because the lower bound of the score scale is fixed for the run.

- `scale_max: int | float`  
  Lives on run-level because the upper bound of the score scale is fixed for the run.

### Minimal run-level metadata example

```json
{
  "judge_model": "openai/gpt-4.1-mini",
  "paraphrase_model": "qwen/qwen2.5-7b-instruct",
  "dataset_path": "data/examples.jsonl",
  "rubric_path": "data/rubric.json",
  "subset_name": "subset_001",
  "scale_min": 1,
  "scale_max": 5
}
```

## `Run folder layout`

One run is stored in one dedicated run folder.

Inside that folder, `V1` uses fixed, explicit file names:

- `run.json`
- `variants.json`
- `call_log.jsonl`
- `score_log.jsonl`
- `core_output.json`

### Why these files exist

- `run.json` stores run-level metadata.
- `variants.json` stores the generated perturbation variants and is required.
- `call_log.jsonl` stores raw calls for reproducibility and debugging.
- `score_log.jsonl` stores validated score rows and is the source of truth for metrics.
- `core_output.json` stores the analysis results produced by `core`.

### Resume rule

Resume must rely on `score_log.jsonl`.

A call is considered completed only if it has already produced valid score rows in `score_log.jsonl`.

If `--run-dir` is not provided, `V1` should auto-create:

`runs/<timestamp>__<judge_model_slug>__<subset_name>__<scale_min>-<scale_max>`

## `Call log`

`Call log` is required.

It is the source of truth for reproducibility and debugging. It stores the exact prompt sent to the model and the raw model output returned by the model.

To support an exact linkage between logs, each call log entry must have a `call_id`.
There is no bijection between individual rows of `call log` and `score log`.
The relationship is `one call log row -> many score log rows`, and it is recovered through `call_id`.

The adapter must use OpenRouter Structured Outputs (`response_format` + `json_schema`) together with `Instructor` and a typed schema. Parsing and validation are adapter responsibilities.

### Retry policy

The adapter should retry structured-output calls up to `10` times.

If a call still cannot be validated after all retries, the run should fail.
Because resume relies on `score_log.jsonl`, the run must still be resumable from the same point later.

In `V1`, the requested number of `seed` repeats applies to all perturbations, not only to baseline.

### Minimal call log example

```json
{
  "call_id": "call_000001",
  "example_id": "example_001",
  "perturbation": "baseline",
  "seed": 0,
  "prompt_text": "Judge this candidate output using the rubric.",
  "raw_model_output": "{\"criterion_1\": 4}"
}
```

## `Score log`

`Score log` stores validated atomic score observations and is the source of truth for metrics.

- `call_id: str`
- `example_id: str`
- `criterion_id: str`
- `perturbation: str`
- `seed: int | str`
- `score: int | float`
- `metadata: object` (optional)

### Notes

- `call_id` links each score row to the exact call log entry it was parsed from.
- The linkage is one-to-many: one call may produce multiple score rows.
- There is no row-level bijection between `call log` and `score log`.
- `seed` is the axis of repeated measurements under the same condition, even if it is not literally a controllable API seed.
- `perturbation` stores a single string identifier such as:
  - `baseline`
  - `paraphrase__criterion_1__01`
  - `delete__fluency`

This keeps the score schema minimal. Perturbation type or group can be recovered later by parsing this field.
Every score row points back to exactly one call log entry through `call_id`.

For a minimal project, `call_id` should be assigned as a deterministic sequential identifier within the run, for example `call_000001`, `call_000002`, and so on.

### Minimal score log example

```json
{
  "call_id": "call_000001",
  "example_id": "example_001",
  "criterion_id": "criterion_1",
  "perturbation": "baseline",
  "seed": 0,
  "score": 4,
  "metadata": {
    "provider": "openrouter"
  }
}
```

## `CoreInput`

`CoreInput` is the minimal contract for the analysis layer in `FAIR-J`.

The `core` reads:
- run-level metadata
- score log

The score log is the main input for metrics.

The call log is a required system artifact, but it is not the main input for metrics.
`core` uses `run-level metadata + score log` for metrics.
The call log is available for audit, debugging, and reproducibility.

`core` does not analyze parsing quality. Structured output parsing and validation are completed by the adapter before score rows are written.

Any tensor-like representation or pivoted table is optional and may be built internally only as a derived view for computation.

### Required score log fields used by `core`

- `call_id`
- `example_id`
- `criterion_id`
- `perturbation`
- `seed`
- `score`

### Baseline matching rule

For a fixed run, `core` matches each non-baseline row to its baseline row(s) using the same:
- `example_id`
- `criterion_id`
- `seed`

where the baseline row is defined by:
- `perturbation = "baseline"`

For statistical tests, `core` builds paired differences on the same examples:

`delta = baseline_score - perturbation_score`

This matching is always done from the score log. Tensor-like or pivoted representations, if used, are only internal derived views built after this canonical linkage is defined.

## `CoreOutput`

`CoreOutput` is the minimal output contract for the analysis layer in `FAIR-J`.

In `V1`, `CoreOutput` is `per_criterion` only.
It does not include:
- `sum_over_criteria`
- `rank_difference_analysis`
- summary / verdict layer

The minimal output shape is:

```json
{
  "run_metadata": {
    "judge_model": "string",
    "paraphrase_model": "string",
    "dataset_path": "string",
    "rubric_path": "string",
    "subset_name": "string",
    "scale_min": "number",
    "scale_max": "number"
  },
  "dataset_level_scores": {
    "criterion_id_1": {
      "std_seed_default": "number",
      "std_perturbations": "number",
      "std_paraphrases": "number",
      "std_deletions": "number",
      "std_total": "number",
      "bias_paraphrases": "number",
      "mad_paraphrases": "number",
      "bias_deletions": "number",
      "mad_deletions": "number",
      "stat_tests": {
        "tested_quantity": "difference_vs_zero",
        "alternative": "two_sided",
        "n_pairs": "integer",
        "tests": {
          "paired_permutation_test": {
            "p_value": "number"
          },
          "wilcoxon_signed_rank": {
            "p_value": "number"
          },
          "paired_t_test": {
            "p_value": "number"
          }
        }
      }
    }
  },
  "ranking_consistency": {
    "criterion_id_1": {
      "kendall_paraphrases_mean": "number",
      "kendall_paraphrases_std": "number",
      "kendall_paraphrases_min": "number",
      "kendall_paraphrases_max": "number",
      "kendall_deletions_mean": "number",
      "kendall_deletions_std": "number",
      "kendall_deletions_min": "number",
      "kendall_deletions_max": "number"
    }
  }
}
```

### `dataset_level_scores`

This block stores dataset-level stability metrics for each criterion.

It contains:
- standard deviations across seeds and perturbations
- group-level standard deviations for paraphrases and deletions
- `bias` and `MAD`
- a dedicated `stat_tests` sub-block

By default, uncertainty estimates based on dataset-level `std_*` metrics should be normalized by `sqrt(n_examples - 1)`.
If the real target dataset size is larger than the current subset, they may instead be normalized by `sqrt(n_dataset_size - 1)`.
If `V1` does not yet implement dataset-size-aware normalization, that part should be treated as deferred.

### `stat_tests`

This block stores raw statistical test results for paired, two-sided tests of:

`delta = baseline_score - perturbation_score`

against zero on the same examples.

It does not store a final significance verdict in `V1`.

### `ranking_consistency`

This block stores ranking stability metrics for each criterion using Kendall-based comparisons between baseline and perturbations.

In `V1`, it stores only grouped paraphrase/deletion aggregates:
- mean
- std
- min
- max
