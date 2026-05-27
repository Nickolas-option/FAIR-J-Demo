# `fair_j` module map

`fair_j` is split into two phases:

- Judge run: prepare rubric variants, call the judge, validate outputs, and write run artifacts.
- Judge evaluation: read completed run artifacts, compute robustness metrics, and render reports.

One-line guide to each file:

- `__init__.py` - package version and package-level exports.
- `cli.py` - command-line entry points for running the pipeline and rendering reports.
- `run_judge.py` - execution stage that creates variants, calls the judge, logs outputs, and supports resume.
- `evaluate_judge.py` - analysis stage that reads score logs and computes robustness metrics.
- `openrouter_judge.py` - OpenRouter client setup and structured judge-call logic.
- `perturbations.py` - rubric perturbation generation, including paraphrases and deletions.
- `schemas.py` - typed dataclasses for datasets, rubrics, logs, and analysis outputs.
- `io_utils.py` - JSON/JSONL/dataset I/O plus run-directory helpers.
- `score_grouping.py` - grouping helpers for baseline-vs-perturbation comparisons and ranking metrics.
- `stats_utils.py` - statistical utilities used by the evaluation stage.
- `report_html.py` - static HTML report generation from precomputed `core_output.json` files.
