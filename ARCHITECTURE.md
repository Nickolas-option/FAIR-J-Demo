# FAIR-J Architecture

## Table Of Contents

- [Overview](#overview)
- [Repository Map](#repository-map)
- [Phase 1: Run The Judge](#phase-1-run-the-judge)
- [Phase 2: Evaluate The Judge](#phase-2-evaluate-the-judge)
- [Run Artifacts](#run-artifacts)

## Overview

`FAIR-J` is organized around two phases:

1. `Run the judge`: build rubric perturbations, call the judge model, validate the structured outputs, and write run artifacts.
2. `Evaluate the judge`: read completed run artifacts, compute robustness metrics, and render human-readable reports.

This split keeps the execution path simple and resumable, while keeping the analysis path deterministic and easy to iterate on.

At a high level:

- `AdapterInput`, `DatasetExample`, `Rubric`, and `RubricVariant` define the inputs needed to execute judge calls.
- `RunMetadata`, `CallLogRow`, and `ScoreLogRow` define the artifacts produced by a run.
- `CoreOutput` defines the final evaluation payload consumed by reporting.

## Repository Map

- `src/` - installable project source tree.
- `src/fair_j/` - package code for the CLI, judge execution, evaluation, stats, and HTML reporting.
- `data/` - tiny committed fixture data for first-run testing only.
- `examples/` - runnable example assets, including a shell script and an alternate CSV input example.
- `runs/` - generated outputs; the repo keeps only an example HTML report, while full run artifacts stay out of git.
- `summeval_conversion/` - legacy benchmark-specific conversion helper for SummEval raw data.
- `README.md` - quick-start usage and CLI entry points.
- `ARCHITECTURE.md` - high-level technical map of the repository.
- `limitations.md` - explicit current boundaries and known gaps in the project.

## Phase 1: Run The Judge

The first phase turns a dataset plus rubric into validated judge scores.

Main modules:

- `cli.py` - exposes `run-pipeline`, `run-adapter`, `make-variants`, `run-core`, and report rendering commands.
- `run_judge.py` - orchestrates one run: load inputs, create variants, call the judge, resume safely, and write logs.
- `openrouter_judge.py` - configures the OpenRouter client, enforces structured outputs, and validates returned scores.
- `perturbations.py` - generates baseline, paraphrase, and deletion rubric variants.
- `io_utils.py` - loads datasets and rubrics, manages run directories, and reads/writes JSON and JSONL artifacts.
- `schemas.py` - defines the typed contracts used across the run pipeline.

Key data structures:

- `AdapterInput` - minimal execution config for judge runs.
- `DatasetExample` - one example to be judged.
- `Rubric` - baseline rubric definition.
- `RubricVariant` - one perturbed rubric variant.
- `RunMetadata` - run-level config stored once per run.
- `CallLogRow` - exact prompt/output record for one model call.
- `ScoreLogRow` - validated atomic score record for one criterion.

Why this phase exists:

- It isolates all networked and non-deterministic work.
- It makes resume possible from `score_log.jsonl`.
- It keeps benchmark-specific preprocessing outside the core package.

## Phase 2: Evaluate The Judge

The second phase reads finished run artifacts and converts them into robustness metrics and reports.

Main modules:

- `evaluate_judge.py` - entry point for deterministic post-processing of completed runs.
- `score_grouping.py` - groups baseline and perturbation scores into comparison-ready structures.
- `stats_utils.py` - computes variability, paired tests, effect sizes, and ranking diagnostics.
- `report_html.py` - renders single-run and multi-run HTML summaries from analysis outputs.

Key data structures:

- `CoreOutput` - top-level evaluation result written to `core_output.json`.

Why this phase exists:

- It keeps evaluation reproducible once the raw scores are written.
- It allows report iteration without re-running judge calls.
- It separates metric definitions from provider-specific execution details.

## Run Artifacts

Each run lives in its own directory under `runs/` during local work.

Important files:

- `run.json` - run-level metadata such as models, dataset path, rubric path, and score scale.
- `variants.json` - generated rubric variants used in the run.
- `call_log.jsonl` - raw prompt/output audit log for each judge call.
- `score_log.jsonl` - validated score rows; this is the main source of truth for evaluation.
- `core_output.json` - computed metrics derived from `score_log.jsonl`.
- `report.html` - optional single-run human-readable HTML summary.

The repository intentionally does not commit full run directories. For demo purposes, we keep only an example comparison HTML file in `runs/` so a new reader can immediately see what the final output looks like.
