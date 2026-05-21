from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AdapterInput:
    judge_model: str
    paraphrase_model: str
    dataset_path: Path
    rubric_path: Path
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    dataset_id_column: str = "id"
    dataset_context_column: str = "context"
    dataset_candidate_column: str = "candidate"
    provider_only: str | None = None
    provider_quantization: str | None = None
    request_timeout_seconds: float = 90.0


@dataclass
class DatasetExample:
    id: str
    context: str
    candidate: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Criterion:
    id: str
    text: str


@dataclass
class Rubric:
    name: str
    scale_min: int | float
    scale_max: int | float
    criteria: list[Criterion]


@dataclass
class RubricVariant:
    perturbation: str
    rubric_name: str
    criteria: list[Criterion]


@dataclass
class RunMetadata:
    judge_model: str
    paraphrase_model: str
    dataset_path: str
    rubric_path: str
    subset_name: str
    scale_min: int | float
    scale_max: int | float
    seeds: int = 1
    paraphrases_per_criterion: int = 1
    limit_examples: int | None = None
    dataset_format: str | None = None
    dataset_id_column: str = "id"
    dataset_context_column: str = "context"
    dataset_candidate_column: str = "candidate"


@dataclass
class CallLogRow:
    call_id: str
    example_id: str
    perturbation: str
    seed: int
    prompt_text: str
    raw_model_output: str


@dataclass
class ScoreLogRow:
    call_id: str
    example_id: str
    criterion_id: str
    perturbation: str
    seed: int
    score: int | float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreOutput:
    run_metadata: RunMetadata
    dataset_level_scores: dict[str, Any]
    ranking_consistency: dict[str, Any]
    per_example_seed_std: dict[str, Any]
