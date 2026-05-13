from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fair_j.schemas import (
    Criterion,
    DatasetExample,
    Rubric,
    RubricVariant,
    RunMetadata,
    ScoreLogRow,
)


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


def default_subset_name(dataset_path: Path) -> str:
    return dataset_path.stem


def build_default_run_dir(
    base_dir: Path,
    judge_model: str,
    subset_name: str,
    scale_min: int | float,
    scale_max: int | float,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_slug = slugify_model_name(judge_model)
    run_name = f"{timestamp}__{model_slug}__{subset_name}__{scale_min}-{scale_max}"
    return base_dir / run_name


def create_run_dir(base_dir: Path, run_name: str) -> Path:
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, data: Any) -> None:
    serializable = _to_serializable(data)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]] | list[Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_serializable(row), ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_dataset_format(dataset_path: Path) -> str:
    name = dataset_path.name.lower()
    suffixes = [suffix.lower() for suffix in dataset_path.suffixes]
    if name.endswith(".jsonl"):
        return "jsonl"
    if suffixes and suffixes[-1] == ".csv":
        return "csv"
    if suffixes and suffixes[-1] in {".parquet", ".pq"}:
        return "parquet"
    if suffixes and suffixes[-1] == ".json":
        return "json"
    raise ValueError(
        f"Unsupported dataset format for {dataset_path}. Expected one of: .jsonl, .csv, .parquet, .json."
    )


def load_dataset_examples(
    dataset_path: Path,
    *,
    id_column: str = "id",
    context_column: str = "context",
    candidate_column: str = "candidate",
    limit: int | None = None,
) -> list[DatasetExample]:
    rows = read_dataset_rows(dataset_path)
    if limit is not None:
        rows = rows[:limit]
    return [
        build_dataset_example(
            row,
            id_column=id_column,
            context_column=context_column,
            candidate_column=candidate_column,
        )
        for row in rows
    ]


def read_dataset_rows(dataset_path: Path) -> list[dict[str, Any]]:
    dataset_format = infer_dataset_format(dataset_path)
    if dataset_format == "jsonl":
        return read_jsonl(dataset_path)
    if dataset_format == "csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if dataset_format == "parquet":
        table = pq.read_table(dataset_path)
        return table.to_pylist()
    if dataset_format == "json":
        data = read_json(dataset_path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array dataset in {dataset_path}.")
        return [dict(row) for row in data]
    raise ValueError(f"Unsupported dataset format for {dataset_path}.")


def build_dataset_example(
    row: dict[str, Any],
    *,
    id_column: str,
    context_column: str,
    candidate_column: str,
) -> DatasetExample:
    missing = [
        column
        for column in (id_column, context_column, candidate_column)
        if column not in row
    ]
    if missing:
        available = ", ".join(sorted(row.keys()))
        raise ValueError(
            "Dataset row is missing required columns "
            f"{missing}. Available columns: [{available}]"
        )

    metadata = normalize_metadata(row.get("metadata"))
    for key, value in row.items():
        if key in {id_column, context_column, candidate_column, "metadata"}:
            continue
        if value in (None, ""):
            continue
        metadata[key] = value

    return DatasetExample(
        id=str(row[id_column]),
        context=str(row[context_column]),
        candidate=str(row[candidate_column]),
        metadata=metadata,
    )


def normalize_metadata(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {"raw_metadata": value}
        if isinstance(loaded, dict):
            return loaded
        return {"raw_metadata": loaded}
    return {"raw_metadata": value}


def load_rubric(rubric_path: Path) -> Rubric:
    data = read_json(rubric_path)
    criteria = [Criterion(**item) for item in data["criteria"]]
    return Rubric(
        name=data["name"],
        scale_min=data["scale"]["min"],
        scale_max=data["scale"]["max"],
        criteria=criteria,
    )


def load_variants(variants_path: Path) -> list[RubricVariant]:
    data = read_json(variants_path)
    return [
        RubricVariant(
            perturbation=item["perturbation"],
            rubric_name=item["rubric_name"],
            criteria=[Criterion(**criterion) for criterion in item["criteria"]],
        )
        for item in data
    ]


def read_run_artifacts(run_dir: Path) -> tuple[RunMetadata, list[ScoreLogRow]]:
    run_data = read_json(run_dir / "run.json")
    score_rows_data = read_jsonl(run_dir / "score_log.jsonl")
    run_metadata = RunMetadata(**run_data)
    score_rows = [ScoreLogRow(**row) for row in score_rows_data]
    return run_metadata, score_rows


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    return value
