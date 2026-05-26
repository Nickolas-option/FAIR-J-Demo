from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

from fair_j.io_utils import (
    append_jsonl,
    infer_dataset_format,
    load_dataset_examples,
    load_rubric,
    load_variants,
    read_jsonl,
    write_json,
)
from fair_j.openrouter_judge import call_openrouter_judge
from fair_j.perturbations import make_variants
from fair_j.schemas import (
    AdapterInput,
    CallLogRow,
    DatasetExample,
    RubricVariant,
    RunMetadata,
    ScoreLogRow,
)


@dataclass(frozen=True)
class PendingCall:
    call_id: str
    example: DatasetExample
    variant: RubricVariant
    seed: int
    prompt_text: str


def run_judge(
    adapter_input: AdapterInput,
    run_dir: Path,
    subset_name: str,
    seeds: int,
    paraphrases_per_criterion: int | None = None,
    variants_path: Path | None = None,
    workers: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
    limit: int | None = None,
) -> dict[str, int | str]:
    if paraphrases_per_criterion is not None and paraphrases_per_criterion < 1:
        raise SystemExit("--paraphrases-per-criterion must be at least 1.")

    dataset = load_dataset_examples(
        adapter_input.dataset_path,
        id_column=adapter_input.dataset_id_column,
        context_column=adapter_input.dataset_context_column,
        candidate_column=adapter_input.dataset_candidate_column,
        limit=limit,
    )
    rubric = load_rubric(adapter_input.rubric_path)

    run_variants_path = run_dir / "variants.json"
    if variants_path is not None:
        if not variants_path.exists():
            raise SystemExit(f"--variants-path does not exist: {variants_path}")
        variants = load_variants(variants_path)
        if run_variants_path.exists():
            existing = load_variants(run_variants_path)
            if variant_signatures(existing) != variant_signatures(variants):
                raise SystemExit(
                    "run_dir already has a different variants.json. "
                    "Use a new --run-dir or provide a matching --variants-path."
                )
        else:
            write_json(run_variants_path, variants)
    elif run_variants_path.exists():
        variants = load_variants(run_variants_path)
    else:
        paraphrases_for_generation = paraphrases_per_criterion or 1
        variants = make_variants(
            rubric=rubric,
            paraphrase_model=adapter_input.paraphrase_model,
            openrouter_api_key=adapter_input.openrouter_api_key,
            openai_api_key=adapter_input.openai_api_key,
            anthropic_api_key=adapter_input.anthropic_api_key,
            paraphrases_per_criterion=paraphrases_for_generation,
            bedrock_region=adapter_input.bedrock_region,
        )
        write_json(run_variants_path, variants)

    inferred_paraphrases_per_criterion = infer_paraphrases_per_criterion(variants)
    if (
        paraphrases_per_criterion is not None
        and inferred_paraphrases_per_criterion != paraphrases_per_criterion
    ):
        raise SystemExit(
            "Provided --paraphrases-per-criterion does not match variants. "
            f"Provided={paraphrases_per_criterion}, inferred={inferred_paraphrases_per_criterion}."
        )

    run_metadata = RunMetadata(
        judge_model=adapter_input.judge_model,
        paraphrase_model=adapter_input.paraphrase_model,
        dataset_path=str(adapter_input.dataset_path),
        rubric_path=str(adapter_input.rubric_path),
        subset_name=subset_name,
        scale_min=rubric.scale_min,
        scale_max=rubric.scale_max,
        seeds=seeds,
        paraphrases_per_criterion=inferred_paraphrases_per_criterion,
        limit_examples=limit,
        dataset_format=infer_dataset_format(adapter_input.dataset_path),
        dataset_id_column=adapter_input.dataset_id_column,
        dataset_context_column=adapter_input.dataset_context_column,
        dataset_candidate_column=adapter_input.dataset_candidate_column,
    )
    write_json(run_dir / "run.json", run_metadata)

    call_log_path = run_dir / "call_log.jsonl"
    score_log_path = run_dir / "score_log.jsonl"
    call_log_path.touch(exist_ok=True)
    score_log_path.touch(exist_ok=True)

    completed = resume_state_from_score_log(run_dir, variants)
    next_call_number = get_next_call_number(run_dir)
    pending_calls = collect_pending_calls(
        dataset=dataset,
        variants=variants,
        seeds=seeds,
        completed=completed,
        scale_min=rubric.scale_min,
        scale_max=rubric.scale_max,
        next_call_number=next_call_number,
    )
    total_calls = len(dataset) * len(variants) * seeds

    skipped_calls = total_calls - len(pending_calls)
    report_progress(progress_callback, len(completed), total_calls)
    written_calls = run_pending_calls(
        adapter_input=adapter_input,
        pending_calls=pending_calls,
        call_log_path=call_log_path,
        score_log_path=score_log_path,
        scale_min=rubric.scale_min,
        scale_max=rubric.scale_max,
        workers=workers,
        progress_callback=progress_callback,
        initial_completed=len(completed),
        total_calls=total_calls,
    )

    return {
        "examples": len(dataset),
        "variants": len(variants),
        "seeds": seeds,
        "completed_calls": len(completed),
        "written_calls": written_calls,
        "skipped_calls": skipped_calls,
        "run_dir": str(run_dir),
    }


# Backwards-compatible alias for older imports.
run_adapter = run_judge


def collect_pending_calls(
    dataset: list[DatasetExample],
    variants: list[RubricVariant],
    seeds: int,
    completed: set[tuple[str, str, int]],
    scale_min: int | float,
    scale_max: int | float,
    next_call_number: int,
) -> list[PendingCall]:
    pending_calls: list[PendingCall] = []
    call_number = next_call_number
    for example in dataset:
        for variant in variants:
            for seed in range(seeds):
                call_key = (example.id, variant.perturbation, seed)
                if call_key in completed:
                    continue

                pending_calls.append(
                    PendingCall(
                        call_id=format_call_id(call_number),
                        example=example,
                        variant=variant,
                        seed=seed,
                        prompt_text=build_judge_prompt(
                            example=example,
                            variant=variant,
                            scale_min=scale_min,
                            scale_max=scale_max,
                        ),
                    )
                )
                call_number += 1
    return pending_calls


def get_pending_call_scores(
    adapter_input: AdapterInput,
    pending_call: PendingCall,
    scale_min: int | float,
    scale_max: int | float,
) -> tuple[dict[str, int | float], str]:
    expected_ids = {criterion.id for criterion in pending_call.variant.criteria}
    return call_openrouter_judge(
        adapter_input=adapter_input,
        judge_model=adapter_input.judge_model,
        expected_ids=expected_ids,
        scale_min=scale_min,
        scale_max=scale_max,
        prompt_text=pending_call.prompt_text,
    )


def write_completed_call(
    call_log_path: Path,
    score_log_path: Path,
    pending_call: PendingCall,
    scores_and_output: tuple[dict[str, int | float], str],
) -> None:
    scores, raw_model_output = scores_and_output
    call_row = CallLogRow(
        call_id=pending_call.call_id,
        example_id=pending_call.example.id,
        perturbation=pending_call.variant.perturbation,
        seed=pending_call.seed,
        prompt_text=pending_call.prompt_text,
        raw_model_output=raw_model_output,
    )
    score_rows = [
        ScoreLogRow(
            call_id=pending_call.call_id,
            example_id=pending_call.example.id,
            criterion_id=criterion.id,
            perturbation=pending_call.variant.perturbation,
            seed=pending_call.seed,
            score=scores[criterion.id],
        )
        for criterion in pending_call.variant.criteria
    ]

    append_jsonl(call_log_path, [call_row])
    append_jsonl(score_log_path, score_rows)


def run_pending_calls(
    adapter_input: AdapterInput,
    pending_calls: list[PendingCall],
    call_log_path: Path,
    score_log_path: Path,
    scale_min: int | float,
    scale_max: int | float,
    workers: int,
    progress_callback: Callable[[int, int], None] | None,
    initial_completed: int,
    total_calls: int,
) -> int:
    completed_futures = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_call = {}
        for pending_call in pending_calls:
            report_call_event("started", pending_call)
            future = executor.submit(
                get_pending_call_scores,
                adapter_input=adapter_input,
                pending_call=pending_call,
                scale_min=scale_min,
                scale_max=scale_max,
            )
            future_to_call[future] = pending_call

        for future in as_completed(future_to_call):
            pending_call = future_to_call[future]
            try:
                scores_and_output = future.result()
            except Exception:
                report_call_event("failed", pending_call)
                raise

            completed_futures += 1
            write_completed_call(
                call_log_path=call_log_path,
                score_log_path=score_log_path,
                pending_call=pending_call,
                scores_and_output=scores_and_output,
            )
            report_call_event("completed", pending_call)
            report_progress(progress_callback, initial_completed + completed_futures, total_calls)

    return completed_futures


def report_progress(
    progress_callback: Callable[[int, int], None] | None,
    current: int,
    total: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(current, total)


def resume_state_from_score_log(
    run_dir: Path,
    variants: list[RubricVariant],
) -> set[tuple[str, str, int]]:
    score_rows = read_jsonl(run_dir / "score_log.jsonl")
    expected_counts = {
        variant.perturbation: len(variant.criteria)
        for variant in variants
    }
    counts: dict[tuple[str, str, int], int] = {}
    for row in score_rows:
        key = (row["example_id"], row["perturbation"], int(row["seed"]))
        counts[key] = counts.get(key, 0) + 1

    return {
        key
        for key, count in counts.items()
        if count >= expected_counts.get(key[1], 0)
    }


def get_next_call_number(run_dir: Path) -> int:
    call_rows = read_jsonl(run_dir / "call_log.jsonl")
    return len(call_rows) + 1


def format_call_id(call_number: int) -> str:
    return f"call_{call_number:06d}"


def build_judge_prompt(
    example: DatasetExample,
    variant: RubricVariant,
    scale_min: int | float,
    scale_max: int | float,
) -> str:
    criteria_lines = "\n".join(f"- {criterion.id}: {criterion.text}" for criterion in variant.criteria)
    return (
        f"Context:\n{example.context}\n\n"
        f"Candidate:\n{example.candidate}\n\n"
        f"Score each criterion on the scale [{scale_min}, {scale_max}].\n"
        f"Criteria:\n{criteria_lines}\n\n"
        "Return structured output with one numeric score for each criterion id."
    )


def report_call_event(status: str, pending_call: PendingCall) -> None:
    sys.stderr.write(f"\n[{status}] {format_pending_call_label(pending_call)}\n")
    sys.stderr.flush()


def format_pending_call_label(pending_call: PendingCall) -> str:
    return (
        f"example_id={pending_call.example.id} "
        f"perturbation={pending_call.variant.perturbation} "
        f"seed={pending_call.seed}"
    )


def variant_signatures(variants: list[RubricVariant]) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    return [
        (
            variant.perturbation,
            tuple((criterion.id, criterion.text) for criterion in variant.criteria),
        )
        for variant in variants
    ]


PARAPHRASE_PERTURBATION_PATTERN = re.compile(r"^paraphrase__.+__(\d+)$")


def infer_paraphrases_per_criterion(variants: list[RubricVariant]) -> int:
    counts_by_criterion: dict[str, int] = {}
    for variant in variants:
        perturbation = variant.perturbation
        if not perturbation.startswith("paraphrase__"):
            continue
        parts = perturbation.split("__")
        if len(parts) != 3:
            continue
        if not PARAPHRASE_PERTURBATION_PATTERN.match(perturbation):
            continue
        criterion_id = parts[1]
        counts_by_criterion[criterion_id] = counts_by_criterion.get(criterion_id, 0) + 1

    if not counts_by_criterion:
        return 0

    unique_counts = set(counts_by_criterion.values())
    if len(unique_counts) != 1:
        details = ", ".join(
            f"{criterion_id}:{count}"
            for criterion_id, count in sorted(counts_by_criterion.items())
        )
        raise SystemExit(
            "Inconsistent paraphrase counts across criteria in variants file: "
            f"{details}."
        )
    return unique_counts.pop()
