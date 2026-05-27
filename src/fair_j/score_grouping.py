from __future__ import annotations

from itertools import combinations

from scipy.stats import kendalltau

from fair_j.schemas import ScoreLogRow

PARAPHRASES_GROUP = "paraphrases"
PARAPHRASES_ALL_GROUP = "paraphrases_all"
PARAPHRASES_SAME_CRITERION_GROUP = "paraphrases_same_criterion"
PARAPHRASES_CROSS_CRITERION_GROUP = "paraphrases_cross_criterion"
DELETIONS_GROUP = "deletions"


def compute_dataset_level_means(
    score_rows: list[ScoreLogRow],
) -> dict[str, dict[str, dict[int, float]]]:
    grouped_scores: dict[str, dict[str, dict[int, list[float]]]] = {}

    for row in score_rows:
        criterion_scores = grouped_scores.setdefault(row.criterion_id, {})
        perturbation_scores = criterion_scores.setdefault(row.perturbation, {})
        seed_scores = perturbation_scores.setdefault(row.seed, [])
        seed_scores.append(float(row.score))

    result: dict[str, dict[str, dict[int, float]]] = {}
    for criterion_id, criterion_scores in grouped_scores.items():
        result[criterion_id] = {}
        for perturbation, perturbation_scores in criterion_scores.items():
            result[criterion_id][perturbation] = {}
            for seed, scores in perturbation_scores.items():
                result[criterion_id][perturbation][seed] = sum(scores) / len(scores)
    return result


def group_scores_by_condition(
    score_rows: list[ScoreLogRow],
) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    grouped: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for row in score_rows:
        criterion_scores = grouped.setdefault(row.criterion_id, {})
        perturbation_scores = criterion_scores.setdefault(row.perturbation, {})
        seed_scores = perturbation_scores.setdefault(row.seed, {})
        seed_scores[row.example_id] = float(row.score)
    return grouped


def count_unique_examples_per_criterion(score_rows: list[ScoreLogRow]) -> dict[str, int]:
    grouped_examples: dict[str, set[str]] = {}
    for row in score_rows:
        grouped_examples.setdefault(row.criterion_id, set()).add(row.example_id)
    return {
        criterion_id: len(example_ids)
        for criterion_id, example_ids in grouped_examples.items()
    }


def parse_perturbation(perturbation: str) -> dict[str, str | None]:
    if perturbation == "baseline":
        return {"kind": "baseline", "criterion_id": None}

    parts = perturbation.split("__")
    if perturbation.startswith("paraphrase__") and len(parts) >= 2:
        return {"kind": "paraphrase", "criterion_id": parts[1]}
    if perturbation.startswith("delete__") and len(parts) >= 2:
        return {"kind": "deletion", "criterion_id": parts[1]}
    return {"kind": "other", "criterion_id": None}


def collect_grouped_comparisons(
    score_rows: list[ScoreLogRow],
) -> dict[str, dict[str, dict[str, object]]]:
    baseline_scores: dict[tuple[str, str, int], float] = {}
    grouped_comparisons: dict[str, dict[str, dict[str, object]]] = {}

    for row in score_rows:
        if row.perturbation == "baseline":
            key = (row.example_id, row.criterion_id, row.seed)
            baseline_scores[key] = float(row.score)

    for row in score_rows:
        if row.perturbation == "baseline":
            continue

        parsed = parse_perturbation(row.perturbation)
        kind = parsed["kind"]
        if kind not in ("paraphrase", "deletion"):
            continue

        key = (row.example_id, row.criterion_id, row.seed)
        baseline_score = baseline_scores.get(key)
        if baseline_score is None:
            continue

        difference = baseline_score - float(row.score)
        criterion_groups = grouped_comparisons.setdefault(
            row.criterion_id,
            {
                PARAPHRASES_ALL_GROUP: empty_group_comparison(),
                PARAPHRASES_SAME_CRITERION_GROUP: empty_group_comparison(),
                PARAPHRASES_CROSS_CRITERION_GROUP: empty_group_comparison(),
                DELETIONS_GROUP: empty_group_comparison(),
            },
        )

        if kind == "paraphrase":
            append_group_comparison(
                criterion_groups[PARAPHRASES_ALL_GROUP],
                example_id=row.example_id,
                baseline_score=baseline_score,
                perturbation_score=float(row.score),
                difference=difference,
            )
            target_criterion_id = parsed["criterion_id"]
            paraphrase_group = (
                PARAPHRASES_SAME_CRITERION_GROUP
                if target_criterion_id == row.criterion_id
                else PARAPHRASES_CROSS_CRITERION_GROUP
            )
            append_group_comparison(
                criterion_groups[paraphrase_group],
                example_id=row.example_id,
                baseline_score=baseline_score,
                perturbation_score=float(row.score),
                difference=difference,
            )
            continue

        append_group_comparison(
            criterion_groups[DELETIONS_GROUP],
            example_id=row.example_id,
            baseline_score=baseline_score,
            perturbation_score=float(row.score),
            difference=difference,
        )

    return grouped_comparisons


def append_group_comparison(
    group_comparison: dict[str, object],
    *,
    example_id: str,
    baseline_score: float,
    perturbation_score: float,
    difference: float,
) -> None:
    group_comparison["baseline"].append(baseline_score)
    group_comparison["perturbation"].append(perturbation_score)
    group_comparison["differences"].append(difference)
    group_comparison["example_ids"].append(example_id)
    pairs_by_example = group_comparison["pairs_by_example"]
    example_entry = pairs_by_example.setdefault(
        example_id,
        {
            "baseline": [],
            "perturbation": [],
            "differences": [],
        },
    )
    example_entry["baseline"].append(baseline_score)
    example_entry["perturbation"].append(perturbation_score)
    example_entry["differences"].append(difference)


def collect_grouped_rank_metrics(
    criterion_scores: dict[str, dict[int, dict[str, float]]],
    criterion_id: str,
) -> dict[str, dict[str, list[float]]]:
    baseline_by_seed = criterion_scores.get("baseline", {})
    grouped_metrics: dict[str, dict[str, list[float]]] = {
        PARAPHRASES_GROUP: empty_group_rank_metrics(),
        PARAPHRASES_ALL_GROUP: empty_group_rank_metrics(),
        PARAPHRASES_SAME_CRITERION_GROUP: empty_group_rank_metrics(),
        PARAPHRASES_CROSS_CRITERION_GROUP: empty_group_rank_metrics(),
        DELETIONS_GROUP: empty_group_rank_metrics(),
    }

    for perturbation, perturbation_by_seed in criterion_scores.items():
        group_names = rank_metric_group_names(perturbation, criterion_id)
        if not group_names:
            continue

        for seed, perturbation_scores in perturbation_by_seed.items():
            baseline_scores = baseline_by_seed.get(seed)
            if baseline_scores is None:
                continue

            shared_example_ids = sorted(set(baseline_scores) & set(perturbation_scores))
            if len(shared_example_ids) < 2:
                continue

            baseline_values = [baseline_scores[example_id] for example_id in shared_example_ids]
            perturbation_values = [
                perturbation_scores[example_id]
                for example_id in shared_example_ids
            ]
            pair_counts = count_pair_relationships(baseline_values, perturbation_values)
            result = kendalltau(baseline_values, perturbation_values)
            tau = float(result.statistic)
            for group_name in group_names:
                grouped_metrics[group_name]["tie_pct"].append(pair_counts["tie_pct"])
                grouped_metrics[group_name]["concordant_pairs"].append(pair_counts["concordant_pairs"])
                grouped_metrics[group_name]["discordant_pairs"].append(pair_counts["discordant_pairs"])
                if tau == tau:
                    grouped_metrics[group_name]["tau_b"].append(tau)

    return grouped_metrics


def rank_metric_group_names(perturbation: str, criterion_id: str) -> list[str]:
    parsed = parse_perturbation(perturbation)
    if parsed["kind"] == "paraphrase":
        group_names = [PARAPHRASES_GROUP, PARAPHRASES_ALL_GROUP]
        if parsed["criterion_id"] == criterion_id:
            group_names.append(PARAPHRASES_SAME_CRITERION_GROUP)
        else:
            group_names.append(PARAPHRASES_CROSS_CRITERION_GROUP)
        return group_names
    if parsed["kind"] == "deletion":
        return [DELETIONS_GROUP]
    return []


def empty_group_rank_metrics() -> dict[str, list[float]]:
    return {
        "tau_b": [],
        "tie_pct": [],
        "concordant_pairs": [],
        "discordant_pairs": [],
    }


def count_pair_relationships(
    baseline_values: list[float],
    perturbation_values: list[float],
) -> dict[str, float | None]:
    total_pairs = len(baseline_values) * (len(baseline_values) - 1) // 2
    concordant = 0
    discordant = 0
    tied_pairs = 0

    for left_index, right_index in combinations(range(len(baseline_values)), 2):
        baseline_delta = baseline_values[left_index] - baseline_values[right_index]
        perturbation_delta = perturbation_values[left_index] - perturbation_values[right_index]

        if baseline_delta == 0 or perturbation_delta == 0:
            tied_pairs += 1
            continue

        if baseline_delta * perturbation_delta > 0:
            concordant += 1
        elif baseline_delta * perturbation_delta < 0:
            discordant += 1

    tie_pct = 0.0
    if total_pairs > 0:
        tie_pct = 100 * tied_pairs / total_pairs

    return {
        "concordant_pairs": float(concordant),
        "discordant_pairs": float(discordant),
        "tie_pct": tie_pct,
    }


def empty_group_comparison() -> dict[str, object]:
    return {
        "baseline": [],
        "perturbation": [],
        "differences": [],
        "example_ids": [],
        "pairs_by_example": {},
    }


def perturbation_group_name(perturbation: str) -> str | None:
    parsed = parse_perturbation(perturbation)
    if parsed["kind"] == "paraphrase":
        return PARAPHRASES_GROUP
    if parsed["kind"] == "deletion":
        return DELETIONS_GROUP
    return None
