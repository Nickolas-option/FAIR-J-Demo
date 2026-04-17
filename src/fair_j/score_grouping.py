from __future__ import annotations

from itertools import combinations

from scipy.stats import kendalltau

from fair_j.schemas import ScoreLogRow


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


def collect_grouped_comparisons(
    score_rows: list[ScoreLogRow],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    baseline_scores: dict[tuple[str, str, int], float] = {}
    grouped_comparisons: dict[str, dict[str, dict[str, list[float]]]] = {}

    for row in score_rows:
        if row.perturbation == "baseline":
            key = (row.example_id, row.criterion_id, row.seed)
            baseline_scores[key] = float(row.score)

    for row in score_rows:
        if row.perturbation == "baseline":
            continue

        group_name = perturbation_group_name(row.perturbation)
        if group_name is None:
            continue

        key = (row.example_id, row.criterion_id, row.seed)
        baseline_score = baseline_scores.get(key)
        if baseline_score is None:
            continue

        difference = baseline_score - float(row.score)
        criterion_groups = grouped_comparisons.setdefault(
            row.criterion_id,
            {
                "paraphrases": empty_group_comparison(),
                "deletions": empty_group_comparison(),
            },
        )
        criterion_groups[group_name]["baseline"].append(baseline_score)
        criterion_groups[group_name]["perturbation"].append(float(row.score))
        criterion_groups[group_name]["differences"].append(difference)

    return grouped_comparisons


def collect_grouped_rank_metrics(
    criterion_scores: dict[str, dict[int, dict[str, float]]],
) -> dict[str, dict[str, list[float]]]:
    baseline_by_seed = criterion_scores.get("baseline", {})
    grouped_metrics: dict[str, dict[str, list[float]]] = {
        "paraphrases": {
            "tau_b": [],
            "tie_pct": [],
            "concordant_pairs": [],
            "discordant_pairs": [],
        },
        "deletions": {
            "tau_b": [],
            "tie_pct": [],
            "concordant_pairs": [],
            "discordant_pairs": [],
        },
    }

    for perturbation, perturbation_by_seed in criterion_scores.items():
        group_name = perturbation_group_name(perturbation)
        if group_name is None:
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
            result = kendalltau(baseline_values, perturbation_values)
            tau = float(result.statistic)
            if tau != tau:
                continue
            pair_counts = count_pair_relationships(baseline_values, perturbation_values)
            grouped_metrics[group_name]["tau_b"].append(tau)
            grouped_metrics[group_name]["tie_pct"].append(pair_counts["tie_pct"])
            grouped_metrics[group_name]["concordant_pairs"].append(pair_counts["concordant_pairs"])
            grouped_metrics[group_name]["discordant_pairs"].append(pair_counts["discordant_pairs"])

    return grouped_metrics


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


def empty_group_comparison() -> dict[str, list[float]]:
    return {
        "baseline": [],
        "perturbation": [],
        "differences": [],
    }


def perturbation_group_name(perturbation: str) -> str | None:
    if perturbation.startswith("paraphrase__"):
        return "paraphrases"
    if perturbation.startswith("delete__"):
        return "deletions"
    return None
