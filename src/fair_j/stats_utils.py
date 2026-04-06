from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.stats import permutation_test, ttest_rel, wilcoxon

from fair_j.score_grouping import perturbation_group_name


def compute_std_seed_default(
    criterion_means: dict[str, dict[int, float]],
    n_examples: int,
) -> float | None:
    baseline_by_seed = criterion_means.get("baseline", {})
    return normalized_std(list(baseline_by_seed.values()), n_examples)


def compute_std_perturbations(
    criterion_means: dict[str, dict[int, float]],
    n_examples: int,
) -> float | None:
    seed_to_values: dict[int, list[float]] = {}
    for perturbation_scores in criterion_means.values():
        for seed, value in perturbation_scores.items():
            seed_to_values.setdefault(seed, []).append(value)

    per_seed_stds = [
        std_value
        for values in seed_to_values.values()
        if (std_value := raw_std(values)) is not None
    ]
    return normalize_uncertainty(mean_or_none(per_seed_stds), n_examples)


def compute_std_for_group(
    criterion_means: dict[str, dict[int, float]],
    group_name: str,
    n_examples: int,
) -> float | None:
    values: list[float] = []
    for perturbation, perturbation_scores in criterion_means.items():
        if perturbation_group_name(perturbation) == group_name:
            values.extend(perturbation_scores.values())
    return normalized_std(values, n_examples)


def compute_std_total(
    criterion_means: dict[str, dict[int, float]],
    n_examples: int,
) -> float | None:
    values: list[float] = []
    for perturbation_scores in criterion_means.values():
        values.extend(perturbation_scores.values())
    return normalized_std(values, n_examples)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def mean_absolute_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(abs(value) for value in values) / len(values)


def min_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return min(values)


def max_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values)


def build_stat_test_block(group_comparison: dict[str, list[float]]) -> dict[str, object]:
    baseline = group_comparison["baseline"]
    perturbation = group_comparison["perturbation"]
    differences = group_comparison["differences"]
    return {
        "tested_quantity": "difference_vs_zero",
        "alternative": "two_sided",
        "n_pairs": len(differences),
        "tests": {
            "paired_permutation_test": {
                "p_value": paired_permutation_test(differences),
            },
            "wilcoxon_signed_rank": {
                "p_value": wilcoxon_signed_rank_test(differences),
            },
            "paired_t_test": {
                "p_value": paired_t_test(baseline, perturbation),
            },
        },
    }


def paired_permutation_test(differences: list[float]) -> float | None:
    if not differences:
        return None

    values = np.asarray(differences, dtype=float)
    n_resamples = np.inf if len(values) <= 20 else 10000
    result = permutation_test(
        data=(values,),
        statistic=mean_statistic,
        permutation_type="samples",
        alternative="two-sided",
        n_resamples=n_resamples,
        random_state=0,
        vectorized=True,
    )
    return float(result.pvalue)


def wilcoxon_signed_rank_test(differences: list[float]) -> float | None:
    if len(differences) < 1:
        return None
    try:
        result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return None
    p_value = float(result.pvalue)
    if p_value != p_value:
        return None
    return p_value


def paired_t_test(baseline: list[float], perturbation: list[float]) -> float | None:
    if len(baseline) < 2 or len(perturbation) < 2:
        return None
    result = ttest_rel(baseline, perturbation, alternative="two-sided")
    p_value = float(result.pvalue)
    if p_value != p_value:
        return None
    return p_value


def mean_statistic(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(values, axis=axis)


def normalized_std(values: list[float], n_examples: int) -> float | None:
    return normalize_uncertainty(raw_std(values), n_examples)


def raw_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance)


def normalize_uncertainty(value: float | None, n_examples: int) -> float | None:
    if value is None or n_examples <= 1:
        return None
    return value / sqrt(n_examples - 1)


def compute_score_scale_span(scale_min: int | float, scale_max: int | float) -> float | None:
    span = float(scale_max) - float(scale_min)
    if span <= 0:
        return None
    return span


def normalize_by_scale(value: float | None, score_scale_span: float | None) -> float | None:
    if value is None or score_scale_span is None:
        return None
    return value / score_scale_span
