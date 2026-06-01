from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import permutation_test, tstd, ttest_rel, wilcoxon
from statsmodels.stats.power import TTestPower

from fair_j.score_grouping import perturbation_group_name

PAIRED_T_TEST_ALPHA = 0.05
PAIRED_T_TEST_POWER = 0.80
_TTEST_POWER_ANALYSIS = TTestPower()


def compute_std_seed_default(
    criterion_means: dict[str, dict[int, float]],
) -> float | None:
    baseline_by_seed = criterion_means.get("baseline", {})
    return raw_std(list(baseline_by_seed.values()))


def compute_std_perturbations(
    criterion_means: dict[str, dict[int, float]],
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
    return mean_or_none(per_seed_stds)


def compute_std_for_group(
    criterion_means: dict[str, dict[int, float]],
    group_name: str,
) -> float | None:
    values: list[float] = []
    for perturbation, perturbation_scores in criterion_means.items():
        if perturbation_group_name(perturbation) == group_name:
            values.extend(perturbation_scores.values())
    return raw_std(values)


def compute_std_total(
    criterion_means: dict[str, dict[int, float]],
) -> float | None:
    values: list[float] = []
    for perturbation_scores in criterion_means.values():
        values.extend(perturbation_scores.values())
    return raw_std(values)


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
    aggregated = aggregate_comparison_by_example(group_comparison)
    baseline = aggregated["baseline"]
    perturbation = aggregated["perturbation"]
    differences = aggregated["differences"]
    n_unique_examples = aggregated["n_unique_examples"]
    n_raw_pairs = aggregated["n_raw_pairs"]
    mean_difference = mean_or_none(differences)
    effect_size_dz = paired_effect_size_dz(differences)
    mde_effect_size_dz = paired_mde_effect_size_dz(n_unique_examples)
    mde_mean_difference = scale_effect_size_to_mean_difference(mde_effect_size_dz, differences)
    return {
        "tested_quantity": "difference_vs_zero",
        "alternative": "two_sided",
        "aggregation_method": "mean_difference_per_example",
        "n_pairs": n_unique_examples,
        "n_unique_examples": n_unique_examples,
        "n_raw_pairs": n_raw_pairs,
        "mean_difference": mean_difference,
        "mde_mean_difference": mde_mean_difference,
        "effect_size_dz": effect_size_dz,
        "mde_effect_size_dz": mde_effect_size_dz,
        "observed_abs_mean_difference_over_mde": ratio_or_none(
            abs(mean_difference) if mean_difference is not None else None,
            mde_mean_difference,
        ),
        "alpha": PAIRED_T_TEST_ALPHA,
        "target_power": PAIRED_T_TEST_POWER,
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


def aggregate_comparison_by_example(group_comparison: dict[str, object]) -> dict[str, object]:
    pairs_by_example = group_comparison.get("pairs_by_example", {})
    if not pairs_by_example:
        baseline = [float(value) for value in group_comparison.get("baseline", [])]
        perturbation = [float(value) for value in group_comparison.get("perturbation", [])]
        differences = [float(value) for value in group_comparison.get("differences", [])]
        unique_example_ids = set(group_comparison.get("example_ids", []))
        if not unique_example_ids and differences:
            unique_example_ids = {str(index) for index, _value in enumerate(differences)}
        return {
            "baseline": baseline,
            "perturbation": perturbation,
            "differences": differences,
            "n_unique_examples": len(unique_example_ids) if unique_example_ids else len(differences),
            "n_raw_pairs": len(differences),
        }

    baseline_values: list[float] = []
    perturbation_values: list[float] = []
    difference_values: list[float] = []
    for example_id in sorted(pairs_by_example):
        example_pairs = pairs_by_example[example_id]
        baseline_scores = [float(value) for value in example_pairs.get("baseline", [])]
        perturbation_scores = [float(value) for value in example_pairs.get("perturbation", [])]
        difference_scores = [float(value) for value in example_pairs.get("differences", [])]
        if not difference_scores:
            continue
        baseline_mean = mean_or_none(baseline_scores)
        perturbation_mean = mean_or_none(perturbation_scores)
        difference_mean = mean_or_none(difference_scores)
        if baseline_mean is None or perturbation_mean is None or difference_mean is None:
            continue
        baseline_values.append(baseline_mean)
        perturbation_values.append(perturbation_mean)
        difference_values.append(difference_mean)

    return {
        "baseline": baseline_values,
        "perturbation": perturbation_values,
        "differences": difference_values,
        "n_unique_examples": len(difference_values),
        "n_raw_pairs": len(group_comparison.get("differences", [])),
    }


def paired_permutation_test(differences: list[float]) -> float | None:
    if len(differences) < 2:
        return None

    values = np.asarray(differences, dtype=float)
    n_resamples = np.inf if len(values) <= 20 else 1000
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
    if len(differences) < 2:
        return None
    values = np.asarray(differences, dtype=float)
    if np.allclose(values, 0.0):
        return 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return None
    p_value = float(result.pvalue)
    if p_value != p_value:
        return None
    return p_value


def paired_t_test(baseline: list[float], perturbation: list[float]) -> float | None:
    if len(baseline) < 2 or len(perturbation) < 2:
        return None
    baseline_values = np.asarray(baseline, dtype=float)
    perturbation_values = np.asarray(perturbation, dtype=float)
    differences = baseline_values - perturbation_values
    if np.allclose(differences, 0.0):
        return 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ttest_rel(baseline_values, perturbation_values, alternative="two-sided")
    p_value = float(result.pvalue)
    if p_value != p_value:
        return None
    return p_value


def mean_statistic(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(values, axis=axis)


def raw_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    if np.allclose(array, array[0]):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(tstd(array))


def bias_to_mae_ratio(differences: list[float]) -> float | None:
    return ratio_or_none(mean_or_none(differences), mean_absolute_or_none(differences))


def paired_effect_size_dz(differences: list[float]) -> float | None:
    std_difference = raw_std(differences)
    if std_difference in (None, 0):
        return None
    mean_difference = mean_or_none(differences)
    if mean_difference is None:
        return None
    return mean_difference / std_difference


def paired_mde_effect_size_dz(
    n_pairs: int,
    alpha: float = PAIRED_T_TEST_ALPHA,
    power: float = PAIRED_T_TEST_POWER,
) -> float | None:
    if n_pairs < 2:
        return None
    result = _TTEST_POWER_ANALYSIS.solve_power(
        effect_size=None,
        nobs=n_pairs,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    return float(np.asarray(result).item())


def scale_effect_size_to_mean_difference(
    effect_size: float | None,
    differences: list[float],
) -> float | None:
    std_difference = raw_std(differences)
    if effect_size is None or std_difference is None:
        return None
    return abs(effect_size) * std_difference


def ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def compute_score_scale_span(scale_min: int | float, scale_max: int | float) -> float | None:
    span = float(scale_max) - float(scale_min)
    if span <= 0:
        return None
    return span


def normalize_by_scale(value: float | None, score_scale_span: float | None) -> float | None:
    if value is None or score_scale_span is None:
        return None
    return value / score_scale_span
