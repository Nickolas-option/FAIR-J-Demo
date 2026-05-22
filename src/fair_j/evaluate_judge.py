from __future__ import annotations

from pathlib import Path

from fair_j.io_utils import read_run_artifacts, write_json
from fair_j.schemas import CoreOutput, ScoreLogRow
from fair_j.score_grouping import (
    DELETIONS_GROUP,
    PARAPHRASES_ALL_GROUP,
    PARAPHRASES_CROSS_CRITERION_GROUP,
    PARAPHRASES_SAME_CRITERION_GROUP,
    collect_grouped_comparisons,
    collect_grouped_rank_metrics,
    compute_dataset_level_means,
    empty_group_comparison,
    group_scores_by_condition,
    parse_perturbation,
)
from fair_j.stats_utils import (
    bias_to_mae_ratio,
    build_stat_test_block,
    compute_score_scale_span,
    compute_std_for_group,
    compute_std_perturbations,
    compute_std_seed_default,
    compute_std_total,
    max_or_none,
    mean_absolute_or_none,
    mean_or_none,
    min_or_none,
    normalize_by_scale,
    raw_std,
)

TOTAL_CRITERION_ID = "__total__"


def evaluate_judge(run_dir: Path) -> CoreOutput:
    run_metadata, score_rows = read_run_artifacts(run_dir)
    analysis_score_rows = score_rows + build_total_score_rows(score_rows)
    score_scale_span = compute_score_scale_span(run_metadata.scale_min, run_metadata.scale_max)
    output = CoreOutput(
        run_metadata=run_metadata,
        dataset_level_scores=compute_dataset_level_scores(analysis_score_rows, score_scale_span),
        ranking_consistency=compute_ranking_consistency(analysis_score_rows),
        per_example_seed_std=compute_per_example_seed_std(analysis_score_rows, score_scale_span),
    )
    write_json(run_dir / "core_output.json", output)
    return output


# Backwards-compatible alias for older imports.
run_core = evaluate_judge


def compute_dataset_level_scores(
    score_rows: list[ScoreLogRow],
    score_scale_span: float | None,
) -> dict[str, object]:
    grouped_comparisons = collect_grouped_comparisons(score_rows)
    dataset_level_means = compute_dataset_level_means(score_rows)
    criterion_ids = sorted({row.criterion_id for row in score_rows})
    results: dict[str, object] = {}

    for criterion_id in criterion_ids:
        paraphrase_all_comparison = grouped_comparisons.get(criterion_id, {}).get(
            PARAPHRASES_ALL_GROUP,
            empty_group_comparison(),
        )
        paraphrase_same_comparison = grouped_comparisons.get(criterion_id, {}).get(
            PARAPHRASES_SAME_CRITERION_GROUP,
            empty_group_comparison(),
        )
        paraphrase_cross_comparison = grouped_comparisons.get(criterion_id, {}).get(
            PARAPHRASES_CROSS_CRITERION_GROUP,
            empty_group_comparison(),
        )
        deletion_comparison = grouped_comparisons.get(criterion_id, {}).get(
            DELETIONS_GROUP,
            empty_group_comparison(),
        )
        paraphrase_all_diffs = paraphrase_all_comparison["differences"]
        paraphrase_same_diffs = paraphrase_same_comparison["differences"]
        paraphrase_cross_diffs = paraphrase_cross_comparison["differences"]
        deletion_diffs = deletion_comparison["differences"]
        criterion_means = dataset_level_means.get(criterion_id, {})
        baseline_scores = collect_baseline_scores(score_rows, criterion_id)
        criterion_score_scale_span = compute_criterion_score_scale_span(
            criterion_id,
            score_rows,
            score_scale_span,
        )
        results[criterion_id] = {
            "baseline_score_std": raw_std(baseline_scores),
            "std_seed_default": compute_std_seed_default(criterion_means),
            "std_perturbations": compute_std_perturbations(criterion_means),
            "std_paraphrases": compute_std_for_group(criterion_means, "paraphrases"),
            "std_paraphrases_all": compute_std_for_group(criterion_means, "paraphrases"),
            "std_paraphrases_same_criterion": compute_std_for_paraphrase_scope(
                criterion_means,
                criterion_id,
                PARAPHRASES_SAME_CRITERION_GROUP,
            ),
            "std_paraphrases_cross_criterion": compute_std_for_paraphrase_scope(
                criterion_means,
                criterion_id,
                PARAPHRASES_CROSS_CRITERION_GROUP,
            ),
            "std_deletions": compute_std_for_group(criterion_means, "deletions"),
            "std_total": compute_std_total(criterion_means),
            "bias_paraphrases": mean_or_none(paraphrase_all_diffs),
            "bias_paraphrases_all": mean_or_none(paraphrase_all_diffs),
            "bias_paraphrases_same_criterion": mean_or_none(paraphrase_same_diffs),
            "bias_paraphrases_cross_criterion": mean_or_none(paraphrase_cross_diffs),
            "bias_to_mae_ratio_paraphrases": bias_to_mae_ratio(paraphrase_all_diffs),
            "bias_to_mae_ratio_paraphrases_all": bias_to_mae_ratio(paraphrase_all_diffs),
            "bias_to_mae_ratio_paraphrases_same_criterion": bias_to_mae_ratio(paraphrase_same_diffs),
            "bias_to_mae_ratio_paraphrases_cross_criterion": bias_to_mae_ratio(paraphrase_cross_diffs),
            "bias_paraphrases_scale_normalized": normalize_by_scale(
                mean_or_none(paraphrase_all_diffs),
                criterion_score_scale_span,
            ),
            "bias_paraphrases_all_scale_normalized": normalize_by_scale(
                mean_or_none(paraphrase_all_diffs),
                criterion_score_scale_span,
            ),
            "bias_paraphrases_same_criterion_scale_normalized": normalize_by_scale(
                mean_or_none(paraphrase_same_diffs),
                criterion_score_scale_span,
            ),
            "bias_paraphrases_cross_criterion_scale_normalized": normalize_by_scale(
                mean_or_none(paraphrase_cross_diffs),
                criterion_score_scale_span,
            ),
            "mad_paraphrases": mean_absolute_or_none(paraphrase_all_diffs),
            "mad_paraphrases_all": mean_absolute_or_none(paraphrase_all_diffs),
            "mad_paraphrases_same_criterion": mean_absolute_or_none(paraphrase_same_diffs),
            "mad_paraphrases_cross_criterion": mean_absolute_or_none(paraphrase_cross_diffs),
            "mad_paraphrases_scale_normalized": normalize_by_scale(
                mean_absolute_or_none(paraphrase_all_diffs),
                criterion_score_scale_span,
            ),
            "mad_paraphrases_all_scale_normalized": normalize_by_scale(
                mean_absolute_or_none(paraphrase_all_diffs),
                criterion_score_scale_span,
            ),
            "mad_paraphrases_same_criterion_scale_normalized": normalize_by_scale(
                mean_absolute_or_none(paraphrase_same_diffs),
                criterion_score_scale_span,
            ),
            "mad_paraphrases_cross_criterion_scale_normalized": normalize_by_scale(
                mean_absolute_or_none(paraphrase_cross_diffs),
                criterion_score_scale_span,
            ),
            "bias_deletions": mean_or_none(deletion_diffs),
            "bias_to_mae_ratio_deletions": bias_to_mae_ratio(deletion_diffs),
            "bias_deletions_scale_normalized": normalize_by_scale(
                mean_or_none(deletion_diffs),
                criterion_score_scale_span,
            ),
            "mad_deletions": mean_absolute_or_none(deletion_diffs),
            "mad_deletions_scale_normalized": normalize_by_scale(
                mean_absolute_or_none(deletion_diffs),
                criterion_score_scale_span,
            ),
            "score_scale_span": criterion_score_scale_span,
            "stat_tests": {
                "paraphrases_vs_baseline": build_stat_test_block(paraphrase_all_comparison),
                "paraphrases_all_vs_baseline": build_stat_test_block(paraphrase_all_comparison),
                "paraphrases_same_criterion_vs_baseline": build_stat_test_block(
                    paraphrase_same_comparison
                ),
                "paraphrases_cross_criterion_vs_baseline": build_stat_test_block(
                    paraphrase_cross_comparison
                ),
                "deletions_vs_baseline": build_stat_test_block(deletion_comparison),
            },
        }

    return results


def collect_baseline_scores(score_rows: list[ScoreLogRow], criterion_id: str) -> list[float]:
    return [
        float(row.score)
        for row in score_rows
        if row.criterion_id == criterion_id and row.perturbation == "baseline"
    ]


def compute_ranking_consistency(score_rows: list[ScoreLogRow]) -> dict[str, object]:
    grouped_scores = group_scores_by_condition(score_rows)
    criterion_ids = sorted(grouped_scores.keys())
    results: dict[str, object] = {}

    for criterion_id in criterion_ids:
        rank_metrics = collect_grouped_rank_metrics(grouped_scores[criterion_id])
        paraphrases = rank_metrics.get("paraphrases", {})
        deletions = rank_metrics.get("deletions", {})
        results[criterion_id] = {
            "kendall_paraphrases_mean": mean_or_none(paraphrases.get("tau_b", [])),
            "kendall_paraphrases_std": raw_std(paraphrases.get("tau_b", [])),
            "kendall_paraphrases_min": min_or_none(paraphrases.get("tau_b", [])),
            "kendall_paraphrases_max": max_or_none(paraphrases.get("tau_b", [])),
            "concordant_paraphrases_pooled": pooled_pair_count(paraphrases, "concordant_pairs"),
            "discordant_paraphrases_pooled": pooled_pair_count(paraphrases, "discordant_pairs"),
            "kendall_deletions_mean": mean_or_none(deletions.get("tau_b", [])),
            "kendall_deletions_std": raw_std(deletions.get("tau_b", [])),
            "kendall_deletions_min": min_or_none(deletions.get("tau_b", [])),
            "kendall_deletions_max": max_or_none(deletions.get("tau_b", [])),
            "concordant_deletions_pooled": pooled_pair_count(deletions, "concordant_pairs"),
            "discordant_deletions_pooled": pooled_pair_count(deletions, "discordant_pairs"),
            "gamma_paraphrases_pooled": compute_pooled_gamma(
                paraphrases.get("concordant_pairs", []),
                paraphrases.get("discordant_pairs", []),
            ),
            "gamma_deletions_pooled": compute_pooled_gamma(
                deletions.get("concordant_pairs", []),
                deletions.get("discordant_pairs", []),
            ),
            "tie_pct_paraphrases_mean": mean_or_none(paraphrases.get("tie_pct", [])),
            "tie_pct_paraphrases_std": raw_std(paraphrases.get("tie_pct", [])),
            "tie_pct_paraphrases_min": min_or_none(paraphrases.get("tie_pct", [])),
            "tie_pct_paraphrases_max": max_or_none(paraphrases.get("tie_pct", [])),
            "tie_pct_deletions_mean": mean_or_none(deletions.get("tie_pct", [])),
            "tie_pct_deletions_std": raw_std(deletions.get("tie_pct", [])),
            "tie_pct_deletions_min": min_or_none(deletions.get("tie_pct", [])),
            "tie_pct_deletions_max": max_or_none(deletions.get("tie_pct", [])),
        }

    return results


def compute_std_for_paraphrase_scope(
    criterion_means: dict[str, dict[int, float]],
    criterion_id: str,
    scope: str,
) -> float | None:
    values: list[float] = []
    for perturbation, perturbation_scores in criterion_means.items():
        parsed = parse_perturbation(perturbation)
        if parsed["kind"] != "paraphrase":
            continue
        target_criterion_id = parsed["criterion_id"]
        if scope == PARAPHRASES_SAME_CRITERION_GROUP and target_criterion_id != criterion_id:
            continue
        if scope == PARAPHRASES_CROSS_CRITERION_GROUP and target_criterion_id == criterion_id:
            continue
        values.extend(perturbation_scores.values())
    return raw_std(values)


def compute_pooled_gamma(
    concordant_pairs: list[float],
    discordant_pairs: list[float],
) -> float | None:
    total_concordant = sum(concordant_pairs)
    total_discordant = sum(discordant_pairs)
    total_comparable = total_concordant + total_discordant
    if total_comparable == 0:
        return None
    return (total_concordant - total_discordant) / total_comparable


def pooled_pair_count(rank_metrics: dict[str, list[float]], key: str) -> int:
    return int(round(sum(rank_metrics.get(key, []))))


def build_total_score_rows(score_rows: list[ScoreLogRow]) -> list[ScoreLogRow]:
    grouped_scores: dict[tuple[str, str, int, str], dict[str, float]] = {}
    grouped_call_ids: dict[tuple[str, str, int, str], str] = {}
    baseline_scores: dict[tuple[str, int], dict[str, float]] = {}

    for row in score_rows:
        if row.criterion_id == TOTAL_CRITERION_ID:
            continue
        key = (row.example_id, row.perturbation, row.seed, row.call_id)
        grouped_scores.setdefault(key, {})[row.criterion_id] = float(row.score)
        grouped_call_ids[key] = row.call_id
        if row.perturbation == "baseline":
            baseline_key = (row.example_id, row.seed)
            baseline_scores.setdefault(baseline_key, {})[row.criterion_id] = float(row.score)

    total_rows: list[ScoreLogRow] = []
    for (example_id, perturbation, seed, call_id), criterion_scores in grouped_scores.items():
        total_score = compute_comparable_total_score(
            criterion_scores=criterion_scores,
            baseline_scores=baseline_scores.get((example_id, seed), {}),
        )
        total_rows.append(
            ScoreLogRow(
                call_id=grouped_call_ids[(example_id, perturbation, seed, call_id)],
                example_id=example_id,
                criterion_id=TOTAL_CRITERION_ID,
                perturbation=perturbation,
                seed=seed,
                score=total_score,
            )
        )
    return total_rows


def compute_comparable_total_score(
    criterion_scores: dict[str, float],
    baseline_scores: dict[str, float],
) -> float:
    if not baseline_scores:
        return sum(criterion_scores.values())
    return sum(
        criterion_scores.get(criterion_id, baseline_score)
        for criterion_id, baseline_score in baseline_scores.items()
    )


def compute_criterion_score_scale_span(
    criterion_id: str,
    score_rows: list[ScoreLogRow],
    score_scale_span: float | None,
) -> float | None:
    if score_scale_span is None:
        return None
    if criterion_id != TOTAL_CRITERION_ID:
        return score_scale_span
    baseline_criterion_ids = {
        row.criterion_id
        for row in score_rows
        if row.perturbation == "baseline" and row.criterion_id != TOTAL_CRITERION_ID
    }
    if not baseline_criterion_ids:
        return None
    return score_scale_span * len(baseline_criterion_ids)


def compute_per_example_seed_std(
    score_rows: list[ScoreLogRow],
    score_scale_span: float | None,
) -> dict[str, object]:
    grouped_scores: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for row in score_rows:
        criterion_scores = grouped_scores.setdefault(row.criterion_id, {})
        perturbation_scores = criterion_scores.setdefault(row.perturbation, {})
        example_scores = perturbation_scores.setdefault(row.example_id, {})
        example_scores[row.seed] = float(row.score)

    results: dict[str, object] = {}
    for criterion_id, criterion_scores in grouped_scores.items():
        criterion_score_scale_span = compute_criterion_score_scale_span(
            criterion_id,
            score_rows,
            score_scale_span,
        )
        results[criterion_id] = {}
        for perturbation, perturbation_scores in criterion_scores.items():
            example_std_by_id: dict[str, float] = {}
            for example_id, seed_scores in perturbation_scores.items():
                values = list(seed_scores.values())
                std_value = raw_std(values)
                if std_value is None:
                    continue
                example_std_by_id[example_id] = std_value

            example_std_values = list(example_std_by_id.values())
            results[criterion_id][perturbation] = {
                "n_examples_with_multiple_seeds": len(example_std_values),
                "example_std_mean": mean_or_none(example_std_values),
                "example_std_std": raw_std(example_std_values),
                "example_std_min": min_or_none(example_std_values),
                "example_std_max": max_or_none(example_std_values),
                "example_std_mean_scale_normalized": normalize_by_scale(
                    mean_or_none(example_std_values),
                    criterion_score_scale_span,
                ),
                "example_std_max_scale_normalized": normalize_by_scale(
                    max_or_none(example_std_values),
                    criterion_score_scale_span,
                ),
                "example_std_by_id": example_std_by_id,
            }
    return results
