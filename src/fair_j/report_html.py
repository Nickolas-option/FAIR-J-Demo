from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TOTAL_CRITERION_ID = "__total__"
MAX_EXAMPLE_VIEWER_RECORDS_PER_MODEL = 100
STAT_TEST_GROUPS = (
    ("paraphrases_vs_baseline", "Paraphrased vs Base"),
    ("deletions_vs_baseline", "Deleted vs Base"),
)
SECTION_EXPLANATIONS = {
    "dataset_level": (
        "How much do scores change when the same criterion is paraphrased? Lower score "
        "instability values and lower spillover mean cleaner dataset-level comparisons."
    ),
    "ranking_consistency": (
        "Does the judge preserve example ordering when the same criterion is "
        "rewritten? Higher ranking stability and lower flips are better."
    ),
}
METRIC_HELP = {
    "mad_paraphrases_same_criterion_scale_normalized": (
        "Mean absolute score shift when the same criterion is paraphrased, "
        "normalized by score scale span. Lower is better."
    ),
    "mad_paraphrases_cross_criterion_scale_normalized": (
        "Mean absolute score shift on other criteria when one criterion is "
        "paraphrased, normalized by score scale span."
    ),
    "spillover_ratio": (
        "Cross-criterion normalized MAD divided by same-criterion normalized MAD. "
        "Lower means paraphrasing stays more isolated to the intended criterion."
    ),
    "flip_rate_paraphrases_same_criterion": (
        "Estimated share of ranking pairs that flip when the same criterion is paraphrased."
    ),
    "kendall_paraphrases_same_criterion_mean": (
        "Average Kendall's tau-b between baseline ranking and same-criterion "
        "paraphrased ranking. Higher is better."
    ),
    "tie_pct_paraphrases_same_criterion_mean": (
        "Share of example pairs that receive tied scores when the same criterion is paraphrased."
    ),
    "n_pairs": "Number of paired observations used by the statistical test.",
    "mean_difference": "Average baseline minus perturbation score difference.",
    "paired_t_test": "Paired t-test p-value for baseline versus perturbation differences.",
    "wilcoxon_signed_rank": "Wilcoxon signed-rank p-value for paired differences.",
    "paired_permutation_test": "Permutation test p-value for paired differences.",
}
DATASET_OVERVIEW_ROWS = (
    ("mad_paraphrases_same_criterion_scale_normalized", "Score Instability"),
    ("spillover_ratio", "Spillover"),
)
RANKING_OVERVIEW_ROWS = (
    ("flip_rate_paraphrases_same_criterion", "Flips"),
    ("kendall_paraphrases_same_criterion_mean", "Ranking Stability"),
    ("tie_pct_paraphrases_same_criterion_mean", "Ties"),
)
STAT_EXPLORER_ROWS = {
    "paraphrases_vs_baseline": (
        ("n_pairs", "N"),
        ("mean_difference", "Δ"),
        ("paired_t_test", "Paired t-test p-value"),
        ("wilcoxon_signed_rank", "Wilcoxon p-value"),
        ("paired_permutation_test", "Permutation p-value"),
    ),
    "deletions_vs_baseline": (
        ("n_pairs", "N"),
        ("mean_difference", "Δ"),
        ("paired_t_test", "Paired t-test p-value"),
        ("wilcoxon_signed_rank", "Wilcoxon p-value"),
        ("paired_permutation_test", "Permutation p-value"),
    ),
}


@dataclass(frozen=True)
class ModelReport:
    label: str
    full_label: str
    run_dir: Path
    run_meta: dict[str, object]
    data: dict[str, object]


@dataclass(frozen=True)
class ReportExampleData:
    records: list[dict[str, object]]
    examples_used: int | None


def render_comparison_report(run_dirs: list[Path], output_path: Path, title: str | None = None) -> Path:
    reports = [load_model_report(run_dir) for run_dir in run_dirs]
    document_title = title or build_default_title(reports)
    context = build_template_context(reports, document_title)
    environment = build_template_environment()
    template = environment.get_template("report.html.jinja")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.render(**context), encoding="utf-8")
    return output_path


def load_model_report(run_dir: Path) -> ModelReport:
    core_output_path = run_dir / "core_output.json"
    run_meta_path = run_dir / "run.json"
    data = json.loads(core_output_path.read_text(encoding="utf-8"))
    run_meta = (
        json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta_path.exists()
        else expect_mapping(data, "run_metadata", core_output_path)
    )
    judge_model = str(run_meta.get("judge_model", run_dir.name))
    return ModelReport(
        label=build_short_label(judge_model),
        full_label=f"{judge_model} ({run_dir.name})",
        run_dir=run_dir,
        run_meta=run_meta,
        data=data,
    )


def build_template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(Path(__file__).with_name("templates")),
        autoescape=select_autoescape(enabled_extensions=("html", "jinja"), default_for_string=True),
    )


def build_template_context(reports: list[ModelReport], title: str) -> dict[str, object]:
    criteria = get_display_criteria(reports)
    example_data = build_report_example_data(reports)
    example_records = [
        record for report in reports for record in example_data[report.run_dir].records
    ]
    return {
        "title": title,
        "run_context": build_run_context_context(reports, example_data),
        "methodology": build_methodology_context(),
        "summary_cards": build_summary_cards_context(reports, criteria),
        "dataset_calculator": build_dataset_calculator_context(reports, criteria),
        "dataset_overview": build_overview_section_context(
            reports=reports,
            title="Dataset Level",
            explanation=SECTION_EXPLANATIONS["dataset_level"],
            group_title="Paraphrased vs Base",
            row_specs=DATASET_OVERVIEW_ROWS,
            fetch_value=lambda report, metric_key: fetch_dataset_metric(report, metric_key, criteria),
        ),
        "ranking_overview": build_overview_section_context(
            reports=reports,
            title="Ranking Consistency",
            explanation=SECTION_EXPLANATIONS["ranking_consistency"],
            group_title="Paraphrased vs Base",
            row_specs=RANKING_OVERVIEW_ROWS,
            fetch_value=lambda report, metric_key: fetch_ranking_metric(report, metric_key, criteria),
        ),
        "criterion_sections": build_criterion_sections_context(reports, criteria),
        "stat_explorer": build_stat_tests_explorer_context(reports, criteria),
        "example_viewer": {
            "enabled": bool(example_records),
            "title": "Example Viewer",
            "description": (
                "Compare real examples where the judge score changed after paraphrasing "
                "the evaluated criterion."
            ),
        },
        "scripts": build_page_scripts_context(reports, criteria, example_records),
    }


def build_summary_cards_context(reports: list[ModelReport], criteria: list[str]) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    for report in reports:
        score_instability = fetch_dataset_metric(
            report, "mad_paraphrases_same_criterion_scale_normalized", criteria
        )
        avg_flip = fetch_ranking_metric(report, "flip_rate_paraphrases_same_criterion", criteria)
        ranking_stability = fetch_ranking_metric(report, "kendall_paraphrases_same_criterion_mean", criteria)
        spillover = fetch_dataset_metric(report, "spillover_ratio", criteria)
        cards.append(
            {
                "label": report.label,
                "stats": [
                    {
                        "label": "Flips",
                        "value": format_optional_percent(avg_flip),
                        "help_text": METRIC_HELP["flip_rate_paraphrases_same_criterion"],
                    },
                    {
                        "label": "Score Instability",
                        "value": format_optional_number(score_instability, digits=4),
                        "help_text": METRIC_HELP["mad_paraphrases_same_criterion_scale_normalized"],
                    },
                    {
                        "label": "Ranking Stability",
                        "value": format_optional_fixed(ranking_stability, digits=3),
                        "help_text": METRIC_HELP["kendall_paraphrases_same_criterion_mean"],
                    },
                    {
                        "label": "Spillover",
                        "value": format_optional_number(spillover, digits=3),
                        "help_text": METRIC_HELP["spillover_ratio"],
                    },
                ],
            }
        )
    return cards


def build_dataset_calculator_context(reports: list[ModelReport], criteria: list[str]) -> dict[str, object]:
    return {
        "title": "Noise Calculator",
        "description": "Estimate how same-criterion score instability shrinks as dataset size grows.",
        "input_label": "Dataset Size (N):",
        "default_value": 1000,
        "calc_data": {
            report.label: fetch_dataset_metric(report, "mad_paraphrases_same_criterion_scale_normalized", criteria)
            or 0.0
            for report in reports
        },
    }


def build_run_context_context(
    reports: list[ModelReport], example_data: dict[Path, ReportExampleData]
) -> dict[str, object]:
    return {
        "title": "Run Context",
        "cards": [
            {
                "label": report.label,
                "rows": [
                    ("Judge model", str(report.run_meta.get("judge_model", "-"))),
                    ("Paraphrase model", str(report.run_meta.get("paraphrase_model") or "-")),
                    (
                        "Examples used",
                        format_optional_int(example_data[report.run_dir].examples_used),
                    ),
                    (
                        "Paraphrases per criterion",
                        format_optional_int(report.run_meta.get("paraphrases_per_criterion")),
                    ),
                    ("Repeated calls", format_optional_int(report.run_meta.get("seeds"))),
                    (
                        "Score scale",
                        f"{report.run_meta.get('scale_min', '-')}"
                        f" - {report.run_meta.get('scale_max', '-')}",
                    ),
                ],
            }
            for report in reports
        ],
    }


def build_methodology_context() -> dict[str, object]:
    return {
        "title": "Methodology",
        "description": (
            "All stability metrics below read precomputed values from run.json and core_output.json. "
            "Scale-normalized means the score shift is divided by score scale span."
        ),
        "rows": [
            {
                "label": "Score Instability",
                "meaning": "How much the score changes when the same criterion is paraphrased.",
                "formula": "same-criterion MAD / scale span",
            },
            {
                "label": "Ranking Stability",
                "meaning": "Whether example ordering stays consistent when the same criterion is paraphrased.",
                "formula": "mean same-criterion Kendall tau-b",
            },
            {
                "label": "Flips",
                "meaning": "Estimated share of ranking pairs that reverse order after same-criterion paraphrasing.",
                "formula": "same-criterion discordant pairs / same-criterion total ranking pairs",
            },
            {
                "label": "Spillover",
                "meaning": "Whether paraphrasing one criterion changes scores on other criteria.",
                "formula": "cross MAD / same-criterion MAD",
            },
        ],
    }


def build_overview_section_context(
    *,
    reports: list[ModelReport],
    title: str,
    explanation: str,
    group_title: str,
    row_specs: tuple[tuple[str, str], ...],
    fetch_value,
) -> dict[str, object]:
    return {
        "title": title,
        "help_text": explanation,
        "groups": [
            {
                "title": group_title,
                "table": build_matrix_table_context(
                    reports, row_specs, fetch_value, format_metric_cell_context
                ),
            }
        ],
    }


def build_criterion_sections_context(
    reports: list[ModelReport], criteria: list[str]
) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    for criterion_id in criteria:
        sections.append(
            {
                "title": format_criterion_label(criterion_id),
                "groups": [
                    {
                        "title": "Scoring Instability",
                        "table": build_matrix_table_context(
                            reports,
                            DATASET_OVERVIEW_ROWS,
                            lambda report, key: fetch_dataset_metric(report, key, [criterion_id]),
                            format_metric_cell_context,
                        ),
                    },
                    {
                        "title": "Ranking Consistency",
                        "table": build_matrix_table_context(
                            reports,
                            RANKING_OVERVIEW_ROWS,
                            lambda report, key: fetch_ranking_metric(report, key, [criterion_id]),
                            format_metric_cell_context,
                        ),
                    },
                ],
            }
        )
    return sections


def build_stat_tests_explorer_context(
    reports: list[ModelReport], criteria: list[str]
) -> dict[str, object]:
    options = [{"value": criterion_id, "label": format_criterion_label(criterion_id)} for criterion_id in criteria]
    options.append({"value": TOTAL_CRITERION_ID, "label": "Total Score (All criteria pooled)"})
    scopes: list[dict[str, object]] = []
    for index, option in enumerate(options):
        scopes.append(
            {
                "value": option["value"],
                "is_active": index == 0,
                "groups": [
                    {
                        "title": group_label,
                        "table": build_matrix_table_context(
                            reports,
                            STAT_EXPLORER_ROWS[group_key],
                            lambda report, metric_key, gk=group_key, criterion_id=option["value"]: fetch_stat_metric(
                                report, criterion_id, gk, metric_key
                            ),
                            format_stat_metric_cell_context,
                        ),
                    }
                    for group_key, group_label in STAT_TEST_GROUPS
                ],
            }
        )
    return {
        "title": "Statistical Tests",
        "help_text": (
            "These tables show the already computed paired test outputs stored "
            "in core_output.json. No baseline or perturbation scores are "
            "recomputed during rendering."
        ),
        "select_label": "Score",
        "options": options,
        "scopes": scopes,
    }


def build_matrix_table_context(
    reports: list[ModelReport],
    row_specs: tuple[tuple[str, str], ...],
    fetch_value,
    cell_formatter,
) -> dict[str, object]:
    return {
        "headers": [report.label for report in reports],
        "rows": [
            {
                "label": {"text": label, "help_text": METRIC_HELP.get(metric_key)},
                "cells": [
                    cell_formatter(fetch_value(report, metric_key), metric_key)
                    for report in reports
                ],
            }
            for metric_key, label in row_specs
        ],
    }


def build_page_scripts_context(
    reports: list[ModelReport],
    criteria: list[str],
    example_records: list[dict[str, object]],
) -> dict[str, object]:
    calc_data = {
        report.label: fetch_dataset_metric(report, "mad_paraphrases_same_criterion_scale_normalized", criteria)
        or 0.0
        for report in reports
    }
    return {
        "calc_data_json": json.dumps(calc_data),
        "example_records_json": json.dumps(
            example_records, ensure_ascii=False, separators=(",", ":")
        ).replace("</", "<\\/"),
    }


def build_report_example_data(reports: list[ModelReport]) -> dict[Path, ReportExampleData]:
    dataset_cache: dict[Path, dict[str, dict[str, object]]] = {}
    return {
        report.run_dir: load_report_example_data(report, dataset_cache) for report in reports
    }


def load_report_example_data(
    report: ModelReport,
    dataset_cache: dict[Path, dict[str, dict[str, object]]],
) -> ReportExampleData:
    score_log_path = report.run_dir / "score_log.jsonl"
    configured_limit = optional_int(report.run_meta.get("limit_examples"))
    dataset_path = resolve_run_path(report, report.run_meta.get("dataset_path"))
    dataset_records: dict[str, dict[str, object]] = {}
    if dataset_path is not None:
        if dataset_path not in dataset_cache:
            dataset_cache[dataset_path] = load_dataset_records(dataset_path, report.run_meta)
        dataset_records = dataset_cache[dataset_path]

    if not score_log_path.exists():
        examples_used = configured_limit
        if examples_used is None and dataset_records:
            examples_used = len(dataset_records)
        return ReportExampleData(records=[], examples_used=examples_used)

    baselines, observed_example_ids = load_baseline_scores(score_log_path)
    examples_used = len(observed_example_ids)
    variants_path = report.run_dir / "variants.json"
    if not variants_path.exists() or not dataset_records:
        return ReportExampleData(records=[], examples_used=examples_used)

    baseline_criteria, changed_variants = load_changed_variants(variants_path)
    records = collect_changed_score_examples(
        report=report,
        score_log_path=score_log_path,
        dataset_records=dataset_records,
        baselines=baselines,
        baseline_criteria=baseline_criteria,
        changed_variants=changed_variants,
    )
    return ReportExampleData(records=records, examples_used=examples_used)


def resolve_run_path(report: ModelReport, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    if not path.is_absolute():
        candidates.extend(parent / path for parent in (report.run_dir, *report.run_dir.parents))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_dataset_records(
    dataset_path: Path, run_meta: dict[str, object]
) -> dict[str, dict[str, object]]:
    if dataset_path.suffix.lower() == ".json":
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {dataset_path}")
        rows = payload
    else:
        rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    id_column = str(run_meta.get("dataset_id_column") or "id")
    records: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or id_column not in row:
            continue
        records[str(row[id_column])] = row
    return records


def load_baseline_scores(
    score_log_path: Path,
) -> tuple[dict[tuple[str, str, str], float], set[str]]:
    baselines: dict[tuple[str, str, str], float] = {}
    example_ids: set[str] = set()
    with score_log_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_id = str(row["example_id"])
            example_ids.add(example_id)
            score = safe_float(row.get("score"))
            if row.get("perturbation") != "baseline" or score is None:
                continue
            key = (example_id, str(row["criterion_id"]), str(row["seed"]))
            baselines[key] = score
    return baselines, example_ids


def load_changed_variants(
    variants_path: Path,
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    if not isinstance(variants, list):
        raise ValueError(f"Expected a JSON array in {variants_path}")
    baseline_variant = next(
        (
            variant
            for variant in variants
            if isinstance(variant, dict) and variant.get("perturbation") == "baseline"
        ),
        None,
    )
    if not isinstance(baseline_variant, dict) or not isinstance(
        baseline_variant.get("criteria"), list
    ):
        raise ValueError(f"Missing baseline criteria in {variants_path}")
    baseline_criteria = {
        str(criterion["id"]): str(criterion["text"])
        for criterion in baseline_variant["criteria"]
        if isinstance(criterion, dict) and "id" in criterion and "text" in criterion
    }
    changed_variants: dict[str, tuple[str, str]] = {}
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        perturbation = str(variant.get("perturbation", ""))
        criteria = variant.get("criteria")
        if not perturbation.startswith("paraphrase__") or not isinstance(criteria, list):
            continue
        for criterion in criteria:
            if not isinstance(criterion, dict) or "id" not in criterion or "text" not in criterion:
                continue
            criterion_id = str(criterion["id"])
            criterion_text = str(criterion["text"])
            if baseline_criteria.get(criterion_id) != criterion_text:
                changed_variants[perturbation] = (criterion_id, criterion_text)
                break
    return baseline_criteria, changed_variants


def collect_changed_score_examples(
    *,
    report: ModelReport,
    score_log_path: Path,
    dataset_records: dict[str, dict[str, object]],
    baselines: dict[tuple[str, str, str], float],
    baseline_criteria: dict[str, str],
    changed_variants: dict[str, tuple[str, str]],
) -> list[dict[str, object]]:
    context_column = str(report.run_meta.get("dataset_context_column") or "context")
    candidate_column = str(report.run_meta.get("dataset_candidate_column") or "candidate")
    strongest_by_pair: dict[tuple[str, str, str], dict[str, object]] = {}
    with score_log_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            perturbation = str(row.get("perturbation", ""))
            changed_variant = changed_variants.get(perturbation)
            score = safe_float(row.get("score"))
            if changed_variant is None or score is None:
                continue
            criterion_id, paraphrased_criterion = changed_variant
            if str(row["criterion_id"]) != criterion_id:
                continue
            example_id = str(row["example_id"])
            seed = str(row["seed"])
            baseline_score = baselines.get((example_id, criterion_id, seed))
            dataset_record = dataset_records.get(example_id)
            if baseline_score is None or dataset_record is None:
                continue
            delta = score - baseline_score
            if delta == 0:
                continue
            record = {
                "model": report.label,
                "example_id": example_id,
                "criterion_id": criterion_id,
                "seed": row["seed"],
                "original_criterion": baseline_criteria[criterion_id],
                "paraphrased_criterion": paraphrased_criterion,
                "baseline_score": normalize_score(baseline_score),
                "paraphrased_score": normalize_score(score),
                "delta": normalize_score(delta),
                "context": str(dataset_record.get(context_column, "")),
                "candidate": str(dataset_record.get(candidate_column, "")),
            }
            pair_key = (example_id, criterion_id, paraphrased_criterion)
            previous = strongest_by_pair.get(pair_key)
            if previous is None or abs(delta) > abs(float(previous["delta"])):
                strongest_by_pair[pair_key] = record
    ranked = sorted(
        strongest_by_pair.values(),
        key=lambda item: (
            -abs(float(item["delta"])),
            str(item["criterion_id"]),
            str(item["example_id"]),
            str(item["paraphrased_criterion"]),
        ),
    )
    return ranked[:MAX_EXAMPLE_VIEWER_RECORDS_PER_MODEL]


def normalize_score(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def get_display_criteria(reports: list[ModelReport]) -> list[str]:
    criterion_order: list[str] = []
    seen: set[str] = set()
    for report in reports:
        dataset_scores = expect_mapping(report.data, "dataset_level_scores", report.run_dir / "core_output.json")
        for criterion_id in dataset_scores:
            if criterion_id == TOTAL_CRITERION_ID or criterion_id in seen:
                continue
            seen.add(criterion_id)
            criterion_order.append(criterion_id)
    return criterion_order


def fetch_dataset_metric(report: ModelReport, metric_key: str, criteria: list[str]) -> float | None:
    dataset_scores = expect_mapping(report.data, "dataset_level_scores", report.run_dir / "core_output.json")
    if metric_key == "spillover_ratio":
        same_values = [
            safe_float(
                expect_mapping(dataset_scores, criterion_id, report.run_dir / "core_output.json").get(
                    "mad_paraphrases_same_criterion_scale_normalized"
                )
            )
            for criterion_id in criteria
            if criterion_id in dataset_scores
        ]
        cross_values = [
            safe_float(
                expect_mapping(dataset_scores, criterion_id, report.run_dir / "core_output.json").get(
                    "mad_paraphrases_cross_criterion_scale_normalized"
                )
            )
            for criterion_id in criteria
            if criterion_id in dataset_scores
        ]
        same_mean = mean_non_null(same_values)
        cross_mean = mean_non_null(cross_values)
        if same_mean is None or cross_mean is None or same_mean == 0:
            return None
        return cross_mean / same_mean
    if metric_key == "std_total":
        total_block = dataset_scores.get(TOTAL_CRITERION_ID)
        if isinstance(total_block, dict) and isinstance(total_block.get("std_total"), (int, float)):
            return float(total_block["std_total"])
    values = [
        safe_float(expect_mapping(dataset_scores, criterion_id, report.run_dir / "core_output.json").get(metric_key))
        for criterion_id in criteria
        if criterion_id in dataset_scores
    ]
    return mean_non_null(values)


def fetch_ranking_metric(report: ModelReport, metric_key: str, criteria: list[str]) -> float | None:
    ranking_scores = expect_mapping(report.data, "ranking_consistency", report.run_dir / "core_output.json")
    if metric_key in {
        "flip_rate_paraphrases",
        "flip_rate_paraphrases_same_criterion",
        "flip_rate_deletions",
    }:
        if metric_key == "flip_rate_paraphrases_same_criterion":
            mode = "paraphrases_same_criterion"
        else:
            mode = "paraphrases" if "paraphrases" in metric_key else "deletions"
        values = [
            compute_flip_rate(expect_mapping(ranking_scores, criterion_id, report.run_dir / "core_output.json"), mode)
            for criterion_id in criteria
            if criterion_id in ranking_scores
        ]
        return mean_non_null(values)
    values = [
        safe_float(expect_mapping(ranking_scores, criterion_id, report.run_dir / "core_output.json").get(metric_key))
        for criterion_id in criteria
        if criterion_id in ranking_scores
    ]
    return mean_non_null(values)


def fetch_stat_metric(
    report: ModelReport,
    criterion_id: str,
    group_key: str,
    metric_key: str,
) -> float | None:
    group = stat_test_group(report, criterion_id, group_key)
    if metric_key in {"paired_t_test", "wilcoxon_signed_rank", "paired_permutation_test"}:
        tests = group.get("tests")
        if not isinstance(tests, dict):
            return None
        result = tests.get(metric_key)
        if not isinstance(result, dict):
            return None
        return safe_float(result.get("p_value"))
    return safe_float(group.get(metric_key))


def stat_test_group(report: ModelReport, criterion_id: str, group_key: str) -> dict[str, object]:
    dataset_scores = expect_mapping(report.data, "dataset_level_scores", report.run_dir / "core_output.json")
    criterion_block = expect_mapping(dataset_scores, criterion_id, report.run_dir / "core_output.json")
    stat_tests = criterion_block.get("stat_tests")
    if not isinstance(stat_tests, dict):
        return {}
    group = stat_tests.get(group_key)
    return group if isinstance(group, dict) else {}


def compute_flip_rate(ranking_block: dict[str, object], mode: str) -> float | None:
    discordant = safe_float(ranking_block.get(f"discordant_{mode}_pooled"))
    concordant = safe_float(ranking_block.get(f"concordant_{mode}_pooled"))
    tie_pct = safe_float(ranking_block.get(f"tie_pct_{mode}_mean"))
    if discordant is None or concordant is None or tie_pct is None:
        return None
    total_comparable = discordant + concordant
    if total_comparable <= 0:
        return None
    tie_ratio = tie_pct if tie_pct <= 1.0 else tie_pct / 100.0
    total_pairs = total_comparable if tie_ratio >= 1.0 else total_comparable / (1.0 - tie_ratio)
    if total_pairs <= 0:
        return None
    return (discordant / total_pairs) * 100.0


def format_metric_cell_context(value: float | None, metric_key: str) -> dict[str, str]:
    if value is None:
        return {"display": "N/A", "css_class": "muted"}
    if metric_key.startswith("flip_rate") or "pct" in metric_key:
        return {"display": f"{value:.1f}%", "css_class": ""}
    if "kendall" in metric_key or "gamma" in metric_key:
        return {"display": f"{value:.3f}", "css_class": ""}
    if metric_key in {
        "mad_paraphrases_same_criterion_scale_normalized",
        "mad_paraphrases_cross_criterion_scale_normalized",
    }:
        return {"display": f"{value:.4f}", "css_class": ""}
    if metric_key == "spillover_ratio":
        return {"display": f"{value:.4f}", "css_class": ""}
    return {"display": format_number(value, digits=2), "css_class": ""}


def format_stat_metric_cell_context(value: float | None, metric_key: str) -> dict[str, str]:
    if value is None:
        return {"display": "N/A", "css_class": "muted"}
    if metric_key == "n_pairs":
        return {"display": str(int(round(value))), "css_class": ""}
    if "test" in metric_key or "signed_rank" in metric_key:
        css_class = "good" if value < 0.05 else "muted"
        return {"display": f"{value:.3f}", "css_class": css_class}
    return {"display": format_number(value, digits=3), "css_class": ""}


def safe_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean_non_null(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def build_default_title(reports: list[ModelReport]) -> str:
    return "FAIR-J: LLM-as-a-Judge Robustness Report"


def build_short_label(judge_model: str) -> str:
    known = {
        "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
        "meta-llama/llama-4-scout": "Llama 4 Scout",
        "openai/gpt-5-nano": "GPT-5 Nano",
        "qwen/qwen3-8b": "Qwen3 8B",
        "qwen/qwen3-14b": "Qwen3 14B",
        "qwen/qwen3-32b": "Qwen3 32B",
        "bedrock/claude-haiku-4-5": "Claude Haiku 4.5",
        "bedrock/claude-opus-4": "Claude Opus 4",
        "bedrock/claude-opus-4-5": "Claude Opus 4.5",
        "bedrock/claude-opus-4-7": "Claude Opus 4.7",
        "bedrock/claude-sonnet-4-6": "Claude Sonnet 4.6",
        "bedrock/gpt-oss-120b": "GPT OSS 120B",
    }
    return known.get(judge_model, judge_model.rsplit("/", 1)[-1])


def format_number(value: float, digits: int) -> str:
    if value == 0:
        return "0"
    if math.isnan(value):
        return "N/A"
    if abs(value) >= 1000 or abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def format_optional_number(value: float | None, digits: int) -> str:
    if value is None:
        return "N/A"
    return format_number(value, digits)


def format_optional_fixed(value: float | None, digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def format_optional_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def format_criterion_label(criterion_id: str) -> str:
    return criterion_id.replace("_", " ")


def format_optional_int(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def expect_mapping(container: dict[str, object], key: str, source_path: Path) -> dict[str, object]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be an object in {source_path}")
    return value
