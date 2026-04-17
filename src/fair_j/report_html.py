from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from fair_j.io_utils import load_dataset_examples, load_variants, read_run_artifacts
from fair_j.stats_utils import build_stat_test_block


TOTAL_CRITERION_ID = "__total__"
DATASET_GENERAL_METRICS = (
    "baseline_score_std",
    "std_seed_default",
    "std_total",
)
DATASET_GROUP_METRICS = (
    "std_paraphrases",
    "bias_paraphrases",
    "mad_paraphrases",
    "bias_to_mae_ratio_paraphrases",
)
RANKING_GROUP_METRICS = (
    "kendall_paraphrases_mean",
    "kendall_paraphrases_std",
    "kendall_paraphrases_min",
    "kendall_paraphrases_max",
    "concordant_paraphrases_pooled",
    "discordant_paraphrases_pooled",
    "gamma_paraphrases_pooled",
    "tie_pct_paraphrases_mean",
)
STAT_TEST_VALUE_KEYS = (
    "n_pairs",
    "mean_difference",
    "mde_mean_difference",
    "effect_size_dz",
    "mde_effect_size_dz",
    "observed_abs_mean_difference_over_mde",
)
STAT_TEST_GROUPS = (
    ("paraphrases_vs_baseline", "Paraphrases vs baseline"),
    ("deletions_vs_baseline", "Deletions vs baseline"),
)
STAT_TESTS = (
    ("paired_permutation_test", "Permutation"),
    ("wilcoxon_signed_rank", "Wilcoxon"),
    ("paired_t_test", "Paired t-test"),
)
DATASET_GROUP_SUFFIXES = {
    "paraphrases": (
        "std_paraphrases",
        "bias_paraphrases",
        "mad_paraphrases",
        "bias_to_mae_ratio_paraphrases",
    ),
    "deletions": (
        "std_deletions",
        "bias_deletions",
        "mad_deletions",
        "bias_to_mae_ratio_deletions",
    ),
}
RANKING_GROUP_SUFFIXES = {
    "paraphrases": (
        "kendall_paraphrases_mean",
        "kendall_paraphrases_std",
        "kendall_paraphrases_min",
        "kendall_paraphrases_max",
        "concordant_paraphrases_pooled",
        "discordant_paraphrases_pooled",
        "gamma_paraphrases_pooled",
        "tie_pct_paraphrases_mean",
    ),
    "deletions": (
        "kendall_deletions_mean",
        "kendall_deletions_std",
        "kendall_deletions_min",
        "kendall_deletions_max",
        "concordant_deletions_pooled",
        "discordant_deletions_pooled",
        "gamma_deletions_pooled",
        "tie_pct_deletions_mean",
    ),
}
DATASET_METRICS = (
    *DATASET_GENERAL_METRICS,
    "std_paraphrases",
    "std_deletions",
    "bias_paraphrases",
    "mad_paraphrases",
    "bias_to_mae_ratio_paraphrases",
    "bias_deletions",
    "mad_deletions",
    "bias_to_mae_ratio_deletions",
)
RANKING_METRICS = (
    "kendall_paraphrases_mean",
    "kendall_paraphrases_std",
    "kendall_paraphrases_min",
    "kendall_paraphrases_max",
    "gamma_paraphrases_pooled",
    "tie_pct_paraphrases_mean",
    "kendall_deletions_mean",
    "kendall_deletions_std",
    "kendall_deletions_min",
    "kendall_deletions_max",
    "gamma_deletions_pooled",
    "tie_pct_deletions_mean",
)
SECTION_EXPLANATIONS = {
    "ranking_consistency": (
        "We take all examples for one criterion, rank them by the baseline score, and rank them again under each paraphrase or deletion variant. "
        "Then we compute Kendall tau-b between the baseline ranking and each perturbed ranking, and aggregate those tau values across variants. "
        "We also track the percentage of tied pairs and Goodman-Kruskal gamma as a diagnostic metric on comparable pairs only. "
        "Use this section when you care about preserving example ordering, for example in reward modeling, reranking, or pairwise preference training."
    ),
    "dataset_level": (
        "We first reduce scores to dataset-level summaries for each criterion and perturbation. "
        "Then we inspect average shifts with bias, average absolute shifts with MAD, and test whether baseline-minus-perturbation differences are centered at zero using statistical tests. "
        "Use this section when you care about whether aggregate judge scores stay stable enough for method comparison."
    ),
}
METRIC_EXPLANATIONS = {
    "baseline_score_std": (
        "We take all baseline example-level scores for this criterion and compute their standard deviation. "
        "Higher means the baseline scoring distribution is more spread out, while lower means the judge compresses more examples into a narrower score range."
    ),
    "std_total": (
        "We take the dataset-level mean score for baseline, all paraphrase variants, and all deletion variants for this criterion. "
        "Then we compute the standard deviation across those perturbation-level means. Lower means the overall average score moves less when the rubric changes."
    ),
    "std_seed_default": (
        "We take the dataset-level mean score for the baseline rubric for each seed. "
        "Then we compute the standard deviation across those baseline seed-level means. Lower means the default rubric itself is more stable across repeated runs."
    ),
    "std_paraphrases": (
        "We take the dataset-level mean score for each paraphrase variant of this criterion. "
        "Then we compute the standard deviation across those paraphrase means. Lower means paraphrases of the rubric produce more similar average scores."
    ),
    "std_deletions": (
        "We take the dataset-level mean score for each deletion variant of this criterion. "
        "Then we compute the standard deviation across those deletion means. Lower means deletions of the rubric produce more similar average scores."
    ),
    "bias_paraphrases": (
        "We take every example-level difference between a paraphrase variant and the baseline for this criterion. "
        "Then we sum those signed differences and divide by the number of baseline-paraphrase pairs. Values near zero mean little average shift; positive means paraphrases tend to score lower than baseline."
    ),
    "mad_paraphrases": (
        "We take every example-level difference between a paraphrase variant and the baseline for this criterion. "
        "Then we take the absolute value, sum those absolute differences, and divide by the number of baseline-paraphrase pairs. Lower means paraphrases change the score less on average."
    ),
    "bias_to_mae_ratio_paraphrases": (
        "We divide the signed paraphrase bias by the paraphrase MAE (the same quantity as MAD here). "
        "Values near 0 mean the total shift is mostly canceling noise, while values near -1 or +1 mean the perturbation effect is mostly systematic and one-directional."
    ),
    "bias_deletions": (
        "We take every example-level difference between a deletion variant and the baseline for this criterion. "
        "Then we sum those signed differences and divide by the number of baseline-deletion pairs. Values near zero mean little average shift; positive means deletions tend to score lower than baseline."
    ),
    "mad_deletions": (
        "We take every example-level difference between a deletion variant and the baseline for this criterion. "
        "Then we take the absolute value, sum those absolute differences, and divide by the number of baseline-deletion pairs. Lower means deletions change the score less on average."
    ),
    "bias_to_mae_ratio_deletions": (
        "We divide the signed deletion bias by the deletion MAE (the same quantity as MAD here). "
        "Values near 0 mean the total shift is mostly canceling noise, while values near -1 or +1 mean the perturbation effect is mostly systematic and one-directional."
    ),
    "kendall_paraphrases_mean": (
        "For each paraphrase variant, we rank all examples by score for this criterion and compare that ranking against the baseline ranking using Kendall tau-b. "
        "Then we average those tau-b values across paraphrase variants. Higher means the model preserves the example ordering better under paraphrased rubric wording while correctly accounting for tied scores."
        '<div class="tooltip-formula">τ<sub>b</sub> = (C - D) / √((C + D + T<sub>x</sub>)(C + D + T<sub>y</sub>))</div>'
        '<div class="tooltip-formula">C = concordant pairs, D = discordant pairs</div>'
        '<div class="tooltip-formula">T<sub>x</sub> = ties only in baseline, T<sub>y</sub> = ties only in perturbation</div>'
    ),
    "kendall_paraphrases_std": (
        "We compute Kendall tau between each paraphrase ranking and the baseline ranking for this criterion. "
        "Then we compute the standard deviation of those tau values across paraphrase variants. Lower means ranking stability is more consistent across paraphrases."
    ),
    "kendall_paraphrases_min": (
        "We compute Kendall tau between each paraphrase ranking and the baseline ranking for this criterion. "
        "Then we take the minimum tau across paraphrase variants. Higher means even the worst paraphrase stays relatively close to the baseline ranking."
    ),
    "kendall_paraphrases_max": (
        "We compute Kendall tau between each paraphrase ranking and the baseline ranking for this criterion. "
        "Then we take the maximum tau across paraphrase variants. Higher means the best paraphrase keeps the ranking very close to baseline."
        '<div class="tooltip-formula">max over per-variant τ<sub>b</sub></div>'
    ),
    "kendall_deletions_mean": (
        "For each deletion variant, we rank all examples by score for this criterion and compare that ranking against the baseline ranking using Kendall tau-b. "
        "Then we average those tau-b values across deletion variants. Higher means the model preserves the example ordering better when rubric text is removed while correctly accounting for tied scores."
        '<div class="tooltip-formula">τ<sub>b</sub> = (C - D) / √((C + D + T<sub>x</sub>)(C + D + T<sub>y</sub>))</div>'
        '<div class="tooltip-formula">C = concordant pairs, D = discordant pairs</div>'
        '<div class="tooltip-formula">T<sub>x</sub> = ties only in baseline, T<sub>y</sub> = ties only in perturbation</div>'
    ),
    "gamma_paraphrases_pooled": (
        "We pool concordant and discordant comparable pairs across all paraphrase variants for this criterion and then compute one Goodman-Kruskal gamma value. "
        "This metric ignores tied pairs and shows how stable the ordering is when we focus only on pairs whose order is actually distinguishable."
        '<div class="tooltip-formula">γ = (C - D) / (C + D)</div>'
        '<div class="tooltip-formula">C = concordant comparable pairs, D = discordant comparable pairs</div>'
        '<div class="tooltip-formula">pooled across all variants</div>'
    ),
    "gamma_deletions_pooled": (
        "We pool concordant and discordant comparable pairs across all deletion variants for this criterion and then compute one Goodman-Kruskal gamma value. "
        "This metric ignores tied pairs and shows how stable the ordering is when we focus only on pairs whose order is actually distinguishable."
        '<div class="tooltip-formula">γ = (C - D) / (C + D)</div>'
        '<div class="tooltip-formula">C = concordant comparable pairs, D = discordant comparable pairs</div>'
        '<div class="tooltip-formula">pooled across all variants</div>'
    ),
    "tie_pct_paraphrases_mean": (
        "For each paraphrase variant, we look at all example pairs for this criterion and count the share of pairs that are tied in the baseline ranking or in the perturbed ranking. "
        "Then we average that tied-pair percentage across paraphrase variants. Higher means the score scale is coarser and provides less ranking resolution."
    ),
    "tie_pct_deletions_mean": (
        "For each deletion variant, we look at all example pairs for this criterion and count the share of pairs that are tied in the baseline ranking or in the perturbed ranking. "
        "Then we average that tied-pair percentage across deletion variants. Higher means the score scale is coarser and provides less ranking resolution."
    ),
    "kendall_deletions_std": (
        "We compute Kendall tau between each deletion ranking and the baseline ranking for this criterion. "
        "Then we compute the standard deviation of those tau values across deletion variants. Lower means ranking stability is more consistent across deletions."
    ),
    "kendall_deletions_min": (
        "We compute Kendall tau between each deletion ranking and the baseline ranking for this criterion. "
        "Then we take the minimum tau across deletion variants. Higher means even the most damaging deletion still keeps the ranking relatively close to baseline."
    ),
    "kendall_deletions_max": (
        "We compute Kendall tau between each deletion ranking and the baseline ranking for this criterion. "
        "Then we take the maximum tau across deletion variants. Higher means the easiest deletion leaves the ranking very close to baseline."
        '<div class="tooltip-formula">max over per-variant τ<sub>b</sub></div>'
    ),
    "paraphrases_vs_baseline:n_pairs": (
        "We pair every paraphrase score with the matching baseline score for the same example and criterion. "
        "This number is how many paired observations go into the paraphrase-vs-baseline statistical tests."
    ),
    "deletions_vs_baseline:n_pairs": (
        "We pair every deletion score with the matching baseline score for the same example and criterion. "
        "This number is how many paired observations go into the deletion-vs-baseline statistical tests."
    ),
    "paraphrases_vs_baseline:paired_permutation_test": (
        "We take all baseline-minus-paraphrase differences for this criterion. "
        "Then we repeatedly flip their signs at random and ask how often we get a mean difference at least as extreme as the observed one. A smaller p-value means the paraphrase shift is harder to explain by chance."
    ),
    "paraphrases_vs_baseline:mean_difference": (
        "This is the observed average paired difference baseline minus perturbation for this criterion."
        '<div class="tooltip-formula">d<sub>i</sub> = baseline<sub>i</sub> - perturbation<sub>i</sub></div>'
        '<div class="tooltip-formula">mean difference = mean(d)</div>'
    ),
    "paraphrases_vs_baseline:effect_size_dz": (
        "This is Cohen's dz for the paired differences baseline minus perturbation. "
        "It rescales the observed mean difference by the standard deviation of the paired differences, so it is directly comparable across datasets and criteria."
        '<div class="tooltip-formula">d<sub>z</sub> = mean(d) / std(d)</div>'
    ),
    "paraphrases_vs_baseline:mde_effect_size_dz": (
        "This is the minimum detectable paired effect size for a two-sided paired t-test with alpha 0.05 and target power 0.80, using the current number of pairs. "
        "Smaller values mean the current sample size can reliably detect smaller standardized shifts."
        '<div class="tooltip-formula">power = 0.80, α = 0.05</div>'
        '<div class="tooltip-formula">solve for the smallest detectable |d<sub>z</sub>|</div>'
    ),
    "paraphrases_vs_baseline:mde_mean_difference": (
        "This is the minimum detectable absolute mean difference in score units for a two-sided paired t-test with alpha 0.05 and target power 0.80, using the current variability of paired differences. "
        "It translates the standardized MDE into the same raw score units as the observed mean shift."
        '<div class="tooltip-formula">MDE mean difference = MDE d<sub>z</sub> × std(d)</div>'
    ),
    "paraphrases_vs_baseline:observed_abs_mean_difference_over_mde": (
        "This ratio compares the observed absolute mean difference against the minimum detectable effect. "
        "Values above 1 mean the observed shift is larger than the estimated MDE; values below 1 mean it is smaller."
        '<div class="tooltip-formula">|mean(d)| / MDE mean difference</div>'
    ),
    "paraphrases_vs_baseline:wilcoxon_signed_rank": (
        "We take all baseline-minus-paraphrase differences for this criterion, rank their absolute sizes, and compare the positive and negative signed ranks. "
        "A smaller p-value means the paraphrase differences are systematically shifted away from zero."
    ),
    "paraphrases_vs_baseline:paired_t_test": (
        "We take all baseline-minus-paraphrase differences for this criterion, compute their mean, and divide by the estimated standard error of that mean. "
        "The p-value tells us how plausible it is to see a mean difference this large if the true average shift were zero."
    ),
    "deletions_vs_baseline:paired_permutation_test": (
        "We take all baseline-minus-deletion differences for this criterion. "
        "Then we repeatedly flip their signs at random and ask how often we get a mean difference at least as extreme as the observed one. A smaller p-value means the deletion shift is harder to explain by chance."
    ),
    "deletions_vs_baseline:mean_difference": (
        "This is the observed average paired difference baseline minus perturbation for this criterion."
        '<div class="tooltip-formula">d<sub>i</sub> = baseline<sub>i</sub> - perturbation<sub>i</sub></div>'
        '<div class="tooltip-formula">mean difference = mean(d)</div>'
    ),
    "deletions_vs_baseline:effect_size_dz": (
        "This is Cohen's dz for the paired differences baseline minus perturbation. "
        "It rescales the observed mean difference by the standard deviation of the paired differences, so it is directly comparable across datasets and criteria."
        '<div class="tooltip-formula">d<sub>z</sub> = mean(d) / std(d)</div>'
    ),
    "deletions_vs_baseline:mde_effect_size_dz": (
        "This is the minimum detectable paired effect size for a two-sided paired t-test with alpha 0.05 and target power 0.80, using the current number of pairs. "
        "Smaller values mean the current sample size can reliably detect smaller standardized shifts."
        '<div class="tooltip-formula">power = 0.80, α = 0.05</div>'
        '<div class="tooltip-formula">solve for the smallest detectable |d<sub>z</sub>|</div>'
    ),
    "deletions_vs_baseline:mde_mean_difference": (
        "This is the minimum detectable absolute mean difference in score units for a two-sided paired t-test with alpha 0.05 and target power 0.80, using the current variability of paired differences. "
        "It translates the standardized MDE into the same raw score units as the observed mean shift."
        '<div class="tooltip-formula">MDE mean difference = MDE d<sub>z</sub> × std(d)</div>'
    ),
    "deletions_vs_baseline:observed_abs_mean_difference_over_mde": (
        "This ratio compares the observed absolute mean difference against the minimum detectable effect. "
        "Values above 1 mean the observed shift is larger than the estimated MDE; values below 1 mean it is smaller."
        '<div class="tooltip-formula">|mean(d)| / MDE mean difference</div>'
    ),
    "deletions_vs_baseline:wilcoxon_signed_rank": (
        "We take all baseline-minus-deletion differences for this criterion, rank their absolute sizes, and compare the positive and negative signed ranks. "
        "A smaller p-value means the deletion differences are systematically shifted away from zero."
    ),
    "deletions_vs_baseline:paired_t_test": (
        "We take all baseline-minus-deletion differences for this criterion, compute their mean, and divide by the estimated standard error of that mean. "
        "The p-value tells us how plausible it is to see a mean difference this large if the true average shift were zero."
    ),
}


@dataclass(frozen=True)
class ModelReport:
    label: str
    full_label: str
    run_dir: Path
    core_output_path: Path
    data: dict[str, object]
    dataset_examples: dict[str, dict[str, object]]
    score_rows: list[dict[str, object]]
    variants: dict[str, list[dict[str, str]]]


def render_comparison_report(run_dirs: list[Path], output_path: Path, title: str | None = None) -> Path:
    reports = [load_model_report(run_dir) for run_dir in run_dirs]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html_document(reports, title=title), encoding="utf-8")
    return output_path


def load_model_report(run_dir: Path) -> ModelReport:
    core_output_path = run_dir / "core_output.json"
    data = json.loads(core_output_path.read_text(encoding="utf-8"))
    if needs_core_refresh(data):
        from fair_j.evaluate_judge import evaluate_judge

        evaluate_judge(run_dir)
        data = json.loads(core_output_path.read_text(encoding="utf-8"))
    run_metadata = data["run_metadata"]
    _, score_rows = read_run_artifacts(run_dir)
    variants = {
        variant.perturbation: [
            {"id": criterion.id, "text": criterion.text}
            for criterion in variant.criteria
        ]
        for variant in load_variants(run_dir / "variants.json")
    }
    dataset_examples = {
        example.id: {
            "context": example.context,
            "candidate": example.candidate,
            "metadata": example.metadata,
        }
        for example in load_dataset_examples(
            Path(run_metadata["dataset_path"]),
            id_column=run_metadata.get("dataset_id_column", "id"),
            context_column=run_metadata.get("dataset_context_column", "context"),
            candidate_column=run_metadata.get("dataset_candidate_column", "candidate"),
        )
    }
    judge_model = run_metadata["judge_model"]
    label = build_short_label(judge_model, run_dir.name)
    full_label = f"{judge_model} ({run_dir.name})"
    return ModelReport(
        label=label,
        full_label=full_label,
        run_dir=run_dir,
        core_output_path=core_output_path,
        data=data,
        dataset_examples=dataset_examples,
        score_rows=[row.__dict__ for row in score_rows],
        variants=variants,
    )


def get_display_criteria(reports: list[ModelReport]) -> list[str]:
    if not reports:
        return []
    criterion_ids = list(reports[0].data["dataset_level_scores"].keys())
    return [criterion_id for criterion_id in criterion_ids if criterion_id != TOTAL_CRITERION_ID]


def needs_core_refresh(data: dict[str, object]) -> bool:
    dataset_level = data.get("dataset_level_scores", {})
    ranking = data.get("ranking_consistency", {})
    if TOTAL_CRITERION_ID not in dataset_level:
        return True
    if not ranking:
        return True
    first_key = next(iter(ranking))
    first_block = ranking[first_key]
    return "concordant_paraphrases_pooled" not in first_block


def build_short_label(judge_model: str, run_dir_name: str) -> str:
    known = {
        "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
        "meta-llama/llama-4-scout": "Llama 4 Scout",
        "openai/gpt-5-nano": "GPT-5 Nano",
        "qwen/qwen3-8b": "Qwen3 8B",
        "qwen/qwen3-14b": "Qwen3 14B",
        "qwen/qwen3-32b": "Qwen3 32B",
    }
    provider = infer_provider_label(run_dir_name)
    base = known.get(judge_model, judge_model.rsplit("/", 1)[-1])
    return f"{base} / {provider}" if provider else base


def infer_provider_label(run_dir_name: str) -> str:
    for key, label in (
        ("alibaba", "Alibaba"),
        ("deepinfra", "DeepInfra"),
        ("groq", "Groq"),
        ("openai", "OpenAI"),
        ("autoroute", "AutoRoute"),
    ):
        if key in run_dir_name:
            return label
    return ""


def build_html_document(reports: list[ModelReport], title: str | None = None) -> str:
    page_title = title or "FAIR-J Model Comparison"
    criteria = get_display_criteria(reports)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page_title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5efe5;
      --panel: rgba(255, 252, 247, 0.92);
      --panel-strong: #fffaf2;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9ccb9;
      --accent: #0f766e;
      --accent-soft: rgba(15, 118, 110, 0.12);
      --bad: #b42318;
      --warn: #b54708;
      --good: #027a48;
      --shadow: 0 18px 44px rgba(62, 39, 8, 0.08);
      --radius: 18px;
      --mono: "SFMono-Regular", ui-monospace, Menlo, monospace;
      --sans: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.09), transparent 32%),
        radial-gradient(circle at top right, rgba(181, 71, 8, 0.10), transparent 28%),
        linear-gradient(180deg, #fdf7ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}

    main {{
      width: min(1440px, calc(100vw - 32px));
      margin: 32px auto 48px;
    }}

    .hero {{
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.92), rgba(17, 94, 89, 0.88));
      color: white;
      padding: 28px 30px;
      border-radius: calc(var(--radius) + 6px);
      box-shadow: var(--shadow);
    }}

    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(32px, 4vw, 52px);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }}

    .hero p {{
      margin: 0;
      max-width: 980px;
      color: rgba(255, 255, 255, 0.9);
      font-size: 17px;
      line-height: 1.45;
    }}

    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 20px 0 24px;
    }}

    button {{
      border: 1px solid rgba(15, 118, 110, 0.24);
      background: var(--panel);
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      font: inherit;
      box-shadow: 0 6px 18px rgba(62, 39, 8, 0.06);
    }}

    button:hover {{
      background: var(--panel-strong);
    }}

    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 22px;
    }}

    .overview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 22px;
    }}

    .summary-card,
    details.section {{
      background: var(--panel);
      border: 1px solid rgba(217, 204, 185, 0.95);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}

    .summary-card {{
      padding: 18px 20px;
    }}

    .overview-card {{
      padding: 20px 22px;
    }}

    .summary-card h2 {{
      margin: 0 0 10px;
      font-size: 21px;
      line-height: 1.1;
    }}

    .overview-card h2 {{
      margin: 0 0 10px;
      font-size: 24px;
      line-height: 1.05;
    }}

    .meta {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 14px;
      word-break: break-word;
    }}

    .stat-list {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 14px;
    }}

    .stat-item {{
      background: rgba(255, 255, 255, 0.72);
      border-radius: 14px;
      padding: 10px 12px;
      border: 1px solid rgba(217, 204, 185, 0.7);
    }}

    .stat-item .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }}

    .stat-item .value {{
      font-family: var(--mono);
      font-size: 16px;
    }}

    details.section {{
      margin-bottom: 16px;
      overflow: hidden;
    }}

    details.section > summary {{
      list-style: none;
      cursor: pointer;
      padding: 18px 22px;
      font-size: 24px;
      font-weight: 700;
      border-bottom: 1px solid transparent;
    }}

    details.section[open] > summary {{
      border-bottom-color: var(--line);
      background: rgba(255, 255, 255, 0.45);
    }}

    details.section > summary::-webkit-details-marker {{
      display: none;
    }}

    .section-body {{
      padding: 18px 22px 24px;
    }}

    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}

    .badge {{
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      padding: 6px 10px;
      font-size: 12px;
      font-family: var(--mono);
    }}

    .metric-block {{
      margin-bottom: 22px;
    }}

    .metric-block h3 {{
      margin: 0 0 10px;
      font-size: 17px;
    }}

    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}

    .group-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}

    .mini-card {{
      background: rgba(255, 255, 255, 0.72);
      border-radius: 16px;
      padding: 14px;
      border: 1px solid rgba(217, 204, 185, 0.7);
    }}

    .mini-card h4 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}

    .group-card {{
      background: rgba(255, 255, 255, 0.72);
      border-radius: 16px;
      padding: 14px;
      border: 1px solid rgba(217, 204, 185, 0.7);
    }}

    .group-card h4 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}

    .control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 14px;
      align-items: end;
    }}

    .control-field {{
      min-width: 180px;
      display: grid;
      gap: 6px;
    }}

    .control-field label {{
      color: var(--muted);
      font-size: 12px;
      font-family: var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    select {{
      border: 1px solid rgba(15, 118, 110, 0.24);
      background: rgba(255, 255, 255, 0.88);
      color: var(--ink);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
    }}

    .viewer-shell {{
      background: rgba(255, 255, 255, 0.72);
      border-radius: 18px;
      border: 1px solid rgba(217, 204, 185, 0.8);
      padding: 18px;
    }}

    .viewer-toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 12px;
      margin-bottom: 14px;
    }}

    .viewer-counter {{
      color: var(--muted);
      font-family: var(--mono);
      font-size: 13px;
      margin-left: auto;
    }}

    .viewer-nav {{
      display: inline-flex;
      gap: 8px;
    }}

    .viewer-nav button {{
      min-width: 44px;
      padding: 10px 12px;
    }}

    .viewer-card {{
      display: grid;
      gap: 14px;
    }}

    .viewer-title {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      align-items: center;
    }}

    .viewer-title h4 {{
      margin: 0;
      font-size: 22px;
    }}

    .viewer-meta {{
      color: var(--muted);
      font-size: 13px;
    }}

    .viewer-summary {{
      border: 1px solid rgba(217, 204, 185, 0.8);
      border-radius: 14px;
      background: rgba(255, 252, 247, 0.92);
      overflow: hidden;
    }}

    .viewer-summary summary {{
      cursor: pointer;
      list-style: none;
      padding: 12px 14px;
      display: grid;
      gap: 4px;
    }}

    .viewer-summary summary::-webkit-details-marker {{
      display: none;
    }}

    .viewer-summary[open] summary {{
      border-bottom: 1px solid rgba(217, 204, 185, 0.75);
      background: rgba(255, 255, 255, 0.78);
    }}

    .viewer-summary-title {{
      font-family: var(--mono);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--accent);
    }}

    .viewer-summary-preview {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}

    .viewer-summary-body {{
      padding: 14px;
    }}

    .rubric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}

    .rubric-panel {{
      border: 1px solid rgba(217, 204, 185, 0.8);
      border-radius: 14px;
      background: rgba(255, 252, 247, 0.9);
      padding: 12px 14px;
    }}

    .rubric-panel h5 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}

    .rubric-list {{
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}

    .rubric-item {{
      border-radius: 12px;
      padding: 8px 10px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(217, 204, 185, 0.55);
    }}

    .rubric-item.is-focus {{
      border-color: rgba(15, 118, 110, 0.4);
      background: rgba(15, 118, 110, 0.08);
    }}

    .rubric-item.is-removed {{
      border-color: rgba(180, 35, 24, 0.35);
      background: rgba(180, 35, 24, 0.06);
    }}

    .rubric-item strong {{
      display: block;
      font-family: var(--mono);
      margin-bottom: 4px;
      font-size: 12px;
    }}

    .stat-scope {{
      display: none;
    }}

    .stat-scope.is-active {{
      display: block;
    }}

    .muted-note {{
      color: var(--muted);
      font-size: 14px;
    }}

    .example-list {{
      display: grid;
      gap: 10px;
    }}

    .example-item {{
      padding: 12px 0 0;
      border-top: 1px solid rgba(217, 204, 185, 0.7);
    }}

    .example-item:first-child {{
      border-top: 0;
      padding-top: 0;
    }}

    .example-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px 10px;
      margin-bottom: 8px;
    }}

    .perturbation-badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      background: rgba(15, 118, 110, 0.12);
      color: var(--accent);
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 700;
    }}

    .example-meta {{
      font-size: 12px;
      color: var(--muted);
    }}

    .example-id {{
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 11px;
    }}

    .example-id summary {{
      cursor: pointer;
      font-family: var(--mono);
    }}

    .example-id-value {{
      margin-top: 6px;
      font-family: var(--mono);
      word-break: break-all;
    }}

    .example-delta {{
      font-family: var(--mono);
      font-size: 13px;
      margin-bottom: 8px;
    }}

    .score-compare {{
      width: 100%;
      min-width: 0;
      border-collapse: separate;
      border-spacing: 0;
      margin-bottom: 10px;
      overflow: hidden;
      border: 1px solid rgba(217, 204, 185, 0.7);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.72);
      table-layout: fixed;
    }}

    .score-compare th,
    .score-compare td {{
      padding: 7px 9px;
      border-bottom: 1px solid rgba(217, 204, 185, 0.5);
      font-size: 12px;
      position: static;
      top: auto;
      z-index: auto;
      white-space: nowrap;
    }}

    .score-compare tr:last-child td {{
      border-bottom: 0;
    }}

    .score-compare th {{
      text-align: left;
      background: rgba(255, 248, 240, 0.95);
      color: var(--muted);
      font-weight: 600;
    }}

    .score-compare td {{
      font-family: var(--mono);
    }}

    .score-compare th:first-child,
    .score-compare td:first-child {{
      width: 42%;
      white-space: normal;
    }}

    .score-compare tr.is-focus td:first-child {{
      color: var(--accent);
      font-weight: 700;
    }}

    .delta-up {{
      color: var(--good);
    }}

    .delta-down {{
      color: var(--bad);
    }}

    .delta-flat {{
      color: var(--muted);
    }}

    .example-text {{
      font-size: 13px;
      line-height: 1.4;
      color: var(--ink);
      white-space: pre-wrap;
      word-break: break-word;
    }}

    .dist-chart {{
      background: rgba(255, 255, 255, 0.72);
      border-radius: 16px;
      padding: 14px;
      border: 1px solid rgba(217, 204, 185, 0.7);
    }}

    .dist-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      margin-bottom: 12px;
      font-size: 13px;
    }}

    .dist-legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}

    .dist-legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}

    .dist-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .dist-axis-label {{
      font-family: var(--mono);
      font-size: 12px;
      fill: var(--muted);
    }}

    .dist-grid {{
      stroke: rgba(82, 96, 109, 0.18);
      stroke-width: 1;
    }}

    .table-wrap {{
      overflow-x: auto;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.76);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}

    th,
    td {{
      padding: 11px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(217, 204, 185, 0.8);
      vertical-align: top;
    }}

    thead th {{
      background: rgba(255, 248, 240, 0.95);
      position: sticky;
      top: 0;
      z-index: 1;
    }}

    tbody tr:last-child td {{
      border-bottom: 0;
    }}

    td.value,
    th.value {{
      font-family: var(--mono);
      white-space: nowrap;
    }}

    .good {{
      color: var(--good);
    }}

    .bad {{
      color: var(--bad);
    }}

    .warn {{
      color: var(--warn);
    }}

    .footer-note {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 18px;
    }}

    .metric-label {{
      border-bottom: 1px dotted rgba(15, 118, 110, 0.55);
      cursor: help;
    }}

    .section-label {{
      border-bottom: 1px dotted rgba(181, 71, 8, 0.6);
      cursor: help;
    }}

    .tooltip-anchor {{
      position: relative;
      display: inline-flex;
      align-items: center;
    }}

    .tooltip-bubble {{
      position: absolute;
      left: 0;
      top: calc(100% + 8px);
      width: min(420px, 70vw);
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(31, 41, 51, 0.96);
      color: #f8fafc;
      box-shadow: 0 18px 44px rgba(15, 23, 42, 0.28);
      font-size: 13px;
      line-height: 1.45;
      z-index: 20;
      opacity: 0;
      visibility: hidden;
      transform: translateY(4px);
      transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
      pointer-events: none;
      font-weight: 400;
      white-space: normal;
    }}

    .tooltip-anchor:hover .tooltip-bubble,
    .tooltip-anchor:focus-within .tooltip-bubble {{
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
    }}

    .tooltip-formula {{
      margin-top: 6px;
      font-family: var(--mono);
      color: #d5f5f1;
    }}

    @media (max-width: 720px) {{
      main {{
        width: min(100vw - 16px, 1440px);
        margin-top: 16px;
      }}

      .hero,
      .summary-card,
      details.section > summary,
      .section-body {{
        padding-left: 16px;
        padding-right: 16px;
      }}

      .stat-list {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{escape(page_title)}</h1>
      <p>Interactive comparison of FAIR-J robustness metrics across multiple judge models. Expand any criterion to compare all models side by side for dataset-level noise, ranking consistency, and statistical tests.</p>
    </section>

    <section class="controls">
      <button type="button" data-open-all="section">Open all criteria</button>
      <button type="button" data-close-all="section">Collapse all criteria</button>
      <button type="button" data-open-all="dataset">Open all dataset metrics</button>
      <button type="button" data-open-all="ranking">Open all ranking metrics</button>
      <button type="button" data-close-all="metric">Collapse all inner sections</button>
    </section>

    <section class="summary-grid">
      {build_summary_cards(reports, criteria)}
    </section>

    <section class="overview-grid">
      {build_overview_cards(reports, criteria)}
    </section>

    {''.join(build_criterion_sections(reports, criteria))}

    {build_stat_tests_explorer(reports, criteria)}

    {build_example_viewer_section(reports, criteria)}

    <p class="footer-note">Lower noise metrics are better. Higher Kendall tau is better. For p-values, values below 0.05 suggest a statistically noticeable shift under rubric perturbation.</p>
  </main>
  <script>
    function setOpen(selector, open) {{
      document.querySelectorAll(selector).forEach((el) => {{
        el.open = open;
      }});
    }}

    document.querySelector('[data-open-all="section"]').addEventListener('click', () => setOpen('details.section', true));
    document.querySelector('[data-close-all="section"]').addEventListener('click', () => setOpen('details.section', false));
    document.querySelector('[data-open-all="dataset"]').addEventListener('click', () => setOpen('details.metric-dataset', true));
    document.querySelector('[data-open-all="ranking"]').addEventListener('click', () => setOpen('details.metric-ranking', true));
    document.querySelector('[data-close-all="metric"]').addEventListener('click', () => setOpen('details.metric-dataset, details.metric-ranking', false));

    const statCriterionSelector = document.querySelector('[data-stat-criterion-select]');
    const statMetricSelector = document.querySelector('[data-stat-metric-select]');
    if (statCriterionSelector && statMetricSelector) {{
      const syncStatScope = () => {{
        const scopeKey = `${{statCriterionSelector.value}}::${{statMetricSelector.value}}`;
        document.querySelectorAll('[data-stat-scope]').forEach((node) => {{
          node.classList.toggle('is-active', node.dataset.statScope === scopeKey);
        }});
      }};
      statCriterionSelector.addEventListener('change', syncStatScope);
      statMetricSelector.addEventListener('change', syncStatScope);
      syncStatScope();
    }}

    const exampleViewer = window.__FAIR_J_EXAMPLE_VIEWER__;
    if (exampleViewer) {{
      exampleViewer.init();
    }}
  </script>
</body>
</html>
"""


def build_summary_cards(reports: list[ModelReport], criteria: list[str]) -> str:
    cards: list[str] = []
    for report in reports:
        ds = report.data["dataset_level_scores"]
        rc = report.data["ranking_consistency"]
        avg_std_seed_default = mean_non_null(ds[criterion]["std_seed_default"] for criterion in criteria)
        avg_std = mean_non_null(ds[criterion]["std_total"] for criterion in criteria)
        avg_kendall_para = mean_non_null(rc[criterion]["kendall_paraphrases_mean"] for criterion in criteria)
        avg_kendall_del = mean_non_null(rc[criterion]["kendall_deletions_mean"] for criterion in criteria)
        significant_para = 0
        significant_del = 0
        for criterion in criteria:
            tests = ds[criterion]["stat_tests"]
            if safe_float(tests["paraphrases_vs_baseline"]["tests"]["paired_t_test"]["p_value"]) is not None and safe_float(tests["paraphrases_vs_baseline"]["tests"]["paired_t_test"]["p_value"]) < 0.05:
                significant_para += 1
            if safe_float(tests["deletions_vs_baseline"]["tests"]["paired_t_test"]["p_value"]) is not None and safe_float(tests["deletions_vs_baseline"]["tests"]["paired_t_test"]["p_value"]) < 0.05:
                significant_del += 1

        cards.append(
            "\n".join(
                [
                    '<article class="summary-card">',
                    f"<h2>{escape(report.label)}</h2>",
                    '<div class="stat-list">',
                    summary_stat("Avg std_seed_default", avg_std_seed_default),
                    summary_stat("Avg std_total", avg_std),
                    summary_stat("Avg Kendall paraphrases", avg_kendall_para),
                    summary_stat("Avg Kendall deletions", avg_kendall_del),
                    summary_stat("Significant paraphrase tests", significant_para, digits=0),
                    summary_stat("Significant deletion tests", significant_del, digits=0),
                    summary_stat("Criteria tracked", len(criteria), digits=0),
                    "</div>",
                    "</article>",
                ]
            )
        )
    return "".join(cards)


def build_overview_cards(reports: list[ModelReport], criteria: list[str]) -> str:
    return "".join(
        [
            build_overview_card(
                section_key="ranking_consistency",
                title="Ranking consistency (Kendall tau)",
                body=build_overview_table(
                    reports=reports,
                    rows=[
                        ("avg_kendall_paraphrases", "Average Kendall tau / paraphrases"),
                        ("avg_kendall_deletions", "Average Kendall tau / deletions"),
                        ("worst_kendall", "Worst Kendall tau over all criteria"),
                    ],
                    fetch=lambda report, metric_key: fetch_ranking_overview_value(report, metric_key, criteria),
                ),
            ),
            build_overview_card(
                section_key="dataset_level",
                title="Dataset-level shifts (bias + MAD + stat tests)",
                body=build_overview_table(
                    reports=reports,
                    rows=[
                        ("avg_std_seed_default", "Average std_seed_default"),
                        ("avg_mad_paraphrases", "Average MAD / paraphrases"),
                        ("avg_mad_deletions", "Average MAD / deletions"),
                        ("significant_paraphrase_tests", "Significant paired t-tests / paraphrases"),
                        ("significant_deletion_tests", "Significant paired t-tests / deletions"),
                    ],
                    fetch=lambda report, metric_key: fetch_dataset_overview_value(report, metric_key, criteria),
                ),
            ),
        ]
    )


def build_overview_card(section_key: str, title: str, body: str) -> str:
    explanation = SECTION_EXPLANATIONS[section_key]
    return (
        '<article class="summary-card overview-card">'
        f"<h2>{render_tooltip_label(title, explanation, 'section-label')}</h2>"
        f'<div class="meta">{escape(explanation)}</div>'
        f"{body}"
        "</article>"
    )


def build_overview_table(reports: list[ModelReport], rows: list[tuple[str, str]], fetch) -> str:
    head = "".join(f"<th>{escape(report.label)}</th>" for report in reports)
    body_rows: list[str] = []
    for metric_key, metric_label in rows:
        cells = "".join(
            f'<td class="value">{escape(format_number(fetch(report, metric_key)))}</td>'
            for report in reports
        )
        body_rows.append(f"<tr><td>{escape(metric_label)}</td>{cells}</tr>")
    return (
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr><th>Summary metric</th>"
        f"{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def fetch_ranking_overview_value(report: ModelReport, metric_key: str, criteria: list[str]) -> float | None:
    rc = report.data["ranking_consistency"]
    paraphrases = [rc[criterion]["kendall_paraphrases_mean"] for criterion in criteria]
    deletions = [rc[criterion]["kendall_deletions_mean"] for criterion in criteria]
    if metric_key == "avg_kendall_paraphrases":
        return mean_non_null(paraphrases)
    if metric_key == "avg_kendall_deletions":
        return mean_non_null(deletions)
    if metric_key == "worst_kendall":
        filtered = [value for value in paraphrases + deletions if value is not None]
        return min(filtered) if filtered else None
    raise KeyError(metric_key)


def fetch_dataset_overview_value(report: ModelReport, metric_key: str, criteria: list[str]) -> float | int | None:
    ds = report.data["dataset_level_scores"]
    if metric_key == "avg_std_seed_default":
        return mean_non_null(ds[criterion]["std_seed_default"] for criterion in criteria)
    if metric_key == "avg_mad_paraphrases":
        return mean_non_null(ds[criterion]["mad_paraphrases"] for criterion in criteria)
    if metric_key == "avg_mad_deletions":
        return mean_non_null(ds[criterion]["mad_deletions"] for criterion in criteria)
    if metric_key == "significant_paraphrase_tests":
        return sum(
            safe_float(ds[criterion]["stat_tests"]["paraphrases_vs_baseline"]["tests"]["paired_t_test"]["p_value"]) is not None
            and safe_float(ds[criterion]["stat_tests"]["paraphrases_vs_baseline"]["tests"]["paired_t_test"]["p_value"]) < 0.05
            for criterion in criteria
        )
    if metric_key == "significant_deletion_tests":
        return sum(
            safe_float(ds[criterion]["stat_tests"]["deletions_vs_baseline"]["tests"]["paired_t_test"]["p_value"]) is not None
            and safe_float(ds[criterion]["stat_tests"]["deletions_vs_baseline"]["tests"]["paired_t_test"]["p_value"]) < 0.05
            for criterion in criteria
        )
    raise KeyError(metric_key)


def summary_stat(label: str, value: float | int, digits: int = 4) -> str:
    return (
        '<div class="stat-item">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{format_number(value, digits=digits)}</div>'
        "</div>"
    )


def build_criterion_sections(reports: list[ModelReport], criteria: list[str]) -> list[str]:
    sections: list[str] = []
    for criterion in criteria:
        sections.append(
            "\n".join(
                [
                    '<details class="section">',
                    f"<summary>{escape(format_criterion_label(criterion))}</summary>",
                    '<div class="section-body">',
                    '<div class="badge-row">',
                    '<span class="badge">dataset-level</span>',
                    '<span class="badge">ranking consistency</span>',
                    "</div>",
                    build_metric_detail(
                        title="Dataset-level metrics",
                        css_class="metric-dataset",
                        body=build_dataset_table(reports, criterion),
                        open_by_default=True,
                    ),
                    build_metric_detail(
                        title="Ranking consistency",
                        css_class="metric-ranking",
                        body=build_ranking_table(reports, criterion),
                    ),
                    build_metric_detail(
                        title="Baseline score distribution",
                        css_class="metric-distribution",
                        body=build_distribution_section(reports, criterion),
                    ),
                    "</div>",
                    "</details>",
                ]
            )
        )
    return sections


def build_metric_detail(title: str, css_class: str, body: str, open_by_default: bool = False) -> str:
    open_attr = " open" if open_by_default else ""
    return (
        f'<details class="metric-block {css_class}"{open_attr}>'
        f"<summary><h3>{escape(title)}</h3></summary>"
        f"{body}"
        "</details>"
    )


def build_group_card(title: str, body: str) -> str:
    return (
        '<section class="group-card">'
        f"<h4>{escape(title)}</h4>"
        f"{body}"
        "</section>"
    )


def build_stat_tests_explorer(reports: list[ModelReport], criteria: list[str]) -> str:
    if not criteria:
        return ""
    selector = (
        '<div class="control-row">'
        '<div class="control-field">'
        '<label for="stat-criterion-select">Changed rubric criterion</label>'
        '<select id="stat-criterion-select" data-stat-criterion-select>'
        + "".join(
            f'<option value="{escape(option_id)}">{escape(option_label)}</option>'
            for option_id, option_label in build_stat_scope_options(criteria)
        )
        + "</select>"
        "</div>"
        '<div class="control-field">'
        '<label for="stat-metric-select">Metric</label>'
        '<select id="stat-metric-select" data-stat-metric-select>'
        '<option value="same_criterion">Same criterion score</option>'
        '<option value="total_score">Total score</option>'
        "</select>"
        "</div>"
        "</div>"
    )
    scopes = "".join(
        [
            (
                f'<div class="stat-scope{" is-active" if index == 0 and metric_mode == "same_criterion" else ""}" data-stat-scope="{escape(scope_id)}::{escape(metric_mode)}">'
                f"{build_stats_table(reports, scope_id, metric_mode)}"
                "</div>"
            )
            for index, (scope_id, _scope_label) in enumerate(build_stat_scope_options(criteria))
            for metric_mode in ("same_criterion", "total_score")
        ]
    )
    return (
        '<details class="section" open>'
        "<summary>Statistical tests explorer</summary>"
        '<div class="section-body">'
        '<div class="badge-row">'
        '<span class="badge">paired tests</span>'
        '<span class="badge">changed criterion selector</span>'
        '<span class="badge">total score</span>'
        "</div>"
        f"{selector}"
        '<p class="muted-note">`Same criterion score` compares the score of the changed rubric criterion when that score exists. For deletions of the selected criterion, the per-criterion block is shown as N/A and `total score` remains available.</p>'
        f"{scopes}"
        "</div>"
        "</details>"
    )


def build_dataset_table(reports: list[ModelReport], criterion: str) -> str:
    tables = [
        build_group_card(
            title="Overall stability",
            body=build_matrix_table(
                reports=reports,
                row_specs=[
                    ("baseline_score_std", "Baseline score std"),
                    ("std_seed_default", "Seed noise / baseline"),
                    ("std_total", "Std across all perturbations"),
                ],
                fetch_value=lambda report, metric: report.data["dataset_level_scores"][criterion][metric],
                value_formatter=lambda value, _metric_key: format_number(value),
            ),
        ),
        build_group_card(
            title="Paraphrases vs baseline",
            body=build_matrix_table(
                reports=reports,
                row_specs=[
                    ("std_paraphrases", "Std across paraphrase variants"),
                    ("bias_paraphrases", "Mean difference"),
                    ("mad_paraphrases", "MAD"),
                    ("bias_to_mae_ratio_paraphrases", "Bias / MAD"),
                ],
                fetch_value=lambda report, metric: report.data["dataset_level_scores"][criterion][metric],
                value_formatter=lambda value, _metric_key: format_number(value),
            ),
        ),
        build_group_card(
            title="Deletions vs baseline",
            body=build_matrix_table(
                reports=reports,
                row_specs=[
                    ("std_deletions", "Std across deletion variants"),
                    ("bias_deletions", "Mean difference"),
                    ("mad_deletions", "MAD"),
                    ("bias_to_mae_ratio_deletions", "Bias / MAD"),
                ],
                fetch_value=lambda report, metric: report.data["dataset_level_scores"][criterion][metric],
                value_formatter=lambda value, _metric_key: format_number(value),
            ),
        ),
    ]
    return f'<div class="group-grid">{"".join(tables)}</div>'


def build_ranking_table(reports: list[ModelReport], criterion: str) -> str:
    tables = [
        build_group_card(
            title="Paraphrases vs baseline",
            body=build_matrix_table(
                reports=reports,
                row_specs=[
                    ("kendall_paraphrases_mean", "Kendall tau-b / mean"),
                    ("kendall_paraphrases_std", "Kendall tau-b / std"),
                    ("kendall_paraphrases_min", "Kendall tau-b / min"),
                    ("kendall_paraphrases_max", "Kendall tau-b / max"),
                    ("concordant_paraphrases_pooled", "Concordant pairs / pooled"),
                    ("discordant_paraphrases_pooled", "Discordant pairs / pooled"),
                    ("gamma_paraphrases_pooled", "Goodman-Kruskal gamma / pooled"),
                    ("tie_pct_paraphrases_mean", "Tied pairs % / mean"),
                ],
                fetch_value=lambda report, metric: report.data["ranking_consistency"][criterion][metric],
                value_formatter=lambda value, _metric_key: format_number(value),
            ),
        ),
        build_group_card(
            title="Deletions vs baseline",
            body=build_matrix_table(
                reports=reports,
                row_specs=[
                    ("kendall_deletions_mean", "Kendall tau-b / mean"),
                    ("kendall_deletions_std", "Kendall tau-b / std"),
                    ("kendall_deletions_min", "Kendall tau-b / min"),
                    ("kendall_deletions_max", "Kendall tau-b / max"),
                    ("concordant_deletions_pooled", "Concordant pairs / pooled"),
                    ("discordant_deletions_pooled", "Discordant pairs / pooled"),
                    ("gamma_deletions_pooled", "Goodman-Kruskal gamma / pooled"),
                    ("tie_pct_deletions_mean", "Tied pairs % / mean"),
                ],
                fetch_value=lambda report, metric: report.data["ranking_consistency"][criterion][metric],
                value_formatter=lambda value, _metric_key: format_number(value),
            ),
        ),
    ]
    return f'<div class="group-grid">{"".join(tables)}</div>'


def build_stats_table(reports: list[ModelReport], changed_criterion: str, metric_mode: str) -> str:
    cards: list[str] = []
    for group_key, group_label in STAT_TEST_GROUPS:
        rows = [
            (f"{group_key}:n_pairs", "Paired observations"),
            (f"{group_key}:mean_difference", "Mean difference"),
            (f"{group_key}:mde_mean_difference", "MDE mean difference"),
            (f"{group_key}:effect_size_dz", "Effect size (Cohen's dz)"),
            (f"{group_key}:mde_effect_size_dz", "MDE effect size (dz)"),
            (f"{group_key}:observed_abs_mean_difference_over_mde", "|Mean diff| / MDE"),
            (f"{group_key}:paired_permutation_test", "Permutation p-value"),
            (f"{group_key}:wilcoxon_signed_rank", "Wilcoxon p-value"),
            (f"{group_key}:paired_t_test", "Paired t-test p-value"),
        ]
        cards.append(
            build_group_card(
                title=group_label,
                body=build_matrix_table(
                    reports=reports,
                    row_specs=rows,
                    fetch_value=lambda report, metric: fetch_stat_metric(
                        report,
                        changed_criterion,
                        metric_mode,
                        metric,
                    ),
                    value_formatter=format_stat_metric_cell,
                    raw_html=True,
                ),
            )
        )
    return f'<div class="group-grid">{"".join(cards)}</div>'


def build_top_examples_section(reports: list[ModelReport]) -> str:
    cards: list[str] = []
    for report in reports:
        examples = find_top_score_jumps(report, limit=3)
        if not examples:
            cards.append(
                '<div class="mini-card">'
                f"<h4>{escape(report.label)}</h4>"
                '<div class="example-text">No comparable examples found.</div>'
                "</div>"
            )
            continue

        items: list[str] = []
        for item in examples:
            items.append(
                '<div class="example-item">'
                '<div class="example-header">'
                f'<span class="perturbation-badge">{escape(item["perturbation"])}</span>'
                f'<span class="example-meta">{escape(item["example_label"])}</span>'
                "</div>"
                '<details class="example-id">'
                "<summary>raw example id</summary>"
                f'<div class="example-id-value">{escape(item["raw_example_id"])}</div>'
                "</details>"
                f'<div class="example-delta">criterion={escape(item["jump_criterion"])} | baseline={escape(item["baseline_score"])} | perturbation={escape(item["perturbation_score"])} | Δ=baseline-perturbation={escape(item["delta"])}</div>'
                f'{render_score_comparison_table(item["score_rows"], str(item["jump_criterion"]))}'
                f'<div class="example-text"><strong>Candidate:</strong> {escape(item["candidate"])}</div>'
                "</div>"
            )

        cards.append(
            '<div class="mini-card">'
            f"<h4>{escape(report.label)}</h4>"
            '<div class="example-list">'
            f"{''.join(items)}"
            "</div>"
            "</div>"
        )

    return f'<div class="card-grid">{"".join(cards)}</div>'


def build_distribution_section(reports: list[ModelReport], criterion: str) -> str:
    return build_overlay_distribution_chart(reports, criterion)


def build_example_viewer_section(reports: list[ModelReport], criteria: list[str]) -> str:
    payload = build_example_viewer_payload(reports, criteria)
    if not payload["models"]:
        return ""
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (
        '<details class="section" open>'
        "<summary>Side-by-side example viewer</summary>"
        '<div class="section-body">'
        '<div class="badge-row">'
        '<span class="badge">single example</span>'
        '<span class="badge">model selector</span>'
        '<span class="badge">criterion selector</span>'
        '<span class="badge">rubric diff</span>'
        "</div>"
        '<div class="viewer-shell">'
        '<div class="control-row">'
        '<div class="control-field">'
        '<label for="viewer-model-select">Model</label>'
        '<select id="viewer-model-select" data-viewer-model></select>'
        "</div>"
        '<div class="control-field">'
        '<label for="viewer-criterion-select">Changed rubric criterion</label>'
        '<select id="viewer-criterion-select" data-viewer-criterion></select>'
        "</div>"
        '<div class="control-field">'
        '<label for="viewer-sort-select">Sort</label>'
        '<select id="viewer-sort-select" data-viewer-sort>'
        '<option value="largest_delta">Largest absolute delta</option>'
        '<option value="example_id">Example id</option>'
        "</select>"
        "</div>"
        "</div>"
        '<div class="viewer-toolbar">'
        '<div class="viewer-nav">'
        '<button type="button" data-viewer-prev aria-label="Previous example">←</button>'
        '<button type="button" data-viewer-next aria-label="Next example">→</button>'
        "</div>"
        '<div class="viewer-counter" data-viewer-counter></div>'
        "</div>"
        '<div class="viewer-card" data-viewer-card></div>'
        "</div>"
        f"<script>{build_example_viewer_script(payload_json)}</script>"
        "</div>"
        "</details>"
    )


def build_example_viewer_payload(reports: list[ModelReport], criteria: list[str]) -> dict[str, object]:
    models = []
    for report in reports:
        items = build_example_viewer_items(report, criteria)
        if not items:
            continue
        models.append(
            {
                "id": report.label,
                "label": report.label,
                "items": items,
            }
        )
    return {
        "models": models,
        "criteria": [
            {"id": "__max__", "label": "All perturbations / largest absolute delta"},
            *[
                {"id": criterion_id, "label": format_criterion_label(criterion_id)}
                for criterion_id in criteria
            ],
        ],
    }


def build_example_viewer_items(report: ModelReport, criteria: list[str]) -> list[dict[str, object]]:
    baseline_rubric = report.variants.get("baseline", [])
    if not baseline_rubric:
        return []

    score_index: dict[tuple[str, int, str, str], int] = {}
    grouped_rows: dict[tuple[str, int, str], dict[str, int]] = {}
    for row in report.score_rows:
        criterion_id = str(row["criterion_id"])
        if criterion_id == TOTAL_CRITERION_ID:
            continue
        key = (
            str(row["example_id"]),
            int(row["seed"]),
            str(row["perturbation"]),
            criterion_id,
        )
        score_value = int(round(float(row["score"])))
        score_index[key] = score_value
        group_key = (str(row["example_id"]), int(row["seed"]), str(row["perturbation"]))
        grouped_rows.setdefault(group_key, {})[criterion_id] = score_value

    items: list[dict[str, object]] = []
    for (example_id, seed, perturbation), perturbation_scores in grouped_rows.items():
        if perturbation == "baseline":
            continue

        criterion_deltas: dict[str, int] = {}
        for criterion_id in criteria:
            baseline_score = score_index.get((example_id, seed, "baseline", criterion_id))
            perturbed_score = perturbation_scores.get(criterion_id)
            if baseline_score is None or perturbed_score is None:
                continue
            criterion_deltas[criterion_id] = perturbed_score - baseline_score

        if not criterion_deltas:
            continue

        focus_criterion = max(
            criterion_deltas,
            key=lambda criterion_id: (abs(criterion_deltas[criterion_id]), criterion_id),
        )
        example = report.dataset_examples.get(example_id, {})
        metadata = example.get("metadata", {})
        rubric_compare = build_rubric_compare_payload(
            baseline_rubric=baseline_rubric,
            perturbation_rubric=report.variants.get(perturbation, baseline_rubric),
            perturbation=perturbation,
        )
        items.append(
            {
                "raw_example_id": example_id,
                "example_label": format_example_label(example_id, metadata, seed),
                "perturbation": perturbation,
                "perturbation_label": format_perturbation_label(perturbation),
                "focus_criterion": focus_criterion,
                "criterion_deltas": criterion_deltas,
                "max_abs_delta": max(abs(delta) for delta in criterion_deltas.values()),
                "score_rows": build_dynamic_example_score_rows(
                    baseline_rubric=baseline_rubric,
                    score_index=score_index,
                    example_id=example_id,
                    seed=seed,
                    perturbation=perturbation,
                ),
                "summary": str(example.get("candidate", "")),
                "rubric_compare": rubric_compare,
            }
        )

    grouped: dict[str, list[dict[str, object]]] = {criterion_id: [] for criterion_id in criteria}
    for item in items:
        grouped[item["rubric_compare"]["changed_criterion"]].append(item)

    limited: list[dict[str, object]] = []
    for criterion_id in criteria:
        group_items = grouped.get(criterion_id, [])
        group_items.sort(
            key=lambda item: (-int(item["max_abs_delta"]), str(item["raw_example_id"]), str(item["perturbation"]))
        )
        limited.extend(group_items[:50])
    return limited


def build_rubric_compare_payload(
    baseline_rubric: list[dict[str, str]],
    perturbation_rubric: list[dict[str, str]],
    perturbation: str,
) -> dict[str, object]:
    perturbation_map = {item["id"]: item["text"] for item in perturbation_rubric}
    change = parse_perturbation(perturbation)
    changed_id = change["criterion_id"]
    baseline_items: list[dict[str, str]] = []
    perturbation_items: list[dict[str, str]] = []

    for criterion in baseline_rubric:
        criterion_id = criterion["id"]
        baseline_items.append(
            {
                "id": criterion_id,
                "text": criterion["text"],
                "status": "focus" if criterion_id == changed_id else "normal",
            }
        )

        altered_text = perturbation_map.get(criterion_id)
        if altered_text is None:
            perturbation_items.append(
                {
                    "id": criterion_id,
                    "text": "[removed from perturbation rubric]",
                    "status": "removed",
                }
            )
            continue

        status = "focus" if criterion_id == changed_id and altered_text != criterion["text"] else "normal"
        perturbation_items.append(
            {
                "id": criterion_id,
                "text": altered_text,
                "status": status,
            }
        )

    return {
        "kind": change["kind"],
        "changed_criterion": changed_id,
        "baseline": baseline_items,
        "perturbation": perturbation_items,
    }


def build_dynamic_example_score_rows(
    baseline_rubric: list[dict[str, str]],
    score_index: dict[tuple[str, int, str, str], int],
    example_id: str,
    seed: int,
    perturbation: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for criterion in baseline_rubric:
        criterion_id = criterion["id"]
        baseline_score = score_index.get((example_id, seed, "baseline", criterion_id))
        perturbed_score = score_index.get((example_id, seed, perturbation, criterion_id))
        delta = None if baseline_score is None or perturbed_score is None else perturbed_score - baseline_score
        rows.append(
            {
                "criterion": criterion_id,
                "baseline": format_integer_score(baseline_score),
                "perturbation": format_integer_score(perturbed_score),
                "delta": format_integer_delta(delta),
                "delta_value": delta,
                "delta_class": classify_display_delta(delta),
            }
        )
    return rows


def build_example_viewer_script(payload_json: str) -> str:
    return f"""
window.__FAIR_J_EXAMPLE_VIEWER_DATA__ = {payload_json};
window.__FAIR_J_EXAMPLE_VIEWER__ = {{
  state: {{
    modelId: null,
    criterionId: "__max__",
    sortMode: "largest_delta",
    index: 0,
  }},
  escapeHtml(value) {{
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }},
  shorten(value, maxLength) {{
    const normalized = String(value).replace(/\\s+/g, " ").trim();
    if (normalized.length <= maxLength) {{
      return normalized;
    }}
    return normalized.slice(0, maxLength - 1).trimEnd() + "…";
  }},
  init() {{
    const data = window.__FAIR_J_EXAMPLE_VIEWER_DATA__;
    if (!data || !data.models.length) {{
      return;
    }}
    this.modelSelect = document.querySelector("[data-viewer-model]");
    this.criterionSelect = document.querySelector("[data-viewer-criterion]");
    this.sortSelect = document.querySelector("[data-viewer-sort]");
    this.counter = document.querySelector("[data-viewer-counter]");
    this.card = document.querySelector("[data-viewer-card]");
    this.prevButton = document.querySelector("[data-viewer-prev]");
    this.nextButton = document.querySelector("[data-viewer-next]");

    this.modelSelect.innerHTML = data.models.map((model) => `<option value="${{this.escapeHtml(model.id)}}">${{this.escapeHtml(model.label)}}</option>`).join("");
    this.criterionSelect.innerHTML = data.criteria.map((criterion) => `<option value="${{this.escapeHtml(criterion.id)}}">${{this.escapeHtml(criterion.label)}}</option>`).join("");
    this.state.modelId = data.models[0].id;

    this.modelSelect.addEventListener("change", () => {{
      this.state.modelId = this.modelSelect.value;
      this.state.index = 0;
      this.render();
    }});
    this.criterionSelect.addEventListener("change", () => {{
      this.state.criterionId = this.criterionSelect.value;
      this.state.index = 0;
      this.render();
    }});
    this.sortSelect.addEventListener("change", () => {{
      this.state.sortMode = this.sortSelect.value;
      this.state.index = 0;
      this.render();
    }});
    this.prevButton.addEventListener("click", () => {{
      const items = this.getItems();
      if (!items.length) {{
        return;
      }}
      this.state.index = (this.state.index - 1 + items.length) % items.length;
      this.render();
    }});
    this.nextButton.addEventListener("click", () => {{
      const items = this.getItems();
      if (!items.length) {{
        return;
      }}
      this.state.index = (this.state.index + 1) % items.length;
      this.render();
    }});
    this.render();
  }},
  getItems() {{
    const data = window.__FAIR_J_EXAMPLE_VIEWER_DATA__;
    const model = data.models.find((candidate) => candidate.id === this.state.modelId) || data.models[0];
    const criterionId = this.state.criterionId;
    const sortMode = this.state.sortMode;
    const filtered = model.items.filter((item) => criterionId === "__max__" || item.rubric_compare.changed_criterion === criterionId);
    filtered.sort((left, right) => {{
      if (sortMode === "example_id") {{
        return left.raw_example_id.localeCompare(right.raw_example_id);
      }}
      const leftScore = left.max_abs_delta;
      const rightScore = right.max_abs_delta;
      if (rightScore !== leftScore) {{
        return rightScore - leftScore;
      }}
      return left.raw_example_id.localeCompare(right.raw_example_id);
    }});
    return filtered;
  }},
  render() {{
    const items = this.getItems();
    if (!items.length) {{
      this.counter.textContent = "0 / 0";
      this.card.innerHTML = '<div class="muted-note">No comparable examples for this selection.</div>';
      return;
    }}
    if (this.state.index >= items.length) {{
      this.state.index = 0;
    }}
    const item = items[this.state.index];
    const focusCriterion = item.focus_criterion;
    const displayedDelta = item.criterion_deltas[focusCriterion];
    const summaryPreview = this.shorten(item.summary, 180);
    this.counter.textContent = `${{this.state.index + 1}} / ${{items.length}}`;

    const scoreRows = item.score_rows.map((row) => `
      <tr class="${{row.criterion === focusCriterion ? "is-focus" : ""}}">
        <td>${{this.escapeHtml(row.criterion)}}</td>
        <td>${{this.escapeHtml(row.baseline)}}</td>
        <td>${{this.escapeHtml(row.perturbation)}}</td>
        <td><span class="${{this.escapeHtml(row.delta_class)}}">${{this.escapeHtml(row.delta)}}</span></td>
      </tr>
    `).join("");

    const renderRubric = (title, items) => `
      <section class="rubric-panel">
        <h5>${{this.escapeHtml(title)}}</h5>
        <ul class="rubric-list">
          ${{items.map((entry) => `
            <li class="rubric-item ${{entry.status === "focus" ? "is-focus" : entry.status === "removed" ? "is-removed" : ""}}">
              <strong>${{this.escapeHtml(entry.id)}}</strong>
              <span>${{this.escapeHtml(entry.text)}}</span>
            </li>
          `).join("")}}
        </ul>
      </section>
    `;

    this.card.innerHTML = `
      <div class="viewer-title">
        <h4>${{this.escapeHtml(item.perturbation_label)}}</h4>
        <span class="perturbation-badge">max Δ on ${{this.escapeHtml(focusCriterion)}}</span>
      </div>
      <div class="viewer-meta">${{this.escapeHtml(item.example_label)}} | Δ perturbation-baseline = ${{this.escapeHtml(displayedDelta >= 0 ? "+" + displayedDelta : String(displayedDelta))}}</div>
      <table class="score-compare">
        <thead><tr><th>Criterion</th><th>Baseline</th><th>Perturbation</th><th>Δ</th></tr></thead>
        <tbody>${{scoreRows}}</tbody>
      </table>
      <div class="rubric-grid">
        ${{renderRubric("Original rubric", item.rubric_compare.baseline)}}
        ${{renderRubric(item.rubric_compare.kind === "paraphrase" ? "Changed rubric (paraphrase)" : "Changed rubric", item.rubric_compare.perturbation)}}
      </div>
      <details class="viewer-summary">
        <summary>
          <span class="viewer-summary-title">Evaluated summary</span>
          <span class="viewer-summary-preview">${{this.escapeHtml(summaryPreview || "Click to expand the full summary text.")}}</span>
        </summary>
        <div class="viewer-summary-body">
          <div class="example-text">${{this.escapeHtml(item.summary)}}</div>
        </div>
      </details>
      <details class="example-id">
        <summary>Raw example id</summary>
        <div class="example-id-value">${{this.escapeHtml(item.raw_example_id)}}</div>
      </details>
    `;
  }},
}};
"""


def build_matrix_table(
    reports: list[ModelReport],
    row_specs: list[tuple[str, str]],
    fetch_value,
    value_formatter,
    raw_html: bool = False,
) -> str:
    head = "".join(f"<th>{escape(report.label)}</th>" for report in reports)
    body_rows: list[str] = []
    for metric_key, metric_label in row_specs:
        cells = []
        for report in reports:
            rendered = value_formatter(fetch_value(report, metric_key), metric_key)
            if raw_html:
                cells.append(f'<td class="value">{rendered}</td>')
            else:
                cells.append(f'<td class="value">{escape(rendered)}</td>')
        label_html = render_metric_label(metric_key, metric_label)
        body_rows.append(
            "<tr>"
            f"<td>{label_html}</td>"
            + "".join(cells)
            + "</tr>"
        )

    return (
        '<div class="table-wrap">'
        "<table>"
        "<thead><tr><th>Metric</th>"
        f"{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def prettify_metric_name(metric: str) -> str:
    return metric.replace("_", " ")


def render_metric_label(metric_key: str, metric_label: str) -> str:
    explanation = METRIC_EXPLANATIONS.get(metric_key)
    if explanation is None:
        return escape(metric_label)
    return render_tooltip_label(metric_label, explanation, "metric-label")


def render_tooltip_label(label: str, tooltip_html: str, css_class: str) -> str:
    return (
        f'<span class="tooltip-anchor {css_class}" tabindex="0">'
        f"{escape(label)}"
        f'<span class="tooltip-bubble">{tooltip_html}</span>'
        "</span>"
    )


def classify_p_value(value: float) -> str:
    if value < 0.01:
        return "bad"
    if value < 0.05:
        return "warn"
    return "good"


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric != numeric:
        return None
    return numeric


def format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.0001:
        return f"{value:.4e}"
    return f"{value:.{digits}f}"


def format_integer_score(value: float | int | None) -> str:
    if value is None:
        return "—"
    return str(int(round(float(value))))


def format_integer_delta(value: float | int | None) -> str:
    if value is None:
        return "—"
    rounded = int(round(float(value)))
    return f"{rounded:+d}"


def classify_display_delta(delta: int | None) -> str:
    if delta is None:
        return "delta-flat"
    if delta > 0:
        return "delta-up"
    if delta < 0:
        return "delta-down"
    return "delta-flat"


def format_criterion_label(criterion_id: str) -> str:
    if criterion_id == TOTAL_CRITERION_ID:
        return "Total score"
    return criterion_id.replace("_", " ").title()


def build_stat_scope_options(criteria: list[str]) -> list[tuple[str, str]]:
    return [(criterion_id, format_criterion_label(criterion_id)) for criterion_id in criteria]


def fetch_stat_metric(
    report: ModelReport,
    changed_criterion: str,
    metric_mode: str,
    metric: str,
) -> float | int | None:
    group_key, suffix = metric.split(":", 1)
    group = compute_stat_test_group(report, changed_criterion, metric_mode, group_key)
    if suffix == "n_pairs":
        return group["n_pairs"]
    if suffix in STAT_TEST_VALUE_KEYS:
        return group[suffix]
    return group["tests"][suffix]["p_value"]


def format_stat_metric_cell(value: float | int | None, metric_key: str) -> str:
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "N/A"
    if metric_key.endswith(("paired_permutation_test", "wilcoxon_signed_rank", "paired_t_test")):
        css_class = classify_p_value(float(value))
        return f'<span class="{css_class}">{format_number(float(value), digits=6)}</span>'
    if metric_key.endswith("observed_abs_mean_difference_over_mde"):
        css_class = "good" if float(value) >= 1 else "warn"
        suffix = "(>1)" if float(value) >= 1 else "(<1)"
        return f'<span class="{css_class}">{format_number(float(value), digits=6)} {escape(suffix)}</span>'
    return escape(format_number(float(value), digits=6))


def parse_perturbation(perturbation: str) -> dict[str, str]:
    parts = perturbation.split("__")
    if perturbation.startswith("paraphrase__") and len(parts) >= 2:
        return {"kind": "paraphrase", "criterion_id": parts[1]}
    if perturbation.startswith("delete__") and len(parts) >= 2:
        return {"kind": "deletion", "criterion_id": parts[1]}
    return {"kind": "other", "criterion_id": parts[1] if len(parts) >= 2 else perturbation}


def format_perturbation_label(perturbation: str) -> str:
    parsed = parse_perturbation(perturbation)
    criterion_label = format_criterion_label(parsed["criterion_id"])
    if parsed["kind"] == "paraphrase":
        return f"Paraphrase / {criterion_label}"
    if parsed["kind"] == "deletion":
        return f"Deletion / {criterion_label}"
    return perturbation.replace("__", " / ")


def compute_stat_test_group(
    report: ModelReport,
    changed_criterion: str,
    metric_mode: str,
    group_key: str,
) -> dict[str, object]:
    if metric_mode == "same_criterion" and group_key == "deletions_vs_baseline":
        return empty_stat_block()

    score_rows = report.score_rows
    baseline_scores: dict[tuple[str, int], float] = {}
    perturbation_scores: dict[tuple[str, int], float] = {}

    if metric_mode == "same_criterion":
        criterion_id = changed_criterion
        perturbation_prefix = "paraphrase__" if group_key == "paraphrases_vs_baseline" else "delete__"
        for row in score_rows:
            row_criterion = str(row["criterion_id"])
            perturbation = str(row["perturbation"])
            if row_criterion != criterion_id:
                continue
            key = (str(row["example_id"]), int(row["seed"]))
            if perturbation == "baseline":
                baseline_scores[key] = float(row["score"])
                continue
            if perturbation_matches_changed_criterion(perturbation, group_key, changed_criterion):
                perturbation_scores[key] = float(row["score"])
    else:
        totals: dict[tuple[str, int, str], float] = {}
        for row in score_rows:
            if str(row["criterion_id"]) == TOTAL_CRITERION_ID:
                continue
            key = (str(row["example_id"]), int(row["seed"]), str(row["perturbation"]))
            totals[key] = totals.get(key, 0.0) + float(row["score"])

        for (example_id, seed, perturbation), total_score in totals.items():
            key = (example_id, seed)
            if perturbation == "baseline":
                baseline_scores[key] = total_score
            elif perturbation_matches_changed_criterion(perturbation, group_key, changed_criterion):
                perturbation_scores[key] = total_score

    keys = sorted(set(baseline_scores) & set(perturbation_scores))
    comparison = {
        "baseline": [baseline_scores[key] for key in keys],
        "perturbation": [perturbation_scores[key] for key in keys],
        "differences": [baseline_scores[key] - perturbation_scores[key] for key in keys],
    }
    if not comparison["differences"]:
        return empty_stat_block()
    return build_stat_test_block(comparison)


def perturbation_matches_changed_criterion(
    perturbation: str,
    group_key: str,
    changed_criterion: str,
) -> bool:
    if group_key == "paraphrases_vs_baseline":
        return perturbation.startswith(f"paraphrase__{changed_criterion}__")
    if group_key == "deletions_vs_baseline":
        return perturbation == f"delete__{changed_criterion}"
    return False


def empty_stat_block() -> dict[str, object]:
    return {
        "tested_quantity": "difference_vs_zero",
        "alternative": "two_sided",
        "n_pairs": 0,
        "mean_difference": None,
        "mde_mean_difference": None,
        "effect_size_dz": None,
        "mde_effect_size_dz": None,
        "observed_abs_mean_difference_over_mde": None,
        "alpha": None,
        "target_power": None,
        "tests": {
            "paired_permutation_test": {"p_value": None},
            "wilcoxon_signed_rank": {"p_value": None},
            "paired_t_test": {"p_value": None},
        },
    }


def escape(value: object) -> str:
    return html.escape(str(value))


def mean_non_null(values) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return mean(filtered)


def find_top_score_jumps(
    report: ModelReport,
    limit: int,
) -> list[dict[str, object]]:
    score_index: dict[tuple[str, int, str, str], int] = {}
    jumps: list[dict[str, object]] = []
    seen_groups: set[tuple[str, int, str]] = set()

    for row in report.score_rows:
        score_index[
            (
                str(row["example_id"]),
                int(row["seed"]),
                str(row["perturbation"]),
                str(row["criterion_id"]),
            )
        ] = int(round(float(row["score"])))

    for row in report.score_rows:
        if row["perturbation"] == "baseline":
            continue

        key = (str(row["example_id"]), int(row["seed"]))
        perturbation = str(row["perturbation"])
        group_key = (key[0], key[1], perturbation)
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)

        score_rows = build_example_score_rows(score_index, key[0], key[1], perturbation)
        comparable_rows = [score_row for score_row in score_rows if score_row["delta_value"] is not None]
        if not comparable_rows:
            continue

        jump_row = max(
            comparable_rows,
            key=lambda score_row: (abs(int(score_row["delta_value"])), str(score_row["criterion"])),
        )
        example = report.dataset_examples.get(str(row["example_id"]), {})
        metadata = example.get("metadata", {})
        jumps.append(
            {
                "example_id": str(row["example_id"]),
                "raw_example_id": str(row["example_id"]),
                "example_label": format_example_label(str(row["example_id"]), metadata, int(row["seed"])),
                "perturbation": perturbation,
                "jump_criterion": str(jump_row["criterion"]),
                "baseline_score": str(jump_row["baseline"]),
                "perturbation_score": str(jump_row["perturbation"]),
                "delta": str(jump_row["delta"]),
                "abs_delta": abs(int(jump_row["delta_value"])),
                "score_rows": score_rows,
                "candidate": shorten_text(str(example.get("candidate", "")), 220),
            }
        )

    jumps.sort(
        key=lambda item: (
            -float(item["abs_delta"]),
            str(item["example_id"]),
            str(item["perturbation"]),
        )
    )
    return jumps[:limit]


def collect_baseline_distribution(report: ModelReport, criterion: str) -> dict[int, int]:
    distribution: dict[int, int] = {}
    for row in report.score_rows:
        if row["criterion_id"] != criterion or row["perturbation"] != "baseline":
            continue
        score = int(round(float(row["score"])))
        distribution[score] = distribution.get(score, 0) + 1
    return distribution


def build_overlay_distribution_chart(reports: list[ModelReport], criterion: str) -> str:
    color_palette = ["#0f766e", "#b54708", "#3b82f6", "#b42318", "#7c3aed", "#027a48"]
    distributions = [collect_baseline_distribution(report, criterion) for report in reports]

    all_scores = sorted({score for distribution in distributions for score in distribution})
    if not all_scores:
        return '<div class="dist-chart">No baseline distribution available.</div>'

    total_count = max(sum(distribution.values()) for distribution in distributions)
    max_pct = 0.0
    for distribution in distributions:
        for score in all_scores:
            pct = 100 * distribution.get(score, 0) / total_count
            if pct > max_pct:
                max_pct = pct
    y_max = max(10.0, float(int((max_pct + 9.999) // 10) * 10))

    width = 760
    height = 320
    left = 54
    right = 24
    top = 24
    bottom = 42
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_pos(score: int) -> float:
        if len(all_scores) == 1:
            return left + plot_width / 2
        return left + (score - all_scores[0]) * plot_width / (all_scores[-1] - all_scores[0])

    def y_pos(pct: float) -> float:
        return top + plot_height * (1 - pct / y_max)

    grid_lines: list[str] = []
    for y_tick in range(0, int(y_max) + 1, 10):
        y = y_pos(float(y_tick))
        grid_lines.append(
            f'<line class="dist-grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"></line>'
        )
        grid_lines.append(
            f'<text class="dist-axis-label" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{y_tick}%</text>'
        )
    for score in all_scores:
        x = x_pos(score)
        grid_lines.append(
            f'<line class="dist-grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}"></line>'
        )
        grid_lines.append(
            f'<text class="dist-axis-label" x="{x:.2f}" y="{height - bottom + 22}" text-anchor="middle">{score}</text>'
        )

    grid_lines.append(
        f'<text class="dist-axis-label" x="{left + plot_width / 2:.2f}" y="{height - 8}" text-anchor="middle">Score</text>'
    )
    grid_lines.append(
        f'<text class="dist-axis-label" x="16" y="{top + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 16 {top + plot_height / 2:.2f})">% of examples</text>'
    )

    series_layers: list[str] = []
    legend_items: list[str] = []
    for index, (report, distribution) in enumerate(zip(reports, distributions)):
        color = color_palette[index % len(color_palette)]
        points = []
        circles = []
        for score in all_scores:
            pct = 100 * distribution.get(score, 0) / total_count
            x = x_pos(score)
            y = y_pos(pct)
            points.append(f"{x:.2f},{y:.2f}")
            circles.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}">'
                f"<title>{escape(report.label)} | score={score} | count={distribution.get(score, 0)} | pct={pct:.1f}%</title>"
                "</circle>"
            )
        series_layers.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"></polyline>'
            + "".join(circles)
        )
        legend_items.append(
            '<span class="dist-legend-item">'
            f'<span class="dist-legend-swatch" style="background:{color}"></span>'
            f"{escape(report.label)}"
            "</span>"
        )

    svg = (
        f'<svg class="dist-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Baseline score distribution overlay chart">'
        + "".join(grid_lines)
        + "".join(series_layers)
        + "</svg>"
    )
    return (
        '<div class="dist-chart">'
        f'<div class="dist-legend">{"".join(legend_items)}</div>'
        f"{svg}"
        "</div>"
    )


def shorten_text(value: str, max_length: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "..."


def format_example_label(example_id: str, metadata: object, seed: int) -> str:
    meta = metadata if isinstance(metadata, dict) else {}
    source_id = str(meta.get("source_id", example_id)).split("__", 1)[0]
    source_short = source_id[:10]
    model_id = str(meta.get("model_id", example_id.rsplit("__", 1)[-1]))
    return f"source {source_short} · {model_id} · seed {seed}"


def build_example_score_rows(
    score_index: dict[tuple[str, int, str, str], int],
    example_id: str,
    seed: int,
    perturbation: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    criterion_ids = sorted(
        {
            criterion_id
            for candidate_example_id, candidate_seed, _candidate_perturbation, criterion_id in score_index
            if candidate_example_id == example_id and candidate_seed == seed
        }
    )
    for criterion_id in criterion_ids:
        baseline = score_index.get((example_id, seed, "baseline", criterion_id))
        perturbed = score_index.get((example_id, seed, perturbation, criterion_id))
        delta = None if baseline is None or perturbed is None else baseline - perturbed
        rows.append(
            {
                "criterion": criterion_id,
                "baseline": format_integer_score(baseline),
                "perturbation": format_integer_score(perturbed),
                "delta_value": delta,
                "delta": format_integer_delta(delta),
                "delta_class": classify_display_delta(delta),
            }
        )
    return rows


def render_score_comparison_table(score_rows: list[dict[str, object]], focus_criterion: str) -> str:
    head = (
        "<table class=\"score-compare\">"
        "<thead><tr><th>Criterion</th><th>Baseline</th><th>Perturbation</th><th>Δ</th></tr></thead><tbody>"
    )
    body: list[str] = []
    for row in score_rows:
        focus_class = " is-focus" if row["criterion"] == focus_criterion else ""
        body.append(
            f'<tr class="{focus_class.strip()}">'
            f"<td>{escape(str(row['criterion']))}</td>"
            f"<td>{escape(str(row['baseline']))}</td>"
            f"<td>{escape(str(row['perturbation']))}</td>"
            f"<td><span class=\"{escape(str(row['delta_class']))}\">{escape(str(row['delta']))}</span></td>"
            "</tr>"
        )
    return head + "".join(body) + "</tbody></table>"


def classify_delta(delta: int | None) -> str:
    return classify_display_delta(delta)
