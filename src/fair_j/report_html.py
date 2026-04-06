from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


CRITERIA = ("coherence", "consistency", "fluency", "relevance")
DATASET_METRICS = (
    "std_seed_default",
    "std_total",
    "std_paraphrases",
    "std_deletions",
    "bias_paraphrases",
    "mad_paraphrases",
    "bias_deletions",
    "mad_deletions",
)
RANKING_METRICS = (
    "kendall_paraphrases_mean",
    "kendall_paraphrases_std",
    "kendall_paraphrases_min",
    "kendall_paraphrases_max",
    "kendall_deletions_mean",
    "kendall_deletions_std",
    "kendall_deletions_min",
    "kendall_deletions_max",
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
SECTION_EXPLANATIONS = {
    "ranking_consistency": (
        "We take all examples for one criterion, rank them by the baseline score, and rank them again under each paraphrase or deletion variant. "
        "Then we compute Kendall tau between the baseline ranking and each perturbed ranking, and aggregate those tau values across variants. "
        "Use this section when you care about preserving example ordering, for example in reward modeling, reranking, or pairwise preference training."
    ),
    "dataset_level": (
        "We first reduce scores to dataset-level summaries for each criterion and perturbation. "
        "Then we inspect average shifts with bias, average absolute shifts with MAD, and test whether baseline-minus-perturbation differences are centered at zero using statistical tests. "
        "Use this section when you care about whether aggregate judge scores stay stable enough for method comparison."
    ),
}
METRIC_EXPLANATIONS = {
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
    "bias_deletions": (
        "We take every example-level difference between a deletion variant and the baseline for this criterion. "
        "Then we sum those signed differences and divide by the number of baseline-deletion pairs. Values near zero mean little average shift; positive means deletions tend to score lower than baseline."
    ),
    "mad_deletions": (
        "We take every example-level difference between a deletion variant and the baseline for this criterion. "
        "Then we take the absolute value, sum those absolute differences, and divide by the number of baseline-deletion pairs. Lower means deletions change the score less on average."
    ),
    "kendall_paraphrases_mean": (
        "For each paraphrase variant, we rank all examples by score for this criterion and compare that ranking against the baseline ranking using Kendall tau. "
        "Then we average those tau values across paraphrase variants. Higher means the model preserves the example ordering better under paraphrased rubric wording."
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
    ),
    "kendall_deletions_mean": (
        "For each deletion variant, we rank all examples by score for this criterion and compare that ranking against the baseline ranking using Kendall tau. "
        "Then we average those tau values across deletion variants. Higher means the model preserves the example ordering better when rubric text is removed."
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


def render_comparison_report(run_dirs: list[Path], output_path: Path, title: str | None = None) -> Path:
    reports = [load_model_report(run_dir) for run_dir in run_dirs]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html_document(reports, title=title), encoding="utf-8")
    return output_path


def load_model_report(run_dir: Path) -> ModelReport:
    core_output_path = run_dir / "core_output.json"
    data = json.loads(core_output_path.read_text(encoding="utf-8"))
    run_metadata = data["run_metadata"]
    judge_model = run_metadata["judge_model"]
    label = build_short_label(judge_model, run_dir.name)
    full_label = f"{judge_model} ({run_dir.name})"
    return ModelReport(
        label=label,
        full_label=full_label,
        run_dir=run_dir,
        core_output_path=core_output_path,
        data=data,
    )


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
      <button type="button" data-open-all="stats">Open all stat tests</button>
      <button type="button" data-close-all="metric">Collapse all inner sections</button>
    </section>

    <section class="summary-grid">
      {build_summary_cards(reports)}
    </section>

    <section class="overview-grid">
      {build_overview_cards(reports)}
    </section>

    {''.join(build_criterion_sections(reports))}

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
    document.querySelector('[data-open-all="stats"]').addEventListener('click', () => setOpen('details.metric-stats', true));
    document.querySelector('[data-close-all="metric"]').addEventListener('click', () => setOpen('details.metric-dataset, details.metric-ranking, details.metric-stats', false));
  </script>
</body>
</html>
"""


def build_summary_cards(reports: list[ModelReport]) -> str:
    cards: list[str] = []
    for report in reports:
        ds = report.data["dataset_level_scores"]
        rc = report.data["ranking_consistency"]
        avg_std_seed_default = mean_non_null(ds[criterion]["std_seed_default"] for criterion in CRITERIA)
        avg_std = mean(ds[criterion]["std_total"] for criterion in CRITERIA)
        avg_kendall_para = mean(rc[criterion]["kendall_paraphrases_mean"] for criterion in CRITERIA)
        avg_kendall_del = mean(rc[criterion]["kendall_deletions_mean"] for criterion in CRITERIA)
        significant_para = 0
        significant_del = 0
        for criterion in CRITERIA:
            tests = ds[criterion]["stat_tests"]
            if tests["paraphrases_vs_baseline"]["tests"]["paired_t_test"]["p_value"] < 0.05:
                significant_para += 1
            if tests["deletions_vs_baseline"]["tests"]["paired_t_test"]["p_value"] < 0.05:
                significant_del += 1

        cards.append(
            "\n".join(
                [
                    '<article class="summary-card">',
                    f"<h2>{escape(report.label)}</h2>",
                    f'<div class="meta">{escape(report.full_label)}<br>{escape(str(report.run_dir))}</div>',
                    '<div class="stat-list">',
                    summary_stat("Avg std_seed_default", avg_std_seed_default),
                    summary_stat("Avg std_total", avg_std),
                    summary_stat("Avg Kendall paraphrases", avg_kendall_para),
                    summary_stat("Avg Kendall deletions", avg_kendall_del),
                    summary_stat("Significant paraphrase tests", significant_para, digits=0),
                    summary_stat("Significant deletion tests", significant_del, digits=0),
                    summary_stat("Criteria tracked", len(CRITERIA), digits=0),
                    "</div>",
                    "</article>",
                ]
            )
        )
    return "".join(cards)


def build_overview_cards(reports: list[ModelReport]) -> str:
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
                    fetch=fetch_ranking_overview_value,
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
                    fetch=fetch_dataset_overview_value,
                ),
            ),
        ]
    )


def build_overview_card(section_key: str, title: str, body: str) -> str:
    explanation = SECTION_EXPLANATIONS[section_key]
    return (
        '<article class="summary-card overview-card">'
        f'<h2><span class="section-label" title="{escape(explanation)}">{escape(title)}</span></h2>'
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


def fetch_ranking_overview_value(report: ModelReport, metric_key: str) -> float:
    rc = report.data["ranking_consistency"]
    paraphrases = [rc[criterion]["kendall_paraphrases_mean"] for criterion in CRITERIA]
    deletions = [rc[criterion]["kendall_deletions_mean"] for criterion in CRITERIA]
    if metric_key == "avg_kendall_paraphrases":
        return mean(paraphrases)
    if metric_key == "avg_kendall_deletions":
        return mean(deletions)
    if metric_key == "worst_kendall":
        return min(paraphrases + deletions)
    raise KeyError(metric_key)


def fetch_dataset_overview_value(report: ModelReport, metric_key: str) -> float | int:
    ds = report.data["dataset_level_scores"]
    if metric_key == "avg_std_seed_default":
        return mean_non_null(ds[criterion]["std_seed_default"] for criterion in CRITERIA)
    if metric_key == "avg_mad_paraphrases":
        return mean(ds[criterion]["mad_paraphrases"] for criterion in CRITERIA)
    if metric_key == "avg_mad_deletions":
        return mean(ds[criterion]["mad_deletions"] for criterion in CRITERIA)
    if metric_key == "significant_paraphrase_tests":
        return sum(
            ds[criterion]["stat_tests"]["paraphrases_vs_baseline"]["tests"]["paired_t_test"]["p_value"] < 0.05
            for criterion in CRITERIA
        )
    if metric_key == "significant_deletion_tests":
        return sum(
            ds[criterion]["stat_tests"]["deletions_vs_baseline"]["tests"]["paired_t_test"]["p_value"] < 0.05
            for criterion in CRITERIA
        )
    raise KeyError(metric_key)


def summary_stat(label: str, value: float | int, digits: int = 4) -> str:
    return (
        '<div class="stat-item">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{format_number(value, digits=digits)}</div>'
        "</div>"
    )


def build_criterion_sections(reports: list[ModelReport]) -> list[str]:
    sections: list[str] = []
    for criterion in CRITERIA:
        sections.append(
            "\n".join(
                [
                    '<details class="section">',
                    f"<summary>{escape(criterion.title())}</summary>",
                    '<div class="section-body">',
                    '<div class="badge-row">',
                    '<span class="badge">dataset-level</span>',
                    '<span class="badge">ranking consistency</span>',
                    '<span class="badge">statistical tests</span>',
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
                        title="Statistical tests",
                        css_class="metric-stats",
                        body=build_stats_table(reports, criterion),
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


def build_dataset_table(reports: list[ModelReport], criterion: str) -> str:
    return build_matrix_table(
        reports=reports,
        row_specs=[(metric, prettify_metric_name(metric)) for metric in DATASET_METRICS],
        fetch_value=lambda report, metric: report.data["dataset_level_scores"][criterion][metric],
        value_formatter=format_number,
    )


def build_ranking_table(reports: list[ModelReport], criterion: str) -> str:
    return build_matrix_table(
        reports=reports,
        row_specs=[(metric, prettify_metric_name(metric)) for metric in RANKING_METRICS],
        fetch_value=lambda report, metric: report.data["ranking_consistency"][criterion][metric],
        value_formatter=format_number,
    )


def build_stats_table(reports: list[ModelReport], criterion: str) -> str:
    rows: list[tuple[str, str]] = []
    for group_key, group_label in STAT_TEST_GROUPS:
        rows.append((f"{group_key}:n_pairs", f"{group_label} / n_pairs"))
        for test_key, test_label in STAT_TESTS:
            rows.append((f"{group_key}:{test_key}", f"{group_label} / {test_label} p-value"))

    def fetch(report: ModelReport, metric: str) -> float | int:
        group_key, suffix = metric.split(":", 1)
        group = report.data["dataset_level_scores"][criterion]["stat_tests"][group_key]
        if suffix == "n_pairs":
            return group["n_pairs"]
        return group["tests"][suffix]["p_value"]

    def fmt(value: float | int) -> str:
        if isinstance(value, int):
            return str(value)
        css_class = classify_p_value(value)
        return f'<span class="{css_class}">{format_number(value, digits=6)}</span>'

    return build_matrix_table(
        reports=reports,
        row_specs=rows,
        fetch_value=fetch,
        value_formatter=fmt,
        raw_html=True,
    )


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
            rendered = value_formatter(fetch_value(report, metric_key))
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
    return (
        f'<span class="metric-label" title="{escape(explanation)}">'
        f"{escape(metric_label)}"
        "</span>"
    )


def classify_p_value(value: float) -> str:
    if value < 0.01:
        return "bad"
    if value < 0.05:
        return "warn"
    return "good"


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


def escape(value: object) -> str:
    return html.escape(str(value))


def mean_non_null(values) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return mean(filtered)
