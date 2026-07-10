from __future__ import annotations

import json
import re
from pathlib import Path

from fair_j.report_html import render_comparison_report


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_sample_run(root: Path, example_count: int = 101) -> Path:
    run_dir = root / "sample-run"
    run_dir.mkdir(parents=True)
    dataset_path = root / "dataset.jsonl"
    dataset_rows = [
        {
            "id": f"example_{index:03d}",
            "context": f"Context {index}",
            "candidate": f"Candidate {index}",
        }
        for index in range(example_count)
    ]
    write_jsonl(dataset_path, dataset_rows)
    write_json(
        run_dir / "run.json",
        {
            "judge_model": "test/example-judge",
            "paraphrase_model": "test/example-paraphraser",
            "dataset_path": str(dataset_path),
            "subset_name": "sample_dataset",
            "scale_min": 1,
            "scale_max": 3,
            "seeds": 1,
            "paraphrases_per_criterion": 1,
            "limit_examples": None,
            "dataset_id_column": "id",
            "dataset_context_column": "context",
            "dataset_candidate_column": "candidate",
        },
    )
    write_json(
        run_dir / "core_output.json",
        {
            "dataset_level_scores": {
                "quality": {
                    "mad_paraphrases_same_criterion_scale_normalized": 0.5,
                    "mad_paraphrases_cross_criterion_scale_normalized": 0.1,
                    "stat_tests": {},
                },
                "__total__": {"std_total": 0.2, "stat_tests": {}},
            },
            "ranking_consistency": {
                "quality": {
                    "discordant_paraphrases_same_criterion_pooled": 1,
                    "concordant_paraphrases_same_criterion_pooled": 9,
                    "tie_pct_paraphrases_same_criterion_mean": 0,
                    "kendall_paraphrases_same_criterion_mean": 0.8,
                }
            },
        },
    )
    write_json(
        run_dir / "variants.json",
        [
            {
                "perturbation": "baseline",
                "rubric_name": "sample",
                "criteria": [{"id": "quality", "text": "Is the response good?"}],
            },
            {
                "perturbation": "paraphrase__quality__01",
                "rubric_name": "sample",
                "criteria": [{"id": "quality", "text": "Is this a good response?"}],
            },
        ],
    )
    score_rows: list[dict[str, object]] = []
    for index in range(example_count):
        example_id = f"example_{index:03d}"
        score_rows.extend(
            [
                {
                    "example_id": example_id,
                    "criterion_id": "quality",
                    "perturbation": "baseline",
                    "seed": 0,
                    "score": 1,
                },
                {
                    "example_id": example_id,
                    "criterion_id": "quality",
                    "perturbation": "paraphrase__quality__01",
                    "seed": 0,
                    "score": 3,
                },
            ]
        )
    write_jsonl(run_dir / "score_log.jsonl", score_rows)
    return run_dir


def test_rendered_report_uses_current_static_html_contract(tmp_path: Path) -> None:
    run_dir = write_sample_run(tmp_path)
    output_path = tmp_path / "report.html"

    render_comparison_report([run_dir], output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "Expand All Sections" not in rendered
    assert "Collapse All Sections" not in rendered
    assert "summary-verdict" not in rendered
    assert ">Review<" not in rendered
    assert ">Stable<" not in rendered
    assert "Score Stability" not in rendered
    assert "Score Instability" in rendered
    assert re.search(r"<dt>Examples used</dt>\s*<dd>101</dd>", rendered)

    details = re.findall(r'<details class="section"[^>]*>', rendered)
    assert details
    assert all(" open" in detail for detail in details)

    match = re.search(
        r"window\.__FAIR_J_EXAMPLES__ = (.*?);\n",
        rendered,
        flags=re.DOTALL,
    )
    assert match is not None
    viewer_records = json.loads(match.group(1).replace("<\\/", "</"))
    assert len(viewer_records) == 100
    assert viewer_records[0]["baseline_score"] == 1
    assert viewer_records[0]["paraphrased_score"] == 3
    assert viewer_records[0]["original_criterion"] == "Is the response good?"
    assert viewer_records[0]["paraphrased_criterion"] == "Is this a good response?"
    assert viewer_records[0]["context"].startswith("Context ")
    assert viewer_records[0]["candidate"].startswith("Candidate ")
