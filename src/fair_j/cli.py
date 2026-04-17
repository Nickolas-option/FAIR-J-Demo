from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fair_j.evaluate_judge import evaluate_judge
from fair_j.io_utils import (
    build_default_run_dir,
    create_run_dir,
    default_subset_name,
    load_rubric,
    write_json,
)
from fair_j.perturbations import make_variants
from fair_j.report_html import render_comparison_report
from fair_j.run_judge import run_judge
from fair_j.schemas import AdapterInput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fair-j")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_variants_parser = subparsers.add_parser("make-variants")
    make_variants_parser.add_argument("--rubric-path", type=Path, required=True)
    make_variants_parser.add_argument("--paraphrase-model", required=True)
    make_variants_parser.add_argument("--openrouter-api-key", required=True)
    make_variants_parser.add_argument("--output-path", type=Path, required=True)
    make_variants_parser.set_defaults(func=cmd_make_variants)

    run_adapter_parser = subparsers.add_parser("run-adapter")
    add_pipeline_arguments(run_adapter_parser)
    run_adapter_parser.set_defaults(func=cmd_run_adapter)

    run_evaluation_parser = subparsers.add_parser("run-evaluation")
    add_pipeline_arguments(run_evaluation_parser)
    run_evaluation_parser.add_argument("--html-output-path", type=Path)
    run_evaluation_parser.add_argument("--html-title")
    run_evaluation_parser.set_defaults(func=cmd_run_evaluation)

    run_pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Unified end-to-end entry point: adapter -> core -> HTML report.",
    )
    add_pipeline_arguments(run_pipeline_parser)
    run_pipeline_parser.add_argument("--html-output-path", type=Path)
    run_pipeline_parser.add_argument("--html-title")
    run_pipeline_parser.set_defaults(func=cmd_run_evaluation)

    run_core_parser = subparsers.add_parser("run-core")
    run_core_parser.add_argument("--run-dir", type=Path, required=True)
    run_core_parser.set_defaults(func=cmd_run_core)

    render_compare_parser = subparsers.add_parser("render-compare-html")
    render_compare_parser.add_argument("--run-dir", type=Path, action="append", required=True)
    render_compare_parser.add_argument("--output-path", type=Path, required=True)
    render_compare_parser.add_argument("--title")
    render_compare_parser.set_defaults(func=cmd_render_compare_html)

    return parser


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--paraphrase-model", required=True)
    parser.add_argument("--openrouter-api-key", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--rubric-path", type=Path, required=True)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--context-column", default="context")
    parser.add_argument("--candidate-column", default="candidate")
    parser.add_argument("--provider")
    parser.add_argument("--provider-quantization")
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--subset-name")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--no-progress", action="store_true")


def cmd_make_variants(args: argparse.Namespace) -> None:
    rubric = load_rubric(args.rubric_path)
    variants = make_variants(
        rubric=rubric,
        paraphrase_model=args.paraphrase_model,
        openrouter_api_key=args.openrouter_api_key,
    )
    write_json(args.output_path, variants)
    print(f"Wrote {len(variants)} variants to {args.output_path}")


def cmd_run_adapter(args: argparse.Namespace) -> None:
    result = run_adapter_from_args(args)
    print(f"Prepared run in {result['run_dir']}")
    print(
        "Calls: "
        f"written={result['written_calls']} "
        f"skipped={result['skipped_calls']} "
        f"already_completed={result['completed_calls']}"
    )


def cmd_run_evaluation(args: argparse.Namespace) -> None:
    result = run_adapter_from_args(args)
    run_dir = Path(str(result["run_dir"]))

    print(f"Prepared run in {run_dir}")
    print(
        "Calls: "
        f"written={result['written_calls']} "
        f"skipped={result['skipped_calls']} "
        f"already_completed={result['completed_calls']}"
    )

    core_output = evaluate_judge(run_dir)
    print(f"Wrote core output to {run_dir / 'core_output.json'}")
    print(f"Dataset metrics keys: {list(core_output.dataset_level_scores.keys())}")

    html_output_path = args.html_output_path or (run_dir / "report.html")
    output_path = render_comparison_report(
        run_dirs=[run_dir],
        output_path=html_output_path,
        title=args.html_title,
    )
    print(f"Wrote comparison report to {output_path}")


def cmd_run_core(args: argparse.Namespace) -> None:
    output = evaluate_judge(args.run_dir)
    print(f"Wrote core output to {args.run_dir / 'core_output.json'}")
    print(f"Dataset metrics keys: {list(output.dataset_level_scores.keys())}")


def cmd_render_compare_html(args: argparse.Namespace) -> None:
    output_path = render_comparison_report(
        run_dirs=args.run_dir,
        output_path=args.output_path,
        title=args.title,
    )
    print(f"Wrote comparison report to {output_path}")


def build_progress_callback():
    def update(current: int, total: int) -> None:
        width = 24
        ratio = 1.0 if total == 0 else current / total
        filled = min(width, int(ratio * width))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(f"\rAdapter progress [{bar}] {current}/{total}")
        if current >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return update


def run_adapter_from_args(args: argparse.Namespace) -> dict[str, int | str]:
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    adapter_input = AdapterInput(
        judge_model=args.judge_model,
        paraphrase_model=args.paraphrase_model,
        openrouter_api_key=args.openrouter_api_key,
        dataset_path=args.dataset_path,
        rubric_path=args.rubric_path,
        dataset_id_column=args.id_column,
        dataset_context_column=args.context_column,
        dataset_candidate_column=args.candidate_column,
        provider_only=args.provider,
        provider_quantization=args.provider_quantization,
        request_timeout_seconds=args.request_timeout,
    )

    rubric = load_rubric(args.rubric_path)
    subset_name = args.subset_name or default_subset_name(args.dataset_path)
    run_dir = args.run_dir or build_default_run_dir(
        base_dir=Path("runs"),
        judge_model=args.judge_model,
        subset_name=subset_name,
        scale_min=rubric.scale_min,
        scale_max=rubric.scale_max,
    )
    run_dir = create_run_dir(run_dir.parent, run_dir.name)

    return run_judge(
        adapter_input=adapter_input,
        run_dir=run_dir,
        subset_name=subset_name,
        seeds=args.seeds,
        workers=args.workers,
        progress_callback=None if args.no_progress else build_progress_callback(),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
