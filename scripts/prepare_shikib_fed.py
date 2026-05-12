#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path


SOURCE_URL = "https://shikib.com/fed_data.json"
DEFAULT_OUTPUT_PATH = Path("data/fed/fed_shikib_dataset.jsonl")
DEFAULT_SUMMARY_PATH = Path("data/fed/fed_shikib_build_summary.json")
ROLE_PREFIX_RE = re.compile(r"^(Speaker|Listener|System|User):\s*")


def normalize_text(value: str) -> str:
    lines = [line.rstrip() for line in value.splitlines()]
    return "\n".join(lines).strip()


def split_candidate_role(response_text: str) -> tuple[str | None, str]:
    normalized = normalize_text(response_text)
    first_line, *rest = normalized.splitlines()
    match = ROLE_PREFIX_RE.match(first_line)
    if not match:
        return None, normalized

    role = match.group(1)
    first_line_without_role = ROLE_PREFIX_RE.sub("", first_line, count=1).strip()
    candidate_lines = [first_line_without_role, *rest]
    candidate = "\n".join(line.rstrip() for line in candidate_lines).strip()
    return role, candidate


def load_source_rows(source_url: str) -> list[dict[str, object]]:
    with urllib.request.urlopen(source_url) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level array from {source_url}.")
    return [dict(row) for row in payload]


def build_dataset_rows(source_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset_rows: list[dict[str, object]] = []
    skipped_missing_response = 0

    for index, row in enumerate(source_rows, start=1):
        response = row.get("response")
        if not isinstance(response, str) or not response.strip():
            skipped_missing_response += 1
            continue

        context = row.get("context")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"Row {index} is missing a non-empty string 'context'.")

        candidate_role, candidate = split_candidate_role(response)
        metadata = {
            "source_url": SOURCE_URL,
            "source_index": index,
            "system": row.get("system"),
            "candidate_role": candidate_role,
            "annotations": row.get("annotations", {}),
        }

        dataset_rows.append(
            {
                "id": f"fed_shikib_{index:04d}",
                "context": normalize_text(context),
                "candidate": candidate,
                "metadata": metadata,
            }
        )

    summary = {
        "source_url": SOURCE_URL,
        "source_rows": len(source_rows),
        "output_rows": len(dataset_rows),
        "skipped_missing_response": skipped_missing_response,
    }
    return dataset_rows, summary


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    source_rows = load_source_rows(SOURCE_URL)
    dataset_rows, summary = build_dataset_rows(source_rows)
    write_jsonl(DEFAULT_OUTPUT_PATH, dataset_rows)
    write_json(DEFAULT_SUMMARY_PATH, summary)
    print(f"Wrote {len(dataset_rows)} rows to {DEFAULT_OUTPUT_PATH}")
    print(f"Wrote build summary to {DEFAULT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
