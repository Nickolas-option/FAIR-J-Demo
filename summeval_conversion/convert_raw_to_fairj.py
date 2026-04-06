from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


RAW_DIR = Path("data/raw/summeval")
OUTPUT_PATH = Path("data/converted/summeval/summeval_from_raw.jsonl")


def main() -> None:
    source_articles = load_source_articles(RAW_DIR / "cnn_dailymail_test.parquet")
    input_path = RAW_DIR / "model_annotations.aligned.jsonl"
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_rows = 0
    missing_source_rows = 0

    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line in input_handle:
            annotation_row = json.loads(line)
            converted_row = convert_row(annotation_row, source_articles)
            if converted_row is None:
                missing_source_rows += 1
                continue
            output_handle.write(json.dumps(converted_row, ensure_ascii=False) + "\n")
            converted_rows += 1

    print(f"Wrote {converted_rows} rows to {output_path}")
    print(f"Missing source rows: {missing_source_rows}")


def load_source_articles(parquet_path: Path) -> dict[str, dict[str, str]]:
    table = pq.read_table(parquet_path)
    return {row["id"]: row for row in table.to_pylist()}


def convert_row(
    annotation_row: dict,
    source_articles: dict[str, dict[str, str]],
) -> dict | None:
    source_id = Path(annotation_row["filepath"]).stem
    source_row = source_articles.get(source_id)
    if source_row is None:
        return None

    return {
        "id": f"{annotation_row['id']}__{annotation_row['model_id']}",
        "context": source_row["article"],
        "candidate": annotation_row["decoded"],
        "metadata": {
            "source_id": source_id,
            "annotation_id": annotation_row["id"],
            "model_id": annotation_row["model_id"],
            "filepath": annotation_row["filepath"],
            "reference_highlights": source_row["highlights"],
            "references": annotation_row.get("references", []),
            "expert_annotations": annotation_row.get("expert_annotations", []),
            "turker_annotations": annotation_row.get("turker_annotations", []),
        },
    }


if __name__ == "__main__":
    main()
