import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = "artifacts/experiments/component6_sql_validation_execution_live/text_to_sql/sql_execution_report.json"


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def format_cell(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value).replace("\n", " ").strip()
    return text if text else "''"


def build_markdown_table(columns: list[str], rows: list[dict[str, Any]], max_rows: int = 10) -> str:
    if not columns:
        return "No columns returned."

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |"
        for row in rows[:max_rows]
    ]

    if not body:
        body = ["| " + " | ".join(["(no rows)"] + [""] * (len(columns) - 1)) + " |"]
    return "\n".join([header, separator, *body])


def build_sql_evidence_text(report: dict[str, Any], preview_rows: list[dict[str, Any]]) -> str:
    lines = [
        "[SQL Evidence]",
        f"User Query: {report['user_query']}",
        f"Target Tables: {', '.join(report.get('target_tables', []))}",
        f"Confidence: {report.get('confidence', 'unknown')}",
        f"Validated SQL: {report['validated_sql']}",
        f"Returned Rows: {report['row_count']}",
        "Result Preview:",
        build_markdown_table(report["columns"], preview_rows),
    ]
    return "\n".join(lines)


def package_sql_evidence(report: dict[str, Any], preview_limit: int = 10) -> dict[str, Any]:
    preview_rows = report["rows"][:preview_limit]
    return {
        "user_query": report["user_query"],
        "database_path": report["database_path"],
        "target_tables": report.get("target_tables", []),
        "validated_sql": report["validated_sql"],
        "confidence": report.get("confidence", "unknown"),
        "bundle_summary": {
            "column_count": len(report["columns"]),
            "row_count": report["row_count"],
            "preview_row_count": len(preview_rows),
        },
        "columns": report["columns"],
        "preview_rows": preview_rows,
        "assembled_sql_evidence_text": build_sql_evidence_text(report, preview_rows),
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "text_to_sql"
        / "sql_evidence_bundle.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package executed SQL results into an answer-ready SQL evidence bundle.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default="component6_sql_evidence_packaging")
    parser.add_argument("--preview-limit", type=int, default=10)
    args = parser.parse_args()

    report = load_report(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    result = package_sql_evidence(report, preview_limit=args.preview_limit)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "row_count": result["bundle_summary"]["row_count"],
                "preview_row_count": result["bundle_summary"]["preview_row_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
