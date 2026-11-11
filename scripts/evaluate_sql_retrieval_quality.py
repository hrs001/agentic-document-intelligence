import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.generate_text_to_sql import generate_text_to_sql, load_schema_package
from agentic_document_intelligence.scripts.validate_and_execute_sql import execute_read_only_sql


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def execute_reference_sql(database_path: Path, sql_query: str) -> dict[str, Any]:
    conn = sqlite3.connect(database_path)
    try:
        cursor = conn.execute(sql_query)
        columns = [column[0] for column in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
    finally:
        conn.close()
    return {
        "columns": columns,
        "rows": [dict(zip(columns, row)) for row in rows],
        "row_count": len(rows),
    }


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def normalize_rows(columns: list[str], rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    normalized = []
    for row in rows:
        normalized.append(tuple(normalize_value(row.get(column)) for column in columns))
    return normalized


def project_rows(columns: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in columns} for row in rows]


def compare_results(
    actual: dict[str, Any],
    expected: dict[str, Any],
    ordered: bool = True,
) -> dict[str, Any]:
    same_columns = actual["columns"] == expected["columns"]
    actual_rows = normalize_rows(actual["columns"], actual["rows"])
    expected_rows = normalize_rows(expected["columns"], expected["rows"])
    if not ordered:
        actual_rows = sorted(actual_rows)
        expected_rows = sorted(expected_rows)
    rows_match = actual_rows == expected_rows
    missing_columns = [column for column in expected["columns"] if column not in actual["columns"]]
    extra_columns = [column for column in actual["columns"] if column not in expected["columns"]]
    projected_rows_match = False
    if not missing_columns:
        projected_actual_rows = normalize_rows(
            expected["columns"],
            project_rows(expected["columns"], actual["rows"]),
        )
        projected_expected_rows = normalize_rows(expected["columns"], expected["rows"])
        if not ordered:
            projected_actual_rows = sorted(projected_actual_rows)
            projected_expected_rows = sorted(projected_expected_rows)
        projected_rows_match = projected_actual_rows == projected_expected_rows
    return {
        "same_columns": same_columns,
        "rows_match": rows_match,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "projected_rows_match": projected_rows_match,
        "strict_passed": same_columns and rows_match,
        "answer_passed": not missing_columns and projected_rows_match,
        "actual_row_count": actual["row_count"],
        "expected_row_count": expected["row_count"],
    }


def evaluate_case(
    database_path: Path,
    schema_package: dict[str, Any],
    case: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    generation = generate_text_to_sql(case["query"], schema_package, model=model)
    actual = execute_read_only_sql(database_path, generation["sql_query"], row_limit=case.get("row_limit", 50))
    expected = execute_reference_sql(database_path, case["reference_sql"])
    comparison = compare_results(actual, expected, ordered=case.get("ordered", True))
    expected_tables = case.get("expected_target_tables", [])
    actual_tables = generation.get("target_tables", [])
    table_hit = set(expected_tables).issubset(set(actual_tables)) if expected_tables else True

    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "expected_target_tables": expected_tables,
        "generated_target_tables": actual_tables,
        "table_hit": table_hit,
        "generated_sql": generation["sql_query"],
        "validated_sql": actual["validated_sql"],
        "confidence": generation["confidence"],
        "actual_result": {
            "columns": actual["columns"],
            "row_count": actual["row_count"],
            "rows": actual["rows"],
        },
        "expected_result": expected,
        "comparison": comparison,
        "strict_passed": table_hit and comparison["strict_passed"],
        "answer_passed": table_hit and comparison["answer_passed"],
        "passed": table_hit and comparison["strict_passed"],
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "text_to_sql"
        / "sql_retrieval_quality_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate text-to-SQL quality against database-grounded benchmark queries.")
    parser.add_argument(
        "--schema-path",
        default="artifacts/experiments/component6_sql_schema_packaging_live/sql_schema/sql_schema_package.json",
    )
    parser.add_argument(
        "--database-path",
        default="artifacts/experiments/component6_sqlite_database_build_live/sql_db/microsoft_fy2025_analyst_demo.sqlite",
    )
    parser.add_argument(
        "--cases-path",
        default="evals/sql_retrieval_quality_cases.json",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--run-id", default="component6_sql_retrieval_quality_eval")
    args = parser.parse_args()

    project_root = PROJECT_ROOT / "agentic_document_intelligence"
    schema_package = load_schema_package(project_root / args.schema_path)
    database_path = project_root / args.database_path
    cases = load_cases(project_root / args.cases_path)

    case_results = []
    for case in cases:
        case_results.append(evaluate_case(database_path, schema_package, case, model=args.model))

    strict_passed_count = sum(1 for item in case_results if item["strict_passed"])
    answer_passed_count = sum(1 for item in case_results if item["answer_passed"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "database_path": str(database_path),
        "case_count": len(case_results),
        "strict_passed_count": strict_passed_count,
        "strict_pass_rate": round(strict_passed_count / len(case_results), 4) if case_results else 0.0,
        "answer_passed_count": answer_passed_count,
        "answer_pass_rate": round(answer_passed_count / len(case_results), 4) if case_results else 0.0,
        "case_results": case_results,
    }
    report_path = write_report(PROJECT_ROOT, args.run_id, report)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "case_count": len(case_results),
                "strict_passed_count": strict_passed_count,
                "strict_pass_rate": report["strict_pass_rate"],
                "answer_passed_count": answer_passed_count,
                "answer_pass_rate": report["answer_pass_rate"],
                "model": args.model,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
