import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORBIDDEN_SQL_PATTERNS = [
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\balter\b",
    r"\bcreate\b",
    r"\battach\b",
    r"\bpragma\b",
    r"\bbegin\b",
    r"\bcommit\b",
    r"\brollback\b",
    r"\breplace\b",
    r"\btruncate\b",
]


def load_generation_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def validate_read_only_sql(sql_query: str) -> str:
    sql = sql_query.strip()
    if not sql:
        raise ValueError("SQL query is empty")

    normalized = sql.lower().strip()
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    if len(statements) != 1:
        raise ValueError("Only a single SQL statement is allowed")
    if not (normalized.startswith("select") or normalized.startswith("with")):
        raise ValueError("Only read-only SELECT queries are allowed")
    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, normalized):
            raise ValueError(f"Forbidden SQL pattern detected: {pattern}")
    return statements[0]


def make_read_only_connection(database_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)

    def authorizer(action_code, _param1, _param2, _db_name, _trigger_name):
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
        }
        return sqlite3.SQLITE_OK if action_code in allowed else sqlite3.SQLITE_DENY

    conn.set_authorizer(authorizer)
    return conn


def execute_read_only_sql(database_path: Path, sql_query: str, row_limit: int = 50) -> dict[str, Any]:
    safe_sql = validate_read_only_sql(sql_query)
    conn = make_read_only_connection(database_path)
    try:
        cursor = conn.execute(safe_sql)
        columns = [column[0] for column in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(row_limit)
    finally:
        conn.close()

    serialized_rows = [dict(zip(columns, row)) for row in rows]
    return {
        "validated_sql": safe_sql,
        "row_limit": row_limit,
        "row_count": len(serialized_rows),
        "columns": columns,
        "rows": serialized_rows,
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "text_to_sql"
        / "sql_execution_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and safely execute generated read-only SQL.")
    parser.add_argument(
        "--generation-report",
        default="artifacts/experiments/component6_text_to_sql_generation_live/text_to_sql/text_to_sql_generation_report.json",
    )
    parser.add_argument(
        "--database-path",
        default="artifacts/experiments/component6_sqlite_database_build_live/sql_db/microsoft_fy2025_analyst_demo.sqlite",
    )
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--run-id", default="component6_sql_validation_execution")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    generation_result = load_generation_report(project_root / "agentic_document_intelligence" / args.generation_report)
    database_path = project_root / "agentic_document_intelligence" / args.database_path
    execution = execute_read_only_sql(database_path, generation_result["sql_query"], row_limit=args.row_limit)

    result = {
        "database_path": str(database_path),
        "user_query": generation_result["user_query"],
        "target_tables": generation_result["target_tables"],
        "generated_sql": generation_result["sql_query"],
        "validated_sql": execution["validated_sql"],
        "row_limit": execution["row_limit"],
        "row_count": execution["row_count"],
        "columns": execution["columns"],
        "rows": execution["rows"],
        "confidence": generation_result["confidence"],
    }
    report_path = write_report(project_root, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "row_count": result["row_count"],
                "columns": result["columns"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
