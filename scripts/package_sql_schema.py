import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def load_table_schema(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [
        {
            "name": row[1],
            "sqlite_type": row[2],
            "notnull": bool(row[3]),
            "default_value": row[4],
            "is_primary_key": bool(row[5]),
        }
        for row in rows
    ]


def load_sample_rows(conn: sqlite3.Connection, table_name: str, limit: int = 2) -> list[dict[str, Any]]:
    cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT {int(limit)}")
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def build_prompt_schema_text(table_summaries: list[dict[str, Any]]) -> str:
    blocks = []
    for table in table_summaries:
        column_lines = [f"- {column['name']} ({column['sqlite_type']})" for column in table["columns"]]
        blocks.append(
            "\n".join(
                [
                    f"Table: {table['table_name']}",
                    f"Row count: {table['row_count']}",
                    "Columns:",
                    *column_lines,
                    f"Sample rows: {json.dumps(table['sample_rows'], ensure_ascii=False)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def package_sql_schema(database_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(database_path)
    try:
        table_summaries = []
        for table_name in load_table_names(conn):
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            table_summaries.append(
                {
                    "table_name": table_name,
                    "row_count": int(row_count),
                    "columns": load_table_schema(conn, table_name),
                    "sample_rows": load_sample_rows(conn, table_name, limit=2),
                }
            )
    finally:
        conn.close()

    return {
        "database_path": str(database_path),
        "table_count": len(table_summaries),
        "tables": table_summaries,
        "prompt_schema_text": build_prompt_schema_text(table_summaries),
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "sql_schema"
        / "sql_schema_package.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the SQLite schema for later text-to-SQL prompting.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--database-path",
        default="artifacts/experiments/component6_sqlite_database_build_live/sql_db/microsoft_fy2025_analyst_demo.sqlite",
    )
    parser.add_argument("--run-id", default="component6_sql_schema_packaging")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent.parent
    database_path = project_root / "agentic_document_intelligence" / args.database_path

    result = package_sql_schema(database_path)
    report_path = write_report(project_root, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "database_path": result["database_path"],
                "report_path": str(report_path),
                "table_count": result["table_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
