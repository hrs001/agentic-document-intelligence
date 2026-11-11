import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TABLE_CONFIG = {
    "financial_performance_by_segment": {
        "path": "corpus/datasets/financial_performance_by_segment.csv",
        "columns": {
            "fiscal_year": "INTEGER",
            "segment_name": "TEXT",
            "revenue_usd_millions": "INTEGER",
            "operating_income_usd_millions": "INTEGER",
            "operating_margin_pct": "REAL",
            "yoy_revenue_growth_pct": "REAL",
            "narrative_driver": "TEXT",
        },
    },
    "geographic_revenue_mix": {
        "path": "corpus/datasets/geographic_revenue_mix.csv",
        "columns": {
            "fiscal_year": "INTEGER",
            "geography": "TEXT",
            "revenue_usd_millions": "INTEGER",
            "revenue_mix_pct": "REAL",
            "yoy_revenue_growth_pct": "REAL",
        },
    },
    "product_family_signals": {
        "path": "corpus/datasets/product_family_signals.csv",
        "columns": {
            "fiscal_year": "INTEGER",
            "product_family": "TEXT",
            "strategic_priority": "TEXT",
            "revenue_signal_index": "INTEGER",
            "margin_profile": "TEXT",
            "ai_relevance_score": "INTEGER",
            "commentary": "TEXT",
        },
    },
}


def convert_value(value: str, sqlite_type: str) -> Any:
    if value == "":
        return None
    if sqlite_type == "INTEGER":
        return int(value)
    if sqlite_type == "REAL":
        return float(value)
    return value


def load_csv_rows(csv_path: Path, columns: dict[str, str]) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {column: convert_value(row[column], sqlite_type) for column, sqlite_type in columns.items()}
            for row in reader
        ]


def create_table(conn: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
    column_sql = ", ".join(f"{column} {sqlite_type}" for column, sqlite_type in columns.items())
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} ({column_sql})")


def insert_rows(conn: sqlite3.Connection, table_name: str, columns: dict[str, str], rows: list[dict[str, Any]]) -> None:
    column_names = list(columns.keys())
    placeholders = ", ".join("?" for _ in column_names)
    insert_sql = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({placeholders})"
    values = [tuple(row[column] for column in column_names) for row in rows]
    conn.executemany(insert_sql, values)


def build_sqlite_database(project_root: Path, database_path: Path) -> dict[str, Any]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    conn = sqlite3.connect(database_path)
    table_summaries: list[dict[str, Any]] = []
    try:
        for table_name, config in TABLE_CONFIG.items():
            columns = config["columns"]
            csv_path = project_root / "agentic_document_intelligence" / config["path"]
            rows = load_csv_rows(csv_path, columns)
            create_table(conn, table_name, columns)
            insert_rows(conn, table_name, columns, rows)
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            table_summaries.append(
                {
                    "table_name": table_name,
                    "source_csv_path": str(csv_path),
                    "row_count": int(row_count),
                    "columns": [{"name": name, "sqlite_type": sqlite_type} for name, sqlite_type in columns.items()],
                }
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "database_path": str(database_path),
        "table_count": len(table_summaries),
        "tables": table_summaries,
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "sql_db"
        / "sqlite_build_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": report,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reusable local SQLite demo database for text-to-SQL.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument("--run-id", default="component6_sqlite_database_build")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent.parent
    database_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / args.run_id
        / "sql_db"
        / "microsoft_fy2025_analyst_demo.sqlite"
    )

    result = build_sqlite_database(project_root, database_path)
    report_path = write_report(project_root, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "database_path": result["database_path"],
                "report_path": str(report_path),
                "table_count": result["table_count"],
                "tables": [
                    {"table_name": table["table_name"], "row_count": table["row_count"]}
                    for table in result["tables"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
