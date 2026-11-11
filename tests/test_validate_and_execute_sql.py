import json
import sqlite3
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.validate_and_execute_sql import (
    execute_read_only_sql,
    make_read_only_connection,
    validate_read_only_sql,
    write_report,
)


class ValidateAndExecuteSqlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.database_path = (
            cls.project_root
            / "artifacts"
            / "experiments"
            / "component6_sqlite_database_build_live"
            / "sql_db"
            / "microsoft_fy2025_analyst_demo.sqlite"
        )

    def test_validate_read_only_sql_accepts_select(self):
        sql = "SELECT * FROM financial_performance_by_segment;"
        self.assertEqual(validate_read_only_sql(sql), "SELECT * FROM financial_performance_by_segment")

    def test_validate_read_only_sql_rejects_delete(self):
        with self.assertRaises(ValueError):
            validate_read_only_sql("DELETE FROM financial_performance_by_segment")

    def test_execute_read_only_sql_returns_rows(self):
        result = execute_read_only_sql(
            self.database_path,
            "SELECT segment_name FROM financial_performance_by_segment WHERE fiscal_year = 2025",
        )
        self.assertGreater(result["row_count"], 0)
        self.assertIn("segment_name", result["columns"])

    def test_read_only_connection_denies_writes(self):
        conn = make_read_only_connection(self.database_path)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM financial_performance_by_segment")
        finally:
            conn.close()

    def test_report_can_be_written(self):
        result = {
            "database_path": "db",
            "user_query": "x",
            "target_tables": ["t1"],
            "generated_sql": "SELECT 1",
            "validated_sql": "SELECT 1",
            "row_limit": 1,
            "row_count": 1,
            "columns": ["x"],
            "rows": [{"x": 1}],
            "confidence": "high",
        }
        path = write_report(self.project_root.parent, "component6_sql_validation_execution_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
