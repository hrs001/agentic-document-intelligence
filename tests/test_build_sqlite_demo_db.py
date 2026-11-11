import json
import sqlite3
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.build_sqlite_demo_db import (
    TABLE_CONFIG,
    build_sqlite_database,
    convert_value,
    write_report,
)


class BuildSqliteDemoDbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_convert_value_handles_numeric_types(self):
        self.assertEqual(convert_value("2025", "INTEGER"), 2025)
        self.assertEqual(convert_value("46.8", "REAL"), 46.8)
        self.assertEqual(convert_value("Azure", "TEXT"), "Azure")

    def test_build_sqlite_database_creates_tables(self):
        database_path = (
            self.project_root.parent
            / "artifacts"
            / "experiments"
            / "component6_sqlite_database_build_test"
            / "sql_db"
            / "test.sqlite"
        )
        result = build_sqlite_database(self.project_root.parent, database_path)
        self.assertEqual(result["table_count"], 3)
        conn = sqlite3.connect(database_path)
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(set(TABLE_CONFIG.keys()), tables)
            count = conn.execute("SELECT COUNT(*) FROM financial_performance_by_segment").fetchone()[0]
            self.assertGreater(count, 0)
        finally:
            conn.close()

    def test_report_can_be_written(self):
        report = {"database_path": "db", "table_count": 3, "tables": []}
        path = write_report(self.project_root.parent, "component6_sqlite_database_build_test_report", report)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["table_count"], 3)


if __name__ == "__main__":
    unittest.main()
