import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.package_sql_schema import (
    build_prompt_schema_text,
    package_sql_schema,
    write_report,
)


class PackageSqlSchemaTests(unittest.TestCase):
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

    def test_package_sql_schema_reads_real_tables(self):
        result = package_sql_schema(self.database_path)
        self.assertEqual(result["table_count"], 3)
        self.assertTrue(any(table["table_name"] == "financial_performance_by_segment" for table in result["tables"]))

    def test_build_prompt_schema_text_contains_columns(self):
        text = build_prompt_schema_text(
            [
                {
                    "table_name": "t1",
                    "row_count": 2,
                    "columns": [{"name": "fiscal_year", "sqlite_type": "INTEGER"}],
                    "sample_rows": [{"fiscal_year": 2025}],
                }
            ]
        )
        self.assertIn("Table: t1", text)
        self.assertIn("fiscal_year (INTEGER)", text)

    def test_report_can_be_written(self):
        result = {"database_path": "db", "table_count": 1, "tables": [], "prompt_schema_text": "x"}
        path = write_report(self.project_root.parent, "component6_sql_schema_packaging_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["table_count"], 1)


if __name__ == "__main__":
    unittest.main()
