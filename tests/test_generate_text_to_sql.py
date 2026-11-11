import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.generate_text_to_sql import (
    load_schema_package,
    sanitize_generation_result,
    write_report,
)


class GenerateTextToSqlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.schema_package = load_schema_package(
            cls.project_root
            / "artifacts"
            / "experiments"
            / "component6_sql_schema_packaging_live"
            / "sql_schema"
            / "sql_schema_package.json"
        )

    def test_sanitize_generation_result_keeps_select(self):
        payload = {
            "user_query": "Which segment had the highest revenue in FY2025?",
            "target_tables": ["financial_performance_by_segment"],
            "sql_query": "SELECT segment_name FROM financial_performance_by_segment WHERE fiscal_year = 2025;",
            "rationale": "Need revenue by segment for FY2025.",
            "confidence": "high",
        }
        result = sanitize_generation_result(payload["user_query"], payload, self.schema_package)
        self.assertEqual(result["sql_query"], "SELECT segment_name FROM financial_performance_by_segment WHERE fiscal_year = 2025")

    def test_sanitize_generation_result_blocks_non_select(self):
        payload = {
            "user_query": "Bad",
            "target_tables": ["financial_performance_by_segment"],
            "sql_query": "DROP TABLE financial_performance_by_segment",
            "rationale": "bad",
            "confidence": "low",
        }
        with self.assertRaises(ValueError):
            sanitize_generation_result(payload["user_query"], payload, self.schema_package)

    def test_report_can_be_written(self):
        result = {
            "user_query": "x",
            "target_tables": ["t1"],
            "sql_query": "SELECT 1",
            "rationale": "r",
            "confidence": "high",
        }
        path = write_report(self.project_root.parent, "component6_text_to_sql_generation_test", result, "gpt-5-mini")
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
