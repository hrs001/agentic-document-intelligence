import json
import sqlite3
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_sql_retrieval_quality import (
    compare_results,
    execute_reference_sql,
    normalize_rows,
    project_rows,
    write_report,
)


class EvaluateSqlRetrievalQualityTests(unittest.TestCase):
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

    def test_execute_reference_sql_returns_expected_rows(self):
        result = execute_reference_sql(
            self.database_path,
            "SELECT geography, revenue_mix_pct FROM geographic_revenue_mix WHERE fiscal_year = 2025 ORDER BY geography",
        )
        self.assertEqual(result["columns"], ["geography", "revenue_mix_pct"])
        self.assertEqual(result["row_count"], 2)

    def test_normalize_rows_rounds_floats(self):
        rows = [{"value": 46.812345}]
        normalized = normalize_rows(["value"], rows)
        self.assertEqual(normalized, [(46.8123,)])

    def test_compare_results_supports_unordered_matches(self):
        actual = {
            "columns": ["name"],
            "rows": [{"name": "Azure"}, {"name": "Windows"}],
            "row_count": 2,
        }
        expected = {
            "columns": ["name"],
            "rows": [{"name": "Windows"}, {"name": "Azure"}],
            "row_count": 2,
        }
        comparison = compare_results(actual, expected, ordered=False)
        self.assertTrue(comparison["strict_passed"])

    def test_project_rows_keeps_expected_columns_only(self):
        rows = [{"name": "Azure", "score": 95, "extra": "x"}]
        projected = project_rows(["name", "score"], rows)
        self.assertEqual(projected, [{"name": "Azure", "score": 95}])

    def test_compare_results_allows_answer_projection_match_with_extra_columns(self):
        actual = {
            "columns": ["name", "score", "extra"],
            "rows": [{"name": "Azure", "score": 95, "extra": "x"}],
            "row_count": 1,
        }
        expected = {
            "columns": ["name", "score"],
            "rows": [{"name": "Azure", "score": 95}],
            "row_count": 1,
        }
        comparison = compare_results(actual, expected, ordered=True)
        self.assertFalse(comparison["strict_passed"])
        self.assertTrue(comparison["answer_passed"])

    def test_report_can_be_written(self):
        report = {
            "model": "gpt-5-mini",
            "database_path": "db.sqlite",
            "case_count": 1,
            "strict_passed_count": 1,
            "strict_pass_rate": 1.0,
            "answer_passed_count": 1,
            "answer_pass_rate": 1.0,
            "case_results": [],
        }
        path = write_report(self.project_root.parent, "component6_sql_retrieval_quality_eval_test", report)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["strict_passed_count"], 1)


if __name__ == "__main__":
    unittest.main()
