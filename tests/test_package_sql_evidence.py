import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.package_sql_evidence import (
    build_markdown_table,
    package_sql_evidence,
    write_report,
)


class PackageSqlEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_build_markdown_table_includes_headers_and_rows(self):
        table = build_markdown_table(
            ["segment_name", "revenue_usd_millions"],
            [{"segment_name": "Intelligent Cloud", "revenue_usd_millions": 108610}],
        )
        self.assertIn("| segment_name | revenue_usd_millions |", table)
        self.assertIn("Intelligent Cloud", table)

    def test_package_sql_evidence_builds_bundle_summary(self):
        report = {
            "user_query": "Which segment had the highest revenue in FY2025?",
            "database_path": "db.sqlite",
            "target_tables": ["financial_performance_by_segment"],
            "validated_sql": "SELECT segment_name, revenue_usd_millions FROM financial_performance_by_segment",
            "row_count": 1,
            "columns": ["segment_name", "revenue_usd_millions"],
            "rows": [{"segment_name": "Intelligent Cloud", "revenue_usd_millions": 108610}],
            "confidence": "high",
        }
        bundle = package_sql_evidence(report)
        self.assertEqual(bundle["bundle_summary"]["row_count"], 1)
        self.assertEqual(bundle["bundle_summary"]["column_count"], 2)
        self.assertIn("[SQL Evidence]", bundle["assembled_sql_evidence_text"])

    def test_report_can_be_written(self):
        result = {
            "user_query": "x",
            "database_path": "db.sqlite",
            "target_tables": ["t1"],
            "validated_sql": "SELECT 1",
            "confidence": "high",
            "bundle_summary": {"column_count": 1, "row_count": 1, "preview_row_count": 1},
            "columns": ["x"],
            "preview_rows": [{"x": 1}],
            "assembled_sql_evidence_text": "[SQL Evidence]",
        }
        path = write_report(self.project_root.parent, "component6_sql_evidence_packaging_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["bundle_summary"]["preview_row_count"], 1)


if __name__ == "__main__":
    unittest.main()
