import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_multi_query_generation import (
    evaluate_case,
    has_duplicate_rewrites,
)
from agentic_document_intelligence.scripts.multi_query_generation import (
    normalize_result,
    write_report,
)


class MultiQueryGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_normalize_result_dedupes_and_caps(self):
        payload = {
            "rewrites": [
                {"query": "What was revenue?", "angle": "direct"},
                {"query": "What was revenue?", "angle": "financial_disclosure"},
                {"query": "Revenue in filings", "angle": "financial_disclosure"},
                {"query": "Revenue in annual report", "angle": "management_commentary"},
            ]
        }
        result = normalize_result("What was revenue?", payload)
        self.assertEqual(result["rewrite_count"], 3)

    def test_duplicate_detection(self):
        self.assertTrue(has_duplicate_rewrites(["Revenue?", "revenue?"]))

    def test_evaluate_case_passes_valid_result(self):
        actual = {
            "rewrites": [
                {"query": "What was Microsoft's FY2025 revenue?", "angle": "direct"},
                {"query": "What revenue did Microsoft report for FY2025?", "angle": "financial_disclosure"},
            ]
        }
        expected = {"rewrite_count_range": [2, 3], "required_angles": ["direct", "financial_disclosure"]}
        evaluation = evaluate_case(actual, expected)
        self.assertTrue(evaluation["passed"])

    def test_report_can_be_written(self):
        result = {
            "original_query": "What was revenue?",
            "rewrite_count": 2,
            "rewrites": [
                {"query": "What was revenue?", "angle": "direct"},
                {"query": "What revenue was reported?", "angle": "financial_disclosure"},
            ],
        }
        path = write_report(self.project_root, "component3_multi_query_generation_test", result, "gpt-5-mini")
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
