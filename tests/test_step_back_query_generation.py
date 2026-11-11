import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_step_back_query_generation import (
    evaluate_case,
)
from agentic_document_intelligence.scripts.step_back_query_generation import (
    normalize_result,
    write_report,
)


class StepBackQueryGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_normalize_result(self):
        payload = {
            "original_query": "What was revenue?",
            "step_back_query": "What broader business context explains Microsoft's revenue performance?",
            "broadening_strategy": "broaden_to_explanation",
        }
        result = normalize_result("What was revenue?", payload)
        self.assertEqual(result["broadening_strategy"], "broaden_to_explanation")

    def test_evaluate_case_passes_valid_result(self):
        actual = {
            "original_query": "What was Microsoft's FY2025 revenue?",
            "step_back_query": "What broader business context explains Microsoft's FY2025 revenue performance?",
        }
        expected = {"required_terms": ["microsoft", "revenue"], "forbidden_terms": ["google", "amazon"]}
        evaluation = evaluate_case(actual, expected)
        self.assertTrue(evaluation["passed"])

    def test_report_can_be_written(self):
        result = {
            "original_query": "What was revenue?",
            "step_back_query": "What broader business context explains revenue performance?",
            "broadening_strategy": "broaden_to_context",
        }
        path = write_report(self.project_root, "component3_step_back_query_generation_test", result, "gpt-5-mini")
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
