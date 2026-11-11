import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_transformation_gating import evaluate_case
from agentic_document_intelligence.scripts.transformation_gating import (
    MAX_SUB_QUERIES,
    classify_query,
    select_transformations,
    write_report,
)


class TransformationGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_max_sub_queries_is_capped(self):
        result = select_transformations("A and B and C and D and E")
        self.assertEqual(result["max_sub_queries"], MAX_SUB_QUERIES)
        self.assertLessEqual(result["estimated_sub_queries"], MAX_SUB_QUERIES)

    def test_ambiguous_case_detected(self):
        features = classify_query("What did they say about AI and revenue?")
        self.assertTrue(features["is_ambiguous_case"])

    def test_comparison_stays_single(self):
        result = select_transformations("Compare Azure and Microsoft 365 revenue drivers.")
        self.assertFalse(result["run_decomposition"])
        self.assertTrue(result["run_multi_query"])
        self.assertTrue(result["run_step_back"])

    def test_evaluate_case_passes(self):
        actual = {
            "run_decomposition": True,
            "run_multi_query": True,
            "run_step_back": False,
            "run_hyde": False,
            "is_ambiguous_case": True,
            "estimated_sub_queries": 2,
        }
        expected = actual.copy()
        evaluation = evaluate_case(actual, expected)
        self.assertTrue(evaluation["passed"])

    def test_report_can_be_written(self):
        result = select_transformations("What was Microsoft's FY2025 revenue?")
        path = write_report(self.project_root, "component3_transformation_gating_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
