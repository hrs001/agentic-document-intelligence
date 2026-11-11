import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_llm_query_decomposition import (
    evaluate_case,
    normalize_text,
)
from agentic_document_intelligence.scripts.llm_query_decomposition import (
    normalize_result,
    write_report,
)


class LLMQueryDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_normalize_result_preserves_valid_payload(self):
        payload = {
            "needs_decomposition": True,
            "sub_queries": ["Who is the CEO of Microsoft?", "What was FY2025 revenue?"],
            "reasoning_type": "multi_question",
            "decomposition_strategy": "question_boundary_split",
        }
        result = normalize_result("Who is the CEO?", payload)
        self.assertTrue(result["needs_decomposition"])
        self.assertEqual(len(result["sub_queries"]), 2)

    def test_normalize_result_falls_back_when_empty(self):
        payload = {
            "needs_decomposition": False,
            "sub_queries": [],
            "reasoning_type": "single_intent",
            "decomposition_strategy": "none",
        }
        result = normalize_result("What was revenue?", payload)
        self.assertEqual(result["sub_queries"], ["What was revenue?"])

    def test_evaluate_case_detects_missing_items(self):
        actual = {"needs_decomposition": True, "sub_queries": ["Who is the CEO of Microsoft?"]}
        expected = {
            "needs_decomposition": True,
            "sub_queries": ["Who is the CEO of Microsoft?", "What was FY2025 revenue?"],
        }
        evaluation = evaluate_case("query", actual, expected)
        self.assertFalse(evaluation["passed"])
        self.assertIn("what was fy2025 revenue", evaluation["missing_sub_queries"])

    def test_evaluate_case_allows_semantic_equivalence(self):
        actual = {
            "needs_decomposition": True,
            "sub_queries": [
                "Who is Microsoft's CEO?",
                "What was Microsoft's fiscal year 2025 revenue?",
            ],
        }
        expected = {
            "needs_decomposition": True,
            "sub_queries": [
                "Who is the CEO of Microsoft?",
                "What was FY2025 revenue?",
            ],
        }
        evaluation = evaluate_case("query", actual, expected)
        self.assertTrue(evaluation["passed"])
        self.assertFalse(evaluation["exact_passed"])

    def test_normalize_text_is_stable(self):
        self.assertEqual(normalize_text("  What was revenue? "), "what was revenue")

    def test_report_can_be_written(self):
        result = {
            "needs_decomposition": False,
            "sub_queries": ["What was revenue?"],
            "reasoning_type": "single_intent",
            "decomposition_strategy": "none",
        }
        path = write_report(
            self.project_root,
            "component3_llm_query_decomposition_test",
            "What was revenue?",
            result,
            "gpt-5-mini",
        )
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
