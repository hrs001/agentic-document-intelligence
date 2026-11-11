import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.query_decomposition import (
    decompose_query,
    sanitize_query,
    write_report,
)


class QueryDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_single_intent_query_is_not_split(self):
        result = decompose_query("What was Microsoft's FY2025 revenue?")
        self.assertFalse(result["needs_decomposition"])
        self.assertEqual(len(result["sub_queries"]), 1)

    def test_multi_question_query_is_split(self):
        result = decompose_query("Who is the CEO of Microsoft? What was FY2025 revenue?")
        self.assertTrue(result["needs_decomposition"])
        self.assertEqual(
            result["sub_queries"],
            ["Who is the CEO of Microsoft", "What was FY2025 revenue"],
        )

    def test_multi_intent_conjunction_query_is_split(self):
        result = decompose_query(
            "Explain Microsoft's FY2025 revenue growth and identify who the CEO is."
        )
        self.assertTrue(result["needs_decomposition"])
        self.assertEqual(len(result["sub_queries"]), 2)

    def test_compare_query_is_kept_atomic(self):
        result = decompose_query("Compare Azure and Microsoft 365 revenue drivers.")
        self.assertFalse(result["needs_decomposition"])
        self.assertEqual(result["sub_queries"], ["Compare Azure and Microsoft 365 revenue drivers?"])

    def test_list_query_with_commas_is_split(self):
        result = decompose_query(
            "List Microsoft's FY2025 revenue, operating income, and net income."
        )
        self.assertTrue(result["needs_decomposition"])
        self.assertEqual(
            result["sub_queries"],
            [
                "List Microsoft's FY2025 revenue?",
                "List Microsoft's FY2025 operating income?",
                "List Microsoft's FY2025 net income?",
            ],
        )

    def test_short_fragments_are_not_promoted(self):
        result = decompose_query("Revenue and CEO")
        self.assertFalse(result["needs_decomposition"])

    def test_whitespace_and_zero_width_are_normalized(self):
        cleaned = sanitize_query("Who\u200b is   the CEO?\r\n\r\nWhat was revenue?")
        self.assertNotIn("\u200b", cleaned)
        self.assertNotIn("  ", cleaned)

    def test_empty_query_returns_empty_result(self):
        result = decompose_query("   ")
        self.assertFalse(result["needs_decomposition"])
        self.assertEqual(result["sub_queries"], [])
        self.assertEqual(result["reasoning_type"], "empty")

    def test_duplicate_clauses_are_deduped(self):
        result = decompose_query("Show revenue and show revenue.")
        self.assertFalse(result["needs_decomposition"])
        self.assertEqual(len(result["sub_queries"]), 1)

    def test_report_can_be_written(self):
        result = decompose_query("Who is the CEO of Microsoft?")
        out_path = write_report(
            self.project_root,
            "component3_query_decomposition_test",
            "Who is the CEO of Microsoft?",
            result,
        )
        self.assertTrue(out_path.exists())
        written = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertIn("result", written)


if __name__ == "__main__":
    unittest.main()
