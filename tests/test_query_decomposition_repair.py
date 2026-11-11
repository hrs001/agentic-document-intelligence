import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.query_decomposition_repair import (
    normalize_repair_result,
    should_repair,
    write_report,
)


class QueryDecompositionRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_gate_skips_clean_result(self):
        result = {
            "needs_decomposition": True,
            "sub_queries": [
                "Who is the CEO of Microsoft?",
                "What was Microsoft's FY2025 revenue?",
            ],
        }
        gate = should_repair("Who is the CEO of Microsoft? What was FY2025 revenue?", result)
        self.assertFalse(gate["should_repair"])

    def test_gate_flags_unresolved_reference(self):
        result = {
            "needs_decomposition": True,
            "sub_queries": ["Who is the CEO?", "What did they say about AI?"],
        }
        gate = should_repair("Who is the CEO and what did they say about AI?", result)
        self.assertTrue(gate["should_repair"])
        self.assertIn("unresolved_reference", gate["reasons"])

    def test_gate_flags_possible_under_split(self):
        result = {
            "needs_decomposition": False,
            "sub_queries": ["Summarize AI demand and cybersecurity risks?"],
        }
        gate = should_repair("Summarize AI demand and cybersecurity risks.", result)
        self.assertTrue(gate["should_repair"])
        self.assertIn("possible_under_split", gate["reasons"])

    def test_normalize_repair_result(self):
        payload = {
            "needs_decomposition": True,
            "sub_queries": ["Who is the CEO of Microsoft?"],
            "reasoning_type": "multi_question",
            "decomposition_strategy": "repair_normalized",
            "repair_notes": ["made subject explicit"],
        }
        result = normalize_repair_result("Who is the CEO?", payload)
        self.assertEqual(result["repair_notes"], ["made subject explicit"])

    def test_report_can_be_written(self):
        original = {"sub_queries": ["Who is the CEO?"]}
        repaired = {"sub_queries": ["Who is the CEO of Microsoft?"], "repair_applied": True}
        path = write_report(
            self.project_root,
            "component3_query_decomposition_repair_test",
            "Who is the CEO?",
            original,
            repaired,
            "gpt-5-mini",
        )
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
