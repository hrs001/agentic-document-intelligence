import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_hyde_query_generation import (
    evaluate_case,
)
from agentic_document_intelligence.scripts.hyde_query_generation import (
    normalize_result,
    write_report,
)


class HyDEQueryGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_normalize_result(self):
        payload = {
            "original_query": "What was revenue?",
            "hypothetical_passage": "Microsoft reported revenue in its fiscal disclosures. The discussion linked revenue performance to business demand.",
            "generation_style": "financial_stub",
        }
        result = normalize_result("What was revenue?", payload)
        self.assertEqual(result["generation_style"], "financial_stub")

    def test_evaluate_case_passes_valid_result(self):
        actual = {
            "hypothetical_passage": "Microsoft discussed Azure growth in relation to AI demand. Management described how AI-related demand influenced Azure performance."
        }
        expected = {"required_terms": ["azure", "ai"], "forbidden_terms": ["google", "amazon"]}
        evaluation = evaluate_case(actual, expected)
        self.assertTrue(evaluation["passed"])

    def test_report_can_be_written(self):
        result = {
            "original_query": "What was revenue?",
            "hypothetical_passage": "Microsoft reported revenue. The filing discussed the drivers behind revenue changes.",
            "generation_style": "financial_stub",
        }
        path = write_report(self.project_root, "component3_hyde_query_generation_test", result, "gpt-5-mini")
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
