import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.llm_transformation_gating import (
    normalize_result,
    write_report,
)


class LLMTransformationGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_normalize_result(self):
        baseline = {
            "estimated_sub_queries": 2,
            "run_decomposition": True,
            "run_multi_query": False,
            "run_step_back": False,
            "run_hyde": False,
            "is_ambiguous_case": True,
            "ambiguity_signals": ["ambiguous_reference"],
            "reasons": ["compound_or_multi_intent"],
        }
        payload = {
            "estimated_sub_queries": 2,
            "run_decomposition": True,
            "run_multi_query": True,
            "run_step_back": False,
            "run_hyde": False,
            "selector_notes": ["lexical variation may help"],
        }
        result = normalize_result("What did they say about AI and margins?", baseline, payload)
        self.assertTrue(result["run_multi_query"])
        self.assertTrue(result["llm_selector_applied"])

    def test_report_can_be_written(self):
        result = {
            "query": "What did they say about AI and margins?",
            "estimated_sub_queries": 2,
            "run_decomposition": True,
            "run_multi_query": True,
            "run_step_back": False,
            "run_hyde": False,
            "is_ambiguous_case": True,
            "ambiguity_signals": ["ambiguous_reference"],
            "reasons": ["compound_or_multi_intent"],
            "selector_notes": ["lexical variation may help"],
            "llm_selector_applied": True,
        }
        path = write_report(self.project_root, "component3_llm_transformation_gating_test", result, "gpt-5-mini")
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
