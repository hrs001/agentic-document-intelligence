import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.corrective_answer_repair import (
    build_repair_input,
    choose_repair_strategy,
    repair_grounded_answer,
    write_report,
)
from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.tests.test_cross_source_evidence_fusion import build_sample_orchestration_result


class CorrectiveAnswerRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.fused_bundle = fuse_cross_source_evidence(build_sample_orchestration_result())

    def test_choose_repair_strategy_returns_no_op_when_not_needed(self):
        critique_result = {
            "final_critique": {
                "needs_correction": False,
                "issues": [],
                "repair_plan": [],
            }
        }
        strategy = choose_repair_strategy(critique_result)
        self.assertEqual(strategy["strategy"], "no_op")

    def test_choose_repair_strategy_prioritizes_coverage_gap(self):
        critique_result = {
            "final_critique": {
                "needs_correction": True,
                "issues": [{"issue_type": "coverage_gap"}],
                "repair_plan": ["Answer the missing sub-query."],
            }
        }
        strategy = choose_repair_strategy(critique_result)
        self.assertEqual(strategy["strategy"], "targeted_coverage_repair")

    def test_build_repair_input_contains_strategy_and_answer_input(self):
        answer_result = {
            "answer_markdown": "GitHub is in Intelligent Cloud [graph_edge::e1].",
            "used_fact_ids": ["graph_edge::e1"],
            "citations": [],
            "unanswered_sub_queries": ["What was its FY2025 revenue?"],
            "confidence": "medium",
        }
        critique_result = {
            "final_critique": {
                "needs_correction": True,
                "issues": [{"issue_type": "coverage_gap"}],
                "repair_plan": ["Answer the revenue part too."],
            },
            "deterministic_signals": {"needs_correction": True},
        }
        strategy = choose_repair_strategy(critique_result)
        repair_input = build_repair_input(self.fused_bundle, answer_result, critique_result, strategy)
        self.assertEqual(repair_input["strategy"]["strategy"], "targeted_coverage_repair")
        self.assertEqual(repair_input["original_query"], self.fused_bundle["original_query"])
        self.assertIn("answer_input", repair_input)

    def test_repair_grounded_answer_returns_original_for_no_op(self):
        answer_result = {
            "original_query": self.fused_bundle["original_query"],
            "model": "gpt-5.1",
            "answer_markdown": "GitHub is included in Intelligent Cloud [graph_edge::e1]. Revenue was $108,610 million [sql::sq_1::1].",
            "used_fact_ids": ["graph_edge::e1", "sql::sq_1::1"],
            "citations": [],
            "unanswered_sub_queries": [],
            "confidence": "high",
        }
        critique_result = {
            "final_critique": {
                "needs_correction": False,
                "issues": [],
                "repair_plan": [],
            }
        }
        result = repair_grounded_answer(self.fused_bundle, answer_result, critique_result, model="gpt-5.1", client=None)
        self.assertFalse(result["repair_applied"])
        self.assertTrue(result["repair_success"])

    def test_report_can_be_written(self):
        result = {
            "original_query": "x",
            "repair_model": "gpt-5.1",
            "repair_strategy": {"strategy": "no_op"},
            "repair_applied": False,
            "repair_success": True,
            "repaired_answer": {"answer_markdown": "x"},
            "post_repair_deterministic_signals": {},
        }
        path = write_report(self.project_root.parent, "component8_corrective_answer_repair_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["repair_model"], "gpt-5.1")


if __name__ == "__main__":
    unittest.main()
