import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.runtime_quality_gating import (
    build_retry_state,
    decide_runtime_action,
    write_report,
)


class RuntimeQualityGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_decide_runtime_action_accepts_when_critique_and_judge_pass(self):
        critique_result = {
            "final_critique": {"needs_correction": False, "issues": []},
            "deterministic_signals": {},
        }
        judge_result = {"metrics": {"overall_verdict": "pass", "faithfulness": 5, "answer_relevancy": 5, "context_precision": 5, "citation_grounding": 5}, "average_score": 5.0}
        state = build_retry_state()
        decision = decide_runtime_action(critique_result, judge_result, state)
        self.assertEqual(decision["action"], "stop_accept")

    def test_decide_runtime_action_stops_when_budget_exhausted(self):
        critique_result = {
            "final_critique": {"needs_correction": True, "issues": [{"issue_type": "coverage_gap"}]},
            "deterministic_signals": {},
        }
        judge_result = {"metrics": {"overall_verdict": "fail", "faithfulness": 2, "answer_relevancy": 2, "context_precision": 2, "citation_grounding": 2}, "average_score": 2.0}
        state = build_retry_state()
        state["total_rounds_used"] = 2
        decision = decide_runtime_action(critique_result, judge_result, state, max_total_rounds=2)
        self.assertEqual(decision["action"], "stop_best_effort")

    def test_decide_runtime_action_prefers_citation_repair(self):
        critique_result = {
            "final_critique": {"needs_correction": True, "issues": []},
            "deterministic_signals": {"missing_inline_for_used": ["sql::sq_1::1"], "inline_not_declared": [], "unknown_used_fact_ids": []},
        }
        judge_result = {"metrics": {"overall_verdict": "borderline", "faithfulness": 4, "answer_relevancy": 4, "context_precision": 4, "citation_grounding": 2}, "average_score": 3.5}
        state = build_retry_state()
        decision = decide_runtime_action(critique_result, judge_result, state)
        self.assertEqual(decision["action"], "citation_strict_repair")

    def test_decide_runtime_action_prefers_answer_regeneration_for_low_relevancy(self):
        critique_result = {
            "final_critique": {"needs_correction": False, "issues": []},
            "deterministic_signals": {},
        }
        judge_result = {"metrics": {"overall_verdict": "borderline", "faithfulness": 4, "answer_relevancy": 2, "context_precision": 4, "citation_grounding": 4}, "average_score": 3.5}
        state = build_retry_state()
        decision = decide_runtime_action(critique_result, judge_result, state)
        self.assertEqual(decision["action"], "answer_regeneration_only")

    def test_decide_runtime_action_allows_pipeline_rerun_for_low_faithfulness(self):
        critique_result = {
            "final_critique": {"needs_correction": False, "issues": []},
            "deterministic_signals": {},
        }
        judge_result = {"metrics": {"overall_verdict": "fail", "faithfulness": 1, "answer_relevancy": 4, "context_precision": 2, "citation_grounding": 4}, "average_score": 2.75}
        state = build_retry_state()
        decision = decide_runtime_action(critique_result, judge_result, state)
        self.assertEqual(decision["action"], "full_pipeline_rerun_once")

    def test_decide_runtime_action_stops_on_non_improving_repeat(self):
        critique_result = {
            "final_critique": {"needs_correction": True, "issues": [{"issue_type": "coverage_gap"}]},
            "deterministic_signals": {},
        }
        judge_result = {"metrics": {"overall_verdict": "fail", "faithfulness": 2, "answer_relevancy": 2, "context_precision": 2, "citation_grounding": 2}, "average_score": 2.0}
        state = build_retry_state()
        state["actions_taken"] = ["targeted_answer_repair"]
        state["last_judge_average_score"] = 2.5
        decision = decide_runtime_action(critique_result, judge_result, state)
        self.assertEqual(decision["action"], "stop_best_effort")

    def test_report_can_be_written(self):
        result = {
            "original_query": "x",
            "termination": {"action": "stop_accept"},
            "retry_state": {"total_rounds_used": 0},
            "history": [],
        }
        path = write_report(self.project_root.parent, "component10_runtime_quality_gating_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["termination"]["action"], "stop_accept")


if __name__ == "__main__":
    unittest.main()
