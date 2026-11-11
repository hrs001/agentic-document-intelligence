import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_final_answer_pipeline import (
    evaluate_case_end_to_end,
    write_report,
)


class EvaluateFinalAnswerPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_evaluate_case_end_to_end_tracks_repair_and_final_pass(self):
        import agentic_document_intelligence.scripts.evaluate_final_answer_pipeline as mod

        original_generate = mod.generate_grounded_answer
        original_critique = mod.critique_grounded_answer
        original_repair = mod.repair_grounded_answer
        original_judge = mod.judge_final_answer
        try:
            mod.generate_grounded_answer = lambda *args, **kwargs: {
                "answer_markdown": "GitHub is included in Intelligent Cloud.",
                "used_fact_ids": ["graph_edge::e1"],
                "unanswered_sub_queries": ["What was its FY2025 revenue?"],
                "citations": [],
                "confidence": "medium",
            }
            mod.critique_grounded_answer = lambda *args, **kwargs: {
                "final_critique": {"needs_correction": True, "issues": [{"issue_type": "coverage_gap"}], "repair_plan": []},
                "deterministic_signals": {"needs_correction": True},
            }
            mod.repair_grounded_answer = lambda *args, **kwargs: {
                "repair_applied": True,
                "repair_success": True,
                "repair_strategy": {"strategy": "targeted_coverage_repair"},
                "repaired_answer": {
                    "answer_markdown": "GitHub is included in Intelligent Cloud [graph_edge::e1]. Revenue was $108,610 million [sql::sq_1::1].",
                    "used_fact_ids": ["graph_edge::e1", "sql::sq_1::1"],
                    "unanswered_sub_queries": [],
                    "citations": [],
                    "confidence": "high",
                },
            }
            mod.judge_final_answer = lambda *args, **kwargs: {
                "metrics": {"overall_verdict": "pass"},
                "average_score": 4.5,
            }
            case = {
                "case_id": "x",
                "query": "Which segment includes GitHub and what was its FY2025 revenue?",
                "required_phrases": ["GitHub", "Intelligent Cloud", "108610"],
                "required_source_types": ["graph_relationships", "sql_structured"],
                "min_citation_count": 2,
            }
            fused_bundle = {
                "normalized_facts": [
                    {"fact_id": "graph_edge::e1", "source_type": "graph_relationships"},
                    {"fact_id": "sql::sq_1::1", "source_type": "sql_structured"},
                ]
            }
            result = evaluate_case_end_to_end(
                case,
                fused_bundle,
                "gpt-5.1",
                "gpt-5-mini",
                "gpt-5.1",
                "gpt-5-mini",
                openai_client=None,
            )
        finally:
            mod.generate_grounded_answer = original_generate
            mod.critique_grounded_answer = original_critique
            mod.repair_grounded_answer = original_repair
            mod.judge_final_answer = original_judge

        self.assertFalse(result["initial_evaluation"]["passed"])
        self.assertTrue(result["final_evaluation"]["passed"])
        self.assertTrue(result["repair_applied"])
        self.assertTrue(result["repair_success"])
        self.assertEqual(result["judge_result"]["metrics"]["overall_verdict"], "pass")

    def test_write_report_can_be_written(self):
        report = {
            "case_count": 1,
            "initial_pass_rate": 0.5,
            "final_pass_rate": 1.0,
            "repaired_count": 1,
            "recovered_count": 1,
            "case_results": [],
        }
        path = write_report(self.project_root.parent, "component9_final_answer_pipeline_eval_test", report)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["final_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
