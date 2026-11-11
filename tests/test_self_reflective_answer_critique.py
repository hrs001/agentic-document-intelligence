import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.scripts.self_reflective_answer_critique import (
    apply_deterministic_overrides,
    build_critique_input,
    deterministic_reflection_checks,
    extract_inline_fact_ids,
    sanitize_critique_payload,
    write_report,
)
from agentic_document_intelligence.tests.test_cross_source_evidence_fusion import build_sample_orchestration_result


class SelfReflectiveAnswerCritiqueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.fused_bundle = fuse_cross_source_evidence(build_sample_orchestration_result())

    def test_extract_inline_fact_ids_reads_bracketed_ids(self):
        answer = "GitHub is in Intelligent Cloud [graph_edge::e1] and revenue was [sql::sq_1::1]."
        self.assertEqual(extract_inline_fact_ids(answer), ["graph_edge::e1", "sql::sq_1::1"])

    def test_deterministic_checks_flag_missing_inline_and_unanswered(self):
        answer_result = {
            "answer_markdown": "GitHub is in Intelligent Cloud.",
            "used_fact_ids": ["graph_edge::e1", "sql::sq_1::1"],
            "unanswered_sub_queries": ["What was its FY2025 revenue?"],
            "confidence": "medium",
        }
        signals = deterministic_reflection_checks(self.fused_bundle, answer_result)
        self.assertTrue(signals["needs_correction"])
        self.assertIn("graph_edge::e1", signals["missing_inline_for_used"])
        self.assertIn("What was its FY2025 revenue?", signals["unanswered_sub_queries"])

    def test_deterministic_checks_flag_inline_not_declared_and_unknown_fact(self):
        answer_result = {
            "answer_markdown": "Claim [bad::fact].",
            "used_fact_ids": ["unknown::1"],
            "unanswered_sub_queries": [],
            "confidence": "medium",
        }
        signals = deterministic_reflection_checks(self.fused_bundle, answer_result)
        self.assertIn("bad::fact", signals["inline_not_declared"])
        self.assertIn("unknown::1", signals["unknown_used_fact_ids"])

    def test_apply_deterministic_overrides_forces_correction(self):
        llm_critique = {
            "grounded": True,
            "complete": True,
            "needs_correction": False,
            "confidence": "high",
            "issue_summary": "Looks fine.",
            "strengths": ["Cited answer."],
            "issues": [],
            "repair_plan": [],
        }
        deterministic_signals = {
            "missing_inline_for_used": ["graph_edge::e1"],
            "inline_not_declared": [],
            "unknown_used_fact_ids": [],
            "unanswered_sub_queries": [],
            "uncovered_sub_queries": [],
            "used_source_types": ["graph_relationships"],
            "used_conflict_fact_ids": [],
            "conflict_acknowledged": False,
            "needs_correction": True,
        }
        merged = apply_deterministic_overrides(llm_critique, deterministic_signals)
        self.assertTrue(merged["needs_correction"])
        self.assertGreaterEqual(len(merged["issues"]), 1)

    def test_build_critique_input_keeps_used_fact_summaries(self):
        answer_result = {
            "answer_markdown": "GitHub is in Intelligent Cloud [graph_edge::e1].",
            "used_fact_ids": ["graph_edge::e1", "sql::sq_1::1"],
            "unanswered_sub_queries": [],
            "confidence": "medium",
        }
        signals = deterministic_reflection_checks(self.fused_bundle, answer_result)
        critique_input = build_critique_input(self.fused_bundle, answer_result, signals)
        self.assertEqual(critique_input["original_query"], self.fused_bundle["original_query"])
        self.assertEqual(len(critique_input["used_facts"]), 2)

    def test_sanitize_critique_payload_normalizes_issue_shape(self):
        payload = sanitize_critique_payload(
            {
                "grounded": True,
                "complete": False,
                "needs_correction": True,
                "confidence": "medium",
                "issue_summary": "Coverage gap.",
                "strengths": ["Some claims are grounded."],
                "issues": [
                    {
                        "issue_type": "coverage_gap",
                        "severity": "high",
                        "description": "One sub-query is missing.",
                        "affected_sub_queries": ["sq_1"],
                        "repair_action": "Answer missing sub-query.",
                    }
                ],
                "repair_plan": ["Answer missing sub-query."],
            }
        )
        self.assertEqual(payload["issues"][0]["issue_type"], "coverage_gap")
        self.assertTrue(payload["needs_correction"])

    def test_report_can_be_written(self):
        result = {
            "original_query": "x",
            "answer_model": "gpt-5.1",
            "critique_model": "gpt-5-mini",
            "deterministic_signals": {},
            "llm_critique": {},
            "final_critique": {"needs_correction": False, "grounded": True, "complete": True},
        }
        path = write_report(self.project_root.parent, "component8_self_reflective_answer_critique_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["answer_model"], "gpt-5.1")


if __name__ == "__main__":
    unittest.main()
