import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.latency_optimized_orchestration_policy import (
    build_latency_optimized_policy,
    extract_policy_signals,
    sanitize_policy_decision,
    write_report,
)


class LatencyOptimizedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_extract_policy_signals_detects_reference(self):
        signals = extract_policy_signals(
            "What narrative driver was mentioned for that segment?",
            ["sql_structured", "vector_document"],
        )
        self.assertTrue(signals["has_reference"])
        self.assertTrue(signals["has_vector"])

    def test_sanitize_policy_decision_forces_vector_profile_when_vector_selected(self):
        signals = extract_policy_signals("What did management say about AI demand?", ["vector_document"])
        decision = sanitize_policy_decision(
            "What did management say about AI demand?",
            ["vector_document"],
            {"vector_profile": "skip", "parallel_safe": True, "reasoning": "x", "confidence": "medium"},
            signals,
        )
        self.assertNotEqual(decision["vector_profile"], "skip")
        self.assertFalse(decision["parallel_safe"])

    def test_sanitize_policy_decision_raises_vector_depth_for_executive_role_query(self):
        signals = extract_policy_signals("Who is the CEO of Microsoft?", ["vector_document"])
        decision = sanitize_policy_decision(
            "Who is the CEO of Microsoft?",
            ["vector_document"],
            {"vector_profile": "fast", "parallel_safe": False, "reasoning": "x", "confidence": "medium"},
            signals,
        )
        self.assertEqual(decision["vector_profile"], "balanced")
        self.assertTrue(decision["signals"]["asks_for_executive_role"])

    def test_build_latency_optimized_policy_collects_profile_counts(self):
        transformed_bundle = {"original_query": "x"}
        routing_plan = {
            "sub_query_plans": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "Which segment had the highest revenue?",
                    "routing_decision": {"selected_sources": ["sql_structured"]},
                },
                {
                    "sub_query_id": "sq_2",
                    "original_sub_query": "What did management say about AI demand?",
                    "routing_decision": {"selected_sources": ["vector_document"]},
                },
            ]
        }

        import agentic_document_intelligence.scripts.latency_optimized_orchestration_policy as mod

        original = mod.plan_sub_query_execution

        def fake_plan(sub_query, selected_sources, model="gpt-5-mini", client=None):
            return {
                "sub_query": sub_query,
                "selected_sources": selected_sources,
                "vector_profile": "skip" if "vector_document" not in selected_sources else "balanced",
                "parallel_safe": False,
                "reasoning": "x",
                "confidence": "high",
                "signals": {},
            }

        mod.plan_sub_query_execution = fake_plan
        try:
            result = build_latency_optimized_policy("x", transformed_bundle, routing_plan)
        finally:
            mod.plan_sub_query_execution = original

        self.assertEqual(result["policy_summary"]["vector_profile_counts"]["skip"], 1)
        self.assertEqual(result["policy_summary"]["vector_profile_counts"]["balanced"], 1)

    def test_sanitize_policy_decision_marks_independent_sql_graph_as_parallel_safe(self):
        query = "Which segment includes GitHub, and what was FY2025 revenue mix by geography?"
        signals = extract_policy_signals(query, ["graph_relationships", "sql_structured"])
        decision = sanitize_policy_decision(
            query,
            ["graph_relationships", "sql_structured"],
            {
                "active_sources": ["graph_relationships", "sql_structured"],
                "vector_profile": "skip",
                "parallel_safe": False,
                "reasoning": "x",
                "confidence": "medium",
            },
            signals,
        )
        self.assertTrue(decision["parallel_safe"])
        self.assertEqual(decision["active_sources"], ["graph_relationships", "sql_structured"])

    def test_sanitize_policy_decision_marks_independent_graph_vector_as_parallel_safe(self):
        query = "Which segment includes GitHub?"
        signals = extract_policy_signals(query, ["graph_relationships", "vector_document"])
        decision = sanitize_policy_decision(
            query,
            ["graph_relationships", "vector_document"],
            {
                "active_sources": ["graph_relationships", "vector_document"],
                "vector_profile": "balanced",
                "parallel_safe": False,
                "reasoning": "x",
                "confidence": "medium",
            },
            signals,
        )
        self.assertTrue(decision["parallel_safe"])
        self.assertEqual(decision["active_sources"], ["graph_relationships", "vector_document"])

    def test_report_can_be_written(self):
        path = write_report(
            self.project_root.parent,
            "component7_latency_policy_test",
            {"policy_summary": {"sub_query_count": 1, "vector_profile_counts": {}, "parallel_safe_count": 0}},
        )
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
