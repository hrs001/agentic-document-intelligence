import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.multi_source_routing import (
    build_graph_capability_summary,
    build_multi_source_routing_plan,
    build_sql_capability_summary,
    extract_routing_signals,
    load_json_result,
    sanitize_routing_decision,
    write_report,
)


class MultiSourceRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        schema_package = load_json_result(
            cls.project_root
            / "artifacts"
            / "experiments"
            / "component6_sql_schema_packaging_live"
            / "sql_schema"
            / "sql_schema_package.json"
        )
        cls.sql_capability_summary = build_sql_capability_summary(schema_package)
        cls.graph_capability_summary = build_graph_capability_summary()

    def test_extract_routing_signals_detects_numeric_and_relationship_intent(self):
        signals = extract_routing_signals("Which segment includes GitHub and had the highest revenue?")
        self.assertGreater(signals["numeric_signal_count"], 0)
        self.assertGreater(signals["relationship_signal_count"], 0)
        self.assertTrue(signals["looks_multi_source"])

    def test_extract_routing_signals_detects_executive_role_intent(self):
        signals = extract_routing_signals("Who is the CEO of Microsoft?")
        self.assertTrue(signals["executive_role_signal"])
        self.assertGreater(signals["executive_role_signal_count"], 0)

    def test_sanitize_routing_decision_adds_vector_for_narrative_query(self):
        signals = extract_routing_signals("What did management say about AI demand?")
        decision = sanitize_routing_decision(
            "What did management say about AI demand?",
            {
                "primary_source": "graph_relationships",
                "selected_sources": ["graph_relationships"],
                "reasoning": "x",
                "confidence": "medium",
            },
            signals,
        )
        self.assertIn("vector_document", decision["selected_sources"])

    def test_sanitize_routing_decision_forces_vector_for_executive_role_query(self):
        signals = extract_routing_signals("Who is the CEO of Microsoft?")
        decision = sanitize_routing_decision(
            "Who is the CEO of Microsoft?",
            {
                "primary_source": "graph_relationships",
                "selected_sources": ["graph_relationships"],
                "reasoning": "x",
                "confidence": "medium",
            },
            signals,
        )
        self.assertEqual(decision["primary_source"], "vector_document")
        self.assertIn("vector_document", decision["selected_sources"])
        self.assertNotIn("graph_relationships", decision["selected_sources"])

    def test_sanitize_routing_decision_pairs_graph_with_vector(self):
        signals = extract_routing_signals("Which segment includes GitHub?")
        decision = sanitize_routing_decision(
            "Which segment includes GitHub?",
            {
                "primary_source": "graph_relationships",
                "selected_sources": ["graph_relationships"],
                "reasoning": "relationship query",
                "confidence": "high",
            },
            signals,
        )
        self.assertEqual(decision["primary_source"], "graph_relationships")
        self.assertIn("graph_relationships", decision["selected_sources"])
        self.assertIn("vector_document", decision["selected_sources"])

    def test_build_multi_source_routing_plan_aggregates_source_usage(self):
        bundle = {
            "original_query": "x",
            "sub_query_bundles": [
                {"sub_query_id": "sq_1", "original_sub_query": "Who is the CEO?"},
                {"sub_query_id": "sq_2", "original_sub_query": "Which segment had the highest revenue?"},
            ],
        }

        def fake_route(sub_query, _sql_summary, _graph_summary, _model):
            if "revenue" in sub_query.lower():
                return {
                    "sub_query": sub_query,
                    "primary_source": "sql_structured",
                    "selected_sources": ["sql_structured"],
                    "reasoning": "metric lookup",
                    "confidence": "high",
                }
            return {
                "sub_query": sub_query,
                "primary_source": "graph_relationships",
                "selected_sources": ["graph_relationships", "vector_document"],
                "reasoning": "relationship query",
                "confidence": "high",
            }

        result = build_multi_source_routing_plan(
            bundle,
            self.sql_capability_summary,
            self.graph_capability_summary,
            route_fn=fake_route,
        )
        self.assertEqual(result["routing_summary"]["sub_query_count"], 2)
        self.assertEqual(result["routing_summary"]["source_usage"]["sql_structured"], 1)
        self.assertEqual(result["routing_summary"]["source_usage"]["graph_relationships"], 1)

    def test_report_can_be_written(self):
        result = {
            "original_query": "x",
            "policy": {"routing_model": "gpt-5-mini"},
            "sub_query_plans": [],
            "routing_summary": {"sub_query_count": 0, "source_usage": {}},
        }
        path = write_report(self.project_root.parent, "component7_multi_source_routing_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
