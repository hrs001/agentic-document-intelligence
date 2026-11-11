import json
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.execute_latency_optimized_orchestration import (
    assemble_fast_vector_evidence,
    execute_latency_optimized_orchestration,
    write_report,
)


class ExecuteLatencyOptimizedOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_assemble_fast_vector_evidence_builds_context(self):
        merged = {
            "original_query": "x",
            "policy": {},
            "sub_query_results": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "What did management say about AI demand?",
                    "merged_matches": [
                        {
                            "source_chunk_id": "c1",
                            "metadata": {
                                "parent_id": "p1",
                                "section_title": "AI",
                                "page": 1,
                                "page_end": 2,
                                "content_type": "text",
                                "document_id": "d1",
                                "source_file": "f",
                            },
                            "child_text": "AI demand remained strong.",
                            "parent_text": "Parent context.",
                            "best_score": 2.1,
                            "match_count": 1,
                            "variant_types": ["original_sub_query"],
                            "query_angles": ["original"],
                        }
                    ],
                }
            ],
        }
        bundle = assemble_fast_vector_evidence(merged)
        self.assertEqual(bundle["bundle_summary"]["total_evidence_items"], 1)
        self.assertIn("AI demand remained strong.", bundle["assembled_evidence_text"])

    def test_report_can_be_written(self):
        path = write_report(
            self.project_root.parent,
            "component7_latency_optimized_exec_test",
            {"execution_summary": {"sub_query_count": 0, "source_usage": {}, "vector_profile_usage": {}}},
        )
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)

    def test_parallel_sql_graph_execution_runs_concurrently(self):
        transformed_bundle = {
            "original_query": "x",
            "sub_query_bundles": [{"sub_query_id": "sq_1", "original_sub_query": "x"}],
        }
        routing_plan = {
            "policy": {},
            "routing_summary": {"sub_query_count": 1},
            "sub_query_plans": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "x",
                    "routing_decision": {"selected_sources": ["sql_structured", "graph_relationships"]},
                }
            ],
        }
        policy_plan = {
            "policy": {},
            "policy_summary": {"sub_query_count": 1},
            "sub_query_execution_plans": [
                {
                    "sub_query_id": "sq_1",
                    "execution_policy": {
                        "active_sources": ["sql_structured", "graph_relationships"],
                        "vector_profile": "skip",
                        "parallel_safe": True,
                    },
                }
            ],
        }

        import agentic_document_intelligence.scripts.execute_latency_optimized_orchestration as mod

        original_resolve = mod.resolve_sub_query_with_context
        original_graph = mod.execute_graph_source
        original_sql = mod.execute_sql_source
        try:
            mod.resolve_sub_query_with_context = lambda *args, **kwargs: {
                "resolved_sub_query": "x",
                "used_context": False,
                "reasoning": "none",
            }

            def fake_graph(*args, **kwargs):
                time.sleep(0.2)
                return {"source": "graph_relationships", "evidence_bundle": {"assembled_graph_evidence_text": "g"}}

            def fake_sql(*args, **kwargs):
                time.sleep(0.2)
                return {"source": "sql_structured", "evidence_bundle": {"assembled_sql_evidence_text": "s"}}

            mod.execute_graph_source = fake_graph
            mod.execute_sql_source = fake_sql

            start = time.perf_counter()
            result = execute_latency_optimized_orchestration(
                query="x",
                transformed_bundle=transformed_bundle,
                routing_plan=routing_plan,
                policy_plan=policy_plan,
                schema_package={},
                sql_database_path=Path("db.sqlite"),
                graph_database_path=Path("graph.kuzu"),
                index=None,
                namespace="ns",
                alpha=0.6,
                top_k=4,
                openai_client=None,
                pinecone_client=None,
                child_index={},
                parent_index={},
                chunk_to_record_id={},
                model="gpt-5-mini",
            )
            elapsed = time.perf_counter() - start
        finally:
            mod.resolve_sub_query_with_context = original_resolve
            mod.execute_graph_source = original_graph
            mod.execute_sql_source = original_sql

        self.assertLess(elapsed, 0.35)
        outputs = result["sub_query_results"][0]["source_outputs"]
        self.assertEqual([item["source"] for item in outputs], ["sql_structured", "graph_relationships"])

    def test_parallel_graph_vector_execution_runs_concurrently(self):
        transformed_bundle = {
            "original_query": "x",
            "sub_query_bundles": [{"sub_query_id": "sq_1", "original_sub_query": "x"}],
        }
        routing_plan = {
            "policy": {},
            "routing_summary": {"sub_query_count": 1},
            "sub_query_plans": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "x",
                    "routing_decision": {"selected_sources": ["graph_relationships", "vector_document"]},
                }
            ],
        }
        policy_plan = {
            "policy": {},
            "policy_summary": {"sub_query_count": 1},
            "sub_query_execution_plans": [
                {
                    "sub_query_id": "sq_1",
                    "execution_policy": {
                        "active_sources": ["graph_relationships", "vector_document"],
                        "vector_profile": "balanced",
                        "parallel_safe": True,
                    },
                }
            ],
        }

        import agentic_document_intelligence.scripts.execute_latency_optimized_orchestration as mod

        original_resolve = mod.resolve_sub_query_with_context
        original_graph = mod.execute_graph_source
        original_vector = mod.execute_vector_source_optimized
        try:
            mod.resolve_sub_query_with_context = lambda *args, **kwargs: {
                "resolved_sub_query": "x",
                "used_context": False,
                "reasoning": "none",
            }

            def fake_graph(*args, **kwargs):
                time.sleep(0.2)
                return {"source": "graph_relationships", "evidence_bundle": {"assembled_graph_evidence_text": "g"}}

            def fake_vector(*args, **kwargs):
                time.sleep(0.2)
                return {
                    "source": "vector_document",
                    "execution_profile": "balanced",
                    "evidence_bundle": {"assembled_evidence_text": "v"},
                }

            mod.execute_graph_source = fake_graph
            mod.execute_vector_source_optimized = fake_vector

            start = time.perf_counter()
            result = execute_latency_optimized_orchestration(
                query="x",
                transformed_bundle=transformed_bundle,
                routing_plan=routing_plan,
                policy_plan=policy_plan,
                schema_package={},
                sql_database_path=Path("db.sqlite"),
                graph_database_path=Path("graph.kuzu"),
                index=None,
                namespace="ns",
                alpha=0.6,
                top_k=4,
                openai_client=None,
                pinecone_client=None,
                child_index={},
                parent_index={},
                chunk_to_record_id={},
                model="gpt-5-mini",
            )
            elapsed = time.perf_counter() - start
        finally:
            mod.resolve_sub_query_with_context = original_resolve
            mod.execute_graph_source = original_graph
            mod.execute_vector_source_optimized = original_vector

        self.assertLess(elapsed, 0.35)
        outputs = result["sub_query_results"][0]["source_outputs"]
        self.assertEqual([item["source"] for item in outputs], ["graph_relationships", "vector_document"])


if __name__ == "__main__":
    unittest.main()
