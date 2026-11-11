import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.execute_multi_source_orchestration import (
    build_single_sub_query_bundle,
    execute_routed_orchestration,
    write_report,
)


class ExecuteMultiSourceOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_build_single_sub_query_bundle_filters_to_one_sub_query(self):
        transformed_bundle = {
            "original_query": "x",
            "policy": {},
            "decomposition_result": {},
            "sub_query_bundles": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "q1",
                    "multi_query_result": {"rewrite_count": 2},
                    "step_back_result": {},
                    "hyde_recommendation": {"should_consider_hyde_after_weak_retrieval": False},
                },
                {
                    "sub_query_id": "sq_2",
                    "original_sub_query": "q2",
                    "multi_query_result": {"rewrite_count": 1},
                    "step_back_result": {},
                    "hyde_recommendation": {"should_consider_hyde_after_weak_retrieval": True},
                },
            ],
        }
        mini = build_single_sub_query_bundle(transformed_bundle, "sq_2")
        self.assertEqual(len(mini["sub_query_bundles"]), 1)
        self.assertEqual(mini["sub_query_bundles"][0]["sub_query_id"], "sq_2")

    def test_execute_routed_orchestration_dispatches_selected_sources(self):
        transformed_bundle = {
            "original_query": "x",
            "policy": {},
            "decomposition_result": {},
            "sub_query_bundles": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "who is the ceo",
                    "multi_query_result": {"rewrite_count": 1},
                    "step_back_result": {},
                    "hyde_recommendation": {"should_consider_hyde_after_weak_retrieval": False},
                }
            ],
        }
        routing_plan = {
            "policy": {},
            "routing_summary": {"sub_query_count": 1},
            "sub_query_plans": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "who is the ceo",
                    "routing_decision": {
                        "selected_sources": ["graph_relationships", "vector_document"],
                    },
                }
            ],
        }

        def fake_graph(*_args, **_kwargs):
            return {"source": "graph_relationships", "evidence_bundle": {"assembled_graph_evidence_text": "Satya Nadella"}}

        def fake_sql(*_args, **_kwargs):
            return {"source": "sql_structured", "evidence_bundle": {"assembled_sql_evidence_text": "x"}}

        def fake_vector(*_args, **_kwargs):
            return {"source": "vector_document", "evidence_bundle": {"assembled_evidence_text": "CEO evidence"}}

        result = execute_routed_orchestration(
            query="x",
            transformed_bundle=transformed_bundle,
            routing_plan=routing_plan,
            schema_package={},
            sql_database_path=Path("db.sqlite"),
            graph_database_path=Path("graph.kuzu"),
            index=None,
            namespace="ns",
            alpha=0.6,
            top_k=6,
            openai_client=None,
            pinecone_client=None,
            child_index={},
            parent_index={},
            chunk_to_record_id={},
            model="gpt-5-mini",
            rerank_top_n=6,
            mmr_top_m=4,
            mmr_lambda=0.75,
            graph_executor=fake_graph,
            sql_executor=fake_sql,
            vector_executor=fake_vector,
        )
        outputs = result["sub_query_results"][0]["source_outputs"]
        self.assertEqual([item["source"] for item in outputs], ["graph_relationships", "vector_document"])

    def test_report_can_be_written(self):
        result = {
            "original_query": "x",
            "policy": {},
            "routing_summary": {},
            "execution_summary": {"sub_query_count": 0, "source_usage": {}},
            "sub_query_results": [],
        }
        path = write_report(self.project_root.parent, "component7_multi_source_orchestration_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
