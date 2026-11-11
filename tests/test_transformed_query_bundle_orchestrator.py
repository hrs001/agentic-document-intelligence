import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.transformed_query_bundle_orchestrator import (
    MAX_SUB_QUERIES,
    build_transformed_query_bundle,
    recommend_hyde,
    write_report,
)


class TransformedQueryBundleOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_recommend_hyde_for_short_sparse_query(self):
        result = recommend_hyde("Azure growth?")
        self.assertTrue(result["should_consider_hyde_after_weak_retrieval"])

    def test_bundle_caps_sub_queries(self):
        def fake_decompose(_: str):
            return {
                "needs_decomposition": True,
                "sub_queries": ["Q1?", "Q2?", "Q3?", "Q4?"],
                "reasoning_type": "multi_intent",
                "decomposition_strategy": "question_boundary_split",
            }

        def fake_multi(q: str):
            return {
                "original_query": q,
                "rewrite_count": 2,
                "rewrites": [
                    {"query": f"{q} alt 1", "angle": "direct"},
                    {"query": f"{q} alt 2", "angle": "financial_disclosure"},
                ],
            }

        def fake_step_back(q: str):
            return {
                "original_query": q,
                "step_back_query": f"broader {q}",
                "broadening_strategy": "broaden_to_explanation",
            }

        bundle = build_transformed_query_bundle(
            "complex query",
            decomposition_fn=fake_decompose,
            multi_query_fn=fake_multi,
            step_back_fn=fake_step_back,
        )
        self.assertEqual(bundle["bundle_summary"]["sub_query_count"], MAX_SUB_QUERIES)
        self.assertEqual(bundle["bundle_summary"]["total_multi_query_rewrites"], MAX_SUB_QUERIES * 2)

    def test_bundle_writes_report(self):
        bundle = {
            "original_query": "test",
            "bundle_summary": {
                "sub_query_count": 1,
                "total_multi_query_rewrites": 2,
                "step_back_count": 1,
                "hyde_candidate_count": 0,
            },
        }
        path = write_report(self.project_root, "component3_transformed_query_bundle_test", bundle)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
