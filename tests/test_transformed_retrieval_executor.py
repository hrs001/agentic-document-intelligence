import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.transformed_retrieval_executor import (
    build_query_variants,
    execute_transformed_retrieval,
    write_report,
)


class TransformedRetrievalExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_build_query_variants_counts_all_types(self):
        bundle = {
            "original_query": "query",
            "policy": {},
            "bundle_summary": {"sub_query_count": 1},
            "sub_query_bundles": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "Who is the CEO?",
                    "multi_query_result": {
                        "rewrites": [
                            {"query": "Identify the CEO.", "angle": "direct"},
                            {"query": "CEO in filings", "angle": "financial_disclosure"},
                        ]
                    },
                    "step_back_result": {
                        "step_back_query": "How can the CEO be identified from authoritative sources?",
                        "broadening_strategy": "broaden_to_explanation",
                    },
                    "hyde_recommendation": {"should_consider_hyde_after_weak_retrieval": True},
                }
            ],
        }
        variants = build_query_variants(bundle)
        self.assertEqual(len(variants), 4)

    def test_execute_transformed_retrieval_uses_variant_executor(self):
        bundle = {
            "original_query": "query",
            "policy": {},
            "bundle_summary": {"sub_query_count": 1},
            "sub_query_bundles": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "Who is the CEO?",
                    "multi_query_result": {
                        "rewrites": [
                            {"query": "Identify the CEO.", "angle": "direct"},
                        ],
                        "rewrite_count": 1,
                    },
                    "step_back_result": {
                        "step_back_query": "How can the CEO be identified from authoritative sources?",
                        "broadening_strategy": "broaden_to_explanation",
                    },
                    "hyde_recommendation": {"should_consider_hyde_after_weak_retrieval": True},
                }
            ],
        }

        def fake_executor(variant, *_args):
            return {
                "variant": variant,
                "raw_match_count": 2,
                "deduped_match_count": 1,
                "matches": [
                    {
                        "id": "m1",
                        "score": 0.9,
                        "child_text": "child",
                        "parent_text": "parent",
                        "metadata": {},
                        "provenance": {
                            "sub_query_id": variant["sub_query_id"],
                            "variant_id": variant["variant_id"],
                            "variant_type": variant["variant_type"],
                            "query_text": variant["query_text"],
                            "query_angle": variant["query_angle"],
                        },
                    }
                ],
            }

        result = execute_transformed_retrieval(
            bundle,
            index=None,
            namespace="ns",
            alpha=0.6,
            top_k=5,
            openai_client=None,
            pinecone_client=None,
            child_index={},
            parent_index={},
            variant_executor=fake_executor,
        )
        self.assertEqual(result["retrieval_summary"]["variant_count"], 3)
        self.assertEqual(result["retrieval_summary"]["total_deduped_matches"], 3)

    def test_report_can_be_written(self):
        result = {"retrieval_summary": {"variant_count": 3}}
        path = write_report(self.project_root, "component4_transformed_retrieval_executor_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
