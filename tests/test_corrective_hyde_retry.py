import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.corrective_hyde_retry import (
    merge_retry_matches,
    run_corrective_hyde_retry,
    should_trigger_hyde,
    write_report,
)


def make_existing_sub_query(sub_query_id: str) -> dict:
    return {
        "sub_query_id": sub_query_id,
        "original_sub_query": "What did the CEO say about AI?",
        "merged_match_count": 1,
        "merged_matches": [
            {
                "id": "chunk_1_record",
                "source_chunk_id": "chunk_1",
                "best_score": 0.9,
                "match_count": 1,
                "metadata": {
                    "source_chunk_id": "chunk_1",
                    "section_title": "Finance",
                    "page": 1,
                    "page_end": 1,
                    "parent_id": "parent_1",
                    "content_type": "text",
                },
                "child_text": "Revenue rose in FY2025.",
                "parent_text": "Revenue rose in FY2025.",
                "provenance_list": [
                    {
                        "sub_query_id": sub_query_id,
                        "variant_id": f"{sub_query_id}_original",
                        "variant_type": "original_sub_query",
                        "query_text": "What did the CEO say about AI?",
                        "query_angle": "original",
                    }
                ],
                "variant_types": ["original_sub_query"],
                "query_angles": ["original"],
                "matched_query_texts": ["What did the CEO say about AI?"],
            }
        ],
    }


class CorrectiveHydeRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_should_trigger_hyde_for_clearly_weak_case(self):
        score = {
            "coverage_label": "moderate",
            "signal_breakdown": {
                "top_score": 1.3,
                "score_strength": 0.2,
                "query_overlap_ratio": 0.08,
                "reinforced_match_strength": 0.0,
                "result_depth_strength": 0.2,
            },
        }
        triggered, reasons = should_trigger_hyde(score)
        self.assertTrue(triggered)
        self.assertTrue(reasons)

    def test_should_not_trigger_hyde_for_strong_case(self):
        score = {
            "coverage_label": "strong",
            "signal_breakdown": {
                "top_score": 3.5,
                "score_strength": 1.0,
                "query_overlap_ratio": 0.6,
                "reinforced_match_strength": 1.0,
                "result_depth_strength": 1.0,
            },
        }
        triggered, reasons = should_trigger_hyde(score)
        self.assertFalse(triggered)
        self.assertEqual(reasons, [])

    def test_merge_retry_matches_adds_hyde_provenance(self):
        existing = make_existing_sub_query("sq_1")
        retry_matches = [
            {
                "id": "chunk_2_record",
                "score": 1.8,
                "metadata": {
                    "source_chunk_id": "chunk_2",
                    "section_title": "AI",
                    "page": 2,
                    "page_end": 2,
                    "parent_id": "parent_2",
                    "content_type": "text",
                },
                "child_text": "Management said AI demand is increasing.",
                "parent_text": "Management said AI demand is increasing.",
                "provenance": {
                    "sub_query_id": "sq_1",
                    "variant_id": "sq_1_hyde",
                    "variant_type": "hyde",
                    "query_text": "AI demand and CEO AI commentary",
                    "query_angle": "factual_stub",
                },
            }
        ]
        merged = merge_retry_matches(existing, retry_matches)
        self.assertEqual(merged["merged_match_count"], 2)
        self.assertEqual(merged["merged_matches"][0]["source_chunk_id"], "chunk_2")

    def test_run_corrective_hyde_retry_triggers_only_weak_sub_queries(self):
        coverage_result = {
            "original_query": "query",
            "policy": {},
            "sub_query_scores": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "What did the CEO say about AI?",
                    "coverage_label": "moderate",
                    "signal_breakdown": {
                        "top_score": 1.3,
                        "score_strength": 0.2,
                        "query_overlap_ratio": 0.08,
                        "reinforced_match_strength": 0.0,
                        "result_depth_strength": 0.2,
                    },
                },
                {
                    "sub_query_id": "sq_2",
                    "original_sub_query": "Who is the CEO of Microsoft?",
                    "coverage_label": "strong",
                    "signal_breakdown": {
                        "top_score": 3.5,
                        "score_strength": 1.0,
                        "query_overlap_ratio": 0.8,
                        "reinforced_match_strength": 1.0,
                        "result_depth_strength": 1.0,
                    },
                },
            ],
        }
        merge_result = {
            "original_query": "query",
            "policy": {},
            "sub_query_results": [
                make_existing_sub_query("sq_1"),
                {
                    **make_existing_sub_query("sq_2"),
                    "sub_query_id": "sq_2",
                    "original_sub_query": "Who is the CEO of Microsoft?",
                },
            ],
        }

        def fake_hyde(query: str) -> dict:
            return {
                "original_query": query,
                "hypothetical_passage": "Management said AI demand remained a major growth theme.",
                "generation_style": "factual_stub",
            }

        def fake_executor(variant, *_args):
            return {
                "variant": variant,
                "raw_match_count": 1,
                "deduped_match_count": 1,
                "matches": [
                    {
                        "id": "chunk_2_record",
                        "score": 1.8,
                        "metadata": {
                            "source_chunk_id": "chunk_2",
                            "section_title": "AI",
                            "page": 2,
                            "page_end": 2,
                            "parent_id": "parent_2",
                            "content_type": "text",
                        },
                        "child_text": "Management said AI demand is increasing.",
                        "parent_text": "Management said AI demand is increasing.",
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

        result = run_corrective_hyde_retry(
            coverage_result,
            merge_result,
            index=None,
            namespace="ns",
            alpha=0.6,
            top_k=5,
            openai_client=None,
            pinecone_client=None,
            child_index={},
            parent_index={},
            hyde_generator=fake_hyde,
            retry_executor=fake_executor,
        )
        self.assertEqual(result["retry_summary"]["triggered_hyde_count"], 1)
        self.assertEqual(result["retry_summary"]["triggered_sub_query_ids"], ["sq_1"])

    def test_write_report_persists_payload(self):
        result = {
            "original_query": "query",
            "policy": {},
            "retry_summary": {"triggered_hyde_count": 0},
        }
        path = write_report(self.project_root, "component4_corrective_hyde_retry_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
