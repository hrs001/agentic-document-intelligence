import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.retrieval_merge_dedup import (
    merge_variant_results,
    write_report,
)


def make_match(
    source_chunk_id: str,
    score: float,
    sub_query_id: str,
    variant_id: str,
    variant_type: str,
    query_text: str,
    query_angle: str,
) -> dict:
    return {
        "id": f"{source_chunk_id}_record",
        "score": score,
        "metadata": {
            "source_chunk_id": source_chunk_id,
            "section_title": "Section",
            "page": 1,
            "page_end": 1,
            "parent_id": f"{source_chunk_id}_parent",
            "content_type": "text",
        },
        "child_text": f"child {source_chunk_id}",
        "parent_text": f"parent {source_chunk_id}",
        "provenance": {
            "sub_query_id": sub_query_id,
            "variant_id": variant_id,
            "variant_type": variant_type,
            "query_text": query_text,
            "query_angle": query_angle,
        },
    }


class RetrievalMergeDedupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_merge_deduplicates_within_sub_query(self):
        result = {
            "original_query": "query",
            "policy": {},
            "retrieval_summary": {"variant_count": 2},
            "variant_results": [
                {
                    "variant": {
                        "sub_query_id": "sq_1",
                        "variant_id": "sq_1_original",
                        "variant_type": "original_sub_query",
                        "query_text": "Who is the CEO?",
                    },
                    "matches": [
                        make_match(
                            "chunk_1",
                            1.2,
                            "sq_1",
                            "sq_1_original",
                            "original_sub_query",
                            "Who is the CEO?",
                            "original",
                        ),
                        make_match(
                            "chunk_2",
                            0.8,
                            "sq_1",
                            "sq_1_original",
                            "original_sub_query",
                            "Who is the CEO?",
                            "original",
                        ),
                    ],
                },
                {
                    "variant": {
                        "sub_query_id": "sq_1",
                        "variant_id": "sq_1_mq_1",
                        "variant_type": "multi_query",
                        "query_text": "Chief executive officer",
                    },
                    "matches": [
                        make_match(
                            "chunk_1",
                            2.4,
                            "sq_1",
                            "sq_1_mq_1",
                            "multi_query",
                            "Chief executive officer",
                            "direct",
                        ),
                    ],
                },
            ],
        }

        merged = merge_variant_results(result)
        self.assertEqual(merged["merge_summary"]["duplicates_removed"], 1)
        self.assertEqual(merged["sub_query_results"][0]["merged_match_count"], 2)
        top_match = merged["sub_query_results"][0]["merged_matches"][0]
        self.assertEqual(top_match["source_chunk_id"], "chunk_1")
        self.assertEqual(top_match["best_score"], 2.4)
        self.assertEqual(top_match["match_count"], 2)

    def test_merge_preserves_provenance_and_variant_types(self):
        result = {
            "original_query": "query",
            "policy": {},
            "retrieval_summary": {"variant_count": 2},
            "variant_results": [
                {
                    "variant": {
                        "sub_query_id": "sq_1",
                        "variant_id": "sq_1_original",
                        "variant_type": "original_sub_query",
                        "query_text": "Q1",
                    },
                    "matches": [
                        make_match(
                            "chunk_1",
                            1.0,
                            "sq_1",
                            "sq_1_original",
                            "original_sub_query",
                            "Q1",
                            "original",
                        ),
                    ],
                },
                {
                    "variant": {
                        "sub_query_id": "sq_1",
                        "variant_id": "sq_1_step_back",
                        "variant_type": "step_back",
                        "query_text": "Q1 broader",
                    },
                    "matches": [
                        make_match(
                            "chunk_1",
                            0.7,
                            "sq_1",
                            "sq_1_step_back",
                            "step_back",
                            "Q1 broader",
                            "broaden_to_context",
                        ),
                    ],
                },
            ],
        }

        merged = merge_variant_results(result)
        match = merged["sub_query_results"][0]["merged_matches"][0]
        self.assertEqual(len(match["provenance_list"]), 2)
        self.assertIn("original_sub_query", match["variant_types"])
        self.assertIn("step_back", match["variant_types"])
        self.assertIn("Q1 broader", match["matched_query_texts"])

    def test_same_chunk_is_not_cross_deduped_between_sub_queries(self):
        result = {
            "original_query": "query",
            "policy": {},
            "retrieval_summary": {"variant_count": 2},
            "variant_results": [
                {
                    "variant": {
                        "sub_query_id": "sq_1",
                        "variant_id": "sq_1_original",
                        "variant_type": "original_sub_query",
                        "query_text": "Q1",
                    },
                    "matches": [
                        make_match(
                            "chunk_shared",
                            1.0,
                            "sq_1",
                            "sq_1_original",
                            "original_sub_query",
                            "Q1",
                            "original",
                        ),
                    ],
                },
                {
                    "variant": {
                        "sub_query_id": "sq_2",
                        "variant_id": "sq_2_original",
                        "variant_type": "original_sub_query",
                        "query_text": "Q2",
                    },
                    "matches": [
                        make_match(
                            "chunk_shared",
                            1.1,
                            "sq_2",
                            "sq_2_original",
                            "original_sub_query",
                            "Q2",
                            "original",
                        ),
                    ],
                },
            ],
        }

        merged = merge_variant_results(result)
        self.assertEqual(merged["merge_summary"]["duplicates_removed"], 0)
        self.assertEqual(len(merged["sub_query_results"]), 2)

    def test_write_report_persists_payload(self):
        merged = {
            "original_query": "query",
            "policy": {},
            "merge_summary": {"sub_query_count": 1},
            "sub_query_results": [],
        }
        path = write_report(self.project_root, "component4_retrieval_merge_dedup_test", merged)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
