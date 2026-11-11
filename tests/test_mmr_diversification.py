import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.mmr_diversification import (
    cosine_similarity,
    diversify_sub_query_results,
    load_embedding_records,
    select_with_mmr,
    write_report,
)


class FakeVector:
    def __init__(self, values):
        self.values = values


class FakeFetchResponse:
    def __init__(self, vectors):
        self.vectors = vectors


class FakeIndex:
    def __init__(self, vectors):
        self._vectors = vectors

    def fetch(self, ids, namespace):
        return FakeFetchResponse({vector_id: FakeVector(self._vectors[vector_id]) for vector_id in ids})


def make_match(match_id: str, source_chunk_id: str, rerank_score: float) -> dict:
    return {
        "id": match_id,
        "source_chunk_id": source_chunk_id,
        "rerank_score": rerank_score,
        "best_score": rerank_score,
        "match_count": 1,
        "metadata": {"source_chunk_id": source_chunk_id},
        "child_text": f"child {source_chunk_id}",
        "parent_text": f"parent {source_chunk_id}",
        "provenance_list": [],
        "variant_types": [],
        "query_angles": [],
        "matched_query_texts": [],
    }


class MmrDiversificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_cosine_similarity_basic(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_select_with_mmr_prefers_diverse_second_choice(self):
        matches = [
            make_match("m1", "c1", 0.9),
            make_match("m2", "c2", 0.89),
            make_match("m3", "c3", 0.8),
        ]
        vectors_by_match_id = {
            "m1": [1.0, 0.0],
            "m2": [0.99, 0.01],
            "m3": [0.0, 1.0],
        }
        selected = select_with_mmr(matches, vectors_by_match_id, top_m=2, lambda_weight=0.7)
        self.assertEqual(selected[0]["id"], "m1")
        self.assertEqual(selected[1]["id"], "m3")

    def test_diversify_sub_query_results_builds_output(self):
        result = {
            "original_query": "query",
            "policy": {},
            "sub_query_results": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "Who is the CEO?",
                    "candidate_count": 3,
                    "reranked_count": 3,
                    "reranked_matches": [
                        make_match("m1", "c1", 0.9),
                        make_match("m2", "c2", 0.89),
                        make_match("m3", "c3", 0.8),
                    ],
                }
            ],
        }
        chunk_to_record_id = {"c1": "r1", "c2": "r2", "c3": "r3"}
        index = FakeIndex({"r1": [1.0, 0.0], "r2": [0.99, 0.01], "r3": [0.0, 1.0]})
        diversified = diversify_sub_query_results(
            result,
            index,
            namespace="ns",
            chunk_to_record_id=chunk_to_record_id,
            top_m=2,
            lambda_weight=0.7,
        )
        self.assertEqual(diversified["mmr_summary"]["sub_query_count"], 1)
        self.assertEqual(diversified["sub_query_results"][0]["diversified_count"], 2)

    def test_load_embedding_records_maps_source_chunk_to_record_id(self):
        path = (
            self.project_root
            / "artifacts"
            / "experiments"
            / "component2_embedding_ready_records"
            / "embeddings"
            / "microsoft_fy2025_10k_summary_embedding_records.json"
        )
        mapping = load_embedding_records(path)
        self.assertTrue(mapping)

    def test_write_report_persists_payload(self):
        result = {
            "original_query": "query",
            "policy": {},
            "mmr_summary": {"sub_query_count": 1},
            "sub_query_results": [],
        }
        path = write_report(self.project_root, "component4_mmr_diversification_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
