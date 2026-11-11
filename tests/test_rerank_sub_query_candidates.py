import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.rerank_sub_query_candidates import (
    apply_rerank,
    build_rerank_documents,
    rerank_sub_query_results,
    write_report,
)


class FakeRerankItem:
    def __init__(self, score: float, document: dict[str, str]):
        self.score = score
        self.document = document


class FakeRerankResponse:
    def __init__(self, data):
        self.data = data


class FakeInference:
    def rerank(self, **kwargs):
        docs = kwargs["documents"]
        ordered = list(reversed(docs[: kwargs["top_n"]]))
        return FakeRerankResponse(
            [
                FakeRerankItem(score=1.0 - (i * 0.1), document={"id": doc["id"]})
                for i, doc in enumerate(ordered)
            ]
        )


class FakePinecone:
    def __init__(self):
        self.inference = FakeInference()


def make_match(match_id: str, source_chunk_id: str) -> dict:
    return {
        "id": match_id,
        "source_chunk_id": source_chunk_id,
        "best_score": 1.2,
        "match_count": 2,
        "metadata": {
            "section_title": "Section",
            "page": 1,
            "content_type": "text",
        },
        "child_text": f"child {source_chunk_id}",
        "parent_text": f"parent {source_chunk_id}",
        "provenance_list": [],
        "variant_types": ["multi_query"],
        "query_angles": ["direct"],
        "matched_query_texts": ["query"],
    }


class RerankSubQueryCandidatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_build_rerank_documents_includes_context(self):
        docs = build_rerank_documents([make_match("m1", "chunk_1")])
        self.assertEqual(len(docs), 1)
        self.assertIn("Section:", docs[0]["text"])
        self.assertIn("Child Text:", docs[0]["text"])

    def test_apply_rerank_attaches_scores(self):
        matches = [make_match("m1", "chunk_1"), make_match("m2", "chunk_2")]
        rerank_result = FakeRerankResponse(
            [
                FakeRerankItem(0.9, {"id": "m2"}),
                FakeRerankItem(0.8, {"id": "m1"}),
            ]
        )
        reranked = apply_rerank(matches, rerank_result)
        self.assertEqual(reranked[0]["id"], "m2")
        self.assertEqual(reranked[0]["rerank_score"], 0.9)

    def test_rerank_sub_query_results_builds_per_sub_query_output(self):
        result = {
            "original_query": "query",
            "policy": {},
            "updated_sub_query_results": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "Who is the CEO?",
                    "merged_matches": [make_match("m1", "chunk_1"), make_match("m2", "chunk_2")],
                }
            ],
        }
        reranked = rerank_sub_query_results(result, FakePinecone(), top_n=2)
        self.assertEqual(reranked["rerank_summary"]["sub_query_count"], 1)
        self.assertEqual(reranked["sub_query_results"][0]["reranked_count"], 2)

    def test_write_report_persists_payload(self):
        result = {
            "original_query": "query",
            "policy": {},
            "rerank_summary": {"sub_query_count": 1},
            "sub_query_results": [],
        }
        path = write_report(self.project_root, "component4_sub_query_rerank_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
