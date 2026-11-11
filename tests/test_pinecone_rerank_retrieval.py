import json
import unittest
from pathlib import Path

from scripts.pinecone_rerank_retrieval import apply_rerank, build_rerank_documents, write_report


class _MockRerankItem:
    def __init__(self, score, index, document=None):
        self.score = score
        self.index = index
        self.document = document


class _MockRerankResult:
    def __init__(self, data):
        self.data = data


class PineconeRerankRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.matches = [
            {
                "id": "doc1",
                "metadata": {"section_title": "Section A", "page": 1, "content_type": "text", "source_chunk_id": "c1"},
                "child_text": "Satya Nadella is the CEO of Microsoft.",
                "parent_text": "Satya Nadella is the CEO of Microsoft and Chairman.",
            },
            {
                "id": "doc2",
                "metadata": {"section_title": "Section B", "page": 2, "content_type": "text", "source_chunk_id": "c2"},
                "child_text": "Azure grew 34%.",
                "parent_text": "Azure grew 34% in FY2025.",
            },
        ]

    def test_build_rerank_documents(self):
        docs = build_rerank_documents(self.matches)
        self.assertEqual(len(docs), 2)
        self.assertIn("Section: Section A", docs[0]["text"])

    def test_apply_rerank_uses_returned_order(self):
        rerank = _MockRerankResult([_MockRerankItem(0.9, 1), _MockRerankItem(0.8, 0)])
        ordered = apply_rerank(self.matches, rerank)
        self.assertEqual(ordered[0]["id"], "doc2")
        self.assertIn("rerank_score", ordered[0])

    def test_report_can_be_written(self):
        report = {"ok": True, "reranked_count": 2}
        out_path = write_report(self.project_root, "component2_pinecone_rerank_test", report)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["reranked_count"], 2)


if __name__ == "__main__":
    unittest.main()
