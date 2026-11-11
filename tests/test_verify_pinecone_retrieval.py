import json
import unittest
from pathlib import Path

from scripts.extract_layout_aware_document import build_layout_artifact
from scripts.generate_chunk_records import build_chunk_artifact
from scripts.verify_pinecone_retrieval import build_chunk_indexes, hydrate_matches, write_report


class _MockMatch:
    def __init__(self, match_id: str, score: float, metadata: dict):
        self.id = match_id
        self.score = score
        self.metadata = metadata


class PineconeRetrievalVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        layout = build_layout_artifact(cls.project_root, "microsoft_fy2025_10k_summary")
        cls.chunk_artifact = build_chunk_artifact(layout)
        cls.child_index, cls.parent_index = build_chunk_indexes(cls.chunk_artifact)

    def test_indexes_build(self):
        self.assertGreater(len(self.child_index), 0)
        self.assertGreater(len(self.parent_index), 0)

    def test_hydration_restores_child_and_parent_text(self):
        first_chunk = self.chunk_artifact["chunks"][0]
        mock = _MockMatch(
            match_id="mock1",
            score=0.99,
            metadata={
                "source_chunk_id": first_chunk["child_id"],
                "parent_id": first_chunk["parent_id"],
                "section_title": first_chunk["metadata"]["section_title"],
                "page": first_chunk["metadata"]["page"],
            },
        )
        hydrated = hydrate_matches([mock], self.child_index, self.parent_index)
        self.assertEqual(len(hydrated), 1)
        self.assertTrue(hydrated[0]["child_text"])
        self.assertTrue(hydrated[0]["parent_text"])

    def test_hydration_preserves_metadata(self):
        first_chunk = self.chunk_artifact["chunks"][0]
        mock = _MockMatch(
            match_id="mock1",
            score=0.5,
            metadata={
                "source_chunk_id": first_chunk["child_id"],
                "parent_id": first_chunk["parent_id"],
                "content_type": first_chunk["metadata"].get("content_type", "text"),
            },
        )
        hydrated = hydrate_matches([mock], self.child_index, self.parent_index)
        self.assertEqual(hydrated[0]["metadata"]["parent_id"], first_chunk["parent_id"])

    def test_report_can_be_written(self):
        report = {"ok": True, "match_count": 1}
        out_path = write_report(self.project_root, "component2_pinecone_retrieval_test", report)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["match_count"], 1)


if __name__ == "__main__":
    unittest.main()
