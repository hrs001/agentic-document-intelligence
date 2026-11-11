import json
import unittest
from pathlib import Path

from scripts.extract_layout_aware_document import build_layout_artifact
from scripts.generate_chunk_records import build_chunk_artifact
from scripts.pinecone_hybrid_retrieval import dedupe_matches, hydrate_matches, build_chunk_indexes, write_report


class _MockMatch:
    def __init__(self, match_id: str, score: float, metadata: dict):
        self.id = match_id
        self.score = score
        self.metadata = metadata


class PineconeHybridRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        layout = build_layout_artifact(cls.project_root, "microsoft_fy2025_10k_summary")
        chunks = build_chunk_artifact(layout)
        cls.child_index, cls.parent_index = build_chunk_indexes(chunks)
        cls.sample_chunk = chunks["chunks"][0]

    def test_dedup_removes_duplicate_source_chunk_ids(self):
        metadata = {"source_chunk_id": self.sample_chunk["child_id"], "parent_id": self.sample_chunk["parent_id"]}
        matches = [_MockMatch("a", 0.9, metadata), _MockMatch("b", 0.8, metadata)]
        deduped = dedupe_matches(matches)
        self.assertEqual(len(deduped), 1)

    def test_hydration_recovers_parent_text(self):
        match = _MockMatch(
            "a",
            0.9,
            {"source_chunk_id": self.sample_chunk["child_id"], "parent_id": self.sample_chunk["parent_id"]},
        )
        hydrated = hydrate_matches([match], self.child_index, self.parent_index)
        self.assertTrue(hydrated[0]["child_text"])
        self.assertTrue(hydrated[0]["parent_text"])

    def test_report_can_be_written(self):
        report = {"ok": True, "deduped_match_count": 3}
        out_path = write_report(self.project_root, "component2_pinecone_hybrid_retrieval_test", report)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["deduped_match_count"], 3)


if __name__ == "__main__":
    unittest.main()
