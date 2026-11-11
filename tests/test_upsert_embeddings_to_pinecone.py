import json
import unittest
from pathlib import Path

from scripts.extract_layout_aware_document import build_layout_artifact
from scripts.generate_chunk_records import build_chunk_artifact
from scripts.generate_embedding_ready_records import build_embedding_ready_artifact
from scripts.upsert_embeddings_to_pinecone import (
    build_pinecone_metadata,
    build_vectors,
    filter_child_records,
    write_report,
)


class PineconeUpsertPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        layout = build_layout_artifact(cls.project_root, "microsoft_fy2025_10k_summary")
        chunks = build_chunk_artifact(layout)
        ready = build_embedding_ready_artifact(chunks)
        cls.child_records = filter_child_records(ready)
        cls.ready = ready

    def test_only_child_records_are_selected(self):
        self.assertEqual(len(self.child_records), self.ready["record_count"])
        self.assertTrue(all("_ch" in record["source_chunk_id"] for record in self.child_records))

    def test_metadata_contains_parent_linkage(self):
        first = self.child_records[0]
        meta = build_pinecone_metadata(first)
        self.assertIn("parent_id", meta)
        self.assertIn("section_title", meta)
        self.assertIn("page", meta)

    def test_build_vectors_preserves_record_identity(self):
        vectors = build_vectors(self.child_records[:2], [[0.1] * 1536, [0.2] * 1536])
        self.assertEqual(vectors[0]["id"], self.child_records[0]["record_id"])
        self.assertEqual(len(vectors[0]["values"]), 1536)

    def test_table_metadata_survives(self):
        table = next(record for record in self.child_records if record["content_type"] == "table")
        meta = build_pinecone_metadata(table)
        self.assertEqual(meta["content_type"], "table")
        self.assertIn("table_header", meta)

    def test_report_can_be_written(self):
        report = {"ok": True, "child_record_count": len(self.child_records)}
        out_path = write_report(self.project_root, "component2_pinecone_upsert_test", report)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["child_record_count"], len(self.child_records))


if __name__ == "__main__":
    unittest.main()
