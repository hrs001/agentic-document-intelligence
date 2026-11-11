import json
import unittest
from pathlib import Path

from scripts.extract_layout_aware_document import build_layout_artifact
from scripts.generate_chunk_records import build_chunk_artifact
from scripts.generate_embedding_ready_records import build_embedding_ready_artifact, write_artifact


class EmbeddingReadyRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        layout = build_layout_artifact(cls.project_root, "microsoft_fy2025_10k_summary")
        chunks = build_chunk_artifact(layout)
        cls.artifact = build_embedding_ready_artifact(chunks)
        cls.chunks = chunks

    def test_record_count_matches_chunk_count(self):
        self.assertEqual(self.artifact["record_count"], len(self.chunks["chunks"]))

    def test_record_fields_exist(self):
        first = self.artifact["records"][0]
        required = {"record_id", "source_chunk_id", "parent_id", "text", "content_type", "text_hash", "metadata"}
        self.assertTrue(required.issubset(first.keys()))

    def test_record_ids_are_unique(self):
        ids = [record["record_id"] for record in self.artifact["records"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_table_records_are_preserved(self):
        table_records = [record for record in self.artifact["records"] if record["content_type"] == "table"]
        self.assertGreater(len(table_records), 0)

    def test_write_artifact(self):
        out_path = write_artifact(self.project_root, "component2_embedding_ready_records_test", self.artifact)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["record_count"], self.artifact["record_count"])


if __name__ == "__main__":
    unittest.main()
