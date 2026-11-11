import json
import unittest
from pathlib import Path

from scripts.extract_layout_aware_document import build_layout_artifact
from scripts.generate_chunk_records import build_chunk_artifact
from scripts.build_sparse_index import build_sparse_index, tokenize, write_artifact


class SparseIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        layout = build_layout_artifact(cls.project_root, "microsoft_fy2025_10k_summary")
        chunks = build_chunk_artifact(layout)
        cls.index_artifact = build_sparse_index(chunks)
        cls.chunks = chunks

    def test_tokenizer_is_non_empty(self):
        tokens = tokenize("Revenue grew 15% in FY2025.")
        self.assertIn("revenue", tokens)
        self.assertIn("fy2025.", tokens)

    def test_document_count_matches_chunks(self):
        self.assertEqual(self.index_artifact["document_count"], len(self.chunks["chunks"]))

    def test_index_contains_vocabulary(self):
        self.assertGreater(self.index_artifact["vocabulary_size"], 0)
        self.assertIn("revenue", self.index_artifact["idf"])

    def test_documents_preserve_parent_linkage(self):
        first = self.index_artifact["documents"][0]
        self.assertIn("doc_id", first)
        self.assertIn("parent_id", first)
        self.assertIn("metadata", first)

    def test_table_documents_are_preserved(self):
        self.assertTrue(any(doc["content_type"] == "table" for doc in self.index_artifact["documents"]))

    def test_artifact_can_be_written(self):
        out_path = write_artifact(self.project_root, "component2_sparse_index_test", self.index_artifact)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["document_count"], self.index_artifact["document_count"])


if __name__ == "__main__":
    unittest.main()
