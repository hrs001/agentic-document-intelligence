import json
import unittest
from pathlib import Path

from scripts.extract_layout_aware_document import build_layout_artifact
from scripts.generate_chunk_records import (
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_SIZE,
    build_chunk_artifact,
    write_artifact,
)


class ChunkGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        layout = build_layout_artifact(cls.project_root, "microsoft_fy2025_10k_summary")
        cls.chunk_artifact = build_chunk_artifact(layout)

    def test_chunk_artifact_is_non_empty(self):
        self.assertGreater(self.chunk_artifact["parent_chunk_count"], 0)
        self.assertGreater(self.chunk_artifact["child_chunk_count"], 0)
        self.assertTrue(self.chunk_artifact["chunks"])

    def test_chunk_ids_and_parent_ids_exist(self):
        first = self.chunk_artifact["chunks"][0]
        self.assertTrue(first["child_id"])
        self.assertTrue(first["parent_id"])

    def test_chunk_metadata_is_complete(self):
        first = self.chunk_artifact["chunks"][0]
        required = {
            "document_id",
            "source_file",
            "page",
            "page_end",
            "section_title",
            "section_index",
            "parent_index_on_page",
            "child_index_in_parent",
        }
        self.assertTrue(required.issubset(first["metadata"].keys()))

    def test_child_text_is_bounded(self):
        text_chunks = [chunk for chunk in self.chunk_artifact["chunks"] if chunk["metadata"].get("content_type") != "table"]
        longest = max(len(chunk["child_text"]) for chunk in text_chunks)
        self.assertLessEqual(longest, CHILD_CHUNK_SIZE + 60)

    def test_parent_text_is_bounded(self):
        text_chunks = [chunk for chunk in self.chunk_artifact["chunks"] if chunk["metadata"].get("content_type") != "table"]
        longest = max(len(chunk["parent_text"]) for chunk in text_chunks)
        self.assertLessEqual(longest, PARENT_CHUNK_SIZE + 250)

    def test_each_child_belongs_to_parent_text(self):
        text_chunks = [chunk for chunk in self.chunk_artifact["chunks"] if chunk["metadata"].get("content_type") != "table"]
        for chunk in text_chunks[:50]:
            self.assertIn(chunk["child_text"][:50].strip(), chunk["parent_text"])

    def test_structure_aware_sections_exist(self):
        self.assertGreater(self.chunk_artifact["section_count"], 5)
        titles = [section["section_title"] for section in self.chunk_artifact["sections"]]
        self.assertIn("1. Executive summary", titles)
        self.assertIn("4. Fiscal 2025 financial performance", titles)

    def test_chunk_section_titles_are_not_generic_for_main_content(self):
        titled_chunks = [chunk for chunk in self.chunk_artifact["chunks"] if chunk["metadata"]["page"] >= 3]
        self.assertTrue(any(chunk["metadata"]["section_title"] == "1. Executive summary" for chunk in titled_chunks))

    def test_table_chunks_exist(self):
        table_chunks = [chunk for chunk in self.chunk_artifact["chunks"] if chunk["metadata"].get("content_type") == "table"]
        self.assertGreater(len(table_chunks), 0)
        self.assertTrue(any("| Metric | FY2025 | FY2024 | Change |" in chunk["child_text"] for chunk in table_chunks))

    def test_artifact_can_be_written(self):
        out_path = write_artifact(self.project_root, "component2_chunk_generation_test", self.chunk_artifact)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["document_id"], self.chunk_artifact["document_id"])


if __name__ == "__main__":
    unittest.main()
