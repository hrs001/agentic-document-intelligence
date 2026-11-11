import json
import unittest
from pathlib import Path

from scripts.extract_layout_aware_document import build_layout_artifact, write_artifact


class LayoutAwareExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.artifact = build_layout_artifact(cls.project_root, "microsoft_fy2025_10k_summary")

    def test_layout_artifact_is_non_empty(self):
        self.assertGreater(self.artifact["page_count"], 0)
        self.assertTrue(self.artifact["pages"])

    def test_pages_contain_items(self):
        self.assertTrue(any(page["items"] for page in self.artifact["pages"]))

    def test_table_extraction_detects_real_tables(self):
        table_pages = [page for page in self.artifact["pages"] if page["table_count"] > 0]
        self.assertGreater(len(table_pages), 0)
        first_table = next(item for item in table_pages[0]["items"] if item["type"] == "table")
        self.assertIn("|", first_table["markdown"])

    def test_text_blocks_are_present(self):
        text_pages = [page for page in self.artifact["pages"] if page["text_block_count"] > 0]
        self.assertGreater(len(text_pages), 0)

    def test_artifact_can_be_written(self):
        out_path = write_artifact(self.project_root, "component2_layout_aware_extraction_test", self.artifact)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["document_id"], self.artifact["document_id"])


if __name__ == "__main__":
    unittest.main()
