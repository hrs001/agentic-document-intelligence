import json
import unittest
from pathlib import Path

from scripts.extract_document_text import build_document_text_artifact, write_artifact


class DocumentTextExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.artifact = build_document_text_artifact(cls.project_root, "microsoft_fy2025_10k_summary")

    def test_artifact_is_non_empty(self):
        self.assertGreater(self.artifact["page_count"], 0)
        self.assertGreater(self.artifact["character_count"], 0)
        self.assertTrue(self.artifact["text"].strip())

    def test_text_contains_page_markers(self):
        self.assertIn("[PAGE_MARKER_1]", self.artifact["text"])
        self.assertIn("page_number", self.artifact["pages"][0])

    def test_pages_shape_is_complete(self):
        first_page = self.artifact["pages"][0]
        self.assertEqual(
            sorted(first_page.keys()),
            ["character_count", "marker", "page_number", "text"],
        )

    def test_contract_required_fields_exist(self):
        required = {
            "document_id",
            "text",
            "pages",
            "extraction_method",
            "generated_at",
        }
        self.assertTrue(required.issubset(self.artifact.keys()))

    def test_artifact_can_be_written(self):
        out_path = write_artifact(self.project_root, "component2_document_text_extraction_test", self.artifact)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["document_id"], self.artifact["document_id"])


if __name__ == "__main__":
    unittest.main()
