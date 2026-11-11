import json
import unittest
from pathlib import Path

from scripts.clean_document_text import build_cleaned_artifact, normalize_text, write_artifact
from scripts.extract_document_text import build_document_text_artifact


class DocumentCleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.raw_artifact = build_document_text_artifact(cls.project_root, "microsoft_fy2025_10k_summary")
        cls.cleaned_artifact = build_cleaned_artifact(cls.raw_artifact)

    def test_normalize_text_fixes_common_pdf_artifacts(self):
        raw = "Page 3\nAâ€™B ï‚· item\n\n\nmore"
        cleaned = normalize_text(raw)
        self.assertNotIn("â€™", cleaned)
        self.assertNotIn("ï‚·", cleaned)
        self.assertIn("A'B", cleaned)
        self.assertIn("- item", cleaned)

    def test_cleaned_artifact_is_non_empty(self):
        self.assertGreater(self.cleaned_artifact["page_count"], 0)
        self.assertTrue(self.cleaned_artifact["text"].strip())

    def test_page_markers_are_preserved(self):
        self.assertIn("[PAGE_MARKER_1]", self.cleaned_artifact["text"])

    def test_page_labels_are_removed_from_cleaned_text(self):
        first_page_text = self.cleaned_artifact["pages"][0]["text"]
        self.assertNotIn("Page 1\n", first_page_text)

    def test_artifact_can_be_written(self):
        out_path = write_artifact(self.project_root, "component2_document_text_cleaning_test", self.cleaned_artifact)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["document_id"], self.cleaned_artifact["document_id"])


if __name__ == "__main__":
    unittest.main()
