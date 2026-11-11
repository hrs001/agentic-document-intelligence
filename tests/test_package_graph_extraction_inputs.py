import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.package_graph_extraction_inputs import (
    build_extraction_text,
    build_graph_extraction_input_artifact,
    write_artifact,
)


class PackageGraphExtractionInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_build_extraction_text_prefixes_section_when_missing(self):
        result = build_extraction_text("Section A", "Some paragraph text.")
        self.assertTrue(result.startswith("Section A"))

    def test_build_extraction_text_does_not_duplicate_section_title(self):
        result = build_extraction_text("Section A", "Section A\n\nSome paragraph text.")
        self.assertEqual(result, "Section A\n\nSome paragraph text.")

    def test_build_graph_input_artifact_keeps_unique_text_parents(self):
        chunk_artifact = {
            "document_id": "doc1",
            "page_count": 2,
            "section_count": 1,
            "chunks": [
                {
                    "child_id": "p1_ch0",
                    "parent_id": "p1",
                    "child_text": "a",
                    "parent_text": "Section 1\n\nParent A",
                    "metadata": {
                        "document_id": "doc1",
                        "source_file": "f.pdf",
                        "page": 1,
                        "page_end": 1,
                        "section_title": "Section 1",
                    },
                },
                {
                    "child_id": "p1_ch1",
                    "parent_id": "p1",
                    "child_text": "b",
                    "parent_text": "Section 1\n\nParent A",
                    "metadata": {
                        "document_id": "doc1",
                        "source_file": "f.pdf",
                        "page": 1,
                        "page_end": 1,
                        "section_title": "Section 1",
                    },
                },
                {
                    "child_id": "table0_ch0",
                    "parent_id": "table0",
                    "child_text": "| x |",
                    "parent_text": "| x |",
                    "metadata": {
                        "document_id": "doc1",
                        "source_file": "f.pdf",
                        "page": 1,
                        "page_end": 1,
                        "section_title": "Section 1",
                        "content_type": "table",
                    },
                },
            ],
        }
        artifact = build_graph_extraction_input_artifact(chunk_artifact)
        self.assertEqual(artifact["graph_input_count"], 1)
        self.assertEqual(artifact["skipped_table_chunk_count"], 1)
        self.assertEqual(artifact["graph_inputs"][0]["source_child_count"], 2)

    def test_graph_input_preserves_child_linkage(self):
        chunk_artifact = {
            "document_id": "doc1",
            "page_count": 1,
            "section_count": 1,
            "chunks": [
                {
                    "child_id": "c1",
                    "parent_id": "p1",
                    "child_text": "a",
                    "parent_text": "Parent A",
                    "metadata": {
                        "document_id": "doc1",
                        "source_file": "f.pdf",
                        "page": 1,
                        "page_end": 1,
                        "section_title": "Section 1",
                    },
                }
            ],
        }
        artifact = build_graph_extraction_input_artifact(chunk_artifact)
        self.assertEqual(artifact["graph_inputs"][0]["source_child_ids"], ["c1"])

    def test_report_can_be_written(self):
        artifact = {
            "document_id": "doc1",
            "graph_input_count": 1,
            "skipped_table_chunk_count": 0,
            "graph_inputs": [],
        }
        path = write_artifact(self.project_root, "component5_graph_input_packaging_test", artifact)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["document_id"], "doc1")


if __name__ == "__main__":
    unittest.main()
