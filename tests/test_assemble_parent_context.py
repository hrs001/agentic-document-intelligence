import json
import unittest
from pathlib import Path

from scripts.assemble_parent_context import assemble_parent_context, write_artifact


class ParentContextAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.report = {
            "query": "What were Microsoft's FY2025 revenue, operating income, and net income?",
            "matches": [
                {
                    "id": "a",
                    "score": 1.0,
                    "rerank_score": 0.9,
                    "metadata": {
                        "parent_id": "parent_1",
                        "source_chunk_id": "chunk_1",
                        "section_title": "Section A",
                        "page": 3,
                        "page_end": 4,
                        "content_type": "text",
                    },
                    "child_text": "Revenue rose strongly.",
                    "parent_text": "Full parent text A.",
                },
                {
                    "id": "b",
                    "score": 0.9,
                    "rerank_score": 0.8,
                    "metadata": {
                        "parent_id": "parent_1",
                        "source_chunk_id": "chunk_2",
                        "section_title": "Section A",
                        "page": 3,
                        "page_end": 4,
                        "content_type": "table",
                    },
                    "child_text": "Table with totals.",
                    "parent_text": "Full parent text A.",
                },
                {
                    "id": "c",
                    "score": 0.7,
                    "rerank_score": 0.6,
                    "metadata": {
                        "parent_id": "parent_2",
                        "source_chunk_id": "chunk_3",
                        "section_title": "Section B",
                        "page": 8,
                        "page_end": 9,
                        "content_type": "text",
                    },
                    "child_text": "Operating income details.",
                    "parent_text": "Full parent text B.",
                },
            ],
        }
        cls.artifact = assemble_parent_context(cls.report)

    def test_parent_contexts_are_grouped(self):
        self.assertEqual(self.artifact["parent_context_count"], 2)
        self.assertEqual(self.artifact["contexts"][0]["parent_id"], "parent_1")

    def test_child_evidence_is_limited_and_preserved(self):
        ctx = next(c for c in self.artifact["contexts"] if c["parent_id"] == "parent_1")
        self.assertEqual(len(ctx["children"]), 2)
        self.assertIn("table", ctx["content_types"])

    def test_assembled_text_is_non_empty(self):
        self.assertIn("[Context 1]", self.artifact["assembled_context_text"])
        self.assertIn("Parent Context:", self.artifact["assembled_context_text"])

    def test_report_can_be_written(self):
        out_path = write_artifact(self.project_root, "component2_parent_context_test", self.artifact)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["parent_context_count"], self.artifact["parent_context_count"])


if __name__ == "__main__":
    unittest.main()
