import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.normalize_graph_extraction import (
    build_normalized_graph_artifact,
    clean_name,
    normalize_key,
    write_artifact,
)


class NormalizeGraphExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_clean_name_removes_trailing_acronym_alias(self):
        self.assertEqual(clean_name("Productivity and Business Processes (PBP)"), "Productivity and Business Processes")

    def test_normalize_key_is_case_insensitive(self):
        self.assertEqual(normalize_key("GitHub"), normalize_key("github"))

    def test_build_normalized_graph_artifact_merges_aliases(self):
        extraction_artifact = {
            "document_id": "doc1",
            "record_count": 1,
            "entity_count": 2,
            "relationship_count": 1,
            "graph_extraction_method": {"model": "gpt-5.4-mini"},
            "records": [
                {
                    "entities": [
                        {
                            "name": "Productivity and Business Processes",
                            "entity_type": "segment",
                            "evidence": "Productivity and Business Processes",
                            "source_graph_input_id": "g1",
                            "source_parent_id": "p1",
                            "source_child_ids": ["c1"],
                            "section_title": "Section",
                            "page_start": 1,
                            "page_end": 1,
                        },
                        {
                            "name": "Productivity and Business Processes (PBP)",
                            "entity_type": "segment",
                            "evidence": "Productivity and Business Processes (PBP)",
                            "source_graph_input_id": "g1",
                            "source_parent_id": "p1",
                            "source_child_ids": ["c1"],
                            "section_title": "Section",
                            "page_start": 1,
                            "page_end": 1,
                        },
                    ],
                    "relationships": [
                        {
                            "source": "Productivity and Business Processes (PBP)",
                            "relation_type": "includes",
                            "target": "GitHub",
                            "evidence": "includes GitHub",
                            "source_graph_input_id": "g1",
                            "source_parent_id": "p1",
                            "source_child_ids": ["c1"],
                            "section_title": "Section",
                            "page_start": 1,
                            "page_end": 1,
                        }
                    ],
                }
            ],
        }
        # add target node in separate record so edge survives
        extraction_artifact["records"][0]["entities"].append(
            {
                "name": "GitHub",
                "entity_type": "product_or_service",
                "evidence": "GitHub",
                "source_graph_input_id": "g1",
                "source_parent_id": "p1",
                "source_child_ids": ["c1"],
                "section_title": "Section",
                "page_start": 1,
                "page_end": 1,
            }
        )
        artifact = build_normalized_graph_artifact(extraction_artifact)
        self.assertEqual(artifact["normalized_node_count"], 2)
        node = next(item for item in artifact["nodes"] if item["canonical_name"] == "Productivity and Business Processes")
        self.assertIn("Productivity and Business Processes (PBP)", node["aliases"])
        self.assertEqual(artifact["normalized_edge_count"], 1)

    def test_report_can_be_written(self):
        artifact = {
            "document_id": "doc1",
            "raw_entity_count": 2,
            "raw_relationship_count": 1,
            "normalized_node_count": 2,
            "normalized_edge_count": 1,
            "nodes": [],
            "edges": [],
        }
        path = write_artifact(self.project_root, "component5_graph_normalization_test", artifact)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["document_id"], "doc1")


if __name__ == "__main__":
    unittest.main()
