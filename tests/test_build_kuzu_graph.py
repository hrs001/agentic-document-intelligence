import json
import sys
import unittest
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.build_kuzu_graph import (
    build_kuzu_graph_database,
    write_report,
)


class BuildKuzuGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_build_kuzu_graph_database_loads_nodes_and_edges(self):
        artifact = {
            "document_id": "doc1",
            "validated_node_count": 2,
            "validated_edge_count": 1,
            "validated_nodes": [
                {
                    "node_id": "n1",
                    "canonical_name": "Microsoft",
                    "entity_type": "organization",
                    "aliases": [],
                    "evidence_snippets": ["Microsoft"],
                    "source_graph_input_ids": ["g1"],
                    "source_parent_ids": ["p1"],
                    "source_child_ids": ["c1"],
                    "section_titles": ["Section 1"],
                    "page_ranges": [{"page_start": 1, "page_end": 1}],
                    "mention_count": 2,
                },
                {
                    "node_id": "n2",
                    "canonical_name": "GitHub",
                    "entity_type": "product_or_service",
                    "aliases": [],
                    "evidence_snippets": ["GitHub"],
                    "source_graph_input_ids": ["g1"],
                    "source_parent_ids": ["p1"],
                    "source_child_ids": ["c1"],
                    "section_titles": ["Section 1"],
                    "page_ranges": [{"page_start": 1, "page_end": 1}],
                    "mention_count": 2,
                },
            ],
            "validated_edges": [
                {
                    "edge_id": "e1",
                    "source_node_id": "n1",
                    "source_canonical_name": "Microsoft",
                    "relation_type": "includes",
                    "target_node_id": "n2",
                    "target_canonical_name": "GitHub",
                    "evidence_snippets": ["GitHub"],
                    "source_graph_input_ids": ["g1"],
                    "source_parent_ids": ["p1"],
                    "source_child_ids": ["c1"],
                    "section_titles": ["Section 1"],
                    "page_ranges": [{"page_start": 1, "page_end": 1}],
                    "mention_count": 2,
                }
            ],
        }
        database_path = (
            self.project_root
            / "artifacts"
            / "experiments"
            / "component5_kuzu_graph_build_test_db"
            / "kuzu_db"
            / "doc1.kuzu"
        )
        if database_path.parent.exists():
            shutil.rmtree(database_path.parent)
        report = build_kuzu_graph_database(artifact, database_path)
        self.assertEqual(report["stored_node_count"], 2)
        self.assertEqual(report["stored_edge_count"], 1)

    def test_report_can_be_written(self):
        report = {
            "document_id": "doc1",
            "database_path": "x",
            "stored_node_count": 2,
            "stored_edge_count": 1,
            "distinct_relation_type_count": 1,
        }
        path = write_report(self.project_root, "component5_kuzu_graph_build_test", report)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["document_id"], "doc1")


if __name__ == "__main__":
    unittest.main()
