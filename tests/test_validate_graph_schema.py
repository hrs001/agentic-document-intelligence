import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.validate_graph_schema import (
    build_graph_schema_validation_artifact,
    validate_edge,
    validate_node,
    write_artifact,
)


class ValidateGraphSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_validate_node_keeps_high_value_type(self):
        node = {
            "node_id": "n1",
            "canonical_name": "Microsoft",
            "entity_type": "organization",
            "mention_count": 5,
            "evidence_snippets": ["Microsoft"],
        }
        is_valid, reasons = validate_node(node)
        self.assertTrue(is_valid)
        self.assertEqual(reasons, [])

    def test_validate_node_rejects_generic_other(self):
        node = {
            "node_id": "n2",
            "canonical_name": "full FY2025 Microsoft Form 10-K",
            "entity_type": "other",
            "mention_count": 1,
            "evidence_snippets": ["full FY2025 Microsoft Form 10-K"],
        }
        is_valid, reasons = validate_node(node)
        self.assertFalse(is_valid)
        self.assertIn("generic_other_type", reasons)

    def test_validate_edge_rejects_single_mention_supports(self):
        edge = {
            "edge_id": "e1",
            "source_node_id": "n1",
            "target_node_id": "n2",
            "relation_type": "supports",
            "evidence_snippets": ["supports"],
            "mention_count": 1,
        }
        is_valid, reasons = validate_edge(edge, {"n1", "n2"})
        self.assertFalse(is_valid)
        self.assertIn("weak_single_mention_relation", reasons)

    def test_build_validation_artifact_filters_nodes_and_edges(self):
        artifact = {
            "document_id": "doc1",
            "normalized_node_count": 2,
            "normalized_edge_count": 1,
            "nodes": [
                {
                    "node_id": "n1",
                    "canonical_name": "Microsoft",
                    "entity_type": "organization",
                    "mention_count": 3,
                    "evidence_snippets": ["Microsoft"],
                },
                {
                    "node_id": "n2",
                    "canonical_name": "full FY2025 Microsoft Form 10-K",
                    "entity_type": "other",
                    "mention_count": 1,
                    "evidence_snippets": ["full FY2025 Microsoft Form 10-K"],
                },
            ],
            "edges": [
                {
                    "edge_id": "e1",
                    "source_node_id": "n1",
                    "source_canonical_name": "Microsoft",
                    "relation_type": "supports",
                    "target_node_id": "n2",
                    "target_canonical_name": "full FY2025 Microsoft Form 10-K",
                    "evidence_snippets": ["supports"],
                    "mention_count": 1,
                }
            ],
        }
        validated = build_graph_schema_validation_artifact(artifact)
        self.assertEqual(validated["validated_node_count"], 1)
        self.assertEqual(validated["validated_edge_count"], 0)
        self.assertEqual(validated["rejected_node_count"], 1)

    def test_report_can_be_written(self):
        artifact = {
            "document_id": "doc1",
            "validated_node_count": 1,
            "validated_edge_count": 1,
            "rejected_node_count": 0,
            "rejected_edge_count": 0,
            "validated_nodes": [],
            "validated_edges": [],
            "rejected_nodes": [],
            "rejected_edges": [],
        }
        path = write_artifact(self.project_root, "component5_graph_schema_validation_test", artifact)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["document_id"], "doc1")


if __name__ == "__main__":
    unittest.main()
