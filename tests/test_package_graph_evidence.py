import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.package_graph_evidence import (
    assemble_graph_evidence_bundle,
    write_report,
)


class PackageGraphEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_assemble_graph_evidence_bundle_preserves_counts(self):
        report = {
            "query": "Which segment includes GitHub?",
            "database_path": "db",
            "matched_nodes": [
                {
                    "node_id": "n1",
                    "canonical_name": "GitHub",
                    "entity_type": "product_or_service",
                    "aliases": [],
                    "node_score": 5.0,
                    "mention_count": 2,
                    "evidence_snippets": ["GitHub"],
                    "source_graph_input_ids": ["g1"],
                    "source_parent_ids": ["p1"],
                    "source_child_ids": ["c1"],
                    "section_titles": ["Section 1"],
                    "page_ranges": [{"page_start": 1, "page_end": 1}],
                }
            ],
            "matched_edges": [
                {
                    "edge_id": "e1",
                    "source_node_id": "n2",
                    "source_canonical_name": "Intelligent Cloud",
                    "relation_type": "includes",
                    "target_node_id": "n1",
                    "target_canonical_name": "GitHub",
                    "edge_score": 4.0,
                    "mention_count": 2,
                    "evidence_snippets": ["Intelligent Cloud includes GitHub"],
                    "source_graph_input_ids": ["g1"],
                    "source_parent_ids": ["p1"],
                    "source_child_ids": ["c1"],
                    "section_titles": ["Section 1"],
                    "page_ranges": [{"page_start": 1, "page_end": 1}],
                }
            ],
        }
        bundle = assemble_graph_evidence_bundle(report)
        self.assertEqual(bundle["bundle_summary"]["matched_node_count"], 1)
        self.assertEqual(bundle["bundle_summary"]["matched_edge_count"], 1)
        self.assertIn("Intelligent Cloud -> includes -> GitHub", bundle["assembled_graph_evidence_text"])

    def test_report_can_be_written(self):
        result = {
            "query": "x",
            "database_path": "db",
            "bundle_summary": {"matched_node_count": 1, "matched_edge_count": 1},
            "matched_nodes": [],
            "matched_edges": [],
            "assembled_graph_evidence_text": "",
        }
        path = write_report(self.project_root, "component5_graph_evidence_packaging_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["query"], "x")


if __name__ == "__main__":
    unittest.main()
