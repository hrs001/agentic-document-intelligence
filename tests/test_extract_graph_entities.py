import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.extract_graph_entities import (
    build_graph_extraction_artifact,
    enrich_entities,
    enrich_relationships,
    write_artifact,
)


def fake_extractor(passage_text: str, section_title: str, model: str) -> dict:
    return {
        "entities": [
            {"name": "Microsoft", "entity_type": "organization", "evidence": "Microsoft"},
            {"name": "GitHub", "entity_type": "product_or_service", "evidence": "GitHub"},
        ],
        "relationships": [
            {
                "source": "Intelligent Cloud",
                "relation_type": "includes",
                "target": "GitHub",
                "evidence": "GitHub",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "estimated_cost_usd": 0.001,
        },
    }


class ExtractGraphEntitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.graph_input = {
            "graph_input_id": "g1",
            "source_parent_id": "p1",
            "source_child_ids": ["c1", "c2"],
            "section_title": "Section 1",
            "page_start": 1,
            "page_end": 2,
        }

    def test_enrich_entities_preserves_provenance(self):
        entities = enrich_entities(
            self.graph_input,
            [{"name": "Microsoft", "entity_type": "organization", "evidence": "Microsoft"}],
        )
        self.assertEqual(entities[0]["source_parent_id"], "p1")
        self.assertEqual(entities[0]["source_child_ids"], ["c1", "c2"])

    def test_enrich_relationships_preserves_provenance(self):
        relationships = enrich_relationships(
            self.graph_input,
            [
                {
                    "source": "Section 1",
                    "relation_type": "includes",
                    "target": "GitHub",
                    "evidence": "GitHub",
                }
            ],
        )
        self.assertEqual(relationships[0]["section_title"], "Section 1")

    def test_build_graph_extraction_artifact_aggregates_counts(self):
        graph_input_artifact = {
            "document_id": "doc1",
            "graph_input_count": 2,
            "graph_inputs": [
                {
                    "graph_input_id": "g1",
                    "source_parent_id": "p1",
                    "source_child_ids": ["c1"],
                    "section_title": "Section 1",
                    "page_start": 1,
                    "page_end": 1,
                    "extraction_text": "Text 1",
                },
                {
                    "graph_input_id": "g2",
                    "source_parent_id": "p2",
                    "source_child_ids": ["c2"],
                    "section_title": "Section 2",
                    "page_start": 2,
                    "page_end": 2,
                    "extraction_text": "Text 2",
                },
            ],
        }
        artifact = build_graph_extraction_artifact(graph_input_artifact, extractor=fake_extractor, model="gpt-5.4-mini")
        self.assertEqual(artifact["record_count"], 2)
        self.assertEqual(artifact["entity_count"], 4)
        self.assertEqual(artifact["relationship_count"], 2)
        self.assertEqual(artifact["usage"]["prompt_tokens"], 20)

    def test_report_can_be_written(self):
        artifact = {
            "document_id": "doc1",
            "record_count": 1,
            "entity_count": 2,
            "relationship_count": 1,
            "usage": {"estimated_cost_usd": 0.001},
            "records": [],
        }
        path = write_artifact(self.project_root, "component5_graph_entity_extraction_test", artifact)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["document_id"], "doc1")


if __name__ == "__main__":
    unittest.main()
