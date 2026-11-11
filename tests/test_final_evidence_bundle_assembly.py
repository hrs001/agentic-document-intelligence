import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.final_evidence_bundle_assembly import (
    assemble_final_evidence_bundle,
    build_citation,
    write_report,
)


def make_match(source_chunk_id: str, parent_id: str) -> dict:
    return {
        "source_chunk_id": source_chunk_id,
        "best_score": 1.2,
        "rerank_score": 0.8,
        "mmr_score": 0.7,
        "match_count": 2,
        "metadata": {
            "document_id": "doc_1",
            "source_file": "file.pdf",
            "section_title": "Section A",
            "page": 3,
            "page_end": 4,
            "parent_id": parent_id,
            "content_type": "text",
        },
        "child_text": "child text",
        "parent_text": "parent text",
        "variant_types": ["multi_query"],
        "query_angles": ["direct"],
        "matched_query_texts": ["query"],
        "provenance_list": [],
    }


class FinalEvidenceBundleAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_build_citation_extracts_expected_fields(self):
        match = make_match("chunk_1", "parent_1")
        citation = build_citation(match)
        self.assertEqual(citation["source_chunk_id"], "chunk_1")
        self.assertEqual(citation["parent_id"], "parent_1")
        self.assertEqual(citation["section_title"], "Section A")

    def test_assemble_final_evidence_bundle_builds_sub_query_bundle(self):
        result = {
            "original_query": "query",
            "policy": {},
            "sub_query_results": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "Who is the CEO?",
                    "diversified_matches": [
                        make_match("chunk_1", "parent_1"),
                        make_match("chunk_2", "parent_2"),
                    ],
                }
            ],
        }
        bundle = assemble_final_evidence_bundle(result)
        self.assertEqual(bundle["bundle_summary"]["sub_query_count"], 1)
        self.assertEqual(bundle["bundle_summary"]["total_evidence_items"], 2)
        self.assertIn("assembled_evidence_text", bundle)

    def test_write_report_persists_payload(self):
        bundle = {
            "original_query": "query",
            "policy": {},
            "bundle_summary": {"sub_query_count": 1, "total_evidence_items": 1},
            "sub_query_bundles": [],
            "assembled_evidence_text": "",
        }
        path = write_report(self.project_root, "component4_final_evidence_bundle_test", bundle)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
