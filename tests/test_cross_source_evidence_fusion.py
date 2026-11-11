import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.cross_source_evidence_fusion import (
    build_sub_query_fusions,
    detect_conflict_signals,
    detect_overlap_signals,
    fuse_cross_source_evidence,
    load_report,
    write_report,
)


def build_sample_orchestration_result() -> dict:
    return {
        "original_query": "Which segment includes GitHub and what was its FY2025 revenue?",
        "policy": {"routing_policy": "x"},
        "routing_summary": {"sub_query_count": 1},
        "execution_summary": {"sub_query_count": 1, "source_usage": {"graph_relationships": 1, "sql_structured": 1, "vector_document": 1}},
        "sub_query_results": [
            {
                "sub_query_id": "sq_1",
                "original_sub_query": "Which segment includes GitHub and what was its FY2025 revenue?",
                "resolved_sub_query": "Which segment includes GitHub and what was its FY2025 revenue?",
                "source_outputs": [
                    {
                        "source": "graph_relationships",
                        "evidence_bundle": {
                            "matched_nodes": [
                                {
                                    "node_id": "n1",
                                    "canonical_name": "GitHub",
                                    "entity_type": "product_or_service",
                                    "aliases": [],
                                    "node_score": 7.4,
                                    "mention_count": 2,
                                    "evidence_snippets": ["GitHub is included under Intelligent Cloud"],
                                    "source_graph_input_ids": ["g1"],
                                    "source_parent_ids": ["parent_1"],
                                    "source_child_ids": ["child_1"],
                                    "section_titles": ["Segment and product portfolio review"],
                                    "page_ranges": [{"page_start": 4, "page_end": 5}],
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
                                    "edge_score": 8.1,
                                    "mention_count": 2,
                                    "evidence_snippets": ["Intelligent Cloud includes GitHub"],
                                    "source_graph_input_ids": ["g1"],
                                    "source_parent_ids": ["parent_1"],
                                    "source_child_ids": ["child_1"],
                                    "section_titles": ["Segment and product portfolio review"],
                                    "page_ranges": [{"page_start": 4, "page_end": 5}],
                                }
                            ],
                            "assembled_graph_evidence_text": "Intelligent Cloud includes GitHub",
                        },
                    },
                    {
                        "source": "sql_structured",
                        "evidence_bundle": {
                            "target_tables": ["financial_performance_by_segment"],
                            "validated_sql": "SELECT segment_name, revenue_usd_millions FROM financial_performance_by_segment WHERE fiscal_year = 2025",
                            "confidence": "high",
                            "preview_rows": [
                                {"segment_name": "Intelligent Cloud", "revenue_usd_millions": 108610}
                            ],
                            "assembled_sql_evidence_text": "Intelligent Cloud 108610",
                        },
                    },
                    {
                        "source": "vector_document",
                        "evidence_bundle": {
                            "sub_query_bundles": [
                                {
                                    "sub_query_id": "sq_1",
                                    "evidence_items": [
                                        {
                                            "source_chunk_id": "chunk_1",
                                            "parent_id": "parent_1",
                                            "child_text": "Intelligent Cloud includes GitHub and other cloud services.",
                                            "parent_text": "Segment and product portfolio review describes Intelligent Cloud.",
                                            "citation": {
                                                "parent_id": "parent_1",
                                                "section_title": "Segment and product portfolio review",
                                                "page": 4,
                                                "page_end": 5,
                                            },
                                            "best_score": 1.9,
                                            "rerank_score": 0.92,
                                            "mmr_score": 0.81,
                                        }
                                    ],
                                    "evidence_count": 1,
                                }
                            ],
                            "assembled_evidence_text": "Intelligent Cloud includes GitHub and other cloud services.",
                        },
                    },
                ],
            }
        ],
    }


class CrossSourceEvidenceFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_fuse_cross_source_evidence_collects_mixed_source_facts(self):
        result = fuse_cross_source_evidence(build_sample_orchestration_result())
        self.assertEqual(result["bundle_summary"]["sub_query_count"], 1)
        self.assertGreaterEqual(result["bundle_summary"]["fact_count"], 4)
        self.assertIn("[Fused Sub-query]", result["assembled_fused_evidence_text"])
        self.assertIn("[SQL]", result["assembled_fused_evidence_text"])
        self.assertIn("[Graph]", result["assembled_fused_evidence_text"])
        self.assertIn("[Vector]", result["assembled_fused_evidence_text"])

    def test_detect_overlap_signals_finds_cross_source_entity_and_parent_overlap(self):
        result = fuse_cross_source_evidence(build_sample_orchestration_result())
        overlap_types = {item["overlap_type"] for item in result["overlap_signals"]}
        self.assertIn("entity", overlap_types)
        self.assertIn("parent_context", overlap_types)

    def test_detect_conflict_signals_flags_sql_value_mismatch(self):
        orchestration_result = build_sample_orchestration_result()
        orchestration_result["sub_query_results"][0]["source_outputs"].append(
            {
                "source": "sql_structured",
                "evidence_bundle": {
                    "target_tables": ["financial_performance_by_segment"],
                    "validated_sql": "SELECT segment_name, revenue_usd_millions FROM financial_performance_by_segment WHERE fiscal_year = 2025",
                    "confidence": "medium",
                    "preview_rows": [
                        {"segment_name": "Intelligent Cloud", "revenue_usd_millions": 999999}
                    ],
                    "assembled_sql_evidence_text": "Intelligent Cloud 999999",
                },
            }
        )
        result = fuse_cross_source_evidence(orchestration_result)
        conflict_types = {item["conflict_type"] for item in result["conflict_signals"]}
        self.assertIn("sql_value_mismatch", conflict_types)

    def test_detect_conflict_signals_flags_graph_relation_mismatch(self):
        orchestration_result = build_sample_orchestration_result()
        orchestration_result["sub_query_results"][0]["source_outputs"][0]["evidence_bundle"]["matched_edges"].append(
            {
                "edge_id": "e2",
                "source_node_id": "n2",
                "source_canonical_name": "Intelligent Cloud",
                "relation_type": "excludes",
                "target_node_id": "n1",
                "target_canonical_name": "GitHub",
                "edge_score": 4.1,
                "mention_count": 1,
                "evidence_snippets": ["Intelligent Cloud excludes GitHub"],
                "source_graph_input_ids": ["g2"],
                "source_parent_ids": ["parent_2"],
                "source_child_ids": ["child_2"],
                "section_titles": ["Segment and product portfolio review"],
                "page_ranges": [{"page_start": 6, "page_end": 6}],
            }
        )
        result = fuse_cross_source_evidence(orchestration_result)
        conflict_types = {item["conflict_type"] for item in result["conflict_signals"]}
        self.assertIn("graph_relation_mismatch", conflict_types)

    def test_build_sub_query_fusions_prioritizes_sql_then_graph_then_vector(self):
        fused = fuse_cross_source_evidence(build_sample_orchestration_result())
        sub_query = fused["sub_query_fusions"][0]
        lines = sub_query["assembled_fused_sub_query_text"].splitlines()
        sql_index = next(index for index, line in enumerate(lines) if line.startswith("[SQL]"))
        graph_index = next(index for index, line in enumerate(lines) if line.startswith("[Graph]"))
        vector_index = next(index for index, line in enumerate(lines) if line.startswith("[Vector]"))
        self.assertLess(sql_index, graph_index)
        self.assertLess(graph_index, vector_index)

    def test_write_and_load_report_round_trip(self):
        fused = fuse_cross_source_evidence(build_sample_orchestration_result())
        path = write_report(self.project_root.parent, "component8_cross_source_evidence_fusion_test", fused)
        self.assertTrue(path.exists())
        reloaded = load_report(path)
        self.assertEqual(reloaded["original_query"], fused["original_query"])
        self.assertEqual(reloaded["bundle_summary"]["fact_count"], fused["bundle_summary"]["fact_count"])


if __name__ == "__main__":
    unittest.main()
