import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.sub_query_coverage_scoring import (
    compute_overlap_ratio,
    score_coverage,
    score_sub_query_coverage,
    write_report,
)


def make_match(source_chunk_id: str, best_score: float, match_count: int, section_title: str, child_text: str) -> dict:
    return {
        "id": f"{source_chunk_id}_record",
        "source_chunk_id": source_chunk_id,
        "best_score": best_score,
        "match_count": match_count,
        "metadata": {
            "source_chunk_id": source_chunk_id,
            "section_title": section_title,
            "page": 1,
            "page_end": 1,
            "parent_id": f"{source_chunk_id}_parent",
            "content_type": "text",
        },
        "child_text": child_text,
        "parent_text": child_text,
        "provenance_list": [],
        "variant_types": [],
        "query_angles": [],
        "matched_query_texts": [],
    }


class SubQueryCoverageScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_overlap_ratio_detects_shared_terms(self):
        matches = [
            make_match(
                "chunk_1",
                2.0,
                2,
                "Governance",
                "Satya Nadella is the chairman and CEO of Microsoft.",
            )
        ]
        overlap = compute_overlap_ratio("Who is the CEO of Microsoft?", matches)
        self.assertGreater(overlap, 0.5)

    def test_strong_coverage_marks_no_hyde(self):
        sub_query_result = {
            "sub_query_id": "sq_1",
            "original_sub_query": "Who is the CEO of Microsoft?",
            "merged_match_count": 4,
            "merged_matches": [
                make_match(
                    "chunk_1",
                    3.6,
                    4,
                    "Governance and executives",
                    "Satya Nadella is the Chairman and CEO of Microsoft.",
                ),
                make_match(
                    "chunk_2",
                    2.4,
                    2,
                    "Executive officers",
                    "The executive officers listed include Satya Nadella.",
                ),
            ],
        }
        scored = score_sub_query_coverage(sub_query_result)
        self.assertEqual(scored["coverage_label"], "strong")
        self.assertFalse(scored["should_consider_hyde"])

    def test_weak_coverage_marks_hyde_candidate(self):
        sub_query_result = {
            "sub_query_id": "sq_2",
            "original_sub_query": "What did the CEO say about AI?",
            "merged_match_count": 2,
            "merged_matches": [
                make_match(
                    "chunk_3",
                    0.7,
                    1,
                    "Financial performance",
                    "Revenue rose and operating margin remained strong.",
                ),
                make_match(
                    "chunk_4",
                    0.6,
                    1,
                    "Risk factors",
                    "Cybersecurity remains a key risk factor.",
                ),
            ],
        }
        scored = score_sub_query_coverage(sub_query_result)
        self.assertEqual(scored["coverage_label"], "weak")
        self.assertTrue(scored["should_consider_hyde"])

    def test_score_coverage_builds_summary(self):
        result = {
            "original_query": "query",
            "policy": {},
            "sub_query_results": [
                {
                    "sub_query_id": "sq_1",
                    "original_sub_query": "Who is the CEO of Microsoft?",
                    "merged_match_count": 2,
                    "merged_matches": [
                        make_match(
                            "chunk_1",
                            3.5,
                            2,
                            "Governance",
                            "Satya Nadella is CEO of Microsoft.",
                        )
                    ],
                },
                {
                    "sub_query_id": "sq_2",
                    "original_sub_query": "What did the CEO say about AI?",
                    "merged_match_count": 1,
                    "merged_matches": [
                        make_match(
                            "chunk_2",
                            0.4,
                            1,
                            "Finance",
                            "Revenue rose in FY2025.",
                        )
                    ],
                },
            ],
        }
        scored = score_coverage(result)
        self.assertEqual(scored["coverage_summary"]["sub_query_count"], 2)
        self.assertEqual(scored["coverage_summary"]["weak_sub_query_count"], 1)

    def test_write_report_persists_payload(self):
        scored = {
            "original_query": "query",
            "policy": {},
            "coverage_summary": {"sub_query_count": 1},
            "sub_query_scores": [],
        }
        path = write_report(self.project_root, "component4_sub_query_coverage_test", scored)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)


if __name__ == "__main__":
    unittest.main()
