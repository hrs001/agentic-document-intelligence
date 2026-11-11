import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.benchmark_graph_entity_extraction import (
    estimate_cost_usd,
    evaluate_case,
    is_grounded,
    load_cases,
    normalize_extraction_payload,
    normalize_relation,
    rank_models,
    write_report,
)


class GraphEntityExtractionBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_load_cases_enriches_text(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 3)
        self.assertTrue(cases[0]["text"])

    def test_normalize_extraction_payload_dedupes(self):
        payload = {
            "entities": [
                {"name": "Microsoft", "entity_type": "organization", "evidence": "Microsoft"},
                {"name": "microsoft", "entity_type": "organization", "evidence": "Microsoft"},
            ],
            "relationships": [
                {
                    "source": "Microsoft",
                    "relation_type": "invests_in",
                    "target": "AI infrastructure",
                    "evidence": "AI infrastructure",
                },
                {
                    "source": "Microsoft",
                    "relation_type": "invests_in",
                    "target": "AI infrastructure",
                    "evidence": "AI infrastructure",
                },
            ],
        }
        normalized = normalize_extraction_payload(payload)
        self.assertEqual(len(normalized["entities"]), 1)
        self.assertEqual(len(normalized["relationships"]), 1)

    def test_grounding_check_uses_passage_text(self):
        passage = "Microsoft integrated Activision Blizzard in gaming."
        self.assertTrue(is_grounded(passage, "integrated Activision Blizzard"))
        self.assertFalse(is_grounded(passage, "Google Cloud"))

    def test_evaluate_case_scores_expected_matches(self):
        case = {
            "case_id": "x",
            "section_title": "Test",
            "page_start": 1,
            "page_end": 1,
            "text": "Microsoft integrated Activision Blizzard and invested in AI infrastructure.",
            "expected_entities": ["Microsoft", "Activision Blizzard", "AI infrastructure"],
            "expected_relationships": [
                {
                    "source": "Microsoft",
                    "relation_type": "integrates",
                    "target": "Activision Blizzard",
                }
            ],
        }
        actual = {
            "entities": [
                {"name": "Microsoft", "entity_type": "organization", "evidence": "Microsoft"},
                {
                    "name": "Activision Blizzard",
                    "entity_type": "organization",
                    "evidence": "Activision Blizzard",
                },
            ],
            "relationships": [
                {
                    "source": "Microsoft",
                    "relation_type": "integrates",
                    "target": "Activision Blizzard",
                    "evidence": "integrated Activision Blizzard",
                }
            ],
        }
        evaluation = evaluate_case(case, actual)
        self.assertEqual(evaluation["entity_recall"], 0.6667)
        self.assertEqual(evaluation["relation_recall"], 1.0)

    def test_cost_estimation_returns_positive_value(self):
        self.assertGreater(estimate_cost_usd("gpt-5-mini", 1000, 200), 0)

    def test_relation_normalization_is_case_insensitive(self):
        self.assertEqual(
            normalize_relation("Microsoft", "INCLUDES", "GitHub"),
            normalize_relation("microsoft", "includes", "github"),
        )

    def test_report_can_be_written(self):
        model_results = [
            {
                "model": "gpt-5-mini",
                "summary": {
                    "avg_quality_score": 0.5,
                    "avg_relation_recall": 0.4,
                    "estimated_cost_usd": 0.01,
                },
                "case_results": [],
            }
        ]
        path = write_report(self.project_root, "component5_graph_entity_benchmark_test", model_results)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["results"][0]["model"], "gpt-5-mini")

    def test_rank_models_prefers_quality_then_cost(self):
        ranked = rank_models(
            [
                {
                    "model": "a",
                    "summary": {"avg_quality_score": 0.6, "avg_relation_recall": 0.4, "estimated_cost_usd": 0.02},
                },
                {
                    "model": "b",
                    "summary": {"avg_quality_score": 0.6, "avg_relation_recall": 0.5, "estimated_cost_usd": 0.03},
                },
            ]
        )
        self.assertEqual(ranked[0]["model"], "b")


if __name__ == "__main__":
    unittest.main()
