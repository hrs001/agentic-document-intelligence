import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.benchmark_answer_generation_models import evaluate_answer, phrase_present
from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.tests.test_cross_source_evidence_fusion import build_sample_orchestration_result


class BenchmarkAnswerGenerationModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fused_bundle = fuse_cross_source_evidence(build_sample_orchestration_result())

    def test_evaluate_answer_requires_phrases_sources_and_inline_citations(self):
        answer_result = {
            "answer_markdown": "GitHub belongs to Intelligent Cloud [graph_edge::e1]. FY2025 revenue was 108610 [sql::sq_1::1].",
            "used_fact_ids": ["graph_edge::e1", "sql::sq_1::1"],
            "unanswered_sub_queries": [],
        }
        case = {
            "required_phrases": ["GitHub", "Intelligent Cloud", "108610"],
            "required_source_types": ["graph_relationships", "sql_structured"],
            "min_citation_count": 2,
        }
        evaluation = evaluate_answer(answer_result, self.fused_bundle, case)
        self.assertTrue(evaluation["passed"])

    def test_phrase_present_normalizes_numeric_formatting(self):
        self.assertTrue(phrase_present("Revenue was $108,610 million.", "108610"))
        self.assertTrue(phrase_present("Operating margin was 50.6%.", "50.6"))

    def test_evaluate_answer_fails_when_required_source_missing(self):
        answer_result = {
            "answer_markdown": "GitHub belongs to Intelligent Cloud [graph_edge::e1].",
            "used_fact_ids": ["graph_edge::e1"],
            "unanswered_sub_queries": [],
        }
        case = {
            "required_phrases": ["GitHub", "Intelligent Cloud"],
            "required_source_types": ["graph_relationships", "sql_structured"],
            "min_citation_count": 1,
        }
        evaluation = evaluate_answer(answer_result, self.fused_bundle, case)
        self.assertFalse(evaluation["passed"])
        self.assertIn("sql_structured", evaluation["missing_source_types"])


if __name__ == "__main__":
    unittest.main()
