import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_parallel_safe_orchestration import (
    build_single_subquery_bundle,
    evaluate_parallel_safe_case,
    write_report,
)


class EvaluateParallelSafeOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_build_single_subquery_bundle_creates_atomic_bundle(self):
        bundle = build_single_subquery_bundle(
            "Which segment includes GitHub, and what was FY2025 revenue mix by geography?"
        )
        self.assertEqual(bundle["bundle_summary"]["sub_query_count"], 1)
        self.assertEqual(bundle["sub_query_bundles"][0]["sub_query_id"], "sq_1")
        self.assertTrue(bundle["policy"]["forced_single_subquery_eval"])

    def test_evaluate_parallel_safe_case_requires_parallel_safe_activation(self):
        result = {
            "sub_query_results": [
                {
                    "execution_policy": {
                        "parallel_safe": False,
                        "active_sources": ["graph_relationships", "sql_structured"],
                    },
                    "source_outputs": [
                        {
                            "source": "graph_relationships",
                            "evidence_bundle": {"assembled_graph_evidence_text": "GitHub Intelligent Cloud"},
                        },
                        {
                            "source": "sql_structured",
                            "evidence_bundle": {"assembled_sql_evidence_text": "United States Other Countries"},
                        },
                    ],
                }
            ]
        }
        case = {
            "required_sources": ["graph_relationships", "sql_structured"],
            "expect_parallel_safe": True,
            "keyword_expectations": [
                {"source": "graph_relationships", "keywords": ["GitHub", "Intelligent Cloud"]},
                {"source": "sql_structured", "keywords": ["United States", "Other Countries"]},
            ],
        }
        evaluation = evaluate_parallel_safe_case(result, case)
        self.assertFalse(evaluation["passed"])
        self.assertTrue(evaluation["parallel_safe_mismatch"])

    def test_evaluate_parallel_safe_case_passes_when_sources_keywords_and_policy_match(self):
        result = {
            "sub_query_results": [
                {
                    "execution_policy": {
                        "parallel_safe": True,
                        "active_sources": ["graph_relationships", "sql_structured"],
                    },
                    "source_outputs": [
                        {
                            "source": "graph_relationships",
                            "evidence_bundle": {"assembled_graph_evidence_text": "Azure GitHub"},
                        },
                        {
                            "source": "sql_structured",
                            "evidence_bundle": {
                                "assembled_sql_evidence_text": "Productivity and Business Processes 50.6"
                            },
                        },
                    ],
                }
            ]
        }
        case = {
            "required_sources": ["graph_relationships", "sql_structured"],
            "expect_parallel_safe": True,
            "keyword_expectations": [
                {"source": "graph_relationships", "keywords": ["Azure", "GitHub"]},
                {"source": "sql_structured", "keywords": ["Productivity and Business Processes", "50.6"]},
            ],
        }
        evaluation = evaluate_parallel_safe_case(result, case)
        self.assertTrue(evaluation["passed"])
        self.assertFalse(evaluation["parallel_safe_mismatch"])

    def test_report_can_be_written(self):
        path = write_report(
            self.project_root.parent,
            "component7_parallel_safe_eval_test",
            {"case_count": 1, "passed_count": 1, "parallel_safe_count": 1, "case_results": []},
        )
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["passed_count"], 1)


if __name__ == "__main__":
    unittest.main()
