import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.scripts.ragas_style_llm_judge import (
    build_judge_input,
    sanitize_judge_payload,
    write_report,
)
from agentic_document_intelligence.tests.test_cross_source_evidence_fusion import build_sample_orchestration_result


class RagasStyleLlmJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.fused_bundle = fuse_cross_source_evidence(build_sample_orchestration_result())

    def test_build_judge_input_collects_used_fact_summaries(self):
        answer_result = {
            "answer_markdown": "GitHub is in Intelligent Cloud [graph_edge::e1]. Revenue was $108,610 million [sql::sq_1::1].",
            "used_fact_ids": ["graph_edge::e1", "sql::sq_1::1"],
        }
        judge_input = build_judge_input(self.fused_bundle, answer_result)
        self.assertEqual(judge_input["original_query"], self.fused_bundle["original_query"])
        self.assertEqual(len(judge_input["used_facts"]), 2)

    def test_sanitize_judge_payload_normalizes_metrics(self):
        payload = sanitize_judge_payload(
            {
                "faithfulness": 5,
                "answer_relevancy": 4,
                "context_precision": 4,
                "citation_grounding": 5,
                "overall_verdict": "pass",
                "summary": "Strong answer.",
                "strengths": ["Grounded."],
                "weaknesses": ["None."],
                "recommendations": ["Keep."],
            }
        )
        self.assertEqual(payload["faithfulness"], 5)
        self.assertEqual(payload["overall_verdict"], "pass")

    def test_report_can_be_written(self):
        result = {
            "original_query": "x",
            "judge_model": "gpt-5-mini",
            "metrics": {"overall_verdict": "pass"},
            "average_score": 4.5,
        }
        path = write_report(self.project_root.parent, "component9_ragas_style_llm_judge_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["judge_model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
