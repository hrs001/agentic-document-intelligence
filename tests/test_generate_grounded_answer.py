import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.generate_grounded_answer import (
    build_answer_input,
    sanitize_answer_payload,
    write_report,
)
from agentic_document_intelligence.tests.test_cross_source_evidence_fusion import build_sample_orchestration_result
from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence


class GenerateGroundedAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.fused_bundle = fuse_cross_source_evidence(build_sample_orchestration_result())

    def test_build_answer_input_limits_and_collects_fact_ids(self):
        answer_input = build_answer_input(self.fused_bundle, max_facts_per_sub_query=2)
        self.assertEqual(answer_input["sub_queries"][0]["sub_query_id"], "sq_1")
        self.assertEqual(len(answer_input["sub_queries"][0]["facts"]), 2)
        self.assertEqual(len(answer_input["allowed_fact_ids"]), 2)

    def test_sanitize_answer_payload_filters_unknown_fact_ids(self):
        fact_catalog = build_answer_input(self.fused_bundle)
        allowed_ids = set(fact_catalog["allowed_fact_ids"])
        source_map = {item["fact_id"]: item["source_type"] for item in fact_catalog["fact_catalog"]}
        payload = sanitize_answer_payload(
            {
                "answer_markdown": "Answer [sql::sq_1::1] [bad_fact]",
                "used_fact_ids": ["sql::sq_1::1", "bad_fact"],
                "citations": [
                    {"fact_id": "sql::sq_1::1", "source_type": "sql_structured", "reason": "numeric support"},
                    {"fact_id": "bad_fact", "source_type": "sql_structured", "reason": "bad"},
                ],
                "unanswered_sub_queries": [],
                "confidence": "high",
            },
            allowed_ids,
            source_map,
        )
        self.assertEqual(payload["used_fact_ids"], ["sql::sq_1::1"])
        self.assertEqual(len(payload["citations"]), 1)

    def test_report_can_be_written(self):
        result = {
            "original_query": "x",
            "model": "gpt-5-mini",
            "answer_markdown": "x [sql::sq_1::1]",
            "used_fact_ids": ["sql::sq_1::1"],
            "citations": [],
            "unanswered_sub_queries": [],
            "confidence": "medium",
        }
        path = write_report(self.project_root.parent, "component8_grounded_answer_generation_test", result)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"]["model"], "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
