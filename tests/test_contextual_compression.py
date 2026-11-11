import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.backend.app.service import build_ui_citations_and_answer
from agentic_document_intelligence.scripts.contextual_compression import (
    attach_contextual_compression,
    build_compression_input,
    sanitize_compression_payload,
    should_apply_contextual_compression,
)
from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.scripts.generate_grounded_answer import build_answer_input
from agentic_document_intelligence.tests.test_cross_source_evidence_fusion import build_sample_orchestration_result


class _FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 1000, "completion_tokens": 200}


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeChatCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeChatCompletions(content)


class FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


class ContextualCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fused_bundle = fuse_cross_source_evidence(build_sample_orchestration_result())

    def test_should_apply_contextual_compression_for_multi_subquery_bundle(self):
        self.assertFalse(should_apply_contextual_compression(self.fused_bundle))
        synthetic_bundle = {
            "bundle_summary": {"fact_count": 35, "sub_query_count": 3},
            "normalized_facts": [{"source_type": "vector_document"} for _ in range(9)],
        }
        self.assertTrue(should_apply_contextual_compression(synthetic_bundle))

    def test_build_compression_input_collects_allowed_fact_ids(self):
        compression_input = build_compression_input(self.fused_bundle, max_facts_per_sub_query=2)
        self.assertEqual(len(compression_input["sub_queries"]), 1)
        self.assertGreaterEqual(len(compression_input["allowed_fact_ids"]), 2)

    def test_sanitize_compression_payload_filters_unknown_fact_ids(self):
        compression_input = build_compression_input(self.fused_bundle, max_facts_per_sub_query=2)
        payload = {
            "sub_queries": [
                {
                    "sub_query_id": "sq_1",
                    "compressed_units": [
                        {
                            "unit_id": "sq_1_u1",
                            "summary_text": "Revenue fact",
                            "supported_fact_ids": ["sql::sq_1::1", "bad_fact"],
                            "compression_type": "query_focused_summary",
                        }
                    ],
                }
            ],
            "confidence": "high",
            "compression_notes": ["ok"],
        }
        sanitized = sanitize_compression_payload(payload, compression_input)
        self.assertEqual(sanitized["sub_queries"][0]["compressed_units"][0]["supported_fact_ids"], ["sql::sq_1::1"])

    def test_attach_contextual_compression_preserves_original_citation_resolution(self):
        client = FakeClient(
            json.dumps(
                {
                    "sub_queries": [
                        {
                            "sub_query_id": "sq_1",
                            "compressed_units": [
                                {
                                    "unit_id": "sq_1_u1",
                                    "summary_text": "Intelligent Cloud revenue was $108,610 million.",
                                    "supported_fact_ids": ["sql::sq_1::1"],
                                    "compression_type": "query_focused_summary",
                                }
                            ],
                        },
                        {
                            "sub_query_id": "sq_2",
                            "compressed_units": [
                                {
                                    "unit_id": "sq_2_u1",
                                    "summary_text": "AI demand supported capital intensity.",
                                    "supported_fact_ids": ["vector::chunk_1"],
                                    "compression_type": "query_focused_summary",
                                }
                            ],
                        },
                    ],
                    "confidence": "high",
                    "compression_notes": ["compression ok"],
                }
            )
        )
        compressed_bundle = attach_contextual_compression(
            self.fused_bundle,
            client=client,
            min_facts_for_compression=1,
            min_sub_queries_for_compression=1,
            min_vector_facts_for_compression=1,
        )
        self.assertTrue(compressed_bundle["compression_context"]["applied"])
        answer_result = {
            "answer_markdown": "Revenue was $108,610 million.[sql::sq_1::1] AI demand was strong.[vector::chunk_1]"
        }
        answer_text, citations = build_ui_citations_and_answer(compressed_bundle, answer_result)
        self.assertIn("[Evidence 1]", answer_text)
        self.assertIn("[Evidence 2]", answer_text)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["source_type"], "database")
        self.assertEqual(citations[1]["source_type"], "internal_document")

    def test_build_answer_input_prefers_compressed_units_when_present(self):
        client = FakeClient(
            json.dumps(
                {
                    "sub_queries": [
                        {
                            "sub_query_id": "sq_1",
                            "compressed_units": [
                                {
                                    "unit_id": "sq_1_u1",
                                    "summary_text": "Revenue fact.",
                                    "supported_fact_ids": ["sql::sq_1::1"],
                                    "compression_type": "query_focused_summary",
                                }
                            ],
                        },
                        {
                            "sub_query_id": "sq_2",
                            "compressed_units": [
                                {
                                    "unit_id": "sq_2_u1",
                                    "summary_text": "Narrative fact.",
                                    "supported_fact_ids": ["vector::chunk_1"],
                                    "compression_type": "query_focused_summary",
                                }
                            ],
                        },
                    ],
                    "confidence": "high",
                    "compression_notes": ["compression ok"],
                }
            )
        )
        compressed_bundle = attach_contextual_compression(
            self.fused_bundle,
            client=client,
            min_facts_for_compression=1,
            min_sub_queries_for_compression=1,
            min_vector_facts_for_compression=1,
        )
        answer_input = build_answer_input(compressed_bundle)
        self.assertIn("compressed_units", answer_input["sub_queries"][0])
        self.assertTrue(answer_input["compression_context"]["applied"])
        self.assertGreaterEqual(len(answer_input["sub_queries"][0]["compressed_units"]), 1)


if __name__ == "__main__":
    unittest.main()
