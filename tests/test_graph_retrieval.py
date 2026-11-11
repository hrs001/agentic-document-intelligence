import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_graph_retrieval import evaluate_case
from agentic_document_intelligence.scripts.graph_retrieval import (
    normalize_text,
    score_edge,
    score_node,
    tokenize,
    write_report,
)


class GraphRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_tokenize_removes_stopwords(self):
        self.assertEqual(tokenize("Which segment includes GitHub?"), ["segment", "github"])

    def test_score_node_prefers_phrase_match(self):
        node = {
            "canonical_name": "GitHub",
            "aliases": [],
            "evidence_snippets": ["GitHub deepens developer ecosystem control"],
            "mention_count": 2,
        }
        higher = score_node("Which segment includes GitHub?", node)
        lower = score_node("What risk does Microsoft face?", node)
        self.assertGreater(higher, lower)

    def test_score_edge_uses_query_overlap(self):
        edge = {
            "source_canonical_name": "Intelligent Cloud",
            "target_canonical_name": "GitHub",
            "relation_type": "includes",
            "evidence_snippets": ["Intelligent Cloud includes GitHub"],
            "source_node_id": "n1",
            "target_node_id": "n2",
            "mention_count": 2,
        }
        score = score_edge("Which segment includes GitHub?", edge, {"n1"})
        self.assertGreater(score, 0)

    def test_evaluate_case_passes_expected_hits(self):
        case = {
            "case_id": "x",
            "query": "Which segment includes GitHub?",
            "expected_node_names": ["GitHub", "Intelligent Cloud"],
            "expected_relations": [{"source": "Intelligent Cloud", "relation_type": "includes", "target": "GitHub"}],
        }
        result = {
            "matched_nodes": [
                {"canonical_name": "GitHub"},
                {"canonical_name": "Intelligent Cloud"},
            ],
            "matched_edges": [
                {
                    "source_canonical_name": "Intelligent Cloud",
                    "relation_type": "includes",
                    "target_canonical_name": "GitHub",
                }
            ],
        }
        evaluation = evaluate_case(result, case)
        self.assertTrue(evaluation["passed"])

    def test_report_can_be_written(self):
        report = {"query": "x", "matched_nodes": [], "matched_edges": []}
        path = write_report(self.project_root, "component5_graph_retrieval_test", report)
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["query"], "x")


if __name__ == "__main__":
    unittest.main()
