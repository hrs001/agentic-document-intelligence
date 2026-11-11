import json
import unittest
from pathlib import Path

from scripts.extract_layout_aware_document import build_layout_artifact
from scripts.generate_chunk_records import build_chunk_artifact
from scripts.generate_embedding_ready_records import build_embedding_ready_artifact
from scripts.upsert_pinecone_sparse_vectors import build_hybrid_vectors, filter_child_records, write_report


class _MockExistingVector:
    def __init__(self, values, metadata):
        self.values = values
        self.metadata = metadata


class PineconeSparseVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        layout = build_layout_artifact(cls.project_root, "microsoft_fy2025_10k_summary")
        chunks = build_chunk_artifact(layout)
        ready = build_embedding_ready_artifact(chunks)
        cls.child_records = filter_child_records(ready)

    def test_child_filter_keeps_child_records(self):
        self.assertTrue(all("_ch" in record["source_chunk_id"] for record in self.child_records))

    def test_build_hybrid_vectors_merges_dense_and_sparse(self):
        sample_records = self.child_records[:2]
        fetched = {
            sample_records[0]["record_id"]: _MockExistingVector([0.1] * 3, {"parent_id": sample_records[0]["parent_id"]}),
            sample_records[1]["record_id"]: _MockExistingVector([0.2] * 3, {"parent_id": sample_records[1]["parent_id"]}),
        }
        sparse = [
            {"sparse_indices": [1, 5], "sparse_values": [0.9, 1.1]},
            {"sparse_indices": [2], "sparse_values": [0.7]},
        ]
        vectors = build_hybrid_vectors(sample_records, fetched, sparse)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0]["sparse_values"]["indices"], [1, 5])
        self.assertEqual(vectors[0]["metadata"]["parent_id"], sample_records[0]["parent_id"])

    def test_report_can_be_written(self):
        report = {"ok": True, "child_record_count": len(self.child_records)}
        out_path = write_report(self.project_root, "component2_pinecone_sparse_upsert_test", report)
        self.assertTrue(out_path.exists())
        with out_path.open("r", encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["child_record_count"], len(self.child_records))


if __name__ == "__main__":
    unittest.main()
