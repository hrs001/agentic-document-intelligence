import json
import unittest
from pathlib import Path

from scripts.validate_corpus import validate


class CorpusValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.report = validate(cls.project_root)
        with (cls.project_root / "corpus" / "metadata" / "corpus_manifest.json").open("r", encoding="utf-8") as f:
            cls.manifest = json.load(f)
        with (cls.project_root / "corpus" / "metadata" / "dataset_schema.json").open("r", encoding="utf-8") as f:
            cls.dataset_schema = json.load(f)

    def test_validation_report_is_ok(self):
        self.assertTrue(self.report["ok"], msg=f"Blocking issues: {self.report['blocking_issues']}")

    def test_manifest_declares_single_v1_document(self):
        self.assertEqual(len(self.manifest["documents"]), 1)
        self.assertEqual(self.manifest["documents"][0]["document_id"], "microsoft_fy2025_10k_summary")

    def test_manifest_declares_sql_dataset(self):
        datasets = self.manifest["datasets"]
        self.assertEqual(len(datasets), 1)
        self.assertTrue(datasets[0]["synthetic"])
        self.assertEqual(len(datasets[0]["tables"]), 3)

    def test_dataset_schema_contains_expected_tables(self):
        names = [table["table_name"] for table in self.dataset_schema["tables"]]
        self.assertEqual(
            names,
            [
                "financial_performance_by_segment",
                "geographic_revenue_mix",
                "product_family_signals",
            ],
        )

    def test_every_check_passes(self):
        failed = [check for check in self.report["checks"] if check["status"] != "pass"]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
