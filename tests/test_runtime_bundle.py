import os
import unittest
from pathlib import Path

from agentic_document_intelligence.backend.app.runtime_bundle import (
    _build_s3_key,
    ensure_runtime_bundle_ready,
)


class RuntimeBundleTest(unittest.TestCase):
    def test_build_s3_key_with_prefix(self):
        key = _build_s3_key("runtime-bundle/demo", Path("agentic_document_intelligence/corpus/metadata/corpus_manifest.json"))
        self.assertEqual(
            key,
            "runtime-bundle/demo/agentic_document_intelligence/corpus/metadata/corpus_manifest.json",
        )

    def test_build_s3_key_without_prefix(self):
        key = _build_s3_key("", Path("agentic_document_intelligence/corpus/metadata/corpus_manifest.json"))
        self.assertEqual(
            key,
            "agentic_document_intelligence/corpus/metadata/corpus_manifest.json",
        )

    def test_returns_code_root_when_bucket_not_configured(self):
        code_root = Path.cwd()
        prior_bucket = os.environ.pop("ADI_RUNTIME_BUNDLE_BUCKET", None)
        try:
            self.assertEqual(ensure_runtime_bundle_ready(code_root), code_root)
        finally:
            if prior_bucket is not None:
                os.environ["ADI_RUNTIME_BUNDLE_BUCKET"] = prior_bucket


if __name__ == "__main__":
    unittest.main()
