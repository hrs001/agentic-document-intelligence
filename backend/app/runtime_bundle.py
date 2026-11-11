from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


RUNTIME_BUCKET_ENV = "ADI_RUNTIME_BUNDLE_BUCKET"
RUNTIME_PREFIX_ENV = "ADI_RUNTIME_BUNDLE_PREFIX"
RUNTIME_LOCAL_ROOT_ENV = "ADI_RUNTIME_LOCAL_ROOT"
DEFAULT_RUNTIME_LOCAL_ROOT = "/tmp/adi_runtime"

APP_ROOT_RELATIVE = Path("agentic_document_intelligence")
BOOTSTRAP_RUNTIME_PATHS = [
    APP_ROOT_RELATIVE / "corpus" / "metadata" / "corpus_metadata.json",
    APP_ROOT_RELATIVE / "corpus" / "metadata" / "corpus_manifest.json",
]
STATIC_RUNTIME_PATHS = [
    APP_ROOT_RELATIVE
    / "artifacts"
    / "experiments"
    / "component6_sql_schema_packaging_live"
    / "sql_schema"
    / "sql_schema_package.json",
    APP_ROOT_RELATIVE
    / "artifacts"
    / "experiments"
    / "component6_sqlite_database_build_live"
    / "sql_db"
    / "microsoft_fy2025_analyst_demo.sqlite",
    APP_ROOT_RELATIVE
    / "artifacts"
    / "experiments"
    / "component5_kuzu_graph_build_live"
    / "kuzu_db"
    / "microsoft_fy2025_10k_summary.kuzu",
    APP_ROOT_RELATIVE
    / "artifacts"
    / "experiments"
    / "component5_graph_schema_validation_live"
    / "graph_validated"
    / "microsoft_fy2025_10k_summary_graph_validated.json",
    APP_ROOT_RELATIVE
    / "artifacts"
    / "experiments"
    / "component2_chunk_generation"
    / "chunks"
    / "microsoft_fy2025_10k_summary_chunks.json",
    APP_ROOT_RELATIVE
    / "artifacts"
    / "experiments"
    / "component2_embedding_ready_records"
    / "embeddings"
    / "microsoft_fy2025_10k_summary_embedding_records.json",
]

_HYDRATION_LOCK = threading.Lock()
_HYDRATED_ROOT: Path | None = None


def ensure_runtime_bundle_ready(code_project_root: Path) -> Path:
    global _HYDRATED_ROOT
    bucket = os.getenv(RUNTIME_BUCKET_ENV, "").strip()
    if not bucket:
        return code_project_root

    with _HYDRATION_LOCK:
        if _HYDRATED_ROOT is not None:
            return _HYDRATED_ROOT

        runtime_root = Path(os.getenv(RUNTIME_LOCAL_ROOT_ENV, DEFAULT_RUNTIME_LOCAL_ROOT)).resolve()
        runtime_root.mkdir(parents=True, exist_ok=True)
        _hydrate_runtime_bundle(runtime_root, bucket, os.getenv(RUNTIME_PREFIX_ENV, "").strip())
        _HYDRATED_ROOT = runtime_root
        return runtime_root


def _hydrate_runtime_bundle(runtime_root: Path, bucket: str, prefix: str) -> None:
    client = _build_s3_client()

    for relative_path in BOOTSTRAP_RUNTIME_PATHS:
        _download_relative_path(client, bucket, prefix, runtime_root, relative_path)

    manifest_path = runtime_root / APP_ROOT_RELATIVE / "corpus" / "metadata" / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required_paths = list(STATIC_RUNTIME_PATHS)
    for document in manifest.get("documents", []):
        required_paths.append(APP_ROOT_RELATIVE / Path(document["relative_path"]))
    for dataset in manifest.get("datasets", []):
        for table in dataset.get("tables", []):
            required_paths.append(APP_ROOT_RELATIVE / Path(table["relative_path"]))

    for relative_path in required_paths:
        _download_relative_path(client, bucket, prefix, runtime_root, relative_path)


def _build_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required for Lambda runtime bundle hydration.") from exc
    return boto3.client("s3")


def _download_relative_path(client: Any, bucket: str, prefix: str, runtime_root: Path, relative_path: Path) -> None:
    destination = runtime_root / relative_path
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    key = _build_s3_key(prefix, relative_path)
    client.download_file(bucket, key, str(destination))


def _build_s3_key(prefix: str, relative_path: Path) -> str:
    normalized_relative = str(relative_path).replace("\\", "/")
    normalized_prefix = prefix.strip("/")
    if normalized_prefix:
        return f"{normalized_prefix}/{normalized_relative}"
    return normalized_relative
