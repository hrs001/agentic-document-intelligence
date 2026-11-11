import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_METADATA_KEYS = {"schema_version", "corpus_id", "provenance", "document_metadata", "dataset_metadata"}
REQUIRED_CONTRACT_KEYS = {"schema_version", "corpus_id", "artifact_layout", "expected_outputs"}
REQUIRED_EXPECTED_OUTPUT_KEYS = {"artifact_name", "relative_path_pattern", "required_fields"}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition: bool, check_id: str, description: str, details: str = "") -> Dict[str, Any]:
    return {
        "check_id": check_id,
        "description": description,
        "status": "pass" if condition else "fail",
        "details": details,
    }


def validate(project_root: Path) -> Dict[str, Any]:
    manifest_path = project_root / "corpus" / "metadata" / "corpus_manifest.json"
    metadata_path = project_root / "corpus" / "metadata" / "corpus_metadata.json"
    schema_path = project_root / "corpus" / "metadata" / "dataset_schema.json"
    contract_path = project_root / "corpus" / "contracts" / "preprocessing_contract.json"

    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    blocking_issues: List[str] = []

    manifest = _load_json(manifest_path)
    metadata = _load_json(metadata_path)
    schema = _load_json(schema_path)
    contract = _load_json(contract_path)

    checks.append(_check(isinstance(manifest, dict), "manifest_parse", "Manifest parses as JSON object."))
    checks.append(_check(isinstance(metadata, dict), "metadata_parse", "Metadata parses as JSON object."))
    checks.append(_check(isinstance(schema, dict), "dataset_schema_parse", "Dataset schema parses as JSON object."))
    checks.append(_check(isinstance(contract, dict), "contract_parse", "Preprocessing contract parses as JSON object."))

    checks.append(
        _check(
            REQUIRED_METADATA_KEYS.issubset(metadata.keys()),
            "metadata_required_keys",
            "Metadata includes all required top-level keys.",
            f"required={sorted(REQUIRED_METADATA_KEYS)}",
        )
    )
    checks.append(
        _check(
            REQUIRED_CONTRACT_KEYS.issubset(contract.keys()),
            "contract_required_keys",
            "Preprocessing contract includes all required top-level keys.",
            f"required={sorted(REQUIRED_CONTRACT_KEYS)}",
        )
    )

    doc_checks = []
    for doc in manifest.get("documents", []):
        doc_path = project_root / doc["relative_path"]
        doc_exists = doc_path.exists()
        doc_checks.append(doc_exists)
        checks.append(
            _check(
                doc_exists,
                f"document_exists::{doc['document_id']}",
                f"Document exists: {doc['document_id']}",
                str(doc_path),
            )
        )
        if doc_exists:
            actual_hash = _sha256(doc_path)
            expected_hash = str(doc.get("sha256", "")).lower()
            checks.append(
                _check(
                    actual_hash == expected_hash,
                    f"document_hash::{doc['document_id']}",
                    f"Document hash matches manifest: {doc['document_id']}",
                    f"expected={expected_hash} actual={actual_hash}",
                )
            )

    dataset_tables = {table["table_name"]: table for table in schema.get("tables", [])}
    all_dataset_paths_exist = True
    all_columns_match = True

    for dataset in manifest.get("datasets", []):
        for table in dataset.get("tables", []):
            table_name = table["table_name"]
            csv_path = project_root / table["relative_path"]
            exists = csv_path.exists()
            all_dataset_paths_exist &= exists
            checks.append(
                _check(
                    exists,
                    f"dataset_exists::{table_name}",
                    f"Dataset table exists: {table_name}",
                    str(csv_path),
                )
            )
            if not exists:
                continue

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                actual_columns = reader.fieldnames or []
                first_row = next(reader, None)
            expected_columns = [col["name"] for col in dataset_tables.get(table_name, {}).get("columns", [])]
            columns_match = actual_columns == expected_columns
            all_columns_match &= columns_match
            checks.append(
                _check(
                    columns_match,
                    f"dataset_columns::{table_name}",
                    f"Dataset columns match schema: {table_name}",
                    f"expected={expected_columns} actual={actual_columns}",
                )
            )
            checks.append(
                _check(
                    first_row is not None,
                    f"dataset_non_empty::{table_name}",
                    f"Dataset has at least one data row: {table_name}",
                )
            )

    capability_flags = manifest.get("capability_flags", {})
    has_doc = any(doc.get("capabilities", {}).get("document_rag", False) for doc in manifest.get("documents", []))
    has_graph = any(doc.get("capabilities", {}).get("graph_rag", False) for doc in manifest.get("documents", []))
    has_sql = any(ds.get("capabilities", {}).get("text_to_sql", False) for ds in manifest.get("datasets", []))
    checks.append(
        _check(
            capability_flags == {
                "document_rag": has_doc,
                "graph_rag": has_graph,
                "text_to_sql": has_sql,
            },
            "capability_consistency",
            "Manifest capability flags match declared assets.",
            f"computed=document:{has_doc},graph:{has_graph},sql:{has_sql}",
        )
    )

    expected_outputs = contract.get("expected_outputs", [])
    contract_complete = all(REQUIRED_EXPECTED_OUTPUT_KEYS.issubset(item.keys()) for item in expected_outputs)
    checks.append(
        _check(
            contract_complete and len(expected_outputs) >= 6,
            "contract_expected_outputs",
            "Preprocessing contract defines the required downstream artifact interfaces.",
            f"count={len(expected_outputs)}",
        )
    )

    if metadata.get("provenance", {}).get("document_source_kind") != "source_derived":
        warnings.append("Document provenance is not marked source_derived.")
    if metadata.get("provenance", {}).get("dataset_source_kind") != "synthetic":
        warnings.append("Dataset provenance is not marked synthetic.")

    for item in checks:
        if item["status"] == "fail":
            blocking_issues.append(item["check_id"])

    ok = len(blocking_issues) == 0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "component": "component_1_corpus_foundation",
        "ok": ok,
        "checks": checks,
        "warnings": warnings,
        "blocking_issues": blocking_issues,
        "summary": {
          "documents_declared": len(manifest.get("documents", [])),
          "datasets_declared": len(manifest.get("datasets", [])),
          "tables_declared": len(dataset_tables),
          "checks_run": len(checks)
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Component 1 corpus package.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument("--write-report", action="store_true", help="Write validation report to artifacts/validation.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    report = validate(project_root)

    print(json.dumps(report, indent=2))

    if args.write_report:
        out_path = project_root / "artifacts" / "validation" / "component_1_corpus_validation_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
