import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = (
    "artifacts/experiments/component4_mmr_diversification_live/retrieval/"
    "mmr_diversification_report.json"
)


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"]


def build_citation(match: dict[str, Any]) -> dict[str, Any]:
    metadata = match["metadata"]
    return {
        "source_chunk_id": match["source_chunk_id"],
        "parent_id": metadata.get("parent_id", ""),
        "document_id": metadata.get("document_id", ""),
        "source_file": metadata.get("source_file", ""),
        "section_title": metadata.get("section_title", ""),
        "page": metadata.get("page"),
        "page_end": metadata.get("page_end"),
        "content_type": metadata.get("content_type", "text"),
    }


def build_context_block(index: int, sub_query: str, match: dict[str, Any]) -> str:
    citation = build_citation(match)
    return "\n".join(
        [
            f"[Evidence {index}] Sub-query: {sub_query}",
            f"Section: {citation['section_title']}",
            f"Pages: {citation['page']} - {citation['page_end']}",
            f"Parent ID: {citation['parent_id']}",
            f"Content Type: {citation['content_type']}",
            f"Rerank Score: {match.get('rerank_score', 0.0)}",
            f"MMR Score: {match.get('mmr_score', 0.0)}",
            "Child Evidence:",
            match.get("child_text", ""),
            "Parent Context:",
            match.get("parent_text", ""),
        ]
    )


def assemble_final_evidence_bundle(result: dict[str, Any]) -> dict[str, Any]:
    sub_query_bundles = []
    all_context_blocks: list[str] = []
    total_evidence_items = 0

    for sub_query_result in result["sub_query_results"]:
        evidence_items = []
        for index, match in enumerate(sub_query_result["diversified_matches"], start=1):
            citation = build_citation(match)
            evidence_item = {
                "source_chunk_id": match["source_chunk_id"],
                "parent_id": citation["parent_id"],
                "child_text": match.get("child_text", ""),
                "parent_text": match.get("parent_text", ""),
                "citation": citation,
                "best_score": match.get("best_score", 0.0),
                "rerank_score": match.get("rerank_score", 0.0),
                "mmr_score": match.get("mmr_score", 0.0),
                "match_count": match.get("match_count", 0),
                "variant_types": match.get("variant_types", []),
                "query_angles": match.get("query_angles", []),
                "matched_query_texts": match.get("matched_query_texts", []),
                "provenance_list": match.get("provenance_list", []),
            }
            evidence_items.append(evidence_item)
            all_context_blocks.append(
                build_context_block(index, sub_query_result["original_sub_query"], match)
            )

        total_evidence_items += len(evidence_items)
        sub_query_bundles.append(
            {
                "sub_query_id": sub_query_result["sub_query_id"],
                "original_sub_query": sub_query_result["original_sub_query"],
                "evidence_count": len(evidence_items),
                "evidence_items": evidence_items,
            }
        )

    return {
        "original_query": result["original_query"],
        "policy": result["policy"],
        "bundle_summary": {
            "sub_query_count": len(sub_query_bundles),
            "total_evidence_items": total_evidence_items,
        },
        "sub_query_bundles": sub_query_bundles,
        "assembled_evidence_text": "\n\n---\n\n".join(all_context_blocks),
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answer_context"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "final_evidence_bundle.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the final evidence bundle for answer generation.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default="component4_final_evidence_bundle")
    args = parser.parse_args()

    result = load_report(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    bundle = assemble_final_evidence_bundle(result)
    report_path = write_report(PROJECT_ROOT, args.run_id, bundle)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "sub_query_count": bundle["bundle_summary"]["sub_query_count"],
                "total_evidence_items": bundle["bundle_summary"]["total_evidence_items"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
