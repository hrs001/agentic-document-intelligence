import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RERANK_MODEL = "pinecone-rerank-v0"
DEFAULT_INPUT = (
    "artifacts/experiments/component4_corrective_hyde_retry_live/retrieval/"
    "corrective_hyde_retry_report.json"
)
DEFAULT_TOP_N = 6


def load_retry_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"]


def build_rerank_documents(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for match in matches:
        metadata = match.get("metadata", {})
        parent_preview = (match.get("parent_text") or "")[:800]
        text = "\n".join(
            part
            for part in [
                f"Section: {metadata.get('section_title', '')}",
                f"Page: {metadata.get('page', '')}",
                f"Content Type: {metadata.get('content_type', 'text')}",
                f"Child Text: {match.get('child_text', '')}",
                f"Parent Context: {parent_preview}",
            ]
            if part
        )
        docs.append({"id": match["id"], "text": text})
    return docs


def apply_rerank(matches: list[dict[str, Any]], rerank_result: Any) -> list[dict[str, Any]]:
    by_id = {match["id"]: match for match in matches}
    reranked = []
    for item in rerank_result.data:
        document = getattr(item, "document", None)
        doc_id = document.get("id") if isinstance(document, dict) else getattr(document, "id", None)
        index = getattr(item, "index", None)
        source = None
        if doc_id and doc_id in by_id:
            source = by_id[doc_id]
        elif index is not None and index < len(matches):
            source = matches[index]
        if source is None:
            continue
        enriched = dict(source)
        enriched["rerank_score"] = float(item.score)
        reranked.append(enriched)
    return reranked


def rerank_sub_query_results(result: dict[str, Any], pc: Pinecone, top_n: int) -> dict[str, Any]:
    reranked_sub_queries = []
    for sub_query_result in result["updated_sub_query_results"]:
        matches = sub_query_result["merged_matches"]
        docs = build_rerank_documents(matches)
        rerank_result = pc.inference.rerank(
            model=RERANK_MODEL,
            query=sub_query_result["original_sub_query"],
            documents=docs,
            top_n=min(top_n, len(docs)),
            return_documents=True,
            parameters={"truncate": "END"},
        )
        reranked_matches = apply_rerank(matches, rerank_result)
        reranked_sub_queries.append(
            {
                "sub_query_id": sub_query_result["sub_query_id"],
                "original_sub_query": sub_query_result["original_sub_query"],
                "candidate_count": len(matches),
                "reranked_count": len(reranked_matches),
                "reranked_matches": reranked_matches,
            }
        )

    return {
        "original_query": result["original_query"],
        "policy": {
            **result["policy"],
            "sub_query_rerank_model": RERANK_MODEL,
        },
        "rerank_summary": {
            "sub_query_count": len(reranked_sub_queries),
            "top_n": top_n,
        },
        "sub_query_results": reranked_sub_queries,
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "retrieval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rerank_sub_query_candidates_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank merged sub-query candidates against each original sub-query.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--run-id", default="component4_sub_query_rerank")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env")
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    result = load_retry_report(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    pc = Pinecone(api_key=pinecone_api_key)
    reranked = rerank_sub_query_results(result, pc, args.top_n)
    report_path = write_report(PROJECT_ROOT, args.run_id, reranked)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "sub_query_count": reranked["rerank_summary"]["sub_query_count"],
                "top_n": reranked["rerank_summary"]["top_n"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
