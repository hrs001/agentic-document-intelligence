import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pinecone import Pinecone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pinecone_hybrid_retrieval import (
    build_chunk_indexes,
    dedupe_matches,
    embed_dense_query,
    embed_sparse_query,
    hydrate_matches,
    load_chunk_artifact,
    run_hybrid_query,
    scale_dense,
    scale_sparse,
)
from openai import OpenAI


RERANK_MODEL = "pinecone-rerank-v0"


def build_rerank_documents(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def apply_rerank(matches: List[Dict[str, Any]], rerank_result: Any) -> List[Dict[str, Any]]:
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


def write_report(project_root: Path, run_id: str, report: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "retrieval"
        / "pinecone_rerank_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pinecone reranking on top of hybrid retrieval candidates.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--chunk-input",
        default="artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json",
    )
    parser.add_argument("--run-id", default="component2_pinecone_rerank")
    parser.add_argument("--namespace", default="microsoft_fy2025_fixed_corpus")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--query", default="What were Microsoft's FY2025 revenue, operating income, and net income?")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    load_dotenv(project_root / ".env")

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    chunk_artifact = load_chunk_artifact(project_root / args.chunk_input)
    child_index, parent_index = build_chunk_indexes(chunk_artifact)
    openai_client = OpenAI(api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    dense = embed_dense_query(openai_client, args.query)
    sparse = embed_sparse_query(pc, args.query)
    raw_matches = run_hybrid_query(index, args.namespace, scale_dense(dense, args.alpha), scale_sparse(sparse["indices"], sparse["values"], args.alpha), args.top_k)
    deduped = dedupe_matches(raw_matches)
    hydrated = hydrate_matches(deduped, child_index, parent_index)
    rerank_docs = build_rerank_documents(hydrated)
    rerank_result = pc.inference.rerank(
        model=RERANK_MODEL,
        query=args.query,
        documents=rerank_docs,
        top_n=args.top_n,
        return_documents=True,
        parameters={"truncate": "END"},
    )
    reranked = apply_rerank(hydrated, rerank_result)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "namespace": args.namespace,
        "index_name": index_name,
        "query": args.query,
        "top_k": args.top_k,
        "top_n": args.top_n,
        "alpha": args.alpha,
        "rerank_model": RERANK_MODEL,
        "candidate_count": len(hydrated),
        "reranked_count": len(reranked),
        "matches": reranked,
    }
    out_path = write_report(project_root, args.run_id, report)
    print(json.dumps({"ok": True, "report_path": str(out_path), "candidate_count": len(hydrated), "reranked_count": len(reranked)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
