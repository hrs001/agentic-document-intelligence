import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import os


GOLD_QUERIES = [
    {
        "query": "Who is the CEO of Microsoft?",
        "expected_terms": ["satya nadella", "chief executive officer"],
        "expected_sections": ["Appendix B. Executive officers and selected reference data", "9. Operations, people, properties, and governance"],
    },
    {
        "query": "What were Microsoft's FY2025 revenue, operating income, and net income?",
        "expected_terms": ["$281,724", "$128,528", "$101,832"],
        "expected_sections": ["4. Fiscal 2025 financial performance", "1. Executive summary"],
    },
    {
        "query": "How fast did Azure and other cloud services grow in FY2025?",
        "expected_terms": ["34%", "azure"],
        "expected_sections": ["5. Segment results and operating drivers", "1. Executive summary"],
    },
    {
        "query": "What does the filing say about cybersecurity risk and governance?",
        "expected_terms": ["cybersecurity", "risk", "governance"],
        "expected_sections": ["8. Risk factors and cybersecurity posture", "9. Operations, people, properties, and governance"],
    },
]


def evaluate_matches(matches: List[Dict[str, Any]], expected_terms: List[str], expected_sections: List[str]) -> Dict[str, Any]:
    texts = [((match.get("child_text") or "") + "\n" + (match.get("parent_text") or "")).lower() for match in matches]
    sections = [str(match.get("metadata", {}).get("section_title", "")) for match in matches]
    matched_terms = sorted({term for term in expected_terms if any(term.lower() in text for text in texts)})
    matched_sections = sorted({section for section in expected_sections if section in sections})
    duplicate_ids = len({match["metadata"].get("source_chunk_id", match["id"]) for match in matches}) != len(matches)
    return {
        "matched_terms": matched_terms,
        "matched_sections": matched_sections,
        "duplicate_results": duplicate_ids,
        "passed": bool(matched_terms or matched_sections) and not duplicate_ids,
    }


def write_report(project_root: Path, run_id: str, report: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "retrieval"
        / "pinecone_hybrid_eval_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a gold-query evaluation over native Pinecone hybrid retrieval.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--chunk-input",
        default="artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json",
    )
    parser.add_argument("--run-id", default="component2_pinecone_hybrid_eval")
    parser.add_argument("--namespace", default="microsoft_fy2025_fixed_corpus")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.6)
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

    cases = []
    for case in GOLD_QUERIES:
        dense = embed_dense_query(openai_client, case["query"])
        sparse = embed_sparse_query(pc, case["query"])
        raw_matches = run_hybrid_query(index, args.namespace, scale_dense(dense, args.alpha), scale_sparse(sparse["indices"], sparse["values"], args.alpha), args.top_k)
        deduped = dedupe_matches(raw_matches)
        hydrated = hydrate_matches(deduped, child_index, parent_index)
        evaluation = evaluate_matches(hydrated, case["expected_terms"], case["expected_sections"])
        cases.append(
            {
                "query": case["query"],
                "expected_terms": case["expected_terms"],
                "expected_sections": case["expected_sections"],
                "raw_match_count": len(raw_matches),
                "deduped_match_count": len(deduped),
                "top_sections": [match["metadata"].get("section_title", "") for match in hydrated[:3]],
                "top_child_snippets": [match["child_text"][:220] for match in hydrated[:3]],
                "evaluation": evaluation,
            }
        )

    passed = sum(1 for case in cases if case["evaluation"]["passed"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "namespace": args.namespace,
        "index_name": index_name,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "case_count": len(cases),
        "passed_count": passed,
        "cases": cases,
    }
    out_path = write_report(project_root, args.run_id, report)
    print(json.dumps({"ok": True, "report_path": str(out_path), "passed_count": passed, "case_count": len(cases)}, indent=2))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
