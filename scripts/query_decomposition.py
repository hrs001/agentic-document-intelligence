import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


CONJUNCTION_SPLIT_PATTERNS = [
    re.compile(r"\s+(?:and|as well as)\s+", re.IGNORECASE),
    re.compile(r"\s*,\s*(?:and\s+)?", re.IGNORECASE),
]

QUESTION_SPLIT_PATTERN = re.compile(r"\?\s+")
COMPARISON_PATTERN = re.compile(r"\b(compare|versus|vs\.?|difference between)\b", re.IGNORECASE)
MULTI_INTENT_VERBS = re.compile(
    r"\b(compare|explain|summarize|list|identify|show|describe|tell me|who is|what is|how much)\b",
    re.IGNORECASE,
)
LIST_PREFIX_PATTERN = re.compile(
    r"^(list|show|identify|summarize|describe)\s+(?P<body>.+?)\.?$",
    re.IGNORECASE,
)
METRIC_PATTERN = re.compile(
    r"\b(revenue growth|operating income|net income|gross margin|operating margin|revenue|azure growth|capital expenditures|cash flow)\b",
    re.IGNORECASE,
)


def sanitize_query(text: str) -> str:
    cleaned = text.replace("\u200b", " ").replace("\ufeff", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def ensure_query_suffix(text: str) -> str:
    normalized = text.strip()
    if normalized.endswith("?"):
        return normalized
    normalized = normalized.rstrip(".")
    return f"{normalized}?"


def _is_atomic_clause(clause: str) -> bool:
    if len(clause.split()) < 3:
        return False
    if clause.lower() in {"and", "or", "also"}:
        return False
    return True


def _split_question_boundaries(query: str) -> list[str]:
    clauses = []
    for part in QUESTION_SPLIT_PATTERN.split(query.replace("?", "? ")):
        part = part.strip(" .?")
        if _is_atomic_clause(part):
            clauses.append(part)
    return clauses


def _split_multi_clause_sentence(query: str) -> list[str]:
    if COMPARISON_PATTERN.search(query):
        return [query.strip(" .?")]

    candidate_clauses = [query]
    for pattern in CONJUNCTION_SPLIT_PATTERNS:
        next_clauses = []
        for clause in candidate_clauses:
            parts = [part.strip(" .?") for part in pattern.split(clause) if _is_atomic_clause(part.strip(" .?"))]
            if len(parts) > 1:
                next_clauses.extend(parts)
            else:
                next_clauses.append(clause.strip(" .?"))
        candidate_clauses = next_clauses

    deduped = []
    seen = set()
    for clause in candidate_clauses:
        normalized = clause.lower()
        if normalized not in seen and _is_atomic_clause(clause):
            deduped.append(clause)
            seen.add(normalized)
    return deduped


def _split_list_query(query: str) -> list[str]:
    match = LIST_PREFIX_PATTERN.match(query.strip())
    if not match:
        return []

    command = match.group(1).strip()
    body = match.group("body").strip()
    metric_matches = list(METRIC_PATTERN.finditer(body))
    if len(metric_matches) < 2:
        return []

    prefix = body[: metric_matches[0].start()].strip(" ,")
    clauses = []
    seen = set()
    for metric_match in metric_matches:
        metric = metric_match.group(0).strip()
        clause = f"{command} {prefix} {metric}".strip()
        normalized = clause.lower()
        if normalized not in seen and _is_atomic_clause(clause):
            clauses.append(clause)
            seen.add(normalized)
    return clauses


def decompose_query(text: str) -> dict:
    sanitized_query = sanitize_query(text)
    if not sanitized_query:
        return {
            "sanitized_query": sanitized_query,
            "needs_decomposition": False,
            "sub_queries": [],
            "reasoning_type": "empty",
            "decomposition_strategy": "none",
        }

    question_clauses = _split_question_boundaries(sanitized_query)
    if len(question_clauses) > 1:
        return {
            "sanitized_query": sanitized_query,
            "needs_decomposition": True,
            "sub_queries": question_clauses,
            "reasoning_type": "multi_question",
            "decomposition_strategy": "question_boundary_split",
        }

    list_clauses = _split_list_query(sanitized_query)
    if len(list_clauses) > 1:
        return {
            "sanitized_query": sanitized_query,
            "needs_decomposition": True,
            "sub_queries": [ensure_query_suffix(clause) for clause in list_clauses],
            "reasoning_type": "list_intent",
            "decomposition_strategy": "list_item_split",
        }

    clause_splits = _split_multi_clause_sentence(sanitized_query)
    unique_sub_queries = [ensure_query_suffix(clause) for clause in clause_splits]
    if len(unique_sub_queries) > 1 and MULTI_INTENT_VERBS.search(sanitized_query):
        return {
            "sanitized_query": sanitized_query,
            "needs_decomposition": True,
            "sub_queries": unique_sub_queries,
            "reasoning_type": "multi_intent",
            "decomposition_strategy": "conjunction_split",
        }

    return {
        "sanitized_query": sanitized_query,
        "needs_decomposition": False,
        "sub_queries": [ensure_query_suffix(sanitized_query)],
        "reasoning_type": "single_intent",
        "decomposition_strategy": "none",
    }


def write_report(project_root: Path, run_id: str, query: str, result: dict) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "query_transforms"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "query_decomposition_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic query decomposition.")
    parser.add_argument(
        "--query",
        default="Compare Microsoft's FY2025 revenue growth and explain who the CEO is.",
        help="Query to inspect.",
    )
    parser.add_argument(
        "--run-id",
        default="component3_query_decomposition",
        help="Artifact run id.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    result = decompose_query(args.query)
    report_path = write_report(project_root, args.run_id, args.query, result)

    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "needs_decomposition": result["needs_decomposition"],
                "sub_query_count": len(result["sub_queries"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
