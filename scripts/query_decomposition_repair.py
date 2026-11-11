import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.llm_query_decomposition import (
    MODEL_NAME,
    decompose_query_with_llm,
)


REPAIR_MODEL = MODEL_NAME

REPAIR_PROMPT = """You repair query decomposition outputs for a production RAG system.
Return only valid JSON.

Goals:
- Preserve the original user meaning.
- Fix under-splitting, unresolved references, and overly generic phrasing.
- Keep comparisons atomic unless the user clearly asked separate questions.
- Avoid vague placeholders such as "the company", "they", "it", "the same periods", or "relevant periods".
- Never introduce placeholders such as `{company}`, `[company]`, `{period}`, or `[period]`.
- Never ask a clarification question.
- Never expand the user's request into a broader analysis plan.
- If the original query makes the subject explicit, keep that explicit in repaired sub-queries.
- Do not add new asks that were not implied by the original query.
- Prefer concise retrieval-ready question forms.
- If the original query does not contain enough information to make a subject or timeframe explicit, keep the wording close to the original instead of inventing one.
"""

REPAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "needs_decomposition": {"type": "boolean"},
        "sub_queries": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning_type": {
            "type": "string",
            "enum": [
                "single_intent",
                "multi_question",
                "multi_intent",
                "list_intent",
                "comparison",
                "ambiguous",
            ],
        },
        "decomposition_strategy": {
            "type": "string",
            "enum": [
                "none",
                "question_boundary_split",
                "conjunction_split",
                "list_item_split",
                "comparison_atomic",
                "semantic_split",
                "repair_normalized",
            ],
        },
        "repair_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "needs_decomposition",
        "sub_queries",
        "reasoning_type",
        "decomposition_strategy",
        "repair_notes",
    ],
}

UNRESOLVED_REFERENCE_PATTERN = re.compile(
    r"\b(the company|company|they|them|their|it|its|those|these|same periods|relevant periods|the ceo|the cfo|the segment|the business segment)\b",
    re.IGNORECASE,
)
SPLIT_CUE_PATTERN = re.compile(
    r"\b(and|also|plus|along with|as well as)\b",
    re.IGNORECASE,
)
GENERIC_METRIC_PATTERN = re.compile(
    r"\b(what is the|what are the|which segment has|what did they say)\b",
    re.IGNORECASE,
)


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def normalize_repair_result(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    sub_queries = [str(item).strip() for item in payload.get("sub_queries", []) if str(item).strip()]
    if not sub_queries:
        sub_queries = [query.strip()]
    return {
        "sanitized_query": query.strip(),
        "needs_decomposition": bool(payload.get("needs_decomposition", len(sub_queries) > 1)),
        "sub_queries": sub_queries,
        "reasoning_type": payload.get("reasoning_type", "ambiguous"),
        "decomposition_strategy": payload.get("decomposition_strategy", "repair_normalized"),
        "repair_notes": [str(item).strip() for item in payload.get("repair_notes", []) if str(item).strip()],
    }


def has_unresolved_reference(text: str) -> bool:
    normalized = text.strip().lower()
    if not UNRESOLVED_REFERENCE_PATTERN.search(normalized):
        return False

    resolved_role_patterns = [
        r"\bthe ceo of [a-z0-9& ,.'-]+\b",
        r"\bthe cfo of [a-z0-9& ,.'-]+\b",
        r"\bthe business segment of [a-z0-9& ,.'-]+\b",
        r"\bthe segment of [a-z0-9& ,.'-]+\b",
    ]
    return not any(re.search(pattern, normalized) for pattern in resolved_role_patterns)


def should_repair(query: str, decomposition: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    sub_queries = decomposition.get("sub_queries", [])
    lower_query = query.lower()
    is_comparison_query = any(token in lower_query for token in ["compare", "compared with", "versus", " vs "])

    if not sub_queries:
        reasons.append("empty_sub_queries")

    if len(set(item.lower().strip() for item in sub_queries)) < len(sub_queries):
        reasons.append("duplicate_sub_queries")

    if decomposition.get("needs_decomposition") is False and SPLIT_CUE_PATTERN.search(query) and not is_comparison_query:
        reasons.append("possible_under_split")

    if any(has_unresolved_reference(item or "") for item in sub_queries):
        reasons.append("unresolved_reference")

    if any(GENERIC_METRIC_PATTERN.search(item or "") for item in sub_queries) and not is_comparison_query:
        reasons.append("generic_retrieval_form")

    if any(SPLIT_CUE_PATTERN.search(item or "") for item in sub_queries) and not is_comparison_query:
        reasons.append("multi_intent_subquery")

    return {
        "should_repair": bool(reasons),
        "reasons": sorted(set(reasons)),
    }


def repair_query_decomposition(
    query: str,
    decomposition: dict[str, Any],
    model: str = REPAIR_MODEL,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    gate = should_repair(query, decomposition)
    if not gate["should_repair"]:
        return {
            **decomposition,
            "repair_applied": False,
            "repair_reasons": [],
            "repair_notes": [],
        }

    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "query_decomposition_repair",
                "schema": REPAIR_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": REPAIR_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_query": query,
                        "current_decomposition": decomposition,
                        "repair_reasons": gate["reasons"],
                    }
                ),
            },
        ],
    )
    content = completion.choices[0].message.content or "{}"
    payload = json.loads(content)
    normalized = normalize_repair_result(query, payload)
    normalized["repair_applied"] = True
    normalized["repair_reasons"] = gate["reasons"]
    return normalized


def write_report(project_root: Path, run_id: str, query: str, original: dict[str, Any], repaired: dict[str, Any], model: str) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "query_transforms"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "query_decomposition_repair_report.json"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "query": query,
        "original_result": original,
        "repaired_result": repaired,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair LLM query decomposition results.")
    parser.add_argument(
        "--query",
        default="Who is the CEO and what segment drove the most revenue growth and what did they say about AI?",
    )
    parser.add_argument("--model", default=REPAIR_MODEL)
    parser.add_argument("--run-id", default="component3_query_decomposition_repair")
    args = parser.parse_args()

    original = decompose_query_with_llm(args.query, model=args.model)
    repaired = repair_query_decomposition(args.query, original, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, args.query, original, repaired, args.model)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "repair_applied": repaired["repair_applied"],
                "sub_query_count": len(repaired["sub_queries"]),
                "model": args.model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
