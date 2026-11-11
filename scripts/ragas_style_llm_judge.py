import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agentic_document_intelligence.scripts.self_reflective_answer_critique import load_result


DEFAULT_FUSED_INPUT = (
    "artifacts/experiments/component8_cross_source_evidence_fusion_sample/"
    "answer_context/cross_source_fused_evidence.json"
)
DEFAULT_ANSWER_INPUT = (
    "artifacts/experiments/component8_grounded_answer_generation_live_smoke/"
    "answers/grounded_answer_report.json"
)
DEFAULT_MODEL = "gpt-5-mini"

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "answer_relevancy": {"type": "integer", "minimum": 1, "maximum": 5},
        "context_precision": {"type": "integer", "minimum": 1, "maximum": 5},
        "citation_grounding": {"type": "integer", "minimum": 1, "maximum": 5},
        "overall_verdict": {"type": "string", "enum": ["pass", "borderline", "fail"]},
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "citation_grounding",
        "overall_verdict",
        "summary",
        "strengths",
        "weaknesses",
        "recommendations",
    ],
}

SYSTEM_PROMPT = """You are an LLM judge for a production RAG system.
Return only valid JSON.

Use RAGAS-style metrics:
- faithfulness: are the answer claims supported by the provided evidence?
- answer_relevancy: does the answer actually address the user query?
- context_precision: are the cited/used facts relevant and not bloated with irrelevant evidence?
- citation_grounding: are citations attached appropriately to major claims?

Scoring:
- 5 = excellent
- 4 = strong
- 3 = mixed / borderline
- 2 = weak
- 1 = poor

Verdict guidance:
- pass: strong enough to return to the user
- borderline: acceptable with caveats, but likely worth monitoring or repair
- fail: should not be accepted as final
"""


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def build_judge_input(
    fused_bundle: dict[str, Any],
    answer_result: dict[str, Any],
    critique_result: dict[str, Any] | None = None,
    max_used_facts: int = 12,
) -> dict[str, Any]:
    fact_map = {fact["fact_id"]: fact for fact in fused_bundle.get("normalized_facts", [])}
    used_fact_ids = list(answer_result.get("used_fact_ids", []))
    used_facts = [
        {
            "fact_id": fact["fact_id"],
            "source_type": fact["source_type"],
            "fact_type": fact["fact_type"],
            "summary": fact["summary"],
            "sub_query_id": fact["sub_query_id"],
        }
        for fact_id in used_fact_ids[:max_used_facts]
        for fact in [fact_map.get(fact_id)]
        if fact
    ]
    return {
        "original_query": fused_bundle["original_query"],
        "answer_markdown": answer_result.get("answer_markdown", ""),
        "used_fact_ids": used_fact_ids,
        "used_facts": used_facts,
        "sub_queries": [
            {
                "sub_query_id": sub_query["sub_query_id"],
                "original_sub_query": sub_query["original_sub_query"],
                "active_sources": sub_query.get("active_sources", []),
                "fact_count": sub_query.get("fact_count", 0),
            }
            for sub_query in fused_bundle.get("sub_query_fusions", [])
        ],
        "conflict_signal_count": fused_bundle.get("bundle_summary", {}).get("conflict_signal_count", 0),
        "critique_summary": (critique_result or {}).get("final_critique", {}),
    }


def sanitize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "faithfulness": int(payload.get("faithfulness", 1)),
        "answer_relevancy": int(payload.get("answer_relevancy", 1)),
        "context_precision": int(payload.get("context_precision", 1)),
        "citation_grounding": int(payload.get("citation_grounding", 1)),
        "overall_verdict": str(payload.get("overall_verdict", "fail")).strip() or "fail",
        "summary": str(payload.get("summary", "")).strip(),
        "strengths": [str(item).strip() for item in payload.get("strengths", []) if str(item).strip()],
        "weaknesses": [str(item).strip() for item in payload.get("weaknesses", []) if str(item).strip()],
        "recommendations": [str(item).strip() for item in payload.get("recommendations", []) if str(item).strip()],
    }


def judge_final_answer(
    fused_bundle: dict[str, Any],
    answer_result: dict[str, Any],
    critique_result: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and client is None:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    judge_input = build_judge_input(fused_bundle, answer_result, critique_result)
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ragas_style_llm_judge",
                "schema": JUDGE_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(judge_input)},
        ],
    )
    payload = json.loads(completion.choices[0].message.content or "{}")
    sanitized = sanitize_judge_payload(payload)
    scores = [
        sanitized["faithfulness"],
        sanitized["answer_relevancy"],
        sanitized["context_precision"],
        sanitized["citation_grounding"],
    ]
    return {
        "original_query": fused_bundle["original_query"],
        "judge_model": model,
        "metrics": sanitized,
        "average_score": round(sum(scores) / len(scores), 3),
        "usage": getattr(completion, "usage", None).model_dump() if getattr(completion, "usage", None) else {},
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ragas_style_llm_judge.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a RAGAS-style LLM judge over a final answer.")
    parser.add_argument("--fused-input", default=DEFAULT_FUSED_INPUT)
    parser.add_argument("--answer-input", default=DEFAULT_ANSWER_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="component9_ragas_style_llm_judge")
    args = parser.parse_args()

    fused_bundle = load_result(PROJECT_ROOT / "agentic_document_intelligence" / args.fused_input)
    answer_result = load_result(PROJECT_ROOT / "agentic_document_intelligence" / args.answer_input)
    result = judge_final_answer(fused_bundle, answer_result, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "overall_verdict": result["metrics"]["overall_verdict"],
                "average_score": result["average_score"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
