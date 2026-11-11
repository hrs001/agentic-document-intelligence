import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.hyde_query_generation import (
    MODEL_NAME,
    generate_hyde_passage,
)


DEFAULT_CASES_PATH = (
    PROJECT_ROOT
    / "agentic_document_intelligence"
    / "evals"
    / "hyde_query_generation_cases.json"
)


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def sentence_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    return len(parts)


def contains_any(text: str, terms: list[str]) -> bool:
    normalized = normalize_text(text)
    return all(term in normalized for term in terms)


def evaluate_case(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    passage = actual["hypothetical_passage"]
    term_ok = contains_any(passage, expected["required_terms"])
    forbidden_ok = not any(term in normalize_text(passage) for term in expected.get("forbidden_terms", []))
    sentence_ok = 2 <= sentence_count(passage) <= 4
    pass_flag = term_ok and forbidden_ok and sentence_ok
    return {
        "passed": pass_flag,
        "term_ok": term_ok,
        "forbidden_ok": forbidden_ok,
        "sentence_ok": sentence_ok,
        "sentence_count": sentence_count(passage),
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(run_id: str, payload: dict[str, Any]) -> Path:
    output_dir = (
        PROJECT_ROOT
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "evaluations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hyde_query_generation_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HyDE generation quality.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--run-id", default="component3_hyde_query_generation_eval")
    args = parser.parse_args()

    cases_path = Path(args.cases_path)
    cases = load_cases(cases_path)
    results = []

    for case in cases:
        query = case["query"]
        generated = generate_hyde_passage(query, model=args.model)
        results.append(
            {
                "query": query,
                "result": generated,
                "evaluation": evaluate_case(generated, case["expected"]),
            }
        )

    passed_count = sum(item["evaluation"]["passed"] for item in results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "cases_path": str(cases_path),
        "case_count": len(cases),
        "passed_count": passed_count,
        "results": results,
    }
    report_path = write_report(args.run_id, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "case_count": len(cases),
                "passed_count": passed_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
