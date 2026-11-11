from __future__ import annotations

import concurrent.futures
import json
import mimetypes
import os
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import boto3
from openai import OpenAI
from pinecone import Pinecone

from .runtime_bundle import ensure_runtime_bundle_ready

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.scripts.corrective_answer_repair import repair_grounded_answer
from agentic_document_intelligence.scripts.conversation_query_resolution import resolve_conversational_query
from agentic_document_intelligence.scripts.contextual_compression import attach_contextual_compression
from agentic_document_intelligence.scripts.execute_latency_optimized_orchestration import (
    DEFAULT_EMBEDDING_INPUT,
    DEFAULT_GRAPH_DATABASE_PATH,
    DEFAULT_SQL_DATABASE_PATH,
    DEFAULT_SQL_SCHEMA_PATH,
    MAX_SUB_QUERIES,
    build_minimal_orchestration_bundle,
    execute_latency_optimized_orchestration,
)
from agentic_document_intelligence.scripts.execute_multi_source_orchestration import (
    DEFAULT_CHUNK_INPUT,
    DEFAULT_NAMESPACE,
    load_project_env,
)
from agentic_document_intelligence.scripts.generate_grounded_answer import generate_grounded_answer
from agentic_document_intelligence.scripts.input_query_guardrails import (
    MODEL_NAME as DEFAULT_GUARDRAIL_MODEL,
    inspect_query,
    sanitize_conversation_history,
)
from agentic_document_intelligence.scripts.latency_optimized_orchestration_policy import build_latency_optimized_policy
from agentic_document_intelligence.scripts.mmr_diversification import load_embedding_records
from agentic_document_intelligence.scripts.multi_source_routing import (
    build_graph_capability_summary,
    build_multi_source_routing_plan,
    build_sql_capability_summary,
    load_json_result,
)
from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import build_chunk_indexes, load_chunk_artifact
from agentic_document_intelligence.scripts.ragas_style_llm_judge import judge_final_answer
from agentic_document_intelligence.scripts.runtime_quality_gating import decide_runtime_action
from agentic_document_intelligence.scripts.self_reflective_answer_critique import critique_grounded_answer


DEFAULT_PIPELINE_MODEL = "gpt-5-mini"
DEFAULT_ANSWER_MODEL = "gpt-5.1"
DEFAULT_COMPRESSION_MODEL = "gpt-5-mini"
DEFAULT_MIN_FACTS_FOR_COMPRESSION = 30
DEFAULT_GUARDRAIL_BLOCK_MESSAGE = (
    "I blocked this request because it appears to contain prompt-injection, secret-exfiltration, "
    "or unsafe personal-data extraction instructions."
)
DEFAULT_CRITIQUE_MODEL = "gpt-5-mini"
DEFAULT_REPAIR_MODEL = "gpt-5.1"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"

DEFAULT_MAX_TOTAL_ROUNDS = 2
DEFAULT_MAX_PIPELINE_RERUNS = 1
DEFAULT_MAX_ANSWER_REGENERATIONS = 1
DEFAULT_MAX_REPAIRS = 1
DEFAULT_SESSION_QUESTION_LIMIT = 25
DEFAULT_CONVERSATION_MEMORY_TURNS = 10

GRAPH_VALIDATED_PATH = (
    Path("artifacts")
    / "experiments"
    / "component5_graph_schema_validation_live"
    / "graph_validated"
    / "microsoft_fy2025_10k_summary_graph_validated.json"
)
CORPUS_METADATA_PATH = Path("corpus") / "metadata" / "corpus_metadata.json"
CORPUS_MANIFEST_PATH = Path("corpus") / "metadata" / "corpus_manifest.json"

FACT_ID_PATTERN = re.compile(r"\[([^\[\]]+::[^\[\]]+)\]")
PREVIOUS_QUESTION_PATTERN = re.compile(
    r"\b(what|which)\s+(is|was)\s+(my|the)\s+(previous|last|prior)\s+question\b",
    re.IGNORECASE,
)
PREVIOUS_ANSWER_PATTERN = re.compile(
    r"\b(what|which)\s+(is|was)\s+(your|the)\s+(previous|last|prior)\s+(answer|response)\b",
    re.IGNORECASE,
)
CONVERSATION_SUMMARY_PATTERN = re.compile(
    r"\b(summarize|summary of|recap)\b.*\b(conversation|chat|discussion)\b",
    re.IGNORECASE,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_conversation_meta_intent(query: str) -> str | None:
    normalized = query.strip()
    if PREVIOUS_QUESTION_PATTERN.search(normalized):
        return "previous_question"
    if PREVIOUS_ANSWER_PATTERN.search(normalized):
        return "previous_answer"
    if CONVERSATION_SUMMARY_PATTERN.search(normalized):
        return "conversation_summary"
    return None


def build_unavailable_evaluation(reasoning: str) -> dict[str, Any]:
    return {
        "faithfulness": -1.0,
        "answer_relevancy": -1.0,
        "context_precision": -1.0,
        "citation_grounding": -1.0,
        "overall_badge": "N/A",
        "reasoning": reasoning,
        "strengths": [],
        "weaknesses": [],
        "recommendations": [],
    }


def build_conversation_meta_response(
    query: str,
    meta_intent: str,
    conversation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    if meta_intent == "previous_question":
        if conversation_history:
            previous_turn = conversation_history[-1]
            answer = f'Your previous question was: "{previous_turn.get("user_query", "").strip()}"'
            reasoning = "Answered directly from the supplied conversation history."
        else:
            answer = "I do not have any earlier question in the current chat history."
            reasoning = "No prior conversation turn was available in the supplied chat history."
    elif meta_intent == "previous_answer":
        if conversation_history:
            previous_turn = conversation_history[-1]
            answer = f'Your previous answer was: "{str(previous_turn.get("answer_summary", "")).strip()}"'
            reasoning = "Answered directly from the supplied conversation history."
        else:
            answer = "I do not have any earlier answer in the current chat history."
            reasoning = "No prior conversation turn was available in the supplied chat history."
    else:
        if conversation_history:
            lines = []
            for turn in conversation_history[-5:]:
                user_query = str(turn.get("user_query", "")).strip()
                answer_summary = str(turn.get("answer_summary", "")).strip()
                if user_query:
                    lines.append(f'- You asked: "{user_query}"')
                if answer_summary:
                    lines.append(f'  I answered: "{answer_summary}"')
            answer = "Here is a short recap of our recent conversation:\n" + "\n".join(lines)
            reasoning = "Summarized directly from the supplied conversation history."
        else:
            answer = "I do not have any earlier conversation history to summarize."
            reasoning = "No prior conversation turn was available in the supplied chat history."

    return {
        "answer": answer,
        "citations": [],
        "evaluation": build_unavailable_evaluation(reasoning),
        "evaluation_pending": False,
        "truncated_subqueries": False,
        "subquery_cap": MAX_SUB_QUERIES,
        "judge_average_score": None,
        "runtime_history": [{"round": 0, "action": "conversation_meta_direct_answer"}],
        "termination": {"action": "conversation_meta_direct_answer", "reason": reasoning},
        "critique": {
            "grounded": True,
            "complete": True,
            "needs_correction": False,
            "issues": [],
            "repair_plan": [],
        },
        "sources_used": ["conversation_memory"],
        "status": "ok",
        "original_query": query,
        "resolved_query": query,
        "conversation_resolution": {
            "original_query": query,
            "resolved_query": query,
            "used_history": bool(conversation_history),
            "referenced_turn_ids": [str(conversation_history[-1].get("turn_id"))] if conversation_history else [],
            "confidence": "high" if conversation_history else "low",
            "clarification_needed": False,
            "notes": reasoning,
            "recent_turn_count": len(conversation_history),
        },
        "meta_intent": meta_intent,
    }


def build_guardrail_ui_payload(guardrail_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "blocked": bool(guardrail_result.get("blocked", False)),
        "reason": str(guardrail_result.get("reason", "")).strip(),
        "risk_level": str(guardrail_result.get("risk_level", "")).strip(),
        "warnings": list(guardrail_result.get("warnings", [])),
        "attack_types": list(guardrail_result.get("attack_types", [])),
        "pii_findings": list(guardrail_result.get("pii_findings", [])),
        "sanitized_query": str(guardrail_result.get("sanitized_query", "")).strip(),
        "policy_version": str(guardrail_result.get("policy_version", "")).strip(),
        "confidence": str(guardrail_result.get("confidence", "")).strip(),
    }


def build_guardrail_block_response(
    original_query: str,
    sanitized_query: str,
    conversation_history: list[dict[str, Any]],
    guardrail_result: dict[str, Any],
) -> dict[str, Any]:
    reasoning = str(guardrail_result.get("user_message", "")).strip() or DEFAULT_GUARDRAIL_BLOCK_MESSAGE
    return {
        "answer": reasoning,
        "citations": [],
        "evaluation": build_unavailable_evaluation("Blocked by input guardrails before retrieval."),
        "evaluation_pending": False,
        "truncated_subqueries": False,
        "subquery_cap": MAX_SUB_QUERIES,
        "judge_average_score": None,
        "runtime_history": [{"round": 0, "action": "input_guardrail_block"}],
        "termination": {"action": "input_guardrail_block", "reason": guardrail_result.get("reason", "blocked")},
        "critique": {
            "grounded": True,
            "complete": True,
            "needs_correction": False,
            "issues": [],
            "repair_plan": [],
        },
        "sources_used": [],
        "status": "blocked",
        "original_query": original_query,
        "resolved_query": sanitized_query or original_query,
        "conversation_resolution": {
            "original_query": original_query,
            "resolved_query": sanitized_query or original_query,
            "used_history": bool(conversation_history),
            "referenced_turn_ids": [],
            "confidence": "high",
            "clarification_needed": False,
            "notes": "Blocked before conversational resolution.",
            "recent_turn_count": len(conversation_history),
        },
        "guardrails": build_guardrail_ui_payload(guardrail_result),
    }


@dataclass
class JobEvent:
    timestamp: str
    component: str
    data: str


@dataclass
class JobRecord:
    job_id: str
    kind: str
    session_id: str
    status: str = "running"
    result: dict[str, Any] | None = None
    error: str | None = None
    events: list[JobEvent] = field(default_factory=list)


@dataclass
class SessionRecord:
    session_id: str
    hydrated: bool = False
    question_limit: int = DEFAULT_SESSION_QUESTION_LIMIT
    questions_used: int = 0

    @property
    def questions_remaining(self) -> int:
        return max(self.question_limit - self.questions_used, 0)


class S3StateStore:
    def __init__(self, bucket: str, prefix: str) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/").strip()
        self.s3 = boto3.client("s3")

    def _key(self, kind: str, identifier: str) -> str:
        return f"{self.prefix}/{kind}/{identifier}.json"

    def load_json(self, kind: str, identifier: str) -> dict[str, Any] | None:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self._key(kind, identifier))
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if error_code in {"NoSuchKey", "404"}:
                return None
            raise
        return json.loads(response["Body"].read().decode("utf-8"))

    def save_json(self, kind: str, identifier: str, payload: dict[str, Any]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key(kind, identifier),
            Body=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            ContentType="application/json",
        )


class DemoResources:
    def __init__(self) -> None:
        load_project_env()
        self.runtime_project_root = ensure_runtime_bundle_ready(PROJECT_ROOT)
        self.runtime_app_root = self.runtime_project_root / "agentic_document_intelligence"
        self._lock = threading.Lock()
        self._initialized = False
        self.openai_client: OpenAI | None = None
        self.pinecone_client: Pinecone | None = None
        self.index: Any | None = None
        self.schema_package: dict[str, Any] | None = None
        self.sql_capability_summary: str | None = None
        self.graph_capability_summary: dict[str, Any] | None = None
        self.child_index: dict[str, dict[str, Any]] | None = None
        self.parent_index: dict[str, str] | None = None
        self.chunk_to_record_id: dict[str, str] | None = None
        self.graph_payload: dict[str, Any] | None = None
        self.graph_max_page: int = 0
        self.corpus_metadata = json.loads((self.runtime_app_root / CORPUS_METADATA_PATH).read_text(encoding="utf-8"))
        self.corpus_manifest = json.loads((self.runtime_app_root / CORPUS_MANIFEST_PATH).read_text(encoding="utf-8"))
        self.demo_asset_catalog = build_demo_asset_catalog(self.runtime_app_root, self.corpus_manifest)

    def ensure_initialized(self) -> None:
        with self._lock:
            if self._initialized:
                return

            openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
            pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
            index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
            if not openai_api_key:
                raise RuntimeError("OPENAI_API_KEY must be set in agentic_document_intelligence/.env")
            if not pinecone_api_key:
                raise RuntimeError("PINECONE_API_KEY must be set in agentic_document_intelligence/.env")

            if self.openai_client is None:
                self.openai_client = OpenAI(api_key=openai_api_key)
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            self.index = self.pinecone_client.Index(index_name)
            self.schema_package = load_json_result(self.runtime_app_root / DEFAULT_SQL_SCHEMA_PATH)
            self.sql_capability_summary = build_sql_capability_summary(self.schema_package)
            self.graph_capability_summary = build_graph_capability_summary()
            chunk_artifact = load_chunk_artifact(self.runtime_app_root / DEFAULT_CHUNK_INPUT)
            self.child_index, self.parent_index = build_chunk_indexes(chunk_artifact)
            self.chunk_to_record_id = load_embedding_records(self.runtime_app_root / DEFAULT_EMBEDDING_INPUT)
            self.graph_payload = self._load_graph_payload()
            self._initialized = True

    def ensure_openai_client(self) -> OpenAI:
        with self._lock:
            if self.openai_client is not None:
                return self.openai_client
            openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not openai_api_key:
                raise RuntimeError("OPENAI_API_KEY must be set in agentic_document_intelligence/.env")
            self.openai_client = OpenAI(api_key=openai_api_key)
            return self.openai_client

    def _load_graph_payload(self) -> dict[str, Any]:
        payload = json.loads((self.runtime_app_root / GRAPH_VALIDATED_PATH).read_text(encoding="utf-8"))
        nodes = []
        edges = []
        max_page = 0

        for node in payload.get("validated_nodes", []):
            page_ranges = node.get("page_ranges", [])
            for page in page_ranges:
                max_page = max(max_page, int(page.get("page_end") or 0))
            nodes.append(
                {
                    "id": node["node_id"],
                    "name": node["canonical_name"],
                    "type": node["entity_type"],
                    "aliases": node.get("aliases", []),
                    "mention_count": node.get("mention_count", 0),
                    "section_titles": node.get("section_titles", []),
                    "page_ranges": page_ranges,
                    "evidence_snippets": node.get("evidence_snippets", []),
                }
            )

        for edge in payload.get("validated_edges", []):
            page_ranges = edge.get("page_ranges", [])
            for page in page_ranges:
                max_page = max(max_page, int(page.get("page_end") or 0))
            edges.append(
                {
                    "id": edge["edge_id"],
                    "source": edge["source_node_id"],
                    "target": edge["target_node_id"],
                    "relationship": edge["relation_type"],
                    "mention_count": edge.get("mention_count", 0),
                    "section_titles": edge.get("section_titles", []),
                    "page_ranges": page_ranges,
                    "evidence_snippets": edge.get("evidence_snippets", []),
                }
            )

        self.graph_max_page = max_page
        return {
            "nodes": nodes,
            "edges": edges,
            "graph_summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "max_page": max_page,
            },
        }


class DemoAppService:
    def __init__(self, resources: DemoResources | None = None, max_workers: int = 4) -> None:
        self.resources = resources or DemoResources()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self.sessions: dict[str, SessionRecord] = {}
        self.jobs: dict[str, JobRecord] = {}
        state_bucket = os.getenv("ADI_RUNTIME_STATE_BUCKET", "").strip()
        state_prefix = os.getenv("ADI_RUNTIME_STATE_PREFIX", "runtime-state").strip()
        self.state_store = S3StateStore(state_bucket, state_prefix) if state_bucket else None
        self.lambda_function_name = os.getenv("AWS_LAMBDA_FUNCTION_NAME", "").strip()
        self.lambda_client = boto3.client("lambda") if self.state_store and self.lambda_function_name else None

    def get_or_create_session(self, session_id: str) -> SessionRecord:
        if self.state_store:
            payload = self.state_store.load_json("sessions", session_id)
            if payload is not None:
                return SessionRecord(**payload)
            session = SessionRecord(session_id=session_id)
            self._save_session(session)
            return session
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                session = SessionRecord(session_id=session_id)
                self.sessions[session_id] = session
            return session

    def _save_session(self, session: SessionRecord) -> None:
        if self.state_store:
            self.state_store.save_json("sessions", session.session_id, session.__dict__)
            return
        with self._lock:
            self.sessions[session.session_id] = session

    def get_session_status(self, session_id: str) -> dict[str, Any]:
        session = self.get_or_create_session(session_id)
        corpus_items = []
        if session.hydrated:
            corpus_items.extend(document["display_name"] for document in self.resources.corpus_manifest["documents"])
            corpus_items.extend(dataset["display_name"] for dataset in self.resources.corpus_manifest["datasets"])
        return {
            "session_id": session_id,
            "has_any_data": session.hydrated,
            "uploaded_files": corpus_items,
            "questions_remaining": session.questions_remaining,
            "graphrag_max_pages": self.resources.graph_max_page or 0,
            "demo_corpus": self._build_demo_corpus_payload(),
        }

    def _build_demo_corpus_payload(self) -> dict[str, Any]:
        document = self.resources.corpus_metadata["document_metadata"][0]
        dataset = self.resources.corpus_metadata["dataset_metadata"][0]
        document_manifest = self.resources.corpus_manifest["documents"][0]
        dataset_manifest = self.resources.corpus_manifest["datasets"][0]
        return {
            "corpus_id": self.resources.corpus_metadata["corpus_id"],
            "document": {
                "title": document["title"],
                "issuer": document["issuer"],
                "fiscal_period": document["fiscal_period"],
                "topic_tags": document["topic_tags"],
                "download": {
                    "asset_id": "document:microsoft_fy2025_10k_summary",
                    "label": "Download PDF",
                    "file_name": Path(document_manifest["relative_path"]).name,
                    "download_path": "/api/v1/demo-assets/document:microsoft_fy2025_10k_summary",
                },
            },
            "dataset": {
                "display_name": dataset["display_name"],
                "synthetic": dataset["synthetic"],
                "tables": [
                    {
                        "table_name": table["table_name"],
                        "download": {
                            "asset_id": f"dataset:{table['table_name']}",
                            "label": "Download CSV",
                            "file_name": Path(manifest_table["relative_path"]).name,
                            "download_path": f"/api/v1/demo-assets/dataset:{table['table_name']}",
                        },
                    }
                    for table, manifest_table in zip(dataset["tables"], dataset_manifest["tables"], strict=True)
                ],
            },
        }

    def get_demo_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self.resources.demo_asset_catalog.get(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        return asset

    def create_demo_hydrate_job(self, session_id: str) -> str:
        job = self._create_job("demo_hydrate", session_id)
        if self.lambda_client:
            self._invoke_async_job(
                {
                    "job_id": job.job_id,
                    "kind": "demo_hydrate",
                    "session_id": session_id,
                }
            )
        else:
            self.executor.submit(self._run_job, job.job_id, lambda: self._run_demo_hydrate_pipeline(job.job_id, session_id))
        return job.job_id

    def create_chat_job(
        self,
        session_id: str,
        query: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        job = self._create_job("chat", session_id)
        if self.lambda_client:
            self._invoke_async_job(
                {
                    "job_id": job.job_id,
                    "kind": "chat",
                    "session_id": session_id,
                    "query": query,
                    "conversation_history": conversation_history or [],
                }
            )
        else:
            self.executor.submit(
                self._run_job,
                job.job_id,
                lambda: self._run_chat_pipeline(job.job_id, session_id, query, conversation_history or []),
            )
        return job.job_id

    def get_job_status(self, job_id: str, after: int) -> dict[str, Any]:
        if self.state_store:
            payload = self.state_store.load_json("jobs", job_id)
            if payload is None:
                return {
                    "success": False,
                    "status": "not_found",
                    "error": "Unknown job id.",
                    "hint": "Start a new job and poll using the returned job_id.",
                }
            events = payload.get("events", [])
            return {
                "success": True,
                "status": payload.get("status", "running"),
                "events": events[after:],
                "next_after": len(events),
                "result": payload.get("result"),
                "error": payload.get("error"),
            }
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return {
                    "success": False,
                    "status": "not_found",
                    "error": "Unknown job id.",
                    "hint": "Start a new job and poll using the returned job_id.",
                }
            return {
                "success": True,
                "status": job.status,
                "events": [event.__dict__ for event in job.events[after:]],
                "next_after": len(job.events),
                "result": job.result,
                "error": job.error,
            }

    def get_graph_payload(self, session_id: str) -> dict[str, Any]:
        session = self.get_or_create_session(session_id)
        if not session.hydrated:
            return {"nodes": [], "edges": []}
        self.resources.ensure_initialized()
        return self.resources.graph_payload or {"nodes": [], "edges": []}

    def _create_job(self, kind: str, session_id: str) -> JobRecord:
        job_id = f"{kind}-{uuid.uuid4().hex[:20]}"
        job = JobRecord(job_id=job_id, kind=kind, session_id=session_id)
        if self.state_store:
            self.state_store.save_json(
                "jobs",
                job_id,
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "session_id": job.session_id,
                    "status": job.status,
                    "result": job.result,
                    "error": job.error,
                    "events": [],
                },
            )
            return job
        with self._lock:
            self.jobs[job_id] = job
        return job

    def _add_event(self, job_id: str, component: str, data: str) -> None:
        if self.state_store:
            payload = self.state_store.load_json("jobs", job_id)
            if payload is None:
                return
            payload.setdefault("events", []).append(
                JobEvent(timestamp=now_iso(), component=component, data=data).__dict__
            )
            self.state_store.save_json("jobs", job_id, payload)
            return
        with self._lock:
            self.jobs[job_id].events.append(JobEvent(timestamp=now_iso(), component=component, data=data))

    def _run_job(self, job_id: str, runner: Callable[[], dict[str, Any]]) -> None:
        try:
            result = runner()
            if self.state_store:
                payload = self.state_store.load_json("jobs", job_id) or {}
                payload["status"] = "completed"
                payload["result"] = result
                payload["error"] = None
                self.state_store.save_json("jobs", job_id, payload)
                return
            with self._lock:
                self.jobs[job_id].status = "completed"
                self.jobs[job_id].result = result
        except Exception as exc:  # pragma: no cover
            if self.state_store:
                payload = self.state_store.load_json("jobs", job_id) or {}
                payload["status"] = "failed"
                payload["error"] = str(exc)
                payload["result"] = {"status": str(exc)}
                self.state_store.save_json("jobs", job_id, payload)
                self._add_event(job_id, "Error", str(exc))
                return
            with self._lock:
                self.jobs[job_id].status = "failed"
                self.jobs[job_id].error = str(exc)
                self.jobs[job_id].result = {"status": str(exc)}
            self._add_event(job_id, "Error", str(exc))

    def _invoke_async_job(self, payload: dict[str, Any]) -> None:
        assert self.lambda_client is not None
        assert self.lambda_function_name
        self.lambda_client.invoke(
            FunctionName=self.lambda_function_name,
            InvocationType="Event",
            Payload=json.dumps({"adi_async_job": payload}).encode("utf-8"),
        )

    def run_async_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = str(payload["job_id"])
        kind = str(payload["kind"])
        session_id = str(payload["session_id"])
        if kind == "demo_hydrate":
            self._run_job(job_id, lambda: self._run_demo_hydrate_pipeline(job_id, session_id))
        elif kind == "chat":
            self._run_job(
                job_id,
                lambda: self._run_chat_pipeline(
                    job_id,
                    session_id,
                    str(payload["query"]),
                    list(payload.get("conversation_history") or []),
                ),
            )
        else:
            raise RuntimeError(f"Unsupported async job kind: {kind}")
        return {"ok": True, "job_id": job_id, "kind": kind}

    def _run_demo_hydrate_pipeline(self, job_id: str, session_id: str) -> dict[str, Any]:
        self.resources.ensure_initialized()
        self._add_event(job_id, "Session", "Initializing fixed Microsoft FY2025 demo corpus...")
        self._add_event(job_id, "Corpus", "Precomputed document, graph, and SQL assets loaded from local artifacts.")
        session = self.get_or_create_session(session_id)
        session.hydrated = True
        self._save_session(session)
        self._add_event(job_id, "GraphRAG", f"Validated graph ready with {self.resources.graph_payload['graph_summary']['node_count']} nodes and {self.resources.graph_payload['graph_summary']['edge_count']} edges.")
        self._add_event(job_id, "SQL", "SQLite analyst demo dataset ready for read-only querying.")
        self._add_event(job_id, "Orchestrator", "Demo Hydration Complete.")
        return self.get_session_status(session_id)

    def _run_chat_pipeline(
        self,
        job_id: str,
        session_id: str,
        query: str,
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        session = self.get_or_create_session(session_id)
        if not session.hydrated:
            raise RuntimeError("Load Demo Experience before asking questions.")
        if session.questions_remaining <= 0:
            raise RuntimeError("Question limit reached for this session.")
        self.resources.ensure_openai_client()
        result = self._run_query_pipeline(job_id, query, conversation_history)
        if result.get("status") != "blocked":
            session.questions_used += 1
        self._save_session(session)
        return result

    def _run_query_pipeline(
        self,
        job_id: str,
        query: str,
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        resources = self.resources
        assert resources.openai_client is not None

        safe_history = sanitize_conversation_history(conversation_history)
        guardrail_result = inspect_query(
            query,
            client=resources.openai_client,
            model=DEFAULT_GUARDRAIL_MODEL,
        )
        sanitized_query = guardrail_result["sanitized_query"] or query.strip()
        if guardrail_result["pii_findings"]:
            self._add_event(
                job_id,
                "Guardrails",
                f"Redacted {len(guardrail_result['pii_findings'])} sensitive item(s) from the user input before retrieval.",
            )
        if guardrail_result["blocked"]:
            self._add_event(
                job_id,
                "Guardrails",
                f"Blocked request. reason={guardrail_result['reason']} attacks={','.join(guardrail_result['attack_types']) or 'none'}",
            )
            return build_guardrail_block_response(query, sanitized_query, safe_history, guardrail_result)

        if sanitized_query != query.strip():
            self._add_event(job_id, "Guardrails", "Accepted query after privacy-preserving redaction.")
        else:
            self._add_event(job_id, "Guardrails", "User query accepted for multi-source RAG execution.")

        meta_intent = detect_conversation_meta_intent(sanitized_query)
        if meta_intent:
            self._add_event(job_id, "Conversation Memory", f"Detected conversation-meta query: {meta_intent}.")
            self._add_event(job_id, "Orchestrator", "Answered directly from chat history without corpus retrieval.")
            response = build_conversation_meta_response(sanitized_query, meta_intent, safe_history)
            response["original_query"] = query
            response["resolved_query"] = sanitized_query
            response["guardrails"] = build_guardrail_ui_payload(guardrail_result)
            return response

        resources.ensure_initialized()
        assert resources.pinecone_client is not None
        assert resources.index is not None
        assert resources.schema_package is not None
        assert resources.sql_capability_summary is not None
        assert resources.graph_capability_summary is not None
        assert resources.child_index is not None
        assert resources.parent_index is not None
        assert resources.chunk_to_record_id is not None

        conversation_resolution = resolve_conversational_query(
            sanitized_query,
            safe_history,
            model=DEFAULT_PIPELINE_MODEL,
            client=resources.openai_client,
            max_recent_turns=DEFAULT_CONVERSATION_MEMORY_TURNS,
        )
        resolved_query = conversation_resolution["resolved_query"]
        if conversation_resolution["used_history"]:
            self._add_event(
                job_id,
                "Conversation Memory",
                (
                    f"Resolved follow-up query using {len(conversation_resolution['referenced_turn_ids'])} prior turn(s): "
                    f"{resolved_query}"
                ),
            )
        else:
            self._add_event(job_id, "Conversation Memory", "Current query was treated as standalone.")

        transformed_bundle = build_minimal_orchestration_bundle(
            resolved_query,
            model=DEFAULT_PIPELINE_MODEL,
            openai_client=resources.openai_client,
        )
        decomposition = transformed_bundle.get("decomposition_result", {})
        sub_queries = decomposition.get("sub_queries", [])
        self._add_event(job_id, "Query Decomposition", f"Prepared {len(sub_queries)} sub-queries: " + " | ".join(sub_queries))

        routing_plan = build_multi_source_routing_plan(
            transformed_bundle,
            resources.sql_capability_summary,
            resources.graph_capability_summary,
            model=DEFAULT_PIPELINE_MODEL,
            client=resources.openai_client,
        )
        routing_descriptions = []
        for plan in routing_plan.get("sub_query_plans", []):
            sources = ", ".join(plan["routing_decision"]["selected_sources"])
            routing_descriptions.append(f"{plan['sub_query_id']} -> {sources}")
        self._add_event(job_id, "Source Router", "Routing selected: " + "; ".join(routing_descriptions))

        policy_plan = build_latency_optimized_policy(
            resolved_query,
            transformed_bundle,
            routing_plan,
            model=DEFAULT_PIPELINE_MODEL,
            client=resources.openai_client,
        )
        policy_descriptions = []
        for plan in policy_plan.get("sub_query_execution_plans", []):
            execution_policy = plan["execution_policy"]
            policy_descriptions.append(
                f"{plan['sub_query_id']} profile={execution_policy['vector_profile']} sources={','.join(execution_policy['active_sources'])}"
            )
        self._add_event(job_id, "Latency Policy", "Execution plan: " + "; ".join(policy_descriptions))

        orchestration_result = execute_latency_optimized_orchestration(
            query=resolved_query,
            transformed_bundle=transformed_bundle,
            routing_plan=routing_plan,
            policy_plan=policy_plan,
            schema_package=resources.schema_package,
            sql_database_path=resources.runtime_app_root / DEFAULT_SQL_DATABASE_PATH,
            graph_database_path=resources.runtime_app_root / DEFAULT_GRAPH_DATABASE_PATH,
            index=resources.index,
            namespace=DEFAULT_NAMESPACE,
            alpha=0.6,
            top_k=6,
            openai_client=resources.openai_client,
            pinecone_client=resources.pinecone_client,
            child_index=resources.child_index,
            parent_index=resources.parent_index,
            chunk_to_record_id=resources.chunk_to_record_id,
            model=DEFAULT_PIPELINE_MODEL,
        )
        self._add_event(job_id, "Retriever", f"Execution summary: {json.dumps(orchestration_result['execution_summary'])}")

        fused_bundle = fuse_cross_source_evidence(orchestration_result)
        self._add_event(job_id, "Evidence Fusion", f"Fused {fused_bundle['bundle_summary']['fact_count']} facts across {fused_bundle['bundle_summary']['sub_query_count']} sub-queries.")
        fused_bundle = attach_contextual_compression(
            fused_bundle,
            model=DEFAULT_COMPRESSION_MODEL,
            client=resources.openai_client,
            min_facts_for_compression=DEFAULT_MIN_FACTS_FOR_COMPRESSION,
        )
        compression_context = fused_bundle.get("compression_context", {})
        self._add_event(
            job_id,
            "Context Compression",
            (
                f"Applied={compression_context.get('applied', False)}"
                f" model={compression_context.get('compression_model')}"
                f" cost={compression_context.get('usage', {}).get('estimated_cost_usd', 0.0)}"
            ),
        )

        retry_state = {
            "total_rounds_used": 0,
            "pipeline_reruns_used": 0,
            "answer_regenerations_used": 0,
            "repairs_used": 0,
            "actions_taken": [],
            "last_judge_average_score": None,
        }
        cycle = self._run_answer_cycle(job_id, fused_bundle)
        history = [
            {
                "round": 0,
                "action": "initial_generation",
                "judge_average_score": cycle["judge_result"]["average_score"],
                "judge_verdict": cycle["judge_result"]["metrics"]["overall_verdict"],
                "critique_needs_correction": cycle["critique_result"]["final_critique"]["needs_correction"],
            }
        ]
        retry_state["last_judge_average_score"] = cycle["judge_result"]["average_score"]

        while True:
            decision = decide_runtime_action(
                cycle["critique_result"],
                cycle["judge_result"],
                retry_state,
                max_total_rounds=DEFAULT_MAX_TOTAL_ROUNDS,
                max_pipeline_reruns=DEFAULT_MAX_PIPELINE_RERUNS,
                max_answer_regenerations=DEFAULT_MAX_ANSWER_REGENERATIONS,
                max_repairs=DEFAULT_MAX_REPAIRS,
            )
            self._add_event(job_id, "Runtime Gate", f"Action={decision['action']} | Reason={decision['reason']}")

            if decision["action"] in {"stop_accept", "stop_best_effort"}:
                return build_ui_answer_payload(
                    query=query,
                    resolved_query=resolved_query,
                    conversation_resolution=conversation_resolution,
                    transformed_bundle=transformed_bundle,
                    fused_bundle=fused_bundle,
                    answer_result=cycle["answer_result"],
                    judge_result=cycle["judge_result"],
                    critique_result=cycle["critique_result"],
                    runtime_history=history,
                    termination=decision,
                    guardrails=guardrail_result,
                )

            retry_state["total_rounds_used"] += 1
            retry_state["actions_taken"].append(decision["action"])

            if decision["action"] == "answer_regeneration_only":
                retry_state["answer_regenerations_used"] += 1
                cycle = self._run_answer_cycle(job_id, fused_bundle)
            elif decision["action"] in {"targeted_answer_repair", "citation_strict_repair"}:
                retry_state["repairs_used"] += 1
                self._add_event(job_id, "Repair", "Running targeted grounded-answer repair.")
                repair_result = repair_grounded_answer(
                    fused_bundle,
                    cycle["answer_result"],
                    cycle["critique_result"],
                    model=DEFAULT_REPAIR_MODEL,
                    client=resources.openai_client,
                )
                answer_result = repair_result["repaired_answer"] if repair_result["repair_applied"] else cycle["answer_result"]
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    critique_future = executor.submit(
                        critique_grounded_answer,
                        fused_bundle,
                        answer_result,
                        DEFAULT_CRITIQUE_MODEL,
                        resources.openai_client,
                    )
                    judge_future = executor.submit(
                        judge_final_answer,
                        fused_bundle,
                        answer_result,
                        None,
                        DEFAULT_JUDGE_MODEL,
                        resources.openai_client,
                    )
                    critique_result = critique_future.result()
                    judge_result = judge_future.result()
                self._add_event(job_id, "Repair", f"Repair strategy={repair_result['repair_strategy']['strategy']} applied={repair_result['repair_applied']}")
                cycle = {
                    "answer_result": answer_result,
                    "critique_result": critique_result,
                    "judge_result": judge_result,
                }
            elif decision["action"] == "full_pipeline_rerun_once":
                retry_state["pipeline_reruns_used"] += 1
                self._add_event(job_id, "Retriever", "Re-running the full pipeline once under bounded retry policy.")
                orchestration_result = execute_latency_optimized_orchestration(
                    query=resolved_query,
                    transformed_bundle=transformed_bundle,
                    routing_plan=routing_plan,
                    policy_plan=policy_plan,
                    schema_package=resources.schema_package,
                    sql_database_path=resources.runtime_app_root / DEFAULT_SQL_DATABASE_PATH,
                    graph_database_path=resources.runtime_app_root / DEFAULT_GRAPH_DATABASE_PATH,
                    index=resources.index,
                    namespace=DEFAULT_NAMESPACE,
                    alpha=0.6,
                    top_k=6,
                    openai_client=resources.openai_client,
                    pinecone_client=resources.pinecone_client,
                    child_index=resources.child_index,
                    parent_index=resources.parent_index,
                    chunk_to_record_id=resources.chunk_to_record_id,
                    model=DEFAULT_PIPELINE_MODEL,
                )
                fused_bundle = fuse_cross_source_evidence(orchestration_result)
                cycle = self._run_answer_cycle(job_id, fused_bundle)

            retry_state["last_judge_average_score"] = cycle["judge_result"]["average_score"]
            history.append(
                {
                    "round": retry_state["total_rounds_used"],
                    "action": decision["action"],
                    "reason": decision["reason"],
                    "judge_average_score": cycle["judge_result"]["average_score"],
                    "judge_verdict": cycle["judge_result"]["metrics"]["overall_verdict"],
                    "critique_needs_correction": cycle["critique_result"]["final_critique"]["needs_correction"],
                }
            )

            # loop continues until runtime gate stops

        raise RuntimeError("Runtime quality gating loop exited unexpectedly.")

    def _run_answer_cycle(self, job_id: str, fused_bundle: dict[str, Any]) -> dict[str, Any]:
        resources = self.resources
        assert resources.openai_client is not None
        self._add_event(job_id, "Answer Generator", "Generating grounded answer from fused evidence.")
        answer_result = generate_grounded_answer(fused_bundle, model=DEFAULT_ANSWER_MODEL, client=resources.openai_client)
        self._add_event(job_id, "Self-Reflective RAG", "Critiquing the answer for grounding, coverage, and citation integrity.")
        self._add_event(job_id, "LLM Judge", "Scoring the answer with RAGAS-style metrics.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            critique_future = executor.submit(
                critique_grounded_answer,
                fused_bundle,
                answer_result,
                DEFAULT_CRITIQUE_MODEL,
                resources.openai_client,
            )
            judge_future = executor.submit(
                judge_final_answer,
                fused_bundle,
                answer_result,
                None,
                DEFAULT_JUDGE_MODEL,
                resources.openai_client,
            )
            critique_result = critique_future.result()
            judge_result = judge_future.result()
        return {
            "answer_result": answer_result,
            "critique_result": critique_result,
            "judge_result": judge_result,
        }


def build_ui_answer_payload(
    query: str,
    resolved_query: str,
    conversation_resolution: dict[str, Any],
    transformed_bundle: dict[str, Any],
    fused_bundle: dict[str, Any],
    answer_result: dict[str, Any],
    judge_result: dict[str, Any],
    critique_result: dict[str, Any],
    runtime_history: list[dict[str, Any]],
    termination: dict[str, Any],
    guardrails: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer_text, citations = build_ui_citations_and_answer(fused_bundle, answer_result)
    metrics = judge_result["metrics"]
    decomposition = transformed_bundle.get("decomposition_result", {})
    payload = {
        "answer": answer_text,
        "citations": citations,
        "evaluation": {
            "faithfulness": round(float(metrics["faithfulness"]) / 5.0, 3),
            "answer_relevancy": round(float(metrics["answer_relevancy"]) / 5.0, 3),
            "context_precision": round(float(metrics["context_precision"]) / 5.0, 3),
            "citation_grounding": round(float(metrics["citation_grounding"]) / 5.0, 3),
            "overall_badge": metrics["overall_verdict"].upper(),
            "reasoning": metrics["summary"],
            "strengths": metrics.get("strengths", []),
            "weaknesses": metrics.get("weaknesses", []),
            "recommendations": metrics.get("recommendations", []),
        },
        "evaluation_pending": False,
        "truncated_subqueries": bool(decomposition.get("truncated_subqueries", False)),
        "subquery_cap": int(decomposition.get("subquery_cap", MAX_SUB_QUERIES)),
        "judge_average_score": judge_result["average_score"],
        "runtime_history": runtime_history,
        "termination": termination,
        "critique": critique_result.get("final_critique", {}),
        "sources_used": sorted(
            {
                fact["source_type"]
                for fact in fused_bundle.get("normalized_facts", [])
                if fact["fact_id"] in answer_result.get("used_fact_ids", [])
            }
        ),
        "status": "ok",
        "original_query": query,
        "resolved_query": resolved_query,
        "conversation_resolution": conversation_resolution,
        "guardrails": build_guardrail_ui_payload(guardrails or {}),
    }
    return payload


def build_ui_citations_and_answer(fused_bundle: dict[str, Any], answer_result: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    facts_by_id = {fact["fact_id"]: fact for fact in fused_bundle.get("normalized_facts", [])}
    answer_markdown = answer_result.get("answer_markdown", "")
    ordered_fact_ids: list[str] = []
    seen = set()
    for match in FACT_ID_PATTERN.finditer(answer_markdown):
        fact_id = match.group(1)
        if fact_id in seen:
            continue
        seen.add(fact_id)
        ordered_fact_ids.append(fact_id)

    normalized_answer = answer_markdown
    citations = []
    for index, fact_id in enumerate(ordered_fact_ids, start=1):
        display_label = f"[Evidence {index}]"
        normalized_answer = normalized_answer.replace(f"[{fact_id}]", display_label)
        citations.append(build_ui_citation(index, display_label, fact_id, facts_by_id.get(fact_id, {})))

    return normalized_answer, citations


def build_ui_citation(index: int, display_label: str, fact_id: str, fact: dict[str, Any]) -> dict[str, Any]:
    source_type = fact.get("source_type")
    if source_type == "vector_document":
        citation = fact.get("citation", {})
        return {
            "index": index,
            "source_type": "internal_document",
            "display_label": display_label,
            "title": fact.get("summary", "") or citation.get("section_title", "Document evidence"),
            "source_file": citation.get("source_file"),
            "page_number": citation.get("page"),
            "section": citation.get("section_title"),
            "chunk_text": fact.get("child_text") or fact.get("summary"),
            "fact_id": fact_id,
        }

    if source_type == "graph_relationships":
        page_ranges = fact.get("page_ranges") or [{}]
        snippets = fact.get("evidence_snippets", [])
        return {
            "index": index,
            "source_type": "graph_entity",
            "display_label": display_label,
            "title": fact.get("summary", "Graph evidence"),
            "entity_names": [name for name in [fact.get("source_entity"), fact.get("target_entity"), fact.get("entity_name")] if name],
            "relationship_type": fact.get("relation_type"),
            "section": ", ".join(fact.get("section_titles", [])[:2]),
            "page_number": page_ranges[0].get("page_start"),
            "chunk_text": "\n".join(snippets[:2]),
            "fact_id": fact_id,
        }

    if source_type == "sql_structured":
        row = fact.get("row", {})
        return {
            "index": index,
            "source_type": "database",
            "display_label": display_label,
            "title": fact.get("summary", "Structured data row"),
            "table_name": (fact.get("target_tables") or [None])[0],
            "sql_query": fact.get("validated_sql"),
            "chunk_text": ", ".join(f"{key}={value}" for key, value in row.items()),
            "fact_id": fact_id,
        }

    return {
        "index": index,
        "source_type": "internal_document",
        "display_label": display_label,
        "title": fact.get("summary", fact_id),
        "chunk_text": fact.get("summary", ""),
        "fact_id": fact_id,
    }


def build_demo_asset_catalog(runtime_app_root: Path, corpus_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}

    for document in corpus_manifest.get("documents", []):
        relative_path = Path(document["relative_path"])
        asset_id = f"document:{document['document_id']}"
        catalog[asset_id] = {
            "asset_id": asset_id,
            "label": document["display_name"],
            "path": runtime_app_root / relative_path,
            "filename": relative_path.name,
            "media_type": mimetypes.guess_type(relative_path.name)[0] or "application/octet-stream",
            "kind": "document",
        }

    for dataset in corpus_manifest.get("datasets", []):
        for table in dataset.get("tables", []):
            relative_path = Path(table["relative_path"])
            asset_id = f"dataset:{table['table_name']}"
            catalog[asset_id] = {
                "asset_id": asset_id,
                "label": table["table_name"],
                "path": runtime_app_root / relative_path,
                "filename": relative_path.name,
                "media_type": mimetypes.guess_type(relative_path.name)[0] or "text/csv",
                "kind": "dataset",
            }

    return catalog
