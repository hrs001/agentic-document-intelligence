from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .service import DemoAppService


class ChatJobRequest(BaseModel):
    query: str
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)


def create_app(service: DemoAppService | None = None) -> FastAPI:
    app = FastAPI(title="Agentic Document Intelligence API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = service

    def get_runtime() -> DemoAppService:
        runtime = app.state.runtime
        if runtime is None:
            runtime = DemoAppService()
            app.state.runtime = runtime
        return runtime

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/v1/session-status")
    def session_status(session_id: str = Query(...)) -> dict[str, Any]:
        return get_runtime().get_session_status(session_id)

    @app.post("/api/v1/demo-hydrate/jobs")
    def demo_hydrate(session_id: str = Query(...)) -> dict[str, Any]:
        return {"success": True, "job_id": get_runtime().create_demo_hydrate_job(session_id)}

    @app.post("/api/v1/chat/jobs")
    def chat_job(payload: ChatJobRequest, session_id: str = Query(...)) -> dict[str, Any]:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty.")
        return {
            "success": True,
            "job_id": get_runtime().create_chat_job(
                session_id,
                payload.query.strip(),
                payload.conversation_history,
            ),
        }

    @app.get("/api/v1/jobs/{job_id}")
    def job_status(job_id: str, after: int = Query(0, ge=0)) -> dict[str, Any]:
        result = get_runtime().get_job_status(job_id, after)
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/api/v1/graph")
    def graph(session_id: str = Query(...)) -> dict[str, Any]:
        return get_runtime().get_graph_payload(session_id)

    @app.get("/api/v1/demo-assets/{asset_id:path}")
    def demo_asset(asset_id: str) -> FileResponse:
        try:
            asset = get_runtime().get_demo_asset(asset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown demo asset: {asset_id}") from exc
        return FileResponse(
            path=asset["path"],
            media_type=asset["media_type"],
            filename=asset["filename"],
        )

    return app


app = create_app()
