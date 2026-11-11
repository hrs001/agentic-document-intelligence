from __future__ import annotations

from mangum import Mangum

from .main import app


mangum_handler = Mangum(app, lifespan="off")


def handler(event, context):
    async_payload = event.get("adi_async_job") if isinstance(event, dict) else None
    if async_payload:
        return app.state.runtime.run_async_job(async_payload)
    return mangum_handler(event, context)
