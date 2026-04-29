"""
Application entry point.

Startup sequence:
1. Configure structured logging.
2. Start Prometheus metrics server.
3. Initialise the database (create tables if not exist).
4. Register all tool modules (side-effect of importing them).
5. Start FastAPI with middleware.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.routes import router
from src.config import get_settings
from src.memory.long_term import LongTermMemory
from src.services.monitoring import configure_logging, start_metrics_server
from src.services.rate_limiter import RateLimitMiddleware

# Import tool modules to trigger @tool_registry.register() decorators
import src.tools.customer_tools  # noqa: F401
import src.tools.internal_tools  # noqa: F401
import src.tools.search_tools    # noqa: F401

log = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    cfg = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("app.starting", environment=cfg.environment.value)
        start_metrics_server()
        db = LongTermMemory()
        await db.init_db()
        log.info("app.started")
        yield
        log.info("app.shutting_down")

    app = FastAPI(
        title="AI Support Agent",
        description="Production-grade multi-agent customer support system.",
        version="1.0.0",
        docs_url="/docs" if not cfg.is_production else None,  # Disable Swagger in prod
        redoc_url=None,
        lifespan=lifespan,
    )

    # --- Rate limiting ---
    limiter = RateLimitMiddleware.build()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not cfg.is_production else ["https://yourdomain.com"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # --- Request ID middleware ---
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # --- Global error handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error("unhandled_exception", error=str(exc), path=request.url.path, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # --- Routes ---
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "src.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=not cfg.is_production,
        log_level=cfg.log_level.lower(),
        workers=1,  # Use multiple workers in production behind a load balancer
    )
