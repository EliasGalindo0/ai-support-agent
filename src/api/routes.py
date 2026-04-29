"""
FastAPI route definitions.

Endpoints:
- POST /chat              : Customer-facing conversation endpoint.
- POST /internal/query    : Internal ops query (authenticated).
- GET  /health            : Health check (liveness + readiness).
- GET  /metrics/summary   : Cost and usage summary (internal only).
- POST /admin/kb/ingest   : Ingest a document into the knowledge base.
- DELETE /admin/session   : Clear a session's short-term memory.
"""
from __future__ import annotations

import secrets
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.agents.orchestrator import Orchestrator
from src.llm.client import LLMClient
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.memory.vector_store import VectorDocument, VectorStore
from src.services.cache import ResponseCache
from src.services.monitoring import (
    ACTIVE_SESSIONS,
    record_agent_request,
)

log = structlog.get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------
_llm_client = LLMClient()
_short_term = ShortTermMemory()
_long_term = LongTermMemory()
_cache = ResponseCache()
_kb_store = VectorStore(namespace="knowledge_base")


def get_orchestrator() -> Orchestrator:
    return Orchestrator(
        llm_client=_llm_client,
        short_term=_short_term,
        long_term=_long_term,
    )


def _require_internal_token(x_internal_token: str = Header(default="")):
    """Simple internal API token check. Replace with proper auth (OAuth2, JWT)."""
    from src.config import get_settings
    expected = get_settings().api_secret_key
    # Use compare_digest to prevent timing-based token enumeration attacks.
    if not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=403, detail="Forbidden")
    return x_internal_token


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "I need help tracking my order ORD-12345.",
                "user_email": "customer@example.com",
            }
        }
    )

    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        description="Omit to start a new session",
    )
    user_id: str | None = None
    user_email: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    agent_type: str
    escalated: bool = False
    cost_usd: float | None = None   # Omit in production customer responses
    latency_ms: float | None = None


class InternalQueryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str | None = None
    user_id: str | None = None
    force_agent: str | None = Field(
        default=None,
        description="Force a specific agent: internal_ops | knowledge_base | escalation",
    )


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    components: dict[str, str]


class KBIngestRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(
        ...,
        description="List of {id, text, metadata} objects",
    )


# ---------------------------------------------------------------------------
# Customer chat
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse, tags=["Customer"])
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """
    Main customer conversation endpoint.

    - Creates a new session if session_id is not provided.
    - Routes to the appropriate agent automatically.
    - Caches KB responses for cost efficiency.
    """
    session_id = request.session_id or str(uuid.uuid4())
    ACTIVE_SESSIONS.inc()

    # Check cache first (KB namespace — only KB responses are cached)
    cached = await _cache.get(request.message, namespace="kb")
    if cached:
        ACTIVE_SESSIONS.dec()
        return ChatResponse(
            session_id=session_id,
            response=cached,
            agent_type="cached",
        )

    try:
        agent_response = await orchestrator.route(
            user_message=request.message,
            session_id=session_id,
            user_id=request.user_id,
            user_email=request.user_email,
        )

        # Background: persist conversation metadata
        background_tasks.add_task(
            _persist_conversation,
            session_id=session_id,
            agent_type=agent_response.agent_type,
            user_id=request.user_id,
        )

        record_agent_request(
            agent_type=agent_response.agent_type,
            status="escalated" if agent_response.escalated else "success",
            latency_seconds=agent_response.latency_ms / 1000,
        )

        # Opportunistically cache
        await _cache.set(
            agent_type=agent_response.agent_type,
            message=request.message,
            response=agent_response.text,
        )

        return ChatResponse(
            session_id=session_id,
            response=agent_response.text,
            agent_type=agent_response.agent_type,
            escalated=agent_response.escalated,
            # Don't expose cost to customers; include in internal logging
        )

    except Exception as exc:
        log.error("api.chat.error", error=str(exc), session_id=session_id)
        record_agent_request("unknown", "error", 0.0)
        raise HTTPException(status_code=500, detail="An internal error occurred.")
    finally:
        ACTIVE_SESSIONS.dec()


# ---------------------------------------------------------------------------
# Internal query
# ---------------------------------------------------------------------------
@router.post("/internal/query", response_model=ChatResponse, tags=["Internal"])
async def internal_query(
    request: InternalQueryRequest,
    _: str = Depends(_require_internal_token),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Internal operations query — authenticated endpoint."""
    session_id = request.session_id or str(uuid.uuid4())

    agent_response = await orchestrator.route(
        user_message=request.message,
        session_id=session_id,
        user_id=request.user_id,
        force_agent=request.force_agent or "internal_ops",
    )

    return ChatResponse(
        session_id=session_id,
        response=agent_response.text,
        agent_type=agent_response.agent_type,
        escalated=agent_response.escalated,
        cost_usd=round(agent_response.cost_usd, 6),
        latency_ms=round(agent_response.latency_ms, 1),
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    """Liveness + readiness health check."""
    components: dict[str, str] = {}

    # Check Redis
    try:
        import asyncio
        import redis.asyncio as aioredis
        from src.config import get_settings
        r = aioredis.from_url(get_settings().redis_url)
        await asyncio.wait_for(r.ping(), timeout=1.5)
        components["redis"] = "ok"
        await r.aclose()
    except Exception:
        components["redis"] = "unavailable"

    # Check DB (just verify engine connects)
    try:
        await _long_term.conversation_count_today()
        components["database"] = "ok"
    except Exception:
        components["database"] = "unavailable"

    # Check LLM (don't make a real API call — just verify config)
    from src.config import get_settings
    cfg = get_settings()
    components["llm"] = "configured" if cfg.anthropic_api_key or cfg.openai_api_key else "not_configured"

    overall = "ok" if all(v in ("ok", "configured") for v in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components)


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------
@router.get("/metrics/summary", tags=["Operations"])
async def metrics_summary(
    _: str = Depends(_require_internal_token),
):
    """Cost and usage summary for internal dashboards."""
    daily_cost = await _long_term.total_cost_today()
    conversation_count = await _long_term.conversation_count_today()
    llm_daily = await LLMClient.get_daily_cost()

    return {
        "today": {
            "conversations": conversation_count,
            "llm_cost_usd": round(llm_daily, 4),
            "db_cost_usd": round(daily_cost, 4),
        },
        "vector_store": {
            "knowledge_base_docs": _kb_store.count(),
        },
    }


# ---------------------------------------------------------------------------
# Knowledge base ingestion
# ---------------------------------------------------------------------------
@router.post("/admin/kb/ingest", tags=["Admin"])
async def ingest_kb(
    request: KBIngestRequest,
    _: str = Depends(_require_internal_token),
):
    """Ingest documents into the knowledge base vector store."""
    docs = [
        VectorDocument(
            id=d.get("id", str(uuid.uuid4())),
            text=d["text"],
            metadata=d.get("metadata", {}),
        )
        for d in request.documents
        if d.get("text")
    ]

    await _kb_store.add(docs)
    await _cache.invalidate("kb")  # Invalidate KB response cache

    return {"ingested": len(docs), "total_docs": _kb_store.count()}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
@router.delete("/admin/session/{session_id}", tags=["Admin"])
async def clear_session(
    session_id: str,
    _: str = Depends(_require_internal_token),
):
    """Clear a session's short-term memory (e.g. for GDPR erasure)."""
    await _short_term.clear_session(session_id)
    return {"cleared": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------
async def _persist_conversation(
    session_id: str,
    agent_type: str,
    user_id: str | None,
) -> None:
    try:
        existing = await _long_term.get_conversation(session_id)
        if not existing:
            await _long_term.create_conversation(
                session_id=session_id,
                agent_type=agent_type,
                user_id=user_id,
            )
    except Exception as exc:
        log.error("api.persist_conversation.error", error=str(exc))
