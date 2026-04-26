"""
Observability: structured logging + Prometheus metrics.

Metrics exposed:
- agent_requests_total         (counter, labels: agent_type, status)
- agent_latency_seconds        (histogram, labels: agent_type)
- agent_tool_calls_total       (counter, labels: tool_name, status)
- llm_tokens_total             (counter, labels: model, type=input|output)
- llm_cost_usd_total           (counter, labels: model)
- rate_limit_exceeded_total    (counter, labels: identifier)
- cache_hits_total             (counter, labels: agent_type)

Structured logging:
- Uses structlog with JSON renderer in production.
- Human-readable console renderer in development.
"""
from __future__ import annotations

import logging
import sys

import structlog
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.config import Environment, get_settings

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
AGENT_REQUESTS = Counter(
    "agent_requests_total",
    "Total agent requests",
    ["agent_type", "status"],
)

AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Agent response latency",
    ["agent_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"],
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    ["model", "token_type"],
)

LLM_COST = Counter(
    "llm_cost_usd_total",
    "Total LLM cost in USD",
    ["model"],
)

RATE_LIMIT_EXCEEDED = Counter(
    "rate_limit_exceeded_total",
    "Rate limit exceeded events",
    ["identifier_type"],
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Cache hit events",
    ["agent_type"],
)

ACTIVE_SESSIONS = Gauge(
    "active_sessions",
    "Currently active support sessions",
)

DAILY_COST_USD = Gauge(
    "daily_llm_cost_usd",
    "Running daily LLM cost in USD",
)


def record_agent_request(
    agent_type: str,
    status: str,  # "success" | "error" | "escalated"
    latency_seconds: float,
) -> None:
    AGENT_REQUESTS.labels(agent_type=agent_type, status=status).inc()
    AGENT_LATENCY.labels(agent_type=agent_type).observe(latency_seconds)


def record_tool_call(tool_name: str, success: bool) -> None:
    TOOL_CALLS.labels(tool_name=tool_name, status="success" if success else "error").inc()


def record_llm_usage(model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    LLM_TOKENS.labels(model=model, token_type="input").inc(input_tokens)
    LLM_TOKENS.labels(model=model, token_type="output").inc(output_tokens)
    LLM_COST.labels(model=model).inc(cost_usd)


# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------
def configure_logging() -> None:
    cfg = get_settings()

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if cfg.environment == Environment.PRODUCTION:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        stream=sys.stdout,
        format="%(message)s",
    )


def start_metrics_server() -> None:
    cfg = get_settings()
    try:
        start_http_server(cfg.prometheus_port)
        structlog.get_logger(__name__).info(
            "monitoring.prometheus_started",
            port=cfg.prometheus_port,
        )
    except Exception as exc:
        structlog.get_logger(__name__).warning(
            "monitoring.prometheus_failed",
            error=str(exc),
        )
