# AI Support Agent — Production-Grade Multi-Agent System

A production-ready multi-agent AI system for customer support and internal operations, built with Claude (Anthropic) or GPT-4 (OpenAI), FastAPI, Redis, and SQLite/PostgreSQL.

---

## Architecture Overview

```
                          ┌─────────────────────────────────────────────┐
                          │              FastAPI Application             │
                          │  ┌──────────────┐    ┌──────────────────┐   │
  Customer ───── POST /chat──▶│   Routes     │    │  Rate Limiter    │   │
  Internal ── POST /internal──▶ + Middleware │    │  Response Cache  │   │
                          │  └──────┬───────┘    └──────────────────┘   │
                          └─────────┼───────────────────────────────────┘
                                    │
                          ┌─────────▼───────────┐
                          │     Orchestrator     │  ← Routes requests
                          │  (LLM classification │     to correct agent
                          │   + sticky routing)  │
                          └──────┬───────────────┘
                ┌────────────────┼─────────────────────┐
                │                │                     │
     ┌──────────▼──┐    ┌────────▼────┐    ┌──────────▼──────┐
     │  Customer   │    │  Internal   │    │  Knowledge Base │
     │  Support    │    │  Ops Agent  │    │  Agent          │
     │  Agent      │    │             │    │                 │
     └──────┬──────┘    └──────┬──────┘    └──────┬──────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Escalation Agent  │  ← High-severity cases
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Tool Registry    │
                    │  ┌───┐ ┌───┐ ┌───┐  │
                    │  │T1 │ │T2 │ │T3 │  │  T1: Order/Refund/Product
                    │  └───┘ └───┘ └───┘  │  T2: Tickets/SLA/Teams
                    └──────────┬──────────┘  T3: KB Search / Memory
                               │
              ┌────────────────┼───────────────┐
     ┌────────▼───┐   ┌────────▼────┐   ┌──────▼──────┐
     │  Short-Term│   │  Long-Term  │   │   Vector    │
     │  Memory    │   │  Memory     │   │   Store     │
     │  (Redis)   │   │  (SQLite/PG)│   │  (numpy)   │
     └────────────┘   └─────────────┘   └─────────────┘
```

### Agent Roles

| Agent | Responsibility | Model Tier | Tools |
|-------|---------------|------------|-------|
| **Orchestrator** | Route requests, detect topic shifts, fallback chain | light | — |
| **CustomerSupport** | Front-line: orders, refunds, product info | standard | 7 tools |
| **InternalOps** | Tickets, SLAs, team coordination | standard | 7 tools |
| **KnowledgeBase** | Policy retrieval, semantic search | light | 3 tools |
| **Escalation** | Human handoff, P1/P2/P3 triage | heavy | 5 tools |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/yourorg/ai-support-agent
cd ai-support-agent
cp .env.example .env
# Edit .env: add your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### 2. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs  
Metrics: http://localhost:9090

### 3. Run locally with `just` (recommended)

This project uses [`just`](https://github.com/casey/just) as a command runner for common tasks.

**Install just:**
```bash
cargo install just  # or: brew install just  |  scoop install just
```

**Usage:**
```bash
just --list        # Show all available recipes
just setup         # First-time setup: .env, venv, dependencies
just seed          # Initialize DB and seed knowledge base + sample data
just dev           # Start the API server locally
just test          # Run the test suite
just docker-up     # Start all services with Docker Compose
just logs          # View live logs for all services
```

**Key recipes:**

| Recipe | Description |
|--------|-------------|
| `setup` | First-time setup: creates `.env`, venv, installs dependencies |
| `seed` | Initialize DB tables and seed KB + sample data |
| `dev` | Start API server on http://localhost:8000 |
| `test` | Run pytest suite |
| `test-unit` | Run only fast unit tests (skip integration) |
| `docker-up` | Start all services with Docker Compose |
| `docker-down` | Stop all Docker Compose services |
| `logs` | View live logs for all services |
| `logs-api` | View logs for API service only |
| `seed-docker` | Seed KB and DB data inside running Docker container |
| `reset` | Reset local data (DB + embeddings) and re-seed |
| `tools` | List all registered tools |
| `cost` | Show current API cost usage (today) |

**Quick curl demos:**
```bash
just curl-health     # Check API health
just curl-chat       # Send a customer chat message
just curl-internal   # Send an internal query
just curl-escalate   # Trigger an escalation (legal keyword)
```

### 4. Run locally (manual)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/seed_kb.py        # Seed knowledge base
python -m src.main
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Usage Examples

### Customer conversation

```bash
# Start a new session
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need to return my order ORD-12345"}'

# Continue the session
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "It arrived damaged",
    "session_id": "uuid-from-previous-response"
  }'
```

### Internal operations

```bash
curl -X POST http://localhost:8000/api/v1/internal/query \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: your-secret-key" \
  -d '{"message": "Show me all P1 tickets breaching SLA"}'
```

### Knowledge base ingestion

```bash
curl -X POST http://localhost:8000/api/v1/admin/kb/ingest \
  -H "X-Internal-Token: your-secret-key" \
  -d '{
    "documents": [
      {
        "id": "policy-001",
        "text": "Our return window is 30 days for all products.",
        "metadata": {"category": "returns", "last_updated": "2025-04-01"}
      }
    ]
  }'
```

---

## Project Structure

```
ai-support-agent/
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Pydantic settings from .env
│   ├── agents/
│   │   ├── base_agent.py       # ReAct loop, context compression, guardrails
│   │   ├── orchestrator.py     # Multi-agent routing and coordination
│   │   ├── customer_support.py # Customer-facing agent
│   │   ├── internal_ops.py     # Internal workflows agent
│   │   ├── knowledge_base.py   # Document retrieval agent
│   │   └── escalation.py       # Human handoff agent
│   ├── tools/
│   │   ├── registry.py         # Tool registration, execution, error boundary
│   │   ├── customer_tools.py   # Order, refund, product, escalation tools
│   │   ├── internal_tools.py   # Ticket, SLA, audit tools
│   │   └── search_tools.py     # KB search, web search, memory recall
│   ├── memory/
│   │   ├── short_term.py       # Redis-backed conversation history
│   │   ├── long_term.py        # SQLAlchemy persistent memory + facts
│   │   └── vector_store.py     # Semantic search (numpy, swappable)
│   ├── llm/
│   │   ├── client.py           # Anthropic/OpenAI abstraction + cost tracking
│   │   └── prompts.py          # System prompt templates
│   ├── api/
│   │   └── routes.py           # FastAPI routes + middleware
│   └── services/
│       ├── rate_limiter.py     # Token bucket + Redis rate limiting
│       ├── cache.py            # Response cache (exact-match)
│       └── monitoring.py       # Prometheus metrics + structlog
├── tests/
│   ├── test_agents.py          # Agent unit + integration tests
│   ├── test_api.py             # API endpoint tests
│   └── conftest.py
├── scripts/
│   ├── seed_kb.py              # Seed knowledge base documents
│   └── prometheus.yml          # Prometheus scrape config
├── data/
│   ├── sqlite/                 # SQLite database
│   └── embeddings/             # Vector store files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## Adding a New Tool

Tools are registered with a decorator — no boilerplate needed:

```python
# src/tools/my_tools.py
from src.tools.registry import tool_registry

@tool_registry.register(
    name="check_subscription",
    description="Check a customer's subscription tier and renewal date.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
        },
        "required": ["customer_id"],
    },
    timeout_seconds=10.0,
)
async def check_subscription(customer_id: str) -> dict:
    # Call your real API here
    return {"tier": "pro", "renewal_date": "2026-01-01"}
```

Then:
1. Import the module in `src/main.py` (to trigger decorator registration).
2. Add `"check_subscription"` to the `tool_names` list of the relevant agent.

---

## Senior Engineering Notes

### Reducing Hallucinations
- **Tool-first policy**: agents are instructed to call tools before answering factual questions.
- **Hedge detection**: responses containing uncertainty phrases get a soft disclaimer.
- **Source citations**: KB agent always cites the source document and date.
- **Temperature 0.2**: lower temperature reduces creative confabulation.

### Context Management
- **Sliding window**: Redis lists keep the N most recent messages per session.
- **Automatic compression**: when context exceeds 80% of token budget, the oldest half is summarised by the light model and replaced with a compact summary.
- **Persistent facts**: structured facts extracted from conversations are stored in the DB and injected into future system prompts.

### Cost Optimisation
- **Tiered models**: light (routing, summaries) → standard (most tasks) → heavy (escalation only).
- **Response caching**: identical KB queries hit Redis instead of the LLM.
- **Daily budget guard**: hard stop when daily spend exceeds `COST_BUDGET_DAILY_USD`.
- **Token budgets**: `max_tokens_per_request` limits per-call cost.

### Multi-Agent Collaboration
- **Synchronous delegation**: Orchestrator → Agent → sub-agent calls when needed.
- **Sticky routing**: after the first turn, routing persists to avoid re-routing identical sessions.
- **Fallback chain**: agent failure → knowledge_base → escalation (never shows raw errors to users).

### Production Readiness
- **Rate limiting**: per-IP (slowapi) + per-user (Redis token bucket).
- **Structured logging**: JSON in production (structlog + Prometheus).
- **Health endpoint**: Redis + DB + LLM configuration checks.
- **Graceful degradation**: Redis down → in-process fallback; DB down → log-only mode.
- **Non-root Docker**: container runs as `appuser` (UID 1000).
- **Request ID tracing**: every request gets a UUID propagated through all log lines.

### Guardrails
- Hard-coded keyword escalation (no LLM needed for legal/threat language).
- Internal tool mutation (refunds > $200, ticket creation) always go through `log_action`.
- Customer-facing responses strip internal system names from output.
- Max iteration guard prevents runaway tool loops (cost + latency protection).

---

## Scaling to Production

| Concern | Current | Production upgrade |
|---------|---------|-------------------|
| Database | SQLite | PostgreSQL + pgvector |
| Vector search | numpy flat index | Qdrant / Pinecone / pgvector |
| Caching | Redis | Redis Cluster |
| Auth | Static token | OAuth2 / JWT |
| LLM provider | Single | Fallback chain (Anthropic → OpenAI) |
| Deployment | Docker Compose | Kubernetes + HPA |
| Secrets | .env file | AWS Secrets Manager / Vault |
| Monitoring | Prometheus + structlog | OpenTelemetry + Datadog/Grafana |

---

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `LLM_PROVIDER` | `anthropic` or `openai` | `anthropic` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `DATABASE_URL` | SQLAlchemy async URL | SQLite |
| `COST_BUDGET_DAILY_USD` | Hard daily LLM spend cap | `50.0` |
| `MAX_AGENT_ITERATIONS` | ReAct loop iteration limit | `10` |
| `RATE_LIMIT_PER_MINUTE` | Requests per minute per IP | `60` |
| `ENVIRONMENT` | `development`, `staging`, `production` | `development` |
# ai-support-agent
