# ─────────────────────────────────────────────────────────────────────────────
# AI Support Agent — task runner (https://github.com/casey/just)
#
# Install just:   cargo install just  |  brew install just  |  scoop install just
# Usage:          just <recipe>
#                 just --list         # show all recipes
# ─────────────────────────────────────────────────────────────────────────────

set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true           # auto-loads .env if present

# Paths
VENV     := ".venv"
PYTHON   := VENV + "/bin/python"
PIP      := VENV + "/bin/pip"
DATA_DIR := "data"

# Colours (no-op on Windows if your shell doesn't support ANSI)
GREEN  := "\\033[92m"
YELLOW := "\\033[93m"
RESET  := "\\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT: show help
# ─────────────────────────────────────────────────────────────────────────────

[private]
default:
    @just --list

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

# First-time setup: copy .env, create venv, install dependencies
setup:
    @echo -e "{{GREEN}}▶ Setting up project…{{RESET}}"
    @if [ ! -f .env ]; then \
        cp .env.example .env; \
        echo -e "{{YELLOW}}  .env created from .env.example — add your ANTHROPIC_API_KEY{{RESET}}"; \
    else \
        echo "  .env already exists, skipping."; \
    fi
    @if [ ! -d "{{VENV}}" ]; then \
        python3 -m venv "{{VENV}}"; \
        echo "  Virtual environment created at {{VENV}}/"; \
    fi
    "{{PIP}}" install --quiet --upgrade pip
    "{{PIP}}" install --quiet -r requirements.txt
    mkdir -p "{{DATA_DIR}}/sqlite" "{{DATA_DIR}}/embeddings"
    @echo -e "{{GREEN}}✔ Setup complete. Next: edit .env, then run: just seed{{RESET}}"

# Install / refresh Python dependencies only
deps:
    "{{PIP}}" install --quiet -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE & SEED
# ─────────────────────────────────────────────────────────────────────────────

# Create database tables (idempotent)
db-init:
    @echo -e "{{GREEN}}▶ Initialising database…{{RESET}}"
    "{{PYTHON}}" -c "
import asyncio, sys
sys.path.insert(0, '.')
from src.memory.long_term import LongTermMemory
asyncio.run(LongTermMemory().init_db())
print('  Tables created (or already exist).')
"

# Seed knowledge base documents into the vector store
seed-kb:
    @echo -e "{{GREEN}}▶ Seeding knowledge base…{{RESET}}"
    "{{PYTHON}}" scripts/seed_kb.py

# Seed knowledge base from scratch (wipes existing index first)
seed-kb-clean:
    @echo -e "{{YELLOW}}▶ Clearing and re-seeding knowledge base…{{RESET}}"
    "{{PYTHON}}" scripts/seed_kb.py --clear

# Seed sample conversations and customer facts into the database
seed-db:
    @echo -e "{{GREEN}}▶ Seeding sample database data…{{RESET}}"
    "{{PYTHON}}" scripts/seed_db.py

# Full seed: init DB + seed KB + seed sample data
seed: db-init seed-kb seed-db
    @echo -e "{{GREEN}}✔ All seed data loaded. Run: just dev{{RESET}}"

# ─────────────────────────────────────────────────────────────────────────────
# RUNNING (LOCAL)
# ─────────────────────────────────────────────────────────────────────────────

# Start the API server locally (requires Redis — use just docker-redis first)
dev:
    @echo -e "{{GREEN}}▶ Starting API server on http://localhost:8000{{RESET}}"
    @echo "  Docs: http://localhost:8000/docs"
    "{{PYTHON}}" -m uvicorn src.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info

# Start only Redis in Docker (for local dev without full compose)
redis:
    @echo -e "{{GREEN}}▶ Starting Redis container…{{RESET}}"
    docker run -d --rm \
        --name ai-support-redis \
        -p 6379:6379 \
        redis:7-alpine redis-server --appendonly yes \
    || echo "  Redis already running."

# Stop the standalone Redis container
redis-stop:
    docker stop ai-support-redis 2>/dev/null || echo "Redis not running."

# ─────────────────────────────────────────────────────────────────────────────
# RUNNING (DOCKER COMPOSE)
# ─────────────────────────────────────────────────────────────────────────────

# Build and start all services with Docker Compose
docker-up:
    @echo -e "{{GREEN}}▶ Starting all services with Docker Compose…{{RESET}}"
    docker compose up --build -d
    @echo ""
    @echo "  API     : http://localhost:8000"
    @echo "  Docs    : http://localhost:8000/docs"
    @echo "  Metrics : http://localhost:9090"
    @echo ""
    @echo "  Tip: run 'just seed-docker' to load seed data into the container."

# Stop all Docker Compose services
docker-down:
    docker compose down

# Stop and remove volumes (full reset)
docker-reset:
    @echo -e "{{YELLOW}}▶ Stopping Docker Compose and removing volumes…{{RESET}}"
    docker compose down -v
    @echo "  All containers and volumes removed."

# View live logs for all services
logs:
    docker compose logs -f

# View logs for the API service only
logs-api:
    docker compose logs -f api

# ─────────────────────────────────────────────────────────────────────────────
# SEED INTO RUNNING DOCKER CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

# Seed KB and DB data inside the running Docker container
seed-docker:
    @echo -e "{{GREEN}}▶ Seeding data inside Docker container…{{RESET}}"
    docker compose exec api python scripts/seed_kb.py
    docker compose exec api python scripts/seed_db.py
    @echo -e "{{GREEN}}✔ Docker seed complete.{{RESET}}"

# Seed KB only inside Docker
seed-docker-kb:
    docker compose exec api python scripts/seed_kb.py

# ─────────────────────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────────────────────

# Run the full pytest suite
test:
    @echo -e "{{GREEN}}▶ Running test suite…{{RESET}}"
    "{{PYTHON}}" -m pytest tests/ -v --tb=short

# Run tests and show coverage
test-cov:
    "{{PYTHON}}" -m pip install --quiet pytest-cov
    "{{PYTHON}}" -m pytest tests/ -v --cov=src --cov-report=term-missing --tb=short

# Run only fast unit tests (skip integration tests)
test-unit:
    "{{PYTHON}}" -m pytest tests/ -v -k "not Integration" --tb=short

# Run smoke tests against the running API (local)
smoke:
    @echo -e "{{GREEN}}▶ Running smoke tests against http://localhost:8000…{{RESET}}"
    "{{PYTHON}}" scripts/smoke_test.py --base-url http://localhost:8000

# Run smoke tests against a custom URL
smoke-url URL:
    "{{PYTHON}}" scripts/smoke_test.py --base-url "{{URL}}"

# ─────────────────────────────────────────────────────────────────────────────
# QUICK CURL DEMOS (for manual testing without smoke_test.py)
# ─────────────────────────────────────────────────────────────────────────────

# Check API health
curl-health:
    curl -s http://localhost:8000/api/v1/health | python3 -m json.tool

# Send a customer chat message
curl-chat MSG="Hi, where is my order ORD-12345?":
    curl -s -X POST http://localhost:8000/api/v1/chat \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"{{MSG}}\"}" | python3 -m json.tool

# Send an internal query (uses API_SECRET_KEY from .env)
curl-internal MSG="Show me all P1 tickets breaching SLA":
    curl -s -X POST http://localhost:8000/api/v1/internal/query \
        -H "Content-Type: application/json" \
        -H "X-Internal-Token: ${API_SECRET_KEY:-change-me}" \
        -d "{\"message\": \"{{MSG}}\"}" | python3 -m json.tool

# Trigger an escalation (uses legal keyword)
curl-escalate:
    curl -s -X POST http://localhost:8000/api/v1/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "This is unacceptable. I am contacting my lawyer about this data breach."}' \
        | python3 -m json.tool

# Get metrics summary (internal)
curl-metrics:
    curl -s http://localhost:8000/api/v1/metrics/summary \
        -H "X-Internal-Token: ${API_SECRET_KEY:-change-me}" \
        | python3 -m json.tool

# ─────────────────────────────────────────────────────────────────────────────
# MAINTENANCE
# ─────────────────────────────────────────────────────────────────────────────

# Reset local data (DB + embeddings) and re-seed from scratch
reset: db-init seed-kb-clean seed-db
    @echo -e "{{GREEN}}✔ Local data reset and re-seeded.{{RESET}}"

# Remove generated files (venv, data, __pycache__)
clean:
    @echo -e "{{YELLOW}}▶ Cleaning generated files…{{RESET}}"
    rm -rf "{{VENV}}" "{{DATA_DIR}}/sqlite" "{{DATA_DIR}}/embeddings"
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    @echo "  Done."

# Show current API cost usage (today)
cost:
    @"{{PYTHON}}" -c "
import asyncio, sys
sys.path.insert(0, '.')
from src.llm.client import LLMClient
from src.memory.long_term import LongTermMemory
async def show():
    llm_cost = await LLMClient.get_daily_cost()
    db_cost  = await LongTermMemory().total_cost_today()
    print(f'  LLM cost today (in-process tracker): \${llm_cost:.4f}')
    print(f'  DB-logged cost today                : \${db_cost:.4f}')
asyncio.run(show())
"

# List all registered tools
tools:
    @"{{PYTHON}}" -c "
import sys; sys.path.insert(0, '.')
import src.tools.customer_tools, src.tools.internal_tools, src.tools.search_tools
from src.tools.registry import tool_registry
tools = tool_registry.list_all()
print(f'Registered tools ({len(tools)}):')
for t in sorted(tools):
    print(f'  - {t}')
"
