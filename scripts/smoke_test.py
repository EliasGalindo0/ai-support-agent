"""
Smoke test: sends realistic requests against the running API and prints results.

Requires the API to be running (just dev or just docker-up).
Run: python scripts/smoke_test.py [--base-url http://localhost:8000]
"""
import argparse
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
BLUE  = "\033[94m"
RESET = "\033[0m"
BOLD  = "\033[1m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
INFO = f"{BLUE}INFO{RESET}"


def _print_header(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


def _print_result(label: str, ok: bool, detail: str = "") -> None:
    badge = PASS if ok else FAIL
    print(f"  [{badge}] {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {YELLOW}{line}{RESET}")


async def run_smoke(base_url: str, secret: str) -> int:
    failures = 0
    headers_internal = {"X-Internal-Token": secret}

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:

        # ── 1. Health check ────────────────────────────────────────────────
        _print_header("1 · Health check")
        r = await client.get("/api/v1/health")
        ok = r.status_code == 200
        data = r.json()
        _print_result("GET /health returns 200", ok, json.dumps(data, indent=2))
        if not ok:
            failures += 1

        # ── 2. Customer chat — new session ─────────────────────────────────
        _print_header("2 · Customer chat (new session)")
        r = await client.post("/api/v1/chat", json={
            "message": "Hi, I need help tracking my order ORD-99123.",
            "user_email": "demo@example.com",
        })
        ok = r.status_code == 200
        data = r.json()
        session_id = data.get("session_id", "")
        _print_result("POST /chat returns 200", ok)
        _print_result("Response has session_id", bool(session_id))
        _print_result("Response has text", bool(data.get("response")))
        print(f"\n         {YELLOW}Agent   : {data.get('agent_type')}{RESET}")
        print(f"         {YELLOW}Response: {data.get('response', '')[:120]}...{RESET}")
        if not ok:
            failures += 1

        # ── 3. Continue the session ────────────────────────────────────────
        _print_header("3 · Continue existing session (sticky routing)")
        r = await client.post("/api/v1/chat", json={
            "message": "Can I get a refund for it? It arrived damaged.",
            "session_id": session_id,
            "user_email": "demo@example.com",
        })
        ok = r.status_code == 200
        data2 = r.json()
        same_session = data2.get("session_id") == session_id
        _print_result("POST /chat continues same session", same_session)
        _print_result("Returns 200", ok)
        print(f"\n         {YELLOW}Response: {data2.get('response', '')[:120]}...{RESET}")
        if not ok or not same_session:
            failures += 1

        # ── 4. Escalation trigger ──────────────────────────────────────────
        _print_header("4 · Escalation trigger (legal keyword bypass)")
        r = await client.post("/api/v1/chat", json={
            "message": "This is unacceptable. I'm going to file a chargeback and contact my lawyer.",
            "user_email": "angry@example.com",
        })
        ok = r.status_code == 200
        data3 = r.json()
        is_escalation = data3.get("agent_type") == "escalation" or data3.get("escalated") is True
        _print_result("Returns 200", ok)
        _print_result("Routed to escalation agent", is_escalation,
                      f"agent_type={data3.get('agent_type')}, escalated={data3.get('escalated')}")
        if not ok:
            failures += 1

        # ── 5. Internal ops ────────────────────────────────────────────────
        _print_header("5 · Internal ops query (authenticated)")
        r = await client.post("/api/v1/internal/query",
            json={"message": "Show me all P1 tickets breaching SLA right now."},
            headers=headers_internal,
        )
        ok = r.status_code == 200
        data4 = r.json()
        _print_result("POST /internal/query returns 200", ok)
        _print_result("Agent is internal_ops", data4.get("agent_type") == "internal_ops",
                      f"agent_type={data4.get('agent_type')}")
        print(f"\n         {YELLOW}Response: {data4.get('response', '')[:120]}...{RESET}")
        if not ok:
            failures += 1

        # ── 6. Auth guard ──────────────────────────────────────────────────
        _print_header("6 · Internal ops without token (auth guard)")
        r = await client.post("/api/v1/internal/query",
            json={"message": "Show tickets"},
        )
        _print_result("Returns 403 without token", r.status_code == 403,
                      f"status={r.status_code}")
        if r.status_code != 403:
            failures += 1

        # ── 7. KB ingestion ────────────────────────────────────────────────
        _print_header("7 · Knowledge base ingestion")
        r = await client.post("/api/v1/admin/kb/ingest",
            json={"documents": [
                {
                    "id": "smoke-test-doc-001",
                    "text": "Smoke test policy: This document was created during smoke testing and can be safely deleted.",
                    "metadata": {"category": "test", "last_updated": "2025-01-01"},
                }
            ]},
            headers=headers_internal,
        )
        ok = r.status_code == 200
        data5 = r.json()
        _print_result("POST /admin/kb/ingest returns 200", ok)
        _print_result("Reports ingested count", data5.get("ingested", 0) == 1,
                      f"ingested={data5.get('ingested')}, total={data5.get('total_docs')}")
        if not ok:
            failures += 1

        # ── 8. Metrics summary ─────────────────────────────────────────────
        _print_header("8 · Metrics summary")
        r = await client.get("/api/v1/metrics/summary", headers=headers_internal)
        ok = r.status_code == 200
        data6 = r.json()
        _print_result("GET /metrics/summary returns 200", ok)
        _print_result("Contains today.conversations", "today" in data6)
        print(f"\n         {YELLOW}{json.dumps(data6, indent=2)}{RESET}")
        if not ok:
            failures += 1

        # ── 9. Validation guard ────────────────────────────────────────────
        _print_header("9 · Input validation (empty message)")
        r = await client.post("/api/v1/chat", json={"message": ""})
        _print_result("Returns 422 for empty message", r.status_code == 422,
                      f"status={r.status_code}")
        if r.status_code != 422:
            failures += 1

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    total = 9
    passed = total - failures  # approximate — some checks are compound
    if failures == 0:
        print(f"{GREEN}{BOLD}  All smoke tests passed.{RESET}")
    else:
        print(f"{RED}{BOLD}  {failures} section(s) had failures. Check output above.{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}\n")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test the AI Support Agent API.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--secret", default=os.getenv("API_SECRET_KEY", "change-me"),
                        help="Internal API secret key")
    args = parser.parse_args()

    print(f"\n{BOLD}AI Support Agent — Smoke Test{RESET}")
    print(f"Base URL : {args.base_url}")
    print(f"Secret   : {'*' * len(args.secret)}")

    exit_code = asyncio.run(run_smoke(args.base_url, args.secret))
    sys.exit(exit_code)
