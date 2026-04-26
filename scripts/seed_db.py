"""
Seed the long-term database with realistic initial data for testing.

Creates:
- Sample conversations (one customer, one internal)
- Customer facts (tier, preferences)
- Sample interaction log entries

Run: python scripts/seed_db.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.long_term import LongTermMemory

SAMPLE_USER_ID = "usr_demo_001"
SAMPLE_SESSION_CUSTOMER = "sess_demo_customer_001"
SAMPLE_SESSION_INTERNAL = "sess_demo_internal_001"


async def main():
    db = LongTermMemory()
    print("Initialising database tables...")
    await db.init_db()

    # --- Customer conversation ---
    print("Creating demo customer conversation...")
    await db.create_conversation(
        session_id=SAMPLE_SESSION_CUSTOMER,
        agent_type="customer_support",
        user_id=SAMPLE_USER_ID,
    )
    await db.log_interaction(
        session_id=SAMPLE_SESSION_CUSTOMER,
        role="user",
        content="Hi, I need help with my order ORD-55123. It's been 10 days and hasn't arrived.",
        agent_type="customer_support",
        input_tokens=0, output_tokens=0, cost_usd=0,
    )
    await db.log_interaction(
        session_id=SAMPLE_SESSION_CUSTOMER,
        role="assistant",
        content=(
            "Hi! I checked your order ORD-55123 — it's currently out for delivery and should "
            "arrive today. The tracking number is TRK892341 with FedEx. "
            "Is there anything else I can help with?"
        ),
        agent_type="customer_support",
        input_tokens=312, output_tokens=68, cost_usd=0.00098,
    )
    await db.update_summary(
        session_id=SAMPLE_SESSION_CUSTOMER,
        summary="Customer enquired about delayed order ORD-55123. Status confirmed as out-for-delivery. No escalation needed.",
    )

    # --- Internal ops conversation ---
    print("Creating demo internal conversation...")
    await db.create_conversation(
        session_id=SAMPLE_SESSION_INTERNAL,
        agent_type="internal_ops",
        user_id="agent_alice@company.com",
    )
    await db.log_interaction(
        session_id=SAMPLE_SESSION_INTERNAL,
        role="user",
        content="Show me all P1 tickets currently breaching SLA.",
        agent_type="internal_ops",
        input_tokens=0, output_tokens=0, cost_usd=0,
    )
    await db.log_interaction(
        session_id=SAMPLE_SESSION_INTERNAL,
        role="assistant",
        content=(
            "Found 1 P1 ticket breaching SLA:\n\n"
            "- **TKT-11111** — 'Payment gateway timeout' | Breached by 45 min | Unassigned\n\n"
            "2 tickets at risk (P2, <2h remaining):\n"
            "- TKT-22222 — 'Login broken for SSO users' | 30 min left\n"
            "- TKT-33333 — 'Wrong item shipped' | 95 min left\n\n"
            "Recommend immediate assignment for TKT-11111."
        ),
        agent_type="internal_ops",
        input_tokens=287, output_tokens=112, cost_usd=0.00154,
    )

    # --- Customer facts ---
    print("Seeding customer facts for demo user...")
    facts = {
        "account_tier": "pro",
        "preferred_contact": "email",
        "lifetime_orders": "14",
        "member_since": "2023-03-10",
        "language": "en-US",
        "last_issue_type": "shipping_delay",
    }
    for key, value in facts.items():
        await db.upsert_fact(
            user_id=SAMPLE_USER_ID,
            key=key,
            value=value,
            source_session=SAMPLE_SESSION_CUSTOMER,
            confidence=0.95,
        )

    print(f"\nDone.")
    print(f"  Demo customer session : {SAMPLE_SESSION_CUSTOMER}")
    print(f"  Demo internal session : {SAMPLE_SESSION_INTERNAL}")
    print(f"  Demo user ID          : {SAMPLE_USER_ID}")
    print(f"\nUse session IDs above with POST /api/v1/chat to continue existing conversations.")


if __name__ == "__main__":
    asyncio.run(main())
