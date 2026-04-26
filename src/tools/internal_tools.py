"""
Internal operations tools available to the internal_ops agent.

These tools power:
- Ticket management workflows.
- SLA tracking and alerting.
- Team coordination.
- Audit logging.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from src.tools.registry import tool_registry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Ticket management
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="create_ticket",
    description="Create a new internal support ticket.",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "priority": {
                "type": "string",
                "enum": ["P1", "P2", "P3", "P4"],
            },
            "category": {
                "type": "string",
                "enum": ["billing", "technical", "account", "feedback", "other"],
            },
            "assigned_to": {"type": "string", "description": "Agent ID or team queue"},
            "customer_email": {"type": "string"},
        },
        "required": ["title", "description", "priority", "category"],
    },
)
async def create_ticket(
    title: str,
    description: str,
    priority: str,
    category: str,
    assigned_to: str = "unassigned",
    customer_email: str = "",
) -> dict[str, Any]:
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    # --- Replace with Zendesk/Jira/ServiceNow API ---
    return {
        "ticket_id": ticket_id,
        "title": title,
        "priority": priority,
        "category": category,
        "status": "open",
        "assigned_to": assigned_to,
        "created_at": _now_iso(),
        "sla_deadline": (
            datetime.now(timezone.utc) + {
                "P1": timedelta(hours=1),
                "P2": timedelta(hours=4),
                "P3": timedelta(hours=24),
                "P4": timedelta(days=3),
            }.get(priority, timedelta(days=3))
        ).isoformat(),
    }


@tool_registry.register(
    name="update_ticket",
    description="Update the status, assignment, or notes of an existing ticket.",
    parameters={
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "pending_customer", "resolved", "closed"],
            },
            "notes": {"type": "string"},
            "assigned_to": {"type": "string"},
        },
        "required": ["ticket_id"],
    },
)
async def update_ticket(
    ticket_id: str,
    status: str | None = None,
    notes: str = "",
    assigned_to: str | None = None,
) -> dict[str, Any]:
    # --- Replace with real ticketing API ---
    return {
        "ticket_id": ticket_id,
        "updated": True,
        "changes": {
            k: v for k, v in {
                "status": status,
                "notes": notes or None,
                "assigned_to": assigned_to,
            }.items() if v is not None
        },
        "updated_at": _now_iso(),
    }


@tool_registry.register(
    name="list_open_tickets",
    description="List open tickets filtered by priority, category, or assignee.",
    parameters={
        "type": "object",
        "properties": {
            "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            "assigned_to": {"type": "string"},
            "category": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": [],
    },
)
async def list_open_tickets(
    priority: str | None = None,
    assigned_to: str | None = None,
    category: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    # --- Replace with real query ---
    tickets = [
        {
            "ticket_id": f"TKT-{random.randint(10000,99999)}",
            "title": f"Sample ticket {i}",
            "priority": random.choice(["P1", "P2", "P3"]),
            "status": "open",
            "category": random.choice(["billing", "technical", "account"]),
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=random.randint(1,48))).isoformat(),
        }
        for i in range(min(limit, 5))
    ]
    return {"tickets": tickets, "total": len(tickets)}


# ---------------------------------------------------------------------------
# SLA tracking
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="check_sla_status",
    description="Check the SLA status of tickets. Returns breached and at-risk tickets.",
    parameters={
        "type": "object",
        "properties": {
            "queue": {"type": "string", "description": "Team queue name"},
        },
        "required": [],
    },
)
async def check_sla_status(queue: str = "all") -> dict[str, Any]:
    # --- Replace with real SLA engine ---
    return {
        "queue": queue,
        "breached": [
            {"ticket_id": "TKT-11111", "title": "Payment failed", "breached_by_minutes": 45, "priority": "P1"},
        ],
        "at_risk": [
            {"ticket_id": "TKT-22222", "title": "Login issue", "minutes_remaining": 30, "priority": "P2"},
            {"ticket_id": "TKT-33333", "title": "Wrong item received", "minutes_remaining": 95, "priority": "P2"},
        ],
        "healthy": 23,
        "total_open": 26,
        "breach_rate_24h": 0.04,
    }


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="log_action",
    description=(
        "Write an entry to the immutable audit log. "
        "MUST be called for every significant agent action (refunds, escalations, status changes)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action taken"},
            "actor": {"type": "string", "description": "Agent or user who performed it"},
            "resource_type": {"type": "string", "description": "e.g. 'order', 'ticket', 'customer'"},
            "resource_id": {"type": "string"},
            "details": {"type": "object", "description": "Arbitrary structured details"},
        },
        "required": ["action", "actor", "resource_type", "resource_id"],
    },
)
async def log_action(
    action: str,
    actor: str,
    resource_type: str,
    resource_id: str,
    details: dict | None = None,
) -> dict[str, Any]:
    audit_id = str(uuid.uuid4())
    entry = {
        "audit_id": audit_id,
        "timestamp": _now_iso(),
        "action": action,
        "actor": actor,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "details": details or {},
    }
    # --- In production, write to immutable store (CloudTrail, BigQuery, etc.) ---
    return {"logged": True, "audit_id": audit_id}


# ---------------------------------------------------------------------------
# Team availability
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="get_team_availability",
    description="Check which support agents are currently available to handle escalations.",
    parameters={
        "type": "object",
        "properties": {
            "tier": {
                "type": "string",
                "enum": ["tier1", "tier2", "tier3", "manager"],
                "description": "Agent tier required",
            },
        },
        "required": ["tier"],
    },
)
async def get_team_availability(tier: str) -> dict[str, Any]:
    # --- Replace with real workforce management API ---
    available = random.randint(0, 5)
    return {
        "tier": tier,
        "available_agents": available,
        "estimated_wait_minutes": max(0, (5 - available) * 8),
        "agents": [
            {"id": f"agent_{i:03d}", "name": f"Agent {i}", "load": random.randint(1, 5)}
            for i in range(available)
        ],
    }
