"""
Customer-facing tools available to the customer support agent.

Each tool is a realistic stub with the shape of a real integration.
To connect to your actual backend, replace the stub body with the real API call.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from src.tools.registry import tool_registry

# ---------------------------------------------------------------------------
# Order lookup
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="get_order_status",
    description=(
        "Look up the current status and details of a customer order by order ID. "
        "Returns status, items, estimated delivery, and tracking info."
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID (e.g. ORD-12345)",
            },
            "customer_email": {
                "type": "string",
                "description": "Customer email to verify ownership of the order",
            },
        },
        "required": ["order_id"],
    },
)
async def get_order_status(order_id: str, customer_email: str = "") -> dict[str, Any]:
    # --- Replace with real OMS API call ---
    statuses = ["processing", "shipped", "out_for_delivery", "delivered", "cancelled"]
    status = random.choice(statuses)
    eta = (datetime.now(timezone.utc) + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d")
    return {
        "order_id": order_id,
        "status": status,
        "items": [
            {"sku": "PROD-001", "name": "Widget Pro", "qty": 2, "unit_price": 29.99},
        ],
        "total_usd": 59.98,
        "estimated_delivery": eta if status not in ("delivered", "cancelled") else None,
        "tracking_number": f"TRK{random.randint(100000, 999999)}" if status in ("shipped", "out_for_delivery") else None,
        "carrier": "FedEx" if status in ("shipped", "out_for_delivery") else None,
    }


# ---------------------------------------------------------------------------
# Refund
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="process_refund",
    description=(
        "Initiate a refund for an order. "
        "Requires order_id and reason. "
        "Maximum automatic refund is $200; larger amounts require escalation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Order to refund"},
            "amount_usd": {
                "type": "number",
                "description": "Amount to refund in USD. Must be <= 200.",
            },
            "reason": {
                "type": "string",
                "enum": [
                    "damaged_item",
                    "wrong_item",
                    "not_received",
                    "changed_mind",
                    "quality_issue",
                    "other",
                ],
                "description": "Reason code for the refund",
            },
            "notes": {"type": "string", "description": "Optional freeform notes"},
        },
        "required": ["order_id", "amount_usd", "reason"],
    },
)
async def process_refund(
    order_id: str,
    amount_usd: float,
    reason: str,
    notes: str = "",
) -> dict[str, Any]:
    if amount_usd > 200:
        return {
            "success": False,
            "error": "Amount exceeds automatic refund limit of $200. Escalation required.",
            "requires_escalation": True,
        }
    # --- Replace with real payment/OMS API call ---
    refund_id = f"REF-{random.randint(10000, 99999)}"
    return {
        "success": True,
        "refund_id": refund_id,
        "order_id": order_id,
        "amount_usd": amount_usd,
        "reason": reason,
        "status": "initiated",
        "eta_days": 3,
        "message": f"Refund of ${amount_usd:.2f} initiated. Reference: {refund_id}. Credit in 3-5 business days.",
    }


# ---------------------------------------------------------------------------
# Product info
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="get_product_info",
    description="Get product details including specs, pricing, and availability.",
    parameters={
        "type": "object",
        "properties": {
            "product_id": {"type": "string"},
            "query": {
                "type": "string",
                "description": "Specific info to retrieve, e.g. 'warranty' or 'compatibility'",
            },
        },
        "required": ["product_id"],
    },
)
async def get_product_info(product_id: str, query: str = "") -> dict[str, Any]:
    # --- Replace with real product catalogue API ---
    products = {
        "PROD-001": {
            "name": "Widget Pro",
            "price_usd": 29.99,
            "in_stock": True,
            "stock_count": 145,
            "specs": {"weight": "250g", "dimensions": "10x5x3cm", "warranty": "2 years"},
            "categories": ["widgets", "pro"],
        },
        "PROD-002": {
            "name": "Gadget Basic",
            "price_usd": 14.99,
            "in_stock": False,
            "stock_count": 0,
            "restock_eta": "2025-06-01",
            "specs": {"weight": "100g", "dimensions": "5x3x2cm", "warranty": "1 year"},
            "categories": ["gadgets", "basic"],
        },
    }
    product = products.get(product_id)
    if not product:
        return {"found": False, "product_id": product_id}
    return {"found": True, "product_id": product_id, **product}


# ---------------------------------------------------------------------------
# Customer history
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="get_customer_history",
    description=(
        "Retrieve a customer's order history and support ticket history. "
        "Use this to personalise responses and understand account status."
    ),
    parameters={
        "type": "object",
        "properties": {
            "customer_email": {"type": "string"},
            "limit": {
                "type": "integer",
                "description": "Number of most recent records to return (default 5)",
                "default": 5,
            },
        },
        "required": ["customer_email"],
    },
)
async def get_customer_history(customer_email: str, limit: int = 5) -> dict[str, Any]:
    # --- Replace with real CRM API call ---
    return {
        "customer_email": customer_email,
        "account_tier": random.choice(["free", "pro", "enterprise"]),
        "member_since": "2023-01-15",
        "lifetime_value_usd": round(random.uniform(50, 2000), 2),
        "recent_orders": [
            {"order_id": f"ORD-{random.randint(10000,99999)}", "date": "2025-04-10", "total": 59.98, "status": "delivered"},
            {"order_id": f"ORD-{random.randint(10000,99999)}", "date": "2025-03-02", "total": 14.99, "status": "refunded"},
        ][:limit],
        "open_tickets": 0,
        "satisfaction_score": round(random.uniform(3.5, 5.0), 1),
    }


# ---------------------------------------------------------------------------
# Escalate to human
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="escalate_to_human",
    description=(
        "Escalate the conversation to a human support agent. "
        "Use when the issue is complex, the customer is very unhappy, "
        "or automated resolution has failed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "reason": {"type": "string", "description": "Why escalation is needed"},
            "priority": {
                "type": "string",
                "enum": ["P1", "P2", "P3"],
                "description": "P1=immediate, P2=within 4h, P3=within 24h",
            },
            "context_summary": {
                "type": "string",
                "description": "Brief summary of the conversation for the human agent",
            },
        },
        "required": ["session_id", "reason", "priority"],
    },
)
async def escalate_to_human(
    session_id: str,
    reason: str,
    priority: str,
    context_summary: str = "",
) -> dict[str, Any]:
    ticket_id = f"ESC-{random.randint(10000, 99999)}"
    # --- Replace with real ticketing system (Zendesk, ServiceNow, etc.) ---
    return {
        "success": True,
        "ticket_id": ticket_id,
        "priority": priority,
        "queue": "tier2_support",
        "estimated_response": {
            "P1": "15 minutes",
            "P2": "4 hours",
            "P3": "24 hours",
        }.get(priority, "24 hours"),
        "message": (
            f"Your case has been escalated (Ticket {ticket_id}). "
            f"A specialist will contact you within "
            f"{'15 minutes' if priority == 'P1' else '4 hours' if priority == 'P2' else '24 hours'}."
        ),
    }
