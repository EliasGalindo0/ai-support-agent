"""
Knowledge base and search tools shared across agents.
"""
from __future__ import annotations

import uuid
from typing import Any

from src.tools.registry import tool_registry

# ---------------------------------------------------------------------------
# Knowledge base search (semantic)
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="search_knowledge_base",
    description=(
        "Search the company knowledge base for policies, FAQs, and documentation. "
        "Always use this tool before answering policy or procedure questions. "
        "Returns the most relevant document chunks with source citations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "category": {
                "type": "string",
                "enum": ["policy", "faq", "technical", "billing", "shipping", "returns"],
                "description": "Optional category filter",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-10)",
                "default": 3,
            },
        },
        "required": ["query"],
    },
)
async def search_knowledge_base(
    query: str,
    category: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    # In production, this calls VectorStore.search() with the KB namespace.
    # Stub returns hardcoded realistic results.
    stub_docs = [
        {
            "id": "DOC-001",
            "title": "Return & Refund Policy",
            "category": "returns",
            "content": (
                "Customers may return items within 30 days of delivery for a full refund. "
                "Items must be unused and in original packaging. "
                "Digital products are non-refundable. "
                "Refunds are processed within 3-5 business days."
            ),
            "last_updated": "2025-01-15",
            "relevance": 0.92,
        },
        {
            "id": "DOC-002",
            "title": "Shipping FAQ",
            "category": "shipping",
            "content": (
                "Standard shipping: 5-7 business days, free for orders over $50. "
                "Express shipping: 2-3 business days, $12.99 flat fee. "
                "Overnight: next business day for orders before 2pm ET, $24.99."
            ),
            "last_updated": "2025-03-01",
            "relevance": 0.78,
        },
        {
            "id": "DOC-003",
            "title": "Billing & Payment Methods",
            "category": "billing",
            "content": (
                "We accept Visa, Mastercard, Amex, PayPal, and Apple Pay. "
                "Invoices are issued within 24 hours of order. "
                "For enterprise billing, contact billing@company.com."
            ),
            "last_updated": "2025-02-10",
            "relevance": 0.65,
        },
    ]

    if category:
        stub_docs = [d for d in stub_docs if d["category"] == category]

    return {
        "query": query,
        "results": stub_docs[:top_k],
        "total_found": len(stub_docs),
    }


# ---------------------------------------------------------------------------
# Web search (for agents that need live information)
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="web_search",
    description=(
        "Search the web for current information not in the knowledge base. "
        "Use sparingly — prefer the knowledge base for company-specific questions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    },
    timeout_seconds=15.0,
)
async def web_search(query: str, max_results: int = 3) -> dict[str, Any]:
    # --- Replace with Bing/Brave/SerpAPI call ---
    # Stub for development
    return {
        "query": query,
        "results": [
            {
                "title": f"Search result for: {query}",
                "url": "https://example.com/result",
                "snippet": "This is a stub web search result. Connect a real search API.",
            }
        ],
        "note": "Web search stub — connect real search API for production.",
    }


# ---------------------------------------------------------------------------
# Semantic memory retrieval
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="recall_from_memory",
    description=(
        "Search long-term memory for relevant past interactions with this customer. "
        "Use to personalise responses and avoid asking for info already provided."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for in memory"},
            "user_id": {"type": "string", "description": "Customer/user identifier"},
        },
        "required": ["query", "user_id"],
    },
)
async def recall_from_memory(query: str, user_id: str) -> dict[str, Any]:
    # In production, this calls VectorStore.search() on the user's memory namespace.
    # The agent layer injects the real vector store via dependency injection.
    return {
        "query": query,
        "user_id": user_id,
        "memories": [],
        "note": "No relevant memories found. (Stub — real vector search in production.)",
    }
