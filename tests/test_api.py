"""
API endpoint tests using FastAPI test client.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.agents.base_agent import AgentResponse


@pytest.fixture
def mock_orchestrator_response():
    return AgentResponse(
        text="Your order is on its way!",
        session_id="test-session",
        agent_type="customer_support",
        iterations=1,
        cost_usd=0.001,
        latency_ms=500,
    )


@pytest.mark.asyncio
async def test_chat_endpoint(mock_orchestrator_response):
    with patch(
        "src.api.routes.Orchestrator.route",
        new_callable=AsyncMock,
        return_value=mock_orchestrator_response,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Where is my order?"},
            )

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "session_id" in data
    assert data["agent_type"] == "customer_support"


@pytest.mark.asyncio
async def test_chat_empty_message():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/chat",
            json={"message": ""},
        )
    assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data


@pytest.mark.asyncio
async def test_internal_query_requires_auth():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/internal/query",
            json={"message": "Show me open tickets"},
            # No token header
        )
    assert response.status_code == 403
