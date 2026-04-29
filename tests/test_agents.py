"""
Tests for the agent system.

Strategy:
- Mock the LLM client to avoid real API calls in CI.
- Test the ReAct loop independently of the LLM.
- Test routing logic (both hard-override and LLM-based paths).
- Test context compression trigger.
- Integration test the full orchestrator flow.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.agents.base_agent import BaseAgent
from src.agents.orchestrator import (
    Orchestrator,
    _has_escalation_signal,
    _is_internal_user,
)
from src.llm.client import LLMResponse, Message
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory

import src.tools.customer_tools   # noqa: F401 — register tools
import src.tools.internal_tools   # noqa: F401
import src.tools.search_tools     # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_llm_response(text: str, tool_calls=None) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=tool_calls or [],
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        model="claude-sonnet-4-6",
        latency_ms=200,
    )


def make_mock_llm(responses: list[LLMResponse]):
    """Create a mock LLMClient that returns responses in sequence."""
    mock = AsyncMock()
    mock.complete = AsyncMock(side_effect=responses)
    return mock


def make_mock_memory():
    """In-memory short-term memory (no Redis needed)."""
    mem = ShortTermMemory()
    mem._use_fallback = True
    return mem


async def make_mock_long_term():
    """In-memory long-term memory (SQLite)."""
    db = LongTermMemory()
    # Override to use in-memory SQLite for tests
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    db._engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    db._session_factory = async_sessionmaker(db._engine, expire_on_commit=False)
    await db.init_db()
    return db


# ---------------------------------------------------------------------------
# Unit tests: hard override routing
# ---------------------------------------------------------------------------
class TestRoutingHeuristics:
    def test_escalation_keywords(self):
        assert _has_escalation_signal("I'm going to sue you") is True
        assert _has_escalation_signal("I want a refund for my order") is False
        assert _has_escalation_signal("This is a data breach!") is True
        assert _has_escalation_signal("Can you help me track my package?") is False

    def test_internal_user_detection(self):
        assert _is_internal_user("alice@company.com") is True
        assert _is_internal_user("bob@gmail.com") is False
        assert _is_internal_user(None) is False
        assert _is_internal_user("") is False


# ---------------------------------------------------------------------------
# Unit tests: BaseAgent ReAct loop
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestBaseAgentLoop:
    async def test_simple_response_no_tools(self):
        """Agent returns text directly when no tool calls are needed."""
        mock_llm = make_mock_llm([
            make_llm_response("Hello! How can I help you today?")
        ])
        long_term = await make_mock_long_term()
        agent = BaseAgent(
            llm_client=mock_llm,
            short_term=make_mock_memory(),
            long_term=long_term,
        )

        response = await agent.run(
            user_message="Hi there",
            session_id="test-session-1",
        )

        assert response.text == "Hello! How can I help you today?"
        assert response.iterations == 1
        assert response.cost_usd == 0.001
        assert not response.escalated

    async def test_tool_call_then_response(self):
        """Agent makes a tool call, gets result, then responds."""
        from src.llm.client import ToolCall

        # First LLM call: requests tool
        tool_call_response = make_llm_response(
            text="Let me look that up.",
            tool_calls=[
                ToolCall(
                    id="tc_001",
                    name="get_order_status",
                    arguments={"order_id": "ORD-12345"},
                )
            ],
        )
        # Second LLM call: incorporates tool result
        final_response = make_llm_response(
            "Your order ORD-12345 is currently in transit. Expected delivery: tomorrow."
        )

        mock_llm = make_mock_llm([tool_call_response, final_response])
        long_term = await make_mock_long_term()

        agent = BaseAgent(
            llm_client=mock_llm,
            short_term=make_mock_memory(),
            long_term=long_term,
        )
        agent.tool_names = ["get_order_status"]

        response = await agent.run(
            user_message="Where is my order ORD-12345?",
            session_id="test-session-2",
        )

        assert response.iterations == 2
        assert "ORD-12345" in response.text
        assert "get_order_status" in response.tool_calls_made

    async def test_max_iterations_guard(self):
        """Agent stops after max_iterations and marks as escalated."""
        from src.llm.client import ToolCall
        from src.config import get_settings

        # Always returns a tool call — causes infinite loop prevention
        infinite_tool_response = make_llm_response(
            text="Checking...",
            tool_calls=[ToolCall(id="x", name="get_order_status", arguments={"order_id": "X"})],
        )
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=infinite_tool_response)

        long_term = await make_mock_long_term()
        agent = BaseAgent(
            llm_client=mock_llm,
            short_term=make_mock_memory(),
            long_term=long_term,
        )
        agent.tool_names = ["get_order_status"]

        # Temporarily reduce max iterations
        with patch.object(get_settings(), 'max_agent_iterations', 3):
            response = await agent.run(
                user_message="Loop forever",
                session_id="test-session-loop",
            )

        assert response.escalated is True

    async def test_agent_timeout_returns_escalated_response(self):
        """If the agent exceeds its latency budget, it returns an escalated response."""
        from src.config import get_settings

        mock_llm = AsyncMock()
        # Simulate a very slow LLM call (agent timeout should interrupt first)
        mock_llm.complete = AsyncMock(side_effect=lambda **_: asyncio.sleep(10))

        long_term = await make_mock_long_term()
        agent = BaseAgent(
            llm_client=mock_llm,
            short_term=make_mock_memory(),
            long_term=long_term,
        )

        with patch.object(get_settings(), "agent_timeout_seconds", 0.01):
            response = await agent.run(
                user_message="This should time out",
                session_id="test-session-timeout",
            )

        assert response.escalated is True
        assert "couldn't finish in time" in response.text.lower()


# ---------------------------------------------------------------------------
# Unit tests: Memory
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestShortTermMemory:
    async def test_add_and_retrieve(self):
        mem = ShortTermMemory()
        mem._use_fallback = True

        await mem.add_message("s1", Message(role="user", content="Hello"))
        await mem.add_message("s1", Message(role="assistant", content="Hi!"))

        history = await mem.get_history("s1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].content == "Hi!"

    async def test_session_isolation(self):
        mem = ShortTermMemory()
        mem._use_fallback = True

        await mem.add_message("s1", Message(role="user", content="Session 1"))
        await mem.add_message("s2", Message(role="user", content="Session 2"))

        s1_history = await mem.get_history("s1")
        s2_history = await mem.get_history("s2")

        assert len(s1_history) == 1
        assert s1_history[0].content == "Session 1"
        assert s2_history[0].content == "Session 2"

    async def test_clear_session(self):
        mem = ShortTermMemory()
        mem._use_fallback = True

        await mem.add_message("s3", Message(role="user", content="To be cleared"))
        await mem.clear_session("s3")
        history = await mem.get_history("s3")
        assert len(history) == 0


# ---------------------------------------------------------------------------
# Unit tests: Tool registry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestToolRegistry:
    async def test_tool_execution_success(self):
        from src.tools.registry import tool_registry
        result = await tool_registry.execute(
            "get_order_status",
            {"order_id": "ORD-99999"},
        )
        assert result.success is True
        assert result.output["order_id"] == "ORD-99999"
        assert "status" in result.output

    async def test_unknown_tool(self):
        from src.tools.registry import tool_registry
        result = await tool_registry.execute("nonexistent_tool", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_refund_limit_enforced(self):
        from src.tools.registry import tool_registry
        result = await tool_registry.execute(
            "process_refund",
            {"order_id": "ORD-1", "amount_usd": 999.99, "reason": "damaged_item"},
        )
        assert result.success is True  # Tool ran, but response indicates failure
        assert result.output["success"] is False
        assert result.output["requires_escalation"] is True

    async def test_tool_definitions_retrieved(self):
        from src.tools.registry import tool_registry
        defs = tool_registry.get_definitions(["get_order_status", "process_refund"])
        assert len(defs) == 2
        names = [d.name for d in defs]
        assert "get_order_status" in names
        assert "process_refund" in names


# ---------------------------------------------------------------------------
# Integration test: full Orchestrator flow (mocked LLM)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestOrchestrator:
    async def test_customer_support_route(self):
        routing_response = make_llm_response(
            '{"target_agent": "customer_support", "reason": "order question", "confidence": 0.9, "context": ""}'
        )
        final_response = make_llm_response("Your order is being processed.")

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=[routing_response, final_response])

        long_term = await make_mock_long_term()
        short_term = make_mock_memory()

        orchestrator = Orchestrator(
            llm_client=mock_llm,
            short_term=short_term,
            long_term=long_term,
        )

        response = await orchestrator.route(
            user_message="Where is my order?",
            session_id="orch-test-1",
            user_email="user@gmail.com",
        )

        assert "order" in response.text.lower() or len(response.text) > 0
        assert response.agent_type == "customer_support"

    async def test_internal_user_routes_to_internal_ops(self):
        final_response = make_llm_response("Here are the open P1 tickets.")
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=final_response)

        long_term = await make_mock_long_term()
        orchestrator = Orchestrator(
            llm_client=mock_llm,
            short_term=make_mock_memory(),
            long_term=long_term,
        )

        response = await orchestrator.route(
            user_message="Show me all P1 tickets",
            session_id="orch-test-2",
            user_email="agent@company.com",
        )

        assert response.agent_type == "internal_ops"

    async def test_escalation_keyword_bypass(self):
        esc_response = make_llm_response("I've escalated your case to our legal team.")
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=esc_response)

        long_term = await make_mock_long_term()
        orchestrator = Orchestrator(
            llm_client=mock_llm,
            short_term=make_mock_memory(),
            long_term=long_term,
        )

        response = await orchestrator.route(
            user_message="I'm going to sue your company for this!",
            session_id="orch-test-3",
            user_email="angry@customer.com",
        )

        assert response.agent_type == "escalation"
