"""
LLM client abstraction supporting Anthropic and OpenAI.

Design principles:
- Provider-agnostic interface: swap backends without changing agent code.
- Built-in retry with exponential backoff (tenacity).
- Token and cost tracking per request and accumulated daily.
- Streaming support for low-latency UX.
- Structured output parsing for tool-call responses.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import LLMProvider, get_settings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Cost tables (USD per 1M tokens, as of 2025)
# ---------------------------------------------------------------------------
_ANTHROPIC_COSTS: dict[str, dict[str, float]] = {
    "claude-opus-4-6":              {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":            {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001":    {"input": 0.80,  "output": 4.00},
}
_OPENAI_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o":           {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-3.5-turbo":    {"input": 0.50,  "output": 1.50},
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    table = {**_ANTHROPIC_COSTS, **_OPENAI_COSTS}
    prices = table.get(model, {"input": 5.0, "output": 15.0})
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    latency_ms: float = 0.0
    stop_reason: str = "end_turn"


@dataclass
class DailyCostTracker:
    """Thread-safe daily cost accumulator."""
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _total: float = 0.0
    _date: str = ""

    async def add(self, cost: float) -> float:
        from datetime import date
        today = date.today().isoformat()
        async with self._lock:
            if self._date != today:
                self._total = 0.0
                self._date = today
            self._total += cost
            return self._total

    async def get(self) -> float:
        async with self._lock:
            return self._total


# Singleton tracker shared across all LLM calls.
_daily_cost = DailyCostTracker()


class BudgetExceededError(Exception):
    """Raised when daily LLM cost budget is exhausted."""


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------
class _AnthropicBackend:
    def __init__(self, api_key: str) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        model: str,
        messages: list[Message],
        system: str,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        anthropic_msgs = [{"role": m.role, "content": m.content} for m in messages]
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_msgs,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        t0 = time.perf_counter()
        resp = await self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Parse response
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )

        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        cost = _cost_usd(model, in_tok, out_tok)

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            model=model,
            latency_ms=latency_ms,
            stop_reason=resp.stop_reason or "end_turn",
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        anthropic_msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_msgs,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for chunk in stream.text_stream:
                yield chunk


class _OpenAIBackend:
    def __init__(self, api_key: str) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        model: str,
        messages: list[Message],
        system: str,
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        import json

        oai_msgs: list[dict] = []
        if system:
            oai_msgs.append({"role": "system", "content": system})
        oai_msgs += [{"role": m.role, "content": m.content} for m in messages]

        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": oai_msgs,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"

        t0 = time.perf_counter()
        resp = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                    )
                )

        in_tok = resp.usage.prompt_tokens if resp.usage else 0
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        cost = _cost_usd(model, in_tok, out_tok)

        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            model=model,
            latency_ms=latency_ms,
            stop_reason=choice.finish_reason or "stop",
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        oai_msgs: list[dict] = []
        if system:
            oai_msgs.append({"role": "system", "content": system})
        oai_msgs += [{"role": m.role, "content": m.content} for m in messages]

        stream = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=oai_msgs,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Public LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Provider-agnostic LLM client with retry, cost tracking, and streaming.

    Usage::

        client = LLMClient()
        response = await client.complete(
            messages=[Message("user", "Hello")],
            system="You are a helpful assistant.",
            model_tier="standard",
        )
    """

    def __init__(self) -> None:
        cfg = get_settings()
        if cfg.llm_provider == LLMProvider.ANTHROPIC:
            self._backend: _AnthropicBackend | _OpenAIBackend = _AnthropicBackend(
                cfg.anthropic_api_key
            )
        else:
            self._backend = _OpenAIBackend(cfg.openai_api_key)
        self._cfg = cfg

    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        tools: list[ToolDefinition] | None = None,
        model_tier: str = "standard",
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Run a completion with automatic retry and cost tracking."""
        cfg = self._cfg
        model = cfg.model_for(model_tier)  # type: ignore[arg-type]
        max_tok = max_tokens or cfg.max_tokens_per_request

        # Budget guard
        daily = await _daily_cost.get()
        if daily >= cfg.cost_budget_daily_usd:
            raise BudgetExceededError(
                f"Daily budget ${cfg.cost_budget_daily_usd:.2f} exceeded "
                f"(current: ${daily:.4f})"
            )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_not_exception_type(BudgetExceededError),
            reraise=True,
        ):
            with attempt:
                response = await asyncio.wait_for(
                    self._backend.complete(
                        model=model,
                        messages=messages,
                        system=system,
                        tools=tools or [],
                        max_tokens=max_tok,
                        temperature=temperature,
                    ),
                    timeout=cfg.llm_timeout_seconds,
                )

        total = await _daily_cost.add(response.cost_usd)
        log.info(
            "llm.complete",
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=f"${response.cost_usd:.6f}",
            daily_total_usd=f"${total:.4f}",
            latency_ms=f"{response.latency_ms:.0f}",
        )

        if total >= cfg.cost_alert_threshold_usd:
            log.warning(
                "llm.budget_alert",
                daily_total_usd=total,
                threshold=cfg.cost_alert_threshold_usd,
            )

        return response

    async def stream(
        self,
        messages: list[Message],
        system: str = "",
        model_tier: str = "standard",
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """
        Return an async iterator that streams tokens for low-latency responses.

        Made async so the budget check can run before streaming starts.
        Callers: ``async for chunk in await client.stream(...):``.
        """
        cfg = self._cfg
        daily = await _daily_cost.get()
        if daily >= cfg.cost_budget_daily_usd:
            raise BudgetExceededError(
                f"Daily budget ${cfg.cost_budget_daily_usd:.2f} exceeded "
                f"(current: ${daily:.4f})"
            )
        model = cfg.model_for(model_tier)  # type: ignore[arg-type]
        max_tok = max_tokens or cfg.max_tokens_per_request
        return self._backend.stream(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tok,
            temperature=temperature,
        )

    @staticmethod
    async def get_daily_cost() -> float:
        return await _daily_cost.get()
