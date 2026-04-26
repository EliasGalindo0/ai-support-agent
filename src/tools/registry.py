"""
Tool registry: the single source of truth for all tools available to agents.

Adding a new tool:
1. Define an async function with a clear docstring.
2. Decorate it with @tool_registry.register(name, description, parameters_schema).
3. The function will be available to any agent that lists the tool name in its
   `tool_names` attribute.

Design:
- Separation of tool definition (schema) from implementation (handler).
- Per-tool execution timeout.
- Structured result: every tool returns a ToolResult, not a raw value.
  This ensures consistent handling in the agent's tool-use loop.
- Tool-level error boundaries: one tool failure doesn't crash the agent.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import structlog

from src.llm.client import ToolDefinition

log = structlog.get_logger(__name__)


@dataclass
class ToolResult:
    success: bool
    output: Any              # serialisable value returned to the LLM
    error: str | None = None
    latency_ms: float = 0.0

    def to_llm_content(self) -> str:
        """Convert to a string the LLM can parse."""
        import json
        if not self.success:
            return f"[TOOL ERROR] {self.error}"
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, default=str)


@dataclass
class _ToolEntry:
    definition: ToolDefinition
    handler: Callable[..., Coroutine[Any, Any, Any]]
    timeout_seconds: float = 30.0


class ToolRegistry:
    """Central registry for all agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        timeout_seconds: float = 30.0,
    ) -> Callable:
        """Decorator to register a tool handler."""
        def decorator(fn: Callable) -> Callable:
            definition = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
            )
            self._tools[name] = _ToolEntry(
                definition=definition,
                handler=fn,
                timeout_seconds=timeout_seconds,
            )
            log.debug("tool.registered", name=name)
            return fn
        return decorator

    def get_definitions(self, names: list[str]) -> list[ToolDefinition]:
        """Return ToolDefinition objects for a subset of tools."""
        return [
            self._tools[n].definition
            for n in names
            if n in self._tools
        ]

    def list_all(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool with timeout and error boundary."""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                entry.handler(**arguments),
                timeout=entry.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            log.info("tool.executed", name=name, latency_ms=f"{latency_ms:.0f}")
            return ToolResult(success=True, output=result, latency_ms=latency_ms)
        except asyncio.TimeoutError:
            log.error("tool.timeout", name=name, timeout=entry.timeout_seconds)
            return ToolResult(
                success=False,
                error=f"Tool '{name}' timed out after {entry.timeout_seconds}s",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            log.error("tool.error", name=name, error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )


# Singleton — imported by all tool modules and agents.
tool_registry = ToolRegistry()
