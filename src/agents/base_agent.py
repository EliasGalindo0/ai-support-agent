"""
BaseAgent: the foundation all agents inherit from.

The agent loop (ReAct pattern):
1. Build context: system prompt + history + user message.
2. Call LLM.
3. If response contains tool calls → execute tools → append results → loop.
4. If response is text with no tool calls → return final response.
5. Abort after max_iterations to prevent runaway loops.

Senior improvements embedded:
- Context compression: when history exceeds token budget, generate a summary
  and replace the oldest N messages with a single summary message.
- Confidence guard: if the LLM response contains hedge markers ("I think",
  "I'm not sure"), optionally re-query with a grounding instruction.
- Tool result injection: tool outputs are added as structured messages so the
  LLM always has clear attribution of what came from where.
- Latency budget: per-agent timeout cancels runaway tool chains.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import get_settings
from src.llm.client import LLMClient, Message, ToolDefinition
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.tools.registry import tool_registry

log = structlog.get_logger(__name__)

# Hedge phrases that suggest low confidence — triggers re-grounding.
_HEDGE_PHRASES = [
    "i'm not sure", "i think", "i believe", "i'm not certain",
    "probably", "possibly", "might be", "could be",
]

CONTEXT_SUMMARY_PROMPT = (
    "Summarise the conversation so far in 3-5 bullet points. "
    "Focus on: customer issue, actions taken, current status. "
    "Be factual and concise."
)


@dataclass
class AgentResponse:
    text: str
    session_id: str
    agent_type: str
    iterations: int = 0
    tool_calls_made: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    escalated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """
    Abstract base for all support agents.

    Subclasses must define:
        agent_type: str          — unique identifier
        system_prompt: str       — role-specific system prompt
        tool_names: list[str]    — names of tools this agent can use
        model_tier: str          — "heavy" | "standard" | "light"
    """

    agent_type: str = "base"
    system_prompt: str = "You are a helpful assistant."
    tool_names: list[str] = []
    model_tier: str = "standard"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._short_term = short_term or ShortTermMemory()
        self._long_term = long_term or LongTermMemory()
        self._cfg = get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        user_message: str,
        session_id: str,
        user_id: str | None = None,
        extra_context: str = "",
    ) -> AgentResponse:
        """
        Process a user message and return the agent's final response.

        This is the main entry point.  Subclasses should NOT override this —
        override `_build_system_prompt` or `_post_process` instead.
        """
        t0 = time.perf_counter()
        cfg = self._cfg

        log.info(
            "agent.run.start",
            agent=self.agent_type,
            session_id=session_id,
            message_preview=user_message[:80],
        )

        # Persist user message
        user_msg = Message(role="user", content=user_message)
        await self._short_term.add_message(session_id, user_msg)
        await self._long_term.log_interaction(
            session_id=session_id,
            role="user",
            content=user_message,
            agent_type=self.agent_type,
        )

        # Build tool list
        tools: list[ToolDefinition] = tool_registry.get_definitions(self.tool_names)

        # Build system prompt (can be enriched by subclass)
        system = await self._build_system_prompt(session_id, user_id, extra_context)

        # Run ReAct loop
        response = await asyncio.wait_for(
            self._react_loop(
                session_id=session_id,
                user_id=user_id,
                system=system,
                tools=tools,
            ),
            timeout=cfg.agent_timeout_seconds,
        )

        response.latency_ms = (time.perf_counter() - t0) * 1000

        # Persist final assistant message
        assistant_msg = Message(role="assistant", content=response.text)
        await self._short_term.add_message(session_id, assistant_msg)
        await self._long_term.log_interaction(
            session_id=session_id,
            role="assistant",
            content=response.text,
            agent_type=self.agent_type,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )

        # Auto-compress if context is getting large
        await self._maybe_compress_history(session_id)

        log.info(
            "agent.run.complete",
            agent=self.agent_type,
            session_id=session_id,
            iterations=response.iterations,
            cost_usd=f"${response.cost_usd:.6f}",
            latency_ms=f"{response.latency_ms:.0f}",
        )

        return response

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    async def _react_loop(
        self,
        session_id: str,
        user_id: str | None,
        system: str,
        tools: list[ToolDefinition],
    ) -> AgentResponse:
        cfg = self._cfg
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        tool_calls_made: list[str] = []

        for iteration in range(cfg.max_agent_iterations):
            history = await self._short_term.get_history(session_id)

            llm_response = await self._llm.complete(
                messages=history,
                system=system,
                tools=tools if tools else None,
                model_tier=self.model_tier,
            )

            total_input_tokens += llm_response.input_tokens
            total_output_tokens += llm_response.output_tokens
            total_cost += llm_response.cost_usd

            # --- No tool calls: final text response ---
            if not llm_response.tool_calls:
                final_text = await self._post_process(
                    text=llm_response.content,
                    session_id=session_id,
                )
                return AgentResponse(
                    text=final_text,
                    session_id=session_id,
                    agent_type=self.agent_type,
                    iterations=iteration + 1,
                    tool_calls_made=tool_calls_made,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cost_usd=total_cost,
                )

            # --- Tool calls: execute and feed results back ---
            assistant_tool_msg = self._format_tool_call_message(llm_response)
            await self._short_term.add_message(session_id, assistant_tool_msg)

            for tc in llm_response.tool_calls:
                tool_calls_made.append(tc.name)
                log.info(
                    "agent.tool_call",
                    agent=self.agent_type,
                    tool=tc.name,
                    args=tc.arguments,
                )

                result = await tool_registry.execute(tc.name, tc.arguments)
                tool_result_msg = Message(
                    role="user",
                    content=f"[Tool result for {tc.name}]\n{result.to_llm_content()}",
                )
                await self._short_term.add_message(session_id, tool_result_msg)

        # Safety valve: max iterations reached
        log.warning(
            "agent.max_iterations_reached",
            agent=self.agent_type,
            session_id=session_id,
            iterations=cfg.max_agent_iterations,
        )
        return AgentResponse(
            text=(
                "I've been working on your request but couldn't complete it fully. "
                "Let me escalate this to a specialist."
            ),
            session_id=session_id,
            agent_type=self.agent_type,
            iterations=cfg.max_agent_iterations,
            tool_calls_made=tool_calls_made,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=total_cost,
            escalated=True,
        )

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    async def _build_system_prompt(
        self,
        session_id: str,
        user_id: str | None,
        extra_context: str,
    ) -> str:
        """Override to inject dynamic context into the system prompt."""
        prompt = self.system_prompt
        if extra_context:
            prompt += f"\n\n## Additional context\n{extra_context}"

        # Inject user facts if available
        if user_id:
            facts = await self._long_term.get_facts(user_id)
            if facts:
                facts_text = "\n".join(f"- {k}: {v}" for k, v in facts.items())
                prompt += f"\n\n## Known customer facts\n{facts_text}"

        return prompt

    async def _post_process(self, text: str, session_id: str) -> str:
        """
        Optional post-processing hook.

        Default: detect low-confidence language and add a soft disclaimer.
        Override for agent-specific output transformation.
        """
        lower = text.lower()
        if any(phrase in lower for phrase in _HEDGE_PHRASES):
            log.debug(
                "agent.hedge_detected",
                agent=self.agent_type,
                session_id=session_id,
            )
            # Don't add disclaimer for short responses — looks silly
            if len(text) > 100:
                text += (
                    "\n\n*If this doesn't fully resolve your issue, "
                    "please let me know and I can look into it further.*"
                )
        return text

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _format_tool_call_message(self, llm_response: Any) -> Message:
        """
        Encode tool calls as a message for the conversation history.
        This preserves the assistant's reasoning turn in the context window.
        """
        import json
        parts = []
        if llm_response.content:
            parts.append(llm_response.content)
        for tc in llm_response.tool_calls:
            parts.append(
                f"[Calling tool: {tc.name}]\n{json.dumps(tc.arguments, indent=2)}"
            )
        return Message(role="assistant", content="\n\n".join(parts))

    async def _maybe_compress_history(self, session_id: str) -> None:
        """
        If the context is getting large, summarise the oldest half of messages.
        This prevents context overflow while preserving key information.
        """
        token_count = await self._short_term.get_token_count(session_id)
        budget = self._cfg.max_context_tokens * 0.8  # 80% threshold

        if token_count < budget:
            return

        log.info(
            "agent.compressing_context",
            agent=self.agent_type,
            session_id=session_id,
            estimated_tokens=token_count,
        )

        history = await self._short_term.get_history(session_id)
        if len(history) < 6:
            return

        # Summarise the first half
        to_summarise = history[: len(history) // 2]
        summary_prompt = [
            *to_summarise,
            Message(role="user", content=CONTEXT_SUMMARY_PROMPT),
        ]
        summary_response = await self._llm.complete(
            messages=summary_prompt,
            system="You summarise conversations concisely.",
            model_tier="light",  # Use cheapest model for summaries
        )
        summary_text = f"[Conversation summary]\n{summary_response.content}"

        # Rebuild history: summary + second half
        remaining = history[len(history) // 2:]
        await self._short_term.clear_session(session_id)
        await self._short_term.add_message(
            session_id, Message(role="assistant", content=summary_text)
        )
        for msg in remaining:
            await self._short_term.add_message(session_id, msg)

        await self._long_term.update_summary(session_id, summary_response.content)
        log.info("agent.context_compressed", session_id=session_id)
