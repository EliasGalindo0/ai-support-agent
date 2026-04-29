"""
Orchestrator: the top-level multi-agent router.

Responsibilities:
1. Classify the incoming request and route to the correct specialist agent.
2. Maintain routing state across turns (a customer mid-conversation shouldn't
   be re-routed unless they explicitly change topic).
3. Handle agent failures gracefully — fallback chain:
   primary agent → knowledge_base → escalation.
4. Aggregate responses from multiple agents when needed (e.g. KB lookup
   augmenting a customer support reply).

Routing strategy:
- First turn: LLM-based classification via ORCHESTRATOR_SYSTEM prompt.
- Subsequent turns: sticky routing unless topic_shift is detected.
- Hard overrides: explicit escalation signals bypass LLM routing.

Multi-agent collaboration pattern:
    CustomerSupport → needs policy info → calls KnowledgeBase sub-agent
    InternalOps → needs to escalate → calls Escalation sub-agent

This is synchronous-delegation (not parallel fan-out) to keep context coherent.
For parallel fan-out (e.g. summarise + translate simultaneously), use asyncio.gather
on independent agents.
"""
from __future__ import annotations

import json
import re
from typing import Any

import structlog

from src.agents.base_agent import AgentResponse, BaseAgent
from src.agents.customer_support import CustomerSupportAgent
from src.agents.escalation import EscalationAgent
from src.agents.internal_ops import InternalOpsAgent
from src.agents.knowledge_base import KnowledgeBaseAgent
from src.llm.client import LLMClient, Message
from src.llm.prompts import ORCHESTRATOR_SYSTEM
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory

log = structlog.get_logger(__name__)

# Hard-coded escalation signals — bypass LLM routing for speed and reliability.
_ESCALATION_KEYWORDS = [
    "lawyer", "lawsuit", "legal action", "sue", "attorney",
    "data breach", "hacked", "fraud", "scam", "stolen",
    "terrible", "unacceptable", "worst", "disgusting",
    "refund immediately", "chargeback",
]

_INTERNAL_DOMAINS = ["@company.com", "@internal.company.com"]


def _is_internal_user(user_email: str | None) -> bool:
    if not user_email:
        return False
    return any(user_email.endswith(d) for d in _INTERNAL_DOMAINS)


def _has_escalation_signal(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _ESCALATION_KEYWORDS)


class Orchestrator:
    """
    Top-level multi-agent orchestrator.

    All external requests enter here. The orchestrator routes, delegates,
    and optionally augments responses before returning to the caller.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        # Shared memory instances — passed down to every agent.
        self._llm = llm_client or LLMClient()
        self._short_term = short_term or ShortTermMemory()
        self._long_term = long_term or LongTermMemory()

        # Lazy-init agents (avoid creating LLM clients until needed)
        self._agents: dict[str, BaseAgent] = {}

    _KNOWN_AGENTS: dict[str, type[BaseAgent]] = {}  # set below after imports resolve

    def _get_agent(self, agent_type: str) -> BaseAgent:
        if agent_type not in self._agents:
            agent_map: dict[str, type[BaseAgent]] = {
                "customer_support": CustomerSupportAgent,
                "internal_ops": InternalOpsAgent,
                "knowledge_base": KnowledgeBaseAgent,
                "escalation": EscalationAgent,
            }
            cls = agent_map.get(agent_type)
            if cls is None:
                log.warning(
                    "orchestrator.unknown_agent_type",
                    requested=agent_type,
                    fallback="customer_support",
                )
                cls = CustomerSupportAgent
            self._agents[agent_type] = cls(
                llm_client=self._llm,
                short_term=self._short_term,
                long_term=self._long_term,
            )
        return self._agents[agent_type]

    async def route(
        self,
        user_message: str,
        session_id: str,
        user_id: str | None = None,
        user_email: str | None = None,
        force_agent: str | None = None,
    ) -> AgentResponse:
        """
        Main entry point. Route the message to the appropriate agent.

        Args:
            user_message: The incoming message text.
            session_id: Unique conversation session ID.
            user_id: Optional authenticated user ID.
            user_email: Optional user email (used to detect internal users).
            force_agent: Skip routing and go directly to this agent type.
        """
        log.info(
            "orchestrator.route",
            session_id=session_id,
            message_preview=user_message[:80],
        )

        # 1. Hard overrides (fastest path, no LLM call)
        if force_agent:
            target = force_agent
            reason = "forced by caller"
            confidence = 1.0
        elif _has_escalation_signal(user_message):
            target = "escalation"
            reason = "escalation keyword detected"
            confidence = 0.95
        elif _is_internal_user(user_email):
            target = "internal_ops"
            reason = "internal user email domain"
            confidence = 0.95
        else:
            # 2. Check for sticky routing (same topic, continuing conversation)
            sticky = await self._short_term.get_metadata(session_id, "active_agent")
            topic_shift = await self._detect_topic_shift(session_id, user_message)

            if sticky and not topic_shift:
                target = sticky
                reason = "sticky routing (continuing conversation)"
                confidence = 0.9
            else:
                # 3. LLM-based routing classification
                routing = await self._llm_route(user_message, session_id)
                target = routing.get("target_agent", "customer_support")
                reason = routing.get("reason", "llm routing")
                confidence = float(routing.get("confidence", 0.7))

                # If confidence is low, enrich with KB first
                if confidence < 0.5:
                    log.info(
                        "orchestrator.low_confidence_routing",
                        target=target,
                        confidence=confidence,
                    )

        log.info(
            "orchestrator.target_selected",
            target=target,
            reason=reason,
            confidence=confidence,
            session_id=session_id,
        )

        # Persist routing decision for sticky routing
        await self._short_term.set_metadata(session_id, "active_agent", target)

        # 3. Delegate to target agent
        try:
            agent = self._get_agent(target)
            response = await agent.run(
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
            )

            # 4. Post-routing: if agent signalled escalation, re-route
            if response.escalated and target != "escalation":
                log.info(
                    "orchestrator.agent_triggered_escalation",
                    session_id=session_id,
                    original_agent=target,
                )
                escalation_agent = self._get_agent("escalation")
                esc_response = await escalation_agent.run(
                    user_message=f"Escalation from {target}: {response.text}",
                    session_id=session_id,
                    user_id=user_id,
                )
                # Merge costs
                esc_response.input_tokens += response.input_tokens
                esc_response.output_tokens += response.output_tokens
                esc_response.cost_usd += response.cost_usd
                return esc_response

            return response

        except Exception as exc:
            log.error(
                "orchestrator.agent_error",
                target=target,
                session_id=session_id,
                error=str(exc),
            )
            # Fallback: graceful degradation
            return AgentResponse(
                text=(
                    "I'm sorry, I encountered an unexpected issue. "
                    "I've flagged this for our team. Please try again or "
                    "contact support directly."
                ),
                session_id=session_id,
                agent_type="orchestrator_fallback",
                escalated=True,
            )

    async def _llm_route(
        self,
        user_message: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Use the LLM to classify the request and choose an agent."""
        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=user_message)],
                system=ORCHESTRATOR_SYSTEM,
                model_tier="light",  # Routing is cheap — use light model
                temperature=0.0,    # Deterministic routing
            )
            # Extract JSON from response
            text = response.content.strip()
            # Handle markdown code fences if model wraps the JSON
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as exc:
            log.warning(
                "orchestrator.llm_route_failed",
                error=str(exc),
                fallback="customer_support",
            )
            return {
                "target_agent": "customer_support",
                "reason": "routing failed, using default",
                "confidence": 0.5,
                "context": "",
            }

    async def _detect_topic_shift(
        self,
        session_id: str,
        new_message: str,
    ) -> bool:
        """
        Lightweight topic shift detection.

        Uses a heuristic first (question words, new keywords), falls back to
        LLM for ambiguous cases. Avoids LLM calls for trivially continuing
        conversations to save cost.
        """
        history = await self._short_term.get_history(session_id)
        if len(history) < 2:
            return False  # First message, no shift possible

        # Heuristic: very short acknowledgement messages are continuations
        if len(new_message.split()) < 4:
            return False

        # Heuristic: check for new topic keywords
        shift_indicators = [
            "also", "another question", "different issue",
            "by the way", "one more thing", "unrelated",
            "new problem", "different order",
        ]
        lower = new_message.lower()
        return any(ind in lower for ind in shift_indicators)
