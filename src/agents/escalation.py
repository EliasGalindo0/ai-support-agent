"""
Escalation Agent: handles high-severity cases requiring human handoff.
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent, AgentResponse
from src.llm.prompts import ESCALATION_SYSTEM


class EscalationAgent(BaseAgent):
    agent_type = "escalation"
    system_prompt = ESCALATION_SYSTEM
    model_tier = "heavy"   # High-stakes: use best model
    tool_names = [
        "escalate_to_human",
        "create_ticket",
        "get_customer_history",
        "log_action",
        "get_team_availability",
    ]

    async def run(self, user_message: str, session_id: str, **kwargs) -> AgentResponse:
        response = await super().run(user_message, session_id, **kwargs)
        response.escalated = True
        return response
