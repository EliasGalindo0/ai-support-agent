"""
Internal Operations Agent: handles internal support workflows.
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.llm.prompts import INTERNAL_OPS_SYSTEM


class InternalOpsAgent(BaseAgent):
    agent_type = "internal_ops"
    system_prompt = INTERNAL_OPS_SYSTEM
    model_tier = "standard"
    tool_names = [
        "create_ticket",
        "update_ticket",
        "list_open_tickets",
        "check_sla_status",
        "get_team_availability",
        "log_action",
        "search_knowledge_base",
    ]
