"""
Knowledge Base Agent: retrieval and synthesis specialist.
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.llm.prompts import KNOWLEDGE_BASE_SYSTEM


class KnowledgeBaseAgent(BaseAgent):
    agent_type = "knowledge_base"
    system_prompt = KNOWLEDGE_BASE_SYSTEM
    model_tier = "light"   # Cost-optimised: KB lookups don't need heavy models
    tool_names = [
        "search_knowledge_base",
        "web_search",
        "recall_from_memory",
    ]
