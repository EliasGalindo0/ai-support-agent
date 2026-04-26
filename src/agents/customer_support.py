"""
Customer Support Agent: front-line agent for customer-facing interactions.
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent, AgentResponse
from src.llm.prompts import CUSTOMER_SUPPORT_SYSTEM


class CustomerSupportAgent(BaseAgent):
    agent_type = "customer_support"
    system_prompt = CUSTOMER_SUPPORT_SYSTEM
    model_tier = "standard"
    tool_names = [
        "get_order_status",
        "process_refund",
        "get_product_info",
        "get_customer_history",
        "search_knowledge_base",
        "recall_from_memory",
        "escalate_to_human",
    ]

    async def _post_process(self, text: str, session_id: str) -> str:
        """
        Customer-facing: ensure we never expose internal system details.
        """
        # Strip any accidental tool-call syntax that leaked into the output
        import re
        text = re.sub(r"\[(?:Tool result|Calling tool)[^\]]*\]", "", text).strip()
        return await super()._post_process(text, session_id)
