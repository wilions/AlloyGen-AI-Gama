from __future__ import annotations
import logging
from backend.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OnlineSearchAgent(BaseAgent):
    async def process(self, target_variable: str, task_type: str) -> str:
        """Simulate an online search for relevant supplementary data."""
        prompt = (
            f"You are a metallurgical data scientist and web search agent. The user wants "
            f"to build a machine learning model for a {task_type} task predicting "
            f"'{target_variable}'. Please describe what kind of relevant supplementary data "
            f"or literature you would search for online to help with this task. "
            "Keep it to a concise, engaging paragraph, indicating that you have 'searched' "
            "for this information to supplement the user's dataset."
        )
        try:
            result = await self._chat([{"role": "user", "content": prompt}])
            return f"Online Search Agent: {result}"
        except Exception as e:
            return f"Error during online search: {e}"
