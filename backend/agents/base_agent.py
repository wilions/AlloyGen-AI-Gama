from __future__ import annotations

import logging
from typing import Optional

from backend.llm.provider import get_llm_client, get_llm_config

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, provider: Optional[str] = None):
        config = get_llm_config(provider)
        self.client = get_llm_client(provider)
        self.flash_model = config.flash_model
        self.pro_model = config.pro_model
        self._supports_json_mode = config.supports_json_mode

    async def _chat(
        self,
        messages: list,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """Shared LLM call with logging and error handling."""
        model = model or self.flash_model
        kwargs: dict = {}

        if json_mode:
            if self._supports_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            else:
                # Fallback for providers that don't support response_format (e.g. Claude)
                # Inject JSON instruction into the system message
                if messages and messages[0].get("role") == "system":
                    messages = list(messages)  # don't mutate original
                    messages[0] = {
                        **messages[0],
                        "content": messages[0]["content"]
                        + "\n\nIMPORTANT: You MUST respond with valid JSON only. No markdown, no explanation, just JSON.",
                    }

        try:
            response = await self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM call failed in %s: %s", self.__class__.__name__, e)
            raise
