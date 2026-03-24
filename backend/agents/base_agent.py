from __future__ import annotations
import logging
from openai import AsyncOpenAI
from backend.config import GEMINI_API_KEY, GEMINI_BASE_URL, FLASH_MODEL, PRO_MODEL

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
        )
        self.flash_model = FLASH_MODEL
        self.pro_model = PRO_MODEL

    async def _chat(
        self,
        messages: list[dict],
        model: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Shared LLM call with logging and error handling."""
        model = model or self.flash_model
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM call failed in %s: %s", self.__class__.__name__, e)
            raise
