"""Multi-LLM provider abstraction.

All three target providers (Gemini, OpenAI, Claude) can be accessed through
the `openai` Python SDK by simply changing base_url and api_key.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for a single LLM provider."""
    name: str
    api_key: str
    base_url: Optional[str]  # None = default OpenAI URL
    flash_model: str  # Fast/cheap model
    pro_model: str    # Powerful/expensive model
    supports_json_mode: bool = True


def _get_gemini_config() -> LLMConfig:
    return LLMConfig(
        name="gemini",
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        base_url=os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        flash_model=os.environ.get("FLASH_MODEL", "gemini-2.5-flash"),
        pro_model=os.environ.get("PRO_MODEL", "gemini-2.5-flash"),
        supports_json_mode=True,
    )


def _get_openai_config() -> LLMConfig:
    return LLMConfig(
        name="openai",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=None,  # default OpenAI URL
        flash_model=os.environ.get("OPENAI_FLASH_MODEL", "gpt-4o-mini"),
        pro_model=os.environ.get("OPENAI_PRO_MODEL", "gpt-4o"),
        supports_json_mode=True,
    )


def _get_claude_config() -> LLMConfig:
    return LLMConfig(
        name="claude",
        api_key=os.environ.get("CLAUDE_API_KEY", ""),
        base_url="https://api.anthropic.com/v1/",
        flash_model=os.environ.get("CLAUDE_FLASH_MODEL", "claude-sonnet-4-6"),
        pro_model=os.environ.get("CLAUDE_PRO_MODEL", "claude-opus-4-6"),
        # Claude's OpenAI-compat endpoint may not fully support response_format
        supports_json_mode=False,
    )


PROVIDER_REGISTRY = {
    "gemini": _get_gemini_config,
    "openai": _get_openai_config,
    "claude": _get_claude_config,
}


def get_llm_config(provider: Optional[str] = None) -> LLMConfig:
    """Get LLM configuration for the specified provider (or from env)."""
    provider = provider or os.environ.get("LLM_PROVIDER", "gemini")
    factory = PROVIDER_REGISTRY.get(provider)
    if not factory:
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(PROVIDER_REGISTRY.keys())}")
    config = factory()
    if not config.api_key:
        logger.warning("API key not set for provider '%s'", provider)
    return config


def get_llm_client(provider: Optional[str] = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client configured for the specified provider."""
    config = get_llm_config(provider)
    kwargs = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return AsyncOpenAI(**kwargs)
