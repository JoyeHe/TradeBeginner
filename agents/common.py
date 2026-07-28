"""Shared helpers for constructing Agno agents."""

from __future__ import annotations

import structlog
from openai import AsyncOpenAI

from config.settings import Settings

logger = structlog.get_logger(__name__)


# OpenAI Chat Completions (newer) accepts role "developer"; DeepSeek and most
# OpenAI-compatible APIs only allow system/user/assistant/tool.
_COMPAT_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "model": "assistant",
}


def build_agno_model(settings: Settings):
    """Build an Agno model instance from settings."""
    provider = settings.llm_provider.lower()
    if provider == "openai":
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id=settings.llm_model, api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    if provider == "anthropic":
        from agno.models.anthropic import Claude

        return Claude(id=settings.llm_model, api_key=settings.llm_api_key)
    if provider == "deepseek":
        from agno.models.deepseek import DeepSeek

        # DeepSeek extends OpenAILike (system→system). Do not use OpenAIChat:
        # its default_role_map remaps system→developer and DeepSeek rejects that.
        return DeepSeek(
            id=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or "https://api.deepseek.com",
        )
    from agno.models.openai import OpenAILike

    return OpenAILike(
        id=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        role_map=_COMPAT_ROLE_MAP,
    )


def build_openai_client(settings: Settings) -> AsyncOpenAI:
    """Build a raw AsyncOpenAI client compatible with DeepSeek and OpenAI."""
    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def extract_agent_response(response: object) -> str:
    """Normalize Agno agent.arun() output to plain text (DeepSeek/OpenAI compatible)."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    messages = getattr(response, "messages", None)
    if messages:
        last = messages[-1]
        text = getattr(last, "content", None)
        if isinstance(text, str):
            return text
    return str(response)


async def agent_arun(agent, prompt: str) -> str:
    """Run Agno Agent.arun and return assistant text."""
    try:
        response = await agent.arun(prompt)
        return extract_agent_response(response)
    except Exception as exc:
        logger.warning("agent_arun_failed", error=str(exc))
        return ""


async def llm_chat(settings: Settings, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
    """Direct LLM chat call using standard system/user roles (DeepSeek compatible).

    Bypasses Agno's Agent which may use unsupported message roles.
    Returns the assistant's reply text, or empty string on failure.
    """
    if not settings.llm_api_key:
        return ""
    client = build_openai_client(settings)
    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else settings.llm_temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("llm_chat_failed", error=str(exc))
        return ""

