from __future__ import annotations

from langchain_openai import ChatOpenAI

from mini_ocr.core.config import settings


def build_chat_model() -> ChatOpenAI:
    provider = settings.llm_provider.lower().strip()
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key

    if provider == "ollama":
        base_url = base_url or "http://localhost:11434/v1"
        api_key = api_key or "ollama"

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set. For Ollama use LLM_API_KEY=ollama")

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_tokens=2046,
    )
