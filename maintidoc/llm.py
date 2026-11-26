"""LLM open source local : Qwen2.5 servi par Ollama, exposé via LangChain."""

from __future__ import annotations

from langchain_ollama import ChatOllama

from .config import settings


def get_llm(model: str | None = None, temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(
        model=model or settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=8192,  # fenêtre de contexte : question + passages récupérés
    )
