from __future__ import annotations

import json

from anthropic import Anthropic
from openai import OpenAI

from fair_j.schemas import AdapterInput


OPENAI_MODEL_PREFIX = "openai/"
ANTHROPIC_MODEL_PREFIX = "anthropic/"


def infer_model_provider(model_name: str) -> str:
    if model_name.startswith(OPENAI_MODEL_PREFIX):
        return "openai"
    if model_name.startswith(ANTHROPIC_MODEL_PREFIX):
        return "anthropic"
    return "openrouter"


def build_openai_client(adapter_input: AdapterInput, model_name: str):
    provider = infer_model_provider(model_name)
    if provider == "openai":
        if not adapter_input.openai_api_key:
            raise ValueError("openai_api_key is required for OpenAI models.")
        return OpenAI(
            api_key=adapter_input.openai_api_key,
            timeout=adapter_input.request_timeout_seconds,
        )
    if provider == "anthropic":
        if not adapter_input.anthropic_api_key:
            raise ValueError("anthropic_api_key is required for Anthropic models.")
        return Anthropic(
            api_key=adapter_input.anthropic_api_key,
            timeout=adapter_input.request_timeout_seconds,
        )
    if not adapter_input.openrouter_api_key:
        raise ValueError("openrouter_api_key is required for OpenRouter models.")
    return OpenAI(
        api_key=adapter_input.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=adapter_input.request_timeout_seconds,
    )


def build_paraphrase_client(
    model_name: str,
    *,
    openrouter_api_key: str,
    openai_api_key: str | None,
    anthropic_api_key: str | None,
    timeout_seconds: float,
):
    provider = infer_model_provider(model_name)
    if provider == "openai":
        if not openai_api_key:
            raise ValueError("openai_api_key is required for OpenAI models.")
        return OpenAI(api_key=openai_api_key, timeout=timeout_seconds, max_retries=0)
    if provider == "anthropic":
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key is required for Anthropic models.")
        return Anthropic(api_key=anthropic_api_key, timeout=timeout_seconds)
    if not openrouter_api_key:
        raise ValueError("openrouter_api_key is required for OpenRouter models.")
    return OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout_seconds,
        max_retries=0,
    )


def completion_text(completion: object) -> str:
    if hasattr(completion, "choices"):
        message = completion.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
    if hasattr(completion, "content"):
        parts = []
        for item in completion.content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return json.dumps(getattr(completion, "model_dump", lambda: {})(), ensure_ascii=False)
