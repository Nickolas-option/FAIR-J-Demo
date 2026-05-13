from __future__ import annotations

import json

import boto3
from anthropic import Anthropic
from botocore.config import Config
from openai import OpenAI

from fair_j.schemas import AdapterInput


OPENAI_MODEL_PREFIX = "openai/"
ANTHROPIC_MODEL_PREFIX = "anthropic/"
BEDROCK_MODEL_PREFIX = "bedrock/"

BEDROCK_ALIAS_TO_MODEL_ID: dict[str, str] = {
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-opus-4-5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",
}


def infer_model_provider(model_name: str) -> str:
    if model_name.startswith(OPENAI_MODEL_PREFIX):
        return "openai"
    if model_name.startswith(ANTHROPIC_MODEL_PREFIX):
        return "anthropic"
    if model_name.startswith(BEDROCK_MODEL_PREFIX):
        return "bedrock"
    return "openrouter"


def resolve_bedrock_model_id(model_name: str) -> str:
    alias = model_name.removeprefix(BEDROCK_MODEL_PREFIX)
    model_id = BEDROCK_ALIAS_TO_MODEL_ID.get(alias)
    if model_id is None:
        known = ", ".join(sorted(BEDROCK_ALIAS_TO_MODEL_ID))
        raise ValueError(
            f"Unknown Bedrock model alias '{alias}'. Known aliases: {known}."
        )
    return model_id


def build_bedrock_client(region: str, timeout_seconds: float = 90.0):
    config = Config(
        connect_timeout=10,
        read_timeout=timeout_seconds,
        retries={"max_attempts": 0},
    )
    return boto3.client("bedrock-runtime", region_name=region, config=config)


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
