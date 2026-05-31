from __future__ import annotations

import json
import re
import sys
import time

from anthropic import Anthropic
from openai import OpenAI
from openai.types.chat import ChatCompletion

from fair_j.model_clients import (
    build_bedrock_client,
    build_openai_client,
    completion_text,
    infer_model_provider,
    resolve_bedrock_model_id,
)
from fair_j.schemas import AdapterInput


def build_judge_system_prompt() -> str:
    return (
        "You are a judge model. Reply with valid JSON only. "
        "Return an object with a top-level key 'scores' whose value is an object "
        "mapping each criterion id to one integer score. "
        "Every score must be a whole number on the rubric scale. "
        "Do not include any extra text."
    )


def build_judge_response_format(expected_ids: set[str]) -> dict[str, object]:
    sorted_ids = sorted(expected_ids)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_scores",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "object",
                        "properties": {
                            criterion_id: {"type": "integer"}
                            for criterion_id in sorted_ids
                        },
                        "required": sorted_ids,
                        "additionalProperties": False,
                    }
                },
                "required": ["scores"],
                "additionalProperties": False,
            },
        },
    }


def build_judge_tool_config(expected_ids: set[str]) -> dict[str, object]:
    sorted_ids = sorted(expected_ids)
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": "report_scores",
                    "description": "Return one integer score per rubric criterion.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "scores": {
                                    "type": "object",
                                    "properties": {
                                        criterion_id: {"type": "integer"}
                                        for criterion_id in sorted_ids
                                    },
                                    "required": sorted_ids,
                                    "additionalProperties": False,
                                }
                            },
                            "required": ["scores"],
                            "additionalProperties": False,
                        }
                    },
                }
            }
        ],
        "toolChoice": {"tool": {"name": "report_scores"}},
    }


def build_openrouter_provider_preferences(adapter_input: AdapterInput) -> dict[str, object] | None:
    provider: dict[str, object] = {
        # OpenRouter structured-output docs recommend requiring providers
        # that support the requested parameters when using json_schema.
        "require_parameters": True,
    }
    if adapter_input.provider_only:
        provider["only"] = [adapter_input.provider_only]
        provider["allow_fallbacks"] = False
    if adapter_input.provider_quantization:
        provider["quantizations"] = [adapter_input.provider_quantization]
    return provider or None


def build_openrouter_extra_body(adapter_input: AdapterInput) -> dict[str, object] | None:
    extra_body: dict[str, object] = {
        "reasoning": {
            "enabled": False,
            "effort": "none",
            "exclude": True,
        },
        # OpenRouter Response Healing repairs malformed JSON for non-streaming
        # structured-output requests before our parser sees the content.
        "plugins": [{"id": "response-healing"}],
    }
    provider = build_openrouter_provider_preferences(adapter_input)
    if provider is not None:
        extra_body["provider"] = provider
    return extra_body


def call_openrouter_judge(
    adapter_input: AdapterInput,
    judge_model: str,
    expected_ids: set[str],
    scale_min: int | float,
    scale_max: int | float,
    prompt_text: str,
) -> tuple[dict[str, int | float], str]:
    provider = infer_model_provider(judge_model)
    if provider == "anthropic":
        return call_anthropic_judge(
            adapter_input=adapter_input,
            judge_model=judge_model,
            expected_ids=expected_ids,
            scale_min=scale_min,
            scale_max=scale_max,
            prompt_text=prompt_text,
        )
    if provider == "openai":
        return call_openai_judge(
            adapter_input=adapter_input,
            judge_model=judge_model,
            expected_ids=expected_ids,
            scale_min=scale_min,
            scale_max=scale_max,
            prompt_text=prompt_text,
        )
    if provider == "bedrock":
        return call_bedrock_judge(
            adapter_input=adapter_input,
            judge_model=judge_model,
            expected_ids=expected_ids,
            scale_min=scale_min,
            scale_max=scale_max,
            prompt_text=prompt_text,
        )
    return call_openrouter_judge_json_object(
        openai_client=build_openai_client(adapter_input, judge_model),
        adapter_input=adapter_input,
        judge_model=judge_model,
        expected_ids=expected_ids,
        scale_min=scale_min,
        scale_max=scale_max,
        prompt_text=prompt_text,
    )


def call_bedrock_judge(
    adapter_input: AdapterInput,
    judge_model: str,
    expected_ids: set[str],
    scale_min: int | float,
    scale_max: int | float,
    prompt_text: str,
) -> tuple[dict[str, int | float], str]:
    bedrock_model_id = resolve_bedrock_model_id(judge_model)
    client = build_bedrock_client(adapter_input.bedrock_region, adapter_input.request_timeout_seconds)

    last_error: Exception | None = None
    for attempt in range(1, BEDROCK_JUDGE_MAX_RETRIES + 1):
        try:
            inference_config: dict[str, object] = {"maxTokens": 4096}
            if "opus-4-7" not in bedrock_model_id:
                inference_config["temperature"] = 0
            response = client.converse(
                modelId=bedrock_model_id,
                system=[{"text": build_judge_system_prompt()}],
                messages=[{"role": "user", "content": [{"text": prompt_text}]}],
                inferenceConfig=inference_config,
                toolConfig=build_judge_tool_config(expected_ids),
            )
            payload = extract_bedrock_tool_input(response)
            scores = extract_scores_payload(payload)
            if not isinstance(scores, dict):
                raise ValueError("Bedrock tool input did not contain an object under 'scores'.")
            numeric_scores: dict[str, float] = {}
            for criterion_id, score in scores.items():
                numeric_scores[str(criterion_id)] = float(score)
            validated = validate_openrouter_scores(
                scores=numeric_scores,
                expected_ids=expected_ids,
                scale_min=scale_min,
                scale_max=scale_max,
            )
            return validated, json.dumps(response, default=str)
        except Exception as error:
            last_error = error
            report_retry_error("bedrock", attempt, BEDROCK_JUDGE_MAX_RETRIES, error)
            if attempt < BEDROCK_JUDGE_MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Bedrock judge call failed after {BEDROCK_JUDGE_MAX_RETRIES} attempts."
    ) from last_error


def call_openai_judge(
    adapter_input: AdapterInput,
    judge_model: str,
    expected_ids: set[str],
    scale_min: int | float,
    scale_max: int | float,
    prompt_text: str,
) -> tuple[dict[str, int | float], str]:
    client = build_openai_client(adapter_input, judge_model)
    if not isinstance(client, OpenAI):
        raise TypeError("Expected OpenAI client for OpenAI models.")
    completion = client.chat.completions.create(
        model=judge_model.removeprefix("openai/"),
        temperature=0,
        response_format=build_judge_response_format(expected_ids),
        messages=[
            {"role": "system", "content": build_judge_system_prompt()},
            {"role": "user", "content": prompt_text},
        ],
    )
    return parse_json_object_scores(
        completion=completion,
        expected_ids=expected_ids,
        scale_min=scale_min,
        scale_max=scale_max,
    ), json.dumps(completion.model_dump(), ensure_ascii=False)


def call_anthropic_judge(
    adapter_input: AdapterInput,
    judge_model: str,
    expected_ids: set[str],
    scale_min: int | float,
    scale_max: int | float,
    prompt_text: str,
) -> tuple[dict[str, int | float], str]:
    client = build_openai_client(adapter_input, judge_model)
    if not isinstance(client, Anthropic):
        raise TypeError("Expected Anthropic client for Anthropic models.")
    completion = client.messages.create(
        model=judge_model.removeprefix("anthropic/"),
        max_tokens=1024,
        temperature=0,
        system=build_judge_system_prompt(),
        messages=[{"role": "user", "content": prompt_text}],
    )
    payload_text = strip_markdown_fences(completion_text(completion))
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("Anthropic judge response is not a JSON object.")
    scores = extract_scores_payload(payload)
    if not isinstance(scores, dict):
        raise ValueError("Anthropic judge response does not contain an object under 'scores'.")
    numeric_scores: dict[str, float] = {}
    for criterion_id, score in scores.items():
        numeric_scores[str(criterion_id)] = float(score)
    validated = validate_openrouter_scores(
        scores=numeric_scores,
        expected_ids=expected_ids,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return validated, json.dumps(completion.model_dump(), ensure_ascii=False)


def call_openrouter_judge_json_object(
    openai_client: OpenAI,
    adapter_input: AdapterInput,
    judge_model: str,
    expected_ids: set[str],
    scale_min: int | float,
    scale_max: int | float,
    prompt_text: str,
) -> tuple[dict[str, int | float], str]:
    completion = openai_client.chat.completions.create(
        model=judge_model,
        temperature=0,
        response_format=build_judge_response_format(expected_ids),
        extra_body=build_openrouter_extra_body(adapter_input),
        messages=[
            {"role": "system", "content": build_judge_system_prompt()},
            {"role": "user", "content": prompt_text},
        ],
    )
    scores = parse_json_object_scores(
        completion=completion,
        expected_ids=expected_ids,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    raw_model_output = json.dumps(completion.model_dump(), ensure_ascii=False)
    return scores, raw_model_output


def parse_json_object_scores(
    completion: ChatCompletion,
    expected_ids: set[str],
    scale_min: int | float,
    scale_max: int | float,
) -> dict[str, int | float]:
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("JSON mode returned an empty response.")

    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("JSON mode response is not a JSON object.")

    scores = extract_scores_payload(payload)
    if not isinstance(scores, dict):
        raise ValueError("JSON mode response does not contain an object under 'scores'.")

    numeric_scores: dict[str, float] = {}
    for criterion_id, score in scores.items():
        numeric_scores[str(criterion_id)] = float(score)

    return validate_openrouter_scores(
        scores=numeric_scores,
        expected_ids=expected_ids,
        scale_min=scale_min,
        scale_max=scale_max,
    )


def extract_scores_payload(payload: dict[str, object]) -> object:
    direct_scores = payload.get("scores")
    if isinstance(direct_scores, dict):
        return direct_scores

    direct_score = payload.get("score")
    if isinstance(direct_score, dict):
        return direct_score

    judge_response = payload.get("judge_response")
    if isinstance(judge_response, dict):
        nested_scores = judge_response.get("scores")
        if isinstance(nested_scores, dict):
            return nested_scores
        nested_score = judge_response.get("score")
        if isinstance(nested_score, dict):
            return nested_score

    if len(payload) == 1:
        only_value = next(iter(payload.values()))
        if isinstance(only_value, dict):
            nested_scores = only_value.get("scores")
            if isinstance(nested_scores, dict):
                return nested_scores
            nested_score = only_value.get("score")
            if isinstance(nested_score, dict):
                return nested_score

    return direct_scores


def validate_openrouter_scores(
    scores: dict[str, int | float],
    expected_ids: set[str],
    scale_min: int | float,
    scale_max: int | float,
) -> dict[str, int | float]:
    if set(scores.keys()) != expected_ids:
        missing = sorted(expected_ids - set(scores.keys()))
        extra = sorted(set(scores.keys()) - expected_ids)
        raise ValueError(f"Judge response criteria mismatch. Missing={missing} extra={extra}.")
    for score in scores.values():
        if score < scale_min or score > scale_max:
            raise ValueError(f"Score {score} is outside rubric scale [{scale_min}, {scale_max}].")
    return scores


BEDROCK_JUDGE_MAX_RETRIES = 8

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    match = _MARKDOWN_FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def extract_bedrock_text(response: dict) -> str:
    for block in response.get("output", {}).get("message", {}).get("content", []):
        text = block.get("text", "")
        if text:
            return text
    stop_reason = response.get("stopReason", "unknown")
    raise ValueError(f"Bedrock response contained no text content (stopReason={stop_reason!r}).")


def extract_bedrock_tool_input(response: dict) -> dict:
    for block in response.get("output", {}).get("message", {}).get("content", []):
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict) and tool_use.get("name") == "report_scores":
            tool_input = tool_use.get("input")
            if isinstance(tool_input, dict):
                return tool_input
    stop_reason = response.get("stopReason", "unknown")
    raise ValueError(
        f"Bedrock response did not contain a 'report_scores' tool use (stopReason={stop_reason!r})."
    )


def report_retry_error(
    mode: str,
    attempt: int,
    total_attempts: int,
    error: Exception,
) -> None:
    error_name = type(error).__name__
    error_message = " ".join(str(error).split())
    sys.stderr.write(
        f"Judge retry ({mode}) attempt {attempt}/{total_attempts} failed: "
        f"{error_name}: {error_message}\n"
    )
    sys.stderr.flush()
