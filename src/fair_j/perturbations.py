from __future__ import annotations

from anthropic import Anthropic
from openai import OpenAI

from fair_j.model_clients import build_paraphrase_client, completion_text, infer_model_provider
from fair_j.schemas import Criterion, Rubric, RubricVariant


PARAPHRASE_REQUEST_TIMEOUT_SECONDS = 90.0
PARAPHRASE_MAX_RETRIES = 5


def make_variants(
    rubric: Rubric,
    paraphrase_model: str,
    openrouter_api_key: str,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
) -> list[RubricVariant]:
    variants = [build_baseline_variant(rubric)]
    variants.extend(
        make_paraphrase_variants(
            rubric=rubric,
            paraphrase_model=paraphrase_model,
            openrouter_api_key=openrouter_api_key,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
        )
    )
    variants.extend(make_delete_variants(rubric))
    return variants


def build_baseline_variant(rubric: Rubric) -> RubricVariant:
    return RubricVariant(
        perturbation="baseline",
        rubric_name=rubric.name,
        criteria=_clone_criteria(rubric.criteria),
    )


def make_paraphrase_variants(
    rubric: Rubric,
    paraphrase_model: str,
    openrouter_api_key: str,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
) -> list[RubricVariant]:
    variants: list[RubricVariant] = []
    for index, criterion in enumerate(rubric.criteria, start=1):
        criteria = _clone_criteria(rubric.criteria)
        criteria[index - 1].text = paraphrase_criterion_text(
            criterion_text=criterion.text,
            rubric_name=rubric.name,
            criterion_id=criterion.id,
            paraphrase_model=paraphrase_model,
            openrouter_api_key=openrouter_api_key,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
        )
        variants.append(
            RubricVariant(
                perturbation=f"paraphrase__{criterion.id}__01",
                rubric_name=rubric.name,
                criteria=criteria,
            )
        )
    return variants


def paraphrase_criterion_text(
    criterion_text: str,
    rubric_name: str,
    criterion_id: str,
    paraphrase_model: str,
    openrouter_api_key: str,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
) -> str:
    provider = infer_model_provider(paraphrase_model)
    if provider == "openai":
        if not openai_api_key:
            raise ValueError("openai_api_key is required for OpenAI models.")
    elif provider == "anthropic":
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key is required for Anthropic models.")
    prompt = build_paraphrase_prompt(
        rubric_name=rubric_name,
        criterion_id=criterion_id,
        criterion_text=criterion_text,
    )

    last_error: Exception | None = None
    for _ in range(PARAPHRASE_MAX_RETRIES):
        try:
            completion = create_paraphrase_completion(
                openrouter_api_key=openrouter_api_key,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                paraphrase_model=paraphrase_model,
                prompt=prompt,
            )
            paraphrase = normalize_paraphrase_output(completion_text(completion))
            if paraphrase and paraphrase != criterion_text:
                return paraphrase
            if paraphrase:
                return paraphrase
            raise ValueError("Paraphrase model returned an empty criterion.")
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Failed to paraphrase criterion '{criterion_id}' with model '{paraphrase_model}' "
        f"after {PARAPHRASE_MAX_RETRIES} attempts."
    ) from last_error


def create_paraphrase_completion(
    openrouter_api_key: str,
    openai_api_key: str | None,
    anthropic_api_key: str | None,
    paraphrase_model: str,
    prompt: str,
):
    provider = infer_model_provider(paraphrase_model)
    if provider == "anthropic":
        client = Anthropic(
            api_key=anthropic_api_key,
            timeout=PARAPHRASE_REQUEST_TIMEOUT_SECONDS,
        )
        return client.messages.create(
            model=paraphrase_model.removeprefix("anthropic/"),
            max_tokens=1024,
            temperature=0,
            system=(
                "You rewrite evaluation rubric criteria. "
                "Return exactly one paraphrased criterion sentence. "
                "Preserve meaning, scale, and constraints. "
                "Do not add explanations, bullet points, or quotes."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

    client = build_paraphrase_client(
        paraphrase_model,
        openrouter_api_key=openrouter_api_key,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        timeout_seconds=PARAPHRASE_REQUEST_TIMEOUT_SECONDS,
    )
    kwargs = {
        "model": paraphrase_model.removeprefix("openai/") if provider == "openai" else paraphrase_model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You rewrite evaluation rubric criteria. "
                    "Return exactly one paraphrased criterion sentence. "
                    "Preserve meaning, scale, and constraints. "
                    "Do not add explanations, bullet points, or quotes."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    extra_body = None if provider == "openai" else build_paraphrase_extra_body(paraphrase_model)
    try:
        if extra_body is None:
            return client.chat.completions.create(**kwargs)
        return client.chat.completions.create(
            extra_body=extra_body,
            **kwargs,
        )
    except Exception as error:
        error_text = str(error)
        if "Reasoning is mandatory for this endpoint" not in error_text:
            raise
        return client.chat.completions.create(**kwargs)


def build_paraphrase_extra_body(paraphrase_model: str) -> dict[str, object]:
    extra_body: dict[str, object] = {
        "reasoning": {
            "enabled": False,
            "effort": "none",
            "exclude": True,
        }
    }
    if paraphrase_model == "google/gemini-2.5-flash-lite":
        extra_body["provider"] = {
            "only": ["google-ai-studio"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    return extra_body

def build_paraphrase_prompt(
    rubric_name: str,
    criterion_id: str,
    criterion_text: str,
) -> str:
    return (
        f"Rubric: {rubric_name}\n"
        f"Criterion id: {criterion_id}\n"
        f"Original criterion:\n{criterion_text}\n\n"
        "Rewrite this criterion in different wording while preserving the same evaluation meaning. "
        "Keep it as one criterion text only."
    )


def normalize_paraphrase_output(content: str | None) -> str:
    if not content:
        return ""
    paraphrase = " ".join(content.strip().split())
    if paraphrase.startswith('"') and paraphrase.endswith('"') and len(paraphrase) >= 2:
        paraphrase = paraphrase[1:-1].strip()
    return paraphrase


def make_delete_variants(rubric: Rubric) -> list[RubricVariant]:
    variants: list[RubricVariant] = []
    for criterion in rubric.criteria:
        kept = [Criterion(id=item.id, text=item.text) for item in rubric.criteria if item.id != criterion.id]
        variants.append(
            RubricVariant(
                perturbation=f"delete__{criterion.id}",
                rubric_name=rubric.name,
                criteria=kept,
            )
        )
    return variants


def _clone_criteria(criteria: list[Criterion]) -> list[Criterion]:
    return [Criterion(id=item.id, text=item.text) for item in criteria]
