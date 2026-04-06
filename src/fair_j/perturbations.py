from __future__ import annotations

from openai import OpenAI

from fair_j.schemas import Criterion, Rubric, RubricVariant


PARAPHRASE_REQUEST_TIMEOUT_SECONDS = 90.0
PARAPHRASE_MAX_RETRIES = 5


def make_variants(
    rubric: Rubric,
    paraphrase_model: str,
    openrouter_api_key: str,
) -> list[RubricVariant]:
    variants = [build_baseline_variant(rubric)]
    variants.extend(
        make_paraphrase_variants(
            rubric=rubric,
            paraphrase_model=paraphrase_model,
            openrouter_api_key=openrouter_api_key,
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
) -> str:
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=PARAPHRASE_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    prompt = build_paraphrase_prompt(
        rubric_name=rubric_name,
        criterion_id=criterion_id,
        criterion_text=criterion_text,
    )

    last_error: Exception | None = None
    for _ in range(PARAPHRASE_MAX_RETRIES):
        try:
            completion = create_paraphrase_completion(
                client=client,
                paraphrase_model=paraphrase_model,
                prompt=prompt,
            )
            content = completion.choices[0].message.content
            paraphrase = normalize_paraphrase_output(content)
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
    client: OpenAI,
    paraphrase_model: str,
    prompt: str,
):
    kwargs = {
        "model": paraphrase_model,
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

    try:
        return client.chat.completions.create(
            extra_body=build_paraphrase_extra_body(paraphrase_model),
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
