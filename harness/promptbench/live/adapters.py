"""Explicit request contracts and measured provider-usage normalization."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from ..domain import Profile
from ..storage import canonical
from .transport import response_json

PROMPT_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
    "additionalProperties": False,
}


def dollars(value: Any) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("USD amount is unavailable or invalid")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("USD amount is not numeric") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("USD amount must be finite and nonnegative")
    return amount


def executor_body(
    model: str, prompt: str, profile: Profile, *, think: bool | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": "5m",
        "truncate": False,
        "shift": False,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": canonical(profile.executor_input())},
        ],
        "format": {
            "type": "object",
            "properties": {
                "applicable_cves": {
                    "type": "array",
                    "items": {"type": "string", "enum": [a.id for a in profile.advisories]},
                }
            },
            "required": ["applicable_cves"],
            "additionalProperties": False,
        },
        "options": {"temperature": 0, "seed": 11, "num_ctx": 8192, "num_predict": 512},
    }
    if think is not None:
        body["think"] = think
    return body


def optimizer_body(
    model: str, endpoint_tag: str, permitted_request: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "max_tokens": 8192,
        "temperature": 0,
        "reasoning": {"effort": "low"},
        "provider": {
            "only": [endpoint_tag],
            "order": [endpoint_tag],
            "allow_fallbacks": False,
            "require_parameters": True,
            "quantizations": ["fp8"],
            "max_price": {"prompt": 0.8, "completion": 1.6, "request": 0},
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "candidate_prompt", "strict": True, "schema": PROMPT_SCHEMA},
        },
        "messages": [{"role": "user", "content": canonical(permitted_request)}],
    }


def reservation(body: dict[str, Any], endpoint: dict[str, Any]) -> Decimal:
    """Reserve a whole advertised context at router-enforced prices, not guessed tokens."""
    pricing = endpoint["pricing"]
    caps = body["provider"]["max_price"]
    if dollars(pricing.get("request", "0")) != 0:
        raise ValueError("per-request billing has not been admitted for this smoke")
    for field, cap in (("prompt", "prompt"), ("completion", "completion")):
        if dollars(pricing[field]) * 1_000_000 > dollars(caps[cap]):
            raise ValueError("endpoint tariff exceeds the frozen price cap")
    context = endpoint["context_length"]
    if type(context) is not int or context < 1:
        raise ValueError("missing endpoint context ceiling for cost reservation")
    return (
        Decimal(context) * dollars(caps["prompt"])
        + Decimal(body["max_tokens"]) * dollars(caps["completion"])
    ) / 1_000_000


def normalize(record: dict[str, Any], provider: str) -> dict[str, Any]:
    raw = response_json(record)
    if "error" in raw:
        raise ValueError("provider returned an error object; raw response retained")
    if provider == "ollama":
        message = raw.get("message") or {}
        if not isinstance(message, dict):
            raise ValueError("invalid Ollama message envelope")
        return {
            "text": message.get("content"),
            "reasoning": message.get("thinking"),
            "finish_reason": raw.get("done_reason") if raw.get("done") is True else None,
            "actual_model": raw.get("model"),
            "actual_provider": "local-ollama",
            "generation_id": None,
            "system_fingerprint": None,
            "input_tokens": raw.get("prompt_eval_count"),
            "output_tokens": raw.get("eval_count"),
            "reasoning_tokens": None,
            "cached_input_tokens": None,
            "reported_api_cost_usd": "0",
            "cost_origin": "local_no_external_api_charge",
            "timing_ns": {
                k: raw.get(k)
                for k in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                )
            },
            "usage_origin": "ollama_reported",
            "raw_usage": {
                k: v for k, v in raw.items() if k.endswith("_count") or k.endswith("_duration")
            },
        }
    if provider != "openrouter":
        raise ValueError("unknown provider adapter")
    choices = raw.get("choices") or []
    if not isinstance(choices, list) or any(not isinstance(c, dict) for c in choices):
        raise ValueError("invalid OpenRouter choices envelope")
    choice = choices[0] if len(choices) == 1 else {}
    message = choice.get("message") or {}
    usage = raw.get("usage") or {}
    if (
        not isinstance(message, dict)
        or not isinstance(usage, dict)
        or any(
            not isinstance(usage.get(k) or {}, dict)
            for k in ("completion_tokens_details", "prompt_tokens_details")
        )
    ):
        raise ValueError("invalid OpenRouter message/usage envelope")
    cost = usage.get("cost")
    return {
        "text": message.get("content"),
        "reasoning": message.get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "refusal": message.get("refusal"),
        "actual_model": raw.get("model"),
        "actual_provider": raw.get("provider"),
        "generation_id": raw.get("id"),
        "system_fingerprint": raw.get("system_fingerprint"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reported_api_cost_usd": str(dollars(cost)) if cost is not None else None,
        "cost_origin": "openrouter_usage.cost" if cost is not None else "unavailable",
        "usage_origin": "openrouter_reported",
        "raw_usage": usage,
        "reasoning_count_note": "Reasoning tokens are included in completion_tokens; do not add them again.",
    }
