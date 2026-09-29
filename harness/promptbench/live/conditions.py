"""Frozen public-task request contracts; no provider sees the benchmark label store."""

from __future__ import annotations

from typing import Any

from ..benchmark.model import BenchmarkProfile
from ..config import PREFERRED_ENDPOINTS
from ..domain import TASK as TASK
from ..domain import feedback
from ..storage import Store, canonical
from .adapters import PROMPT_SCHEMA, dollars, reservation
from .transport import OLLAMA, OPENROUTER

NAIVE = 'Identify applicable supplied CVEs for the installed system. Return only {"applicable_cves":["CVE-..."]} with unique supplied identifiers, or an empty array.'


def local_conditions(registry: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for model in registry["local_models"]:
        if "ollama_manifest_digest" not in model:
            continue
        for mode in [False, True] if "thinking" in model["capabilities"] else [None]:
            suffix = "off" if mode is False else "on" if mode is True else "default"
            result.append(
                {
                    "id": model["requested_tag"] + "/" + suffix,
                    "provider": "ollama",
                    "url": OLLAMA + "/api/chat",
                    "model": model["requested_tag"],
                    "identity": model,
                    "family": model["requested_tag"].split(":")[0],
                    "think": mode,
                    "expected_models": [model["requested_tag"]],
                    "expected_provider": "local-ollama",
                    "num_ctx": 16384,
                    "num_predict": 2048 if mode is True else 512,
                    "timeout_seconds": 180,
                    "schema_enum": False,
                    "structured_inventory": False,
                }
            )
    return result


def executor_request(
    condition: dict[str, Any], prompt: str, profile: BenchmarkProfile
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": condition["model"],
        "stream": False,
        "keep_alive": "5m",
        "truncate": False,
        "shift": False,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": canonical(
                    profile.executor_input(
                        structured_inventory=condition.get("structured_inventory", False)
                    )
                ),
            },
        ],
        "format": {
            "type": "object",
            "properties": {
                "applicable_cves": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        **(
                            {"enum": [a.id for a in profile.advisories]}
                            if condition.get("schema_enum", False)
                            else {}
                        ),
                    },
                }
            },
            "required": ["applicable_cves"],
            "additionalProperties": False,
        },
        "options": {
            "temperature": 0,
            "seed": 11,
            "num_ctx": condition["num_ctx"],
            "num_predict": condition["num_predict"],
        },
    }
    if condition["think"] is not None:
        body["think"] = condition["think"]
    if condition.get("output_scaffold", False):
        body["format"]["properties"]["advisory_decisions"] = {
            "type": "array",
            "minItems": len(profile.advisories),
            "maxItems": len(profile.advisories),
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "applicable": {"type": "boolean"}},
                "required": ["id", "applicable"],
                "additionalProperties": False,
            },
        }
        body["format"]["required"].append("advisory_decisions")
    return body


def hosted_conditions(
    registry: dict[str, Any], overrides: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Pinned endpoint per hosted model; `overrides` maps model ID to an alternative endpoint tag."""
    preferred = {**PREFERRED_ENDPOINTS, **(overrides or {})}
    result = []
    for model in registry["hosted_models"]:
        alias = model["requested_model"]
        endpoint = next(
            (
                e
                for e in model.get("endpoints", {}).get("endpoints", [])
                if e["tag"] == preferred[alias]
            ),
            None,
        )
        if endpoint is None or not {"response_format", "structured_outputs", "reasoning"} <= set(
            endpoint["supported_parameters"]
        ):
            if overrides and alias in overrides:
                raise ValueError("override endpoint missing or lacks required parameters")
            continue
        result.append(
            {
                "id": alias,
                "provider": "openrouter",
                "url": OPENROUTER + "/api/v1/chat/completions",
                "model": alias,
                "canonical_model": model["catalog"]["canonical_slug"],
                "expected_models": [alias, model["catalog"]["canonical_slug"]],
                "expected_provider": endpoint["provider_name"],
                "endpoint": endpoint,
                "catalog": model["catalog"],
                "timeout_seconds": 180,
                "weight_digest": None,
                "weight_revision_note": "Hosted canonical ID is not a verifiable weight digest.",
            }
        )
    return result


def optimizer_request(
    condition: dict[str, Any], permitted: dict[str, Any], *, seed: int | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = condition["endpoint"]
    caps = {k: str(dollars(endpoint["pricing"][k]) * 1_000_000) for k in ("prompt", "completion")}
    if seed is not None and "seed" not in endpoint.get("supported_parameters", []):
        raise ValueError("pinned hosted endpoint does not accept a sampling seed")
    body = {
        **({"seed": seed} if seed is not None else {}),
        "model": condition["model"],
        "stream": False,
        "max_tokens": 8192,
        "reasoning": {"effort": "low"},
        "provider": {
            "only": [endpoint["tag"]],
            "order": [endpoint["tag"]],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {**caps, "request": "0"},
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "candidate_prompt", "strict": True, "schema": PROMPT_SCHEMA},
        },
        "messages": [{"role": "user", "content": canonical(permitted)}],
    }
    return {**condition, "reservation_usd": str(reservation(body, endpoint))}, body


def feedback_request(
    prompt: str,
    rows: list[dict[str, Any]],
    profiles: list[BenchmarkProfile],
    *,
    tier: int,
    seed: int,
    candidate: int,
) -> dict[str, Any]:
    if tier not in (1, 2, 3):
        raise ValueError("invalid disclosure tier")
    return {
        "task": TASK,
        "current_prompt": prompt,
        "feedback": feedback(rows, profiles, tier),
        "search_seed": seed,
        "candidate_slot": candidate,
        "tier": tier,
        "feedback_version": "relational-abstractions-v2",
    }


def study_feedback_request(
    prompt: str,
    rows: list[dict[str, Any]],
    profiles: list[BenchmarkProfile],
    *,
    tier: int,
    candidate: int,
) -> dict[str, Any]:
    """Main-study Optimizer view: no search label text; repetition identity is an API seed."""
    if tier not in (1, 2, 3):
        raise ValueError("invalid disclosure tier")
    if any(p.partition != "optimization" for p in profiles) or any(
        r["partition"] != "optimization" for r in rows
    ):
        raise ValueError("Optimizer feedback may only derive from the optimization partition")
    return {
        "task": TASK,
        "current_prompt": prompt,
        "feedback": feedback(rows, profiles, tier),
        "candidate_slot": candidate,
        "tier": tier,
        "feedback_version": "relational-abstractions-v2",
    }


def selected_profiles(
    profiles: list[BenchmarkProfile], partition: str, count: int, *, offset: int = 0
) -> list[BenchmarkProfile]:
    """Interleave difficulty and label presence; rotate deterministically without replacement."""
    buckets = [
        [
            p
            for p in profiles
            if p.partition == partition and p.stratum == s and bool(p.expected) == positive
        ]
        for s in ("easy", "medium", "hard")
        for positive in (False, True)
    ]
    ordered = [p for group in zip(*buckets, strict=True) for p in group]
    if not 0 < count <= len(ordered):
        raise ValueError("profile count exceeds partition")
    offset %= len(ordered)
    return (ordered[offset:] + ordered[:offset])[:count]


def save_registry(source: Store, target: Store) -> dict[str, Any]:
    registry = source.get("model_registry.json")
    target.put("inputs/model_registry.json", registry)
    return dict(registry)
