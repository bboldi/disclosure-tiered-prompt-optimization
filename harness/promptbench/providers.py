"""Deterministic teaching fakes: no HTTP, credentials, GPU, or learned models."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Protocol

from .domain import Advisory, Component
from .storage import canonical, digest


class ProviderFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class Reply:
    text: str
    finish_reason: str
    provider_request_id: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def record(self) -> dict[str, Any]:
        result = asdict(self)
        result["usage_origin"] = "synthetic_fixture"
        result["tokens_are_estimates"] = True
        result["actual_api_cost_usd"] = "0"
        result["synthetic_cost_usd"] = str(
            Decimal(self.input_tokens) * Decimal("0.000001")
            + Decimal(self.output_tokens) * Decimal("0.000002")
        )
        return result


class Provider(Protocol):
    def call(self, request: dict[str, Any], retry: int, fault: str | None) -> Reply: ...


class FakeProvider:
    def call(self, request: dict[str, Any], retry: int, fault: str | None) -> Reply:
        if fault == "auth":
            raise ProviderFailure("simulated authentication failure", retryable=False)
        if fault == "timeout_always" or (fault == "timeout_once" and retry == 0):
            raise ProviderFailure("simulated transport timeout", retryable=True)
        if request["role"] == "optimizer":
            slot = request["candidate_slot"]
            detail = (
                "Check vendor/product identity."
                if slot == 0
                else "Check both vendor/product identity and version applicability."
            )
            prompt = (
                f"{detail} Return only JSON with applicable_cves. "
                f"Revision {request['iteration']}, variation {slot}, seed {request['seed']}."
            )
            text = json.dumps({"prompt": prompt})
        else:
            data = request["input"]
            # Fictional fixtures have a deliberately small text grammar. The fake
            # consumes the same unstructured input as the default live Executor.
            components = [
                Component(vendor, product, version)
                for vendor, product, version in re.findall(
                    r"(\w+) (\w+) version ([0-9]+(?:\.[0-9]+)*)", data["system_profile"]
                )
            ]
            advisories = [Advisory(**a) for a in data["advisories"]]
            prompt = request["prompt"]
            if "version applicability" in prompt:
                predicted = [a.id for a in advisories if any(a.applies(c) for c in components)]
            elif "vendor/product identity" in prompt:
                predicted = [
                    a.id for a in advisories if any(a.matches_product(c) for c in components)
                ]
            else:
                predicted = [a.id for a in advisories]
            text = json.dumps({"applicable_cves": predicted})
        finish = "stop"
        if fault == "malformed_always" or (fault == "malformed_once" and retry == 0):
            text = '{"incomplete":'
        elif fault == "reasoning_only":
            text, finish = "", "length"
        elif fault == "refusal":
            text, finish = "Simulated refusal", "refusal"
        return Reply(
            text=text,
            finish_reason=finish,
            provider_request_id="fake-" + digest(request)[:24],
            input_tokens=math.ceil(len(canonical(request)) / 4),
            output_tokens=math.ceil(len(text) / 4),
        )
