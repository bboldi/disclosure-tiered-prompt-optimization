"""Durable per-request execution, typed failures, and bounded recovery."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .domain import ContractError, parse_answer, parse_prompt
from .providers import Provider, ProviderFailure
from .storage import IntegrityError, Store, digest

Hook = Callable[[str, str], None]


class RunPaused(RuntimeError):
    """External/permanent failures or resource caps require an explicit disposition."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def no_hook(point: str, work_id: str) -> None:
    pass


class Runner:
    def __init__(
        self,
        store: Store,
        provider: Provider,
        *,
        max_retries: int,
        max_attempts: int,
        faults: dict[str, str],
        hook: Hook = no_hook,
    ):
        self.store = store
        self.provider = provider
        self.max_retries = max_retries
        self.max_attempts = max_attempts
        self.faults = faults
        self.hook = hook

    def event(self, kind: str, **details: Any) -> None:
        self.store.put(
            f"events/{time.time_ns()}-{uuid.uuid4().hex}.json",
            {
                "timestamp_utc": utc_now(),
                "kind": kind,
                **details,
            },
        )

    def execute(self, key: str, request: dict[str, Any]) -> dict[str, Any]:
        work_id = digest({"key": key, "request": request})
        directory = f"work/{work_id}"
        spec = {"key": key, "request": request, "request_sha256": digest(request)}
        self.store.put(f"{directory}/spec.json", spec)
        self.hook("planned", key)
        result_name = f"{directory}/result.json"
        if self.store.exists(result_name):
            result = self.store.get(result_name)
            self._validate_result(result, spec, directory)
            return dict(result)
        attempts = self.store.names(f"{directory}/attempts/*/request.json")
        retry = 0
        for name in attempts:
            attempt_dir = name.rsplit("/", 1)[0]
            intent = self.store.get(name)
            if intent["request_sha256"] != spec["request_sha256"]:
                raise IntegrityError("attempt request differs from work specification")
            response_name = f"{attempt_dir}/response.json"
            if self.store.exists(response_name):
                response = self.store.get(response_name)
                disposition = self._interpret(request, response)
                self.store.put(f"{attempt_dir}/disposition.json", disposition)
                if disposition["status"] != "retryable":
                    return self._commit(directory, spec, attempt_dir, response, disposition)
                retry = max(retry, intent["retry"] + 1)
            else:
                # An intent without a durable response may already have been billed.
                self.store.put(
                    f"{attempt_dir}/outcome_unknown.json",
                    {
                        "status": "outcome_unknown",
                        "reason": "no durable response after interruption",
                        "actual_api_cost_usd": "0",
                        "synthetic_cost_usd": None,
                        "request_sha256": intent["request_sha256"],
                    },
                )
                # Unknown attempts do not change the logical fake-provider retry state.
                # They DO count against the absolute attempt cap.
                self.event("uncertain_attempt", key=key, attempt=attempt_dir)
        while retry <= self.max_retries:
            if len(self.store.names("work/*/attempts/*/request.json")) >= self.max_attempts:
                raise RunPaused("absolute provider-attempt budget exhausted")
            index = len(self.store.names(f"{directory}/attempts/*/request.json"))
            attempt_dir = f"{directory}/attempts/{index:04d}"
            started = utc_now()
            self.store.put(
                f"{attempt_dir}/request.json",
                {
                    **spec,
                    "attempt_index": index,
                    "retry": retry,
                    "started_utc": started,
                    "model": "deterministic-fake-v1",
                },
            )
            self.event("request_intent", key=key, attempt=attempt_dir)
            self.hook("intent", key)
            start = time.perf_counter_ns()
            try:
                reply = self.provider.call(request, retry, self.faults.get(key))
                response = {"status": "response", "reply": reply.record()}
            except ProviderFailure as exc:
                response = {
                    "status": "transport_error",
                    "error": str(exc),
                    "retryable": exc.retryable,
                    "actual_api_cost_usd": "0",
                    "synthetic_cost_usd": None,
                }
            response.update(
                {
                    "started_utc": started,
                    "finished_utc": utc_now(),
                    "duration_ns": time.perf_counter_ns() - start,
                    "request_sha256": spec["request_sha256"],
                }
            )
            self.hook("provider_returned", key)
            self.store.put(f"{attempt_dir}/response.json", response)
            self.event("response_saved", key=key, attempt=attempt_dir, status=response["status"])
            self.hook("response", key)
            disposition = self._interpret(request, response)
            self.store.put(f"{attempt_dir}/disposition.json", disposition)
            if disposition["status"] != "retryable":
                return self._commit(directory, spec, attempt_dir, response, disposition)
            retry += 1
        if request["role"] == "optimizer":
            raise RunPaused(f"optimizer protocol/transport retries exhausted: {key}")
        # Failure is explicit and no normal empty prediction is fabricated.
        response = self.store.get(f"{attempt_dir}/response.json")
        disposition = {"status": "transport_failure", "value": None}
        return self._commit(directory, spec, attempt_dir, response, disposition)

    def _interpret(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        if response["request_sha256"] != digest(request):
            raise IntegrityError("response belongs to another request")
        if response["status"] == "transport_error":
            if not response["retryable"]:
                raise RunPaused(response["error"])
            return {"status": "retryable", "value": None, "reason": response["error"]}
        reply = response["reply"]
        try:
            if request["role"] == "optimizer":
                value: Any = parse_prompt(reply["text"], reply["finish_reason"])
            else:
                value = parse_answer(
                    reply["text"],
                    {a["id"] for a in request["input"]["advisories"]},
                    reply["finish_reason"],
                )
            return {"status": "valid", "value": value}
        except ContractError as exc:
            return {
                "status": "retryable" if request["role"] == "optimizer" else "invalid_output",
                "value": None,
                "reason": str(exc),
            }

    def _commit(
        self,
        directory: str,
        spec: dict[str, Any],
        attempt_dir: str,
        response: dict[str, Any],
        disposition: dict[str, Any],
    ) -> dict[str, Any]:
        result = {
            **disposition,
            "key": spec["key"],
            "request_sha256": spec["request_sha256"],
            "response_sha256": digest(response),
            "attempt": attempt_dir,
        }
        self.hook("before_commit", spec["key"])
        self.store.put(f"{directory}/result.json", result)
        self.event("work_committed", key=spec["key"], status=result["status"])
        self.hook("committed", spec["key"])
        return result

    def _validate_result(
        self,
        result: dict[str, Any],
        spec: dict[str, Any],
        directory: str,
    ) -> None:
        if result["key"] != spec["key"] or result["request_sha256"] != spec["request_sha256"]:
            raise IntegrityError("committed result differs from work specification")
        if not result["attempt"].startswith(directory + "/attempts/"):
            raise IntegrityError("result references another work item")
        response = self.store.get(result["attempt"] + "/response.json")
        if result["response_sha256"] != digest(response):
            raise IntegrityError("committed response changed")
