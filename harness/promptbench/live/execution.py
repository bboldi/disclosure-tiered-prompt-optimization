"""Campaign-wide attempt journal, reservations, bounded retries and measured run clock."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..domain import ContractError, parse_answer, parse_prompt
from ..runner import Hook, RunPaused, no_hook, utc_now
from ..storage import IntegrityError, Store, digest
from .adapters import dollars, normalize
from .process import invoke
from .progress import Progress
from .telemetry import Telemetry
from .transport import OPENROUTER, Transport, read_key, response_json


class StageExhausted(RunPaused):
    """The predeclared exploratory stage allocation has ended."""


def charge(record: dict[str, Any]) -> Decimal | None:
    try:
        cost = json.loads(record["body_text"]).get("usage", {}).get("cost")
        return dollars(cost) if cost is not None else None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def ledger(root: Path, *, study: bool = False) -> dict[str, Any]:
    """Deduplicate adopted journal records so continuations do not duplicate expenditure."""
    attempts: dict[str, tuple[Decimal, Decimal | None]] = {}
    # Study evidence is one level below the phase folder. Never bill offline fixtures.
    pattern = "*/work/*/attempts/*/request.json" if study else "work/*/attempts/*/request.json"
    for path in sorted(root.glob(pattern)):
        store = Store(path.parent)
        request = store.get("request.json")
        if request.get("provider") != "openrouter":
            continue
        identity = digest(request)
        reserved = dollars(request["reservation_usd"])
        known = charge(store.get("response.json")) if store.exists("response.json") else None
        if identity in attempts:
            old_reservation, old_cost = attempts[identity]
            if reserved != old_reservation or (
                known is not None and old_cost is not None and known != old_cost
            ):
                raise IntegrityError("adopted attempts have inconsistent billing evidence")
            known = old_cost if known is None else known
        attempts[identity] = reserved, known
    return {
        "reported_cost_usd": str(
            sum((cost for _, cost in attempts.values() if cost is not None), Decimal(0))
        ),
        "reserved_unknown_usd": str(
            sum((bound for bound, cost in attempts.values() if cost is None), Decimal(0))
        ),
        "hosted_attempts": len(attempts),
    }


def _without_tariff(spec: dict[str, Any]) -> dict[str, Any]:
    """The request identity with every price-derived field removed."""
    import copy

    view = copy.deepcopy(spec)
    view.get("condition", {}).pop("reservation_usd", None)
    view.get("condition", {}).get("endpoint", {}).pop("pricing", None)
    view.get("body", {}).get("provider", {}).pop("max_price", None)
    return view


def tariff_only_difference(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """True when two request specs differ only in fields derived from a hosted tariff."""
    return old != new and _without_tariff(old) == _without_tariff(new)


class Execution:
    def __init__(
        self,
        store: Store,
        env_file: Path,
        *,
        max_seconds: float,
        max_cost_usd: str,
        max_attempts: int,
        study_root: Path,
        runtime: Path | None = None,
        progress: Progress | None = None,
        worker: Callable[..., dict[str, Any]] = invoke,
        hook: Hook = no_hook,
    ):
        self.store, self.env_file, self.study_root = store, env_file, study_root
        self.max_seconds, self.max_cost = max_seconds, dollars(max_cost_usd)
        if (
            not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 259200
            or not 0 < self.max_cost <= 60
            or max_attempts < 1
        ):
            raise ValueError("invalid campaign limits")
        self.max_attempts, self.runtime, self.progress, self.worker, self.hook = (
            max_attempts,
            runtime,
            progress,
            worker,
            hook,
        )
        self.session, self.checkpoints = uuid.uuid4().hex, 0
        self.started = time.perf_counter()
        self.prior_seconds, self.pending_timeout = 0.0, 0.0
        self.key_usage = Decimal(0)
        self.monitor: Telemetry | None = None
        self.key_reader: Callable[[], dict[str, Any]] | None = None
        self.stage_time_left: Callable[[], float] | None = None
        for folder in (store.root / "sessions").glob("*"):
            names = sorted(folder.glob("[0-9]*.json"))
            if names:
                last = store.get(str(names[-1].relative_to(store.root)))
                self.prior_seconds += last["elapsed_seconds"] + last["pending_timeout_seconds"]
        self.checkpoint()

    def checkpoint(self, pending_timeout: float = 0) -> None:
        self.pending_timeout = pending_timeout
        self.store.put(
            f"sessions/{self.session}/{self.checkpoints:06d}.json",
            {
                "timestamp_utc": utc_now(),
                "elapsed_seconds": time.perf_counter() - self.started,
                "pending_timeout_seconds": pending_timeout,
            },
        )
        self.checkpoints += 1

    def remaining_seconds(self) -> float:
        return self.max_seconds - self.prior_seconds - (time.perf_counter() - self.started)

    def refresh_key(self) -> Decimal:
        if self.key_reader is not None:
            data = self.key_reader()
        else:
            response = Transport(read_key(self.env_file)).request(
                OPENROUTER + "/api/v1/key", authenticated=True, timeout=20
            )
            self.store.put(
                f"key-usage/{utc_now().replace(':', '-')}-{uuid.uuid4().hex}.json", response
            )
            data = response_json(response)
        self.key_usage = dollars(data["data"]["usage"])
        if self.key_usage >= 48:
            print(
                "FUNDING NOTICE: study key usage reached USD 48; review remaining-stage funds.",
                flush=True,
            )
        return self.key_usage

    def accounts(self) -> dict[str, Any]:
        return ledger(self.store.root)

    def admission(self, provider: str, reservation: Decimal) -> float:
        if self.monitor:
            self.monitor.ensure_healthy()
        import shutil

        if shutil.disk_usage(self.store.root).free < 1_073_741_824:
            raise RunPaused("less than 1 GiB of free disk remains")
        if len(self.store.names("work/*/attempts/*/request.json")) >= self.max_attempts:
            raise RunPaused("absolute physical-attempt cap reached")
        remaining = self.remaining_seconds() - 5  # reserve bounded worker cleanup/reporting time
        if self.stage_time_left is not None:
            stage_remaining = self.stage_time_left() - 5
            if stage_remaining < 1:
                raise StageExhausted("exploratory stage time allocation exhausted")
            remaining = min(remaining, stage_remaining)
        if remaining < 1:
            raise RunPaused("cumulative running-time cap reached")
        if provider == "openrouter":
            local = self.accounts()
            if (
                dollars(local["reported_cost_usd"])
                + dollars(local["reserved_unknown_usd"])
                + reservation
                > self.max_cost
            ):
                raise RunPaused("campaign cost cap includes unresolved request reservations")
            self.refresh_key()
            global_account = ledger(self.study_root, study=True)
            if (
                max(self.key_usage, dollars(global_account["reported_cost_usd"]))
                + dollars(global_account["reserved_unknown_usd"])
                + reservation
                > 60
            ):
                raise RunPaused("study-wide USD 60 cap includes earlier unknown reservations")
        return min(300.0, remaining, self.remaining_seconds() - 5)

    def call(
        self,
        key: str,
        condition: dict[str, Any],
        body: dict[str, Any],
        *,
        role: str,
        allowed_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        if role not in ("executor", "optimizer"):
            raise ValueError("unsupported inference role")
        provider = condition["provider"]
        spec = {
            "key": key,
            "condition": condition,
            "body": body,
            "role": role,
            "allowed_ids": sorted(allowed_ids or set()),
        }
        directory = "work/" + digest(spec)
        pointer = "logical-keys/" + digest(key) + ".json"
        if self.store.exists(pointer) and self.store.get(pointer)["spec_sha256"] != digest(spec):
            prior = self.store.get(pointer)["spec_sha256"]
            old_spec = self.store.get(f"work/{prior}/spec.json")
            if not tariff_only_difference(old_spec, spec):
                raise IntegrityError(f"logical key already bound to a different request: {key}")
            if self.store.exists(f"work/{prior}/result.json"):
                # Committed before an accepted tariff change: the result stands as recorded.
                spec, directory = old_spec, "work/" + prior
            else:
                # Never committed (for example refused at the old price ceiling): the key now
                # binds to the repriced request; the old attempts stay as evidence.
                self.store.put(
                    pointer,
                    {
                        "spec_sha256": digest(spec),
                        "superseded_spec_sha256": prior,
                        "reason": "accepted hosted tariff change; only price fields differ",
                    },
                    immutable=False,
                )
        if not self.store.exists(pointer):
            self.store.put(pointer, {"spec_sha256": digest(spec)})
        self.store.put(directory + "/spec.json", spec)
        self.hook("planned", key)
        if self.store.exists(directory + "/result.json"):
            result = self.store.get(directory + "/result.json")
            if result["spec_sha256"] != digest(spec) or (
                result["response_sha256"] is not None
                and digest(self.store.get(result["attempt"] + "/response.json"))
                != result["response_sha256"]
            ):
                raise IntegrityError("committed response/spec changed")
            return dict(result)
        for index in range(3):
            attempt = f"{directory}/attempts/{index:04d}"
            if self.store.exists(attempt + "/request.json") and not self.store.exists(
                attempt + "/response.json"
            ):
                self.store.put(
                    attempt + "/unknown.json",
                    {
                        "status": "unknown",
                        "reason": "dispatch intent without durable response; reservation retained",
                    },
                )
                continue
            if not self.store.exists(attempt + "/response.json"):
                reserved = (
                    dollars(condition.get("reservation_usd", "0"))
                    if provider == "openrouter"
                    else Decimal(0)
                )
                timeout = min(
                    self.admission(provider, reserved), condition.get("timeout_seconds", 120)
                )
                if timeout < 1:
                    raise RunPaused("no request time remains after accounting refresh")
                self.store.put(
                    attempt + "/request.json",
                    {
                        "provider": provider,
                        "url": condition["url"],
                        "body": body,
                        "reservation_usd": str(reserved),
                        "timeout_seconds": max(1, int(timeout)),
                        "started_utc": utc_now(),
                        "dispatch_id": uuid.uuid4().hex,
                        "spec_sha256": digest(spec),
                    },
                )
                self.checkpoint(timeout)
                self.hook("intent", key)
                if self.progress:
                    with self.progress.waiting(key.split("/", 1)[0], key):
                        self.worker(
                            self.store,
                            attempt,
                            self.env_file,
                            timeout=timeout,
                            runtime=self.runtime,
                        )
                else:
                    self.worker(
                        self.store, attempt, self.env_file, timeout=timeout, runtime=self.runtime
                    )
                self.hook("provider_returned", key)
                self.checkpoint()
                if not self.store.exists(attempt + "/response.json"):
                    self.store.put(
                        attempt + "/unknown.json",
                        {
                            "status": "unknown",
                            "reason": "dispatch intent without durable response; reservation retained",
                        },
                    )
                    continue
            response = self.store.get(attempt + "/response.json")
            self.hook("response", key)
            http_status = response["http_status"]
            if http_status != 200 or response["error"]:
                retryable = http_status in (429, 500, 502, 503, 504) or http_status is None
                self.store.put(
                    attempt + "/disposition.json",
                    {
                        "status": "transport_or_http_failure",
                        "retryable": retryable,
                        "http_status": http_status,
                        "error": response["error"],
                    },
                )
                if not retryable:
                    if http_status in (400, 413, 422):
                        # Unsupported settings or oversized requests are terminal evidence
                        # for this item; do not stall an unattended exploratory campaign.
                        result = {
                            "key": key,
                            "status": "request_rejected",
                            "value": None,
                            "attempt": attempt,
                            "response_sha256": digest(response),
                            "spec_sha256": digest(spec),
                            "terminal_http_status": http_status,
                        }
                        self.store.put(directory + "/result.json", result)
                        return result
                    pause_name = attempt + "/operator-pause.json"
                    if (
                        http_status in (401, 402, 403)
                        and self.store.exists(pause_name)
                        and self.store.get(pause_name)["session"] != self.session
                    ):
                        # A later operator resume may follow renewed credentials/credits.
                        # The failed attempt and all original caps remain in force.
                        continue
                    if not self.store.exists(pause_name):
                        self.store.put(
                            pause_name, {"session": self.session, "http_status": http_status}
                        )
                    raise RunPaused(
                        f"permanent provider/API error {http_status}; raw evidence retained"
                    )
                if index < 2:
                    retry_after = response.get("headers", {}).get("retry-after", "")
                    delay = float(retry_after) if str(retry_after).isdigit() else 2**index
                    if delay > 30 or delay + 5 >= self.remaining_seconds():
                        raise RunPaused("provider retry delay exceeds current bounded retry window")
                    print(f"[retry {index + 1}/2] {key}: waiting {delay:.1f}s", flush=True)
                    time.sleep(delay)
                continue
            try:
                reply = normalize(response, provider)
            except (ValueError, TypeError, KeyError) as exc:
                self.store.put(
                    attempt + "/disposition.json",
                    {"status": "provider_envelope_failure", "reason": str(exc)},
                )
                continue
            self.store.put(attempt + "/normalized.json", reply)
            if (
                reply["actual_model"] not in condition["expected_models"]
                or reply["actual_provider"] != condition["expected_provider"]
            ):
                raise IntegrityError("returned model/provider differs from frozen condition")
            try:
                value: Any = (
                    parse_prompt(reply["text"], reply["finish_reason"])
                    if role == "optimizer"
                    else parse_answer(
                        reply["text"],
                        allowed_ids or set(),
                        reply["finish_reason"],
                        scaffold=condition.get("output_scaffold", False),
                    )
                )
                status, reason = "valid", None
            except (ContractError, TypeError) as exc:
                value, status, reason = None, "invalid_output", str(exc)
            self.store.put(attempt + "/disposition.json", {"status": status, "reason": reason})
            if role == "optimizer" and status != "valid":
                continue
            result = {
                "key": key,
                "status": status,
                "value": value,
                "attempt": attempt,
                "response_sha256": digest(response),
                "spec_sha256": digest(spec),
            }
            self.hook("before_commit", key)
            self.store.put(directory + "/result.json", result)
            self.hook("committed", key)
            if self.progress:
                self.progress.emit(
                    key.split("/", 1)[0],
                    key,
                    "committed " + status,
                    elapsed=response["duration_ns"] / 1e9,
                )
            return result
        # Exhausted transient/envelope failures are outcomes, not an endless restart loop.
        # They are never interpreted as a correct empty prediction or a valid candidate.
        result = {
            "key": key,
            "status": "transport_failure",
            "value": None,
            "attempt": attempt,
            "response_sha256": digest(self.store.get(attempt + "/response.json"))
            if self.store.exists(attempt + "/response.json")
            else None,
            "spec_sha256": digest(spec),
        }
        self.store.put(directory + "/result.json", result)
        return result
