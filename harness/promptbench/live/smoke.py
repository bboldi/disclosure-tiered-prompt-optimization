"""A bounded, resumable real-model feedback round on fictional engineering fixtures."""

from __future__ import annotations

import shutil
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..domain import (
    BASELINE,
    TASK,
    ContractError,
    feedback,
    load_profiles,
    metrics,
    parse_answer,
    parse_prompt,
    score,
)
from ..runner import RunPaused, utc_now
from ..storage import IntegrityError, Store, atomic_write, canonical, digest
from .adapters import dollars, executor_body, normalize, optimizer_body, reservation
from .preflight import gpu_snapshot, local_identity, sources
from .progress import Progress
from .transport import OLLAMA, OPENROUTER, Transport, response_json


class Smoke:
    def __init__(
        self,
        root: Path,
        registry: dict[str, Any],
        fixture: dict[str, Any],
        transport: Transport,
        *,
        stop_after: int | None = None,
        max_cost_usd: str = "1",
        local_think: bool | None = None,
    ):
        self.store, self.transport = Store(root), transport
        self.registry, self.fixture = registry, fixture
        self.local = next(
            m for m in registry["local_models"] if m["requested_tag"] == "granite4.2:30b"
        )
        if local_think is not None:
            if type(local_think) is not bool:
                raise ValueError("local thinking must be a boolean or the omitted default")
            if "thinking" not in self.local.get("capabilities", []):
                raise ValueError("local model does not advertise thinking controls")
        self.local_think = local_think
        self.hosted = next(
            m
            for m in registry["hosted_models"]
            if m["requested_model"] == "deepseek/deepseek-v4-pro"
        )
        self.endpoint = next(
            e for e in self.hosted["endpoints"]["endpoints"] if e["tag"] == "streamlake/fp8"
        )
        self.stop_after, self.commits = stop_after, 0
        self.required: set[str] = set()
        self.session = uuid.uuid4().hex
        self.timer = time.perf_counter()
        self.checkpoint_index = 0
        self.prior_seconds = 0.0
        self.pending_timeout = 0
        self.max_cost = dollars(max_cost_usd)
        if not 0 < self.max_cost <= 10:
            raise ValueError("smoke allocation must be positive and within the USD 10 reserve")
        count = sum(p.partition == "optimization" for p in load_profiles(fixture))
        self.progress = Progress(
            self.store,
            {"baseline": count, "optimizer": 1, "proposed": count, "report": 1, "preflight": 7},
        )
        self.preflight_done = 0

    def checkpoint(self, pending_timeout: int = 0) -> None:
        self.pending_timeout = pending_timeout
        self.store.put(
            f"sessions/{self.session}/{self.checkpoint_index:04d}.json",
            {
                "timestamp_utc": utc_now(),
                "elapsed_seconds": time.perf_counter() - self.timer,
                "pending_timeout_seconds": pending_timeout,
            },
        )
        self.checkpoint_index += 1

    def remaining_seconds(self) -> float:
        return 600 - self.prior_seconds - (time.perf_counter() - self.timer)

    def accounting(self) -> dict[str, Any]:
        known, reserved, count = Decimal(0), Decimal(0), 0
        for name in self.store.names("work/*/attempts/*/request.json"):
            request = self.store.get(name)
            count += 1
            if request["provider"] != "openrouter":
                continue
            directory = name.rsplit("/", 1)[0]
            cost = None
            if self.store.exists(directory + "/response.json"):
                try:
                    cost = normalize(self.store.get(directory + "/response.json"), "openrouter")[
                        "reported_api_cost_usd"
                    ]
                except (ValueError, KeyError, TypeError):
                    pass
            if cost is None:
                reserved += dollars(request["reservation_usd"])
            else:
                known += dollars(cost)
        return {
            "known_cost_usd": str(known),
            "unresolved_reserved_usd": str(reserved),
            "attempts": count,
        }

    def metadata(
        self,
        name: str,
        url: str,
        body: dict[str, Any] | None = None,
        *,
        authenticated: bool = False,
    ) -> dict[str, Any]:
        task = "report" if name == "key-after" else "preflight"
        completed = None if task == "report" else self.preflight_done
        with self.progress.waiting(task, name, completed=completed):
            record = self.transport.request(url, body, authenticated=authenticated, timeout=20)
        self.store.put(
            f"metadata/{self.session}/{name}.json",
            {
                "request": {"url": url, "body": body, "authenticated": authenticated},
                "response": record,
            },
        )
        data = response_json(record)
        if task == "preflight":
            self.preflight_done += 1
        self.progress.emit(
            task,
            name,
            "saved",
            completed=None if task == "report" else self.preflight_done,
            elapsed=record["duration_ns"] / 1_000_000_000,
        )
        return data

    def verify_identities(self) -> None:
        version = self.metadata("ollama-version", OLLAMA + "/api/version")["version"]
        tags = self.metadata("ollama-tags", OLLAMA + "/api/tags")["models"]
        tag = next(t for t in tags if t["name"] == self.local["requested_tag"])
        details = self.metadata("ollama-show", OLLAMA + "/api/show", {"model": tag["name"]})
        if local_identity(tag, details, version) != self.local:
            raise IntegrityError(
                "local model digest, template, parameters or Ollama version changed"
            )
        catalog = self.metadata("openrouter-catalog", OPENROUTER + "/api/v1/models")["data"]
        hosted = next(m for m in catalog if m["id"] == self.hosted["requested_model"])
        if hosted.get("canonical_slug") != self.hosted["catalog"].get("canonical_slug"):
            raise IntegrityError("hosted model canonical revision changed")
        endpoints = self.metadata(
            "openrouter-endpoints",
            OPENROUTER + f"/api/v1/models/{self.hosted['requested_model']}/endpoints",
        )["data"]["endpoints"]
        endpoint = next(e for e in endpoints if e["tag"] == self.endpoint["tag"])
        for field in ("name", "model_id", "quantization", "context_length"):
            if endpoint.get(field) != self.endpoint.get(field):
                raise IntegrityError(f"hosted endpoint identity changed: {field}")
        if endpoint["status"] != 0:
            raise RunPaused("pinned hosted endpoint is not healthy")
        body = optimizer_body(self.hosted["requested_model"], self.endpoint["tag"], {})
        reservation(body, endpoint)  # reject a tariff beyond the enforced routing cap
        self.store.put(f"metadata/{self.session}/gpu.json", gpu_snapshot())
        self.preflight_done += 1
        key = self.metadata("key-before", OPENROUTER + "/api/v1/key", authenticated=True)["data"]
        if dollars(key["usage"]) >= 60:
            raise RunPaused("dedicated key has reached the USD 60 study ceiling")
        self.start_key_usage = dollars(key["usage"])
        if self.start_key_usage >= 48:
            self.store.put(
                f"metadata/{self.session}/funding-notice.json",
                {
                    "threshold_usd": "48",
                    "key_usage_usd": str(self.start_key_usage),
                    "action": "Notify the operator before funding becomes a blocker; current study cap remains USD 60.",
                },
            )
            print(
                "FUNDING NOTICE: dedicated-key usage reached USD 48; review the remaining-stage forecast.",
                flush=True,
            )

    def call(
        self, key: str, provider: str, body: dict[str, Any], allowed_ids: set[str] | None = None
    ) -> dict[str, Any]:
        directory = "work/" + digest({"key": key, "body": body, "provider": provider})
        self.required.add(directory)
        self.store.put(directory + "/spec.json", {"key": key, "provider": provider, "body": body})
        if self.store.exists(directory + "/result.json"):
            result = self.store.get(directory + "/result.json")
            if (
                digest(self.store.get(result["attempt"] + "/response.json"))
                != result["response_sha256"]
            ):
                raise IntegrityError("committed raw response changed")
            self.progress.emit(key.split("/", 1)[0], key, "reused saved result")
            return dict(result)
        url = (
            OLLAMA + "/api/chat"
            if provider == "ollama"
            else OPENROUTER + "/api/v1/chat/completions"
        )
        for index in range(2):
            attempt = f"{directory}/attempts/{index:04d}"
            if self.store.exists(attempt + "/request.json") and not self.store.exists(
                attempt + "/response.json"
            ):
                self.store.put(
                    attempt + "/unknown.json",
                    {
                        "outcome": "unknown",
                        "billing": "reservation retained",
                        "reason": "intent without durable response",
                    },
                )
                continue
            if not self.store.exists(attempt + "/response.json"):
                accounts = self.accounting()
                bound = reservation(body, self.endpoint) if provider == "openrouter" else Decimal(0)
                if accounts["attempts"] >= 12 or self.remaining_seconds() < 2:
                    raise RunPaused("smoke attempt/time cap reached")
                if (
                    dollars(accounts["known_cost_usd"])
                    + dollars(accounts["unresolved_reserved_usd"])
                    + bound
                    > self.max_cost
                ):
                    raise RunPaused(
                        f"USD {self.max_cost} smoke cap includes charges and unknown-attempt reservations"
                    )
                if (
                    self.start_key_usage
                    + dollars(accounts["known_cost_usd"])
                    + dollars(accounts["unresolved_reserved_usd"])
                    + bound
                    > 60
                ):
                    raise RunPaused("request reservation exceeds the study ceiling")
                if shutil.disk_usage(self.store.root).free < 1_073_741_824:
                    raise RunPaused("less than 1 GiB of free disk space remains")
                timeout = min(120, int(self.remaining_seconds()))
                self.store.put(
                    attempt + "/request.json",
                    {
                        "provider": provider,
                        "url": url,
                        "body": body,
                        "reservation_usd": str(bound),
                        "timeout_seconds": timeout,
                        "started_utc": utc_now(),
                    },
                )
                self.checkpoint(timeout)
                self.store.put(attempt + "/gpu-before.json", gpu_snapshot())
                with self.progress.waiting(key.split("/", 1)[0], key):
                    record = self.transport.request(
                        url, body, authenticated=provider == "openrouter", timeout=timeout
                    )
                self.store.put(attempt + "/response.json", record)
                self.store.put(attempt + "/gpu-after.json", gpu_snapshot())
                self.checkpoint()
            record = self.store.get(attempt + "/response.json")
            if record["http_status"] != 200 or record["error"]:
                self.store.put(
                    attempt + "/disposition.json",
                    {
                        "status": "transport_or_http_failure",
                        "http_status": record["http_status"],
                        "error": record["error"],
                    },
                )
                # No automatic retry of unknown billing, auth errors or overloaded live servers.
                raise RunPaused("live transport/API failure; raw evidence and reservation retained")
            reply = normalize(record, provider)
            self.store.put(attempt + "/normalized.json", reply)
            expected_models = {body["model"]}
            if provider == "openrouter":
                canonical_model = self.hosted["catalog"].get("canonical_slug")
                if canonical_model:
                    expected_models.add(canonical_model)
                if reply["actual_provider"] != self.endpoint["provider_name"]:
                    raise IntegrityError("returned hosted provider differs from pinned endpoint")
            if reply["actual_model"] not in expected_models:
                raise IntegrityError("returned model differs from requested model")
            try:
                if provider == "openrouter":
                    value: Any = parse_prompt(reply["text"], reply["finish_reason"])
                else:
                    value = parse_answer(
                        reply["text"], allowed_ids or set(), reply["finish_reason"]
                    )
                status, reason = "valid", None
            except (ContractError, TypeError) as exc:
                value, status, reason = None, "invalid_output", str(exc)
            self.store.put(attempt + "/disposition.json", {"status": status, "reason": reason})
            if provider == "openrouter" and status != "valid":
                continue
            result = {
                "key": key,
                "value": value,
                "status": status,
                "attempt": attempt,
                "response_sha256": digest(record),
            }
            self.store.put(directory + "/result.json", result)
            self.commits += 1
            self.progress.emit(
                key.split("/", 1)[0],
                key,
                "committed " + status,
                elapsed=record["duration_ns"] / 1_000_000_000,
            )
            if self.stop_after and self.commits >= self.stop_after:
                raise KeyboardInterrupt("requested live-smoke checkpoint interruption")
            return result
        raise RunPaused("smoke request exhausted its two physical attempts")

    def initialize(self) -> None:
        self.store.put(
            "manifest.json",
            {
                "protocol": "live-fixture-feedback-round-v1",
                "tier": 2,
                "local_identity": self.local,
                "local_think": self.local_think,
                "hosted_identity": self.hosted,
                "endpoint": self.endpoint,
                "fixture_sha256": digest(self.fixture),
                "source_sha256": digest(sources()),
                "max_cost_usd": str(self.max_cost),
                "max_attempts": 12,
                "cumulative_controller_seconds_cap": 600,
                "scope": "three optimization engineering profiles; no validation/test study claims",
                "parent_manifest_sha256": self.store.get("lineage/continuation.json")[
                    "parent_manifest_sha256"
                ]
                if self.store.exists("lineage/continuation.json")
                else None,
            },
        )
        self.store.put("inputs/source.json", sources())
        self.store.put("inputs/fixture.json", self.fixture)

    def prepare_continuation(self, parent_root: Path, reason: str) -> dict[str, Any]:
        """Import unchanged requests/results under an explicit new operational budget."""
        parent = Store(parent_root)
        if parent.root == self.store.root:
            raise IntegrityError("continuation needs a new run directory")
        with parent.lock(), self.store.lock():
            if any(self.store.root.iterdir()):
                raise IntegrityError("continuation target must be empty")
            manifest = parent.get("manifest.json")
            if parent.exists("completion.json"):
                raise IntegrityError("a completed run does not need a budget continuation")
            if self.max_cost <= dollars(manifest["max_cost_usd"]):
                raise IntegrityError("budget continuation must increase the recorded cap")
            for field, value in (
                ("local_identity", self.local),
                ("local_think", self.local_think),
                ("hosted_identity", self.hosted),
                ("endpoint", self.endpoint),
                ("fixture_sha256", digest(self.fixture)),
            ):
                if manifest.get(field) != value:
                    raise IntegrityError(f"cannot import changed scientific inputs: {field}")
            previous_source = parent.get("inputs/source.json")
            if digest(previous_source) != manifest["source_sha256"]:
                raise IntegrityError("parent source fingerprint is invalid")
            if digest(parent.get("inputs/fixture.json")) != manifest["fixture_sha256"]:
                raise IntegrityError("parent fixture fingerprint is invalid")
            current_source = sources()
            changed = sorted(
                name
                for name in set(previous_source) | set(current_source)
                if previous_source.get(name) != current_source.get(name)
            )
            if not set(changed) <= {"live/smoke.py", "live/__main__.py", "live/progress.py"}:
                raise IntegrityError(
                    "budget continuation cannot change adapters, parsers, scoring or model inputs"
                )
            imported = {}
            for folder in ("work", "evaluations", "prompts", "feedback", "sessions"):
                for name in parent.names(folder + "/**/*.json"):
                    imported[name] = self.store.put(name, parent.get(name))
            lineage = {
                "created_utc": utc_now(),
                "parent_run": str(parent.root),
                "parent_manifest_sha256": digest(manifest),
                "reason": reason,
                "old_cap_usd": manifest["max_cost_usd"],
                "new_cap_usd": str(self.max_cost),
                "changed_controller_files": changed,
                "imported_artifact_sha256": imported,
                "semantics": "Adopt existing request/result evidence and all cost/time reservations; do not regenerate completed local work or claim original attempts used new source.",
            }
            self.store.put("lineage/continuation.json", lineage)
            self.store.put("lineage/parent_manifest.json", manifest)
            self.store.put("lineage/parent_source.json", previous_source)
            self.initialize()
            return {
                "target": str(self.store.root),
                "imported_artifacts": len(imported),
                "accounting": self.accounting(),
                "max_cost_usd": str(self.max_cost),
            }

    def run(self) -> dict[str, Any]:
        with self.store.lock():
            self.initialize()
            self.progress.emit("preflight", "identity checks", "starting", completed=0)
            for folder in sorted((self.store.root / "sessions").glob("*")):
                names = sorted(folder.glob("*.json"))
                if names:
                    last = self.store.get(str(names[-1].relative_to(self.store.root)))
                    self.prior_seconds += last["elapsed_seconds"] + last["pending_timeout_seconds"]
            self.checkpoint()
            try:
                self.verify_identities()
                self.checkpoint()
                profiles = [p for p in load_profiles(self.fixture) if p.partition == "optimization"]
                baseline_rows = []
                self.store.put("prompts/baseline.json", {"prompt": BASELINE})
                for profile in profiles:
                    result = self.call(
                        "baseline/" + profile.id,
                        "ollama",
                        executor_body(
                            self.local["requested_tag"], BASELINE, profile, think=self.local_think
                        ),
                        {a.id for a in profile.advisories},
                    )
                    row = score(profile, result["value"], result["status"])
                    self.store.put("evaluations/baseline-" + profile.id + ".json", row)
                    baseline_rows.append(row)
                permitted = {
                    "task": TASK,
                    "feedback": feedback(baseline_rows, profiles, 2),
                    "history": [{"prompt": BASELINE, "metrics": metrics(baseline_rows)}],
                }
                self.store.put("feedback/tier2.json", permitted)
                proposal = self.call(
                    "optimizer/1",
                    "openrouter",
                    optimizer_body(self.hosted["requested_model"], self.endpoint["tag"], permitted),
                )
                self.store.put("prompts/proposal.json", {"prompt": proposal["value"]})
                tuned_rows = []
                for profile in profiles:
                    result = self.call(
                        "proposed/" + profile.id,
                        "ollama",
                        executor_body(
                            self.local["requested_tag"],
                            proposal["value"],
                            profile,
                            think=self.local_think,
                        ),
                        {a.id for a in profile.advisories},
                    )
                    row = score(profile, result["value"], result["status"])
                    self.store.put("evaluations/proposed-" + profile.id + ".json", row)
                    tuned_rows.append(row)
                results = {
                    "evidence_kind": "live_engineering_fixture_only",
                    "baseline_metrics": metrics(baseline_rows),
                    "proposed_metrics": metrics(tuned_rows),
                    "required_work": sorted(self.required),
                    "manifest_sha256": digest(self.store.get("manifest.json")),
                }
                if set(self.store.names("work/*/result.json")) != {
                    w + "/result.json" for w in self.required
                }:
                    raise IntegrityError("completion workset does not match committed work")
                self.store.put("completion.json", results)
                self.export(results)
                self.progress.emit("report", "feedback round", "completed")
                return results
            finally:
                self.checkpoint(self.pending_timeout)

    def export(self, results: dict[str, Any]) -> None:
        account = self.accounting()
        key = self.metadata("key-after", OPENROUTER + "/api/v1/key", authenticated=True)["data"]
        before = self.store.get(f"metadata/{self.session}/key-before.json")["response"]
        before_usage = dollars(response_json(before)["data"]["usage"])
        audit = {
            **account,
            "current_session_key_usage_delta_usd": str(dollars(key["usage"]) - before_usage),
            "key_usage_total_usd": str(dollars(key["usage"])),
            "reconciliation_note": "Session delta is separate from cumulative run cost; charges can settle later.",
            "controller_seconds_measured_or_reserved": self.prior_seconds
            + time.perf_counter()
            - self.timer,
        }
        self.store.put(f"reports/{self.session}.json", {"results": results, "accounting": audit})
        export_rows = [
            self.store.get(name) for name in self.store.names("work/*/attempts/*/normalized.json")
        ]
        atomic_write(
            self.store.root / "exports/usage.jsonl",
            "".join(canonical(r) + "\n" for r in export_rows).encode(),
            immutable=False,
        )
        atomic_write(
            self.store.root / "CONCLUSIONS.md",
            (
                "# Live engineering smoke\n\nCompleted one Tier-2 feedback round on three fictional optimization profiles. This is adapter/recovery evidence, not pilot or manuscript task-performance evidence.\n\n"
                + f"Committed requests: {len(self.required)}. Attempts: {account['attempts']}. Reported API cost: USD {account['known_cost_usd']}. Unresolved reservations: USD {account['unresolved_reserved_usd']}.\n\n"
                + "Exact local digest/template/runtime and hosted catalog/endpoint identities are frozen in manifest.json; requested/returned model/provider IDs and raw usage are retained per attempt. Hosted weight hashes are unavailable. Reports and usage exports are derived from retained raw responses.\n"
            ).encode(),
            immutable=False,
        )
