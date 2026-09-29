"""Offline reconstruction from requests and raw replies, independently of runner summaries."""

from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..benchmark.model import score
from ..domain import ContractError, metrics, parse_answer, parse_prompt
from ..storage import IntegrityError, Store, atomic_write, canonical, digest
from .adapters import normalize
from .campaign import profiles_from
from .conditions import executor_request
from .execution import charge, ledger
from .telemetry import summarize


def reconstruct(root: Path, output: Path) -> dict[str, Any]:
    store, target = Store(root), Store(output)
    profiles = {p.id: p for p in profiles_from(store)}
    attempts, reconstructed = [], []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown, completed, invalid = 0, 0, 0
    for spec_name in store.names("work/*/spec.json"):
        spec = store.get(spec_name)
        folder = spec_name.rsplit("/", 1)[0]
        if folder != "work/" + digest(spec):
            raise IntegrityError("work directory differs from immutable specification")
        if store.get("logical-keys/" + digest(spec["key"]) + ".json")["spec_sha256"] != digest(
            spec
        ):
            raise IntegrityError("logical-key identity differs")
        replies = {}
        for name in store.names(folder + "/attempts/*/request.json"):
            request = store.get(name)
            attempt = name.rsplit("/", 1)[0]
            if request["body"] != spec["body"] or request["spec_sha256"] != digest(spec):
                raise IntegrityError("attempt differs from requested work")
            row: dict[str, Any] = {
                "attempt": attempt,
                "logical_key": spec["key"],
                "stage": spec["key"].split("/")[0],
                "provider": request["provider"],
                "role": spec["role"],
                "condition_id": spec["condition"]["id"],
                "dispatch_id": request["dispatch_id"],
                "started_utc": request["started_utc"],
                "timeout_seconds": request["timeout_seconds"],
                "reservation_usd": request["reservation_usd"],
                "status": "unknown",
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "reported_cost_usd": None,
                "http_seconds": None,
                "http_status": None,
                "returned_model": None,
                "returned_provider": None,
                "generation_id": None,
                "finish_reason": None,
            }
            if store.exists(attempt + "/response.json"):
                response = store.get(attempt + "/response.json")
                row.update(
                    http_status=response["http_status"],
                    http_seconds=response["duration_ns"] / 1e9,
                    status="response",
                )
                cost = charge(response) if request["provider"] == "openrouter" else 0
                row["reported_cost_usd"] = str(cost) if cost is not None else None
                try:
                    reply = normalize(response, request["provider"])
                    replies[attempt] = reply
                    row.update(
                        {
                            key: reply[key]
                            for key in (
                                "input_tokens",
                                "output_tokens",
                                "reasoning_tokens",
                                "generation_id",
                                "finish_reason",
                            )
                        }
                    )
                    row.update(
                        returned_model=reply["actual_model"],
                        returned_provider=reply["actual_provider"],
                    )
                    row["provider_timing_ns"] = reply.get("timing_ns")
                    if (
                        store.exists(attempt + "/normalized.json")
                        and store.get(attempt + "/normalized.json") != reply
                    ):
                        raise IntegrityError("normalized record disagrees with raw response")
                except (ValueError, KeyError, TypeError) as exc:
                    if isinstance(exc, IntegrityError):
                        raise
                    row.update(status="unusable_provider_response", normalization_error=str(exc))
            else:
                unknown += 1
            attempts.append(row)
        result_name = folder + "/result.json"
        if not store.exists(result_name):
            continue
        completed += 1
        result = store.get(result_name)
        if result["key"] != spec["key"] or result["spec_sha256"] != digest(spec):
            raise IntegrityError("committed work identity differs")
        attempt = result["attempt"]
        if (
            result["response_sha256"] is not None
            and digest(store.get(attempt + "/response.json")) != result["response_sha256"]
        ):
            raise IntegrityError("committed response hash differs")
        if result["status"] == "request_rejected":
            value, status = None, "request_rejected"
            if store.get(attempt + "/response.json")["http_status"] not in (400, 413, 422):
                raise IntegrityError("request rejection lacks a terminal client-error response")
        elif result["status"] == "transport_failure":
            value, status = None, "transport_failure"
            if len(store.names(folder + "/attempts/*/request.json")) != 3:
                raise IntegrityError(
                    "transport failure committed before physical attempts exhausted"
                )
        else:
            reply = replies[attempt]
            condition = spec["condition"]
            if (
                reply["actual_model"] not in condition["expected_models"]
                or reply["actual_provider"] != condition["expected_provider"]
            ):
                raise IntegrityError("committed returned identity differs from condition")
            try:
                value = (
                    parse_prompt(reply["text"], reply["finish_reason"])
                    if spec["role"] == "optimizer"
                    else parse_answer(
                        reply["text"],
                        set(spec["allowed_ids"]),
                        reply["finish_reason"],
                        scaffold=condition.get("output_scaffold", False),
                    )
                )
                status = "valid"
            except (ContractError, TypeError):
                value, status = None, "invalid_output"
        if result["value"] != value or result["status"] != status:
            raise IntegrityError("committed interpretation differs from strict raw parsing")
        invalid += int(status != "valid")
        if spec["role"] == "executor":
            profile = profiles[spec["key"].rsplit("/", 1)[1]]
            prompt = spec["body"]["messages"][0]["content"]
            if executor_request(spec["condition"], prompt, profile) != spec["body"] or set(
                spec["allowed_ids"]
            ) != {a.id for a in profile.advisories}:
                raise IntegrityError("Executor payload differs from frozen public profile/contract")
            row = score(profile, value if isinstance(value, list) else None, status)
            evaluation_name = "evaluations/" + digest(spec["key"]) + ".json"
            if store.exists(evaluation_name) and store.get(evaluation_name)["row"] != row:
                raise IntegrityError("stored score differs from independently reconstructed counts")
            reconstructed.append(
                {
                    "key": spec["key"],
                    "condition_id": spec["condition"]["id"],
                    "prompt_sha256": digest(prompt),
                    "row": row,
                }
            )
            groups[spec["key"].split("/")[0] + "/" + spec["condition"]["id"]].append(row)
    telemetry = []
    for name in store.names("telemetry/*/metadata.json"):
        metadata = store.get(name)
        stream = store.root / name.replace("metadata.json", "samples.jsonl")
        telemetry.append(
            {"session": name.split("/")[1], **summarize(stream, metadata["interval_seconds"])}
        )
    sessions = []
    for session_folder in (root / "sessions").glob("*"):
        paths = sorted(session_folder.glob("[0-9]*.json"))
        if paths:
            sessions.append(
                {"session": session_folder.name, **store.get(str(paths[-1].relative_to(root)))}
            )
    report = {
        "evidence_kind": "offline_raw_reply_reconstruction",
        "run": str(root.resolve()),
        "completed_work": completed,
        "physical_attempts": len(attempts),
        "unknown_attempts": unknown,
        "invalid_or_failed_commits": invalid,
        "costs": ledger(root),
        "metrics_by_stage_and_condition": {key: metrics(rows) for key, rows in groups.items()},
        "telemetry_sessions": telemetry,
        "clock_sessions": sessions,
        "running_seconds_accounted": sum(
            s["elapsed_seconds"] + s["pending_timeout_seconds"] for s in sessions
        ),
        "limitations": [
            "Checks raw-response interpretation and arithmetic; not an independent human oracle review.",
            "Repeated profiles/prompts are not independent observations. Pilot and soak are never confirmatory results.",
            "Power is whole-device sampled energy, without interpolation through missing/long intervals.",
            "Unknown responses retain reservations; reasoning tokens are part of output tokens and not added twice.",
        ],
    }
    with target.lock():
        target.put("report.json", report)
        atomic_write(
            target.root / "attempts.jsonl",
            ("".join(canonical(r) + "\n" for r in attempts)).encode(),
        )
        atomic_write(
            target.root / "evaluations.jsonl",
            ("".join(canonical(r) + "\n" for r in reconstructed)).encode(),
        )
        csv_text = io.StringIO()
        fields = sorted({k for row in attempts for k in row})
        writer = csv.DictWriter(csv_text, fields)
        writer.writeheader()
        writer.writerows(attempts)
        atomic_write(target.root / "attempts.csv", csv_text.getvalue().encode())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = reconstruct(args.run_dir.resolve(), args.output_dir.resolve())
    print(
        canonical(
            {
                k: report[k]
                for k in ("completed_work", "physical_attempts", "unknown_attempts", "costs")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
