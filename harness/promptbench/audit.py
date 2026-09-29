"""Reconstruct results from saved requests and raw responses, without model calls."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import Any

from .domain import ContractError, load_profiles, metrics, parse_answer, parse_prompt, score
from .runner import utc_now
from .storage import IntegrityError, Store, atomic_write, canonical, digest


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise IntegrityError(reason)


def audit(store: Store, *, export: bool = True) -> dict[str, Any]:
    with store.lock():
        return _audit(store, export=export)


def _audit(store: Store, *, export: bool) -> dict[str, Any]:
    for name in store.names("**/*.json"):
        if not name.startswith("exports/"):
            store.get(name)  # verify every authoritative record, including attempts and events
    manifest = store.get("manifest.json")
    dataset = store.get("inputs/dataset.json")
    _require(digest(dataset) == manifest["dataset_sha256"], "dataset snapshot changed")
    _require(
        digest(store.get("inputs/source.json")) == manifest["source_sha256"],
        "source snapshot changed",
    )
    profiles = {p.id: p for p in load_profiles(dataset)}
    complete = store.get("completion.json")
    _require(complete["manifest_sha256"] == digest(manifest), "completion manifest mismatch")
    work_ids = complete["required_work"]
    _require(len(work_ids) == len(set(work_ids)), "duplicate work in completion")
    result_names = store.names("work/*/result.json")
    _require(
        set(result_names) == {f"work/{w}/result.json" for w in work_ids},
        "missing or unexpected committed work",
    )
    _require(
        set(store.names("work/*/spec.json")) == {f"work/{w}/spec.json" for w in work_ids},
        "uncommitted work remains",
    )
    work: dict[str, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    for work_id in work_ids:
        spec = store.get(f"work/{work_id}/spec.json")
        result = store.get(f"work/{work_id}/result.json")
        request = spec["request"]
        _require(digest({"key": spec["key"], "request": request}) == work_id, "work ID changed")
        _require(result["key"] == spec["key"], "result key changed")
        _require(
            result["attempt"].startswith(f"work/{work_id}/attempts/"),
            "result references wrong attempt",
        )
        response = store.get(result["attempt"] + "/response.json")
        _require(digest(response) == result["response_sha256"], "result response digest mismatch")
        _require(
            spec["request_sha256"]
            == result["request_sha256"]
            == response["request_sha256"]
            == digest(request),
            "request digest mismatch",
        )
        if response["status"] == "response":
            raw = response["reply"]
            try:
                value: Any = (
                    parse_prompt(raw["text"], raw["finish_reason"])
                    if request["role"] == "optimizer"
                    else parse_answer(
                        raw["text"],
                        {a["id"] for a in request["input"]["advisories"]},
                        raw["finish_reason"],
                    )
                )
                expected_status = "valid"
            except ContractError:
                value, expected_status = None, "invalid_output"
                _require(request["role"] != "optimizer", "invalid optimizer marked committed")
        else:
            value, expected_status = None, "transport_failure"
            _require(
                request["role"] == "executor" and response["retryable"],
                "invalid terminal transport disposition",
            )
        _require(
            result["value"] == value and result["status"] == expected_status,
            "parsed result differs from raw evidence",
        )
        work[work_id] = {"spec": spec, "result": result}
        _require(spec["key"] not in by_key, "duplicate logical work")
        by_key[spec["key"]] = work[work_id]
    evaluations = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    evaluated_work: set[str] = set()
    for name in store.names("evaluations/*.json"):
        saved = store.get(name)
        item = work[saved["work_id"]]
        request, result = item["spec"]["request"], item["result"]
        profile = profiles[saved["score"]["profile_id"]]
        _require(request["role"] == "executor", "evaluation points to optimizer")
        _require(request["input"] == profile.executor_input(), "executor input changed")
        _require(item["spec"]["key"] == saved["key"], "evaluation work key mismatch")
        recomputed = score(profile, result["value"], result["status"])
        _require(saved["score"] == recomputed, "score differs from raw prediction")
        _require(saved["work_id"] not in evaluated_work, "duplicate scored work")
        evaluated_work.add(saved["work_id"])
        groups.setdefault((saved["stage"], saved["candidate_id"]), []).append(recomputed)
        evaluations.append(saved)
    _require(
        evaluated_work
        == {w for w, item in work.items() if item["spec"]["request"]["role"] == "executor"},
        "executor work missing scoring record",
    )
    for name in store.names("metrics/*.json"):
        saved = store.get(name)
        _require(
            metrics(groups[(saved["stage"], saved["candidate_id"])]) == saved["metrics"],
            "cached metrics differ from reconstructed scores",
        )
    expected_keys: set[str] = set()

    def expect_evaluation(stage: str, candidate_id: str, profile_ids: list[str]) -> None:
        expected_keys.update(f"evaluate/{stage}/{candidate_id}/{p}" for p in profile_ids)
        rows = groups[(stage, candidate_id)]
        _require(
            sorted(r["profile_id"] for r in rows) == sorted(profile_ids),
            "evaluation cohort is incomplete",
        )
        prompt = store.get(f"prompts/{candidate_id}.json")["prompt"]
        for p in profile_ids:
            _require(
                by_key[f"evaluate/{stage}/{candidate_id}/{p}"]["spec"]["request"]["prompt"]
                == prompt,
                "evaluated prompt differs from candidate artifact",
            )

    incumbent = "baseline"
    expect_evaluation("baseline", incumbent, store.get("schedules/000.json")["profile_ids"])
    config = manifest["config"]
    for iteration in range(1, config["iterations"] + 1):
        batch = store.get(f"schedules/{iteration:03d}.json")["profile_ids"]
        stage = f"iteration-{iteration}"
        ids = [incumbent]
        for slot in range(config["candidates"]):
            expected_keys.add(f"propose/{iteration}/{slot}")
            candidate_id = f"i{iteration}-c{slot}"
            candidate = store.get(f"prompts/{candidate_id}.json")
            _require(
                candidate["prompt"] == by_key[f"propose/{iteration}/{slot}"]["result"]["value"],
                "candidate differs from optimizer raw response",
            )
            if candidate["duplicate_of"] is None:
                ids.append(candidate_id)
        for candidate_id in ids:
            expect_evaluation(stage, candidate_id, batch)
        winner = max(
            ids,
            key=lambda candidate_id: metrics(groups[(stage, candidate_id)])[
                "failure_aware_lower_bound"
            ]["micro_f1"],
        )
        _require(
            store.get(f"decisions/iteration-{iteration:03d}.json")["selected"] == winner,
            "iteration selected wrong contender",
        )
        incumbent = winner
    selection = store.get("decisions/selection.json")
    shortlist = ["baseline"] if incumbent == "baseline" else ["baseline", incumbent]
    _require(selection["shortlist"] == shortlist, "shortlist changed")
    validation_ids = [p.id for p in profiles.values() if p.partition == "validation"]
    for candidate_id in shortlist:
        expect_evaluation("validation", candidate_id, validation_ids)
    selected_id = max(
        shortlist,
        key=lambda candidate_id: metrics(groups[("validation", candidate_id)])[
            "failure_aware_lower_bound"
        ]["micro_f1"],
    )
    _require(selection["selected"]["id"] == selected_id, "selection differs from validation")
    _require(complete["selected"] == selection["selected"], "test-selected prompt changed")
    expect_evaluation(
        "test", selected_id, [p.id for p in profiles.values() if p.partition == "test"]
    )
    _require(set(by_key) == expected_keys, "completion differs from protocol workset")
    test_metrics = metrics(groups[("test", selected_id)])
    _require(complete["test_metrics"] == test_metrics, "test metrics mismatch")
    attempts: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    synthetic_cost = Decimal(0)
    durations = 0
    for name in store.names("work/*/attempts/*/request.json"):
        directory = name.rsplit("/", 1)[0]
        intent = store.get(name)
        response = (
            store.get(directory + "/response.json")
            if store.exists(directory + "/response.json")
            else None
        )
        if response is None:
            _require(
                store.exists(directory + "/outcome_unknown.json"),
                "unfinished attempt lacks unknown-outcome disposition",
            )
            failures["outcome_unknown"] += 1
        elif response["status"] == "response":
            reply = response["reply"]
            for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
                totals[key] += reply[key]
            expected_cost = Decimal(reply["input_tokens"]) * Decimal("0.000001") + Decimal(
                reply["output_tokens"]
            ) * Decimal("0.000002")
            _require(expected_cost == Decimal(reply["synthetic_cost_usd"]), "usage cost mismatch")
            synthetic_cost += expected_cost
            durations += response["duration_ns"]
        else:
            failures["transport_error"] += 1
            durations += response["duration_ns"]
        if store.exists(directory + "/disposition.json"):
            disposition = store.get(directory + "/disposition.json")
            if disposition["status"] != "valid":
                failures[disposition["status"]] += 1
        attempts.append({"path": directory, "request": intent, "response": response})
    events = [store.get(n) for n in store.names("events/*.json")]
    times = [datetime.fromisoformat(e["timestamp_utc"]) for e in events]
    summary = {
        "evidence_kind": "engineering_fixture_only",
        "status": "completed_and_audited",
        "committed_work": len(work),
        "provider_attempts": len(attempts),
        "actual_api_cost_usd": "0",
        "synthetic_known_cost_usd": str(synthetic_cost),
        "unknown_attempts": failures["outcome_unknown"],
        "usage_origin": "synthetic_fixture",
        "tokens": dict(totals),
        "failures": dict(failures),
        "provider_duration_seconds": durations / 1_000_000_000,
        "recorded_wall_span_seconds": (max(times) - min(times)).total_seconds() if times else 0.0,
        "controller_starts": sum(e["kind"] == "controller_started" for e in events),
        "selected": selection["selected"],
        "test_metrics": test_metrics,
        "checks": [
            "record checksums",
            "source and dataset snapshots",
            "complete workset",
            "raw response parsing",
            "independent fixture labels",
            "score reconstruction",
            "paired candidate cohorts",
            "validation-only selection",
            "usage reconstruction",
        ],
    }
    if export:
        store.put("exports/summary.json", summary, immutable=False)
        for filename, rows in (
            ("attempts", attempts),
            ("evaluations", evaluations),
            ("events", events),
        ):
            atomic_write(
                store.root / f"exports/{filename}.jsonl",
                "".join(canonical(row) + "\n" for row in rows).encode(),
                immutable=False,
            )
        report = (
            "# Phase 1 dry-run conclusions\n\n"
            f"Audited at {utc_now()}. This is deterministic engineering-fixture evidence; "
            "it does not measure LLM quality, privacy effectiveness, or real model pricing.\n\n"
            f"- {len(work)} committed work items; {len(attempts)} provider attempts.\n"
            f"- {summary['controller_starts']} controller starts.\n"
            f"- All {len(summary['checks'])} audit categories passed.\n"
            "- Actual API expenditure: USD 0. No live provider was called.\n"
            f"- Synthetic known usage cost: USD {synthetic_cost}; "
            f"{failures['outcome_unknown']} attempts have an unknown simulated outcome.\n"
            f"- Fixture selected prompt: {selected_id}.\n"
            "- Full requests, raw responses, prompts, feedback and checkpoint evidence "
            "remain in this run directory. JSONL exports are derived views.\n\n"
            "A passing dry run establishes only these exercised harness properties. "
            "Real provider integration, NVD ingestion, GPU serving, live failure recovery "
            "and the small-scale pilot remain subsequent milestones.\n"
        )
        atomic_write(store.root / "CONCLUSIONS.md", report.encode(), immutable=False)
    return summary
