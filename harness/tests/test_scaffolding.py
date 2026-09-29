import copy
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from promptbench.benchmark.model import score
from promptbench.domain import ContractError, parse_answer
from promptbench.live.conditions import NAIVE, executor_request
from promptbench.live.execution import Execution
from promptbench.live.reconstruct import reconstruct
from promptbench.live.scaffolding import (
    CONTROL_WATCHDOG_SECONDS,
    EXECUTORS,
    MAX_ATTEMPTS,
    NEUTRAL_APPEND,
    SCAFFOLD_APPEND,
    VARIANTS,
    control_design,
    observations,
    paired_summary,
    run,
)
from promptbench.storage import IntegrityError, Store, digest
from tests.test_execution import LOCAL, response
from tests.test_feedback_relations import profile


class ScaffoldingTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)

    def test_scaffold_contract_complete_unique_boolean_consistent_and_opt_in(self):
        answer = {
            "applicable_cves": ["CVE-1"],
            "advisory_decisions": [
                {"id": "CVE-1", "applicable": True},
                {"id": "CVE-2", "applicable": False},
            ],
        }
        allowed = {"CVE-1", "CVE-2"}
        self.assertEqual(parse_answer(json.dumps(answer), allowed, scaffold=True), ["CVE-1"])
        with self.assertRaises(ContractError):
            parse_answer(json.dumps(answer), allowed)
        with self.assertRaises(ContractError):
            parse_answer(json.dumps(answer), allowed, "length", scaffold=True)
        for decisions in (
            answer["advisory_decisions"][:1],
            answer["advisory_decisions"] * 2,
            [{"id": "CVE-1", "applicable": True}, {"id": "foreign", "applicable": False}],
            [{"id": "CVE-1", "applicable": 1}, {"id": "CVE-2", "applicable": False}],
            [
                {"id": "CVE-1", "applicable": True, "reason": "extra"},
                {"id": "CVE-2", "applicable": False},
            ],
            [{"id": "CVE-1", "applicable": False}, {"id": "CVE-2", "applicable": False}],
        ):
            with self.subTest(decisions=decisions), self.assertRaises(ContractError):
                parse_answer(
                    json.dumps({**answer, "advisory_decisions": decisions}), allowed, scaffold=True
                )
        negative = {
            "applicable_cves": [],
            "advisory_decisions": [{"id": i, "applicable": False} for i in sorted(allowed)],
        }
        self.assertEqual(parse_answer(json.dumps(negative), allowed, scaffold=True), [])

    def test_design_is_fixed_paired_and_each_variant_changes_only_declared_factor(self):
        profiles = [replace(profile(i), partition="test") for i in range(192)]
        conditions = [
            {
                **LOCAL,
                "id": i,
                "model": "fixture",
                "think": False,
                "num_ctx": 16384,
                "num_predict": 512,
            }
            for i in EXECUTORS
        ]
        verdict = {"passed": True, "promoted": list(EXECUTORS)}
        design = control_design(profiles, conditions, verdict)
        Store(self.root).put("design.json", design)
        self.assertEqual(design["logical_calls"], 1920)
        self.assertEqual(design["optimizer_calls"], 0)
        self.assertEqual(len(design["conditions"]), 10)
        self.assertEqual(len(SCAFFOLD_APPEND.encode("ascii")), len(NEUTRAL_APPEND.encode("ascii")))
        self.assertEqual(design["schedule"][0]["variant"], "naive")
        self.assertEqual(design["schedule"][5]["variant"], "output_scaffold")
        self.assertEqual(
            [s["profile_id"] for s in design["schedule"][:960]],
            [s["profile_id"] for s in design["schedule"][960:]],
        )
        for identifier in EXECUTORS:
            for variant in VARIANTS:
                prefix = f"scaffolding/{identifier}/{variant}"
                condition = design["conditions"][prefix]
                prompt = design["prompts"][variant]
                body = executor_request(condition, prompt, profiles[0])
                self.assertEqual(
                    body["options"],
                    {"temperature": 0, "seed": 11, "num_ctx": 16384, "num_predict": 4096},
                )
                payload = json.loads(body["messages"][1]["content"])
                self.assertEqual("components" in payload, variant == "structured_inventory")
                self.assertNotIn("expected", payload)
                self.assertEqual(
                    "enum" in body["format"]["properties"]["applicable_cves"]["items"],
                    variant == "schema_enum",
                )
                self.assertEqual(
                    "advisory_decisions" in body["format"]["properties"],
                    variant == "output_scaffold",
                )
                if variant == "output_scaffold":
                    scaffold = body["format"]["properties"]["advisory_decisions"]
                    self.assertEqual(scaffold["minItems"], len(profiles[0].advisories))
                    self.assertNotIn("enum", scaffold["items"]["properties"]["id"])
                if variant not in ("output_scaffold", "neutral"):
                    self.assertEqual(prompt, NAIVE)
        for bad_verdict in (
            {"passed": False, "promoted": list(EXECUTORS)},
            {"passed": True, "promoted": [*EXECUTORS, "gemma4:31b/off"]},
        ):
            with self.assertRaises(IntegrityError):
                control_design(profiles, conditions, bad_verdict)
        for bad_profiles in (
            profiles[:-1],
            [profiles[0]] * 192,
            [replace(profiles[0], partition="validation"), *profiles[1:]],
        ):
            with self.assertRaises(IntegrityError):
                control_design(bad_profiles, conditions, verdict)

    def test_execution_retains_scaffold_and_resume_does_not_repeat_inference(self):
        store, calls = Store(self.root), []
        p = profile()
        text = json.dumps(
            {
                "applicable_cves": [],
                "advisory_decisions": [{"id": p.advisories[0].id, "applicable": False}],
            }
        )

        def worker(store, attempt, env_file, **kwargs):
            calls.append(attempt)
            store.put(attempt + "/response.json", response(text))

        engine = Execution(
            store,
            self.root / "unused.env",
            max_seconds=600,
            max_cost_usd="1",
            max_attempts=10,
            study_root=self.root,
            worker=worker,
        )
        prefix = "scaffolding/fixture/output_scaffold"
        kwargs = {"role": "executor", "allowed_ids": {p.advisories[0].id}}
        condition = {**LOCAL, "output_scaffold": True}
        accepted = engine.call(prefix + "/" + p.id, condition, {}, **kwargs)
        self.assertEqual(accepted["status"], "valid")
        self.assertEqual(accepted["value"], [])
        self.assertEqual(engine.call(prefix + "/" + p.id, condition, {}, **kwargs), accepted)
        self.assertEqual(len(calls), 1)
        rejected = engine.call("scaffolding/fixture/naive/" + p.id, LOCAL, {}, **kwargs)
        self.assertEqual(rejected["status"], "invalid_output")
        self.assertEqual(len(calls), 2)
        obs = observations(store, prefix, [p])
        self.assertEqual(obs[0]["output_tokens"], 10)
        self.assertIsNone(obs[0]["input_tokens"])
        self.assertEqual(obs[0]["latency_seconds"], 0.001)
        self.assertEqual(obs[0]["physical_attempts"], 1)

    def test_paired_report_keeps_failure_outcomes_and_missing_usage(self):
        p = profile()
        row = score(p, None, "invalid_output")
        observed = [
            {"profile_id": p.id, "output_tokens": 15, "input_tokens": None, "latency_seconds": 3}
        ]
        naive = [
            {"profile_id": p.id, "output_tokens": 5, "input_tokens": 100, "latency_seconds": 1}
        ]
        report = paired_summary([row], observed, naive)
        self.assertEqual(report["metrics"]["coverage"], 0)
        self.assertEqual(report["paired_difference_from_naive"]["output_tokens"]["total"], 10)
        self.assertEqual(report["paired_difference_from_naive"]["latency_seconds"]["mean_known"], 2)
        self.assertEqual(report["paired_difference_from_naive"]["input_tokens"]["missing"], 1)
        self.assertIsNone(report["paired_difference_from_naive"]["input_tokens"]["total"])
        with self.assertRaises(IntegrityError):
            paired_summary([row], observed, [{**naive[0], "profile_id": "different"}])
        Store(self.root).put("report.json", report)

    def test_completed_run_checks_integrity_and_performs_no_inference(self):
        store = Store(self.root)
        plan = {"kind": "fixed_scaffolding_controls"}
        store.put("manifest.json", {"plan_sha256": digest(plan), "input_sha256": {}})
        store.put("reports/complete.json", {"complete": True})
        with (
            patch("promptbench.live.scaffolding.verify_runtime", return_value=plan),
            patch("promptbench.live.scaffolding.Execution") as engine,
        ):
            self.assertEqual(run(self.root, self.root / "unused.env"), 0)
            engine.assert_not_called()
        bad = copy.deepcopy(plan)
        bad["kind"] = "phase3"
        with (
            patch("promptbench.live.scaffolding.verify_runtime", return_value=bad),
            self.assertRaises(IntegrityError),
        ):
            run(self.root, self.root / "unused.env")

    def test_fresh_run_limits_are_accepted_by_the_real_journal(self):
        store = Store(self.root)
        plan = {
            "kind": "fixed_scaffolding_controls",
            "max_seconds": CONTROL_WATCHDOG_SECONDS,
            "max_cost_usd": "0.01",
            "max_attempts": MAX_ATTEMPTS,
            "initial_step_estimate": {"scaffolding": 1920, "report": 1},
        }
        store.put("plan.json", plan)
        store.put("manifest.json", {"plan_sha256": digest(plan), "input_sha256": {}})
        with (
            patch("promptbench.live.scaffolding.verify_runtime", return_value=plan),
            patch(
                "promptbench.live.scaffolding.execute", return_value={"logical_calls": 1920}
            ) as execute,
            patch("promptbench.live.scaffolding.Telemetry") as telemetry,
        ):
            telemetry.return_value.stop.return_value = {}
            telemetry.return_value.thread.is_alive.return_value = False
            self.assertEqual(run(self.root, self.root / "unused.env"), 0)
            self.assertIsInstance(execute.call_args.args[0], Execution)
            self.assertEqual(store.get("reports/complete.json")["logical_calls"], 1920)
        self.assertEqual(CONTROL_WATCHDOG_SECONDS, 259200)
        self.assertEqual(MAX_ATTEMPTS, 5760)

    def test_raw_reconstruction_honors_scaffold_and_retains_inconsistent_failure(self):
        store = Store(self.root / "campaign")
        p = profile()
        store.put("inputs/profiles.json", [p.record()])
        store.put("inputs/advisories.json", {a.id: a.record() for a in p.advisories})
        condition = {
            **LOCAL,
            "id": "fixture/off",
            "model": "fixture",
            "think": False,
            "num_ctx": 16384,
            "num_predict": 4096,
            "output_scaffold": True,
        }
        calls = []

        def worker(store, attempt, env_file, **kwargs):
            answer = {
                "applicable_cves": [] if not calls else [p.advisories[0].id],
                "advisory_decisions": [{"id": p.advisories[0].id, "applicable": False}],
            }
            calls.append(attempt)
            store.put(attempt + "/response.json", response(json.dumps(answer)))

        engine = Execution(
            store,
            self.root / "unused.env",
            max_seconds=600,
            max_cost_usd="1",
            max_attempts=10,
            study_root=self.root,
            worker=worker,
        )
        for label, status in (("consistent", "valid"), ("inconsistent", "invalid_output")):
            key = f"scaffolding/fixture/{label}/{p.id}"
            result = engine.call(
                key,
                condition,
                executor_request(condition, NAIVE + SCAFFOLD_APPEND, p),
                role="executor",
                allowed_ids={p.advisories[0].id},
            )
            self.assertEqual(result["status"], status)
            store.put(
                "evaluations/" + digest(key) + ".json", {"row": score(p, result["value"], status)}
            )
        audited = reconstruct(store.root, self.root / "reconstruction")
        self.assertEqual(audited["completed_work"], 2)
        self.assertEqual(audited["physical_attempts"], 2)
        self.assertEqual(audited["invalid_or_failed_commits"], 1)
        self.assertEqual(audited["unknown_attempts"], 0)
        self.assertEqual(len(calls), 2)
