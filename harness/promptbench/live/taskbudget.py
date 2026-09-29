"""Additional pre-study scope caps over the unchanged request journal and worker."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from ..runner import RunPaused
from ..storage import Store
from .adapters import dollars
from .execution import Execution, ledger

SCOPE = "pre-full-study-20260913"
INITIAL_GPU_SECONDS = 64.80369261599844
KEY_BASELINE = Decimal("0.25388512")
TOTAL_USD_CEILING = Decimal("20")
CALIBRATION_GPU_CEILING_SECONDS = 12 * 3600


def accounted_seconds(store: Store) -> float:
    total = 0.0
    for session in (store.root / "sessions").glob("*"):
        names = sorted(session.glob("[0-9]*.json"))
        if names:
            last = store.get(str(names[-1].relative_to(store.root)))
            total += last["elapsed_seconds"] + last["pending_timeout_seconds"]
    return total


def scope_usage(
    root: Path, *, exclude: Path | None = None, calibration_only: bool = False
) -> dict[str, Any]:
    seconds, spent, unknown = INITIAL_GPU_SECONDS, Decimal(0), Decimal(0)
    for path in root.glob("*/plan.json"):
        if exclude is not None and path.parent.resolve() == exclude.resolve():
            continue
        store = Store(path.parent)
        plan = store.get("plan.json")
        if plan.get("budget_scope") != SCOPE or (
            calibration_only and plan["kind"] != "calibration"
        ):
            continue
        seconds += accounted_seconds(store)
        seconds += sum(
            float(store.get(n)["charged_seconds"])
            for n in store.names("resource-adjustments/*.json")
        )
        account = ledger(store.root)
        spent += dollars(account["reported_cost_usd"])
        unknown += dollars(account["reserved_unknown_usd"])
    return {
        "charged_gpu_seconds": seconds,
        "reported_cost_usd": str(spent),
        "reserved_unknown_usd": str(unknown),
    }


class ScopedExecution(Execution):
    def remaining_seconds(self) -> float:
        plan = self.store.get("plan.json")
        others = scope_usage(self.study_root, exclude=self.store.root)
        used_here = self.max_seconds - super().remaining_seconds()
        return float(
            min(
                super().remaining_seconds(),
                plan["scope_gpu_ceiling_seconds"] - others["charged_gpu_seconds"] - used_here,
            )
        )

    def admission(self, provider: str, reservation: Decimal) -> float:
        timeout = super().admission(provider, reservation)
        if provider == "openrouter":
            total = scope_usage(self.study_root)
            if self.key_usage < KEY_BASELINE or (
                max(self.key_usage, KEY_BASELINE + dollars(total["reported_cost_usd"]))
                + dollars(total["reserved_unknown_usd"])
                + reservation
                > TOTAL_USD_CEILING
            ):
                raise RunPaused("pre-study USD 20 scope cap or changed key usage baseline")
            if self.store.get("plan.json")["kind"] == "calibration":
                calibration = scope_usage(self.study_root, calibration_only=True)
                if (
                    dollars(calibration["reported_cost_usd"])
                    + dollars(calibration["reserved_unknown_usd"])
                    + reservation
                    > 2
                ):
                    raise RunPaused("combined calibration USD 2 cap includes unknown reservations")
        return timeout
