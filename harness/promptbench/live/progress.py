"""Terminal progress backed by committed work, with a heartbeat during blocking calls."""

from __future__ import annotations

import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TextIO

from ..runner import utc_now
from ..storage import Store


class Progress:
    def __init__(
        self,
        store: Store,
        sizes: dict[str, int],
        *,
        stream: TextIO | None = None,
        interval: float = 10,
    ):
        self.store, self.sizes = store, sizes
        self.stream = stream or sys.stdout
        self.interval = interval
        self.session = uuid.uuid4().hex
        self._lock = threading.Lock()
        self.clock: Callable[[], dict[str, float]] | None = None
        self.estimated = False

    def emit(
        self, task: str, item: str, state: str, *, completed: int | None = None, elapsed: float = 0
    ) -> None:
        with self._lock:
            keys = [self.store.get(n)["key"] for n in self.store.names("work/*/result.json")]
            finalized = bool(self.store.names("reports/*.json"))
            total = sum(size for name, size in self.sizes.items() if name != "preflight")
            done = len(keys) + int(finalized)
            size = max(1, self.sizes[task])
            if completed is None:
                completed = (
                    int(finalized)
                    if task == "report"
                    else sum(k.startswith(task + "/") for k in keys)
                )
            percent = 100 * done / total
            task_percent = 100 * completed / size
            clock_values = self.clock() if self.clock is not None else None
            record = {
                "timestamp_utc": utc_now(),
                "total_done": done,
                "total_steps": total,
                "total_percent": percent,
                "task": task,
                "task_done": completed,
                "task_steps": size,
                "task_percent": task_percent,
                "item": item,
                "state": state,
                "request_elapsed_seconds": elapsed,
                "progress_basis": "committed evaluations/candidate plus final report; not generated tokens or elapsed-time fraction",
                "schedule_is_estimate": self.estimated,
                "run_clock": clock_values,
            }
            self.store.put(
                f"progress/{self.session}/{time.time_ns()}-{uuid.uuid4().hex}.json", record
            )
            clock_text = ""
            if clock_values is not None:
                clock_text = f" | budget left {clock_values['remaining_seconds'] / 3600:.2f}h"
            estimate = "~" if self.estimated else ""
            print(
                f"[TOTAL{estimate} {percent:5.1f}% {done}/{total} | {task} {task_percent:5.1f}% {completed}/{size}{clock_text}] {item}: {state} ({elapsed:.1f}s)",
                file=self.stream,
                flush=True,
            )

    @contextmanager
    def waiting(self, task: str, item: str, *, completed: int | None = None) -> Iterator[None]:
        started = time.perf_counter()
        finished = threading.Event()
        self.emit(task, item, "waiting for response", completed=completed)

        def heartbeat() -> None:
            while not finished.wait(self.interval):
                self.emit(
                    task,
                    item,
                    "waiting for response",
                    completed=completed,
                    elapsed=time.perf_counter() - started,
                )

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            yield
        finally:
            finished.set()
            worker.join()
