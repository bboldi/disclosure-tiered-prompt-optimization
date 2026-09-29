"""Subprocess used by tests to hard-kill the controller at a durable boundary."""

import os
import signal
import sys
from pathlib import Path

from promptbench.experiment import Config, Experiment, read_fixture

root, fixture_path, boundary = sys.argv[1:]
seen = 0


def crash(point: str, key: str) -> None:
    global seen
    if point == boundary:
        seen += 1
        target = 5 if boundary == "committed" else 1
        if seen == target:
            os.kill(os.getpid(), signal.SIGKILL)


Experiment(Path(root), Config(), read_fixture(Path(fixture_path)), hook=crash).run()
raise SystemExit("requested crash point was not exercised")
