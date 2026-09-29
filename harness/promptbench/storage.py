"""Atomic immutable records. All durable evidence lives in the requested run folder."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class IntegrityError(ValueError):
    """A saved record is corrupt or conflicts with the requested run."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_write(path: Path, content: bytes, *, immutable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != content:
            raise IntegrityError(f"immutable artifact conflicts: {path}")
        return
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(temp)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise IntegrityError(f"concurrent artifact conflict: {path}") from None
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[Any]:
    """Read physical JSONL records without treating Unicode text separators as rows."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def put(self, name: str, payload: Any, *, immutable: bool = True) -> str:
        checksum = digest(payload)
        record = {"sha256": checksum, "payload": payload}
        atomic_write(self.root / name, (canonical(record) + "\n").encode(), immutable=immutable)
        return checksum

    def get(self, name: str) -> Any:
        path = self.root / name
        try:
            record = json.loads(path.read_text())
            payload = record["payload"]
            if set(record) != {"sha256", "payload"} or digest(payload) != record["sha256"]:
                raise IntegrityError(f"checksum mismatch: {name}")
            return payload
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IntegrityError(f"cannot validate record: {name}") from exc

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()

    def names(self, pattern: str) -> list[str]:
        return sorted(str(path.relative_to(self.root)) for path in self.root.glob(pattern))

    @contextmanager
    def lock(self) -> Iterator[None]:
        # This is a disposable local process lock, not experimental data.
        lock_root = Path(tempfile.gettempdir()) / f"promptbench-locks-{os.getuid()}"
        lock_root.mkdir(mode=0o700, exist_ok=True)
        path = lock_root / (hashlib.sha256(str(self.root).encode()).hexdigest() + ".lock")
        with path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise IntegrityError("another controller owns this run") from None
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
