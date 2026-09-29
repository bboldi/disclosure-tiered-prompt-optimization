"""Bounded, metadata-only reconciliation after a completed hosted response."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..storage import IntegrityError
from .transport import OPENROUTER, HTTPResponseError


def generation_metadata(
    get: Callable[..., dict[str, Any]],
    wait: Callable[[float], None],
    identifier: str,
    canonical_model: str,
) -> dict[str, Any]:
    if not identifier.startswith("gen-") or not all(c.isalnum() or c in "-_" for c in identifier):
        raise IntegrityError("hosted response lacks a usable generation identifier")
    # API: /api/v1/generation?id=<completion.id>, authenticated, after completion.
    # No documented availability SLA. These are bounded local retry-policy delays.
    delays = (0, 2, 4, 8, 16)
    for index, delay in enumerate(delays):
        if delay:
            wait(delay)
        try:
            data = get(
                "generation", OPENROUTER + "/api/v1/generation?id=" + identifier, authenticated=True
            )["data"]
        except HTTPResponseError as exc:
            if exc.status != 404 or index == len(delays) - 1:
                raise
            continue
        if data.get("id") != identifier or data.get("model") != canonical_model:
            raise IntegrityError("generation metadata differs from requested ID or frozen model")
        return dict(data)
    raise AssertionError("unreachable generation retry state")
