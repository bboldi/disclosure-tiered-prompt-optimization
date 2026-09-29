"""Restricted HTTP transport preserving responses without persisting credentials."""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config import OLLAMA, OPENROUTER  # re-exported for existing imports
from ..runner import utc_now
from ..storage import canonical

__all__ = ["OLLAMA", "OPENROUTER", "Transport", "read_key", "response_json"]


def read_key(path: Path) -> str:
    """Read only the named assignment; never execute shell/dotenv interpolation."""
    found = []
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*?)\s*", line)
        if match:
            value = match[1]
            if value[:1] in ("'", '"'):
                quote = value[0]
                end = value.find(quote, 1)
                if end < 1 or value[end + 1 :].strip().split(" ", 1)[0] not in ("", "#"):
                    raise ValueError("unsupported OPENROUTER_API_KEY assignment")
                value = value[1:end]
            else:
                value = value.split(" #", 1)[0].strip()
            if not value or any(c.isspace() for c in value) or "$" in value or "`" in value:
                raise ValueError("invalid OPENROUTER_API_KEY assignment")
            found.append(value)
    if len(found) != 1:
        raise ValueError("expected exactly one OPENROUTER_API_KEY assignment")
    return found[0]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class Transport:
    def __init__(self, key: str | None = None):
        self._key = key
        self._opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))

    def request(
        self,
        url: str,
        body: dict[str, Any] | None = None,
        *,
        authenticated: bool = False,
        timeout: int = 30,
    ) -> dict[str, Any]:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in (OLLAMA, OPENROUTER) or parts.username or parts.fragment:
            raise ValueError("HTTP origin is outside the admitted provider endpoints")
        if authenticated and (origin != OPENROUTER or not self._key):
            raise ValueError("credentials are only available to the OpenRouter origin")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._key}"
        request = urllib.request.Request(
            url, data=None if body is None else canonical(body).encode(), headers=headers
        )
        started = utc_now()
        timer = time.perf_counter_ns()
        raw, status, error = b"", None, None
        response_headers: dict[str, str] = {}
        headers_ns = None
        try:
            try:
                response = self._opener.open(request, timeout=timeout)  # nosec B310 -- origins checked
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                headers_ns = time.perf_counter_ns() - timer
                status = response.status
                response_headers = {
                    name.lower(): value
                    for name, value in response.headers.items()
                    if name.lower() in {"content-type", "retry-after", "x-request-id", "date"}
                }
                raw = response.read(8_388_609)
                if len(raw) > 8_388_608:
                    error = "response exceeded the 8 MiB admission limit; retained prefix"
        except (OSError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        redacted = bool(self._key and self._key.encode() in raw)
        if self._key:
            raw = raw.replace(self._key.encode(), b"[REDACTED_CREDENTIAL]")
            error = error.replace(self._key, "[REDACTED_CREDENTIAL]") if error else None
        return {
            "url": url,
            "http_status": status,
            "headers": response_headers,
            "body_text": raw.decode("utf-8", errors="replace"),
            "body_base64": base64.b64encode(raw).decode(),
            "credential_redaction_applied": redacted,
            "error": error,
            "started_utc": started,
            "finished_utc": utc_now(),
            "duration_ns": time.perf_counter_ns() - timer,
            "headers_received_ns": headers_ns,
        }


class HTTPResponseError(ValueError):
    def __init__(self, status: int | None, error: str | None):
        self.status = status
        super().__init__(f"HTTP request failed: status={status}, {error}")


def response_json(record: dict[str, Any]) -> dict[str, Any]:
    if record["error"] or record["http_status"] != 200:
        raise HTTPResponseError(record["http_status"], record["error"])
    value = json.loads(record["body_text"])
    if not isinstance(value, dict):
        raise ValueError("provider response must be a JSON object")
    return value
