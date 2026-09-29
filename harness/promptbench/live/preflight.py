"""Read-only identity, hardware, API access and pricing evidence for Phase 2."""

from __future__ import annotations

import platform
import shutil
import subprocess  # nosec B404 -- fixed local hardware command
import time
from pathlib import Path
from typing import Any

from ..config import HOSTED_MODELS, LOCAL_MODELS
from ..runner import utc_now
from ..storage import Store, digest
from .transport import OLLAMA, OPENROUTER, Transport, response_json


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader",
    ]
    started = time.perf_counter_ns()
    try:
        process = subprocess.run(  # nosec B603 B607 -- fixed read-only hardware executable
            command, capture_output=True, text=True, timeout=5, check=False
        )
        return {
            "timestamp_utc": utc_now(),
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "duration_ns": time.perf_counter_ns() - started,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "timestamp_utc": utc_now(),
            "command": command,
            "error": str(exc),
            "returncode": None,
        }


def sources() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(package)): p.read_text() for p in sorted(package.rglob("*.py"))}


def local_identity(tag: dict[str, Any], details: dict[str, Any], version: str) -> dict[str, Any]:
    if tag.get("remote_host") or tag.get("remote_model"):
        raise ValueError("a remote Ollama alias cannot be admitted as a local Executor")
    return {
        "requested_tag": tag["name"],
        "ollama_manifest_digest": tag["digest"],
        "ollama_version": version,
        "modified_at": tag.get("modified_at"),
        "size_bytes": tag["size"],
        "details": details.get("details", tag.get("details")),
        "capabilities": details.get("capabilities", tag.get("capabilities")),
        "template_sha256": digest(details.get("template")),
        "parameters_sha256": digest(details.get("parameters")),
        "metadata_sha256": digest(details),
        "weight_revision": "Ollama content-addressed manifest; raw show metadata retained",
    }


def run_preflight(root: Path, transport: Transport) -> dict[str, Any]:
    store = Store(root)
    with store.lock():
        store.put("source.json", sources())
        store.put(
            "environment.json",
            {
                "created_utc": utc_now(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "hostname": platform.node(),
                "disk_free_bytes": shutil.disk_usage(root).free,
            },
        )
        gpu = gpu_snapshot()
        store.put("hardware.json", gpu)

        def get(
            name: str, url: str, body: dict[str, Any] | None = None, auth: bool = False
        ) -> dict[str, Any]:
            store.put(f"requests/{name}.json", {"url": url, "body": body, "authenticated": auth})
            record = transport.request(url, body, authenticated=auth)
            store.put(f"responses/{name}.json", record)
            return response_json(record)

        version = get("ollama-version", OLLAMA + "/api/version")["version"]
        tags = get("ollama-tags", OLLAMA + "/api/tags")["models"]
        get("ollama-running", OLLAMA + "/api/ps")
        local: list[dict[str, Any]] = []
        for model in LOCAL_MODELS:
            tag = next((t for t in tags if t["name"] == model), None)
            if tag is None:
                local.append({"requested_tag": model, "status": "not_installed"})
                continue
            details = get(
                "ollama-show-" + model.replace(":", "_"), OLLAMA + "/api/show", {"model": model}
            )
            local.append(local_identity(tag, details, version))
        key = get("openrouter-key", OPENROUTER + "/api/v1/key", auth=True)["data"]
        # /credits needs a management key; this study's inference key is sufficient for /key.
        catalog = get("openrouter-catalog", OPENROUTER + "/api/v1/models")["data"]
        hosted: list[dict[str, Any]] = []
        for model in HOSTED_MODELS:
            entry = next((m for m in catalog if m["id"] == model), None)
            if entry is None:
                hosted.append({"requested_model": model, "status": "not_listed"})
                continue
            endpoints = get(
                "openrouter-endpoints-" + model.replace("/", "_"),
                OPENROUTER + f"/api/v1/models/{model}/endpoints",
            )["data"]
            hosted.append(
                {
                    "requested_model": model,
                    "catalog": entry,
                    "endpoints": endpoints,
                    "hosted_weight_revision": None,
                    "revision_note": "Catalog/canonical IDs and endpoint revisions are retained; no independently verifiable weight digest is exposed.",
                }
            )
        result = {
            "evidence_kind": "read_only_live_preflight",
            "created_utc": utc_now(),
            "local_models": local,
            "hosted_models": hosted,
            "key_usage_baseline": key,
            "key_label_for_manuscript": "prompt_optimization_publication_key",
            "gpu_accessible": gpu["returncode"] == 0,
            "inference_requests": 0,
            "inference_cost_usd": "0",
            "account_balance": None,
            "account_balance_reason": "No management key requested; study budget is capped separately at USD 60.",
        }
        store.put("model_registry.json", result)
        return result
