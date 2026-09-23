#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shlex
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


RunnerKind = Literal["python-distribution", "external-process"]


@dataclass(frozen=True, slots=True)
class PeerSpec:
    peer_id: str
    source_revision: str
    runner_kind: RunnerKind
    expected_identity: str
    package: str | None
    command_environment: str | None


@dataclass(frozen=True, slots=True)
class PeerInvocation:
    response: Mapping[str, Any] | None
    reason: str | None
    detail: str | None
    observed_identity: str | None


def stable_fingerprint(value: Any, /) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_peer_specs(path: Path, /) -> dict[str, PeerSpec]:
    manifest = json.loads(path.read_text())
    peers: dict[str, PeerSpec] = {}
    for raw in manifest["peers"]:
        runner = raw["runner"]
        peer_id = str(raw["id"])
        if peer_id in peers:
            raise ValueError(f"Duplicate peer id {peer_id!r}.")
        source_revision = str(raw["revision"])
        if len(source_revision) != 40:
            raise ValueError(f"Peer {peer_id!r} must freeze one full source revision.")
        peers[peer_id] = PeerSpec(
            peer_id,
            source_revision,
            runner["kind"],
            str(runner["expected_identity"]),
            runner.get("package"),
            runner.get("command_environment"),
        )
    return peers


def python_runtime_identity(spec: PeerSpec, /) -> str | None:
    if spec.runner_kind != "python-distribution" or spec.package is None:
        raise TypeError("python_runtime_identity requires a Python distribution peer.")
    if importlib.util.find_spec(spec.package) is None:
        return None
    try:
        version = importlib.metadata.version(spec.package)
    except importlib.metadata.PackageNotFoundError:
        return None
    return f"{spec.package}=={version}"


def _python_source_revision(spec: PeerSpec, /) -> str | None:
    if spec.package is None:
        return None
    try:
        distribution = importlib.metadata.distribution(spec.package)
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    revision = payload.get("vcs_info", {}).get("commit_id")
    return revision if isinstance(revision, str) else None


def make_runner_request(
    spec: PeerSpec,
    family: str,
    case_id: str,
    implementation: str,
    initial_fingerprint: str,
    payload: Mapping[str, Any],
    /,
) -> dict[str, Any]:
    request_id = stable_fingerprint(
        {
            "peer": spec.peer_id,
            "family": family,
            "case": case_id,
            "implementation": implementation,
            "initial_fingerprint": initial_fingerprint,
            "payload": payload,
        }
    )
    return {
        "request_id": request_id,
        "family": family,
        "case_id": case_id,
        "implementation": implementation,
        "initial_fingerprint": initial_fingerprint,
        "expected_identity": spec.expected_identity,
        "source_revision": spec.source_revision,
        "payload": dict(payload),
    }


def validate_peer_response(
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    /,
) -> None:
    if response["request_id"] != request["request_id"]:
        raise ValueError("Runner response request_id mismatch.")
    if response["runner_id"] != request["implementation"]:
        raise ValueError("Runner response implementation mismatch.")
    if response["initial_fingerprint"] != request["initial_fingerprint"]:
        raise ValueError("Runner response initial fingerprint mismatch.")
    if response["observed_identity"] != request["expected_identity"]:
        raise ValueError("Runner runtime identity mismatch.")
    if response["source_revision"] != request["source_revision"]:
        raise ValueError("Runner source revision mismatch.")


def run_python_peer(
    request: Mapping[str, Any],
    spec: PeerSpec,
    callback: Callable[[], Mapping[str, Any]],
    /,
) -> PeerInvocation:
    if spec.runner_kind != "python-distribution":
        raise TypeError("run_python_peer requires a Python distribution peer.")
    observed_identity = python_runtime_identity(spec)
    if observed_identity is None:
        return PeerInvocation(
            None,
            "dependency-missing",
            f"Python distribution {spec.package!r} is not installed.",
            None,
        )
    if observed_identity != spec.expected_identity:
        return PeerInvocation(
            None,
            "revision-mismatch",
            (
                f"Expected runtime identity {spec.expected_identity!r}; "
                f"observed {observed_identity!r}."
            ),
            observed_identity,
        )
    observed_revision = _python_source_revision(spec)
    if observed_revision != spec.source_revision:
        return PeerInvocation(
            None,
            "revision-unverified",
            (
                f"Expected source revision {spec.source_revision!r}; installed "
                f"distribution records {observed_revision!r}."
            ),
            observed_identity,
        )
    try:
        record = callback()
        _validate_execution_record(record)
        response = {
            "request_id": request["request_id"],
            "runner_id": request["implementation"],
            "initial_fingerprint": request["initial_fingerprint"],
            "observed_identity": observed_identity,
            "source_revision": observed_revision,
            "available": True,
            "availability_reason": "available",
            "backend": record["backend"],
            "solution": record["solution"],
            "work_counts": record["work_counts"],
        }
        validate_peer_response(request, response)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
        return PeerInvocation(
            None,
            "runner-error",
            f"Python peer failed: {type(error).__name__}: {error}",
            observed_identity,
        )
    return PeerInvocation(response, None, None, observed_identity)


def run_external_peer(
    request: Mapping[str, Any],
    spec: PeerSpec,
    /,
    *,
    timeout_seconds: float = 300.0,
) -> PeerInvocation:
    if spec.runner_kind != "external-process":
        raise TypeError("run_external_peer requires an external-process peer.")
    environment_name = spec.command_environment
    if environment_name is None:
        raise ValueError("External peer lacks a command environment name.")
    command_text = os.environ.get(environment_name)
    if command_text is None:
        return PeerInvocation(
            None,
            "runtime-missing",
            f"Environment variable {environment_name} is not configured.",
            None,
        )
    try:
        command = shlex.split(command_text)
    except ValueError as error:
        return PeerInvocation(None, "runner-error", f"Invalid runner command: {error}", None)
    if not command:
        return PeerInvocation(
            None,
            "runtime-missing",
            f"Environment variable {environment_name} contains no command.",
            None,
        )
    trusted_revision = os.environ.get(f"{environment_name}_SOURCE_REVISION")
    if trusted_revision != spec.source_revision:
        return PeerInvocation(
            None,
            "revision-unverified",
            (
                f"{environment_name}_SOURCE_REVISION must equal the frozen revision "
                f"{spec.source_revision!r}."
            ),
            None,
        )
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, allow_nan=False, sort_keys=True),
            capture_output=True,
            check=False,
            text=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        return PeerInvocation(
            None,
            "runner-error",
            f"External runner exceeded {float(timeout_seconds):g} seconds.",
            None,
        )
    except OSError as error:
        return PeerInvocation(
            None,
            "runner-error",
            f"External runner could not start: {type(error).__name__}: {error}",
            None,
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return PeerInvocation(
            None,
            "runner-error",
            f"External runner exited {completed.returncode}: {detail}",
            None,
        )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return PeerInvocation(
            None,
            "runner-error",
            "External runner did not return one JSON response.",
            None,
        )
    required = {
        "request_id",
        "runner_id",
        "initial_fingerprint",
        "observed_identity",
        "source_revision",
        "available",
        "availability_reason",
        "backend",
        "solution",
        "work_counts",
    }
    if not isinstance(raw, dict) or not required <= raw.keys():
        return PeerInvocation(
            None,
            "runner-error",
            "External runner response is missing required fields.",
            None,
        )
    try:
        _validate_execution_record(raw)
    except (KeyError, TypeError, ValueError) as error:
        return PeerInvocation(
            None,
            "runner-error",
            f"External runner response is malformed: {error}",
            None,
        )
    try:
        validate_peer_response(request, raw)
    except ValueError as error:
        message = str(error)
        if "initial fingerprint" in message:
            reason = "initial-fingerprint-mismatch"
        elif "identity" in message or "revision" in message:
            reason = "revision-mismatch"
        else:
            reason = "runner-error"
        return PeerInvocation(None, reason, message, raw["observed_identity"])
    return PeerInvocation(raw, None, None, raw["observed_identity"])


def _validate_execution_record(record: Mapping[str, Any], /) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("runner record must be an object")
    if "available" in record and not isinstance(record["available"], bool):
        raise TypeError("runner available field must be boolean")
    backend = record.get("backend")
    work_counts = record.get("work_counts")
    if not isinstance(backend, Mapping):
        raise TypeError("runner backend evidence must be an object")
    for field in ("attempted", "completed"):
        if not isinstance(backend.get(field), bool):
            raise TypeError(f"runner backend {field} must be boolean")
    claimed = backend.get("claimed_success")
    if claimed is not None and not isinstance(claimed, bool):
        raise TypeError("runner backend claimed_success must be boolean or null")
    if not isinstance(work_counts, Mapping):
        raise TypeError("runner work_counts must be an object")
    for key, value in work_counts.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("runner work counts must be finite nonnegative numbers")


__all__ = [
    "PeerInvocation",
    "PeerSpec",
    "load_peer_specs",
    "make_runner_request",
    "python_runtime_identity",
    "run_external_peer",
    "run_python_peer",
    "stable_fingerprint",
    "validate_peer_response",
]
