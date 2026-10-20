# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host-only process-group lifecycle and bounded retained command evidence."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict


TEXT_LIMIT = 20_000
JSON_LIMIT = 64 * 1024 * 1024


class CommandEvidence(TypedDict):
    command: list[str]
    cwd: str
    returncode: int
    timed_out: bool
    stdout: str
    stderr: str
    stdout_path: str
    stderr_path: str
    stdout_sha256: str
    stderr_sha256: str


def _log_evidence(path: Path) -> tuple[str, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(max(0, path.stat().st_size - TEXT_LIMIT))
        text = stream.read(TEXT_LIMIT).decode("utf-8", errors="replace")
    return text, digest


def run(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str], timeout: float
) -> CommandEvidence:
    """Stream logs to disk; terminate the entire session on exit or timeout.

    Evidence embeds at most TEXT_LIMIT bytes per stream and binds retained full
    logs by SHA-256. No unbounded PIPE buffering or orphaned compiler children.
    """
    with (
        tempfile.NamedTemporaryFile(
            prefix="command-", suffix=".stdout", dir=cwd, delete=False
        ) as stdout,
        tempfile.NamedTemporaryFile(
            prefix="command-", suffix=".stderr", dir=cwd, delete=False
        ) as stderr,
    ):
        stdout_path, stderr_path = Path(stdout.name), Path(stderr.name)
        timed_out = False
        with subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        ) as process:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Kill descendants even when their direct parent has exited.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    stdout_text, stdout_digest = _log_evidence(stdout_path)
    stderr_text, stderr_digest = _log_evidence(stderr_path)
    return {
        "command": list(command),
        "cwd": str(cwd),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": stdout_digest,
        "stderr_sha256": stderr_digest,
    }


def write_json_evidence(
    path: Path, record: Mapping[str, object], *, max_bytes: int = JSON_LIMIT
) -> None:
    """Retain complete authoritative JSON or refuse; never truncate a record."""
    payload = json.dumps(dict(record), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(payload) > max_bytes:
        raise ValueError(f"JSON evidence exceeds the {max_bytes}-byte limit")
    path.write_bytes(payload)


def read_json_evidence(path: Path, *, max_bytes: int = JSON_LIMIT) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON evidence exceeds the {max_bytes}-byte limit")
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"JSON evidence exceeds the {max_bytes}-byte limit")
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError("JSON evidence must be an object")
    return record
