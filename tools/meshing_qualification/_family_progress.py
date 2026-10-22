# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host-only, incremental family evidence; never a scientific restart checkpoint."""
from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from threading import Thread
from time import monotonic
from typing import Any

import numpy as np

from phydrax._meshcore import current_native_execution_scope
from phydrax.meshing import CellMeshingResult, NativeMeshingPhaseMeasurement
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    Deadline,
    notify_stage,
    observe_stages,
)


def read_family_progress(path: Path) -> dict[str, Any]:
    """Read the last atomically committed snapshot in a fresh interpreter."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("schema") != "family-progress-v1":
        raise ValueError("Family progress must use the family-progress-v1 schema")
    events = value.get("events")
    if not isinstance(events, list) or any(
        not isinstance(event, dict) or event.get("sequence") != index
        for index, event in enumerate(events)
    ):
        raise ValueError("Family progress must contain a contiguous event prefix")
    return value


def _atomic_snapshot(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _live_resources() -> dict[str, Any]:
    scope = current_native_execution_scope()
    if scope is None:
        return {"status": "no-live-native-scope"}
    root = scope
    while root.parent is not None:
        root = root.parent
    remaining = root.remaining()
    return {
        "status": "live-root-allowance",
        "native_status": int(remaining.status),
        "remaining_work_units": remaining.remaining_work_units,
        "remaining_geometry_queries": remaining.remaining_geometry_queries,
        "remaining_managed_bytes": remaining.remaining_managed_bytes,
        "remaining_seconds": remaining.remaining_seconds,
        "cumulative_admitted_or_consumed_work_units": (
            root.limits.maximum_work_units - remaining.remaining_work_units
        ),
        "cumulative_admitted_or_consumed_geometry_queries": (
            root.limits.maximum_geometry_queries - remaining.remaining_geometry_queries
        ),
        "simultaneous_admitted_managed_bytes": (
            root.limits.maximum_managed_bytes - remaining.remaining_managed_bytes
        ),
        "allocator_bookkeeping_bytes": root.allocator_bookkeeping(),
        "semantics": "Live values include outstanding reservations; not exact consumed counters. Ended receipt totals are recorded separately.",
    }


def _published_result(value: object) -> CellMeshingResult | None:
    # Only known wrapper edges; never walk numerical trees or retain owners.
    for _ in range(4):
        if isinstance(value, CellMeshingResult):
            return value
        if isinstance(value, tuple) and value:
            value = value[0]
        else:
            value = getattr(value, "result", None)
        if value is None:
            break
    return None


class FamilyProgress:
    """Queue immutable metadata without filesystem waits on the science thread.

    The writer fsyncs and atomically replaces each snapshot. Abrupt termination
    preserves a valid committed prefix; queued but uncommitted events may be lost.
    Only close/flush waits, after science or explicitly in reporting tests.
    """

    def __init__(self, path: Path, controls: dict[str, Any]) -> None:
        self.path = path
        self.errors: list[dict[str, str]] = []
        self._queue: Queue[bytes | None] = Queue()
        self._cohort: dict[str, Any] | None = None
        self._deadline: float | None = None
        self._published_identity: dict[str, str] | None = None
        self._thread = Thread(target=self._write, name="family-progress", daemon=True)
        self._initial = json.dumps({
            "schema": "family-progress-v1", "pid": os.getpid(),
            "controls": controls, "events": [], "first_failure": None,
            "scientific_restart_checkpoint": False,
        }).encode()
        self._thread.start()

    def _write(self) -> None:
        state = json.loads(self._initial)
        try:
            if self.path.exists():
                raise FileExistsError(f"Family progress prefix already exists: {self.path}")
            while True:
                payload = self._queue.get()
                try:
                    if payload is None:
                        return
                    event = json.loads(payload)
                    event["sequence"] = len(state["events"])
                    state["events"].append(event)
                    state["cohort"] = event["cohort"]
                    state["latest"] = event
                    for key in ("completed_request", "accepted_mesh"):
                        if event.get(key) is not None:
                            state[key] = event[key]
                    if state["first_failure"] is None and event.get("failure") is not None:
                        state["first_failure"] = event["failure"]
                    _atomic_snapshot(self.path, json.dumps(state, indent=2).encode() + b"\n")
                finally:
                    self._queue.task_done()
        except Exception as error:
            self.errors.append({"type": type(error).__name__, "message": str(error)})
            # Drain until close so producers never block or change scientific work.
            while True:
                payload = self._queue.get()
                self._queue.task_done()
                if payload is None:
                    return

    def __call__(
        self, record: dict[str, Any], name: str, event: str,
        deadline: Deadline | None, value: object, error: BaseException | None,
    ) -> None:
        if self._cohort is None:
            self._cohort = json.loads(json.dumps(record.get("runtime") or _runtime_identity()))
        if deadline is not None:
            self._deadline = deadline.monotonic_expiry()
        try:
            resources = _live_resources()
        except Exception as failure:
            resources = {
                "status": "observation-failed", "type": type(failure).__name__,
                "message": str(failure),
            }
        snapshot: dict[str, Any] = {
            "phase": name, "event": event, "monotonic_seconds": monotonic(),
            "cohort": self._cohort, "resources": resources,
            "deadline_monotonic_seconds": self._deadline,
            "remaining_scientific_seconds": None if self._deadline is None else (
                self._deadline - monotonic()
            ),
            "status": record.get("status"),
            "source_binding_id": record.get("source_binding_id"),
            "request_id": record.get("request_id"), "limits_id": record.get("limits_id"),
            "accepted_certificate_ids": {
                key: item for key, item in (record.get("initial_acceptance") or {}).items()
                if key.endswith("_id") and isinstance(item, str)
            },
            "accepted_epoch_meshes": [
                {key: item for key, item in entry.get("result", {}).items()
                 if key.endswith("_id") and isinstance(item, str)}
                for entry in record.get("epochs", [])
            ],
        }
        if isinstance(value, NativeMeshingPhaseMeasurement):
            snapshot["native_interval"] = {
                "phase": value.phase, "elapsed_seconds": value.elapsed_seconds,
                "work_units": value.work_units, "invocations": value.invocations,
            }
        result = _published_result(value)
        if result is not None:
            snapshot["published_mesh"] = {
                "result_id": str(result.result_id), "mesh_id": str(result.mesh.mesh_id),
                "topology_id": str(result.mesh.topology_id),
            }
            self._published_identity = snapshot["published_mesh"]
            receipt = result.execution_evidence
            if receipt is not None:
                peak = receipt.total_managed_admission_peak_bytes_upper
                snapshot["completed_request"] = {
                    "owner_id": None if receipt.owner_id is None else str(receipt.owner_id),
                    "work_units": int(np.asarray(receipt.total_work_units)),
                    "geometry_queries": int(np.asarray(receipt.total_geometry_queries)),
                    "host_storage_peak_bytes_upper": int(np.asarray(receipt.total_host_storage_peak_bytes_upper)),
                    "managed_admission_peak_bytes_upper": None if peak is None else int(np.asarray(peak)),
                }
        if (record.get("initial_acceptance") or {}).get("passed"):
            snapshot["accepted_mesh"] = self._published_identity
        if error is not None:
            snapshot["failure"] = {
                "phase": name, "type": type(error).__name__, "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
        elif event == "scenario_exit" and record.get("failure") is not None:
            snapshot["failure"] = {
                "phase": record.get("failure_phase", name),
                "type": record.get("failure_type"), "message": record["failure"],
            }
        self._queue.put_nowait(json.dumps(snapshot, allow_nan=False).encode())

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        self._queue.put_nowait(None)
        self._thread.join()


def record_family_native_phase(
    phases: list[NativeMeshingPhaseMeasurement], record: dict[str, Any],
    measurement: NativeMeshingPhaseMeasurement,
) -> None:
    phases.append(measurement)
    notify_stage(record, measurement.phase, "native_interval_exit", None, measurement)


def run_with_family_progress[T](
    controls: dict[str, Any], call: Callable[[], T], *, cold: bool = False,
) -> T:
    prefix = os.environ.get("FAMILY_COMPACT_PROFILE_PREFIX")
    if prefix is None:
        return call()
    suffix = f".cold-{os.getpid()}" if cold else ""
    progress = FamilyProgress(Path(prefix + suffix + "-progress.json"), controls)
    record: dict[str, Any] = {}
    failure: BaseException | None = None
    try:
        with observe_stages(progress):
            value = call()
            candidate = value if isinstance(value, dict) else getattr(value, "record", None)
            if isinstance(candidate, dict):
                record = candidate
            notify_stage(record, "scenario", "scenario_exit", None, value)
            return value
    except BaseException as error:
        failure = error
        with observe_stages(progress):
            notify_stage(record, "scenario", "failure", None, error=error)
        raise
    finally:
        progress.close()
        if progress.errors:
            record.setdefault("reporting_failures", []).extend(progress.errors)
        if record.get("reporting_failures"):
            record["reporting_status"] = "failed"
            if failure is not None:
                failure.add_note(f"Family progress reporting failed: {record['reporting_failures']}")
