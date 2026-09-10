#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Injected, idempotent Slurm and Kubernetes scheduler integrations."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Protocol
from urllib.parse import quote, urlsplit

from .._execution_resources import ResourceRequest
from ..logging import emit
from ._contracts import IntegrityError


class SchedulerState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in (self.SUCCEEDED, self.FAILED, self.CANCELLED)


@dataclass(frozen=True, slots=True)
class SchedulerStatus:
    scheduler_job_id: str
    state: SchedulerState
    reason: str
    exit_code: int | None = None
    resource_version: str | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandExecutor(Protocol):
    def run(
        self, argv: tuple[str, ...], /, *, stdin: bytes | None = None
    ) -> CommandResult: ...


class SubprocessCommandExecutor:
    """Explicit local executor. It never invokes a shell or interpolates arguments."""

    def run(
        self, argv: tuple[str, ...], /, *, stdin: bytes | None = None
    ) -> CommandResult:
        started = time.perf_counter()
        completed = subprocess.run(
            argv,
            input=stdin,
            shell=False,
            check=False,
            capture_output=True,
            text=False,
        )
        result = CommandResult(
            completed.returncode,
            completed.stdout.decode("utf-8", "strict"),
            completed.stderr.decode("utf-8", "replace"),
        )
        emit(
            "ERROR" if result.returncode else "DEBUG",
            (
                "provider.execution.failed"
                if result.returncode
                else "provider.execution.completed"
            ),
            "External command execution completed",
            elapsed_seconds=time.perf_counter() - started,
            executable=Path(argv[0]).name,
            return_code=result.returncode,
            stderr_bytes=len(completed.stderr),
            stdin_bytes=0 if stdin is None else len(stdin),
            stdout_bytes=len(completed.stdout),
        )
        return result


class IdempotencyLedger(Protocol):
    def lookup(self, provider_id: str, key: str, /) -> str | None: ...
    def record(self, provider_id: str, key: str, scheduler_job_id: str, /) -> str: ...


class LocalIdempotencyLedger:
    def __init__(self):
        self._records: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def lookup(self, provider_id: str, key: str, /) -> str | None:
        with self._lock:
            return self._records.get((provider_id, key))

    def record(self, provider_id: str, key: str, scheduler_job_id: str, /) -> str:
        with self._lock:
            current = self._records.setdefault((provider_id, key), scheduler_job_id)
        if current != scheduler_job_id:
            raise IntegrityError(
                "Scheduler idempotency key resolved to conflicting jobs."
            )
        return current


@dataclass(frozen=True, slots=True)
class SlurmJobSpec:
    script_path: str
    arguments: tuple[str, ...]
    idempotency_key: str
    job_name: str
    resources: ResourceRequest
    partition: str | None = None
    account: str | None = None
    time_limit: str | None = None
    ranked: bool = True
    srun_path: str = "srun"

    def __post_init__(self) -> None:
        values = (
            self.script_path,
            self.idempotency_key,
            self.job_name,
            self.srun_path,
            *self.arguments,
        )
        if any(
            not value or "\x00" in value or "\n" in value or "\r" in value
            for value in values
        ):
            raise ValueError(
                "Slurm paths, identifiers, and arguments must be nonempty single-line values."
            )
        for value in (self.partition, self.account, self.time_limit):
            if value is not None and (
                not value or "\x00" in value or "\n" in value or "\r" in value
            ):
                raise ValueError("Slurm options must be nonempty single-line values.")
        object.__setattr__(self, "arguments", tuple(self.arguments))


_SLURM_STATE_MAP = {
    "BOOT_FAIL": SchedulerState.FAILED,
    "CANCELLED": SchedulerState.CANCELLED,
    "COMPLETED": SchedulerState.SUCCEEDED,
    "COMPLETING": SchedulerState.RUNNING,
    "CONFIGURING": SchedulerState.QUEUED,
    "DEADLINE": SchedulerState.FAILED,
    "FAILED": SchedulerState.FAILED,
    "NODE_FAIL": SchedulerState.FAILED,
    "OUT_OF_MEMORY": SchedulerState.FAILED,
    "PENDING": SchedulerState.QUEUED,
    "PREEMPTED": SchedulerState.FAILED,
    "REQUEUED": SchedulerState.QUEUED,
    "RESIZING": SchedulerState.RUNNING,
    "REVOKED": SchedulerState.CANCELLED,
    "RUNNING": SchedulerState.RUNNING,
    "SIGNALING": SchedulerState.RUNNING,
    "SPECIAL_EXIT": SchedulerState.FAILED,
    "STAGE_OUT": SchedulerState.RUNNING,
    "STOPPED": SchedulerState.RUNNING,
    "SUSPENDED": SchedulerState.RUNNING,
    "TIMEOUT": SchedulerState.FAILED,
}


class SlurmScheduler:
    """Slurm provider using only argument vectors and Slurm JSON responses."""

    def __init__(
        self,
        executor: CommandExecutor,
        /,
        *,
        provider_id: str = "slurm",
        support_tuple_id: str | None = None,
        ledger: IdempotencyLedger | None = None,
        sbatch_path: str = "sbatch",
        squeue_path: str = "squeue",
        sacct_path: str = "sacct",
        scancel_path: str = "scancel",
    ):
        tuple_id = provider_id if support_tuple_id is None else support_tuple_id
        if not provider_id or not tuple_id:
            raise ValueError("Slurm provider_id and support_tuple_id must be nonempty.")
        self.provider_id = provider_id
        self.support_tuple_id = tuple_id
        self._executor = executor
        self._ledger = LocalIdempotencyLedger() if ledger is None else ledger
        self._sbatch = sbatch_path
        self._squeue = squeue_path
        self._sacct = sacct_path
        self._scancel = scancel_path
        self._submit_lock = threading.Lock()

    def submit(self, spec: SlurmJobSpec, /) -> str:
        with self._submit_lock:
            return self._submit(spec)

    def _submit(self, spec: SlurmJobSpec, /) -> str:
        existing = self._ledger.lookup(self.provider_id, spec.idempotency_key)
        if existing is not None:
            return existing
        processes = spec.resources.process_count
        hosts = spec.resources.host_count
        if spec.resources.cpu_cores % processes:
            raise ValueError("Slurm cpu_cores must divide exactly across processes.")
        if processes % hosts:
            raise ValueError("Slurm process_count must divide exactly across hosts.")
        cpus_per_task = spec.resources.cpu_cores // processes
        memory_per_host = (spec.resources.memory_bytes + hosts - 1) // hosts
        argv = [
            self._sbatch,
            "--parsable",
            f"--job-name={spec.job_name}",
            f"--comment=phydrax:{spec.idempotency_key}",
            f"--nodes={hosts}",
            f"--ntasks={processes}",
            f"--ntasks-per-node={processes // hosts}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={memory_per_host}B",
        ]
        if spec.resources.accelerator_count:
            if spec.resources.accelerator_count % processes == 0:
                argv.append(
                    f"--gpus-per-task={spec.resources.accelerator_count // processes}"
                )
            else:
                argv.append(f"--gpus={spec.resources.accelerator_count}")
        if spec.partition is not None:
            argv.append(f"--partition={spec.partition}")
        if spec.account is not None:
            argv.append(f"--account={spec.account}")
        if spec.time_limit is not None:
            argv.append(f"--time={spec.time_limit}")
        stdin = None
        if spec.ranked and processes > 1:
            rank_command = (
                spec.srun_path,
                f"--nodes={hosts}",
                f"--ntasks={processes}",
                f"--ntasks-per-node={processes // hosts}",
                "--exact",
                "--",
                spec.script_path,
                *spec.arguments,
            )
            stdin = (
                "#!/bin/sh\nset -eu\nexec " + shlex.join(rank_command) + "\n"
            ).encode("utf-8")
        else:
            argv.extend(("--", spec.script_path, *spec.arguments))
        completed = self._executor.run(tuple(argv), stdin=stdin)
        if completed.returncode:
            raise RuntimeError(
                f"Slurm submission failed: {completed.stderr.strip() or 'unknown error'}"
            )
        # --parsable yields job_id or job_id;cluster. Array suffixes remain identifiers.
        value = completed.stdout.strip().split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value):
            raise IntegrityError(
                "Slurm returned an invalid machine-readable job identifier."
            )
        return self._ledger.record(self.provider_id, spec.idempotency_key, value)

    def status(self, scheduler_job_id: str, /) -> SchedulerStatus:
        _slurm_job_id(scheduler_job_id)
        active = self._executor.run(
            (self._squeue, "--json", f"--jobs={scheduler_job_id}")
        )
        if active.returncode == 0:
            status = _parse_slurm_json(active.stdout, scheduler_job_id)
            if status is not None:
                return status
        historical = self._executor.run(
            (self._sacct, "--json", f"--jobs={scheduler_job_id}", "--allocations")
        )
        if historical.returncode:
            raise RuntimeError(
                f"Slurm state query failed: {historical.stderr.strip() or active.stderr.strip() or 'unknown error'}"
            )
        status = _parse_slurm_json(historical.stdout, scheduler_job_id)
        return status or SchedulerStatus(
            scheduler_job_id, SchedulerState.UNKNOWN, "not reported by Slurm"
        )

    def cancel(self, scheduler_job_id: str, /) -> None:
        _slurm_job_id(scheduler_job_id)
        completed = self._executor.run((self._scancel, "--", scheduler_job_id))
        if completed.returncode:
            raise RuntimeError(
                f"Slurm cancellation failed: {completed.stderr.strip() or 'unknown error'}"
            )


def _slurm_job_id(value: str) -> str:
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value):
        raise ValueError("Slurm job identifier is invalid.")
    return value


def _parse_slurm_json(payload: str, scheduler_job_id: str) -> SchedulerStatus | None:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise IntegrityError("Slurm state response is not valid JSON.") from error
    if not isinstance(decoded, dict):
        raise IntegrityError("Slurm state response must be an object.")
    jobs = decoded.get("jobs", decoded.get("job_records", []))
    if not isinstance(jobs, list):
        raise IntegrityError("Slurm state response has an invalid jobs collection.")
    for job in jobs:
        if not isinstance(job, dict):
            continue
        raw_id = job.get("job_id", job.get("job_id_raw", job.get("id")))
        if str(raw_id).split(".", 1)[0] != scheduler_job_id:
            continue
        raw_state = job.get("job_state", job.get("state", "UNKNOWN"))
        if isinstance(raw_state, list):
            raw_state = raw_state[0] if raw_state else "UNKNOWN"
        state_name = str(raw_state).split("+", 1)[0].split()[0].upper()
        state = _SLURM_STATE_MAP.get(state_name, SchedulerState.UNKNOWN)
        reason = (
            str(job.get("state_reason", job.get("reason", state_name))).strip()
            or state_name
        )
        exit_code = _slurm_exit_code(job.get("exit_code"))
        return SchedulerStatus(scheduler_job_id, state, reason, exit_code)
    return None


def _slurm_exit_code(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("status")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+(?::[0-9]+)?", value):
        return int(value.split(":", 1)[0])
    return None


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


class HTTPTransport(Protocol):
    def request(
        self,
        method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"],
        url: str,
        /,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
    ) -> HTTPResponse: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class UrllibHTTPTransport:
    """Synchronous HTTPS transport; all network effects occur only on request()."""

    def __init__(self, ssl_context: ssl.SSLContext, /, *, timeout_seconds: float = 30.0):
        if timeout_seconds <= 0:
            raise ValueError("HTTPS timeout must be positive.")
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context),
            _NoRedirectHandler(),
        )
        self._timeout = timeout_seconds

    def request(
        self,
        method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"],
        url: str,
        /,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
    ) -> HTTPResponse:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Authenticated transport only accepts absolute HTTPS URLs.")
        request = urllib.request.Request(
            url, data=body, headers=dict(headers), method=method
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return HTTPResponse(
                    response.status,
                    dict(response.headers.items()),
                    response.read(),
                )
        except urllib.error.HTTPError as error:
            return HTTPResponse(error.code, dict(error.headers.items()), error.read())


@dataclass(frozen=True, slots=True)
class KubernetesJobSpec:
    namespace: str
    image: str
    argv: tuple[str, ...]
    idempotency_key: str
    resources: ResourceRequest
    service_account: str | None = None
    labels: Mapping[str, str] | None = None
    coordinator_port: int = 12355
    accelerator_resource: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.namespace, "namespace"),
            (self.image, "image"),
            (self.idempotency_key, "idempotency_key"),
        ):
            if not value or "\x00" in value:
                raise ValueError(f"Kubernetes {name} must be nonempty.")
        if not self.argv or any(not value or "\x00" in value for value in self.argv):
            raise ValueError("Kubernetes argv must contain nonempty arguments.")
        labels = dict(self.labels or {})
        if any(not key or not value for key, value in labels.items()):
            raise ValueError("Kubernetes labels must be nonempty strings.")
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "labels", MappingProxyType(labels))
        if not 0 < int(self.coordinator_port) < 65536:
            raise ValueError("Kubernetes coordinator_port is invalid.")
        if self.accelerator_resource is not None and (
            not self.accelerator_resource or "\x00" in self.accelerator_resource
        ):
            raise ValueError("Kubernetes accelerator_resource must be nonempty.")


class KubernetesScheduler:
    """Authenticated Kubernetes Batch API provider with optimistic concurrency."""

    def __init__(
        self,
        api_server: str,
        bearer_token: str,
        transport: HTTPTransport,
        /,
        *,
        provider_id: str = "kubernetes",
        support_tuple_id: str | None = None,
    ):
        parsed_server = urlsplit(api_server)
        if (
            parsed_server.scheme != "https"
            or not parsed_server.hostname
            or parsed_server.path
            or parsed_server.query
            or parsed_server.fragment
            or parsed_server.username is not None
            or parsed_server.password is not None
            or api_server.endswith("/")
        ):
            raise ValueError(
                "Kubernetes api_server must be an HTTPS origin without credentials."
            )
        if not bearer_token or "\r" in bearer_token or "\n" in bearer_token:
            raise ValueError("Kubernetes bearer token must be nonempty and single-line.")
        tuple_id = provider_id if support_tuple_id is None else support_tuple_id
        if not provider_id or not tuple_id:
            raise ValueError(
                "Kubernetes provider_id and support_tuple_id must be nonempty."
            )
        self.provider_id = provider_id
        self.support_tuple_id = tuple_id
        self._server = api_server
        self._token = bearer_token
        self._transport = transport

    def submit(self, spec: KubernetesJobSpec, /) -> str:
        name = _kubernetes_job_name(spec.idempotency_key)
        digest = _kubernetes_spec_digest(spec)
        existing = self._get(spec.namespace, name)
        if existing is not None:
            self._verify_idempotent(existing, digest)
            if spec.resources.process_count > 1:
                self._ensure_service(spec, name, digest)
            return name
        if spec.resources.process_count > 1:
            self._ensure_service(spec, name, digest)
        body = _kubernetes_job_body(spec, name, digest)
        response = self._request(
            "POST", self._collection_url(spec.namespace), body=_json_bytes(body)
        )
        if response.status == 409:
            existing = self._get(spec.namespace, name)
            if existing is None:
                raise IntegrityError(
                    "Kubernetes reported a conflict without the existing Job."
                )
            self._verify_idempotent(existing, digest)
            return name
        created = _require_kubernetes_object(response, {200, 201}, "create")
        self._verify_idempotent(created, digest)
        return name

    def status(self, namespace: str, scheduler_job_id: str, /) -> SchedulerStatus:
        value = self._get(namespace, scheduler_job_id)
        if value is None:
            return SchedulerStatus(
                scheduler_job_id, SchedulerState.UNKNOWN, "Kubernetes Job not found"
            )
        metadata = _mapping(value.get("metadata"), "Kubernetes metadata")
        status = _mapping(value.get("status", {}), "Kubernetes status")
        version = str(metadata.get("resourceVersion", "")) or None
        if metadata.get("deletionTimestamp"):
            return SchedulerStatus(
                scheduler_job_id,
                SchedulerState.CANCELLED,
                "deletion requested",
                resource_version=version,
            )
        conditions = status.get("conditions", [])
        if not isinstance(conditions, list):
            raise IntegrityError("Kubernetes Job conditions must be a list.")
        for condition in reversed(conditions):
            if not isinstance(condition, dict) or condition.get("status") != "True":
                continue
            kind = condition.get("type")
            reason = str(
                condition.get("reason", condition.get("message", kind or "condition"))
            )
            if kind == "Complete":
                return SchedulerStatus(
                    scheduler_job_id, SchedulerState.SUCCEEDED, reason, 0, version
                )
            if kind in ("Failed", "FailureTarget"):
                return SchedulerStatus(
                    scheduler_job_id, SchedulerState.FAILED, reason, 1, version
                )
        if int(status.get("active", 0) or 0) > 0:
            state = SchedulerState.RUNNING
        else:
            state = SchedulerState.QUEUED
        return SchedulerStatus(
            scheduler_job_id,
            state,
            "active" if state is SchedulerState.RUNNING else "pending",
            resource_version=version,
        )

    def replace(
        self,
        spec: KubernetesJobSpec,
        scheduler_job_id: str,
        expected_resource_version: str,
        /,
    ) -> str:
        if not expected_resource_version:
            raise ValueError("Kubernetes optimistic update requires resourceVersion.")
        name = scheduler_job_id
        digest = _kubernetes_spec_digest(spec)
        body = _kubernetes_job_body(spec, name, digest)
        body["metadata"]["resourceVersion"] = expected_resource_version  # type: ignore[index]
        response = self._request(
            "PUT", self._object_url(spec.namespace, name), body=_json_bytes(body)
        )
        if response.status == 409:
            raise IntegrityError(
                "Kubernetes resourceVersion optimistic-concurrency check failed."
            )
        updated = _require_kubernetes_object(response, {200}, "replace")
        metadata = _mapping(updated.get("metadata"), "Kubernetes metadata")
        return str(metadata.get("resourceVersion", ""))

    def cancel(
        self, namespace: str, scheduler_job_id: str, /, *, expected_resource_version: str
    ) -> None:
        if not expected_resource_version:
            raise ValueError("Kubernetes deletion requires the observed resourceVersion.")
        body = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Foreground",
            "preconditions": {"resourceVersion": expected_resource_version},
        }
        response = self._request(
            "DELETE",
            self._object_url(namespace, scheduler_job_id),
            body=_json_bytes(body),
        )
        if response.status == 409:
            raise IntegrityError(
                "Kubernetes resourceVersion optimistic-concurrency check failed."
            )
        if response.status not in (200, 202, 404):
            _require_kubernetes_object(response, {200, 202, 404}, "delete")
        service_response = self._request(
            "DELETE",
            self._service_object_url(namespace, scheduler_job_id),
            body=_json_bytes(
                {
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "propagationPolicy": "Foreground",
                }
            ),
        )
        if service_response.status not in (200, 202, 404):
            _require_kubernetes_object(
                service_response,
                {200, 202, 404},
                "service delete",
            )

    def _get_service(self, namespace: str, name: str) -> dict[str, object] | None:
        response = self._request("GET", self._service_object_url(namespace, name))
        if response.status == 404:
            return None
        return _require_kubernetes_object(response, {200}, "service read")

    def _ensure_service(
        self,
        spec: KubernetesJobSpec,
        name: str,
        digest: str,
    ) -> None:
        existing = self._get_service(spec.namespace, name)
        if existing is not None:
            self._verify_idempotent(existing, digest)
            return
        response = self._request(
            "POST",
            self._service_collection_url(spec.namespace),
            body=_json_bytes(_kubernetes_service_body(spec, name, digest)),
        )
        if response.status == 409:
            existing = self._get_service(spec.namespace, name)
            if existing is None:
                raise IntegrityError(
                    "Kubernetes reported a Service conflict without the object."
                )
            self._verify_idempotent(existing, digest)
            return
        created = _require_kubernetes_object(
            response,
            {200, 201},
            "service create",
        )
        self._verify_idempotent(created, digest)

    def _get(self, namespace: str, name: str) -> dict[str, object] | None:
        response = self._request("GET", self._object_url(namespace, name))
        if response.status == 404:
            return None
        return _require_kubernetes_object(response, {200}, "read")

    @staticmethod
    def _verify_idempotent(value: Mapping[str, object], digest: str) -> None:
        metadata = _mapping(value.get("metadata"), "Kubernetes metadata")
        annotations = _mapping(metadata.get("annotations", {}), "Kubernetes annotations")
        if annotations.get("phydrax.io/submission-digest") != digest:
            raise IntegrityError(
                "Kubernetes object name exists for a different submission."
            )

    def _request(
        self,
        method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"],
        url: str,
        *,
        body: bytes | None = None,
    ) -> HTTPResponse:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        return self._transport.request(method, url, headers=headers, body=body)

    def _service_collection_url(self, namespace: str) -> str:
        return f"{self._server}/api/v1/namespaces/{quote(namespace, safe='')}/services"

    def _service_object_url(self, namespace: str, name: str) -> str:
        return f"{self._service_collection_url(namespace)}/{quote(name, safe='')}"

    def _collection_url(self, namespace: str) -> str:
        return f"{self._server}/apis/batch/v1/namespaces/{quote(namespace, safe='')}/jobs"

    def _object_url(self, namespace: str, name: str) -> str:
        return f"{self._collection_url(namespace)}/{quote(name, safe='')}"


def _kubernetes_job_name(idempotency_key: str) -> str:
    suffix = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:32]
    return f"phydrax-{suffix}"


def _kubernetes_spec_digest(spec: KubernetesJobSpec) -> str:
    payload = {
        "argv": list(spec.argv),
        "image": spec.image,
        "namespace": spec.namespace,
        "resources": spec.resources.to_payload(),
        "service_account": spec.service_account,
        "labels": dict(spec.labels or {}),
        "coordinator_port": spec.coordinator_port,
        "accelerator_resource": spec.accelerator_resource,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _kubernetes_job_body(
    spec: KubernetesJobSpec, name: str, digest: str
) -> dict[str, object]:
    processes = spec.resources.process_count
    if spec.resources.cpu_cores % processes:
        raise ValueError("Kubernetes cpu_cores must divide exactly across processes.")
    cpu_per_process = spec.resources.cpu_cores // processes
    memory_per_process = (spec.resources.memory_bytes + processes - 1) // processes
    limits = {
        "cpu": str(cpu_per_process),
        "memory": str(memory_per_process),
    }
    if spec.resources.accelerator_count:
        if spec.resources.accelerator_count % processes:
            raise ValueError(
                "Kubernetes accelerator_count must divide exactly across processes."
            )
        accelerator_resource = spec.accelerator_resource
        if accelerator_resource is None:
            vendor = spec.resources.accelerator_vendor
            accelerator_resource = {
                None: "nvidia.com/gpu",
                "nvidia": "nvidia.com/gpu",
                "amd": "amd.com/gpu",
                "intel": "gpu.intel.com/i915",
            }.get(vendor)
        if accelerator_resource is None:
            raise ValueError("Unknown accelerator vendor requires accelerator_resource.")
        limits[accelerator_resource] = str(spec.resources.accelerator_count // processes)
    environment: list[dict[str, object]] = []
    if processes > 1:
        environment = [
            {
                "name": "PHYDRAX_NUM_PROCESSES",
                "value": str(processes),
            },
            {
                "name": "PHYDRAX_PROCESS_ID",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": (
                            "metadata.annotations['batch.kubernetes.io/"
                            "job-completion-index']"
                        )
                    }
                },
            },
            {
                "name": "PHYDRAX_COORDINATOR_ADDRESS",
                "value": (
                    f"{name}-0.{name}.{spec.namespace}.svc.cluster.local:"
                    f"{spec.coordinator_port}"
                ),
            },
            {
                "name": "PHYDRAX_COORDINATOR_BIND_ADDRESS",
                "value": f"0.0.0.0:{spec.coordinator_port}",
            },
        ]
    pod_spec: dict[str, object] = {
        "restartPolicy": "Never",
        "subdomain": name if processes > 1 else None,
        "containers": [
            {
                "name": "worker",
                "image": spec.image,
                "command": list(spec.argv),
                "env": environment,
                "resources": {"requests": limits, "limits": limits},
            }
        ],
    }
    if processes == 1:
        del pod_spec["subdomain"]
    if spec.service_account is not None:
        pod_spec["serviceAccountName"] = spec.service_account
    template_labels = {
        "job-name": name,
        "batch.kubernetes.io/job-name": name,
    }
    job_spec: dict[str, object] = {
        "backoffLimit": 0,
        "template": {
            "metadata": {"labels": template_labels},
            "spec": pod_spec,
        },
    }
    if processes > 1:
        job_spec.update(
            {
                "completionMode": "Indexed",
                "completions": processes,
                "parallelism": processes,
            }
        )
        if spec.resources.host_count > 1:
            pod_spec["topologySpreadConstraints"] = [
                {
                    "maxSkew": 1,
                    "topologyKey": "kubernetes.io/hostname",
                    "whenUnsatisfiable": "DoNotSchedule",
                    "labelSelector": {
                        "matchLabels": {
                            "batch.kubernetes.io/job-name": name,
                        }
                    },
                }
            ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": spec.namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "phydrax",
                **dict(spec.labels or {}),
            },
            "annotations": {
                "phydrax.io/idempotency-key": spec.idempotency_key,
                "phydrax.io/submission-digest": digest,
            },
        },
        "spec": job_spec,
    }


def _kubernetes_service_body(
    spec: KubernetesJobSpec,
    name: str,
    digest: str,
) -> dict[str, object]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "namespace": spec.namespace,
            "labels": {"app.kubernetes.io/managed-by": "phydrax"},
            "annotations": {
                "phydrax.io/idempotency-key": spec.idempotency_key,
                "phydrax.io/submission-digest": digest,
            },
        },
        "spec": {
            "clusterIP": "None",
            "publishNotReadyAddresses": True,
            "selector": {"batch.kubernetes.io/job-name": name},
            "ports": [
                {
                    "name": "coordinator",
                    "port": spec.coordinator_port,
                    "targetPort": spec.coordinator_port,
                }
            ],
        },
    }


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} must be an object.")
    return value


def _require_kubernetes_object(
    response: HTTPResponse, accepted: set[int], action: str
) -> dict[str, object]:
    try:
        value = json.loads(response.body.decode("utf-8")) if response.body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            f"Kubernetes {action} response is not valid JSON."
        ) from error
    if response.status not in accepted:
        reason = (
            value.get("message", value.get("reason", f"HTTP {response.status}"))
            if isinstance(value, dict)
            else f"HTTP {response.status}"
        )
        raise RuntimeError(f"Kubernetes {action} failed: {reason}")
    if not isinstance(value, dict):
        raise IntegrityError(f"Kubernetes {action} response must be an object.")
    return value


__all__ = [
    "CommandExecutor",
    "CommandResult",
    "HTTPResponse",
    "HTTPTransport",
    "IdempotencyLedger",
    "KubernetesJobSpec",
    "KubernetesScheduler",
    "LocalIdempotencyLedger",
    "SchedulerState",
    "SchedulerStatus",
    "SlurmJobSpec",
    "SlurmScheduler",
    "SubprocessCommandExecutor",
    "UrllibHTTPTransport",
]
