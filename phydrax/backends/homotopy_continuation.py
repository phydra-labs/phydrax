#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Pinned, data-only HomotopyContinuation.jl process boundary."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path

import equinox as eqx

from .._external_runtime import (
    EnergyRunResult,
    EnergyRuntimeError,
    PinnedExecutable,
    run_energy_command,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._types import BackendAvailability, BackendCapabilities


HOMOTOPY_CONTINUATION_PROTOCOL = "phydrax.homotopy-continuation"
HOMOTOPY_CONTINUATION_CAPABILITIES = BackendCapabilities(
    backend="homotopy-continuation-jl",
    problem_kinds=("algebraic.isolated-polynomial-roots",),
    execution="host",
    host_only=True,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("complex128",),
    supports_plan_prepare_solve_refresh=True,
    requires_explicit_release=False,
)

_WORKER_PATH = Path(__file__).with_name("_homotopy_continuation_worker.jl")
_WORKER_BYTES = _WORKER_PATH.read_bytes()
HOMOTOPY_CONTINUATION_WORKER_SHA256 = hashlib.sha256(_WORKER_BYTES).hexdigest()


def _sha256(data: bytes, /) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: str, name: str, /) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


class HomotopyContinuationEnvironment(StrictModule):
    """Exact Julia project identity, rechecked before and after every provider run."""

    project_path: str = eqx.field(static=True)
    project_sha256: str = eqx.field(static=True)
    manifest_sha256: str = eqx.field(static=True)
    homotopy_continuation_uuid: str = eqx.field(static=True)
    homotopy_continuation_version: str = eqx.field(static=True)
    depot_path: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)

    def __init__(
        self,
        project_path: str | Path,
        project_sha256: str,
        manifest_sha256: str,
        homotopy_continuation_uuid: str,
        homotopy_continuation_version: str,
        *,
        depot_path: str | Path = "",
    ):
        root = Path(project_path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project_path must be an existing Julia project directory.")
        project_digest = _digest(project_sha256, "project_sha256")
        manifest_digest = _digest(manifest_sha256, "manifest_sha256")
        uuid = str(homotopy_continuation_uuid).strip()
        version = str(homotopy_continuation_version).strip()
        if not uuid or not version:
            raise ValueError("Exact HomotopyContinuation UUID and version are required.")
        depot = ""
        if str(depot_path):
            depot_root = Path(depot_path).expanduser().resolve(strict=True)
            if not depot_root.is_dir():
                raise ValueError("depot_path must be an existing Julia depot directory.")
            depot = str(depot_root)
        self.project_path = str(root)
        self.project_sha256 = project_digest
        self.manifest_sha256 = manifest_digest
        self.homotopy_continuation_uuid = uuid
        self.homotopy_continuation_version = version
        self.depot_path = depot
        self.verify()
        self.environment_id = canonical_fingerprint(
            {
                "kind": "homotopy-continuation-julia-environment",
                "project_sha256": project_digest,
                "manifest_sha256": manifest_digest,
                "homotopy_continuation_uuid": uuid,
                "homotopy_continuation_version": version,
                "depot_path": depot,
            }
        )

    def verify(self) -> tuple[bytes, bytes]:
        root = Path(self.project_path)
        project = (root / "Project.toml").read_bytes()
        manifest = (root / "Manifest.toml").read_bytes()
        if _sha256(project) != self.project_sha256:
            raise ValueError("Project.toml no longer matches its pinned SHA-256.")
        if _sha256(manifest) != self.manifest_sha256:
            raise ValueError("Manifest.toml no longer matches its pinned SHA-256.")
        return project, manifest

    @property
    def project_toml(self) -> bytes:
        return self.verify()[0]

    @property
    def manifest_toml(self) -> bytes:
        return self.verify()[1]


class HomotopyContinuationPolicy(StrictModule):
    """Deterministic start-system choice and hard process/path resource bounds."""

    start_system: str = eqx.field(static=True)
    path_capacity: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    timeout_seconds: float = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        start_system: str = "polyhedral",
        path_capacity: int = 10000,
        seed: int = 0,
        timeout_seconds: float = 3600.0,
        maximum_output_bytes: int = 64 * 1024 * 1024,
    ):
        start = str(start_system).replace("_", "-")
        if start not in ("total-degree", "polyhedral"):
            raise ValueError("start_system must be 'total-degree' or 'polyhedral'.")
        capacity = int(path_capacity)
        seed_ = int(seed)
        timeout = float(timeout_seconds)
        output_bytes = int(maximum_output_bytes)
        if capacity < 1:
            raise ValueError("path_capacity must be positive.")
        if seed_ < 0 or seed_ > 2**32 - 1:
            raise ValueError("seed must fit an unsigned 32-bit integer.")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive.")
        if output_bytes < 1:
            raise ValueError("maximum_output_bytes must be positive.")
        self.start_system = start
        self.path_capacity = capacity
        self.seed = seed_
        self.timeout_seconds = timeout
        self.maximum_output_bytes = output_bytes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "homotopy-continuation-policy",
                "start_system": start,
                "path_capacity": capacity,
                "seed": seed_,
                "timeout_seconds": timeout,
                "maximum_output_bytes": output_bytes,
            }
        )


class HomotopyContinuationProvider(StrictModule):
    """Caller-pinned Julia executable and immutable package environment."""

    executable: PinnedExecutable = eqx.field(static=True)
    environment: HomotopyContinuationEnvironment = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        executable: PinnedExecutable,
        environment: HomotopyContinuationEnvironment,
        *,
        protocol_id: str = HOMOTOPY_CONTINUATION_PROTOCOL,
    ):
        if not isinstance(executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable.")
        if not isinstance(environment, HomotopyContinuationEnvironment):
            raise TypeError("environment must be a HomotopyContinuationEnvironment.")
        protocol = str(protocol_id).strip()
        if not protocol:
            raise ValueError("protocol_id must be nonempty.")
        self.executable = executable
        self.environment = environment
        self.protocol_id = protocol
        self.provider_id = canonical_fingerprint(
            {
                "kind": "homotopy-continuation-provider",
                "executable_sha256": executable.sha256,
                "executable_version": executable.version,
                "executable_license_id": executable.license_id,
                "environment": environment.environment_id,
                "protocol": protocol,
                "worker_sha256": HOMOTOPY_CONTINUATION_WORKER_SHA256,
            }
        )


class HomotopyContinuationPathStatus(str, Enum):
    REGULAR_ENDPOINT = "regular_endpoint"
    SINGULAR_ENDPOINT_CANDIDATE = "singular_endpoint_candidate"
    AT_INFINITY = "at_infinity"
    EXCESS_SOLUTION = "excess_solution"
    TRACKING_FAILED = "tracking_failed"
    INVALID_ENDPOINT = "invalid_endpoint"


class HomotopyContinuationExecutionStatus(str, Enum):
    COMPLETE = "complete"
    PROVIDER_FAILED = "provider_failed"
    INVALID_OUTPUT = "invalid_output"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    PATH_CAPACITY_EXCEEDED = "path_capacity_exceeded"


_PATH_STATUS_VALUES = frozenset(status.value for status in HomotopyContinuationPathStatus)
_COUNT_KEYS = tuple(sorted(_PATH_STATUS_VALUES))


class HomotopyContinuationRequest(StrictModule):
    """Data-only sparse polynomial request; expressions are never provider input."""

    request_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    equation_count: int = eqx.field(static=True)
    variable_count: int = eqx.field(static=True)
    equation_indices: tuple[int, ...] = eqx.field(static=True)
    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    coefficients: tuple[complex, ...] = eqx.field(static=True)

    def __init__(
        self,
        request_id: str,
        support_id: str,
        system_id: str,
        equation_count: int,
        variable_count: int,
        equation_indices,
        exponents,
        coefficients,
    ):
        request = str(request_id).strip()
        support = str(support_id).strip()
        system = str(system_id).strip()
        equation_count_ = int(equation_count)
        variable_count_ = int(variable_count)
        rows = tuple(int(value) for value in equation_indices)
        powers = tuple(tuple(int(value) for value in row) for row in exponents)
        values = tuple(complex(value) for value in coefficients)
        if not request or not support or not system:
            raise ValueError("Request, support and system identities must be nonempty.")
        if equation_count_ < 1 or variable_count_ < 1:
            raise ValueError("Polynomial dimensions must be positive.")
        if len(rows) != len(powers) or len(rows) != len(values):
            raise ValueError("Sparse polynomial term arrays must have equal lengths.")
        if any(row < 0 or row >= equation_count_ for row in rows):
            raise ValueError("equation_indices contain an out-of-range equation.")
        if any(
            len(power) != variable_count_ or any(exponent < 0 for exponent in power)
            for power in powers
        ):
            raise ValueError("Each exponent row must be nonnegative and variable-sized.")
        if any(
            not math.isfinite(value.real) or not math.isfinite(value.imag)
            for value in values
        ):
            raise ValueError("Polynomial coefficients must be finite.")
        self.request_id = request
        self.support_id = support
        self.system_id = system
        self.equation_count = equation_count_
        self.variable_count = variable_count_
        self.equation_indices = rows
        self.exponents = powers
        self.coefficients = values


class HomotopyContinuationPathRecord(StrictModule):
    path_index: int = eqx.field(static=True)
    return_code: str = eqx.field(static=True)
    status: HomotopyContinuationPathStatus = eqx.field(static=True)
    endpoint: tuple[complex, ...] | None = eqx.field(static=True)
    provider_residual_norm: float | None = eqx.field(static=True)
    condition_number: float | None = eqx.field(static=True)


class HomotopyContinuationExecution(StrictModule):
    status: HomotopyContinuationExecutionStatus = eqx.field(static=True)
    paths: tuple[HomotopyContinuationPathRecord, ...] = eqx.field(static=True)
    counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    start_count: int = eqx.field(static=True)
    tracked_path_count: int = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    run: EnergyRunResult | None = eqx.field(static=True)
    error: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)


def homotopy_continuation_availability(
    provider: HomotopyContinuationProvider | None = None, /
) -> BackendAvailability:
    """Report only an explicitly supplied pinned provider; never discover or install one."""

    if provider is None:
        return BackendAvailability(
            capabilities=HOMOTOPY_CONTINUATION_CAPABILITIES,
            available=False,
            requirement="explicit pinned Julia executable and HomotopyContinuation project",
            reason="no explicit HomotopyContinuationProvider was supplied",
        )
    if not isinstance(provider, HomotopyContinuationProvider):
        raise TypeError("provider must be a HomotopyContinuationProvider.")
    provider.environment.verify()
    return BackendAvailability(
        capabilities=HOMOTOPY_CONTINUATION_CAPABILITIES,
        available=True,
        requirement="explicit pinned Julia executable and HomotopyContinuation project",
        reason="pinned executable and Julia project identities were verified",
        versions=(
            ("julia", provider.executable.version),
            ("HomotopyContinuation", provider.environment.homotopy_continuation_version),
        ),
    )


def _execution(
    status: HomotopyContinuationExecutionStatus,
    provider: HomotopyContinuationProvider,
    policy: HomotopyContinuationPolicy,
    request: HomotopyContinuationRequest,
    /,
    *,
    paths: tuple[HomotopyContinuationPathRecord, ...] = (),
    counts: tuple[tuple[str, int], ...] = (),
    start_count: int = 0,
    tracked_path_count: int = 0,
    run: EnergyRunResult | None = None,
    error: str = "",
) -> HomotopyContinuationExecution:
    identifier = canonical_fingerprint(
        {
            "kind": "homotopy-continuation-execution",
            "status": status.value,
            "request": request.request_id,
            "support": request.support_id,
            "system": request.system_id,
            "provider": provider.provider_id,
            "policy": policy.policy_id,
            "run": None if run is None else run.artifact.artifact_id,
            "start_count": start_count,
            "tracked_path_count": tracked_path_count,
            "counts": list(counts),
            "paths": [
                {
                    "index": path.path_index,
                    "return_code": path.return_code,
                    "status": path.status.value,
                    "endpoint": None
                    if path.endpoint is None
                    else [[value.real, value.imag] for value in path.endpoint],
                    "provider_residual_norm": path.provider_residual_norm,
                    "condition_number": path.condition_number,
                }
                for path in paths
            ],
            "error": error,
        }
    )
    return HomotopyContinuationExecution(
        status,
        paths,
        counts,
        int(start_count),
        int(tracked_path_count),
        request.request_id,
        request.support_id,
        request.system_id,
        provider.provider_id,
        policy.policy_id,
        run,
        str(error),
        identifier,
    )


def _payload(
    provider: HomotopyContinuationProvider,
    policy: HomotopyContinuationPolicy,
    request: HomotopyContinuationRequest,
    /,
) -> bytes:
    record = {
        "protocol_id": provider.protocol_id,
        "request_id": request.request_id,
        "support_id": request.support_id,
        "system_id": request.system_id,
        "environment_id": provider.environment.environment_id,
        "policy_id": policy.policy_id,
        "homotopy_continuation_uuid": provider.environment.homotopy_continuation_uuid,
        "homotopy_continuation_version": provider.environment.homotopy_continuation_version,
        "start_system": policy.start_system,
        "path_capacity": policy.path_capacity,
        "seed": policy.seed,
        "equation_count": request.equation_count,
        "variable_count": request.variable_count,
        "equation_indices": list(request.equation_indices),
        "exponents": [list(row) for row in request.exponents],
        "coefficients": [[value.real, value.imag] for value in request.coefficients],
    }
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def _finite_optional(value, name: str, /) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return parsed


def _path_record(record, variable_count: int, /) -> HomotopyContinuationPathRecord:
    required = {
        "path_index",
        "return_code",
        "status",
        "endpoint",
        "provider_residual_norm",
        "condition_number",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("Provider path fields do not match the protocol.")
    index = record["path_index"]
    code = record["return_code"]
    status_value = record["status"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Provider path_index must be a nonnegative integer.")
    if not isinstance(code, str) or not code:
        raise ValueError("Provider return_code must be a nonempty string.")
    if not isinstance(status_value, str) or status_value not in _PATH_STATUS_VALUES:
        raise ValueError("Provider path status is unsupported.")
    endpoint_record = record["endpoint"]
    endpoint = None
    if endpoint_record is not None:
        if (
            not isinstance(endpoint_record, list)
            or len(endpoint_record) != variable_count
        ):
            raise ValueError("Provider endpoint dimension does not match the request.")
        values = []
        for component in endpoint_record:
            if (
                not isinstance(component, list)
                or len(component) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in component
                )
            ):
                raise ValueError(
                    "Provider endpoint components must be [real, imaginary]."
                )
            value = complex(float(component[0]), float(component[1]))
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError("Provider endpoints must be finite.")
            values.append(value)
        endpoint = tuple(values)
    status = HomotopyContinuationPathStatus(status_value)
    if (
        status
        in (
            HomotopyContinuationPathStatus.REGULAR_ENDPOINT,
            HomotopyContinuationPathStatus.SINGULAR_ENDPOINT_CANDIDATE,
        )
        and endpoint is None
    ):
        raise ValueError("A provider endpoint status requires endpoint coordinates.")
    if (
        status
        not in (
            HomotopyContinuationPathStatus.REGULAR_ENDPOINT,
            HomotopyContinuationPathStatus.SINGULAR_ENDPOINT_CANDIDATE,
        )
        and endpoint is not None
    ):
        raise ValueError("A non-endpoint provider status cannot carry coordinates.")
    return HomotopyContinuationPathRecord(
        int(index),
        code,
        status,
        endpoint,
        _finite_optional(record["provider_residual_norm"], "provider_residual_norm"),
        _finite_optional(record["condition_number"], "condition_number"),
    )


def _parse_output(
    data: bytes,
    provider: HomotopyContinuationProvider,
    policy: HomotopyContinuationPolicy,
    request: HomotopyContinuationRequest,
    run: EnergyRunResult,
    /,
) -> HomotopyContinuationExecution:
    record = json.loads(data.decode("ascii", errors="strict"))
    required = {
        "protocol_id",
        "request_id",
        "support_id",
        "system_id",
        "environment_id",
        "policy_id",
        "homotopy_continuation_uuid",
        "homotopy_continuation_version",
        "start_system",
        "execution_status",
        "start_count",
        "tracked_path_count",
        "counts",
        "paths",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("Provider output fields do not match the protocol.")
    expected = (
        ("protocol_id", provider.protocol_id),
        ("request_id", request.request_id),
        ("support_id", request.support_id),
        ("system_id", request.system_id),
        ("environment_id", provider.environment.environment_id),
        ("policy_id", policy.policy_id),
        (
            "homotopy_continuation_uuid",
            provider.environment.homotopy_continuation_uuid,
        ),
        (
            "homotopy_continuation_version",
            provider.environment.homotopy_continuation_version,
        ),
        ("start_system", policy.start_system),
    )
    if any(record[key] != value for key, value in expected):
        return _execution(
            HomotopyContinuationExecutionStatus.SEMANTIC_MISMATCH,
            provider,
            policy,
            request,
            run=run,
            error="Provider output identity differs from the submitted request/environment.",
        )
    execution_status = record["execution_status"]
    if execution_status not in ("complete", "path_capacity_exceeded"):
        raise ValueError("Provider execution_status is unsupported.")
    start_count = record["start_count"]
    tracked_count = record["tracked_path_count"]
    if (
        isinstance(start_count, bool)
        or not isinstance(start_count, int)
        or start_count < 0
        or isinstance(tracked_count, bool)
        or not isinstance(tracked_count, int)
        or tracked_count < 0
    ):
        raise ValueError("Provider path counts must be nonnegative integers.")
    raw_counts = record["counts"]
    if not isinstance(raw_counts, dict) or set(raw_counts) != set(_COUNT_KEYS):
        raise ValueError("Provider path-count fields do not match the protocol.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in raw_counts.values()
    ):
        raise ValueError("Provider classified path counts must be nonnegative integers.")
    if not isinstance(record["paths"], list):
        raise ValueError("Provider paths must be an array.")
    paths = tuple(_path_record(path, request.variable_count) for path in record["paths"])
    if tracked_count != len(paths) or tuple(path.path_index for path in paths) != tuple(
        range(tracked_count)
    ):
        raise ValueError(
            "Provider paths do not exactly account for tracked path indices."
        )
    observed = {key: 0 for key in _COUNT_KEYS}
    for path in paths:
        observed[path.status.value] += 1
    if observed != raw_counts or sum(raw_counts.values()) != tracked_count:
        raise ValueError(
            "Provider classified counts do not account for every tracked path."
        )
    counts = tuple((key, raw_counts[key]) for key in _COUNT_KEYS)
    if execution_status == "path_capacity_exceeded":
        if start_count <= policy.path_capacity or tracked_count != 0:
            raise ValueError("Provider capacity status contradicts path counts.")
        status = HomotopyContinuationExecutionStatus.PATH_CAPACITY_EXCEEDED
    else:
        if start_count > policy.path_capacity or tracked_count != start_count:
            raise ValueError("Provider complete status contradicts path counts/capacity.")
        status = HomotopyContinuationExecutionStatus.COMPLETE
    return _execution(
        status,
        provider,
        policy,
        request,
        paths=paths,
        counts=counts,
        start_count=start_count,
        tracked_path_count=tracked_count,
        run=run,
    )


def execute_homotopy_continuation(
    provider: HomotopyContinuationProvider,
    policy: HomotopyContinuationPolicy,
    request: HomotopyContinuationRequest,
    /,
) -> HomotopyContinuationExecution:
    """Run the bounded JSON worker and retain every raw path and return code."""

    if not isinstance(provider, HomotopyContinuationProvider):
        raise TypeError("provider must be a HomotopyContinuationProvider.")
    if not isinstance(policy, HomotopyContinuationPolicy):
        raise TypeError("policy must be a HomotopyContinuationPolicy.")
    if not isinstance(request, HomotopyContinuationRequest):
        raise TypeError("request must be a HomotopyContinuationRequest.")
    try:
        project, manifest = provider.environment.verify()
    except (OSError, ValueError) as failure:
        return _execution(
            HomotopyContinuationExecutionStatus.PROVIDER_FAILED,
            provider,
            policy,
            request,
            error=f"Pinned Julia project verification failed before execution: {failure}",
        )
    worker = _WORKER_PATH.read_bytes()
    if worker != _WORKER_BYTES:
        return _execution(
            HomotopyContinuationExecutionStatus.PROVIDER_FAILED,
            provider,
            policy,
            request,
            error="The fixed HomotopyContinuation worker changed before execution.",
        )
    environment = {
        "JULIA_PKG_OFFLINE": "true",
        "LC_ALL": "C",
        "OPENBLAS_NUM_THREADS": "1",
    }
    if provider.environment.depot_path:
        environment["JULIA_DEPOT_PATH"] = provider.environment.depot_path
    run: EnergyRunResult | None = None
    runtime_error = ""
    try:
        run = run_energy_command(
            provider.executable,
            (
                "--startup-file=no",
                "--history-file=no",
                "--threads=1",
                "--project=julia-project",
                "homotopy-continuation-worker.jl",
                "homotopy-input.json",
                "homotopy-output.json",
            ),
            inputs={
                "julia-project/Project.toml": project,
                "julia-project/Manifest.toml": manifest,
                "homotopy-continuation-worker.jl": worker,
                "homotopy-input.json": _payload(provider, policy, request),
            },
            outputs=("homotopy-output.json",),
            timeout=policy.timeout_seconds,
            max_output_bytes=policy.maximum_output_bytes,
            environment=environment,
        )
    except EnergyRuntimeError as failure:
        run = failure.result
        runtime_error = str(failure)
    except ValueError as failure:
        runtime_error = str(failure)
    try:
        provider.environment.verify()
    except (OSError, ValueError) as failure:
        return _execution(
            HomotopyContinuationExecutionStatus.PROVIDER_FAILED,
            provider,
            policy,
            request,
            run=run,
            error=f"Pinned Julia project verification failed after execution: {failure}",
        )
    if _WORKER_PATH.read_bytes() != _WORKER_BYTES:
        return _execution(
            HomotopyContinuationExecutionStatus.PROVIDER_FAILED,
            provider,
            policy,
            request,
            run=run,
            error="The fixed HomotopyContinuation worker changed during execution.",
        )
    if runtime_error:
        return _execution(
            HomotopyContinuationExecutionStatus.PROVIDER_FAILED,
            provider,
            policy,
            request,
            run=run,
            error=runtime_error,
        )
    assert run is not None
    try:
        return _parse_output(
            run.output("homotopy-output.json"), provider, policy, request, run
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as failure:
        return _execution(
            HomotopyContinuationExecutionStatus.INVALID_OUTPUT,
            provider,
            policy,
            request,
            run=run,
            error=str(failure),
        )


__all__ = [
    "HOMOTOPY_CONTINUATION_PROTOCOL",
    "HOMOTOPY_CONTINUATION_CAPABILITIES",
    "HomotopyContinuationEnvironment",
    "HOMOTOPY_CONTINUATION_WORKER_SHA256",
    "HomotopyContinuationPolicy",
    "HomotopyContinuationProvider",
    "HomotopyContinuationPathStatus",
    "HomotopyContinuationExecutionStatus",
    "HomotopyContinuationRequest",
    "HomotopyContinuationPathRecord",
    "HomotopyContinuationExecution",
    "homotopy_continuation_availability",
    "execute_homotopy_continuation",
]
