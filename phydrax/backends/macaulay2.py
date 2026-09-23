#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned, data-only Macaulay2 boundary for a closed exact operation inventory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import equinox as eqx
import numpy as np

from .._external_runtime import (
    EnergyRuntimeError,
    PinnedExecutable,
    run_energy_command,
)
from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import StrictModule
from ..algebraic._exact import (
    EliminateArguments,
    ExactSparsePolynomialSystem,
    ExactSymbolicEvidence,
    ExactSymbolicOperation,
    ExactSymbolicPlan,
    ExactSymbolicResult,
    ExactSymbolicStatus,
    PreparedExactSymbolic,
)
from ..algebraic._system import SparsePolynomialSupport
from ._types import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
)


MACAULAY2_CAPABILITIES = BackendCapabilities(
    backend="macaulay2",
    problem_kinds=tuple(
        f"algebraic.exact.{operation.value}" for operation in ExactSymbolicOperation
    ),
    execution="host",
    host_only=True,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("exact-integer", "exact-rational", "prime-field"),
    supports_plan_prepare_solve_refresh=False,
    requires_explicit_release=False,
)

_WORKER_PATH = Path(__file__).with_name("_macaulay2_worker.m2")
_EXTERNAL_CLAIM = "exact_claimed_by_external_provider"


class Macaulay2IdentityError(ValueError):
    """A syntactically valid worker response belongs to another request/provider."""


class Macaulay2Environment(StrictModule):
    """Caller-pinned executable and verified installation inventory."""

    executable: PinnedExecutable = eqx.field(static=True)
    installation_root: str = eqx.field(static=True)
    installation_inventory: tuple[tuple[str, str], ...] = eqx.field(static=True)
    worker_sha256: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)

    def __init__(
        self,
        executable: PinnedExecutable,
        /,
        *,
        installation_root: str | Path | None = None,
        installation_inventory: Sequence[tuple[str, str]] = (),
    ):
        if not isinstance(executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable.")
        inventory = tuple(
            (_safe_inventory_path(path), _sha256_text(digest, "installation digest"))
            for path, digest in installation_inventory
        )
        if inventory != tuple(sorted(inventory)) or len(inventory) != len(
            {path for path, _ in inventory}
        ):
            raise ValueError("Installation inventory paths must be unique and ordered.")
        if inventory and installation_root is None:
            raise ValueError(
                "installation_root is required when installation_inventory is nonempty."
            )
        root = ""
        if installation_root is not None:
            resolved = Path(installation_root).expanduser().resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError("installation_root must be an existing directory.")
            root = str(resolved)
        worker_digest = hashlib.sha256(_WORKER_PATH.read_bytes()).hexdigest()
        self.executable = executable
        self.installation_root = root
        self.installation_inventory = inventory
        self.worker_sha256 = worker_digest
        self.environment_id = canonical_fingerprint(
            {
                "kind": "macaulay2-environment",
                "executable_sha256": executable.sha256,
                "provider_version": executable.version,
                "license_id": executable.license_id,
                "source_url": executable.source_url,
                "worker_sha256": worker_digest,
                "installation_inventory": [list(item) for item in inventory],
            }
        )

    def verify_inventory(self) -> None:
        if not self.installation_inventory:
            return
        root = Path(self.installation_root)
        for relative, expected in self.installation_inventory:
            path = (root / relative).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"Installation inventory path escaped its root: {relative!r}."
                ) from error
            if not path.is_file() or _digest_file(path) != expected:
                raise ValueError(
                    f"Installation inventory identity mismatch: {relative!r}."
                )


class Macaulay2Provider(AbstractExternalBackend):
    """Explicit fresh-process Macaulay2 provider; no discovery or fallback."""

    environment: Macaulay2Environment = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, environment: Macaulay2Environment, /):
        if not isinstance(environment, Macaulay2Environment):
            raise TypeError("environment must be Macaulay2Environment.")
        self.environment = environment
        self.provider_id = canonical_fingerprint(
            {
                "kind": "macaulay2-provider",
                "environment": environment.environment_id,
                "operations": [operation.value for operation in ExactSymbolicOperation],
            }
        )

    @property
    def name(self) -> str:
        return "macaulay2"

    @property
    def capabilities(self) -> BackendCapabilities:
        return MACAULAY2_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return macaulay2_availability(self.environment)


def macaulay2_availability(
    environment: Macaulay2Environment | None = None, /
) -> BackendAvailability:
    """Check only an explicit pin; never search PATH, install, or select a fallback."""
    requirement = "supply a user-installed hash-pinned Macaulay2 executable and exact installation inventory"
    if environment is None:
        return BackendAvailability(
            capabilities=MACAULAY2_CAPABILITIES,
            available=False,
            requirement=requirement,
            reason="no explicit Macaulay2Environment was supplied; discovery is disabled",
        )
    if not isinstance(environment, Macaulay2Environment):
        raise TypeError("environment must be Macaulay2Environment or None.")
    executable = environment.executable
    try:
        observed = _digest_file(Path(executable.path))
        worker_observed = hashlib.sha256(_WORKER_PATH.read_bytes()).hexdigest()
        environment.verify_inventory()
    except (OSError, ValueError) as error:
        return BackendAvailability(
            capabilities=MACAULAY2_CAPABILITIES,
            available=False,
            requirement=requirement,
            reason=f"pinned provider identity could not be read: {type(error).__name__}: {error}",
        )
    if observed != executable.sha256:
        return BackendAvailability(
            capabilities=MACAULAY2_CAPABILITIES,
            available=False,
            requirement=requirement,
            reason="pinned executable SHA-256 does not match its current bytes",
        )
    if worker_observed != environment.worker_sha256:
        return BackendAvailability(
            capabilities=MACAULAY2_CAPABILITIES,
            available=False,
            requirement=requirement,
            reason="packaged worker identity changed after environment preparation",
        )
    return BackendAvailability(
        capabilities=MACAULAY2_CAPABILITIES,
        available=True,
        requirement=requirement,
        reason="explicit executable and packaged worker identities match their pins",
        versions=(
            ("Macaulay2", executable.version),
            ("worker-sha256", environment.worker_sha256),
        ),
    )


def prepare_macaulay2_symbolic(
    plan: ExactSymbolicPlan, provider: Macaulay2Provider, /
) -> PreparedExactSymbolic:
    if not isinstance(plan, ExactSymbolicPlan):
        raise TypeError("plan must be ExactSymbolicPlan.")
    if not isinstance(provider, Macaulay2Provider):
        raise TypeError(
            "provider must be Macaulay2Provider; no implicit fallback exists."
        )
    provider.availability().require(f"algebraic.exact.{plan.operation.value}")
    request_id = canonical_fingerprint(
        {
            "kind": "macaulay2-exact-request",
            "plan": plan.plan_id,
            "provider": provider.provider_id,
            "environment": provider.environment.environment_id,
            "worker": provider.environment.worker_sha256,
        }
    )
    prepared = PreparedExactSymbolic(plan, provider, request_id)
    payload = macaulay2_request_payload(prepared)
    if len(payload) > plan.maximum_input_bytes:
        raise ValueError("Exact symbolic request exceeds maximum_input_bytes.")
    return prepared


def macaulay2_request_record(prepared: PreparedExactSymbolic, /) -> dict[str, object]:
    """Build the exact data-only worker record; it contains no caller code or paths."""
    provider = _prepared_provider(prepared)
    plan = prepared.plan
    system = plan.system
    return {
        "request_id": prepared.request_id,
        "plan_id": plan.plan_id,
        "system_id": system.system_id,
        "support_id": system.support.support_id,
        "domain": system.domain.to_record(),
        "variables": [f"x{index}" for index in range(system.variable_count)],
        "equation_count": system.equation_count,
        "equation_indices": np.asarray(system.support.equation_indices).tolist(),
        "exponents": np.asarray(system.support.exponents).tolist(),
        "coefficients": list(system.coefficients),
        "operation": plan.operation.value,
        "operation_args": plan.arguments.to_record(),
        "environment_id": provider.environment.environment_id,
        "provider_id": provider.provider_id,
        "provider_version": provider.environment.executable.version,
        "executable_sha256": provider.environment.executable.sha256,
        "worker_sha256": provider.environment.worker_sha256,
        "installation_inventory": [
            list(item) for item in provider.environment.installation_inventory
        ],
        "resource_limits": {
            "maximum_variable_count": plan.maximum_variable_count,
            "maximum_equation_count": plan.maximum_equation_count,
            "maximum_term_count": plan.maximum_term_count,
            "maximum_exponent_entries": plan.maximum_exponent_entries,
            "maximum_storage_bytes": plan.maximum_storage_bytes,
        },
    }


def macaulay2_request_payload(prepared: PreparedExactSymbolic, /) -> bytes:
    return (canonical_json(macaulay2_request_record(prepared)) + "\n").encode("ascii")


def execute_macaulay2_symbolic(prepared: PreparedExactSymbolic, /) -> ExactSymbolicResult:
    provider = _prepared_provider(prepared)
    plan = prepared.plan
    availability = provider.availability()
    if not availability.available:
        return ExactSymbolicResult(
            ExactSymbolicStatus.PROVIDER_FAILED,
            None,
            None,
            plan_id=plan.plan_id,
            request_id=prepared.request_id,
            diagnostic=availability.reason,
        )
    try:
        worker = _WORKER_PATH.read_bytes()
    except OSError as error:
        return ExactSymbolicResult(
            ExactSymbolicStatus.PROVIDER_FAILED,
            None,
            None,
            plan_id=plan.plan_id,
            request_id=prepared.request_id,
            diagnostic=f"Packaged Macaulay2 worker could not be read: {error}",
        )
    if hashlib.sha256(worker).hexdigest() != provider.environment.worker_sha256:
        return ExactSymbolicResult(
            ExactSymbolicStatus.PROVIDER_FAILED,
            None,
            None,
            plan_id=plan.plan_id,
            request_id=prepared.request_id,
            diagnostic="Packaged Macaulay2 worker bytes no longer match their pin.",
        )
    payload = macaulay2_request_payload(prepared)
    if len(payload) > plan.maximum_input_bytes:
        raise ValueError("Exact symbolic request exceeds maximum_input_bytes.")
    try:
        run = run_energy_command(
            provider.environment.executable,
            ("--script", "worker.m2", "--no-readline", "--silent"),
            inputs={"worker.m2": worker},
            stdin=payload,
            timeout=plan.timeout_seconds,
            max_output_bytes=plan.maximum_output_bytes,
            environment={"LC_ALL": "C"},
        )
    except EnergyRuntimeError as error:
        artifact_id = "" if error.result is None else error.result.artifact.artifact_id
        return ExactSymbolicResult(
            ExactSymbolicStatus.PROVIDER_FAILED,
            None,
            None,
            plan_id=plan.plan_id,
            request_id=prepared.request_id,
            diagnostic=(
                str(error)
                if not artifact_id
                else f"{error}; run_artifact_id={artifact_id}"
            ),
        )
    availability = provider.availability()
    if not availability.available:
        return ExactSymbolicResult(
            ExactSymbolicStatus.IDENTITY_MISMATCH,
            None,
            None,
            plan_id=plan.plan_id,
            request_id=prepared.request_id,
            diagnostic=(
                "Pinned Macaulay2 identity changed during execution: "
                f"{availability.reason}"
            ),
        )
    try:
        return parse_macaulay2_result(
            run.stdout,
            prepared,
            run_artifact_id=run.artifact.artifact_id,
        )
    except Macaulay2IdentityError as error:
        return ExactSymbolicResult(
            ExactSymbolicStatus.IDENTITY_MISMATCH,
            None,
            None,
            plan_id=plan.plan_id,
            request_id=prepared.request_id,
            diagnostic=str(error),
        )
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
        return ExactSymbolicResult(
            ExactSymbolicStatus.INVALID_OUTPUT,
            None,
            None,
            plan_id=plan.plan_id,
            request_id=prepared.request_id,
            diagnostic=f"{type(error).__name__}: {error}",
        )


def parse_macaulay2_result(
    data: bytes,
    prepared: PreparedExactSymbolic,
    /,
    *,
    run_artifact_id: str = "detached-worker-payload",
) -> ExactSymbolicResult:
    """Parse one deterministic worker result and independently check cheap invariants."""
    provider = _prepared_provider(prepared)
    if not isinstance(data, bytes):
        raise TypeError("Macaulay2 worker output must be exact bytes.")
    if len(data) > prepared.plan.maximum_output_bytes:
        raise ValueError("Macaulay2 worker output exceeds maximum_output_bytes.")
    payload = _decode_object(data)
    expected_keys = {
        "request_id",
        "plan_id",
        "system_id",
        "support_id",
        "domain",
        "operation",
        "environment_id",
        "provider_id",
        "provider_version",
        "executable_sha256",
        "worker_sha256",
        "status",
        "diagnostic",
        "claim",
        "polynomials",
        "provenance",
    }
    if set(payload) != expected_keys:
        raise ValueError("Macaulay2 worker output has an unexpected field inventory.")
    plan = prepared.plan
    environment = provider.environment
    identities = {
        "request_id": prepared.request_id,
        "plan_id": plan.plan_id,
        "system_id": plan.system.system_id,
        "support_id": plan.system.support.support_id,
        "domain": plan.system.domain.to_record(),
        "operation": plan.operation.value,
        "environment_id": environment.environment_id,
        "provider_id": provider.provider_id,
        "provider_version": environment.executable.version,
        "executable_sha256": environment.executable.sha256,
        "worker_sha256": environment.worker_sha256,
        "claim": _EXTERNAL_CLAIM,
    }
    for name, expected in identities.items():
        if payload[name] != expected:
            raise Macaulay2IdentityError(
                f"Macaulay2 worker {name} does not match the prepared request."
            )
    diagnostic = payload["diagnostic"]
    if not isinstance(diagnostic, str):
        raise ValueError("Macaulay2 worker diagnostic must be a string.")
    provenance = payload["provenance"]
    expected_provenance = {
        "provider": "Macaulay2",
        "provider_version": environment.executable.version,
        "executable_sha256": environment.executable.sha256,
        "worker_sha256": environment.worker_sha256,
        "environment_id": environment.environment_id,
        "external_exact_claim": True,
    }
    if provenance != expected_provenance:
        raise Macaulay2IdentityError("Macaulay2 provenance identities do not match.")
    if payload["status"] == "provider_error":
        if payload["polynomials"] is not None or not diagnostic:
            raise ValueError(
                "Provider-error output must contain a diagnostic and no result."
            )
        return ExactSymbolicResult(
            ExactSymbolicStatus.PROVIDER_FAILED,
            None,
            None,
            plan_id=plan.plan_id,
            request_id=prepared.request_id,
            diagnostic=diagnostic,
        )
    if payload["status"] != "success" or diagnostic:
        raise ValueError("Macaulay2 worker status is unsupported or inconsistent.")
    output = _parse_polynomials(payload["polynomials"], prepared)
    evidence = ExactSymbolicEvidence(
        provider_id=provider.provider_id,
        provider_version=environment.executable.version,
        executable_sha256=environment.executable.sha256,
        worker_sha256=environment.worker_sha256,
        run_artifact_id=run_artifact_id,
        independently_checked=(
            "closed-field-inventory",
            "request-and-provider-identities",
            "coefficient-domain-normalization",
            "canonical-sparse-support",
            "operation-variable-order",
        ),
    )
    return ExactSymbolicResult(
        ExactSymbolicStatus.SUCCESS,
        output,
        evidence,
        plan_id=plan.plan_id,
        request_id=prepared.request_id,
    )


def _parse_polynomials(
    value: object, prepared: PreparedExactSymbolic, /
) -> ExactSparsePolynomialSystem:
    if not isinstance(value, dict) or set(value) != {
        "variable_indices",
        "equation_count",
        "equation_indices",
        "exponents",
        "coefficients",
    }:
        raise ValueError("Macaulay2 polynomial output has an invalid field inventory.")
    plan = prepared.plan
    input_system = plan.system
    variable_indices = _integer_list(value["variable_indices"], "variable_indices")
    expected_variables = tuple(range(input_system.variable_count))
    if isinstance(plan.arguments, EliminateArguments):
        eliminated = set(plan.arguments.variable_indices)
        expected_variables = tuple(
            index for index in expected_variables if index not in eliminated
        )
    if variable_indices != expected_variables:
        raise ValueError(
            "Macaulay2 output changed variable order or retained eliminated variables."
        )
    equation_count = _integer(value["equation_count"], "equation_count")
    if equation_count < 1:
        raise ValueError("Macaulay2 output must contain at least one polynomial.")
    if equation_count > plan.maximum_equation_count:
        raise ValueError("Macaulay2 output equation count exceeds its resource bound.")
    equation_indices = _integer_list(value["equation_indices"], "equation_indices")
    raw_exponents = value["exponents"]
    raw_coefficients = value["coefficients"]
    if not isinstance(raw_exponents, list) or not isinstance(raw_coefficients, list):
        raise ValueError("Macaulay2 terms and coefficients must be JSON arrays.")
    if len(equation_indices) != len(raw_exponents) or len(raw_exponents) != len(
        raw_coefficients
    ):
        raise ValueError("Macaulay2 sparse term arrays have inconsistent lengths.")
    term_count = len(equation_indices)
    exponent_entries = term_count * len(variable_indices)
    estimated_storage = (
        len(variable_indices) * 64
        + equation_count * 64
        + term_count * 40
        + exponent_entries * 8
        + sum(
            len(coefficient.encode("ascii"))
            for coefficient in raw_coefficients
            if isinstance(coefficient, str)
        )
    )
    if (
        term_count > plan.maximum_term_count
        or exponent_entries > plan.maximum_exponent_entries
        or estimated_storage > plan.maximum_storage_bytes
    ):
        raise ValueError("Macaulay2 output exceeds its resource bounds.")
    exponents = tuple(
        _integer_list(row, f"exponents[{index}]")
        for index, row in enumerate(raw_exponents)
    )
    if any(len(row) != len(variable_indices) for row in exponents):
        raise ValueError("Macaulay2 exponent rows have the wrong variable count.")
    if any(index < 0 or index >= equation_count for index in equation_indices):
        raise ValueError("Macaulay2 equation index is outside its output range.")
    if any(exponent < 0 for row in exponents for exponent in row):
        raise ValueError("Macaulay2 output exponents must be nonnegative.")
    rows = tuple(zip(equation_indices, exponents, strict=True))
    if rows != tuple(sorted(rows)) or len(rows) != len(set(rows)):
        raise ValueError(
            "Macaulay2 sparse output must be unique and canonically ordered."
        )
    if any(not isinstance(coefficient, str) for coefficient in raw_coefficients):
        raise ValueError("Macaulay2 exact coefficients must be strings.")
    labels = tuple(
        input_system.support.variable_labels[index] for index in variable_indices
    )
    support = SparsePolynomialSupport(
        labels,
        tuple(f"output_{index}" for index in range(equation_count)),
        np.asarray(equation_indices, dtype=np.int64),
        np.asarray(exponents, dtype=np.int64).reshape((len(exponents), len(labels))),
    )
    return ExactSparsePolynomialSystem(
        support,
        tuple(raw_coefficients),
        input_system.domain,
    )


def _prepared_provider(prepared: PreparedExactSymbolic, /) -> Macaulay2Provider:
    if not isinstance(prepared, PreparedExactSymbolic):
        raise TypeError("prepared must be PreparedExactSymbolic.")
    if not isinstance(prepared.provider, Macaulay2Provider):
        raise TypeError("prepared provider must be Macaulay2Provider.")
    return prepared.provider


def _decode_object(data: bytes, /) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate JSON field {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")

    decoded = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(decoded, dict):
        raise ValueError("Macaulay2 worker output must be a JSON object.")
    return decoded


def _integer(value: object, name: str, /) -> int:
    if type(value) is not int:
        raise ValueError(f"Macaulay2 {name} must be an integer.")
    return value


def _integer_list(value: object, name: str, /) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Macaulay2 {name} must be an integer array.")
    return tuple(_integer(item, name) for item in value)


def _safe_inventory_path(value: str, /) -> str:
    path = str(value)
    parts = PurePosixPath(path).parts
    if (
        not path
        or path.startswith(("/", "\\"))
        or "\\" in path
        or ":" in path
        or ".." in parts
    ):
        raise ValueError(
            "Installation inventory paths must be safe relative POSIX paths."
        )
    return path


def _sha256_text(value: str, name: str, /) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _digest_file(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "MACAULAY2_CAPABILITIES",
    "Macaulay2Environment",
    "Macaulay2IdentityError",
    "Macaulay2Provider",
    "execute_macaulay2_symbolic",
    "macaulay2_availability",
    "macaulay2_request_payload",
    "macaulay2_request_record",
    "parse_macaulay2_result",
    "prepare_macaulay2_symbolic",
]
