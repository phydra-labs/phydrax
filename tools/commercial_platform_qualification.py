#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic, injected qualification for commercial platform boundaries.

The runner deliberately has no cloud, scheduler, identity-provider, telemetry, or
subprocess adapter.  A qualification environment supplies a deterministic probe
provider and an exact fault matrix.  The resulting records are candidates only:
they are content-addressed, unsigned, unreleased, and bound to one provider and
one deployment.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import (
    QualificationCoverageReport,
    QualificationEvidence,
    QualificationMatrix,
    SupportDependency,
)


GATES = ("scientific", "performance", "operational", "security")
COMMON_BOUNDARIES = (
    "platform.no-hidden-external-effects",
    "platform.no-secret-leakage",
)
ROUTE_BOUNDARIES: dict[str, tuple[str, ...]] = {
    "posix-repository": (
        "posix.before-manifest",
        "posix.after-manifest",
        "posix.before-commit-marker",
        "posix.after-commit-marker",
        "posix.before-pointer",
        "posix.after-pointer",
        "posix.chunk-corruption",
    ),
    "s3-repository": (
        "s3.conditional-create-conflict",
        "s3.conditional-replace-conflict",
        "s3.stale-writer",
        "s3.bounded-read",
        "s3.object-corruption",
        "s3.metadata-guard-conflict",
        "s3.externally-fenced-guard-recovery",
    ),
    "hpc-filesystem": (
        "hpc.directory-fsync-required",
        "hpc.advisory-locking-required",
        "hpc.attempt-private-staging-required",
        "hpc.provider-mismatch",
    ),
    "chunk-lifecycle": (
        "chunk.duplicate-write",
        "chunk.stale-attempt",
        "lease.active-pins-artifact",
        "lease.expiry",
        "hold.active-pins-artifact",
        "gc.unreachable-history",
        "gc.tombstone-after-lease-and-hold",
    ),
    "topology-restart": (
        "topology.same-bitwise",
        "topology.qualified-change",
        "topology.unsupported-change",
        "topology.chunk-hole",
        "topology.mapping-overlap",
        "topology.bounded-direct-restore",
    ),
    "production-local": (
        "runtime.local.atomic-pointer",
        "runtime.local.monotone-generation",
        "runtime.local.corrupt-pointer",
        "runtime.local.restart",
    ),
    "production-repository": (
        "runtime.repository.before-manifest",
        "runtime.repository.after-manifest",
        "runtime.repository.before-pointer",
        "runtime.repository.resume",
        "runtime.repository.duplicate-output",
        "runtime.repository.outbox-replay",
        "runtime.repository.provider-mismatch",
    ),
    "durable-service": (
        "service.transaction-rollback",
        "service.duplicate-request",
        "service.conflicting-duplicate-request",
        "service.stale-execution-attempt",
        "service.execution-lease-expiry",
        "service.execution-heartbeat-renewal",
        "service.attempt-version-fenced-completion",
        "service.cross-tenant-denial",
        "service.append-only-audit",
        "service.outbox-claim-lease",
        "service.outbox-retry",
        "service.duplicate-delivery",
    ),
    "slurm": (
        "slurm.argv-no-shell",
        "slurm.duplicate-submit",
        "slurm.stale-attempt",
        "slurm.machine-state",
    ),
    "kubernetes": (
        "kubernetes.authenticated-request",
        "kubernetes.duplicate-submit",
        "kubernetes.resource-version-conflict",
        "kubernetes.namespace-isolation",
    ),
    "oidc-jwks": (
        "oidc.issuer",
        "oidc.audience",
        "oidc.expiry",
        "oidc.not-before",
        "jwks.rotation",
        "jwks.key-expiry",
        "jwks.key-revocation",
        "jwks.constructor-no-fetch",
    ),
    "mtls": (
        "mtls.spiffe-san",
        "mtls.issuer",
        "mtls.expiry",
        "mtls.revocation",
    ),
    "ed25519": (
        "ed25519.invalid-signature",
        "ed25519.key-expiry",
        "ed25519.key-revocation",
        "ed25519.rotation",
    ),
    "kms": (
        "kms.injected-only",
        "kms.verify-failure",
        "kms.key-mismatch",
        "kms.key-expiry",
        "kms.key-revocation",
    ),
    "redaction-support": (
        "secret.cross-tenant-denial",
        "secret.expiry",
        "secret.repr-redaction",
        "redaction.recursive",
        "support.allowlist",
        "support.privacy-bound",
    ),
    "provenance-sbom": (
        "provenance.source-digest",
        "provenance.lock-digest",
        "provenance.deterministic-build-record",
        "sbom.spdx-content",
        "sbom.provenance-binding",
    ),
    "exact-admission": (
        "admission.repository",
        "admission.scheduler",
        "admission.authentication-policy",
        "admission.support-tuple",
        "admission.resolved-run-spec",
        "admission.before-allocation",
    ),
    "configuration-migration": (
        "migration.unsupported-format",
        "migration.ambiguous-path",
        "migration.lossy-denial",
        "migration.transform-purity",
        "migration.lineage",
        "migration.immutable-parent-rollback",
        "migration.provider-mismatch",
    ),
}

_ROUTE_BINDING_ROLE = {
    "posix-repository": "repository",
    "s3-repository": "repository",
    "hpc-filesystem": "repository",
    "chunk-lifecycle": "repository",
    "topology-restart": "deployment",
    "production-local": "deployment",
    "production-repository": "repository",
    "durable-service": "deployment",
    "slurm": "scheduler",
    "kubernetes": "scheduler",
    "oidc-jwks": "authentication",
    "mtls": "authentication",
    "ed25519": "deployment",
    "kms": "deployment",
    "redaction-support": "deployment",
    "provenance-sbom": "deployment",
    "exact-admission": "deployment",
    "configuration-migration": "deployment",
}
_SECURITY_ROUTES = frozenset(
    (
        "oidc-jwks",
        "mtls",
        "ed25519",
        "kms",
        "redaction-support",
        "provenance-sbom",
        "exact-admission",
    )
)
_SENSITIVE_KEYS = frozenset(
    (
        "authorization",
        "client_secret",
        "credential",
        "password",
        "private_key",
        "secret",
        "secret_value",
        "token",
    )
)
_OUTCOMES = frozenset(("passed", "failed", "inconclusive"))


def _identifier(value: object, name: str, /) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _timestamp(value: object, name: str, /) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be a non-negative signed 64-bit timestamp.")
    return value


def _json_value(value: object, path: str = "$", /) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{path} has a non-string or empty object key.")
            normalized[key] = _json_value(item, f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} is not canonical JSON data.")


def _json_object(value: object, name: str, /) -> dict[str, object]:
    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must be a JSON object mapping.")
    return normalized


def _identifiers(
    values: Sequence[str], name: str, /, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    normalized = tuple(_identifier(value, name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} cannot be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} cannot contain duplicates.")
    return normalized


def _redact(
    value: object,
    forbidden_values: tuple[str, ...],
    /,
    *,
    key: str | None = None,
) -> tuple[object, bool]:
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        leaked = False
        for child_key, child in value.items():
            name = str(child_key)
            if name.casefold() in _SENSITIVE_KEYS and child != "<redacted>":
                output[name] = "<redacted>"
                leaked = True
                continue
            sanitized, child_leaked = _redact(child, forbidden_values, key=name)
            output[name] = sanitized
            leaked = leaked or child_leaked
        return output, leaked
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        output_items = []
        leaked = False
        for child in value:
            sanitized, child_leaked = _redact(child, forbidden_values, key=key)
            output_items.append(sanitized)
            leaked = leaked or child_leaked
        return output_items, leaked
    if type(value) is str:
        leaked = False
        sanitized = value
        lowered = value.casefold()
        if lowered.startswith("bearer ") or lowered.startswith("basic "):
            sanitized = "<redacted>"
            leaked = True
        for forbidden in forbidden_values:
            if forbidden in sanitized:
                sanitized = sanitized.replace(forbidden, "<redacted>")
                leaked = True
        return sanitized, leaked
    return value, False


def _content_record(
    kind: str, fields: Mapping[str, object], id_name: str, /
) -> dict[str, object]:
    core = _json_object({"kind": kind, **dict(fields)}, "content record")
    return {**core, id_name: canonical_fingerprint(core)}


def required_boundaries(route: str, /) -> tuple[str, ...]:
    """Return the complete ordered fault boundary set for one platform route."""

    route_ = _identifier(route, "route")
    if route_ not in ROUTE_BOUNDARIES:
        raise ValueError(f"Unknown commercial platform route {route_!r}.")
    return COMMON_BOUNDARIES + ROUTE_BOUNDARIES[route_]


def boundary_gate(route: str, boundary_id: str, /) -> str:
    """Return the fixed evidence kind for a platform boundary."""

    if boundary_id == "platform.no-hidden-external-effects":
        return "operational"
    if boundary_id == "platform.no-secret-leakage":
        return "security"
    if boundary_id not in ROUTE_BOUNDARIES.get(route, ()):
        raise ValueError(f"Boundary {boundary_id!r} does not belong to route {route!r}.")
    return "security" if route in _SECURITY_ROUTES else "operational"


class FaultCase:
    """One exact injected stimulus and expected fail-closed observation."""

    __slots__ = (
        "boundary_id",
        "gate",
        "_stimulus_json",
        "expected_facts",
        "allowed_effects",
        "stimulus_id",
        "case_id",
    )

    def __init__(
        self,
        boundary_id: str,
        gate: str,
        stimulus: Mapping[str, object],
        expected_facts: Mapping[str, object],
        /,
        *,
        allowed_effects: Sequence[str] = (),
    ):
        boundary = _identifier(boundary_id, "fault boundary ID")
        gate_ = _identifier(gate, "fault gate")
        if gate_ not in ("operational", "security"):
            raise ValueError("Fault cases must produce operational or security evidence.")
        stimulus_ = _json_object(stimulus, "fault stimulus")
        expected = _json_object(expected_facts, "expected fault facts")
        expected_redacted, expected_leaked = _redact(expected, ())
        if expected_leaked:
            raise ValueError("Expected fault facts must not contain secret material.")
        effects = _identifiers(allowed_effects, "allowed effect IDs", allow_empty=True)
        self.boundary_id = boundary
        self.gate = gate_
        self._stimulus_json = canonical_json(stimulus_)
        self.expected_facts = expected_redacted
        self.allowed_effects = effects
        self.stimulus_id = canonical_fingerprint(
            {"kind": "fault-stimulus", "value": stimulus_}
        )
        core = self._content_record()
        self.case_id = canonical_fingerprint(core)

    @property
    def stimulus(self) -> dict[str, object]:
        """Return an independent copy; stimulus values are never serialized in evidence."""

        value = json.loads(self._stimulus_json)
        if not isinstance(value, dict):
            raise RuntimeError("Stored fault stimulus is not an object.")
        return value

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "commercial-platform-fault-case",
            "boundary_id": self.boundary_id,
            "gate": self.gate,
            "stimulus_id": self.stimulus_id,
            "expected_facts": self.expected_facts,
            "allowed_effects": list(self.allowed_effects),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "case_id": self.case_id}

    def __repr__(self) -> str:
        return f"FaultCase(boundary_id={self.boundary_id!r}, case_id={self.case_id!r})"


class FaultObservation:
    """JSON facts and the exact effects declared by an injected probe."""

    __slots__ = ("facts", "declared_effects")

    def __init__(
        self,
        facts: Mapping[str, object],
        /,
        *,
        declared_effects: Sequence[str] = (),
    ):
        self.facts = _json_object(facts, "fault observation facts")
        self.declared_effects = _identifiers(
            declared_effects, "declared effect IDs", allow_empty=True
        )

    def __repr__(self) -> str:
        return "FaultObservation(<redacted-facts>)"


class DeterministicProbeProvider(Protocol):
    """Injected provider contract; implementations must not use live services."""

    provider_id: str
    deployment_id: str

    def effect_log(self) -> Sequence[str]: ...

    def exercise(self, case: FaultCase, /) -> FaultObservation: ...


class FaultMatrix:
    """Complete fault matrix for one exact provider and deployment route."""

    __slots__ = (
        "route",
        "provider_id",
        "deployment_id",
        "binding_role",
        "support_dependency_id",
        "cases",
        "matrix_id",
    )

    def __init__(
        self,
        route: str,
        provider_id: str,
        deployment_id: str,
        support_dependency_id: str,
        cases: Sequence[FaultCase],
        /,
    ):
        route_ = _identifier(route, "route")
        if route_ not in ROUTE_BOUNDARIES:
            raise ValueError(f"Unknown commercial platform route {route_!r}.")
        if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes, bytearray)):
            raise TypeError("fault cases must be a sequence.")
        cases_ = tuple(cases)
        if any(not isinstance(case, FaultCase) for case in cases_):
            raise TypeError("fault cases must contain FaultCase values.")
        expected = required_boundaries(route_)
        actual = tuple(case.boundary_id for case in cases_)
        if len(set(actual)) != len(actual):
            raise ValueError("Fault matrix contains duplicate boundary IDs.")
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError("Fault matrix is incomplete: " + "; ".join(details) + ".")
        for case in cases_:
            expected_gate = boundary_gate(route_, case.boundary_id)
            if case.gate != expected_gate:
                raise ValueError(
                    f"Boundary {case.boundary_id!r} must use the {expected_gate!r} gate."
                )
        self.route = route_
        self.provider_id = _identifier(provider_id, "provider ID")
        self.deployment_id = _identifier(deployment_id, "deployment ID")
        self.binding_role = _ROUTE_BINDING_ROLE[route_]
        self.support_dependency_id = _identifier(
            support_dependency_id, "support dependency ID"
        )
        by_boundary = {case.boundary_id: case for case in cases_}
        self.cases = tuple(by_boundary[boundary] for boundary in expected)
        self.matrix_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "commercial-platform-fault-matrix",
            "route": self.route,
            "provider_id": self.provider_id,
            "deployment_id": self.deployment_id,
            "binding_role": self.binding_role,
            "support_dependency_id": self.support_dependency_id,
            "cases": [case.to_record() for case in self.cases],
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "matrix_id": self.matrix_id}


def standard_fault_matrix(
    route: str,
    provider_id: str,
    deployment_id: str,
    support_dependency_id: str,
    /,
) -> FaultMatrix:
    """Create the canonical matrix expected from a deterministic injected provider."""

    cases = []
    for boundary_id in required_boundaries(route):
        effects = (
            ()
            if boundary_id == "platform.no-hidden-external-effects"
            else (f"injected:{route}:{boundary_id}",)
        )
        cases.append(
            FaultCase(
                boundary_id,
                boundary_gate(route, boundary_id),
                {"fault": boundary_id, "route": route},
                {"boundary_id": boundary_id, "contract_satisfied": True},
                allowed_effects=effects,
            )
        )
    return FaultMatrix(
        route,
        provider_id,
        deployment_id,
        support_dependency_id,
        cases,
    )


class QualificationContext:
    """Exact build, execution, review, and validity context for generated evidence."""

    __slots__ = (
        "build_id",
        "environment_id",
        "backend",
        "topology",
        "precision",
        "reduction",
        "replay_id",
        "reviewer_id",
        "issued_at",
        "expires_at",
        "evaluated_at",
        "context_id",
    )

    def __init__(
        self,
        *,
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        reduction: str,
        replay_id: str,
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
        evaluated_at: int,
    ):
        self.build_id = _identifier(build_id, "build ID")
        self.environment_id = _identifier(environment_id, "environment ID")
        self.backend = _identifier(backend, "backend")
        self.topology = _identifier(topology, "topology")
        self.precision = _identifier(precision, "precision")
        self.reduction = _identifier(reduction, "reduction")
        self.replay_id = _identifier(replay_id, "replay ID")
        self.reviewer_id = _identifier(reviewer_id, "reviewer ID")
        self.issued_at = _timestamp(issued_at, "issued_at")
        self.expires_at = _timestamp(expires_at, "expires_at")
        self.evaluated_at = _timestamp(evaluated_at, "evaluated_at")
        if self.expires_at <= self.issued_at:
            raise ValueError("Qualification context must expire after it is issued.")
        if not self.issued_at <= self.evaluated_at <= self.expires_at:
            raise ValueError("evaluated_at must lie in the evidence validity window.")
        self.context_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "commercial-platform-qualification-context",
            "build_id": self.build_id,
            "environment_id": self.environment_id,
            "backend": self.backend,
            "topology": self.topology,
            "precision": self.precision,
            "reduction": self.reduction,
            "replay_id": self.replay_id,
            "reviewer_id": self.reviewer_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "evaluated_at": self.evaluated_at,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "context_id": self.context_id}


def _support_bindings(
    values: Sequence[SupportDependency], resolved_run_spec: ResolvedRunSpec, /
) -> tuple[SupportDependency, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("support_dependencies must be a sequence.")
    dependencies = tuple(values)
    if any(not isinstance(value, SupportDependency) for value in dependencies):
        raise TypeError("support_dependencies must contain SupportDependency values.")
    identifiers = tuple(value.dependency_id for value in dependencies)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("support_dependencies cannot contain duplicates.")
    resolved = (
        resolved_run_spec.scientific_dependencies
        + resolved_run_spec.deployment_dependencies
    )
    resolved_by_id = {value.dependency_id: value.to_record() for value in resolved}
    supplied_by_id = {value.dependency_id: value.to_record() for value in dependencies}
    if supplied_by_id != resolved_by_id:
        raise ValueError(
            "support_dependencies must exactly equal the ResolvedRunSpec dependencies."
        )
    return tuple(sorted(dependencies, key=lambda value: value.dependency_id))


def _provider_binding(matrix: FaultMatrix, resolved_run_spec: ResolvedRunSpec, /) -> None:
    if matrix.binding_role == "repository":
        expected = resolved_run_spec.repository_id
    elif matrix.binding_role == "scheduler":
        expected = resolved_run_spec.scheduler_id
    elif matrix.binding_role == "authentication":
        expected = resolved_run_spec.auth_policy_id
    else:
        return
    if matrix.provider_id != expected:
        raise ValueError(
            f"Fault-matrix provider does not match the resolved {matrix.binding_role} binding."
        )


def _evidence_predicate(
    gate: str,
    criterion_id: str,
    context: QualificationContext,
    resolved_run_spec: ResolvedRunSpec,
    /,
) -> dict[str, str]:
    return {
        "evidence_kind": gate,
        "criterion_id": criterion_id,
        "subject_id": resolved_run_spec.spec_id,
        "build_id": context.build_id,
        "environment_id": context.environment_id,
        "backend": context.backend,
        "topology": context.topology,
        "precision": context.precision,
        "reduction": context.reduction,
        "replay_id": context.replay_id,
    }


def _gate_record(
    gate: str,
    criterion_ids: Sequence[str],
    evidence: Sequence[QualificationEvidence],
    context: QualificationContext,
    resolved_run_spec: ResolvedRunSpec,
    /,
) -> dict[str, object]:
    criteria = _identifiers(criterion_ids, f"{gate} criterion IDs")
    matrix = QualificationMatrix(
        {
            f"{gate}:{criterion_id}": _evidence_predicate(
                gate, criterion_id, context, resolved_run_spec
            )
            for criterion_id in criteria
        }
    )
    report = matrix.evaluate(evidence, at_time=context.evaluated_at)
    return {
        "gate": gate,
        "matrix": matrix.to_record(),
        "coverage": report.to_record(),
        "outcome": report.outcome,
    }


def _observation_record(
    matrix: FaultMatrix,
    case: FaultCase,
    observation: FaultObservation,
    before_effects: tuple[str, ...],
    after_effects: tuple[str, ...],
    forbidden_values: tuple[str, ...],
    /,
) -> tuple[dict[str, object], bool, str]:
    sanitized, facts_leaked = _redact(observation.facts, forbidden_values)
    facts = _json_object(sanitized, "sanitized observation facts")
    log_is_append_only = (
        len(after_effects) >= len(before_effects)
        and after_effects[: len(before_effects)] == before_effects
    )
    observed_effects = after_effects[len(before_effects) :] if log_is_append_only else ()
    declared_record, declared_leaked = _redact(
        observation.declared_effects, forbidden_values
    )
    observed_record, observed_leaked = _redact(observed_effects, forbidden_values)
    leaked = facts_leaked or declared_leaked or observed_leaked
    facts_match = canonical_json(facts) == canonical_json(case.expected_facts)
    initial_effects_clean = (
        case.boundary_id != "platform.no-hidden-external-effects" or not before_effects
    )
    effects_match = (
        log_is_append_only
        and initial_effects_clean
        and observed_effects == case.allowed_effects
        and observation.declared_effects == observed_effects
    )
    passed = facts_match and effects_match and not leaked
    if leaked:
        reason = "Injected observation exposed secret material."
    elif not facts_match:
        reason = "Injected observation did not satisfy the exact expected facts."
    elif not effects_match:
        reason = "Injected provider produced undeclared or inconsistent external effects."
    else:
        reason = "Injected observation satisfied the exact fault-boundary contract."
    record = _content_record(
        "commercial-platform-fault-observation",
        {
            "route": matrix.route,
            "provider_id": matrix.provider_id,
            "deployment_id": matrix.deployment_id,
            "case_id": case.case_id,
            "boundary_id": case.boundary_id,
            "gate": case.gate,
            "facts": facts,
            "declared_effects": list(declared_record),
            "observed_effects": list(observed_record),
            "effect_log_append_only": log_is_append_only,
            "initial_effects_clean": initial_effects_clean,
            "facts_matched": facts_match,
            "effects_matched": effects_match,
            "secret_leak_detected": leaked,
            "outcome": "passed" if passed else "failed",
            "reason": reason,
        },
        "observation_id",
    )
    return record, passed, reason


def produce_provider_qualification(
    provider: DeterministicProbeProvider,
    matrix: FaultMatrix,
    /,
    *,
    context: QualificationContext,
    support_dependencies: Sequence[SupportDependency],
    resolved_run_spec: ResolvedRunSpec,
    source_evidence: Sequence[QualificationEvidence],
    scientific_criteria: Sequence[str],
    performance_criteria: Sequence[str],
    forbidden_values: Sequence[str] = (),
) -> dict[str, object]:
    """Exercise one injected matrix and return an unsigned provider candidate.

    Scientific and performance gates consume only caller-supplied exact evidence.
    Operational and security gates consume only evidence generated from the injected
    observations.  Consequently no operational or security record can satisfy a
    scientific predicate.
    """

    if not isinstance(matrix, FaultMatrix):
        raise TypeError("matrix must be a FaultMatrix.")
    if not isinstance(context, QualificationContext):
        raise TypeError("context must be a QualificationContext.")
    if not isinstance(resolved_run_spec, ResolvedRunSpec):
        raise TypeError("resolved_run_spec must be a ResolvedRunSpec.")
    if provider.provider_id != matrix.provider_id:
        raise ValueError("Injected provider identity does not match the fault matrix.")
    if provider.deployment_id != matrix.deployment_id:
        raise ValueError("Injected deployment identity does not match the fault matrix.")
    _provider_binding(matrix, resolved_run_spec)
    dependencies = _support_bindings(support_dependencies, resolved_run_spec)
    if matrix.support_dependency_id not in {
        dependency.dependency_id for dependency in dependencies
    }:
        raise ValueError("Fault matrix is not bound to a resolved SupportDependency.")
    if not isinstance(source_evidence, Sequence) or isinstance(
        source_evidence, (str, bytes, bytearray)
    ):
        raise TypeError("source_evidence must be a sequence.")
    supplied_evidence = tuple(source_evidence)
    if any(not isinstance(value, QualificationEvidence) for value in supplied_evidence):
        raise TypeError("source_evidence must contain QualificationEvidence values.")
    source_ids = tuple(value.evidence_id for value in supplied_evidence)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_evidence cannot contain duplicate evidence IDs.")
    scientific = _identifiers(scientific_criteria, "scientific criterion IDs")
    performance = _identifiers(performance_criteria, "performance criterion IDs")
    if not isinstance(forbidden_values, Sequence) or isinstance(
        forbidden_values, (str, bytes, bytearray)
    ):
        raise TypeError("forbidden_values must be a sequence of strings.")
    forbidden = tuple(_identifier(value, "forbidden value") for value in forbidden_values)
    if any(len(value) < 4 for value in forbidden):
        raise ValueError("Forbidden secret values must contain at least four characters.")
    static_bindings = {
        "fault_matrix": matrix.to_record(),
        "context": context.to_record(),
        "support_dependencies": [value.to_record() for value in dependencies],
        "resolved_run_spec": resolved_run_spec.to_record(),
        "source_evidence": [value.to_record() for value in supplied_evidence],
    }
    _, static_leaked = _redact(static_bindings, forbidden)
    if static_leaked:
        raise ValueError("Qualification bindings contain forbidden secret material.")

    observations = []
    generated_evidence = []
    subject_ids = (resolved_run_spec.spec_id,) + tuple(
        dependency.dependency_id for dependency in dependencies
    )
    for case in matrix.cases:
        before = _identifiers(
            provider.effect_log(), "provider effect log", allow_empty=True
        )
        observation = provider.exercise(case)
        if not isinstance(observation, FaultObservation):
            raise TypeError("Injected providers must return FaultObservation values.")
        after = _identifiers(
            provider.effect_log(), "provider effect log", allow_empty=True
        )
        record, passed, reason = _observation_record(
            matrix, case, observation, before, after, forbidden
        )
        observations.append(record)
        generated_evidence.append(
            QualificationEvidence(
                case.gate,
                "passed" if passed else "failed",
                subject_ids,
                build_id=context.build_id,
                environment_id=context.environment_id,
                backend=context.backend,
                topology=context.topology,
                precision=context.precision,
                reduction=context.reduction,
                replay_id=context.replay_id,
                criteria_ids=(case.case_id,),
                raw_artifact_ids=(str(record["observation_id"]),),
                reviewer_id=context.reviewer_id,
                issued_at=context.issued_at,
                expires_at=context.expires_at,
                reason=reason,
                requalification_triggers=(
                    f"provider:{matrix.provider_id}",
                    f"deployment:{matrix.deployment_id}",
                ),
            )
        )

    operational_evidence = tuple(
        value for value in generated_evidence if value.evidence_kind == "operational"
    )
    security_evidence = tuple(
        value for value in generated_evidence if value.evidence_kind == "security"
    )
    gates = (
        _gate_record(
            "scientific",
            scientific,
            supplied_evidence,
            context,
            resolved_run_spec,
        ),
        _gate_record(
            "performance",
            performance,
            supplied_evidence,
            context,
            resolved_run_spec,
        ),
        _gate_record(
            "operational",
            tuple(value.criteria_ids[0] for value in operational_evidence),
            operational_evidence,
            context,
            resolved_run_spec,
        ),
        _gate_record(
            "security",
            tuple(value.criteria_ids[0] for value in security_evidence),
            security_evidence,
            context,
            resolved_run_spec,
        ),
    )
    outcomes = tuple(str(gate["outcome"]) for gate in gates)
    status = (
        "failed"
        if "failed" in outcomes
        else "inconclusive"
        if "inconclusive" in outcomes
        else "passed"
    )
    core = {
        "kind": "commercial-platform-provider-qualification-candidate",
        "route": matrix.route,
        "provider_id": matrix.provider_id,
        "deployment_id": matrix.deployment_id,
        "fault_matrix": matrix.to_record(),
        "context": context.to_record(),
        "support_dependencies": [value.to_record() for value in dependencies],
        "resolved_run_spec": resolved_run_spec.to_record(),
        "source_evidence": [
            value.to_record()
            for value in sorted(supplied_evidence, key=lambda item: item.evidence_id)
        ],
        "generated_evidence": [
            value.to_record()
            for value in sorted(generated_evidence, key=lambda item: item.evidence_id)
        ],
        "observations": sorted(
            observations, key=lambda value: str(value["observation_id"])
        ),
        "gates": list(gates),
        "status": status,
        "signed": False,
        "release_ready": False,
    }
    return {**core, "artifact_id": canonical_fingerprint(core)}


def _verify_content_address(
    record: Mapping[str, object], id_name: str, label: str, /
) -> None:
    identifier = record.get(id_name)
    core = {name: value for name, value in record.items() if name != id_name}
    if type(identifier) is not str or canonical_fingerprint(core) != identifier:
        raise ValueError(f"{label} has an invalid content address.")


def verify_provider_qualification(record: Mapping[str, object], /) -> None:
    """Fail closed on tampering, release state, or inexact typed bindings."""

    if not isinstance(record, Mapping):
        raise TypeError("Provider qualification must be a mapping.")
    if record.get("kind") != "commercial-platform-provider-qualification-candidate":
        raise ValueError("Record is not a commercial platform provider candidate.")
    if record.get("signed") is not False or record.get("release_ready") is not False:
        raise ValueError("Provider qualification must remain unsigned and unreleased.")
    if "schema_version" in record or "signature" in record:
        raise ValueError(
            "Candidate qualification cannot carry a schema version or signature."
        )
    _verify_content_address(record, "artifact_id", "Provider qualification")
    route = _identifier(record.get("route"), "route")
    provider_id = _identifier(record.get("provider_id"), "provider ID")
    deployment_id = _identifier(record.get("deployment_id"), "deployment ID")
    if route not in ROUTE_BOUNDARIES:
        raise ValueError("Provider qualification has an unknown route.")

    context_record = record.get("context")
    if not isinstance(context_record, Mapping):
        raise TypeError("Provider qualification context must be a mapping.")
    context = QualificationContext(
        build_id=_identifier(context_record.get("build_id"), "build ID"),
        environment_id=_identifier(
            context_record.get("environment_id"), "environment ID"
        ),
        backend=_identifier(context_record.get("backend"), "backend"),
        topology=_identifier(context_record.get("topology"), "topology"),
        precision=_identifier(context_record.get("precision"), "precision"),
        reduction=_identifier(context_record.get("reduction"), "reduction"),
        replay_id=_identifier(context_record.get("replay_id"), "replay ID"),
        reviewer_id=_identifier(context_record.get("reviewer_id"), "reviewer ID"),
        issued_at=_timestamp(context_record.get("issued_at"), "issued_at"),
        expires_at=_timestamp(context_record.get("expires_at"), "expires_at"),
        evaluated_at=_timestamp(context_record.get("evaluated_at"), "evaluated_at"),
    )
    if context.to_record() != dict(context_record):
        raise ValueError("Provider qualification context is not exact.")

    matrix = record.get("fault_matrix")
    if not isinstance(matrix, Mapping):
        raise TypeError("Provider qualification fault_matrix must be a mapping.")
    _verify_content_address(matrix, "matrix_id", "Fault matrix")
    if (
        matrix.get("route") != route
        or matrix.get("provider_id") != provider_id
        or matrix.get("deployment_id") != deployment_id
        or matrix.get("binding_role") != _ROUTE_BINDING_ROLE[route]
    ):
        raise ValueError("Provider qualification and fault matrix identities differ.")
    cases = matrix.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes, bytearray)):
        raise TypeError("Serialized fault cases must be a sequence.")
    boundary_ids = []
    case_by_id: dict[str, Mapping[str, object]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise TypeError("Serialized fault cases must be mappings.")
        _verify_content_address(case, "case_id", "Fault case")
        boundary = _identifier(case.get("boundary_id"), "fault boundary ID")
        case_id = _identifier(case.get("case_id"), "fault case ID")
        if case.get("gate") != boundary_gate(route, boundary):
            raise ValueError("Serialized fault case has an incorrect evidence gate.")
        expected_facts = case.get("expected_facts")
        allowed_effects = case.get("allowed_effects")
        if (
            not isinstance(expected_facts, Mapping)
            or not isinstance(allowed_effects, Sequence)
            or isinstance(allowed_effects, (str, bytes, bytearray))
        ):
            raise TypeError("Serialized fault-case expectations are invalid.")
        _identifier(case.get("stimulus_id"), "fault stimulus ID")
        _json_object(expected_facts, "serialized expected facts")
        _identifiers(
            tuple(allowed_effects), "serialized allowed effect IDs", allow_empty=True
        )
        if case_id in case_by_id:
            raise ValueError("Serialized fault matrix repeats a fault case ID.")
        boundary_ids.append(boundary)
        case_by_id[case_id] = case
    expected_boundaries = required_boundaries(route)
    if set(boundary_ids) != set(expected_boundaries) or len(boundary_ids) != len(
        expected_boundaries
    ):
        raise ValueError("Serialized fault matrix is incomplete.")

    resolved_record = record.get("resolved_run_spec")
    dependency_records = record.get("support_dependencies")
    if (
        not isinstance(resolved_record, Mapping)
        or not isinstance(dependency_records, Sequence)
        or isinstance(dependency_records, (str, bytes, bytearray))
    ):
        raise TypeError("Provider qualification has invalid exact run bindings.")
    resolved = ResolvedRunSpec.from_record(resolved_record)
    dependencies = tuple(
        SupportDependency.from_record(value)
        for value in dependency_records
        if isinstance(value, Mapping)
    )
    if len(dependencies) != len(dependency_records):
        raise TypeError("Serialized support dependencies must be mappings.")
    dependencies = _support_bindings(dependencies, resolved)
    if matrix.get("support_dependency_id") not in {
        value.dependency_id for value in dependencies
    }:
        raise ValueError("Fault matrix has no exact SupportDependency binding.")
    binding_role = str(matrix["binding_role"])
    if binding_role == "repository":
        expected_provider = resolved.repository_id
    elif binding_role == "scheduler":
        expected_provider = resolved.scheduler_id
    elif binding_role == "authentication":
        expected_provider = resolved.auth_policy_id
    else:
        expected_provider = provider_id
    if provider_id != expected_provider:
        raise ValueError("Provider qualification does not match its resolved binding.")

    observations = record.get("observations")
    if not isinstance(observations, Sequence) or isinstance(
        observations, (str, bytes, bytearray)
    ):
        raise TypeError("Serialized observations must be a sequence.")
    observation_by_id: dict[str, Mapping[str, object]] = {}
    observed_case_ids = set()
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise TypeError("Serialized observations must be mappings.")
        _verify_content_address(observation, "observation_id", "Fault observation")
        observation_id = _identifier(
            observation.get("observation_id"), "fault observation ID"
        )
        case_id = _identifier(observation.get("case_id"), "fault case ID")
        if (
            observation.get("route") != route
            or observation.get("provider_id") != provider_id
            or observation.get("deployment_id") != deployment_id
        ):
            raise ValueError(
                "Fault observation has a mismatched route, provider, or deployment."
            )
        if case_id not in case_by_id:
            raise ValueError("Fault observation cites an unknown fault case.")
        case = case_by_id[case_id]
        if (
            observation.get("boundary_id") != case["boundary_id"]
            or observation.get("gate") != case["gate"]
        ):
            raise ValueError("Fault observation does not match its exact fault case.")
        facts = observation.get("facts")
        declared_effects = observation.get("declared_effects")
        observed_effects = observation.get("observed_effects")
        if not isinstance(facts, Mapping):
            raise TypeError("Serialized observation facts must be a mapping.")
        if (
            not isinstance(declared_effects, Sequence)
            or isinstance(declared_effects, (str, bytes, bytearray))
            or not isinstance(observed_effects, Sequence)
            or isinstance(observed_effects, (str, bytes, bytearray))
        ):
            raise TypeError("Serialized observation effects must be sequences.")
        facts_match = canonical_json(_json_object(facts, "observation facts")) == (
            canonical_json(case["expected_facts"])
        )
        declared = _identifiers(
            tuple(declared_effects),
            "serialized declared effect IDs",
            allow_empty=True,
        )
        observed = _identifiers(
            tuple(observed_effects),
            "serialized observed effect IDs",
            allow_empty=True,
        )
        allowed = tuple(str(value) for value in case["allowed_effects"])
        append_only = observation.get("effect_log_append_only")
        initial_effects_clean = observation.get("initial_effects_clean")
        leak_detected = observation.get("secret_leak_detected")
        if (
            type(append_only) is not bool
            or type(initial_effects_clean) is not bool
            or type(leak_detected) is not bool
        ):
            raise TypeError("Serialized observation flags must be booleans.")
        effects_match = (
            append_only and initial_effects_clean and declared == observed == allowed
        )
        if (
            observation.get("facts_matched") is not facts_match
            or observation.get("effects_matched") is not effects_match
        ):
            raise ValueError("Fault observation has inconsistent comparison results.")
        passed = facts_match and effects_match and not leak_detected
        if leak_detected:
            reason = "Injected observation exposed secret material."
        elif not facts_match:
            reason = "Injected observation did not satisfy the exact expected facts."
        elif not effects_match:
            reason = (
                "Injected provider produced undeclared or inconsistent external effects."
            )
        else:
            reason = "Injected observation satisfied the exact fault-boundary contract."
        if (
            observation.get("outcome") != ("passed" if passed else "failed")
            or observation.get("reason") != reason
        ):
            raise ValueError("Fault observation has an inconsistent outcome.")
        if observation_id in observation_by_id:
            raise ValueError("Serialized observations repeat an observation ID.")
        observation_by_id[observation_id] = observation
        observed_case_ids.add(case_id)
    if observed_case_ids != set(case_by_id) or len(observations) != len(case_by_id):
        raise ValueError("Fault observations do not cover every exact fault case.")

    source_records = record.get("source_evidence")
    generated_records = record.get("generated_evidence")
    if (
        not isinstance(source_records, Sequence)
        or not isinstance(generated_records, Sequence)
        or isinstance(source_records, (str, bytes, bytearray))
        or isinstance(generated_records, (str, bytes, bytearray))
    ):
        raise TypeError("Serialized qualification evidence must be sequences.")
    source = tuple(
        QualificationEvidence.from_record(value)
        for value in source_records
        if isinstance(value, Mapping)
    )
    generated = tuple(
        QualificationEvidence.from_record(value)
        for value in generated_records
        if isinstance(value, Mapping)
    )
    if len(source) != len(source_records) or len(generated) != len(generated_records):
        raise TypeError("Serialized qualification evidence must be mappings.")
    all_evidence_ids = tuple(value.evidence_id for value in source + generated)
    if len(set(all_evidence_ids)) != len(all_evidence_ids):
        raise ValueError("Serialized qualification evidence IDs must be unique.")
    expected_subject_ids = {
        resolved.spec_id,
        *(value.dependency_id for value in dependencies),
    }
    generated_by_case: dict[str, QualificationEvidence] = {}
    for evidence in generated:
        if evidence.evidence_kind not in ("operational", "security"):
            raise ValueError(
                "Generated provider evidence crossed an evidence-kind boundary."
            )
        if len(evidence.criteria_ids) != 1 or len(evidence.raw_artifact_ids) != 1:
            raise ValueError(
                "Generated provider evidence must bind one exact case and observation."
            )
        case_id = evidence.criteria_ids[0]
        raw_id = evidence.raw_artifact_ids[0]
        if case_id not in case_by_id or raw_id not in observation_by_id:
            raise ValueError("Generated evidence cites an unknown exact input.")
        case = case_by_id[case_id]
        observation = observation_by_id[raw_id]
        expected_fields = (
            (evidence.build_id, context.build_id),
            (evidence.environment_id, context.environment_id),
            (evidence.backend, context.backend),
            (evidence.topology, context.topology),
            (evidence.precision, context.precision),
            (evidence.reduction, context.reduction),
            (evidence.replay_id, context.replay_id),
            (evidence.reviewer_id, context.reviewer_id),
            (evidence.issued_at, context.issued_at),
            (evidence.expires_at, context.expires_at),
        )
        if any(actual != expected for actual, expected in expected_fields):
            raise ValueError("Generated evidence does not match its exact context.")
        if set(evidence.subject_ids) != expected_subject_ids:
            raise ValueError("Generated evidence does not match its exact run bindings.")
        if (
            evidence.evidence_kind != case["gate"]
            or evidence.outcome != observation["outcome"]
            or evidence.reason != observation["reason"]
        ):
            raise ValueError("Generated evidence does not match its fault observation.")
        if case_id in generated_by_case:
            raise ValueError("Generated evidence repeats a fault case.")
        generated_by_case[case_id] = evidence
    if set(generated_by_case) != set(case_by_id):
        raise ValueError("Generated evidence does not cover every exact fault case.")

    gates = record.get("gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes, bytearray)):
        raise TypeError("Serialized gates must be a sequence.")
    outcomes = []
    gate_names = []
    for gate in gates:
        if not isinstance(gate, Mapping):
            raise TypeError("Serialized gates must be mappings.")
        name = _identifier(gate.get("gate"), "gate")
        if name not in GATES:
            raise ValueError("Serialized gate has an unknown evidence kind.")
        matrix_record = gate.get("matrix")
        coverage_record = gate.get("coverage")
        if not isinstance(matrix_record, Mapping) or not isinstance(
            coverage_record, Mapping
        ):
            raise TypeError("Serialized gate matrix and coverage must be mappings.")
        gate_matrix = QualificationMatrix.from_record(matrix_record)
        predicates = tuple(dict(predicate) for _, predicate in gate_matrix.predicates)
        expected_predicate_fields = {
            "evidence_kind": name,
            "subject_id": resolved.spec_id,
            "build_id": context.build_id,
            "environment_id": context.environment_id,
            "backend": context.backend,
            "topology": context.topology,
            "precision": context.precision,
            "reduction": context.reduction,
            "replay_id": context.replay_id,
        }
        if any(
            any(
                predicate.get(key) != value
                for key, value in expected_predicate_fields.items()
            )
            for predicate in predicates
        ):
            raise ValueError(
                "Gate predicates do not preserve exact context and evidence isolation."
            )
        coverage = QualificationCoverageReport.from_record(coverage_record)
        gate_evidence = (
            source
            if name in ("scientific", "performance")
            else tuple(value for value in generated if value.evidence_kind == name)
        )
        evaluated = gate_matrix.evaluate(gate_evidence, at_time=context.evaluated_at)
        if (
            coverage.to_record() != evaluated.to_record()
            or gate.get("outcome") != evaluated.outcome
        ):
            raise ValueError("Serialized gate coverage does not match exact evidence.")
        outcomes.append(evaluated.outcome)
        gate_names.append(name)
    if tuple(gate_names) != GATES:
        raise ValueError("Provider qualification must keep all four gates separate.")
    expected_status = (
        "failed"
        if "failed" in outcomes
        else "inconclusive"
        if "inconclusive" in outcomes
        else "passed"
    )
    if record.get("status") != expected_status:
        raise ValueError("Provider qualification has an invalid aggregate status.")


def assemble_commercial_platform_candidate(
    artifacts: Sequence[Mapping[str, object]],
    /,
    *,
    required_routes: Sequence[str] | None = None,
) -> dict[str, object]:
    """Assemble provider/deployment artifacts without signing or releasing them."""

    if not isinstance(artifacts, Sequence) or isinstance(
        artifacts, (str, bytes, bytearray)
    ):
        raise TypeError("artifacts must be a sequence.")
    records = tuple(dict(value) for value in artifacts)
    if not records:
        raise ValueError("At least one provider qualification artifact is required.")
    for record in records:
        verify_provider_qualification(record)
    artifact_ids = tuple(str(record["artifact_id"]) for record in records)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("Platform candidate contains duplicate artifact IDs.")
    deployment_keys = tuple(
        (str(record["route"]), str(record["provider_id"]), str(record["deployment_id"]))
        for record in records
    )
    if len(set(deployment_keys)) != len(deployment_keys):
        raise ValueError("Platform candidate repeats a provider/deployment route.")
    required = (
        tuple(ROUTE_BOUNDARIES)
        if required_routes is None
        else _identifiers(required_routes, "required route IDs")
    )
    unknown = sorted(set(required) - set(ROUTE_BOUNDARIES))
    if unknown:
        raise ValueError("Unknown required routes: " + ", ".join(unknown) + ".")
    missing = sorted(set(required) - {key[0] for key in deployment_keys})
    if missing:
        raise ValueError(
            "Platform candidate is missing routes: " + ", ".join(missing) + "."
        )
    statuses = tuple(str(record["status"]) for record in records)
    status = (
        "failed"
        if "failed" in statuses
        else "inconclusive"
        if "inconclusive" in statuses
        else "passed"
    )
    core = {
        "kind": "commercial-platform-qualification-candidate",
        "qualification_artifacts": sorted(
            records, key=lambda value: str(value["artifact_id"])
        ),
        "required_routes": sorted(required),
        "provider_deployments": [
            {"route": route, "provider_id": provider, "deployment_id": deployment}
            for route, provider, deployment in sorted(deployment_keys)
        ],
        "status": status,
        "signed": False,
        "release_ready": False,
    }
    return {**core, "candidate_id": canonical_fingerprint(core)}


def verify_commercial_platform_candidate(record: Mapping[str, object], /) -> None:
    """Verify a complete unsigned, unreleased commercial platform candidate."""

    if not isinstance(record, Mapping):
        raise TypeError("Commercial platform candidate must be a mapping.")
    if record.get("kind") != "commercial-platform-qualification-candidate":
        raise ValueError("Record is not a commercial platform qualification candidate.")
    if record.get("signed") is not False or record.get("release_ready") is not False:
        raise ValueError(
            "Commercial platform candidate must remain unsigned and unreleased."
        )
    if "schema_version" in record or "signature" in record:
        raise ValueError(
            "Candidate qualification cannot carry a schema version or signature."
        )
    _verify_content_address(record, "candidate_id", "Commercial platform candidate")
    artifacts = record.get("qualification_artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(
        artifacts, (str, bytes, bytearray)
    ):
        raise TypeError("Candidate qualification_artifacts must be a sequence.")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise TypeError("Candidate qualification artifacts must be mappings.")
        verify_provider_qualification(artifact)
    required = record.get("required_routes")
    if not isinstance(required, Sequence) or isinstance(
        required, (str, bytes, bytearray)
    ):
        raise TypeError("Candidate required_routes must be a sequence.")
    rebuilt = assemble_commercial_platform_candidate(artifacts, required_routes=required)
    if rebuilt != dict(record):
        raise ValueError(
            "Commercial platform candidate aggregate fields are inconsistent."
        )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble or verify unsigned commercial platform qualification candidates."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("artifacts", type=Path, nargs="+")
    assemble.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("candidate", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "assemble":
        payload = assemble_commercial_platform_candidate(
            tuple(
                json.loads(path.read_text(encoding="utf-8"))
                for path in arguments.artifacts
            )
        )
        arguments.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        verify_commercial_platform_candidate(
            json.loads(arguments.candidate.read_text(encoding="utf-8"))
        )


if __name__ == "__main__":
    _main()


__all__ = [
    "COMMON_BOUNDARIES",
    "DeterministicProbeProvider",
    "FaultCase",
    "FaultMatrix",
    "FaultObservation",
    "GATES",
    "QualificationContext",
    "ROUTE_BOUNDARIES",
    "assemble_commercial_platform_candidate",
    "boundary_gate",
    "produce_provider_qualification",
    "required_boundaries",
    "standard_fault_matrix",
    "verify_commercial_platform_candidate",
    "verify_provider_qualification",
]
