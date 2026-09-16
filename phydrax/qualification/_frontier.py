#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical scientific-closure, resource, and external qualification contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .._fingerprint import canonical_fingerprint
from ..lifecycle import ArrayArtifactProvenance
from ._registry import ReleaseGateEvidence, SupportTuple


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]*$")
_HEX = frozenset("0123456789abcdef")


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result or _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{name} must be a canonical identifier.")
    return result


def _identifiers(
    values: Sequence[str],
    name: str,
    /,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    result = tuple(sorted(_identifier(value, name) for value in values))
    if (not allow_empty and not result) or len(set(result)) != len(result):
        raise ValueError(f"{name} values must be distinct and non-empty.")
    return result


class FrontierGate(StrEnum):
    """Scientific closure gates shared by all frontier capability profiles."""

    CONVENTION = "convention"
    STRUCTURAL = "structural"
    NUMERICAL = "numerical"
    INDEPENDENT_REFERENCE = "independent-reference"
    LIFECYCLE = "lifecycle"
    SCIENTIFIC_SYSTEMATICS = "scientific-systematics"
    RELEASE = "release"


class FrontierDisposition(StrEnum):
    """Claim state; abstention and permanent nonclaims are first-class results."""

    CANDIDATE = "candidate"
    RELEASED = "released"
    ABSTAINED = "abstained"
    PERMANENT_NONCLAIM = "permanent-nonclaim"


@dataclass(frozen=True, slots=True)
class FrontierClosureObligation:
    """One source-bound capability obligation and its exact evidence gates."""

    name: str
    support_tuple: SupportTuple
    source_ids: tuple[str, ...]
    required_gates: tuple[FrontierGate, ...]
    nonclaims: tuple[str, ...]
    obligation_id: str

    def __init__(
        self,
        name: str,
        support_tuple: SupportTuple,
        source_ids: Sequence[str],
        required_gates: Sequence[FrontierGate],
        /,
        *,
        nonclaims: Sequence[str] = (),
    ):
        if not isinstance(support_tuple, SupportTuple):
            raise TypeError("support_tuple must be SupportTuple.")
        name_ = _identifier(name, "obligation name")
        sources = _identifiers(source_ids, "source ID")
        gates = tuple(sorted(set(required_gates), key=lambda item: item.value))
        if not gates or any(not isinstance(item, FrontierGate) for item in gates):
            raise TypeError("required_gates must contain FrontierGate values.")
        nonclaims_ = _identifiers(
            nonclaims,
            "nonclaim",
            allow_empty=True,
        )
        content = {
            "kind": "frontier-closure-obligation",
            "name": name_,
            "support_tuple_id": support_tuple.support_tuple_id,
            "source_ids": list(sources),
            "required_gates": [item.value for item in gates],
            "nonclaims": list(nonclaims_),
        }
        object.__setattr__(self, "name", name_)
        object.__setattr__(self, "support_tuple", support_tuple)
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "required_gates", gates)
        object.__setattr__(self, "nonclaims", nonclaims_)
        object.__setattr__(self, "obligation_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "frontier-closure-obligation",
            "name": self.name,
            "support_tuple": self.support_tuple.to_record(),
            "source_ids": list(self.source_ids),
            "required_gates": [item.value for item in self.required_gates],
            "nonclaims": list(self.nonclaims),
            "obligation_id": self.obligation_id,
        }


@dataclass(frozen=True, slots=True)
class FrontierClaimAssessment:
    """Complete gate assessment for one closure obligation."""

    obligation_id: str
    disposition: FrontierDisposition
    accepted_gates: tuple[str, ...]
    missing_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    expired_gates: tuple[str, ...]
    assessment_id: str

    @property
    def eligible(self) -> bool:
        return self.disposition is FrontierDisposition.RELEASED


def assess_frontier_claim(
    obligation: FrontierClosureObligation,
    evidence: Sequence[ReleaseGateEvidence],
    at_time: int,
    /,
    *,
    permanent_nonclaim: bool = False,
) -> FrontierClaimAssessment:
    """Assess every required gate; duplicate evidence is rejected."""

    if not isinstance(obligation, FrontierClosureObligation):
        raise TypeError("obligation must be FrontierClosureObligation.")
    records = tuple(evidence)
    if any(not isinstance(item, ReleaseGateEvidence) for item in records):
        raise TypeError("evidence must contain ReleaseGateEvidence values.")
    by_gate = {item.gate: item for item in records}
    if len(by_gate) != len(records):
        raise ValueError("Gate evidence must contain at most one record per gate.")
    required = tuple(item.value for item in obligation.required_gates)
    accepted: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    expired: list[str] = []
    for gate in required:
        record = by_gate.get(gate)
        if record is None:
            missing.append(gate)
        elif not record.is_current(at_time):
            expired.append(gate)
        elif not record.accepted:
            failed.append(gate)
        else:
            accepted.append(gate)
    if permanent_nonclaim:
        disposition = FrontierDisposition.PERMANENT_NONCLAIM
    elif missing or failed or expired:
        disposition = FrontierDisposition.ABSTAINED
    else:
        disposition = FrontierDisposition.RELEASED
    content = {
        "kind": "frontier-claim-assessment",
        "obligation_id": obligation.obligation_id,
        "disposition": disposition.value,
        "accepted_gates": accepted,
        "missing_gates": missing,
        "failed_gates": failed,
        "expired_gates": expired,
        "at_time": int(at_time),
    }
    return FrontierClaimAssessment(
        obligation.obligation_id,
        disposition,
        tuple(accepted),
        tuple(missing),
        tuple(failed),
        tuple(expired),
        canonical_fingerprint(content),
    )


@dataclass(frozen=True, slots=True)
class DistributedResourceProfile:
    """Exact device-mesh and bounded-memory support coordinates."""

    backend: str
    device_count: int
    mesh_shape: tuple[int, ...]
    mesh_axes: tuple[str, ...]
    dtype: str
    maximum_local_bytes: int
    maximum_global_bytes: int
    communication_route: str
    profile_id: str

    def __init__(
        self,
        backend: str,
        device_count: int,
        mesh_shape: Sequence[int],
        mesh_axes: Sequence[str],
        dtype: str,
        maximum_local_bytes: int,
        maximum_global_bytes: int,
        communication_route: str,
        /,
    ):
        backend_ = _identifier(backend, "backend")
        dtype_ = _identifier(dtype, "dtype")
        route = _identifier(communication_route, "communication route")
        count = int(device_count)
        shape = tuple(int(value) for value in mesh_shape)
        axes = tuple(_identifier(value, "mesh axis") for value in mesh_axes)
        local = int(maximum_local_bytes)
        global_ = int(maximum_global_bytes)
        if count <= 0 or not shape or any(value <= 0 for value in shape):
            raise ValueError("Device count and mesh extents must be positive.")
        product = 1
        for value in shape:
            product *= value
        if product != count or len(shape) != len(axes) or len(set(axes)) != len(axes):
            raise ValueError("Mesh shape and axes must uniquely cover every device.")
        if local <= 0 or global_ < local or global_ < count * local:
            raise ValueError("Global capacity must cover every positive local capacity.")
        content = {
            "kind": "distributed-resource-profile",
            "backend": backend_,
            "device_count": count,
            "mesh_shape": list(shape),
            "mesh_axes": list(axes),
            "dtype": dtype_,
            "maximum_local_bytes": local,
            "maximum_global_bytes": global_,
            "communication_route": route,
        }
        object.__setattr__(self, "backend", backend_)
        object.__setattr__(self, "device_count", count)
        object.__setattr__(self, "mesh_shape", shape)
        object.__setattr__(self, "mesh_axes", axes)
        object.__setattr__(self, "dtype", dtype_)
        object.__setattr__(self, "maximum_local_bytes", local)
        object.__setattr__(self, "maximum_global_bytes", global_)
        object.__setattr__(self, "communication_route", route)
        object.__setattr__(self, "profile_id", canonical_fingerprint(content))


@dataclass(frozen=True, slots=True)
class ExternalQualificationBoundary:
    """Pinned, bounded process identity required for live qualification."""

    provider: str
    release: str
    executable_sha256: str
    argument_roster: tuple[str, ...]
    required_environment: tuple[str, ...]
    maximum_output_bytes: int
    timeout_seconds: int
    boundary_id: str

    def __init__(
        self,
        provider: str,
        release: str,
        executable_sha256: str,
        argument_roster: Sequence[str],
        required_environment: Sequence[str],
        maximum_output_bytes: int,
        timeout_seconds: int,
        /,
    ):
        provider_ = _identifier(provider, "provider")
        release_ = _identifier(release, "release")
        digest = str(executable_sha256).lower()
        if len(digest) != 64 or any(value not in _HEX for value in digest):
            raise ValueError("executable_sha256 must be a lowercase SHA-256 digest.")
        arguments = tuple(str(value) for value in argument_roster)
        if any("\x00" in value for value in arguments):
            raise ValueError("Process arguments cannot contain NUL characters.")
        environment = _identifiers(
            required_environment,
            "environment variable",
            allow_empty=True,
        )
        output = int(maximum_output_bytes)
        timeout = int(timeout_seconds)
        if output <= 0 or timeout <= 0:
            raise ValueError("Process output and timeout bounds must be positive.")
        content = {
            "kind": "external-qualification-boundary",
            "provider": provider_,
            "release": release_,
            "executable_sha256": digest,
            "argument_roster": list(arguments),
            "required_environment": list(environment),
            "maximum_output_bytes": output,
            "timeout_seconds": timeout,
        }
        object.__setattr__(self, "provider", provider_)
        object.__setattr__(self, "release", release_)
        object.__setattr__(self, "executable_sha256", digest)
        object.__setattr__(self, "argument_roster", arguments)
        object.__setattr__(self, "required_environment", environment)
        object.__setattr__(self, "maximum_output_bytes", output)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "boundary_id", canonical_fingerprint(content))


@dataclass(frozen=True, slots=True)
class FrontierArtifactBinding:
    """Shared provenance coordinates for typed frontier array artifacts."""

    producer_id: str
    support_tuple_id: str
    plan_id: str
    numeric_revision_id: str
    source_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    parent_artifact_ids: tuple[str, ...]
    binding_id: str

    def __init__(
        self,
        producer_id: str,
        support_tuple_id: str,
        plan_id: str,
        numeric_revision_id: str,
        source_ids: Sequence[str],
        profile_ids: Sequence[str],
        unit_ids: Sequence[str],
        evidence_ids: Sequence[str],
        parent_artifact_ids: Sequence[str] = (),
    ):
        producer = _identifier(producer_id, "producer ID")
        support = _identifier(support_tuple_id, "support tuple ID")
        plan = _identifier(plan_id, "plan ID")
        revision = _identifier(numeric_revision_id, "numeric revision ID")
        sources = _identifiers(source_ids, "source ID")
        profiles = _identifiers(profile_ids, "profile ID")
        units = _identifiers(unit_ids, "unit ID")
        evidence = _identifiers(evidence_ids, "evidence ID")
        parents = _identifiers(
            parent_artifact_ids,
            "parent artifact ID",
            allow_empty=True,
        )
        content = {
            "kind": "frontier-artifact-binding",
            "producer_id": producer,
            "support_tuple_id": support,
            "plan_id": plan,
            "numeric_revision_id": revision,
            "source_ids": list(sources),
            "profile_ids": list(profiles),
            "unit_ids": list(units),
            "evidence_ids": list(evidence),
            "parent_artifact_ids": list(parents),
        }
        object.__setattr__(self, "producer_id", producer)
        object.__setattr__(self, "support_tuple_id", support)
        object.__setattr__(self, "plan_id", plan)
        object.__setattr__(self, "numeric_revision_id", revision)
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "profile_ids", profiles)
        object.__setattr__(self, "unit_ids", units)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "parent_artifact_ids", parents)
        object.__setattr__(self, "binding_id", canonical_fingerprint(content))

    def provenance(self) -> ArrayArtifactProvenance:
        return ArrayArtifactProvenance(
            self.producer_id,
            self.source_ids,
            self.profile_ids,
            self.unit_ids,
        )

    def structure_ids(self) -> Mapping[str, str]:
        return {
            "support_tuple": self.support_tuple_id,
            "plan": self.plan_id,
            "numeric_revision": self.numeric_revision_id,
            "binding": self.binding_id,
        }


def frontier_closure_obligations() -> tuple[FrontierClosureObligation, ...]:
    """Return the canonical closure matrix for the computational frontiers."""

    gates = tuple(FrontierGate)
    rows = (
        (
            "quantum-symmetry",
            "quantum.finite-group-irrep",
            {"representation": "matrix-irrep", "route": "matrix-free"},
            ("source-ledger:finite-group-representation",),
            ("no-thermodynamic-phase-inference",),
        ),
        (
            "conformal-bootstrap",
            "conformal.certified-bootstrap",
            {"blocks": "global-and-virasoro", "certificate": "interval-sos"},
            ("source-ledger:conformal-bootstrap",),
            ("no-universal-cft-existence-claim",),
        ),
        (
            "supersymmetric-lattice",
            "lattice.supersymmetric-rhmc",
            {"theory": "twisted-n2-and-bfss", "measure": "phase-reweighted"},
            ("source-ledger:supersymmetric-lattice",),
            ("no-generic-sign-problem-solution",),
        ),
        (
            "calabi-yau",
            "geometry.calabi-yau-harmonic",
            {"geometry": "projective-and-toric", "evidence": "hodge-and-topology"},
            ("source-ledger:calabi-yau",),
            ("no-numerical-proof-of-yau-theorem",),
        ),
        (
            "conformal-ads",
            "relativity.conformal-ads",
            {"evolution": "nonlinear-cfe", "boundary": "timelike-conformal"},
            ("source-ledger:conformal-einstein-ads",),
            ("no-quantum-gravity-validation",),
        ),
        (
            "fuzzy-space",
            "quantum.fuzzy-many-body",
            {"geometry": "sphere", "solver": "exact-and-mps"},
            ("source-ledger:fuzzy-space",),
            ("no-automatic-cft-identification",),
        ),
        (
            "particle-spectrum",
            "particle.native-spectrum",
            {"models": "sm-and-mssm", "interchange": "slha2"},
            ("source-ledger:particle-spectrum",),
            ("no-collider-exclusion-claim",),
        ),
        (
            "soliton-optics",
            "optics.multimode-envelope",
            {"defects": "stationary-and-dynamic", "envelope": "vector-multimode"},
            ("source-ledger:soliton-optics",),
            ("no-full-maxwell-equivalence",),
        ),
        (
            "spin-quantum-geometry",
            "quantum.spin-foam-native",
            {"group": "sl2c", "amplitude": "finite-eprl-complex"},
            ("source-ledger:spin-quantum-geometry",),
            ("no-continuum-quantum-gravity-claim",),
        ),
    )
    return tuple(
        FrontierClosureObligation(
            name,
            SupportTuple(capability, attributes),
            sources,
            gates,
            nonclaims=nonclaims,
        )
        for name, capability, attributes, sources, nonclaims in rows
    )


__all__ = [
    "DistributedResourceProfile",
    "ExternalQualificationBoundary",
    "FrontierArtifactBinding",
    "FrontierClaimAssessment",
    "FrontierClosureObligation",
    "FrontierDisposition",
    "FrontierGate",
    "assess_frontier_claim",
    "frontier_closure_obligations",
]
