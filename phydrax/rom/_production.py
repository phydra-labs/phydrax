#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._execution_plan import ExecutionRequirements, LogicalAxis
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import (
    CapabilityProfile,
    ReleaseGateEvidence,
    SupportTuple,
)


class ROMMaturity(IntEnum):
    INTERNAL = 0
    EXPERIMENTAL = 1
    CANDIDATE = 2
    PRODUCTION = 3
    CERTIFIED = 4
    DISTRIBUTED_PRODUCTION = 5


class ROMAdmissionStatus(IntEnum):
    ADMITTED = 0
    INPUT_SCHEMA = 1
    PARAMETER_SUPPORT = 2
    STATE_SUPPORT = 3
    GEOMETRY_SUPPORT = 4
    DEPENDENCY_REVISION = 5
    RESOURCE_LIMIT = 6
    PROVIDER_UNAVAILABLE = 7


class ROMResourcePolicy(StrictModule, NonTrainableState):
    """Static preparation and execution limits for one ROM deployment."""

    maximum_full_dimension: int = eqx.field(static=True)
    maximum_reduced_dimension: int = eqx.field(static=True)
    maximum_affine_terms: int = eqx.field(static=True)
    maximum_residual_atoms: int = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    maximum_elements: int = eqx.field(static=True)
    maximum_batch_size: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_archive_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_full_dimension: int = 1_000_000,
        maximum_reduced_dimension: int = 4096,
        maximum_affine_terms: int = 4096,
        maximum_residual_atoms: int = 100_000,
        maximum_samples: int = 100_000,
        maximum_elements: int = 1_000_000,
        maximum_batch_size: int = 65_536,
        maximum_workspace_bytes: int = 8 * 1024**3,
        maximum_archive_bytes: int = 16 * 1024**3,
    ):
        values = {
            "maximum_full_dimension": int(maximum_full_dimension),
            "maximum_reduced_dimension": int(maximum_reduced_dimension),
            "maximum_affine_terms": int(maximum_affine_terms),
            "maximum_residual_atoms": int(maximum_residual_atoms),
            "maximum_samples": int(maximum_samples),
            "maximum_elements": int(maximum_elements),
            "maximum_batch_size": int(maximum_batch_size),
            "maximum_workspace_bytes": int(maximum_workspace_bytes),
            "maximum_archive_bytes": int(maximum_archive_bytes),
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError("ROM resource limits must be positive.")
        for name, value in values.items():
            setattr(self, name, value)
        self.policy_id = canonical_fingerprint({"kind": "rom-resource-policy", **values})

    def admit(
        self,
        *,
        full_dimension: int,
        reduced_dimension: int,
        affine_terms: int = 1,
        residual_atoms: int = 1,
        samples: int = 1,
        elements: int = 1,
        batch_size: int = 1,
        workspace_bytes: int = 1,
        archive_bytes: int = 1,
    ) -> bool:
        return (
            0 < int(full_dimension) <= self.maximum_full_dimension
            and 0 < int(reduced_dimension) <= self.maximum_reduced_dimension
            and 0 < int(affine_terms) <= self.maximum_affine_terms
            and 0 < int(residual_atoms) <= self.maximum_residual_atoms
            and 0 < int(samples) <= self.maximum_samples
            and 0 < int(elements) <= self.maximum_elements
            and 0 < int(batch_size) <= self.maximum_batch_size
            and 0 < int(workspace_bytes) <= self.maximum_workspace_bytes
            and 0 < int(archive_bytes) <= self.maximum_archive_bytes
        )


class ROMCostEstimate(StrictModule, NonTrainableState):
    """Operation and storage estimates separated by ROM lifecycle phase."""

    truth_operations: int = eqx.field(static=True)
    preparation_operations: int = eqx.field(static=True)
    online_operations: int = eqx.field(static=True)
    reconstruction_operations: int = eqx.field(static=True)
    local_memory_bytes: int = eqx.field(static=True)
    communication_bytes: int = eqx.field(static=True)
    estimate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        truth_operations: int = 0,
        preparation_operations: int = 0,
        online_operations: int = 0,
        reconstruction_operations: int = 0,
        local_memory_bytes: int = 0,
        communication_bytes: int = 0,
    ):
        values = tuple(
            (
                truth_operations,
                preparation_operations,
                online_operations,
                reconstruction_operations,
                local_memory_bytes,
                communication_bytes,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("ROM cost estimates must be nonnegative.")
        (
            self.truth_operations,
            self.preparation_operations,
            self.online_operations,
            self.reconstruction_operations,
            self.local_memory_bytes,
            self.communication_bytes,
        ) = values
        self.estimate_id = canonical_fingerprint(
            {
                "kind": "rom-cost-estimate",
                "truth_operations": values[0],
                "preparation_operations": values[1],
                "online_operations": values[2],
                "reconstruction_operations": values[3],
                "local_memory_bytes": values[4],
                "communication_bytes": values[5],
            }
        )


class ROMAdmissionEvidence(StrictModule, NonTrainableState):
    """Fixed-shape pre-execution admission result."""

    admitted: Array
    status: Array
    score: Array
    support_id: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        admitted,
        status,
        score,
        /,
        *,
        support_id: str,
        evidence_ids: Sequence[str],
    ):
        support = str(support_id)
        evidence = tuple(str(value) for value in evidence_ids)
        if not support or any(not value for value in evidence):
            raise ValueError("Admission support and evidence IDs must be non-empty.")
        if len(set(evidence)) != len(evidence):
            raise ValueError("Admission evidence IDs must be unique.")
        self.admitted = jnp.asarray(admitted, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.score = jnp.asarray(score)
        self.support_id = support
        self.evidence_ids = evidence
        self.admission_id = canonical_fingerprint(
            {
                "kind": "rom-admission",
                "support": support,
                "evidence": list(evidence),
            }
        )


@dataclass(frozen=True, slots=True)
class ROMPromotionThresholds:
    maximum_relative_error: float = 1.0e-2
    maximum_rollout_error: float = 5.0e-2
    minimum_speedup: float = 2.0
    maximum_memory_ratio: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.maximum_relative_error,
            self.maximum_rollout_error,
            self.minimum_speedup,
            self.maximum_memory_ratio,
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("ROM promotion thresholds must be finite and positive.")


@dataclass(frozen=True, slots=True)
class ROMPromotionEvidence:
    capability: str
    support_id: str
    relative_error: float
    rollout_error: float
    speedup: float
    memory_ratio: float
    support_refusal_passed: bool
    exact_resume_passed: bool
    evidence_ids: tuple[str, ...]
    passed: bool
    evidence_id: str

    def __init__(
        self,
        capability: str,
        support_id: str,
        /,
        *,
        relative_error: float,
        rollout_error: float,
        speedup: float,
        memory_ratio: float,
        support_refusal_passed: bool,
        exact_resume_passed: bool,
        evidence_ids: Sequence[str],
        thresholds: ROMPromotionThresholds | None = None,
    ):
        capability_ = str(capability)
        support_ = str(support_id)
        identifiers = tuple(sorted(str(value) for value in evidence_ids))
        metrics = tuple(
            float(value)
            for value in (relative_error, rollout_error, speedup, memory_ratio)
        )
        if (
            not capability_
            or not support_
            or not identifiers
            or any(not value for value in identifiers)
        ):
            raise ValueError("ROM promotion identities must be complete.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ROM promotion evidence IDs must be unique.")
        if any(not isfinite(value) or value < 0.0 for value in metrics):
            raise ValueError("ROM promotion metrics must be finite and non-negative.")
        limits = ROMPromotionThresholds() if thresholds is None else thresholds
        if not isinstance(limits, ROMPromotionThresholds):
            raise TypeError("thresholds must be ROMPromotionThresholds.")
        passed = (
            metrics[0] <= limits.maximum_relative_error
            and metrics[1] <= limits.maximum_rollout_error
            and metrics[2] >= limits.minimum_speedup
            and metrics[3] <= limits.maximum_memory_ratio
            and bool(support_refusal_passed)
            and bool(exact_resume_passed)
        )
        payload = {
            "kind": "rom-promotion-evidence",
            "capability": capability_,
            "support": support_,
            "relative_error": metrics[0],
            "rollout_error": metrics[1],
            "speedup": metrics[2],
            "memory_ratio": metrics[3],
            "support_refusal_passed": bool(support_refusal_passed),
            "exact_resume_passed": bool(exact_resume_passed),
            "evidence_ids": list(identifiers),
            "thresholds": {
                "maximum_relative_error": limits.maximum_relative_error,
                "maximum_rollout_error": limits.maximum_rollout_error,
                "minimum_speedup": limits.minimum_speedup,
                "maximum_memory_ratio": limits.maximum_memory_ratio,
            },
        }
        object.__setattr__(self, "capability", capability_)
        object.__setattr__(self, "support_id", support_)
        object.__setattr__(self, "relative_error", metrics[0])
        object.__setattr__(self, "rollout_error", metrics[1])
        object.__setattr__(self, "speedup", metrics[2])
        object.__setattr__(self, "memory_ratio", metrics[3])
        object.__setattr__(self, "support_refusal_passed", bool(support_refusal_passed))
        object.__setattr__(self, "exact_resume_passed", bool(exact_resume_passed))
        object.__setattr__(self, "evidence_ids", identifiers)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "evidence_id", canonical_fingerprint(payload))


class ROMCapabilityDeclaration(StrictModule, NonTrainableState):
    """Exact maturity and release-gate declaration for one ROM capability."""

    capability: str = eqx.field(static=True)
    maturity: ROMMaturity = eqx.field(static=True)
    attributes: tuple[tuple[str, str | int | bool], ...] = eqx.field(static=True)
    required_gates: tuple[str, ...] = eqx.field(static=True)
    declaration_id: str = eqx.field(static=True)

    def __init__(
        self,
        capability: str,
        maturity: ROMMaturity,
        attributes: Mapping[str, str | int | bool],
        /,
        *,
        required_gates: Sequence[str],
    ):
        capability_ = str(capability)
        gates = tuple(str(value) for value in required_gates)
        values = tuple(sorted((str(name), value) for name, value in attributes.items()))
        if not capability_ or not values or any(not name for name, _ in values):
            raise ValueError("ROM capability and attributes must be non-empty.")
        if not isinstance(maturity, ROMMaturity):
            raise TypeError("maturity must be a ROMMaturity value.")
        if any(not gate for gate in gates) or len(set(gates)) != len(gates):
            raise ValueError("required_gates must be unique non-empty identifiers.")
        self.capability = capability_
        self.maturity = maturity
        self.attributes = values
        self.required_gates = gates
        self.declaration_id = canonical_fingerprint(
            {
                "kind": "rom-capability-declaration",
                "capability": capability_,
                "maturity": int(maturity),
                "attributes": dict(values),
                "required_gates": list(gates),
            }
        )

    def profile(
        self,
        *,
        provider: str,
        version: str,
        release_evidence: Sequence[ReleaseGateEvidence] = (),
    ) -> CapabilityProfile:
        evidence = tuple(release_evidence)
        released = self.maturity >= ROMMaturity.PRODUCTION
        return CapabilityProfile(
            self.capability,
            provider,
            version,
            (SupportTuple(self.capability, dict(self.attributes)),),
            required_gates=self.required_gates,
            release_evidence=evidence,
            released=released,
        )


def rom_capability_catalog() -> tuple[ROMCapabilityDeclaration, ...]:
    """Return the implemented ROM capability surface with honest maturity."""
    specifications = (
        (
            "rom.affine-steady",
            ROMMaturity.CANDIDATE,
            {"geometry": "fixed", "execution": "reduced-only", "certified": False},
            ("algebra", "scientific", "performance", "archive"),
        ),
        (
            "rom.affine-transient",
            ROMMaturity.EXPERIMENTAL,
            {"geometry": "fixed", "descriptor": "differential"},
            ("algebra", "transient-scientific", "performance"),
        ),
        (
            "rom.hyperreduction",
            ROMMaturity.EXPERIMENTAL,
            {"methods": "deim-gnat-ecsw", "selected-only": True},
            ("algebra", "selected-execution", "nonlinear-rollout"),
        ),
        (
            "rom.identified-dynamics",
            ROMMaturity.CANDIDATE,
            {"methods": "dmd-edmd-sindy-opinf", "rollout-selected": True},
            ("partition", "rollout", "ood", "archive"),
        ),
        (
            "rom.geometry-atlas",
            ROMMaturity.INTERNAL,
            {"geometry": "reference-transfer", "local-atlas": True},
            ("transfer", "atlas", "moving-geometry"),
        ),
        (
            "rom.nonlinear-chart",
            ROMMaturity.INTERNAL,
            {"quadratic": True, "coordinate-conditioned": True},
            ("chart", "immersion", "rollout"),
        ),
        (
            "rom.sensor-estimation",
            ROMMaturity.INTERNAL,
            {"history": True, "assimilation": True},
            ("observability", "uncertainty", "ood"),
        ),
        (
            "rom.spectral-submanifold",
            ROMMaturity.INTERNAL,
            {"local": True, "normal-form": True},
            ("spectral", "invariance", "rollout"),
        ),
        (
            "rom.control-reduction",
            ROMMaturity.EXPERIMENTAL,
            {"balanced": True, "interpolatory": True},
            ("stability", "transfer-response", "error-bound"),
        ),
        (
            "rom.structure-preserving",
            ROMMaturity.INTERNAL,
            {"symplectic": True, "port-hamiltonian": True},
            ("structure", "long-time", "invariants"),
        ),
    )
    return tuple(
        ROMCapabilityDeclaration(
            capability,
            maturity,
            attributes,
            required_gates=gates,
        )
        for capability, maturity, attributes, gates in specifications
    )


def rom_execution_requirements(
    owner_id: str,
    /,
    *,
    batch_axis: int,
    reduced_axis: int,
    dtype: str,
    distributed_output: bool,
) -> ExecutionRequirements:
    return ExecutionRequirements(
        owner_id,
        logical_axes=(
            LogicalAxis("rom-batch", int(batch_axis), splittable=True),
            LogicalAxis("rom-reduced", int(reduced_axis), splittable=False),
        ),
        operations=("rom-admission", "rom-assembly", "rom-solve"),
        dtypes=(str(dtype),),
        transformations=("jit", "vmap"),
        allows_distributed_output=bool(distributed_output),
    )


__all__ = [
    "ROMAdmissionEvidence",
    "ROMAdmissionStatus",
    "ROMCapabilityDeclaration",
    "ROMCostEstimate",
    "ROMPromotionEvidence",
    "ROMPromotionThresholds",
    "ROMMaturity",
    "ROMResourcePolicy",
    "rom_capability_catalog",
    "rom_execution_requirements",
]
