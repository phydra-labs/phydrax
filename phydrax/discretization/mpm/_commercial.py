#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification import ReleaseGateEvidence, SupportTuple


class MPMClaimOutcome(IntEnum):
    SUPPORTED = 0
    REJECTED = 1
    NOT_APPLICABLE = 2
    EXPERIMENTAL = 3


class MPMDerivativeKind(IntEnum):
    SMOOTH_DISCRETE = 0
    BRANCHWISE = 1
    EVENT_AWARE = 2
    GENERALIZED_SET = 3
    SURROGATE = 4
    STOCHASTIC_ESTIMATOR = 5
    NONDIFFERENTIABLE = 6


class MPMReleaseGate(IntEnum):
    INTENDED_USE = 0
    CODE_VERIFICATION = 1
    SOLUTION_VERIFICATION = 2
    VALIDATION_UQ = 3
    DERIVATIVE = 4
    PROVENANCE_SBOM = 5
    QUALITY_RELIABILITY = 6
    RELEASE_DECISION = 7


class MPMOperationalStatus(IntEnum):
    PREPARED = 0
    RUNNING = 1
    CHECKPOINTING = 2
    RECOVERING = 3
    COMPLETED = 4
    FAILED = 5
    QUARANTINED = 6
    RELEASED = 7


class MPMCommercialFailure(IntEnum):
    NONE = 0
    CONFIGURATION_UNSUPPORTED = 1
    PREPARATION_FAILED = 2
    CAPACITY_EXCEEDED = 3
    ROUTE_TOPOLOGY_CHANGED = 4
    MATERIAL_ADMISSIBILITY_FAILED = 5
    LOCAL_ROOT_FAILED = 6
    CONTACT_INFEASIBLE = 7
    CONTACT_NONUNIQUE = 8
    NONLINEAR_NONCONVERGENCE = 9
    LINEAR_OR_ADJOINT_FAILED = 10
    FRACTURE_TOPOLOGY_FAILED = 11
    AMR_TRANSFER_FAILED = 12
    DISTRIBUTED_TRANSACTION_FAILED = 13
    CHECKPOINT_INTEGRITY_FAILED = 14
    OUTPUT_INCOMPLETE = 15
    DERIVATIVE_INVALID = 16
    VALIDATION_OUT_OF_DOMAIN = 17


class MPMIntendedUse(StrictModule, NonTrainableState):
    decision: str = eqx.field(static=True)
    phenomena: tuple[str, ...] = eqx.field(static=True)
    target_observables: tuple[str, ...] = eqx.field(static=True)
    prohibited_uses: tuple[str, ...] = eqx.field(static=True)
    risk_class: str = eqx.field(static=True)
    geometry_loading_scope: str = eqx.field(static=True)
    material_parameter_scope: str = eqx.field(static=True)
    accuracy_uq_goal: str = eqx.field(static=True)
    intended_use_id: str = eqx.field(static=True)

    def __init__(
        self,
        decision: str,
        /,
        *,
        phenomena: Sequence[str],
        target_observables: Sequence[str],
        prohibited_uses: Sequence[str] = (),
        risk_class: str,
        geometry_loading_scope: str,
        material_parameter_scope: str,
        accuracy_uq_goal: str,
    ):
        values = (
            str(decision),
            str(risk_class),
            str(geometry_loading_scope),
            str(material_parameter_scope),
            str(accuracy_uq_goal),
        )
        phenomena_ = tuple(str(value) for value in phenomena)
        observables = tuple(str(value) for value in target_observables)
        prohibited = tuple(str(value) for value in prohibited_uses)
        if any(not value for value in values) or not phenomena_ or not observables:
            raise ValueError("Commercial MPM intended-use fields must be non-empty.")
        self.decision = values[0]
        self.phenomena = phenomena_
        self.target_observables = observables
        self.prohibited_uses = prohibited
        self.risk_class = values[1]
        self.geometry_loading_scope = values[2]
        self.material_parameter_scope = values[3]
        self.accuracy_uq_goal = values[4]
        self.intended_use_id = canonical_fingerprint(
            {
                "kind": "mpm-intended-use",
                "decision": self.decision,
                "phenomena": phenomena_,
                "target_observables": observables,
                "prohibited_uses": prohibited,
                "risk_class": self.risk_class,
                "geometry_loading_scope": self.geometry_loading_scope,
                "material_parameter_scope": self.material_parameter_scope,
                "accuracy_uq_goal": self.accuracy_uq_goal,
            }
        )


class MPMClaimTuple(StrictModule, NonTrainableState):
    """MPM compatibility view over the canonical provider-neutral support tuple."""

    support_tuple: SupportTuple

    def __init__(
        self,
        *,
        equation_family: str,
        dimension: int,
        kinematics: str,
        grid_assignment: str,
        source_domain: str,
        transfer: str,
        schedule: str,
        material: str,
        field_contact: str,
        fracture: str,
        integrator: str,
        storage_backend: str,
        precision_accumulation: str,
        capacity_envelope: str,
        derivative_mode: str,
    ):
        dimension_ = int(dimension)
        values = {
            "equation_family": str(equation_family),
            "dimension": dimension_,
            "kinematics": str(kinematics),
            "grid_assignment": str(grid_assignment),
            "source_domain": str(source_domain),
            "transfer": str(transfer),
            "schedule": str(schedule),
            "material": str(material),
            "field_contact": str(field_contact),
            "fracture": str(fracture),
            "integrator": str(integrator),
            "storage_backend": str(storage_backend),
            "precision_accumulation": str(precision_accumulation),
            "capacity_envelope": str(capacity_envelope),
            "derivative_mode": str(derivative_mode),
        }
        if dimension_ not in (1, 2, 3) or any(
            not value for key, value in values.items() if key != "dimension"
        ):
            raise ValueError("MPM claim tuple is incomplete or dimension is invalid.")
        self.support_tuple = SupportTuple("material-point-method", values)

    def _coordinate(self, name: str, /) -> str | int | bool:
        for coordinate, value in self.support_tuple.attributes:
            if coordinate == name:
                return value
        raise KeyError(f"No MPM support coordinate {name!r}.")

    @property
    def equation_family(self) -> str:
        return str(self._coordinate("equation_family"))

    @property
    def dimension(self) -> int:
        return int(self._coordinate("dimension"))

    @property
    def kinematics(self) -> str:
        return str(self._coordinate("kinematics"))

    @property
    def grid_assignment(self) -> str:
        return str(self._coordinate("grid_assignment"))

    @property
    def source_domain(self) -> str:
        return str(self._coordinate("source_domain"))

    @property
    def transfer(self) -> str:
        return str(self._coordinate("transfer"))

    @property
    def schedule(self) -> str:
        return str(self._coordinate("schedule"))

    @property
    def material(self) -> str:
        return str(self._coordinate("material"))

    @property
    def field_contact(self) -> str:
        return str(self._coordinate("field_contact"))

    @property
    def fracture(self) -> str:
        return str(self._coordinate("fracture"))

    @property
    def integrator(self) -> str:
        return str(self._coordinate("integrator"))

    @property
    def storage_backend(self) -> str:
        return str(self._coordinate("storage_backend"))

    @property
    def precision_accumulation(self) -> str:
        return str(self._coordinate("precision_accumulation"))

    @property
    def capacity_envelope(self) -> str:
        return str(self._coordinate("capacity_envelope"))

    @property
    def derivative_mode(self) -> str:
        return str(self._coordinate("derivative_mode"))

    @property
    def claim_id(self) -> str:
        return self.support_tuple.support_tuple_id


class MPMSupportDecision(StrictModule, NonTrainableState):
    claim: MPMClaimTuple
    outcome: MPMClaimOutcome = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    required_profile: str = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)

    def __init__(
        self,
        claim: MPMClaimTuple,
        outcome: MPMClaimOutcome,
        /,
        *,
        reason: str,
        required_profile: str,
    ):
        if not isinstance(claim, MPMClaimTuple):
            raise TypeError("claim must be MPMClaimTuple.")
        outcome_ = MPMClaimOutcome(outcome)
        reason_ = str(reason)
        profile = str(required_profile)
        if not reason_ or not profile:
            raise ValueError("Support decisions require reason and profile.")
        self.claim = claim
        self.outcome = outcome_
        self.reason = reason_
        self.required_profile = profile
        self.decision_id = canonical_fingerprint(
            {
                "kind": "mpm-support-decision",
                "claim": claim.claim_id,
                "outcome": int(outcome_),
                "reason": reason_,
                "required_profile": profile,
            }
        )

    def require_supported(self) -> str:
        if self.outcome != MPMClaimOutcome.SUPPORTED:
            raise ValueError(
                f"MPM claim {self.claim.claim_id} is {self.outcome.name}: {self.reason}"
            )
        return self.decision_id


class MPMSupportMatrix(StrictModule, NonTrainableState):
    decisions: tuple[MPMSupportDecision, ...]
    matrix_id: str = eqx.field(static=True)

    def __init__(self, decisions: Sequence[MPMSupportDecision], /):
        decisions_ = tuple(decisions)
        if not decisions_ or any(
            not isinstance(value, MPMSupportDecision) for value in decisions_
        ):
            raise TypeError("Support matrix requires typed non-empty decisions.")
        ids = tuple(value.claim.claim_id for value in decisions_)
        if len(set(ids)) != len(ids):
            raise ValueError("Support matrix contains duplicate claim IDs.")
        self.decisions = decisions_
        self.matrix_id = canonical_fingerprint(
            {
                "kind": "mpm-support-matrix",
                "decisions": [d.decision_id for d in decisions_],
            }
        )

    def decision(self, claim_id: str, /) -> MPMSupportDecision:
        identifier = str(claim_id)
        for value in self.decisions:
            if value.claim.claim_id == identifier:
                return value
        raise KeyError(f"No MPM support decision for {identifier}.")


class MPMEventJournal(StrictModule):
    attempted: Array
    event_times: Array
    event_codes: Array
    left_state_digests: Array
    right_state_digests: Array
    transversality_margins: Array
    localized: Array
    capacity: int = eqx.field(static=True)
    journal_id: str = eqx.field(static=True)


class MPMTopologyJournal(StrictModule):
    attempted: Array
    generations: Array
    route_digests: Array
    block_digests: Array
    field_digests: Array
    fracture_digests: Array
    accepted: Array
    capacity: int = eqx.field(static=True)
    journal_id: str = eqx.field(static=True)


class MPMDerivativeEvidence(StrictModule):
    kind: Array
    valid: Array
    branch_margin: Array
    event_time: Array
    transversality_margin: Array
    primal_residual: Array
    transpose_residual: Array
    sample_count: Array
    estimator_variance: Array
    reason_code: Array
    journal_digest: Array
    evidence_id: str = eqx.field(static=True)


class MPMReleaseGateEvidence(StrictModule, NonTrainableState):
    """MPM gate view over canonical, time-bounded release evidence."""

    release_evidence: ReleaseGateEvidence

    def __init__(
        self,
        gate: MPMReleaseGate,
        /,
        *,
        passed: bool,
        evidence_ids: Sequence[str],
        reviewer_id: str,
        deviation_ids: Sequence[str] = (),
        issued_at: int = 0,
        expires_at: int = 2**63 - 1,
    ):
        gate_ = MPMReleaseGate(gate)
        self.release_evidence = ReleaseGateEvidence(
            gate_.name,
            passed=passed,
            evidence_ids=evidence_ids,
            reviewer_id=reviewer_id,
            deviation_ids=deviation_ids,
            issued_at=issued_at,
            expires_at=expires_at,
        )

    @property
    def gate(self) -> MPMReleaseGate:
        return MPMReleaseGate[self.release_evidence.gate]

    @property
    def passed(self) -> bool:
        return self.release_evidence.passed

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.release_evidence.evidence_ids

    @property
    def reviewer_id(self) -> str:
        return self.release_evidence.reviewer_id

    @property
    def deviation_ids(self) -> tuple[str, ...]:
        return self.release_evidence.deviation_ids

    @property
    def gate_id(self) -> str:
        return self.release_evidence.evidence_id


class MPMReleaseEvidenceBundle(StrictModule, NonTrainableState):
    claim: MPMClaimTuple
    intended_use: MPMIntendedUse
    gates: tuple[MPMReleaseGateEvidence, ...]
    independent_approver_id: str = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        claim: MPMClaimTuple,
        intended_use: MPMIntendedUse,
        gates: Sequence[MPMReleaseGateEvidence],
        /,
        *,
        independent_approver_id: str,
    ):
        gates_ = tuple(gates)
        approver = str(independent_approver_id)
        if not isinstance(claim, MPMClaimTuple) or not isinstance(
            intended_use, MPMIntendedUse
        ):
            raise TypeError("Release bundle needs a claim and intended use.")
        if len(gates_) != len(MPMReleaseGate) or {value.gate for value in gates_} != set(
            MPMReleaseGate
        ):
            raise ValueError("Release bundle must contain every G0-G7 gate exactly once.")
        if not approver:
            raise ValueError("Release bundle requires independent approver identity.")
        self.claim = claim
        self.intended_use = intended_use
        self.gates = gates_
        self.independent_approver_id = approver
        self.bundle_id = canonical_fingerprint(
            {
                "kind": "mpm-release-evidence-bundle",
                "claim": claim.claim_id,
                "intended_use": intended_use.intended_use_id,
                "gates": [value.gate_id for value in gates_],
                "approver": approver,
            }
        )

    @property
    def releasable(self) -> bool:
        return all(value.passed for value in self.gates) and all(
            not value.deviation_ids for value in self.gates
        )


class MPMRunProvenance(StrictModule, NonTrainableState):
    compilation_id: str = eqx.field(static=True)
    claim_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    source_commit: str = eqx.field(static=True)
    dependency_lock_digest: str = eqx.field(static=True)
    backend_device: str = eqx.field(static=True)
    input_digest: str = eqx.field(static=True)
    sbom_digest: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        compilation_id: str,
        claim_id: str,
        execution_plan_id: str,
        source_commit: str,
        dependency_lock_digest: str,
        backend_device: str,
        input_digest: str,
        sbom_digest: str,
    ):
        values = {key: str(value) for key, value in locals().items() if key != "self"}
        if any(not value for value in values.values()):
            raise ValueError("Run provenance fields must be non-empty.")
        for key, value in values.items():
            setattr(self, key, value)
        self.provenance_id = canonical_fingerprint(
            {"kind": "mpm-run-provenance", **values}
        )


__all__ = [
    "MPMClaimOutcome",
    "MPMClaimTuple",
    "MPMCommercialFailure",
    "MPMDerivativeEvidence",
    "MPMDerivativeKind",
    "MPMEventJournal",
    "MPMIntendedUse",
    "MPMOperationalStatus",
    "MPMReleaseEvidenceBundle",
    "MPMReleaseGate",
    "MPMReleaseGateEvidence",
    "MPMRunProvenance",
    "MPMSupportDecision",
    "MPMSupportMatrix",
    "MPMTopologyJournal",
]
