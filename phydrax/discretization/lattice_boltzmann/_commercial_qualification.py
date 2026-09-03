#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification._evidence import (
    QualificationCoverageReport,
    QualificationEvidence,
    QualificationMatrix,
    SupportDependency,
)
from ...qualification._registry import SupportTuple, SupportValue
from ._collision import BGKCollisionPlan, CentralMomentCollisionPlan, TRTCollisionPlan
from ._conjugate_thermal import ConjugateThermalPlan
from ._forcing import GuoForcingPlan
from ._interfacial import ConstitutiveDynamicWettingPlan
from ._lattice import D2Q9, D3Q19, D3Q27, LatticeBoltzmannVelocitySet
from ._moments import MomentBasisPlan, RelaxationSpectrumPlan
from ._operating_envelope import (
    LatticeBoltzmannEnvelopeAdmission,
    LatticeBoltzmannHardwareTarget,
    LatticeBoltzmannOperatingEnvelopePlan,
)
from ._precision import LatticeBoltzmannPrecisionPolicy


class LatticeBoltzmannCommercialTier(StrEnum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    CONJUGATE_THERMAL = "conjugate-thermal"


class LatticeBoltzmannQualificationClaim(StrEnum):
    SHEAR_WAVE_DECAY = "shear-wave-decay"
    ACOUSTIC_ATTENUATION = "acoustic-attenuation"
    COUETTE_FLOW = "couette-flow"
    POISEUILLE_FLOW = "poiseuille-flow"
    CYLINDER_WAKE = "cylinder-wake"
    RELAXATION_SWEEP = "relaxation-sweep"
    MACH_SWEEP = "mach-sweep"
    COLLISION_FORCING_CONSERVATION = "collision-forcing-conservation"
    INTERFACE_EQUILIBRIUM = "interface-equilibrium"
    LAPLACE_PRESSURE = "laplace-pressure"
    CAPILLARY_WAVE = "capillary-wave"
    DROPLET_DEFORMATION = "droplet-deformation"
    DYNAMIC_WETTING = "dynamic-wetting"
    THERMAL_CONSERVATION = "thermal-conservation"
    SPECIES_CONSERVATION = "species-conservation"
    REACTIVE_SPLITTING_CONSERVATION = "reactive-splitting-conservation"
    CONJUGATE_THERMAL_CONSERVATION = "conjugate-thermal-conservation"
    FUSED_PARITY = "fused-parity"
    AA_PARITY = "aa-parity"
    SHARDED_PARITY = "sharded-parity"
    CHECKPOINT_PARITY = "checkpoint-parity"
    OUTPUT_PARITY = "output-parity"


class LatticeBoltzmannQualificationError(RuntimeError):
    """Raised when a commercial LBM gate is explicitly required but fails."""


_SCIENTIFIC_CLAIMS = frozenset(
    (
        LatticeBoltzmannQualificationClaim.SHEAR_WAVE_DECAY,
        LatticeBoltzmannQualificationClaim.ACOUSTIC_ATTENUATION,
        LatticeBoltzmannQualificationClaim.COUETTE_FLOW,
        LatticeBoltzmannQualificationClaim.POISEUILLE_FLOW,
        LatticeBoltzmannQualificationClaim.CYLINDER_WAKE,
        LatticeBoltzmannQualificationClaim.RELAXATION_SWEEP,
        LatticeBoltzmannQualificationClaim.MACH_SWEEP,
        LatticeBoltzmannQualificationClaim.COLLISION_FORCING_CONSERVATION,
        LatticeBoltzmannQualificationClaim.INTERFACE_EQUILIBRIUM,
        LatticeBoltzmannQualificationClaim.LAPLACE_PRESSURE,
        LatticeBoltzmannQualificationClaim.CAPILLARY_WAVE,
        LatticeBoltzmannQualificationClaim.DROPLET_DEFORMATION,
        LatticeBoltzmannQualificationClaim.DYNAMIC_WETTING,
        LatticeBoltzmannQualificationClaim.THERMAL_CONSERVATION,
        LatticeBoltzmannQualificationClaim.SPECIES_CONSERVATION,
        LatticeBoltzmannQualificationClaim.REACTIVE_SPLITTING_CONSERVATION,
        LatticeBoltzmannQualificationClaim.CONJUGATE_THERMAL_CONSERVATION,
    )
)
_OPERATIONAL_CLAIMS = frozenset(
    (
        LatticeBoltzmannQualificationClaim.FUSED_PARITY,
        LatticeBoltzmannQualificationClaim.AA_PARITY,
        LatticeBoltzmannQualificationClaim.SHARDED_PARITY,
        LatticeBoltzmannQualificationClaim.CHECKPOINT_PARITY,
        LatticeBoltzmannQualificationClaim.OUTPUT_PARITY,
    )
)
_BASELINE_CLAIMS = (
    LatticeBoltzmannQualificationClaim.SHEAR_WAVE_DECAY,
    LatticeBoltzmannQualificationClaim.ACOUSTIC_ATTENUATION,
    LatticeBoltzmannQualificationClaim.COUETTE_FLOW,
    LatticeBoltzmannQualificationClaim.POISEUILLE_FLOW,
    LatticeBoltzmannQualificationClaim.CYLINDER_WAKE,
    LatticeBoltzmannQualificationClaim.RELAXATION_SWEEP,
    LatticeBoltzmannQualificationClaim.MACH_SWEEP,
    LatticeBoltzmannQualificationClaim.COLLISION_FORCING_CONSERVATION,
    LatticeBoltzmannQualificationClaim.FUSED_PARITY,
    LatticeBoltzmannQualificationClaim.AA_PARITY,
    LatticeBoltzmannQualificationClaim.SHARDED_PARITY,
    LatticeBoltzmannQualificationClaim.CHECKPOINT_PARITY,
    LatticeBoltzmannQualificationClaim.OUTPUT_PARITY,
)


def _identifier(value: str, name: str, /) -> str:
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return result


def _claim_kind(claim: LatticeBoltzmannQualificationClaim, /) -> str:
    if claim in _SCIENTIFIC_CLAIMS:
        return "scientific"
    if claim in _OPERATIONAL_CLAIMS:
        return "operational"
    raise ValueError(f"LBM qualification claim {claim.value!r} has no evidence kind.")


def reference_lattice_boltzmann_hardware(
    *,
    host_count: int = 1,
    devices_per_host: int = 1,
    maximum_device_bytes: int = 8 * 1024**3,
) -> LatticeBoltzmannHardwareTarget:
    """Return the explicit portable CPU target used by unsigned reference profiles."""

    return LatticeBoltzmannHardwareTarget(
        "cpu",
        "phydrax-reference",
        "portable-xla-cpu",
        host_count=host_count,
        devices_per_host=devices_per_host,
        maximum_device_bytes=maximum_device_bytes,
    )


class LatticeBoltzmannDeploymentRecord(StrictModule, NonTrainableState):
    """Exact execution/output/checkpoint topology selected for one run."""

    execution_mode: str = eqx.field(static=True)
    output_mode: str = eqx.field(static=True)
    checkpoint_mode: str = eqx.field(static=True)
    host_count: int = eqx.field(static=True)
    devices_per_host: int = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    output_plan_id: str = eqx.field(static=True)
    checkpoint_plan_id: str = eqx.field(static=True)
    execution_topology_id: str = eqx.field(static=True)
    restart_topology_id: str = eqx.field(static=True)
    topology_restart_relation_id: str | None = eqx.field(static=True)
    parity_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    deployment_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_mode: str,
        output_mode: str,
        checkpoint_mode: str,
        /,
        *,
        host_count: int,
        devices_per_host: int,
        execution_plan_id: str,
        output_plan_id: str,
        checkpoint_plan_id: str,
        execution_topology_id: str,
        restart_topology_id: str,
        topology_restart_relation_id: str | None = None,
        parity_evidence_ids: Sequence[str],
    ):
        execution = _identifier(execution_mode, "execution_mode")
        output = _identifier(output_mode, "output_mode")
        checkpoint = _identifier(checkpoint_mode, "checkpoint_mode")
        hosts = int(host_count)
        devices = int(devices_per_host)
        if hosts <= 0 or devices <= 0:
            raise ValueError("Host and per-host device counts must be positive.")
        plan_ids = tuple(
            _identifier(value, name)
            for value, name in (
                (execution_plan_id, "execution_plan_id"),
                (output_plan_id, "output_plan_id"),
                (checkpoint_plan_id, "checkpoint_plan_id"),
                (execution_topology_id, "execution_topology_id"),
                (restart_topology_id, "restart_topology_id"),
            )
        )
        relation = (
            None
            if topology_restart_relation_id is None
            else _identifier(
                topology_restart_relation_id,
                "topology_restart_relation_id",
            )
        )
        evidence = tuple(
            sorted(
                _identifier(value, "parity evidence ID") for value in parity_evidence_ids
            )
        )
        if not evidence or len(set(evidence)) != len(evidence):
            raise ValueError("Deployment parity evidence must be nonempty and unique.")
        self.execution_mode = execution
        self.output_mode = output
        self.checkpoint_mode = checkpoint
        self.host_count = hosts
        self.devices_per_host = devices
        (
            self.execution_plan_id,
            self.output_plan_id,
            self.checkpoint_plan_id,
            self.execution_topology_id,
            self.restart_topology_id,
        ) = plan_ids
        self.topology_restart_relation_id = relation
        self.parity_evidence_ids = evidence
        self.deployment_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-deployment",
                "execution_mode": execution,
                "output_mode": output,
                "checkpoint_mode": checkpoint,
                "host_count": hosts,
                "devices_per_host": devices,
                "execution_plan": plan_ids[0],
                "output_plan": plan_ids[1],
                "checkpoint_plan": plan_ids[2],
                "execution_topology": plan_ids[3],
                "restart_topology": plan_ids[4],
                "topology_restart_relation": relation,
                "parity_evidence": evidence,
            }
        )

    def to_record(self, /) -> dict[str, object]:
        return {
            "kind": "lattice-boltzmann-deployment",
            "execution_mode": self.execution_mode,
            "output_mode": self.output_mode,
            "checkpoint_mode": self.checkpoint_mode,
            "host_count": self.host_count,
            "devices_per_host": self.devices_per_host,
            "execution_plan_id": self.execution_plan_id,
            "output_plan_id": self.output_plan_id,
            "checkpoint_plan_id": self.checkpoint_plan_id,
            "execution_topology_id": self.execution_topology_id,
            "restart_topology_id": self.restart_topology_id,
            "topology_restart_relation_id": self.topology_restart_relation_id,
            "parity_evidence_ids": list(self.parity_evidence_ids),
            "deployment_id": self.deployment_id,
        }


class LatticeBoltzmannDeploymentCompatibility(StrictModule, NonTrainableState):
    """Named exact deployment predicates with no implicit topology migration."""

    compatible: bool = eqx.field(static=True)
    failed_checks: tuple[str, ...] = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    deployment_id: str = eqx.field(static=True)
    compatibility_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile_id: str,
        deployment_id: str,
        failed_checks: Sequence[str],
        /,
    ):
        profile = _identifier(profile_id, "profile_id")
        deployment = _identifier(deployment_id, "deployment_id")
        failed = tuple(
            sorted(
                _identifier(value, "failed deployment check") for value in failed_checks
            )
        )
        if len(set(failed)) != len(failed):
            raise ValueError("Failed deployment checks must be unique.")
        self.compatible = not failed
        self.failed_checks = failed
        self.profile_id = profile
        self.deployment_id = deployment
        self.compatibility_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-deployment-compatibility",
                "profile": profile,
                "deployment": deployment,
                "failed_checks": failed,
            }
        )


class LatticeBoltzmannQualificationProfile(StrictModule, NonTrainableState):
    """Unsigned candidate profile for one exact LBM operating envelope."""

    name: str = eqx.field(static=True)
    tier: LatticeBoltzmannCommercialTier = eqx.field(static=True)
    envelope: LatticeBoltzmannOperatingEnvelopePlan
    support_tuple: SupportTuple
    dependencies: tuple[SupportDependency, ...]
    required_claims: tuple[LatticeBoltzmannQualificationClaim, ...] = eqx.field(
        static=True
    )
    qualification_matrix: QualificationMatrix
    execution_modes: tuple[str, ...] = eqx.field(static=True)
    output_modes: tuple[str, ...] = eqx.field(static=True)
    checkpoint_modes: tuple[str, ...] = eqx.field(static=True)
    dynamic_wetting: ConstitutiveDynamicWettingPlan | None
    conjugate_thermal: ConjugateThermalPlan | None
    signed: bool = eqx.field(static=True)
    released: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        tier: LatticeBoltzmannCommercialTier,
        envelope: LatticeBoltzmannOperatingEnvelopePlan,
        required_claims: Sequence[LatticeBoltzmannQualificationClaim],
        /,
        *,
        execution_modes: Sequence[str],
        output_modes: Sequence[str],
        checkpoint_modes: Sequence[str],
        additional_support_coordinates: Mapping[str, SupportValue] | None = None,
        dependencies: Sequence[SupportDependency] = (),
        dynamic_wetting: ConstitutiveDynamicWettingPlan | None = None,
        conjugate_thermal: ConjugateThermalPlan | None = None,
    ):
        name_ = _identifier(name, "profile name")
        if not isinstance(tier, LatticeBoltzmannCommercialTier):
            raise TypeError("tier must be a LatticeBoltzmannCommercialTier.")
        if not isinstance(envelope, LatticeBoltzmannOperatingEnvelopePlan):
            raise TypeError("envelope must be a LatticeBoltzmannOperatingEnvelopePlan.")
        claims = tuple(required_claims)
        if not claims or any(
            not isinstance(value, LatticeBoltzmannQualificationClaim) for value in claims
        ):
            raise TypeError(
                "required_claims must contain typed LBM qualification claims."
            )
        if len(set(claims)) != len(claims):
            raise ValueError("LBM required claims must be unique.")
        execution = tuple(
            sorted(_identifier(value, "execution mode") for value in execution_modes)
        )
        output = tuple(
            sorted(_identifier(value, "output mode") for value in output_modes)
        )
        checkpoint = tuple(
            sorted(_identifier(value, "checkpoint mode") for value in checkpoint_modes)
        )
        if not execution or not output or not checkpoint:
            raise ValueError("LBM deployment mode sets must be nonempty.")
        if (
            len(set(execution)) != len(execution)
            or len(set(output)) != len(output)
            or len(set(checkpoint)) != len(checkpoint)
        ):
            raise ValueError("LBM deployment modes must be unique.")
        dependencies_ = tuple(sorted(dependencies, key=lambda value: value.dependency_id))
        if any(not isinstance(value, SupportDependency) for value in dependencies_):
            raise TypeError("dependencies must contain exact SupportDependency values.")
        if len({value.dependency_id for value in dependencies_}) != len(dependencies_):
            raise ValueError("LBM support dependencies must be unique.")
        if dynamic_wetting is not None and not isinstance(
            dynamic_wetting, ConstitutiveDynamicWettingPlan
        ):
            raise TypeError(
                "dynamic_wetting must be ConstitutiveDynamicWettingPlan or None."
            )
        if conjugate_thermal is not None and not isinstance(
            conjugate_thermal, ConjugateThermalPlan
        ):
            raise TypeError("conjugate_thermal must be ConjugateThermalPlan or None.")
        if (
            dynamic_wetting is not None
            and LatticeBoltzmannQualificationClaim.DYNAMIC_WETTING not in claims
        ):
            raise ValueError(
                "A bound dynamic-wetting plan requires its qualification claim."
            )
        if (
            conjugate_thermal is not None
            and tier is not LatticeBoltzmannCommercialTier.CONJUGATE_THERMAL
        ):
            raise ValueError(
                "A conjugate-thermal plan requires the conjugate-thermal tier."
            )
        coordinates: dict[str, SupportValue] = dict(envelope.support_coordinates)
        coordinates["tier"] = tier.value
        additions = (
            {}
            if additional_support_coordinates is None
            else additional_support_coordinates
        )
        for coordinate, value in additions.items():
            key = _identifier(str(coordinate), "support coordinate")
            if key in coordinates:
                raise ValueError(f"Duplicate LBM support coordinate {key!r}.")
            coordinates[key] = value
        labels = tuple(str(value).lower() for value in coordinates.values()) + (
            name_.lower(),
        )
        if any(
            "continuum-dns" in label or "direct-numerical-simulation" in label
            for label in labels
        ):
            raise ValueError("LBM profiles may not claim a continuum-flow method label.")
        support = SupportTuple("lattice-boltzmann", coordinates)
        profile_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-qualification-profile",
                "name": name_,
                "tier": tier.value,
                "envelope": envelope.envelope_id,
                "support_tuple": support.support_tuple_id,
                "dependencies": tuple(value.dependency_id for value in dependencies_),
                "required_claims": tuple(sorted(value.value for value in claims)),
                "execution_modes": execution,
                "output_modes": output,
                "checkpoint_modes": checkpoint,
                "dynamic_wetting": (
                    None if dynamic_wetting is None else dynamic_wetting.plan_id
                ),
                "conjugate_thermal": (
                    None if conjugate_thermal is None else conjugate_thermal.plan_id
                ),
                "signed": False,
                "released": False,
            }
        )
        matrix = QualificationMatrix(
            {
                claim.value: {
                    "evidence_kind": _claim_kind(claim),
                    "subject_id": profile_id,
                    "backend": "jax",
                    "precision": envelope.precision.policy_id,
                    "criterion_id": claim.value,
                }
                for claim in claims
            }
        )
        self.name = name_
        self.tier = tier
        self.envelope = envelope
        self.support_tuple = support
        self.dependencies = dependencies_
        self.required_claims = tuple(sorted(claims, key=lambda value: value.value))
        self.qualification_matrix = matrix
        self.execution_modes = execution
        self.output_modes = output
        self.checkpoint_modes = checkpoint
        self.dynamic_wetting = dynamic_wetting
        self.conjugate_thermal = conjugate_thermal
        self.signed = False
        self.released = False
        self.profile_id = profile_id

    @property
    def support_tuple_id(self) -> str:
        return self.support_tuple.support_tuple_id

    def to_record(self, /) -> dict[str, object]:
        return {
            "kind": "lattice-boltzmann-qualification-profile",
            "name": self.name,
            "tier": self.tier.value,
            "profile_id": self.profile_id,
            "envelope_id": self.envelope.envelope_id,
            "support_tuple": self.support_tuple.to_record(),
            "dependencies": [value.to_record() for value in self.dependencies],
            "required_claims": [value.value for value in self.required_claims],
            "qualification_matrix": self.qualification_matrix.to_record(),
            "execution_modes": list(self.execution_modes),
            "output_modes": list(self.output_modes),
            "checkpoint_modes": list(self.checkpoint_modes),
            "dynamic_wetting_plan_id": (
                None if self.dynamic_wetting is None else self.dynamic_wetting.plan_id
            ),
            "conjugate_thermal_plan_id": (
                None if self.conjugate_thermal is None else self.conjugate_thermal.plan_id
            ),
            "signed": self.signed,
            "released": self.released,
        }

    def deployment_compatibility(
        self, deployment: LatticeBoltzmannDeploymentRecord, /
    ) -> LatticeBoltzmannDeploymentCompatibility:
        if not isinstance(deployment, LatticeBoltzmannDeploymentRecord):
            raise TypeError("deployment must be a LatticeBoltzmannDeploymentRecord.")
        failed: list[str] = []
        if deployment.execution_mode not in self.execution_modes:
            failed.append("execution-mode")
        if deployment.output_mode not in self.output_modes:
            failed.append("output-mode")
        if deployment.checkpoint_mode not in self.checkpoint_modes:
            failed.append("checkpoint-mode")
        if deployment.host_count != self.envelope.hardware.host_count:
            failed.append("host-count")
        if deployment.devices_per_host != self.envelope.hardware.devices_per_host:
            failed.append("devices-per-host")
        if (
            deployment.execution_topology_id != deployment.restart_topology_id
            and deployment.topology_restart_relation_id is None
        ):
            failed.append("restart-topology-relation")
        return LatticeBoltzmannDeploymentCompatibility(
            self.profile_id,
            deployment.deployment_id,
            failed,
        )

    def evaluate(
        self,
        evidence: Sequence[QualificationEvidence],
        admission: LatticeBoltzmannEnvelopeAdmission,
        deployment: LatticeBoltzmannDeploymentRecord,
        /,
        *,
        at_time: int,
        satisfied_dependencies: Sequence[SupportDependency] = (),
    ) -> "LatticeBoltzmannCommercialEvidence":
        if not isinstance(admission, LatticeBoltzmannEnvelopeAdmission):
            raise TypeError("admission must be LatticeBoltzmannEnvelopeAdmission.")
        if admission.envelope_id != self.envelope.envelope_id:
            raise ValueError("Operating admission belongs to a different LBM envelope.")
        evidence_ = tuple(evidence)
        satisfied = tuple(satisfied_dependencies)
        if any(not isinstance(value, SupportDependency) for value in satisfied):
            raise TypeError(
                "satisfied_dependencies must contain exact SupportDependency values."
            )
        report = self.qualification_matrix.evaluate(evidence_, at_time=at_time)
        compatibility = self.deployment_compatibility(deployment)
        evidence_ids = frozenset(value.evidence_id for value in evidence_)
        if not set(deployment.parity_evidence_ids).issubset(evidence_ids):
            compatibility = LatticeBoltzmannDeploymentCompatibility(
                self.profile_id,
                deployment.deployment_id,
                compatibility.failed_checks + ("parity-evidence",),
            )
        return LatticeBoltzmannCommercialEvidence(
            self,
            report,
            admission,
            compatibility,
            evidence_,
            satisfied,
        )


class LatticeBoltzmannCommercialEvidence(StrictModule, NonTrainableState):
    """Commercial gate result retaining coverage, refusal, and deployment evidence."""

    profile: LatticeBoltzmannQualificationProfile
    coverage: QualificationCoverageReport
    admission: LatticeBoltzmannEnvelopeAdmission
    deployment: LatticeBoltzmannDeploymentCompatibility
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    satisfied_dependency_ids: tuple[str, ...] = eqx.field(static=True)
    missing_dependency_ids: tuple[str, ...] = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: LatticeBoltzmannQualificationProfile,
        coverage: QualificationCoverageReport,
        admission: LatticeBoltzmannEnvelopeAdmission,
        deployment: LatticeBoltzmannDeploymentCompatibility,
        evidence: Sequence[QualificationEvidence],
        satisfied_dependencies: Sequence[SupportDependency],
        /,
    ):
        if not isinstance(profile, LatticeBoltzmannQualificationProfile):
            raise TypeError("profile must be LatticeBoltzmannQualificationProfile.")
        if not isinstance(coverage, QualificationCoverageReport):
            raise TypeError("coverage must be QualificationCoverageReport.")
        if not isinstance(admission, LatticeBoltzmannEnvelopeAdmission):
            raise TypeError("admission must be LatticeBoltzmannEnvelopeAdmission.")
        if not isinstance(deployment, LatticeBoltzmannDeploymentCompatibility):
            raise TypeError("deployment must be LatticeBoltzmannDeploymentCompatibility.")
        evidence_ = tuple(evidence)
        if any(not isinstance(value, QualificationEvidence) for value in evidence_):
            raise TypeError("evidence must contain QualificationEvidence values.")
        evidence_ids = tuple(sorted(value.evidence_id for value in evidence_))
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Commercial LBM evidence IDs must be unique.")
        satisfied = tuple(satisfied_dependencies)
        if any(not isinstance(value, SupportDependency) for value in satisfied):
            raise TypeError(
                "satisfied_dependencies must contain exact SupportDependency values."
            )
        satisfied_ids = tuple(sorted(value.dependency_id for value in satisfied))
        if len(set(satisfied_ids)) != len(satisfied_ids):
            raise ValueError("Satisfied LBM dependency IDs must be unique.")
        required_ids = frozenset(value.dependency_id for value in profile.dependencies)
        missing_ids = tuple(sorted(required_ids - set(satisfied_ids)))
        admitted = bool(admission.admitted)
        passed = (
            coverage.passed and admitted and deployment.compatible and not missing_ids
        )
        self.profile = profile
        self.coverage = coverage
        self.admission = admission
        self.deployment = deployment
        self.evidence_ids = evidence_ids
        self.satisfied_dependency_ids = satisfied_ids
        self.missing_dependency_ids = missing_ids
        self.passed = passed
        self.record_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-commercial-evidence",
                "profile": profile.profile_id,
                "support_tuple": profile.support_tuple_id,
                "coverage": coverage.report_id,
                "admission": {
                    "envelope": admission.envelope_id,
                    "admitted": admitted,
                    "failed_checks": admission.failed_checks(),
                },
                "deployment": deployment.compatibility_id,
                "evidence": evidence_ids,
                "satisfied_dependencies": satisfied_ids,
                "missing_dependencies": missing_ids,
                "passed": passed,
            }
        )

    def to_record(self, /) -> dict[str, object]:
        return {
            "kind": "lattice-boltzmann-commercial-evidence",
            "record_id": self.record_id,
            "profile_id": self.profile.profile_id,
            "support_tuple_id": self.profile.support_tuple_id,
            "coverage": self.coverage.to_record(),
            "envelope_id": self.admission.envelope_id,
            "admitted": bool(self.admission.admitted),
            "failed_envelope_checks": list(self.admission.failed_checks()),
            "deployment_compatibility_id": self.deployment.compatibility_id,
            "failed_deployment_checks": list(self.deployment.failed_checks),
            "evidence_ids": list(self.evidence_ids),
            "satisfied_dependency_ids": list(self.satisfied_dependency_ids),
            "missing_dependency_ids": list(self.missing_dependency_ids),
            "passed": self.passed,
        }

    def require(self, /) -> None:
        if self.passed:
            return
        reasons: list[str] = []
        if not self.coverage.passed:
            reasons.append(f"evidence:{self.coverage.outcome}")
        if not bool(self.admission.admitted):
            reasons.extend(
                f"envelope:{value}" for value in self.admission.failed_checks()
            )
        reasons.extend(f"deployment:{value}" for value in self.deployment.failed_checks)
        reasons.extend(f"dependency:{value}" for value in self.missing_dependency_ids)
        raise LatticeBoltzmannQualificationError(
            "Commercial LBM qualification failed: " + ", ".join(reasons) + "."
        )


def _selected_precision(
    precision: LatticeBoltzmannPrecisionPolicy | None, /
) -> LatticeBoltzmannPrecisionPolicy:
    if precision is None:
        return LatticeBoltzmannPrecisionPolicy()
    if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
        raise TypeError("precision must be LatticeBoltzmannPrecisionPolicy or None.")
    return precision


def _selected_hardware(
    hardware: LatticeBoltzmannHardwareTarget | None, /
) -> LatticeBoltzmannHardwareTarget:
    if hardware is None:
        return reference_lattice_boltzmann_hardware()
    if not isinstance(hardware, LatticeBoltzmannHardwareTarget):
        raise TypeError("hardware must be LatticeBoltzmannHardwareTarget or None.")
    return hardware


def _envelope(
    lattice: LatticeBoltzmannVelocitySet,
    collision: BGKCollisionPlan | TRTCollisionPlan,
    precision: LatticeBoltzmannPrecisionPolicy,
    hardware: LatticeBoltzmannHardwareTarget,
    /,
    *,
    physics_model: str,
    boundary_model: str,
    interface: bool = False,
) -> LatticeBoltzmannOperatingEnvelopePlan:
    return LatticeBoltzmannOperatingEnvelopePlan(
        lattice,
        collision,
        GuoForcingPlan(),
        precision,
        hardware,
        physics_model=physics_model,
        boundary_model=boundary_model,
        relaxation_rate_limits=(0.55, 1.90),
        maximum_mach_number=0.10,
        maximum_knudsen_number=0.01,
        density_limits=(0.1, 10.0) if interface else (0.9, 1.1),
        maximum_density_ratio=10.0 if interface else 1.10,
        maximum_force_number=0.01,
        minimum_interface_width_cells=4.0 if interface else 0.0,
        minimum_wall_resolution_cells=8.0,
        maximum_viscosity_ratio=10.0 if interface else 1.10,
        maximum_cahn_number=0.10 if interface else 0.0,
        maximum_capillary_number=0.10 if interface else 0.0,
        maximum_relative_mass_drift=1.0e-8,
        maximum_spurious_current_ratio=1.0e-3 if interface else 0.0,
    )


def c0_guo_baseline_profiles(
    *,
    precision: LatticeBoltzmannPrecisionPolicy | None = None,
    hardware: LatticeBoltzmannHardwareTarget | None = None,
) -> tuple[LatticeBoltzmannQualificationProfile, ...]:
    """Return the four unsigned D2Q9/D3Q19 BGK/TRT Guo candidates."""

    precision_ = _selected_precision(precision)
    hardware_ = _selected_hardware(hardware)
    profiles: list[LatticeBoltzmannQualificationProfile] = []
    for lattice in (D2Q9(), D3Q19()):
        for collision in (BGKCollisionPlan(), TRTCollisionPlan()):
            envelope = _envelope(
                lattice,
                collision,
                precision_,
                hardware_,
                physics_model="athermal-single-phase-lattice-kinetic",
                boundary_model="periodic-halfway-bounce-back",
            )
            profiles.append(
                LatticeBoltzmannQualificationProfile(
                    f"c0-{lattice.name.lower()}-{collision.family}-guo",
                    LatticeBoltzmannCommercialTier.C0,
                    envelope,
                    _BASELINE_CLAIMS,
                    execution_modes=("reference", "fused", "aa", "sharded"),
                    output_modes=("memory", "array-archive"),
                    checkpoint_modes=("kinetic-array-archive", "aa-raw"),
                    additional_support_coordinates={"forcing_route": "population-space"},
                )
            )
    return tuple(profiles)


def c1_collision_native_forcing_profile(
    *,
    precision: LatticeBoltzmannPrecisionPolicy | None = None,
    hardware: LatticeBoltzmannHardwareTarget | None = None,
) -> LatticeBoltzmannQualificationProfile:
    """Return the selected D3Q27 central-moment/Guo forcing candidate."""

    precision_ = _selected_precision(precision)
    hardware_ = _selected_hardware(hardware)
    collision = CentralMomentCollisionPlan(MomentBasisPlan(), RelaxationSpectrumPlan())
    envelope = LatticeBoltzmannOperatingEnvelopePlan(
        D3Q27(),
        collision,
        GuoForcingPlan(),
        precision_,
        hardware_,
        physics_model="athermal-single-phase-lattice-kinetic",
        boundary_model="staged-linkwise",
        relaxation_rate_limits=(0.52, 1.92),
        maximum_mach_number=0.15,
        maximum_knudsen_number=0.01,
        density_limits=(0.9, 1.1),
        maximum_density_ratio=1.20,
        maximum_force_number=0.025,
        minimum_wall_resolution_cells=10.0,
        maximum_viscosity_ratio=1.20,
        maximum_relative_mass_drift=1.0e-9,
    )
    return LatticeBoltzmannQualificationProfile(
        "c1-d3q27-central-moment-guo",
        LatticeBoltzmannCommercialTier.C1,
        envelope,
        _BASELINE_CLAIMS,
        execution_modes=("reference", "fused", "aa", "sharded"),
        output_modes=("memory", "array-archive"),
        checkpoint_modes=("kinetic-array-archive", "aa-raw"),
        additional_support_coordinates={
            "forcing_route": "collision-native-central-moment"
        },
    )


def c2_binary_interface_profiles(
    *,
    precision: LatticeBoltzmannPrecisionPolicy | None = None,
    hardware: LatticeBoltzmannHardwareTarget | None = None,
    dynamic_wetting: ConstitutiveDynamicWettingPlan | None = None,
) -> tuple[LatticeBoltzmannQualificationProfile, ...]:
    """Return selected free-energy and colour-gradient binary-interface candidates."""

    precision_ = _selected_precision(precision)
    hardware_ = _selected_hardware(hardware)
    interface_claims = _BASELINE_CLAIMS + (
        LatticeBoltzmannQualificationClaim.INTERFACE_EQUILIBRIUM,
        LatticeBoltzmannQualificationClaim.LAPLACE_PRESSURE,
        LatticeBoltzmannQualificationClaim.CAPILLARY_WAVE,
        LatticeBoltzmannQualificationClaim.DROPLET_DEFORMATION,
    )
    profiles: list[LatticeBoltzmannQualificationProfile] = []
    for family, wetting_label in (
        ("binary-free-energy", "natural-cubic-wall-energy"),
        (
            "binary-colour-gradient",
            "static-contact-angle"
            if dynamic_wetting is None
            else dynamic_wetting.model_label,
        ),
    ):
        selected_dynamic = dynamic_wetting if family == "binary-colour-gradient" else None
        claims = interface_claims + (
            (LatticeBoltzmannQualificationClaim.DYNAMIC_WETTING,)
            if selected_dynamic is not None
            else ()
        )
        envelope = _envelope(
            D2Q9(),
            TRTCollisionPlan(),
            precision_,
            hardware_,
            physics_model=family,
            boundary_model="periodic-halfway-bounce-back-wetting",
            interface=True,
        )
        profiles.append(
            LatticeBoltzmannQualificationProfile(
                f"c2-d2q9-trt-guo-{family}",
                LatticeBoltzmannCommercialTier.C2,
                envelope,
                claims,
                execution_modes=("reference", "fused", "aa", "sharded"),
                output_modes=("memory", "array-archive"),
                checkpoint_modes=("kinetic-array-archive", "aa-raw"),
                additional_support_coordinates={
                    "interface_family": family,
                    "wetting_model": wetting_label,
                },
                dynamic_wetting=selected_dynamic,
            )
        )
    return tuple(profiles)


def c2_dynamic_wetting_profile(
    dynamic_wetting: ConstitutiveDynamicWettingPlan,
    /,
    *,
    precision: LatticeBoltzmannPrecisionPolicy | None = None,
    hardware: LatticeBoltzmannHardwareTarget | None = None,
) -> LatticeBoltzmannQualificationProfile:
    """Return only the colour-gradient tuple bound to a constitutive wetting law."""

    if not isinstance(dynamic_wetting, ConstitutiveDynamicWettingPlan):
        raise TypeError("dynamic_wetting must be ConstitutiveDynamicWettingPlan.")
    return c2_binary_interface_profiles(
        precision=precision,
        hardware=hardware,
        dynamic_wetting=dynamic_wetting,
    )[1]


def c3_passive_transport_profiles(
    *,
    precision: LatticeBoltzmannPrecisionPolicy | None = None,
    hardware: LatticeBoltzmannHardwareTarget | None = None,
) -> tuple[LatticeBoltzmannQualificationProfile, ...]:
    """Return passive thermal, passive species, and Strang-reactive candidates."""

    precision_ = _selected_precision(precision)
    hardware_ = _selected_hardware(hardware)
    specifications = (
        (
            "passive-sensible-energy",
            (LatticeBoltzmannQualificationClaim.THERMAL_CONSERVATION,),
            {},
        ),
        (
            "passive-fickian-species",
            (LatticeBoltzmannQualificationClaim.SPECIES_CONSERVATION,),
            {},
        ),
        (
            "reactive-thermal-species-strang",
            (
                LatticeBoltzmannQualificationClaim.THERMAL_CONSERVATION,
                LatticeBoltzmannQualificationClaim.SPECIES_CONSERVATION,
                LatticeBoltzmannQualificationClaim.REACTIVE_SPLITTING_CONSERVATION,
            ),
            {"splitting": "symmetric-strang"},
        ),
    )
    profiles: list[LatticeBoltzmannQualificationProfile] = []
    for physics, transport_claims, coordinates in specifications:
        envelope = _envelope(
            D2Q9(),
            TRTCollisionPlan(),
            precision_,
            hardware_,
            physics_model=physics,
            boundary_model="periodic-linkwise-scalar",
        )
        profiles.append(
            LatticeBoltzmannQualificationProfile(
                f"c3-d2q9-trt-guo-{physics}",
                LatticeBoltzmannCommercialTier.C3,
                envelope,
                _BASELINE_CLAIMS + transport_claims,
                execution_modes=("reference", "fused", "aa", "sharded"),
                output_modes=("memory", "array-archive"),
                checkpoint_modes=("kinetic-array-archive", "aa-raw"),
                additional_support_coordinates=coordinates,
            )
        )
    return tuple(profiles)


def conjugate_thermal_qualification_profile(
    conjugate_thermal: ConjugateThermalPlan,
    /,
    *,
    precision: LatticeBoltzmannPrecisionPolicy | None = None,
    hardware: LatticeBoltzmannHardwareTarget | None = None,
) -> LatticeBoltzmannQualificationProfile:
    """Bind a real solid-energy/interface-flux plan to its unsigned tier."""

    if not isinstance(conjugate_thermal, ConjugateThermalPlan):
        raise TypeError("conjugate_thermal must be ConjugateThermalPlan.")
    precision_ = _selected_precision(precision)
    hardware_ = _selected_hardware(hardware)
    envelope = _envelope(
        D2Q9(),
        TRTCollisionPlan(),
        precision_,
        hardware_,
        physics_model=conjugate_thermal.model_label,
        boundary_model="resolved-fluid-solid-interface",
    )
    return LatticeBoltzmannQualificationProfile(
        "conjugate-thermal-d2q9-trt-guo",
        LatticeBoltzmannCommercialTier.CONJUGATE_THERMAL,
        envelope,
        _BASELINE_CLAIMS
        + (
            LatticeBoltzmannQualificationClaim.THERMAL_CONSERVATION,
            LatticeBoltzmannQualificationClaim.CONJUGATE_THERMAL_CONSERVATION,
        ),
        execution_modes=("reference", "fused", "aa", "sharded"),
        output_modes=("memory", "array-archive"),
        checkpoint_modes=("kinetic-array-archive", "aa-raw"),
        additional_support_coordinates={
            "conjugate_thermal_plan": conjugate_thermal.plan_id,
            "solid_energy_state": True,
            "interface_flux": "equal-and-opposite",
        },
        conjugate_thermal=conjugate_thermal,
    )


__all__ = [
    "LatticeBoltzmannCommercialEvidence",
    "LatticeBoltzmannCommercialTier",
    "LatticeBoltzmannDeploymentCompatibility",
    "LatticeBoltzmannDeploymentRecord",
    "LatticeBoltzmannQualificationClaim",
    "LatticeBoltzmannQualificationProfile",
    "LatticeBoltzmannQualificationError",
    "c0_guo_baseline_profiles",
    "c1_collision_native_forcing_profile",
    "c2_binary_interface_profiles",
    "c2_dynamic_wetting_profile",
    "c3_passive_transport_profiles",
    "conjugate_thermal_qualification_profile",
    "reference_lattice_boltzmann_hardware",
]
