#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic, fail-closed qualification records for immersed DNS routes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.incompressible_flow._immersed_admission import (
    ImmersedRuntimeAdmissionResult,
)
from phydrax.applications.incompressible_flow._immersed_profile import (
    ImmersedDNSQualificationProfile,
)
from phydrax.applications.incompressible_flow._immersed_qualification import (
    ImmersedReferenceCampaignPlan,
)
from phydrax.applications.incompressible_flow._immersed_support import (
    ImmersedBodyRegimePlan,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.lifecycle._restart_topology import (
    RestartAdmission,
    TopologyRestartRelation,
)
from phydrax.qualification._evidence import (
    QualificationCoverageReport,
    QualificationEvidence,
    SupportDependency,
)
from phydrax.qualification._reference import ReferenceArtifactManifest


_BODY_METHOD = {
    "prescribed-marker": "marker-regularized",
    "free-rigid-marker": "marker-regularized",
    "fixed-topology-sharp": "sharp-interface",
    "deformable-contact": "marker-regularized",
    "lbm-body": "lattice-boltzmann-body",
    "resolved-cfd-dem": "marker-regularized",
}
_GATE_KINDS = {
    "scientific": frozenset(("scientific", "reference")),
    "performance": frozenset(("performance",)),
    "operational": frozenset(("operational",)),
    "security": frozenset(("security",)),
}


def _record_gate(
    name: str,
    evidence: Sequence[QualificationEvidence],
    /,
    *,
    at_time: int,
    failed_reasons: Sequence[str] = (),
    inconclusive_reasons: Sequence[str] = (),
    require_evidence: bool = True,
) -> dict[str, object]:
    values = tuple(evidence)
    if any(not isinstance(value, QualificationEvidence) for value in values):
        raise TypeError("Gate evidence must contain QualificationEvidence values.")
    wrong = tuple(
        value.evidence_id
        for value in values
        if value.evidence_kind not in _GATE_KINDS[name]
    )
    if wrong:
        raise ValueError(f"{name} gate received evidence of the wrong kind: {wrong}.")
    failed = list(str(value) for value in failed_reasons)
    inconclusive = list(str(value) for value in inconclusive_reasons)
    for value in values:
        if not value.is_current(at_time):
            inconclusive.append(f"not-current:{value.evidence_id}")
        elif value.failed:
            failed.append(f"failed:{value.evidence_id}:{value.reason}")
        elif value.inconclusive:
            inconclusive.append(f"inconclusive:{value.evidence_id}:{value.reason}")
    if require_evidence and not values:
        inconclusive.append("missing-evidence")
    outcome = "failed" if failed else "inconclusive" if inconclusive else "passed"
    return {
        "gate": name,
        "outcome": outcome,
        "evidence_ids": sorted(value.evidence_id for value in values),
        "failed_reasons": sorted(set(failed)),
        "inconclusive_reasons": sorted(set(inconclusive)),
    }


def _restart_record(
    relation: TopologyRestartRelation,
    admission: RestartAdmission,
    /,
) -> dict[str, object]:
    if not isinstance(relation, TopologyRestartRelation) or not isinstance(
        admission, RestartAdmission
    ):
        raise TypeError("Restart evidence requires a typed relation and admission.")
    if admission.relation_id != relation.relation_id:
        raise ValueError("Restart admission belongs to another topology relation.")
    return {
        "kind": "immersed-topology-restart-evidence",
        "relation_id": relation.relation_id,
        "admission_id": admission.admission_id,
        "source_support_tuple_id": relation.source_support_tuple_id,
        "target_support_tuple_id": relation.target_support_tuple_id,
        "restart_class": relation.restart_class,
        "absolute_tolerance": relation.absolute_tolerance,
        "relative_tolerance": relation.relative_tolerance,
        "admitted": admission.admitted,
        "reason": admission.reason,
    }


def _bound_run(
    run_spec: ResolvedRunSpec,
    profile_id: str,
    support_tuple_id: str,
    /,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if not isinstance(run_spec, ResolvedRunSpec):
        raise TypeError("run_spec must be a ResolvedRunSpec.")
    target = SupportDependency(profile_id, support_tuple_id)
    all_dependencies = (
        *run_spec.scientific_dependencies,
        *run_spec.deployment_dependencies,
    )
    reasons = (
        ()
        if any(value.dependency_id == target.dependency_id for value in all_dependencies)
        else ("resolved-run-missing-exact-route-support",)
    )
    return run_spec.to_record(), reasons


def immersed_profile_record(
    profile: ImmersedDNSQualificationProfile | None = None, /
) -> dict[str, object]:
    """Serialize the actual six-route unsigned profile without promoting it."""

    value = ImmersedDNSQualificationProfile() if profile is None else profile
    if not isinstance(value, ImmersedDNSQualificationProfile):
        raise TypeError("profile must be an ImmersedDNSQualificationProfile.")
    supports = tuple(
        {
            "support_tuple": support.to_record(),
            "regime": dict(support.attributes)["regime"],
            "body_method": _BODY_METHOD[str(dict(support.attributes)["regime"])],
        }
        for support in value.support_tuples
    )
    core = {
        "kind": "immersed-commercial-qualification-profile-candidate",
        "source_profile_id": value.profile_id,
        "capability_profile": value.capability_profile.to_record(),
        "qualification_matrix": value.qualification_matrix.to_record(),
        "supports": sorted(supports, key=lambda item: str(item["regime"])),
        "required_reference_cases": list(value.required_reference_cases),
        "tolerances": {
            "load": value.load_tolerance,
            "reference": value.reference_tolerance,
            "conservation": value.conservation_tolerance,
            "sharp_measure": value.sharp_measure_tolerance,
            "marker_condition_limit": value.marker_condition_limit,
        },
        "signed": False,
        "released": False,
        "release_ready": False,
    }
    return {**core, "candidate_profile_id": canonical_fingerprint(core)}


def build_immersed_candidate(
    profile: ImmersedDNSQualificationProfile,
    campaign: ImmersedReferenceCampaignPlan,
    regime: ImmersedBodyRegimePlan,
    coverage: QualificationCoverageReport,
    admission: ImmersedRuntimeAdmissionResult,
    run_spec: ResolvedRunSpec,
    evidence: Sequence[QualificationEvidence],
    /,
    *,
    at_time: int,
    reference_manifests: Sequence[ReferenceArtifactManifest] = (),
    restart_relation: TopologyRestartRelation | None = None,
    restart_admission: RestartAdmission | None = None,
) -> dict[str, object]:
    """Bind one exact immersed route to evidence while remaining unreleased."""

    if not isinstance(profile, ImmersedDNSQualificationProfile):
        raise TypeError("profile must be an ImmersedDNSQualificationProfile.")
    if not isinstance(campaign, ImmersedReferenceCampaignPlan):
        raise TypeError("campaign must be an ImmersedReferenceCampaignPlan.")
    if not isinstance(regime, ImmersedBodyRegimePlan):
        raise TypeError("regime must be an ImmersedBodyRegimePlan.")
    if not isinstance(coverage, QualificationCoverageReport):
        raise TypeError("coverage must be a QualificationCoverageReport.")
    if not isinstance(admission, ImmersedRuntimeAdmissionResult):
        raise TypeError("admission must be an ImmersedRuntimeAdmissionResult.")
    if campaign.profile.profile_id != profile.profile_id:
        raise ValueError("Campaign and candidate use different immersed profiles.")
    if all(value.plan_id != regime.plan_id for value in campaign.regimes):
        raise ValueError("Campaign does not bind the requested immersed regime.")
    if coverage.matrix_id != campaign.qualification_matrix.matrix_id:
        raise ValueError("Coverage report belongs to another immersed campaign matrix.")
    if admission.preflight.plan.regime.plan_id != regime.plan_id:
        raise ValueError("Runtime admission belongs to another immersed regime.")

    evidence_ = tuple(evidence)
    if any(not isinstance(value, QualificationEvidence) for value in evidence_):
        raise TypeError("evidence must contain QualificationEvidence values.")
    if len({value.evidence_id for value in evidence_}) != len(evidence_):
        raise ValueError("Qualification evidence IDs must be unique.")
    references = tuple(reference_manifests)
    if any(not isinstance(value, ReferenceArtifactManifest) for value in references):
        raise TypeError(
            "reference_manifests must contain ReferenceArtifactManifest values."
        )
    if len({value.manifest_id for value in references}) != len(references):
        raise ValueError("Reference manifest IDs must be unique.")

    grouped = {
        name: tuple(value for value in evidence_ if value.evidence_kind in kinds)
        for name, kinds in _GATE_KINDS.items()
    }
    categorized_ids = {
        value.evidence_id for values in grouped.values() for value in values
    }
    if categorized_ids != {value.evidence_id for value in evidence_}:
        raise ValueError(
            "Immersed candidate evidence must use scientific/reference, "
            "performance, operational, or security kinds."
        )
    subject_failures = {
        name: tuple(
            f"route-subject-mismatch:{value.evidence_id}"
            for value in values
            if regime.owner_plan_id not in value.subject_ids
            and regime.plan_id not in value.subject_ids
        )
        for name, values in grouped.items()
    }
    scientific_failures: list[str] = []
    scientific_gaps: list[str] = []
    if coverage.outcome == "failed":
        scientific_failures.extend(
            f"campaign:{predicate}:{reason}"
            for predicate, _, reasons in coverage.gaps
            for reason in reasons
        )
    elif coverage.outcome == "inconclusive":
        scientific_gaps.extend(
            f"campaign:{predicate}:{reason}"
            for predicate, _, reasons in coverage.gaps
            for reason in reasons
        )
    runtime = admission.runtime_evidence
    if admission.preflight.plan.derivative_mode != "none" and not bool(
        np.asarray(runtime.differentiation_routes_frozen)
    ):
        scientific_failures.append("differentiation-route-not-frozen")

    performance_failures = []
    if not bool(np.asarray(admission.preflight.resource_admitted)):
        performance_failures.append("immersed-resource-budget-exceeded")

    operational_failures = []
    operational_gaps = []
    if not bool(np.asarray(admission.admitted)):
        operational_failures.append(
            f"runtime-admission-status:{int(np.asarray(admission.status))}"
        )
    restart_record: dict[str, object] | None
    if restart_relation is None and restart_admission is None:
        restart_record = None
        operational_gaps.append("missing-topology-restart-evidence")
    elif restart_relation is None or restart_admission is None:
        raise ValueError("Restart relation and admission must be supplied together.")
    else:
        restart_record = _restart_record(restart_relation, restart_admission)
        if (
            restart_relation.target_support_tuple_id
            != regime.support_tuple.support_tuple_id
        ):
            operational_failures.append("restart-target-support-mismatch")
        if not restart_admission.admitted:
            operational_failures.append(f"restart-refused:{restart_admission.reason}")

    rights_failures = tuple(
        f"reference:{manifest.manifest_id}:{reason}"
        for manifest in references
        for reason in manifest.rights_refusal_reasons(commercial_use=True)
    )
    run_record, run_failures = _bound_run(
        run_spec,
        profile.capability_profile.profile_id,
        regime.support_tuple.support_tuple_id,
    )
    operational_failures.extend(run_failures)

    gates = {
        "scientific": _record_gate(
            "scientific",
            grouped["scientific"],
            at_time=at_time,
            failed_reasons=(*scientific_failures, *subject_failures["scientific"]),
            inconclusive_reasons=scientific_gaps,
            require_evidence=False,
        ),
        "performance": _record_gate(
            "performance",
            grouped["performance"],
            at_time=at_time,
            failed_reasons=(*performance_failures, *subject_failures["performance"]),
        ),
        "operational": _record_gate(
            "operational",
            grouped["operational"],
            at_time=at_time,
            failed_reasons=(*operational_failures, *subject_failures["operational"]),
            inconclusive_reasons=operational_gaps,
        ),
        "security": _record_gate(
            "security",
            grouped["security"],
            at_time=at_time,
            failed_reasons=(*rights_failures, *subject_failures["security"]),
        ),
    }
    outcomes = tuple(str(value["outcome"]) for value in gates.values())
    status = (
        "failed"
        if "failed" in outcomes
        else "inconclusive"
        if "inconclusive" in outcomes
        else "passed"
    )
    core = {
        "kind": "immersed-commercial-qualification-candidate",
        "route": regime.regime,
        "body_method": _BODY_METHOD[regime.regime],
        "support_tuple": regime.support_tuple.to_record(),
        "profile": immersed_profile_record(profile),
        "campaign": {
            "campaign_id": campaign.campaign_id,
            "required_cases": list(campaign.required_cases),
            "coverage": coverage.to_record(),
        },
        "regime": {
            "plan_id": regime.plan_id,
            "owner_plan_id": regime.owner_plan_id,
            "marker_set_id": regime.marker_set_id,
            "geometry_id": regime.geometry_id,
            "route_id": regime.route_id,
            "topology_epoch_id": regime.topology_epoch_id,
            "motion_epoch_id": regime.motion_epoch_id,
            "geometry_epoch": regime.geometry_epoch,
        },
        "admission": {
            "plan_id": admission.plan_id,
            "preflight_id": admission.preflight.preflight.evidence_id,
            "runtime_evidence_id": runtime.evidence_id,
            "observed_resource_bytes": admission.preflight.preflight.observed_resource_bytes,
            "maximum_resource_bytes": admission.preflight.plan.maximum_resource_bytes,
            "derivative_mode": admission.preflight.plan.derivative_mode,
            "differentiation_routes_frozen": bool(
                np.asarray(runtime.differentiation_routes_frozen)
            ),
            "status": int(np.asarray(admission.status)),
            "admitted": bool(np.asarray(admission.admitted)),
        },
        "restart": restart_record,
        "resolved_run_spec": run_record,
        "support_dependencies": [
            value.to_record()
            for value in (
                *run_spec.scientific_dependencies,
                *run_spec.deployment_dependencies,
            )
        ],
        "reference_manifests": [
            value.to_record()
            for value in sorted(references, key=lambda item: item.manifest_id)
        ],
        "evidence": {
            name: [
                value.to_record()
                for value in sorted(values, key=lambda item: item.evidence_id)
            ]
            for name, values in grouped.items()
        },
        "gates": gates,
        "status": status,
        "signed": False,
        "released": False,
        "release_ready": False,
    }
    return {**core, "candidate_id": canonical_fingerprint(core)}


def write_record(path: str | Path, record: Mapping[str, object], /) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    write_record(arguments.output, immersed_profile_record())


if __name__ == "__main__":
    main()
