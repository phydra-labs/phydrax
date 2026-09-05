#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Route-isolated qualification candidates for NL/QL/GQL and CE2/GCE2."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from phydrax._fingerprint import canonical_fingerprint
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import QualificationEvidence, SupportDependency
from phydrax.qualification._reference import ReferenceArtifactManifest
from phydrax.qualification._registry import CapabilityProfile, SupportTuple
from phydrax.statistical_dynamics._distributed import (
    DistributedRestartRelation,
    DistributedStatisticalLayout,
)
from phydrax.statistical_dynamics._interactions import AbstractInteractionModel
from phydrax.statistical_dynamics._nilss import NILSSPlan
from phydrax.statistical_dynamics._plan import StatisticalDynamicsPlan


_GATE_KINDS = {
    "scientific": frozenset(("scientific", "reference")),
    "performance": frozenset(("performance",)),
    "operational": frozenset(("operational",)),
    "security": frozenset(("security",)),
}


def statistical_model_label(
    model: AbstractInteractionModel | StatisticalDynamicsPlan, /
) -> str:
    """Return the only admissible public label for an exact model API."""

    if isinstance(model, StatisticalDynamicsPlan):
        return model.closure
    if not isinstance(model, AbstractInteractionModel):
        raise TypeError("model must be an interaction model or StatisticalDynamicsPlan.")
    return {
        "nl": "nl-dns",
        "ql": "ql-ensemble",
        "gql": "gql-ensemble",
    }[model.kind]


def _model_id(model: AbstractInteractionModel | StatisticalDynamicsPlan, /) -> str:
    return model.plan_id if isinstance(model, StatisticalDynamicsPlan) else model.model_id


def _validate_route(
    model: AbstractInteractionModel | StatisticalDynamicsPlan,
    nilss: NILSSPlan | None,
    model_label: str | None,
    /,
) -> str:
    label = statistical_model_label(model)
    if model_label is not None and model_label != label:
        raise ValueError(
            f"Model label {model_label!r} cannot describe the bound {label!r} API."
        )
    if nilss is not None and not isinstance(nilss, NILSSPlan):
        raise TypeError("nilss must be a NILSSPlan or None.")
    if nilss is not None and label != "nl-dns":
        raise ValueError("NILSS evidence is admitted only for the NL DNS route.")
    return label


def statistical_support_tuple(
    model: AbstractInteractionModel | StatisticalDynamicsPlan,
    /,
    *,
    distributed_layout: DistributedStatisticalLayout | None = None,
    nilss: NILSSPlan | None = None,
    model_label: str | None = None,
) -> SupportTuple:
    label = _validate_route(model, nilss, model_label)
    if distributed_layout is not None and not isinstance(
        distributed_layout, DistributedStatisticalLayout
    ):
        raise TypeError(
            "distributed_layout must be a DistributedStatisticalLayout or None."
        )
    attributes: dict[str, str | int | bool] = {
        "model_label": label,
        "model_id": _model_id(model),
        "equation_level": "trajectory"
        if label in ("nl-dns", "ql-ensemble", "gql-ensemble")
        else "second-cumulant",
        "interaction_model": model.interaction_model
        if isinstance(model, StatisticalDynamicsPlan)
        else model.kind,
        "closure_exact": model.closure_exact
        if isinstance(model, StatisticalDynamicsPlan)
        else False,
        "topology": "single-process"
        if distributed_layout is None
        else distributed_layout.topology_id,
        "covariance_storage": "none"
        if distributed_layout is None
        else distributed_layout.covariance.storage,
    }
    if nilss is not None:
        attributes["nilss_plan_id"] = nilss.plan_id
        attributes["differentiation"] = "segmented-nilss"
    else:
        attributes["differentiation"] = "route-native"
    return SupportTuple("statistical-dynamics", attributes)


def statistical_candidate_profile(
    model: AbstractInteractionModel | StatisticalDynamicsPlan,
    /,
    *,
    distributed_layout: DistributedStatisticalLayout | None = None,
    nilss: NILSSPlan | None = None,
    model_label: str | None = None,
    dependencies: Sequence[SupportDependency] = (),
) -> CapabilityProfile:
    support = statistical_support_tuple(
        model,
        distributed_layout=distributed_layout,
        nilss=nilss,
        model_label=model_label,
    )
    return CapabilityProfile(
        "statistical-dynamics.candidate",
        "phydrax",
        "candidate",
        (support,),
        dependencies=tuple(dependencies),
        required_gates=("operational", "performance", "scientific", "security"),
        release_evidence=(),
        released=False,
    )


def statistical_profile_record(profile: CapabilityProfile, /) -> dict[str, object]:
    if not isinstance(profile, CapabilityProfile):
        raise TypeError("profile must be a CapabilityProfile.")
    if profile.released or profile.release_evidence:
        raise ValueError("Statistical qualification profiles must remain unreleased.")
    return {
        **profile.to_record(),
        "signed": False,
        "release_ready": False,
    }


def statistical_profile_records(
    models: Sequence[AbstractInteractionModel | StatisticalDynamicsPlan],
    /,
) -> tuple[dict[str, object], ...]:
    """Produce deterministic unsigned records without conflating model families."""

    profiles = tuple(statistical_candidate_profile(model) for model in models)
    labels = tuple(
        dict(profile.support_tuples[0].attributes)["model_label"] for profile in profiles
    )
    if len(set(labels)) != len(labels):
        raise ValueError("Statistical profile inventory contains duplicate model labels.")
    return tuple(
        statistical_profile_record(profile)
        for profile in sorted(profiles, key=lambda value: value.profile_id)
    )


def _gate(
    name: str,
    evidence: Sequence[QualificationEvidence],
    /,
    *,
    at_time: int,
    failed: Sequence[str] = (),
    inconclusive: Sequence[str] = (),
) -> dict[str, object]:
    values = tuple(evidence)
    if any(not isinstance(value, QualificationEvidence) for value in values):
        raise TypeError("Gate evidence must contain QualificationEvidence values.")
    if any(value.evidence_kind not in _GATE_KINDS[name] for value in values):
        raise ValueError(f"{name} gate received evidence of the wrong kind.")
    failures = list(str(value) for value in failed)
    gaps = list(str(value) for value in inconclusive)
    for value in values:
        if not value.is_current(at_time):
            gaps.append(f"not-current:{value.evidence_id}")
        elif value.failed:
            failures.append(f"failed:{value.evidence_id}:{value.reason}")
        elif value.inconclusive:
            gaps.append(f"inconclusive:{value.evidence_id}:{value.reason}")
    if not values:
        gaps.append("missing-evidence")
    outcome = "failed" if failures else "inconclusive" if gaps else "passed"
    return {
        "gate": name,
        "outcome": outcome,
        "evidence_ids": sorted(value.evidence_id for value in values),
        "failed_reasons": sorted(set(failures)),
        "inconclusive_reasons": sorted(set(gaps)),
    }


def _resource_failures(
    model: AbstractInteractionModel | StatisticalDynamicsPlan,
    nilss: NILSSPlan | None,
    measurements: Mapping[str, int | float],
    /,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, float]]:
    normalized: dict[str, float] = {}
    for name, value in measurements.items():
        value_ = float(value)
        if not math.isfinite(value_) or value_ < 0.0:
            raise ValueError("Resource measurements must be finite and non-negative.")
        normalized[str(name)] = value_
    limits: dict[str, int] = {}
    if isinstance(model, StatisticalDynamicsPlan):
        limits.update(
            state_bytes=model.maximum_state_bytes,
            workspace_bytes=model.maximum_workspace_bytes,
        )
    if nilss is not None:
        limits.update(
            retained_bytes=nilss.maximum_retained_bytes,
            nilss_workspace_bytes=nilss.maximum_workspace_bytes,
        )
    failed = tuple(
        f"resource-budget:{name}:{normalized[name]}>{limit}"
        for name, limit in sorted(limits.items())
        if name in normalized and normalized[name] > limit
    )
    missing = tuple(
        f"missing-resource-measurement:{name}"
        for name in sorted(limits)
        if name not in normalized
    )
    return failed, missing, normalized


def build_statistical_dynamics_candidate(
    profile: CapabilityProfile,
    model: AbstractInteractionModel | StatisticalDynamicsPlan,
    run_spec: ResolvedRunSpec,
    evidence: Sequence[QualificationEvidence],
    /,
    *,
    at_time: int,
    resource_measurements: Mapping[str, int | float],
    distributed_layout: DistributedStatisticalLayout | None = None,
    restart_relation: DistributedRestartRelation | None = None,
    nilss: NILSSPlan | None = None,
    model_label: str | None = None,
    reference_manifests: Sequence[ReferenceArtifactManifest] = (),
) -> dict[str, object]:
    """Build one exact model candidate with isolated resource/restart gates."""

    label = _validate_route(model, nilss, model_label)
    expected_support = statistical_support_tuple(
        model,
        distributed_layout=distributed_layout,
        nilss=nilss,
        model_label=label,
    )
    if not isinstance(profile, CapabilityProfile):
        raise TypeError("profile must be a CapabilityProfile.")
    if (
        not profile.supports(expected_support)
        or profile.released
        or profile.release_evidence
    ):
        raise ValueError("Profile must be the matching unsigned statistical candidate.")
    if any(not isinstance(value, SupportDependency) for value in profile.dependencies):
        raise ValueError("Statistical qualification requires exact support dependencies.")
    if not isinstance(run_spec, ResolvedRunSpec):
        raise TypeError("run_spec must be a ResolvedRunSpec.")
    if restart_relation is not None and not isinstance(
        restart_relation, DistributedRestartRelation
    ):
        raise TypeError("restart_relation must be DistributedRestartRelation or None.")
    if distributed_layout is None and restart_relation is not None:
        raise ValueError(
            "A distributed restart relation requires a distributed target layout."
        )
    references = tuple(reference_manifests)
    if any(not isinstance(value, ReferenceArtifactManifest) for value in references):
        raise TypeError(
            "reference_manifests must contain ReferenceArtifactManifest values."
        )
    for reference in references:
        reference.require_uncertainty()
    evidence_ = tuple(evidence)
    if any(not isinstance(value, QualificationEvidence) for value in evidence_):
        raise TypeError("evidence must contain QualificationEvidence values.")
    if len({value.evidence_id for value in evidence_}) != len(evidence_):
        raise ValueError("Qualification evidence IDs must be unique.")

    grouped = {
        name: tuple(value for value in evidence_ if value.evidence_kind in kinds)
        for name, kinds in _GATE_KINDS.items()
    }
    categorized_ids = {
        value.evidence_id for values in grouped.values() for value in values
    }
    if categorized_ids != {value.evidence_id for value in evidence_}:
        raise ValueError(
            "Statistical candidate evidence must use scientific/reference, "
            "performance, operational, or security kinds."
        )
    model_id = _model_id(model)
    subject_failures = {
        name: tuple(
            f"model-subject-mismatch:{value.evidence_id}"
            for value in values
            if model_id not in value.subject_ids
        )
        for name, values in grouped.items()
    }
    resource_failed, resource_gaps, resource_values = _resource_failures(
        model, nilss, resource_measurements
    )
    operational_failures: list[str] = []
    operational_gaps: list[str] = []
    restart_record: dict[str, object] | None = None
    if distributed_layout is not None:
        if restart_relation is None:
            operational_gaps.append("missing-distributed-restart-evidence")
        else:
            if restart_relation.target.topology_id != distributed_layout.topology_id:
                operational_failures.append("restart-target-topology-mismatch")
            if not restart_relation.accepted:
                operational_failures.append("restart-changes-statistical-semantics")
            restart_record = {
                "kind": "distributed-statistical-restart-evidence",
                "relation_id": restart_relation.relation_id,
                "source_semantic_layout_id": restart_relation.source_semantic_layout_id,
                "target_semantic_layout_id": restart_relation.target_semantic_layout_id,
                "source_topology_id": restart_relation.source_topology_id,
                "target_topology_id": restart_relation.target_topology_id,
                "topology_changed": restart_relation.topology_changed,
                "accepted": restart_relation.accepted,
            }
    exact_dependency = SupportDependency(
        profile.profile_id, expected_support.support_tuple_id
    )
    run_dependencies = (
        *run_spec.scientific_dependencies,
        *run_spec.deployment_dependencies,
    )
    run_dependency_ids = frozenset(value.dependency_id for value in run_dependencies)
    operational_failures.extend(
        f"resolved-run-missing-profile-dependency:{value.dependency_id}"
        for value in profile.dependencies
        if value.dependency_id not in run_dependency_ids
    )
    if all(
        value.dependency_id != exact_dependency.dependency_id
        for value in run_dependencies
    ):
        operational_failures.append("resolved-run-missing-exact-statistical-support")
    security_failures = tuple(
        f"reference:{manifest.manifest_id}:{reason}"
        for manifest in references
        for reason in manifest.rights_refusal_reasons(commercial_use=True)
    )
    gates = {
        "scientific": _gate(
            "scientific",
            grouped["scientific"],
            at_time=at_time,
            failed=subject_failures["scientific"],
        ),
        "performance": _gate(
            "performance",
            grouped["performance"],
            at_time=at_time,
            failed=(*resource_failed, *subject_failures["performance"]),
            inconclusive=resource_gaps,
        ),
        "operational": _gate(
            "operational",
            grouped["operational"],
            at_time=at_time,
            failed=(*operational_failures, *subject_failures["operational"]),
            inconclusive=operational_gaps,
        ),
        "security": _gate(
            "security",
            grouped["security"],
            at_time=at_time,
            failed=(*security_failures, *subject_failures["security"]),
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
    model_record: dict[str, object]
    if isinstance(model, StatisticalDynamicsPlan):
        model_record = {
            "plan_id": model.plan_id,
            "closure": model.closure,
            "interaction_model": model.interaction_model,
            "closure_exact": model.closure_exact,
            "exactness": model.exactness,
            "layout_id": model.layout.layout_id,
            "dynamics_id": model.dynamics.dynamics_id,
            "forcing_covariance_id": model.forcing.covariance_id,
        }
    else:
        model_record = {
            "model_id": model.model_id,
            "interaction_kind": model.kind,
        }
    topology_record = None
    if distributed_layout is not None:
        topology_record = {
            "topology_id": distributed_layout.topology_id,
            "semantic_layout_id": distributed_layout.semantic_layout_id,
            "batch_topology_id": distributed_layout.batch.topology_id,
            "covariance_topology_id": distributed_layout.covariance.topology_id,
            "process_count": distributed_layout.batch.process_count,
            "covariance_storage": distributed_layout.covariance.storage,
        }
    core = {
        "kind": "statistical-dynamics-qualification-candidate",
        "model_label": label,
        "support_tuple": expected_support.to_record(),
        "profile": statistical_profile_record(profile),
        "model": model_record,
        "nilss": None
        if nilss is None
        else {
            "plan_id": nilss.plan_id,
            "state_dimension": nilss.state_dimension,
            "unstable_dimension": nilss.unstable_dimension,
            "horizon_steps": nilss.horizon_steps,
            "maximum_retained_bytes": nilss.maximum_retained_bytes,
            "maximum_workspace_bytes": nilss.maximum_workspace_bytes,
        },
        "resources": resource_values,
        "topology": topology_record,
        "restart": restart_record,
        "resolved_run_spec": run_spec.to_record(),
        "support_dependencies": [value.to_record() for value in run_dependencies],
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


def write_candidate(path: str | Path, candidate: Mapping[str, object], /) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(candidate, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
