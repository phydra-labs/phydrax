#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed qualification for offline and deployed closure-data routes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax.closure_data._alignment import (
    ConservativeAlignmentResult,
    PreparedConservativeAlignment,
)
from phydrax.closure_data._analysis import ClosureAnalysisDAG, ClosureQualityReport
from phydrax.closure_data._binding import LearnedClosureBindingPlan
from phydrax.closure_data._dataset import (
    ChunkedClosureDatasetManifest,
    LeakageSafePartition,
    TrainOnlyNormalizer,
)
from phydrax.closure_data._filters import (
    FilterCommutationReport,
    FilterRefinementReport,
    PreparedFilter,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import QualificationEvidence, SupportDependency
from phydrax.qualification._reference import ReferenceArtifactManifest
from phydrax.qualification._registry import CapabilityProfile, SupportTuple


_GATE_KINDS = {
    "scientific": frozenset(("scientific", "reference")),
    "performance": frozenset(("performance",)),
    "operational": frozenset(("operational",)),
    "security": frozenset(("security",)),
}


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


def _validate_pipeline(
    prepared_filter: PreparedFilter,
    alignment: PreparedConservativeAlignment,
    dag: ClosureAnalysisDAG,
    dataset: ChunkedClosureDatasetManifest,
    partition: LeakageSafePartition,
    normalizer: TrainOnlyNormalizer | None,
    binding: LearnedClosureBindingPlan | None,
    /,
) -> None:
    if not isinstance(prepared_filter, PreparedFilter):
        raise TypeError("prepared_filter must be a PreparedFilter.")
    if not isinstance(alignment, PreparedConservativeAlignment):
        raise TypeError("alignment must be a PreparedConservativeAlignment.")
    if not isinstance(dag, ClosureAnalysisDAG):
        raise TypeError("dag must be a ClosureAnalysisDAG.")
    if not isinstance(dataset, ChunkedClosureDatasetManifest):
        raise TypeError("dataset must be a ChunkedClosureDatasetManifest.")
    if not isinstance(partition, LeakageSafePartition):
        raise TypeError("partition must be a LeakageSafePartition.")
    if normalizer is not None and not isinstance(normalizer, TrainOnlyNormalizer):
        raise TypeError("normalizer must be a TrainOnlyNormalizer or None.")
    if binding is not None and not isinstance(binding, LearnedClosureBindingPlan):
        raise TypeError("binding must be a LearnedClosureBindingPlan or None.")
    if dataset.analysis_dag_id != dag.dag_id:
        raise ValueError("Dataset manifest and analysis DAG identities differ.")
    if binding is not None and dataset.schema_id != binding.schema_id:
        raise ValueError("Dataset and deployed closure binding use different schemas.")


def closure_support_tuple(
    prepared_filter: PreparedFilter,
    alignment: PreparedConservativeAlignment,
    dag: ClosureAnalysisDAG,
    dataset: ChunkedClosureDatasetManifest,
    partition: LeakageSafePartition,
    /,
    *,
    normalizer: TrainOnlyNormalizer | None = None,
    binding: LearnedClosureBindingPlan | None = None,
) -> SupportTuple:
    """Create an exact offline or deployed support tuple from actual closure APIs."""

    _validate_pipeline(
        prepared_filter, alignment, dag, dataset, partition, normalizer, binding
    )
    route = (
        "offline-closure-data"
        if binding is None
        else f"deployed-{binding.deployment_kind}"
    )
    attributes: dict[str, str | int | bool] = {
        "route": route,
        "filter_kind": prepared_filter.spec.kind,
        "filter_id": prepared_filter.prepared_id,
        "alignment_kind": alignment.kind,
        "alignment_id": alignment.prepared_id,
        "analysis_dag_id": dag.dag_id,
        "dataset_manifest_id": dataset.manifest_id,
        "partition_level": partition.plan.level,
        "partition_id": partition.partition_id,
        "deployment": "offline" if binding is None else binding.deployment_kind,
    }
    if normalizer is not None:
        attributes["normalizer_id"] = normalizer.normalizer_id
    if binding is not None:
        attributes.update(
            {
                "binding_id": binding.binding_id,
                "model_artifact_id": binding.model_artifact_id,
                "differentiability": binding.differentiability,
            }
        )
    return SupportTuple("closure-data", attributes)


def closure_candidate_profile(
    prepared_filter: PreparedFilter,
    alignment: PreparedConservativeAlignment,
    dag: ClosureAnalysisDAG,
    dataset: ChunkedClosureDatasetManifest,
    partition: LeakageSafePartition,
    /,
    *,
    normalizer: TrainOnlyNormalizer | None = None,
    binding: LearnedClosureBindingPlan | None = None,
    dependencies: Sequence[SupportDependency] = (),
) -> CapabilityProfile:
    support = closure_support_tuple(
        prepared_filter,
        alignment,
        dag,
        dataset,
        partition,
        normalizer=normalizer,
        binding=binding,
    )
    return CapabilityProfile(
        "closure-data.candidate",
        "phydrax",
        "candidate",
        (support,),
        dependencies=tuple(dependencies),
        required_gates=("operational", "performance", "scientific", "security"),
        release_evidence=(),
        released=False,
    )


def closure_profile_record(profile: CapabilityProfile, /) -> dict[str, object]:
    if not isinstance(profile, CapabilityProfile):
        raise TypeError("profile must be a CapabilityProfile.")
    if profile.released or profile.release_evidence:
        raise ValueError("Closure qualification profiles must remain unreleased.")
    return {
        **profile.to_record(),
        "signed": False,
        "release_ready": False,
    }


def _partition_failures(
    partition: LeakageSafePartition,
    normalizer: TrainOnlyNormalizer | None,
    binding: LearnedClosureBindingPlan | None,
    /,
) -> tuple[str, ...]:
    failures: list[str] = []
    group_splits: dict[tuple[str, ...], str] = {}
    for assignment in partition.assignments:
        previous = group_splits.setdefault(assignment.group_key, assignment.split)
        if previous != assignment.split:
            failures.append(f"leakage-group-crosses-splits:{assignment.group_key}")
    if binding is not None and normalizer is None:
        failures.append("deployed-closure-missing-train-only-normalizer")
    if normalizer is not None:
        provenance = normalizer.provenance
        if provenance.partition_id != partition.partition_id:
            failures.append("normalizer-partition-mismatch")
        pairs = zip(
            provenance.training_assignment_ids,
            provenance.training_sample_ids,
            strict=True,
        )
        by_id = {value.assignment_id: value for value in partition.assignments}
        for assignment_id, sample_id in pairs:
            assignment = by_id.get(assignment_id)
            if assignment is None or assignment.sample_id != sample_id:
                failures.append(
                    f"normalizer-training-assignment-mismatch:{assignment_id}"
                )
            elif assignment.split != "train":
                failures.append(f"normalizer-used-nontraining-sample:{sample_id}")
        if binding is not None and (
            binding.normalizer_provenance_id != provenance.provenance_id
        ):
            failures.append("binding-normalizer-provenance-mismatch")
    return tuple(failures)


def build_closure_data_candidate(
    profile: CapabilityProfile,
    prepared_filter: PreparedFilter,
    alignment: PreparedConservativeAlignment,
    dag: ClosureAnalysisDAG,
    dataset: ChunkedClosureDatasetManifest,
    partition: LeakageSafePartition,
    quality: ClosureQualityReport,
    run_spec: ResolvedRunSpec,
    evidence: Sequence[QualificationEvidence],
    /,
    *,
    at_time: int,
    normalizer: TrainOnlyNormalizer | None = None,
    binding: LearnedClosureBindingPlan | None = None,
    alignment_result: ConservativeAlignmentResult | None = None,
    commutation_reports: Sequence[FilterCommutationReport | FilterRefinementReport] = (),
    reference_manifests: Sequence[ReferenceArtifactManifest] = (),
) -> dict[str, object]:
    """Bind filter, alignment, DAG, dataset, leakage, and deployment evidence."""

    _validate_pipeline(
        prepared_filter, alignment, dag, dataset, partition, normalizer, binding
    )
    if not isinstance(profile, CapabilityProfile):
        raise TypeError("profile must be a CapabilityProfile.")
    expected_support = closure_support_tuple(
        prepared_filter,
        alignment,
        dag,
        dataset,
        partition,
        normalizer=normalizer,
        binding=binding,
    )
    if (
        not profile.supports(expected_support)
        or profile.released
        or profile.release_evidence
    ):
        raise ValueError("Profile must be the matching unsigned closure candidate.")
    if any(not isinstance(value, SupportDependency) for value in profile.dependencies):
        raise ValueError("Closure qualification requires exact support dependencies.")
    if not isinstance(quality, ClosureQualityReport):
        raise TypeError("quality must be a ClosureQualityReport.")
    if not isinstance(run_spec, ResolvedRunSpec):
        raise TypeError("run_spec must be a ResolvedRunSpec.")
    if alignment_result is not None and not isinstance(
        alignment_result, ConservativeAlignmentResult
    ):
        raise TypeError("alignment_result must be ConservativeAlignmentResult or None.")
    reports = tuple(commutation_reports)
    if any(
        not isinstance(value, (FilterCommutationReport, FilterRefinementReport))
        for value in reports
    ):
        raise TypeError("commutation_reports contain an unsupported report.")
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
            "Closure candidate evidence must use scientific/reference, performance, "
            "operational, or security kinds."
        )
    subject_id = dataset.manifest_id if binding is None else binding.binding_id
    scientific_failures = (
        [] if quality.passed else [f"closure-quality:{quality.report_id}"]
    )
    for value in grouped["scientific"]:
        if subject_id not in value.subject_ids:
            scientific_failures.append(f"route-subject-mismatch:{value.evidence_id}")
    if (
        alignment_result is not None
        and alignment_result.alignment_id != alignment.prepared_id
    ):
        scientific_failures.append("alignment-evidence-plan-mismatch")
    for report in reports:
        if isinstance(report, FilterCommutationReport):
            if report.filter_id != prepared_filter.prepared_id:
                scientific_failures.append(
                    f"filter-commutation-plan-mismatch:{report.report_id}"
                )
        elif prepared_filter.prepared_id not in (
            report.fine_filter_id,
            report.coarse_filter_id,
        ):
            scientific_failures.append(
                f"filter-refinement-plan-mismatch:{report.report_id}"
            )

    security_failures = list(_partition_failures(partition, normalizer, binding))
    for manifest in references:
        security_failures.extend(
            f"reference:{manifest.manifest_id}:{reason}"
            for reason in manifest.rights_refusal_reasons(
                commercial_use=True,
                training_use=True,
            )
        )
    exact_dependency = SupportDependency(
        profile.profile_id, expected_support.support_tuple_id
    )
    run_dependencies = (
        *run_spec.scientific_dependencies,
        *run_spec.deployment_dependencies,
    )
    operational_failures = []
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
        operational_failures.append("resolved-run-missing-exact-closure-support")
    for value in grouped["operational"]:
        if subject_id not in value.subject_ids:
            operational_failures.append(f"route-subject-mismatch:{value.evidence_id}")

    gates = {
        "scientific": _gate(
            "scientific",
            grouped["scientific"],
            at_time=at_time,
            failed=scientific_failures,
        ),
        "performance": _gate(
            "performance",
            grouped["performance"],
            at_time=at_time,
            failed=tuple(
                f"route-subject-mismatch:{value.evidence_id}"
                for value in grouped["performance"]
                if subject_id not in value.subject_ids
            ),
        ),
        "operational": _gate(
            "operational",
            grouped["operational"],
            at_time=at_time,
            failed=operational_failures,
        ),
        "security": _gate(
            "security",
            grouped["security"],
            at_time=at_time,
            failed=(
                *security_failures,
                *(
                    f"route-subject-mismatch:{value.evidence_id}"
                    for value in grouped["security"]
                    if subject_id not in value.subject_ids
                ),
            ),
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
    route = (
        "offline-closure-data"
        if binding is None
        else f"deployed-{binding.deployment_kind}"
    )
    core = {
        "kind": "closure-data-qualification-candidate",
        "route": route,
        "support_tuple": expected_support.to_record(),
        "profile": closure_profile_record(profile),
        "pipeline": {
            "filter": {
                "spec_id": prepared_filter.spec.spec_id,
                "prepared_id": prepared_filter.prepared_id,
                "kind": prepared_filter.spec.kind,
                "spatial_shape": list(prepared_filter.spatial_shape),
            },
            "alignment": {
                "plan_id": alignment.plan.plan_id,
                "prepared_id": alignment.prepared_id,
                "kind": alignment.kind,
                "source_shape": list(alignment.source_shape),
                "target_shape": list(alignment.target_shape),
                "result_id": None
                if alignment_result is None
                else alignment_result.alignment_id,
            },
            "analysis_dag": {
                "dag_id": dag.dag_id,
                "external_input_ids": list(dag.external_input_ids),
                "node_ids": [value.node_id for value in dag.nodes],
            },
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "manifest_id": dataset.manifest_id,
                "schema_id": dataset.schema_id,
                "analysis_dag_id": dataset.analysis_dag_id,
                "extent_ids": [value.extent_id for value in dataset.extents],
                "chunk_ids": [value.chunk_id for value in dataset.chunks],
            },
            "partition": {
                "partition_id": partition.partition_id,
                "plan_id": partition.plan.plan_id,
                "level": partition.plan.level,
                "assignment_ids": [
                    value.assignment_id for value in partition.assignments
                ],
            },
            "normalizer": None
            if normalizer is None
            else {
                "normalizer_id": normalizer.normalizer_id,
                "provenance_id": normalizer.provenance.provenance_id,
                "partition_id": normalizer.provenance.partition_id,
                "training_sample_ids": list(normalizer.provenance.training_sample_ids),
            },
            "binding": None
            if binding is None
            else {
                "binding_id": binding.binding_id,
                "deployment_kind": binding.deployment_kind,
                "schema_id": binding.schema_id,
                "model_artifact_id": binding.model_artifact_id,
                "normalizer_provenance_id": binding.normalizer_provenance_id,
                "differentiability": binding.differentiability,
            },
        },
        "quality": {
            "report_id": quality.report_id,
            "target_ids": list(quality.target_ids),
            "sample_count": quality.sample_count,
            "nonfinite_count": quality.nonfinite_count,
            "finite_fraction": float(np.asarray(quality.finite_fraction)),
            "rms": float(np.asarray(quality.rms)),
            "maximum_absolute": float(np.asarray(quality.maximum_absolute)),
            "passed": quality.passed,
        },
        "commutation_report_ids": sorted(value.report_id for value in reports),
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
