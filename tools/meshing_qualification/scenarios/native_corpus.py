# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, final

import phydrax as phx
from phydrax import canonical_fingerprint
from tools._meshing_cases import (
    load_corpus_registration,
    MANDATORY_WORKFLOWS,
    POLYHEDRAL_CASE_NAMES,
)
from tools.meshing_qualification._campaign import (
    _native_campaign_sample,
    _native_driver_controls,
    CampaignSample,
)
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _corpus_profiles, _mandatory_result
from tools.meshing_qualification._runtime import _runtime_identity


def _corpus_phase_gaps(
    sample: CampaignSample, *, full_workflow: bool, refusal: bool
) -> list[str]:
    epochs = sample.epochs
    phase_missing = []
    for index, epoch in enumerate(epochs):
        compiler = epoch.get("compiler", {})
        for field in (
            "temporary_bytes",
            "output_bytes",
            "generated_code_bytes",
            "warm_samples_seconds",
        ):
            if (
                compiler.get(field) is None
                or field == "warm_samples_seconds"
                and not compiler[field]
            ):
                phase_missing.append(f"epoch-{index}:{field}")
        if epoch.get("retained_bytes") is None:
            phase_missing.append(f"epoch-{index}:logical_retained_bytes")
    if not epochs and full_workflow and not refusal:
        phase_missing.append("accepted-epoch-phase-measurements")
    return phase_missing


def _corpus_resource_gaps(
    sample: CampaignSample, *, full_workflow: bool, refusal: bool
) -> list[str]:
    publications = []
    if sample.generation is not None:
        publications.append(sample.generation)
    publications.extend(
        epoch["result"]
        for epoch in sample.epochs
        if isinstance(epoch.get("result"), dict)
    )
    checkpoint = sample.checkpoint
    if isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("continued_result"), dict
    ):
        publications.append(checkpoint["continued_result"])
    resource_missing = []
    required_limits = {
        "vertices",
        "edges",
        "faces",
        "cells",
        "connectivity_entries",
        "data_bytes",
        "scratch_bytes",
        "work_units",
        "wall_seconds",
        "cavity_cells",
        "geometry_queries",
    }
    for index, publication in enumerate(publications):
        guarantees = publication.get("resource_guarantees", {})
        resource_missing.extend(
            f"publication-{index}:missing-{name}"
            for name in sorted(required_limits - set(guarantees.get("enforced", ())))
        )
        resource_missing.extend(
            f"publication-{index}:unenforced-{name}"
            for name in guarantees.get("unenforced", ())
        )
        receipt = publication.get("execution_evidence")
        if not isinstance(receipt, dict) or not receipt.get("phases"):
            resource_missing.append(f"publication-{index}:actual-ended-native-receipt")
        elif any(phase.get("status_name") != "OK" for phase in receipt["phases"]):
            resource_missing.append(f"publication-{index}:native-phase-status")
    if not publications and full_workflow and not refusal:
        resource_missing.append("accepted-publication-runtime-and-receipt")
    return resource_missing


def _corpus_profile_assessment(
    profile: dict[str, Any],
    sample: CampaignSample | None,
) -> dict[str, Any]:
    """Keep execution, scientific, workflow and release facts distinct."""
    if sample is None:
        return {
            "execution": "not-run",
            "scientific_certification": "unassessed",
            "mandatory_positive_completion": "incomplete",
            "positive_completion": False,
            "refusal_certified": False,
            "qualification_eligible": False,
            "release": "unassessed",
            "leadership": "unassessed",
            "blocking_status": "not-run",
        }
    matched = sample.status == profile["expected_outcome"]
    refusal = profile["expected_outcome"] == "expected-refusal"
    refusal_evidence = (
        matched
        and refusal
        and (
            sample.failure_evidence_id is not None
            and bool(sample.failure_evidence_id)
            and sample.requested is not None
            and sample.achieved is not None
            and sample.source_bits_preserved is True
        )
    )
    full_workflow = profile.get("full_workflow", True) is True
    phase_missing = _corpus_phase_gaps(
        sample, full_workflow=full_workflow, refusal=refusal
    )
    resource_missing = _corpus_resource_gaps(
        sample, full_workflow=full_workflow, refusal=refusal
    )
    acceptance = sample.independent_acceptance
    scientific = sample.independent_scientific_certification == "passed" or (
        acceptance is not None and acceptance.get("passed") is True
    )
    completion = sample.mandatory_workflow_completion
    case_completion = sample.case_contract_completion
    positive_case = (
        not refusal
        and matched
        and scientific
        and case_completion == "complete"
        and not phase_missing
        and not resource_missing
    )
    eligible = refusal_evidence or positive_case
    mandatory_positive = positive_case and full_workflow and completion == "complete"
    return {
        "deliberate_refusal_certification": (
            "passed" if refusal_evidence else "not-established"
        ),
        "refusal_certified": refusal_evidence,
        "phase_measurement_gaps": phase_missing,
        "resource_evidence_gaps": resource_missing,
        "positive_case_completion": case_completion,
        "case_certified": positive_case,
        "positive_completion": mandatory_positive,
        "execution": sample.status,
        "expected_outcome_matched": matched,
        "scientific_certification": "passed" if scientific else "unassessed",
        "mandatory_positive_completion": completion,
        "qualification_eligible": eligible,
        "release": sample.release_evidence,
        "leadership": sample.leadership_evidence,
        "blocking_status": None if eligible else sample.status,
    }


@final
class _CorpusCampaign(phx.StrictModule):
    selected: tuple[str, ...]
    samples: list[dict[str, Any]]
    by_profile: dict[str, CampaignSample]
    missing_inputs: dict[str, list[str]]


def _execute_corpus_profiles(
    args: QualificationControls, profiles: dict[str, dict[str, Any]]
) -> _CorpusCampaign:
    arguments = args.to_record()
    requested_profiles = args.corpus_profile
    selected = (
        tuple(profiles) if requested_profiles is None else tuple(requested_profiles)
    )
    unknown = set(selected) - profiles.keys()
    if unknown:
        raise ValueError(
            "native_corpus.corpus_profile must name declared profiles; "
            f"got unknown profiles {sorted(unknown)}"
        )
    if len(selected) != len(set(selected)):
        raise ValueError(
            f"native_corpus.corpus_profile must contain unique profiles; got {selected!r}"
        )
    samples = []
    by_profile = {}
    missing_inputs = {}
    for name in selected:
        profile = profiles[name]
        missing = [
            key for key in profile.get("prerequisites", ()) if arguments.get(key) is None
        ]
        if missing:
            missing_inputs[name] = missing
            continue
        scenario = profile["scenario"]
        controls = {} if scenario is None else _native_driver_controls(scenario, args)
        controls.update(profile["controls"])
        rounds = (
            0
            if name in POLYHEDRAL_CASE_NAMES and name != "polyhedral-periodic-cube"
            else args.adaptation_rounds
        )
        with (
            TemporaryDirectory(prefix="phydrax-corpus-distributed-")
            if scenario == "native-distributed-lifecycle"
            and "distributed_checkpoint_root" not in controls
            else nullcontext()
        ) as checkpoint_directory:
            if checkpoint_directory is not None:
                controls["distributed_checkpoint_root"] = str(
                    Path(checkpoint_directory) / "checkpoint"
                )
            sample = _native_campaign_sample(
                profile["case_name"],
                args.resolution,
                args.capacity,
                args.timeout,
                args.repeats,
                args.target_error,
                rounds,
                "native",
                scenario_override=scenario,
                controls=controls,
            )
        sample_record = {
            **sample.to_record(),
            "corpus_profile": name,
            "mandatory_workflow": profile["workflow"],
        }
        samples.append(sample_record)
        by_profile[name] = sample
    return _CorpusCampaign(
        selected=selected,
        samples=samples,
        by_profile=by_profile,
        missing_inputs=missing_inputs,
    )


def _corpus_workflow_results(
    profiles: dict[str, dict[str, Any]], mandatory_results: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    workflow_results = {}
    for workflow in MANDATORY_WORKFLOWS:
        names = sorted(
            name for name, profile in profiles.items() if profile["workflow"] == workflow
        )
        positive_names = [
            name
            for name in names
            if profiles[name]["expected_outcome"] == "passed"
            and profiles[name].get("full_workflow", True) is True
        ]
        case_names = [
            name
            for name in names
            if profiles[name]["expected_outcome"] == "passed"
            and profiles[name].get("full_workflow", True) is not True
        ]
        completed = [
            name
            for name in positive_names
            if mandatory_results[name]["positive_completion"] is True
        ]
        cases_certified = [
            name
            for name in case_names
            if mandatory_results[name]["case_certified"] is True
        ]
        refusals = [
            name for name in names if mandatory_results[name]["refusal_certified"] is True
        ]
        workflow_results[workflow] = {
            "completion": (
                "complete"
                if positive_names
                and len(completed) == len(positive_names)
                and len(cases_certified) == len(case_names)
                else "incomplete"
            ),
            "positive_profiles": positive_names,
            "positive_profiles_completed": completed,
            "routine_case_profiles": case_names,
            "routine_case_profiles_certified": cases_certified,
            "certified_refusal_profiles": refusals,
            "refusals_count_as_positive_completion": False,
        }
    return workflow_results


def _corpus_outcome_counts(
    statuses: list[str], mandatory_results: dict[str, dict[str, Any]]
) -> dict[str, int]:
    outcome_names = (
        "passed",
        "expected-refusal",
        "failed",
        "timeout",
        "missing-dependency",
        "ineligible-comparison",
    )
    counts = {status: statuses.count(status) for status in outcome_names}
    counts.update(
        {
            status: statuses.count(status)
            for status in sorted(set(statuses) - set(outcome_names))
        }
    )
    counts["positive_completed"] = sum(
        result["positive_completion"] is True for result in mandatory_results.values()
    )
    counts["case_certified"] = sum(
        result["case_certified"] is True for result in mandatory_results.values()
    )
    counts["refusal_certified"] = sum(
        result["refusal_certified"] is True for result in mandatory_results.values()
    )
    return counts


def qualify_native_corpus(args: QualificationControls) -> dict[str, object]:
    profiles = _corpus_profiles()
    campaign = _execute_corpus_profiles(args, profiles)
    selected, samples, by_profile, missing_inputs = (
        campaign.selected,
        campaign.samples,
        campaign.by_profile,
        campaign.missing_inputs,
    )
    statuses = [sample.status for sample in by_profile.values()]
    assessments = {
        name: _corpus_profile_assessment(profile, by_profile.get(name))
        for name, profile in profiles.items()
    }
    mandatory_results = {
        name: _mandatory_result(name, profiles[name], assessments[name])
        for name in sorted(profiles)
    }
    workflow_results = _corpus_workflow_results(profiles, mandatory_results)
    counts = _corpus_outcome_counts(statuses, mandatory_results)
    registration = load_corpus_registration()
    record: dict[str, object] = {
        "kind": "native-meshing-corpus-result",
        "scenario": "native-corpus",
        "runtime": _runtime_identity(),
        "corpus_registration_id": registration["registration_id"],
        "samples": samples,
        "counts": counts,
        "mandatory_workflow_results": mandatory_results,
        "mandatory_workflows": workflow_results,
        "mandatory_workflows_incomplete": [
            workflow
            for workflow, result in workflow_results.items()
            if result["completion"] != "complete"
        ],
        "selected_profiles": list(selected),
        "selected_profiles_missing_inputs": missing_inputs,
        "unqualified_selected_profiles": [
            name for name in selected if not assessments[name]["qualification_eligible"]
        ],
        "qualification_dependencies": {
            name: {
                **profile,
                **assessments[name],
                "missing_original_source_inputs": missing_inputs.get(name, []),
                "prerequisites": [
                    "current native source/binary/configuration identity",
                    "all original root resource guarantees",
                    "independent geometry/semantics/physical/state acceptance",
                    "original capacity phase/scaling campaign",
                ],
            }
            for name, profile in profiles.items()
        },
        "reviewed_hybrid_layer_sources": registration["hybrid_layer_sources"],
        "preserved_source_incompatibilities": {
            "original-curved-layer-exact-period-x1": {
                "disposition": "immutable-negative-regression",
                "source_id": "curved-periodic-narrow-gap",
                "source_revision": (
                    "fadef789744332ba906c2bf1b82231d331e7cc7f4128997191b2d97cb9301e4f"
                ),
                "required_period_x": 1,
                "original_top_trace_x_residuals": (
                    "-101121/2361183241434822606848",
                    "-6620711477/77371252455336267181195264",
                ),
                "proof_origin": "tests/data/meshing/curved_periodic_sources.json",
                "reviewed_record": "tests/data/meshing/curved_periodic_sources.json",
                "evidence_scope": (
                    "Original mathematical source incompatibility; the corrected "
                    "positive owns a different source ID, digest, and revision."
                ),
                "rerun_profile": "original-curved-periodic-refusal",
            },
        },
        "research_gates": {
            "general-all-hex": "incomplete",
            "singular-cad-arrangements": "incomplete",
            "underspecified-defective-surface-reconstruction": "incomplete",
        },
        "release": "unassessed",
        "leadership": "unassessed",
    }
    return {**record, "result_id": canonical_fingerprint(record)}


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_native_corpus(args)
