#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Separate numerical implementation closure from rights-blocked scientific release."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from ._closure_matrix import CapabilityClosureMatrix
from ._closure_requirement import CapabilityClosureRequirement, CapabilityGapResolution
from ._closure_taxonomy import (
    CapabilityDepth,
    CarrierRepresentation,
    ClosureDisposition,
    CouplingLocation,
    ExecutionRegime,
    PhysicsField,
    SourceReuseClass,
    TopologyRegime,
    WorkflowClass,
)
from ._registry import CapabilityProfile, SupportTuple
from ._source_reference import PublicationReference, SourceReference, SourceReview


class FuelVerificationCase(NamedTuple):
    case_id: str
    name: str
    reference: str
    metric: str


FUEL_VERIFICATION_CASES = (
    FuelVerificationCase(
        "V1",
        "constant-cylinder-conduction",
        "analytic-cylinder",
        "temperature-error-and-rate",
    ),
    FuelVerificationCase(
        "V2",
        "nonlinear-cylinder-conduction",
        "kirchhoff-transform",
        "temperature-error-and-rate",
    ),
    FuelVerificationCase(
        "V3",
        "lame-generalized-plane-strain",
        "lame-closed-end-tube",
        "stress-and-axial-resultant",
    ),
    FuelVerificationCase(
        "V4", "finite-strain-inflation", "rivlin-tube", "pressure-stretch"
    ),
    FuelVerificationCase(
        "V5", "free-thermal-expansion", "stress-free-eigenstrain", "stress"
    ),
    FuelVerificationCase(
        "V6", "shrink-fit-contact", "lame-shrink-fit", "contact-pressure"
    ),
    FuelVerificationCase(
        "V7", "booth-release", "booth-sphere-series", "released-fraction"
    ),
    FuelVerificationCase(
        "V8", "piecewise-temperature-gas", "communicating-volume-ideal-gas", "pressure"
    ),
    FuelVerificationCase(
        "V9",
        "paired-gap-heat-exchange",
        "concentric-cylinder-heat-flow",
        "heat-flux-and-pair-cancellation",
    ),
    FuelVerificationCase(
        "V10",
        "constant-stress-creep",
        "crepr-closed-form-and-norton-tube",
        "creep-strain-and-tube-equilibrium",
    ),
    FuelVerificationCase(
        "V11",
        "rz-thermomechanics-mms",
        "manufactured-independent-forcing",
        "spatial-and-temporal-rates",
    ),
    FuelVerificationCase(
        "V12", "layered-rz-comparison", "independent-reductions", "midlength-deviation"
    ),
    FuelVerificationCase(
        "V13",
        "whole-history-balances",
        "energy-and-gas-conservation",
        "maximum-relative-imbalance",
    ),
    FuelVerificationCase(
        "V14",
        "fault-rollback-and-restart",
        "accepted-checkpoint-replay",
        "bitwise-state-equality",
    ),
    FuelVerificationCase(
        "V15",
        "whole-history-refinement",
        "richardson",
        "fgr-temperature-pressure-od-convergence",
    ),
)


def nuclear_fuel_release_matrix_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    from ..applications.nuclear_fuel._qualification import nuclear_fuel_profiles

    member = nuclear_fuel_profiles()[0]
    return (
        CapabilityProfile(
            "nuclear-fuel.release-matrix.candidate",
            "phydrax",
            "candidate",
            (
                SupportTuple(
                    "nuclear-fuel.release-matrix",
                    {row.case_id: row.name for row in FUEL_VERIFICATION_CASES},
                ),
            ),
            dependencies=(member.profile_id,),
            required_gates=member.required_gates,
            released=False,
        ),
    )


def nuclear_fuel_closure_matrices(
    verification_evidence: Mapping[str, Sequence[str]] | None = None,
    /,
    *,
    fault_adequacy_evidence: Mapping[str, Sequence[str]] | None = None,
) -> tuple[CapabilityClosureMatrix, ...]:
    """Construct native open closures; only observed campaign artifacts resolve cases.

    Numerical and deliberate-fault evidence are separate observed content
    addresses. A numerical pass alone never resolves the fault-adequacy
    requirement. Restricted data and locked evaluation remain missing even
    when both implementation evidence dimensions are complete.
    """
    observed = {} if verification_evidence is None else verification_evidence
    faults = {} if fault_adequacy_evidence is None else fault_adequacy_evidence
    known = {row.case_id for row in FUEL_VERIFICATION_CASES}
    for records in (observed, faults):
        if set(records) - known or any(not tuple(ids) for ids in records.values()):
            raise ValueError(
                "Verification evidence must name nonempty known V1–V15 cases."
            )
    if any(
        set(observed[name]) & set(faults[name]) for name in set(observed) & set(faults)
    ):
        raise ValueError(
            "Numerical and fault adequacy require distinct observed artifacts."
        )
    requirements: list[CapabilityClosureRequirement] = []
    resolutions: list[CapabilityGapResolution] = []
    scientific_requirements: list[CapabilityClosureRequirement] = []
    scientific_resolutions: list[CapabilityGapResolution] = []
    for name, reference, dimensions, source, scientific in (
        *(
            (
                row.case_id,
                row.reference,
                ("numerical", "fault-adequacy"),
                ("iaea-fumex-public-report",),
                False,
            )
            for row in FUEL_VERIFICATION_CASES
        ),
        (
            "ifpe-data-rights",
            "actual-originator-agreement",
            ("data-rights",),
            ("nea-ifpe-access-metadata",),
            True,
        ),
        (
            "independent-locked-evaluation",
            "ancestry-fabrication-heldout-rods",
            ("locked-evaluation",),
            ("nea-ifpe-access-metadata", "iaea-fumex-public-report"),
            True,
        ),
    ):
        numerical = tuple(observed.get(name, ()))
        fault = tuple(faults.get(name, ()))
        evidence = numerical + fault
        complete = bool(numerical) and bool(fault)
        requirement = CapabilityClosureRequirement.create(
            f"nuclear-fuel-{name}",
            minimum_depth=CapabilityDepth.SCIENTIFICALLY_VALIDATED
            if scientific
            else CapabilityDepth.SPATIAL_COUPLED,
            physical_fields=(
                PhysicsField.THERMAL,
                PhysicsField.SOLID_MECHANICS,
                PhysicsField.CHEMICAL_SPECIES,
            ),
            carriers=(
                CarrierRepresentation.CONTINUUM_VOLUME,
                CarrierRepresentation.INTERFACE_SURFACE,
            ),
            coupling_locations=(
                CouplingLocation.BULK,
                CouplingLocation.CONTACT,
                CouplingLocation.BOUNDARY,
            ),
            execution_regimes=(ExecutionRegime.TRANSIENT,),
            topology_regimes=(TopologyRegime.CONTACT,),
            workflow_classes=(WorkflowClass.QUALIFICATION, WorkflowClass.RESTART_REPLAY),
            required_benchmarks=(
                "tools/nuclear_fuel_qualification.py",
                "benchmarks/nuclear_fuel_performance.py",
            ),
            required_evidence_dimensions=dimensions,
            source_ids=source,
            rationale=reference,
        )
        (scientific_requirements if scientific else requirements).append(requirement)
        (scientific_resolutions if scientific else resolutions).append(
            CapabilityGapResolution.create(
                requirement.requirement_id,
                ClosureDisposition.IMPLEMENTED
                if complete
                else ClosureDisposition.MISSING,
                CapabilityDepth.SPATIAL_COUPLED if complete else CapabilityDepth.SEMANTIC,
                capability_ids=("nuclear-fuel.performance",),
                evidence_ids=evidence,
                rationale="observed-numerical-and-deliberate-fault-campaign"
                if complete
                else (
                    "qualified-external-blocker-actual-data-agreement-and-locked-evaluation-required"
                    if scientific
                    else "observed-numerical-fault-adequacy-required"
                    if numerical
                    else "observed-fault-numerical-verification-required"
                    if fault
                    else "assembled-runtime-verification-evidence-required"
                ),
                release_authorized=False,
            )
        )
    return (
        CapabilityClosureMatrix.create(
            "nuclear-fuel.implementation", requirements, resolutions
        ),
        CapabilityClosureMatrix.create(
            "nuclear-fuel.scientific-release",
            scientific_requirements,
            scientific_resolutions,
        ),
    )


def nuclear_fuel_source_references() -> tuple[SourceReference, ...]:
    """Pin reviewed public documents, never restricted experiment packages.

    FUMEX supplies the comparison/idealized-case protocol, not the individual
    analytic solutions in ``FUEL_VERIFICATION_CASES``. Rights-content digests
    identify extracted notices, separately from original HTML/PDF byte digests.
    Extraction recipes and the technical review scope are documented in
    ``docs/nuclear_fuel_sources.md``; neither record represents legal approval.
    """
    ifpe_url = "https://oecd-nea.org/tools/abstract/detail/nea-1664"
    request_url = "https://oecd-nea.org/jcms/pl_37221/international-fuel-performance-experiments-ifpe-request"
    fumex_url = "https://www-pub.iaea.org/mtcd/publications/pdf/te-1697_cd/pdf/iaea-tecdoc-1697.pdf"
    ifpe_digest = (
        "sha256:06dd6311afdb172400eff4b869a6f9065c2da8bfe257633eea8b9fb1ce4308bf"
    )
    request_digest = (
        "sha256:0f1d930b166b713c4623578a45b131412f99a1ac708693c346b389a6cd57bf75"
    )
    fumex_digest = (
        "sha256:e6efbf9cf532713743f4a1926fabdd78c06e833a672288e30bab81b5f069cd14"
    )
    fumex_rights_digest = "pdftotext-layout-utf8-sha256:9fbb783ab8f9789608c2f00cc3af415c838363c98bd0a1bedcd1ec8a3421fb52"
    return (
        SourceReference.create(
            "nea-ifpe-access-metadata",
            ifpe_url,
            "public-metadata-reviewed-2026-09-30-NEA-1664/03-status-2024-04-08",
            "OECD-copyright-no-copying-permission-established",
            SourceReuseClass.PROPRIETARY,
            concepts=(
                "integral-fuel-validation",
                "heldout-rod-evaluation",
                "restricted-data-access-policy",
            ),
            archive_digest=ifpe_digest,
            license_digest="utf8-excerpt-sha256:9d161f36fc51bceccb5cd5c4c1fcbfb6ef72ede32fbe0f0abe932a21e4b5f865",
            relevant_documents=(
                f"{ifpe_url}@{ifpe_digest}",
                f"{request_url}@{request_digest}",
            ),
            publications=(
                PublicationReference(
                    "IFPE DATABASE: public description and restricted status", ifpe_url
                ),
                PublicationReference(
                    "International Fuel Performance Experiments (IFPE) request",
                    request_url,
                ),
            ),
            data_rights="public-access-metadata-only-masterfiled-restricted-no-local-originator-agreement-evidence",
            technical_reviewer="phydrax-public-document-technical-review-2026-09-30",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "iaea-fumex-public-report",
            fumex_url,
            "IAEA-TECDOC-1697-2013-public-report-reviewed-2026-09-30",
            "IAEA-publication-copyright-permission-required",
            SourceReuseClass.PROPRIETARY,
            concepts=(
                "fuel-code-comparison",
                "simplified-priority-cases",
                "verification-protocol",
            ),
            archive_digest=fumex_digest,
            license_digest=fumex_rights_digest,
            relevant_documents=(
                f"{fumex_url}@{fumex_digest}",
                f"{fumex_url}#page=5@{fumex_rights_digest}",
            ),
            publications=(
                PublicationReference(
                    "Improvement of Computer Codes Used for Fuel Behaviour Simulation (FUMEX-III), "
                    "IAEA-TECDOC-1697: sections 1.2, 2.2 and 3.1.2",
                    fumex_url,
                ),
            ),
            data_rights="public-report-only-experimental-histories-not-acquired-input-and-redistribution-rights-unestablished",
            technical_reviewer="phydrax-public-document-technical-review-2026-09-30",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
    )


__all__ = [
    "FuelVerificationCase",
    "FUEL_VERIFICATION_CASES",
    "nuclear_fuel_closure_matrices",
    "nuclear_fuel_release_matrix_candidate_profiles",
    "nuclear_fuel_source_references",
]
