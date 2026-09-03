#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic commercial qualification candidates for lattice Boltzmann routes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization.lattice_boltzmann._commercial_qualification import (
    c0_guo_baseline_profiles,
    c1_collision_native_forcing_profile,
    c2_binary_interface_profiles,
    c3_passive_transport_profiles,
    conjugate_thermal_qualification_profile,
    LatticeBoltzmannDeploymentRecord,
    LatticeBoltzmannQualificationProfile,
)
from phydrax.discretization.lattice_boltzmann._conjugate_thermal import (
    ConjugateThermalPlan,
)
from phydrax.discretization.lattice_boltzmann._operating_envelope import (
    LatticeBoltzmannOperatingPoint,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import QualificationEvidence, SupportDependency
from phydrax.qualification._reference import ReferenceArtifactManifest


_OPERATIONAL_CLAIMS = frozenset(
    (
        "fused-parity",
        "aa-parity",
        "sharded-parity",
        "checkpoint-parity",
        "output-parity",
    )
)
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
    failures = list(str(value) for value in failed)
    gaps = list(str(value) for value in inconclusive)
    for value in values:
        if not value.is_current(at_time):
            gaps.append(f"not-current:{value.evidence_id}")
        elif value.failed:
            failures.append(f"failed:{value.evidence_id}:{value.reason}")
        elif value.inconclusive:
            gaps.append(f"inconclusive:{value.evidence_id}:{value.reason}")
    if require_evidence and not values:
        gaps.append("missing-evidence")
    outcome = "failed" if failures else "inconclusive" if gaps else "passed"
    return {
        "gate": name,
        "outcome": outcome,
        "evidence_ids": sorted(value.evidence_id for value in values),
        "failed_reasons": sorted(set(failures)),
        "inconclusive_reasons": sorted(set(gaps)),
    }


def lbm_profile_records(
    *, conjugate_thermal: ConjugateThermalPlan | None = None
) -> tuple[dict[str, object], ...]:
    """Return actual C0--C3 profiles and an optional actual CHT profile."""

    if conjugate_thermal is not None and not isinstance(
        conjugate_thermal, ConjugateThermalPlan
    ):
        raise TypeError("conjugate_thermal must be a ConjugateThermalPlan or None.")
    profiles = (
        *c0_guo_baseline_profiles(),
        c1_collision_native_forcing_profile(),
        *c2_binary_interface_profiles(),
        *c3_passive_transport_profiles(),
        *(
            (conjugate_thermal_qualification_profile(conjugate_thermal),)
            if conjugate_thermal is not None
            else ()
        ),
    )
    records = []
    for profile in profiles:
        record = profile.to_record()
        labels = (
            profile.name,
            profile.envelope.physics_model,
            *(str(value) for _, value in profile.support_tuple.attributes),
        )
        if any(
            "continuum-dns" in label.lower()
            or "direct-numerical-simulation" in label.lower()
            for label in labels
        ):
            raise ValueError("LBM candidates cannot carry a continuum-flow method label.")
        records.append(
            {
                **record,
                "method_class": "lattice-kinetic",
                "signed": False,
                "released": False,
                "release_ready": False,
            }
        )
    return tuple(sorted(records, key=lambda value: str(value["profile_id"])))


def _point_record(point: LatticeBoltzmannOperatingPoint, /) -> dict[str, float]:
    return {
        "mach_number": float(np.asarray(point.mach_number)),
        "knudsen_number": float(np.asarray(point.knudsen_number)),
        "relaxation_rate": float(np.asarray(point.relaxation_rate)),
        "minimum_density": float(np.asarray(point.minimum_density)),
        "maximum_density": float(np.asarray(point.maximum_density)),
        "force_number": float(np.asarray(point.force_number)),
        "interface_width_cells": float(np.asarray(point.interface_width_cells)),
        "wall_resolution_cells": float(np.asarray(point.wall_resolution_cells)),
        "viscosity_ratio": float(np.asarray(point.viscosity_ratio)),
        "cahn_number": float(np.asarray(point.cahn_number)),
        "capillary_number": float(np.asarray(point.capillary_number)),
        "relative_mass_drift": float(np.asarray(point.relative_mass_drift)),
        "spurious_current_ratio": float(np.asarray(point.spurious_current_ratio)),
    }


def build_lbm_candidate(
    profile: LatticeBoltzmannQualificationProfile,
    point: LatticeBoltzmannOperatingPoint,
    deployment: LatticeBoltzmannDeploymentRecord,
    run_spec: ResolvedRunSpec,
    evidence: Sequence[QualificationEvidence],
    /,
    *,
    at_time: int,
    resource_counts: Mapping[str, int],
    satisfied_dependencies: Sequence[SupportDependency] = (),
    reference_manifests: Sequence[ReferenceArtifactManifest] = (),
) -> dict[str, object]:
    """Evaluate C0--C3/CHT gates and retain every refusal in an unsigned record."""

    if not isinstance(profile, LatticeBoltzmannQualificationProfile):
        raise TypeError("profile must be a LatticeBoltzmannQualificationProfile.")
    if not isinstance(point, LatticeBoltzmannOperatingPoint):
        raise TypeError("point must be a LatticeBoltzmannOperatingPoint.")
    if not isinstance(deployment, LatticeBoltzmannDeploymentRecord):
        raise TypeError("deployment must be a LatticeBoltzmannDeploymentRecord.")
    if not isinstance(run_spec, ResolvedRunSpec):
        raise TypeError("run_spec must be a ResolvedRunSpec.")
    if not isinstance(resource_counts, Mapping):
        raise TypeError("resource_counts must be a mapping.")
    evidence_ = tuple(evidence)
    if any(not isinstance(value, QualificationEvidence) for value in evidence_):
        raise TypeError("evidence must contain QualificationEvidence values.")
    if len({value.evidence_id for value in evidence_}) != len(evidence_):
        raise ValueError("Qualification evidence IDs must be unique.")
    satisfied = tuple(satisfied_dependencies)
    if any(not isinstance(value, SupportDependency) for value in satisfied):
        raise TypeError("satisfied_dependencies must contain SupportDependency values.")
    references = tuple(reference_manifests)
    if any(not isinstance(value, ReferenceArtifactManifest) for value in references):
        raise TypeError(
            "reference_manifests must contain ReferenceArtifactManifest values."
        )
    if len({value.manifest_id for value in references}) != len(references):
        raise ValueError("Reference manifest IDs must be unique.")

    estimate = profile.envelope.preflight(
        **{str(name): int(value) for name, value in resource_counts.items()}
    )
    admission = profile.envelope.evaluate(point)
    commercial = profile.evaluate(
        evidence_,
        admission,
        deployment,
        at_time=at_time,
        satisfied_dependencies=satisfied,
    )
    grouped = {
        name: tuple(value for value in evidence_ if value.evidence_kind in kinds)
        for name, kinds in _GATE_KINDS.items()
    }
    categorized_ids = {
        value.evidence_id for values in grouped.values() for value in values
    }
    if categorized_ids != {value.evidence_id for value in evidence_}:
        raise ValueError(
            "LBM candidate evidence must use scientific/reference, performance, "
            "operational, or security kinds."
        )
    subject_failures = {
        name: tuple(
            f"profile-subject-mismatch:{value.evidence_id}"
            for value in values
            if profile.profile_id not in value.subject_ids
        )
        for name, values in grouped.items()
    }
    gap_by_predicate = {
        predicate: (outcome, reasons)
        for predicate, outcome, reasons in commercial.coverage.gaps
    }
    scientific_failures = []
    scientific_gaps = []
    operational_failures = []
    operational_gaps = []
    for claim in profile.required_claims:
        outcome_reasons = gap_by_predicate.get(claim.value)
        if outcome_reasons is None:
            continue
        outcome, reasons = outcome_reasons
        destination = (
            operational_failures
            if claim.value in _OPERATIONAL_CLAIMS and outcome == "failed"
            else operational_gaps
            if claim.value in _OPERATIONAL_CLAIMS
            else scientific_failures
            if outcome == "failed"
            else scientific_gaps
        )
        destination.extend(f"claim:{claim.value}:{reason}" for reason in reasons)
    if not bool(np.asarray(admission.admitted)):
        operational_failures.extend(
            f"envelope:{name}" for name in admission.failed_checks()
        )
    operational_failures.extend(
        f"deployment:{name}" for name in commercial.deployment.failed_checks
    )
    operational_failures.extend(
        f"dependency:{identifier}" for identifier in commercial.missing_dependency_ids
    )

    exact_dependency = SupportDependency(profile.profile_id, profile.support_tuple_id)
    run_dependencies = (
        *run_spec.scientific_dependencies,
        *run_spec.deployment_dependencies,
    )
    run_dependency_ids = frozenset(value.dependency_id for value in run_dependencies)
    missing_run_profile_dependencies = tuple(
        value.dependency_id
        for value in profile.dependencies
        if value.dependency_id not in run_dependency_ids
    )
    operational_failures.extend(
        f"resolved-run-missing-profile-dependency:{identifier}"
        for identifier in missing_run_profile_dependencies
    )
    operational_failures.extend(
        f"satisfied-dependency-not-bound-to-run:{value.dependency_id}"
        for value in satisfied
        if value.dependency_id not in run_dependency_ids
    )
    if all(
        value.dependency_id != exact_dependency.dependency_id
        for value in run_dependencies
    ):
        operational_failures.append("resolved-run-missing-exact-lbm-support")
    rights_failures = tuple(
        f"reference:{manifest.manifest_id}:{reason}"
        for manifest in references
        for reason in manifest.rights_refusal_reasons(commercial_use=True)
    )

    performance_failures = (
        ()
        if estimate.fits_budget
        else (f"resource-budget:{estimate.total_bytes}>{estimate.maximum_device_bytes}",)
    )
    gates = {
        "scientific": _gate(
            "scientific",
            grouped["scientific"],
            at_time=at_time,
            failed=(*scientific_failures, *subject_failures["scientific"]),
            inconclusive=scientific_gaps,
        ),
        "performance": _gate(
            "performance",
            grouped["performance"],
            at_time=at_time,
            failed=(*performance_failures, *subject_failures["performance"]),
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
            failed=(*rights_failures, *subject_failures["security"]),
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
        "kind": "lattice-boltzmann-commercial-qualification-candidate",
        "method_class": "lattice-kinetic",
        "tier": profile.tier.value,
        "physics_model": profile.envelope.physics_model,
        "profile": {
            **profile.to_record(),
            "signed": False,
            "released": False,
            "release_ready": False,
        },
        "support_tuple": profile.support_tuple.to_record(),
        "operating_point": _point_record(point),
        "envelope": {
            "envelope_id": profile.envelope.envelope_id,
            "admitted": bool(np.asarray(admission.admitted)),
            "failed_checks": list(admission.failed_checks()),
        },
        "resources": {
            "estimate_id": estimate.estimate_id,
            "state_bytes": estimate.state_bytes,
            "temporary_bytes": estimate.temporary_bytes,
            "halo_bytes": estimate.halo_bytes,
            "checkpoint_bytes": estimate.checkpoint_bytes,
            "output_bytes": estimate.output_bytes,
            "total_bytes": estimate.total_bytes,
            "maximum_device_bytes": estimate.maximum_device_bytes,
            "fits_budget": estimate.fits_budget,
        },
        "commercial_evidence": commercial.to_record(),
        "deployment": deployment.to_record(),
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


def write_records(path: str | Path, records: Sequence[Mapping[str, object]], /) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(list(records), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    write_records(arguments.output, lbm_profile_records())


if __name__ == "__main__":
    main()
