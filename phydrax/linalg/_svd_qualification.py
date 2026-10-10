# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from .._fingerprint import canonical_fingerprint
from ..qualification._core_portfolio import CoreQualificationObservation
from ..qualification._registry import CapabilityProfile, SupportTuple, SupportValue
from ..typing import parse


class SingularSubspaceScenario(StrEnum):
    DENSE_REPEATED = "dense-repeated-projector"
    DENSE_COMPLEX_DIAGONAL = "dense-complex-diagonal-projector"
    RANDOMIZED_RESIDENT = "randomized-resident-projector"
    RANDOMIZED_SHELL = "randomized-shell-stopped"


_REQUIRED_GATES = (
    "original-triplet-residual",
    "metric-orthogonality",
    "rank-evidence",
    "leading-selection",
    "requested-derivative",
    "resource-refusal",
    "stage-admission",
)

_SPECIFICATIONS: tuple[tuple[str, dict[str, SupportValue]], ...] = (
    (
        "dense-repeated-projector",
        {
            "method": "dense-svd",
            "representation": "resident-dense",
            "pairing": "euclidean",
            "precision": "float64",
            "derivative": "mathematical-projector",
            "certificate": "exact-spectrum",
            "target_capacity": 6,
            "source_capacity": 3,
            "retained_capacity": 2,
            "workspace_limit_bytes": 4194304,
        },
    ),
    (
        "dense-complex-diagonal-projector",
        {
            "method": "dense-svd",
            "representation": "resident-dense",
            "pairing": "diagonal",
            "precision": "complex128",
            "derivative": "mathematical-projector",
            "certificate": "exact-spectrum",
            "target_capacity": 8,
            "source_capacity": 5,
            "retained_capacity": 2,
            "workspace_limit_bytes": 4194304,
        },
    ),
    (
        "randomized-resident-projector",
        {
            "method": "randomized-svd",
            "representation": "resident-dense",
            "pairing": "euclidean",
            "precision": "float64",
            "derivative": "fixed-sketch-algorithmic-projector",
            "certificate": "deterministic-frobenius",
            "target_capacity": 64,
            "source_capacity": 32,
            "retained_capacity": 4,
            "oversampling": 4,
            "power_iterations": 1,
            "workspace_limit_bytes": 4194304,
        },
    ),
    (
        "randomized-shell-stopped",
        {
            "method": "randomized-svd",
            "representation": "nonmaterializing-action",
            "pairing": "euclidean",
            "precision": "float64",
            "derivative": "stopped",
            "certificate": "independent-gaussian",
            "target_capacity": 128,
            "source_capacity": 64,
            "retained_capacity": 4,
            "oversampling": 8,
            "power_iterations": 2,
            "audit_capacity": 8,
            "workspace_limit_bytes": 4194304,
        },
    ),
)


def singular_subspace_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Declare exact evidence-free tuples, without claiming release qualification."""
    return tuple(
        CapabilityProfile(
            f"linalg.singular-subspaces.{name}",
            "phydrax",
            (
                SupportTuple(
                    "linalg.singular-subspaces",
                    {
                        **attributes,
                        "scenario": parse(
                            SingularSubspaceScenario(name),
                            SingularSubspaceScenario,
                            "scenario",
                        ).value,
                        "workspace_kind": "admitted-stage-estimate",
                    },
                ),
            ),
            required_gates=_REQUIRED_GATES,
        )
        for name, attributes in _SPECIFICATIONS
    )


def singular_subspace_qualification_record(
    observations: Sequence[CoreQualificationObservation],
    /,
) -> dict[str, object]:
    """Require one actual complete observation per tuple; never sign a release."""
    values = tuple(observations)
    expected = {profile.profile_id for profile in singular_subspace_candidate_profiles()}
    observed = {value.profile_id for value in values}
    if len(values) != len(observed) or observed != expected:
        raise ValueError(
            "Singular-subspace observations must cover each exact profile once."
        )
    ordered = tuple(sorted(values, key=lambda value: value.profile_id))
    payload: dict[str, object] = {
        "kind": "singular-subspace-qualification",
        "release_claim": False,
        "observations": [value.to_record() for value in ordered],
        "passed": all(value.passed for value in ordered),
    }
    return {**payload, "observation_id": canonical_fingerprint(payload)}
