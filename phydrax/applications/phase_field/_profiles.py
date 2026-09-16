#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...qualification import (
    CapabilityProfile,
    ReleaseGateEvidence,
    SupportDependency,
    SupportTuple,
)


def _tuple(name: str, **attributes) -> SupportTuple:
    return SupportTuple(name, attributes)


def phase_field_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return exact unreleased support declarations for closure campaigns."""

    capability = "phase-field-evolution"
    deterministic_support = _tuple(
        capability,
        profile="deterministic-general-binary",
        thermodynamics="registered-bulk-potential",
        mobility="scalar-or-spatial-tensor",
        boundary="wetting-work-periodic-disjoint",
        blocks="heterogeneous-prepared-fe",
        topology="fixed-or-hp-amr",
        storage="dense-binary",
        stochastic="none",
        execution="single-device",
        precision="float64",
    )
    deterministic = CapabilityProfile(
        "phase-field-deterministic-general-binary",
        "phydrax-native",
        "candidate",
        (deterministic_support,),
        required_gates=(
            "behavioral",
            "convergence",
            "conservation",
            "restart",
        ),
    )

    stochastic_support = _tuple(
        capability,
        profile="stochastic-adaptive-binary",
        thermodynamics="registered-bulk-potential",
        mobility="onsager-fdt",
        boundary="periodic-or-no-flux",
        blocks="heterogeneous-prepared-fe",
        topology="hp-amr",
        storage="dense-binary",
        stochastic="global-wiener-spatial-basis",
        execution="distributed-jax",
        precision="float64",
    )
    stochastic = CapabilityProfile(
        "phase-field-stochastic-adaptive-binary",
        "phydrax-native",
        "candidate",
        (stochastic_support,),
        dependencies=(
            SupportDependency(
                deterministic.profile_id, deterministic_support.support_tuple_id
            ),
        ),
        required_gates=(
            "behavioral",
            "statistical",
            "conservation",
            "restart",
            "distributed",
        ),
    )

    active_support = _tuple(
        capability,
        profile="active-grand-potential",
        thermodynamics="isothermal-grand-potential",
        mobility="component-spatial-onsager",
        boundary="wetting-reservoir-periodic-disjoint",
        blocks="heterogeneous-prepared-fe",
        topology="hp-amr",
        storage="fixed-capacity-active-phase-ids",
        stochastic="optional-global-wiener",
        execution="distributed-jax",
        precision="float64",
    )
    active = CapabilityProfile(
        "phase-field-active-grand-potential",
        "phydrax-native",
        "candidate",
        (active_support,),
        dependencies=(
            SupportDependency(
                deterministic.profile_id, deterministic_support.support_tuple_id
            ),
        ),
        required_gates=(
            "behavioral",
            "thermodynamic",
            "conservation",
            "capacity",
            "restart",
            "distributed",
        ),
    )

    flagship_support = _tuple(
        capability,
        profile="fully-integrated-flagship",
        thermodynamics="isothermal-grand-potential",
        mobility="state-aware-onsager-with-drift",
        boundary="wetting-work-reservoir-periodic-disjoint",
        blocks="heterogeneous-prepared-fe",
        topology="hp-amr",
        storage="fixed-capacity-active-phase-ids",
        stochastic="thermal-global-wiener-hierarchical",
        execution="distributed-jax",
        precision="float64",
    )
    flagship = CapabilityProfile(
        "phase-field-fully-integrated-flagship",
        "phydrax-native",
        "candidate",
        (flagship_support,),
        dependencies=(
            SupportDependency(stochastic.profile_id, stochastic_support.support_tuple_id),
            SupportDependency(active.profile_id, active_support.support_tuple_id),
        ),
        required_gates=(
            "behavioral",
            "thermodynamic",
            "statistical",
            "conservation",
            "capacity",
            "convergence",
            "restart",
            "distributed",
        ),
    )
    return deterministic, stochastic, active, flagship


def phase_field_released_profiles(
    qualification_artifact_id: str,
    /,
    *,
    reviewer_id: str,
    issued_at: int,
    expires_at: int,
) -> tuple[CapabilityProfile, ...]:
    """Bind one accepted integrated campaign to exact released support tuples."""

    artifact = str(qualification_artifact_id)
    reviewer = str(reviewer_id)
    if not artifact or not reviewer:
        raise ValueError(
            "Released phase-field profiles require artifact and reviewer IDs."
        )
    candidates = phase_field_candidate_profiles()
    released: list[CapabilityProfile] = []
    for index, candidate in enumerate(candidates):
        if index == 0:
            dependencies = ()
        elif index in (1, 2):
            dependencies = (
                SupportDependency(
                    released[0].profile_id,
                    released[0].support_tuples[0].support_tuple_id,
                ),
            )
        else:
            dependencies = tuple(
                SupportDependency(
                    released[parent].profile_id,
                    released[parent].support_tuples[0].support_tuple_id,
                )
                for parent in (1, 2)
            )
        evidence = tuple(
            ReleaseGateEvidence(
                gate,
                passed=True,
                evidence_ids=(artifact,),
                reviewer_id=reviewer,
                issued_at=issued_at,
                expires_at=expires_at,
            )
            for gate in candidate.required_gates
        )
        released.append(
            CapabilityProfile(
                candidate.name,
                candidate.provider,
                "released",
                candidate.support_tuples,
                dependencies=dependencies,
                required_gates=candidate.required_gates,
                release_evidence=evidence,
                released=True,
            )
        )
    return tuple(released)


__all__ = ["phase_field_candidate_profiles", "phase_field_released_profiles"]
