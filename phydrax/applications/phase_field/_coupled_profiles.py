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


def _support(profile: str, **attributes) -> SupportTuple:
    return SupportTuple(
        "coupled-phase-field-multiphysics", {"profile": profile, **attributes}
    )


def coupled_phase_field_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specifications = (
        (
            "phase-field-nonisothermal-solidification",
            _support(
                "nonisothermal-solidification",
                thermal="enthalpy-latent-heat",
                anti_trapping="none",
                nucleation="none",
                mechanics="none",
                flow="none",
                electrostatic="none",
            ),
            (),
        ),
        (
            "phase-field-quantitative-alloy-solidification",
            _support(
                "quantitative-alloy-solidification",
                thermal="nonisothermal-grand-potential",
                anti_trapping="thin-interface-calibrated",
                nucleation="none",
                mechanics="none",
                flow="none",
                electrostatic="none",
            ),
            (0,),
        ),
        (
            "phase-field-nucleating-solidification",
            _support(
                "nucleating-solidification",
                thermal="nonisothermal-grand-potential",
                anti_trapping="thin-interface-calibrated",
                nucleation="poisson-clock-cnt",
                mechanics="none",
                flow="none",
                electrostatic="none",
            ),
            (1,),
        ),
        (
            "phase-field-elastochemical",
            _support(
                "elastochemical-phase-transformation",
                thermal="optional-isothermal-or-enthalpy",
                anti_trapping="optional",
                nucleation="optional",
                mechanics="small-or-finite-strain",
                flow="none",
                electrostatic="none",
            ),
            (1,),
        ),
        (
            "phase-field-model-h",
            _support(
                "model-h-phase-flow",
                thermal="optional-enthalpy",
                anti_trapping="optional",
                nucleation="optional",
                mechanics="none",
                flow="incompressible-stokes-navier-stokes",
                electrostatic="none",
            ),
            (1,),
        ),
        (
            "phase-field-electrochemical",
            _support(
                "electrochemical-phase-transformation",
                thermal="optional-enthalpy",
                anti_trapping="optional",
                nucleation="optional",
                mechanics="optional-maxwell",
                flow="none",
                electrostatic="poisson-pnp-maxwell",
            ),
            (1,),
        ),
        (
            "phase-field-thermofluid-solidification",
            _support(
                "thermofluid-solidification",
                thermal="nonisothermal-grand-potential",
                anti_trapping="thin-interface-calibrated",
                nucleation="poisson-clock-cnt",
                mechanics="none",
                flow="incompressible-navier-stokes",
                electrostatic="optional",
            ),
            (2, 4),
        ),
        (
            "phase-field-electro-elasto-hydrodynamic-flagship",
            _support(
                "electro-elasto-hydrodynamic-flagship",
                thermal="nonisothermal-grand-potential",
                anti_trapping="thin-interface-calibrated",
                nucleation="poisson-clock-cnt",
                mechanics="small-and-finite-strain",
                flow="incompressible-navier-stokes",
                electrostatic="poisson-pnp-maxwell",
                topology="hp-amr",
                execution="distributed-jax",
            ),
            (2, 3, 4, 5, 6),
        ),
    )
    profiles: list[CapabilityProfile] = []
    for name, support, parent_indices in specifications:
        dependencies = tuple(
            SupportDependency(
                profiles[parent].profile_id,
                profiles[parent].support_tuples[0].support_tuple_id,
            )
            for parent in parent_indices
        )
        profiles.append(
            CapabilityProfile(
                name,
                "phydrax-native",
                "candidate",
                (support,),
                dependencies=dependencies,
                required_gates=(
                    "behavioral",
                    "thermodynamic",
                    "conservation",
                    "entropy",
                    "convergence",
                    "restart",
                    "distributed",
                ),
            )
        )
    return tuple(profiles)


def coupled_phase_field_released_profiles(
    qualification_artifact_id: str,
    /,
    *,
    reviewer_id: str,
    issued_at: int,
    expires_at: int,
) -> tuple[CapabilityProfile, ...]:
    artifact = str(qualification_artifact_id)
    reviewer = str(reviewer_id)
    if not artifact or not reviewer:
        raise ValueError("Released coupled profiles require artifact and reviewer IDs.")
    candidates = coupled_phase_field_candidate_profiles()
    released: list[CapabilityProfile] = []
    for index, candidate in enumerate(candidates):
        parent_indices = tuple(
            range(index)
            if index == len(candidates) - 1
            else (index - 1,)
            if index in (1, 2)
            else ()
        )
        if index in (3, 4, 5):
            parent_indices = (1,)
        if index == 6:
            parent_indices = (2, 4)
        dependencies = tuple(
            SupportDependency(
                released[parent].profile_id,
                released[parent].support_tuples[0].support_tuple_id,
            )
            for parent in parent_indices
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


__all__ = [
    "coupled_phase_field_candidate_profiles",
    "coupled_phase_field_released_profiles",
]
