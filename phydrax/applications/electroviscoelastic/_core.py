#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Two-phase electro-viscoelastic constitutive and interface evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...electrohydrodynamics import (
    electric_traction_jump,
    leaky_dielectric_surface_charge_rate,
    maxwell_stress,
)
from ...qualification import CapabilityProfile, SupportTuple
from ...rheology import ViscoelasticLaw


@dataclass(frozen=True, slots=True)
class ElectroViscoelasticEvaluation:
    polymer_stress_minus: Array
    polymer_stress_plus: Array
    electric_stress_minus: Array
    electric_stress_plus: Array
    electric_traction_jump: Array
    surface_charge_rate: Array
    polymer_energy: Array


def evaluate_two_phase_electroviscoelastic(
    conformation_minus: ArrayLike,
    conformation_plus: ArrayLike,
    law_minus: ViscoelasticLaw | None,
    law_plus: ViscoelasticLaw | None,
    electric_field_minus_v_m: ArrayLike,
    electric_field_plus_v_m: ArrayLike,
    permittivity_minus_f_m: float,
    permittivity_plus_f_m: float,
    normal: ArrayLike,
    surface_charge_c_m2: ArrayLike,
    surface_divergence_s_inv: ArrayLike,
    normal_current_minus_a_m2: ArrayLike,
    normal_current_plus_a_m2: ArrayLike,
    /,
) -> ElectroViscoelasticEvaluation:
    minus_conformation = jnp.asarray(conformation_minus)
    plus_conformation = jnp.asarray(conformation_plus)
    zero_minus = jnp.zeros_like(minus_conformation)
    zero_plus = jnp.zeros_like(plus_conformation)
    polymer_minus = (
        zero_minus if law_minus is None else law_minus.stress(minus_conformation)
    )
    polymer_plus = zero_plus if law_plus is None else law_plus.stress(plus_conformation)
    electric_minus = maxwell_stress(electric_field_minus_v_m, permittivity_minus_f_m)
    electric_plus = maxwell_stress(electric_field_plus_v_m, permittivity_plus_f_m)
    traction = electric_traction_jump(
        electric_field_minus_v_m,
        electric_field_plus_v_m,
        normal,
        permittivity_minus_f_m,
        permittivity_plus_f_m,
    )
    charge_rate = leaky_dielectric_surface_charge_rate(
        surface_charge_c_m2,
        surface_divergence_s_inv,
        normal_current_minus_a_m2,
        normal_current_plus_a_m2,
    )
    energy = (
        jnp.asarray(0.0)
        if law_minus is None
        else jnp.sum(law_minus.free_energy_density(minus_conformation))
    ) + (
        jnp.asarray(0.0)
        if law_plus is None
        else jnp.sum(law_plus.free_energy_density(plus_conformation))
    )
    return ElectroViscoelasticEvaluation(
        polymer_minus,
        polymer_plus,
        electric_minus,
        electric_plus,
        traction,
        charge_rate,
        energy,
    )


def electroviscoelastic_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    regimes = (
        "perfect-dielectric",
        "leaky-dielectric",
    )
    return tuple(
        CapabilityProfile(
            f"electroviscoelastic.two-phase-{regime}.profile",
            "phydrax",
            "candidate",
            (
                SupportTuple(
                    f"electroviscoelastic.two-phase-{regime}",
                    {
                        "interface": "moving",
                        "rheology": "phase-dependent",
                        "electric": regime,
                    },
                ),
            ),
            required_gates=(
                "stress-jump",
                "surface-charge",
                "polymer-energy",
                "limit-recovery",
            ),
        )
        for regime in regimes
    )


__all__ = [
    "ElectroViscoelasticEvaluation",
    "electroviscoelastic_candidate_profiles",
    "evaluate_two_phase_electroviscoelastic",
]
