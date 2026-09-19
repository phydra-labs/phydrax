#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded smart-material constitutive couplings."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class LinearPiezoelectricLaw:
    stiffness: Array
    piezoelectric: Array
    permittivity: Array

    @classmethod
    def create(
        cls, stiffness: ArrayLike, piezoelectric: ArrayLike, permittivity: ArrayLike
    ):
        return cls(
            jnp.asarray(stiffness), jnp.asarray(piezoelectric), jnp.asarray(permittivity)
        )

    def evaluate(
        self, strain: ArrayLike, electric_field: ArrayLike, /
    ) -> tuple[Array, Array]:
        strain_ = jnp.asarray(strain)
        field = jnp.asarray(electric_field)
        stress = (
            self.stiffness @ strain_ - jnp.swapaxes(self.piezoelectric, -1, -2) @ field
        )
        electric_displacement = self.piezoelectric @ strain_ + self.permittivity @ field
        return stress, electric_displacement


def dielectric_elastomer_maxwell_stress(
    electric_field_v_m: ArrayLike, permittivity_f_m: float, /
) -> Array:
    field = jnp.asarray(electric_field_v_m)
    identity = jnp.eye(field.shape[-1], dtype=field.dtype)
    return float(permittivity_f_m) * (
        field[..., :, None] * field[..., None, :]
        - 0.5 * jnp.sum(field * field, axis=-1)[..., None, None] * identity
    )


def cubic_magnetostrictive_strain(
    magnetization_direction: ArrayLike, saturation_magnetostriction: float, /
) -> Array:
    direction = jnp.asarray(magnetization_direction)
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    identity = jnp.eye(direction.shape[-1], dtype=direction.dtype)
    return (
        1.5
        * float(saturation_magnetostriction)
        * (
            direction[..., :, None] * direction[..., None, :]
            - identity / direction.shape[-1]
        )
    )


def thermoelectric_fluxes(
    temperature_gradient_k_m: ArrayLike,
    electric_field_v_m: ArrayLike,
    electrical_conductivity_s_m: float,
    seebeck_v_k: float,
    thermal_conductivity_w_m_k: float,
    temperature_k: float,
    /,
) -> tuple[Array, Array]:
    gradient = jnp.asarray(temperature_gradient_k_m)
    field = jnp.asarray(electric_field_v_m)
    current = float(electrical_conductivity_s_m) * (field - float(seebeck_v_k) * gradient)
    heat = (
        float(seebeck_v_k) * float(temperature_k) * current
        - float(thermal_conductivity_w_m_k) * gradient
    )
    return current, heat


def smart_material_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("smart-materials.piezoelectric", "linear-stress-charge"),
        ("smart-materials.dielectric-elastomer", "maxwell-stress"),
        ("smart-materials.magnetostriction", "cubic-saturation"),
        ("smart-materials.thermoelectric", "seebeck-peltier-fourier"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("reciprocity", "energy-consistency", "analytic-control"),
        )
        for name, formulation in specs
    )


__all__ = [
    "LinearPiezoelectricLaw",
    "cubic_magnetostrictive_strain",
    "dielectric_elastomer_maxwell_stress",
    "smart_material_candidate_profiles",
    "thermoelectric_fluxes",
]
