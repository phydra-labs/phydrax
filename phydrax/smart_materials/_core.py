#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded smart-material constitutive couplings."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
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
        stiffness_ = np.asarray(stiffness)
        piezoelectric_ = np.asarray(piezoelectric)
        permittivity_ = np.asarray(permittivity)
        if (
            stiffness_.ndim != 2
            or stiffness_.shape[0] != stiffness_.shape[1]
            or permittivity_.ndim != 2
            or permittivity_.shape[0] != permittivity_.shape[1]
            or piezoelectric_.shape != (permittivity_.shape[0], stiffness_.shape[0])
        ):
            raise ValueError(
                "Piezoelectric constitutive blocks have incompatible shapes."
            )
        if not all(
            np.all(np.isfinite(value))
            for value in (stiffness_, piezoelectric_, permittivity_)
        ):
            raise ValueError("Piezoelectric constitutive blocks must be finite.")
        if (
            not np.allclose(stiffness_, stiffness_.T)
            or not np.allclose(permittivity_, permittivity_.T)
            or np.min(np.linalg.eigvalsh(stiffness_)) <= 0
            or np.min(np.linalg.eigvalsh(permittivity_)) <= 0
        ):
            raise ValueError(
                "Piezoelectric stiffness and permittivity must be symmetric positive definite."
            )
        return cls(
            jnp.asarray(stiffness_),
            jnp.asarray(piezoelectric_),
            jnp.asarray(permittivity_),
        )

    def evaluate(
        self, strain: ArrayLike, electric_field: ArrayLike, /
    ) -> tuple[Array, Array]:
        strain_ = jnp.asarray(strain)
        field = jnp.asarray(electric_field)
        if strain_.shape[-1:] != (self.stiffness.shape[0],) or field.shape[-1:] != (
            self.permittivity.shape[0],
        ):
            raise ValueError("Piezoelectric strain and field have incompatible shapes.")
        stress = (
            self.stiffness @ strain_ - jnp.swapaxes(self.piezoelectric, -1, -2) @ field
        )
        electric_displacement = self.piezoelectric @ strain_ + self.permittivity @ field
        return stress, electric_displacement


def dielectric_elastomer_maxwell_stress(
    electric_field_v_m: ArrayLike, permittivity_f_m: float, /
) -> Array:
    if not isfinite(permittivity_f_m) or permittivity_f_m <= 0:
        raise ValueError("Dielectric permittivity must be finite and positive.")
    field = jnp.asarray(electric_field_v_m)
    if field.ndim == 0 or field.shape[-1] == 0:
        raise ValueError("Electric field must have a nonempty component axis.")
    field = eqx.error_if(
        field,
        jnp.any(~jnp.isfinite(field)),
        "Electric field must be finite.",
    )
    identity = jnp.eye(field.shape[-1], dtype=field.dtype)
    outer = field[..., :, None] * jnp.conj(field[..., None, :])
    magnitude = jnp.sum(field * jnp.conj(field), axis=-1)
    return float(permittivity_f_m) * jnp.real(
        outer - 0.5 * magnitude[..., None, None] * identity
    )


def cubic_magnetostrictive_strain(
    magnetization_direction: ArrayLike, saturation_magnetostriction: float, /
) -> Array:
    if not isfinite(saturation_magnetostriction):
        raise ValueError("Saturation magnetostriction must be finite.")
    direction = jnp.asarray(magnetization_direction)
    if direction.ndim == 0 or direction.shape[-1] == 0:
        raise ValueError("Magnetization direction must have a component axis.")
    norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    direction = eqx.error_if(
        direction,
        jnp.any(~jnp.isfinite(direction) | (norm <= 0)),
        "Magnetization direction must be finite and nonzero.",
    )
    direction = direction / norm
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
    if (
        not isfinite(electrical_conductivity_s_m)
        or electrical_conductivity_s_m < 0
        or not isfinite(seebeck_v_k)
        or not isfinite(thermal_conductivity_w_m_k)
        or thermal_conductivity_w_m_k < 0
        or not isfinite(temperature_k)
        or temperature_k <= 0
    ):
        raise ValueError("Thermoelectric coefficients are outside physical bounds.")
    gradient = jnp.asarray(temperature_gradient_k_m)
    field = jnp.asarray(electric_field_v_m)
    if gradient.shape != field.shape:
        raise ValueError("Thermal gradient and electric field must have matching shapes.")
    gradient = eqx.error_if(
        gradient,
        jnp.any(~jnp.isfinite(gradient) | ~jnp.isfinite(field)),
        "Thermoelectric fields must be finite.",
    )
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
