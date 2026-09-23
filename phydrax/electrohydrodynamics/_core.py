#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Electrohydrodynamic stresses, charge transport, and interface coupling."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class ElectricMaterial:
    permittivity_f_m: float
    conductivity_s_m: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.permittivity_f_m)
            or self.permittivity_f_m <= 0.0
            or not isfinite(self.conductivity_s_m)
            or self.conductivity_s_m < 0.0
        ):
            raise ValueError("Electric material parameters are outside physical bounds.")


def maxwell_stress(
    electric_field_v_m: ArrayLike, permittivity_f_m: ArrayLike, /
) -> Array:
    field = jnp.asarray(electric_field_v_m)
    permittivity = jnp.asarray(permittivity_f_m)
    if field.ndim == 0 or field.shape[-1] == 0:
        raise ValueError("Electric field must have a nonempty component axis.")
    field = eqx.error_if(
        field,
        jnp.any(~jnp.isfinite(field) | ~jnp.isfinite(permittivity) | (permittivity <= 0)),
        "Electric field/permittivity must be finite with positive permittivity.",
    )
    identity = jnp.eye(field.shape[-1], dtype=field.dtype)
    outer = field[..., :, None] * jnp.conj(field[..., None, :])
    magnitude = jnp.sum(field * jnp.conj(field), axis=-1)
    return jnp.real(
        permittivity[..., None, None]
        * (outer - 0.5 * magnitude[..., None, None] * identity)
    )


def leaky_dielectric_surface_charge_rate(
    surface_charge_c_m2: ArrayLike,
    tangential_divergence_s_inv: ArrayLike,
    normal_current_minus_a_m2: ArrayLike,
    normal_current_plus_a_m2: ArrayLike,
    surface_diffusion_m2_s: float = 0.0,
    surface_laplacian_charge_c_m4: ArrayLike = 0.0,
    /,
) -> Array:
    charge = jnp.asarray(surface_charge_c_m2)
    return (
        jnp.asarray(normal_current_minus_a_m2)
        - jnp.asarray(normal_current_plus_a_m2)
        - charge * jnp.asarray(tangential_divergence_s_inv)
        + float(surface_diffusion_m2_s) * jnp.asarray(surface_laplacian_charge_c_m4)
    )


def electric_traction_jump(
    electric_field_minus_v_m: ArrayLike,
    electric_field_plus_v_m: ArrayLike,
    normal: ArrayLike,
    permittivity_minus_f_m: float,
    permittivity_plus_f_m: float,
    /,
) -> Array:
    normal_ = jnp.asarray(normal)
    minus = maxwell_stress(electric_field_minus_v_m, permittivity_minus_f_m)
    plus = maxwell_stress(electric_field_plus_v_m, permittivity_plus_f_m)
    return (plus - minus) @ normal_


@dataclass(frozen=True, slots=True)
class ElectrohydrodynamicLedger:
    bulk_charge_c: Array
    surface_charge_c: Array
    electric_energy_j: Array
    joule_dissipation_w: Array
    mechanical_power_w: Array

    @property
    def finite(self) -> Array:
        return jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        self.bulk_charge_c,
                        self.surface_charge_c,
                        self.electric_energy_j,
                        self.joule_dissipation_w,
                        self.mechanical_power_w,
                    )
                )
            )
        )


def electrohydrodynamics_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    names = (
        ("electrohydrodynamics.maxwell-stress", "bulk-interface"),
        ("electrohydrodynamics.leaky-dielectric", "surface-charge"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("charge-conservation", "traction-jump", "energy-ledger"),
        )
        for name, formulation in names
    )


__all__ = [
    "ElectricMaterial",
    "ElectrohydrodynamicLedger",
    "electric_traction_jump",
    "electrohydrodynamics_candidate_profiles",
    "leaky_dielectric_surface_charge_rate",
    "maxwell_stress",
]
