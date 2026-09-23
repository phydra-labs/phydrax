#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optical force, photoelastic, thermo-optic, and STOP transfer primitives."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..qualification import CapabilityProfile, SupportTuple


def optical_maxwell_stress(
    electric_field: ArrayLike,
    magnetic_field: ArrayLike,
    permittivity_f_m: float,
    permeability_h_m: float,
    /,
) -> Array:
    electric = jnp.asarray(electric_field)
    magnetic = jnp.asarray(magnetic_field)
    if (
        not isfinite(permittivity_f_m)
        or permittivity_f_m <= 0
        or not isfinite(permeability_h_m)
        or permeability_h_m <= 0
        or electric.shape != magnetic.shape
        or electric.ndim == 0
        or electric.shape[-1] == 0
    ):
        raise ValueError(
            "Optical fields/material constants are incompatible or nonphysical."
        )
    electric = eqx.error_if(
        electric,
        jnp.any(~jnp.isfinite(electric) | ~jnp.isfinite(magnetic)),
        "Optical electromagnetic fields must be finite.",
    )
    identity = jnp.eye(electric.shape[-1], dtype=electric.dtype)
    return (
        float(permittivity_f_m)
        * (
            electric[..., :, None] * jnp.conj(electric[..., None, :])
            - 0.5 * jnp.sum(jnp.abs(electric) ** 2, axis=-1)[..., None, None] * identity
        )
        + float(permeability_h_m)
        * (
            magnetic[..., :, None] * jnp.conj(magnetic[..., None, :])
            - 0.5 * jnp.sum(jnp.abs(magnetic) ** 2, axis=-1)[..., None, None] * identity
        )
    ).real


def photoelastic_permittivity_perturbation(
    strain: ArrayLike,
    photoelastic_tensor: ArrayLike,
    refractive_index: float,
    /,
) -> Array:
    strain_ = jnp.asarray(strain)
    tensor = jnp.asarray(photoelastic_tensor)
    if (
        strain_.ndim < 2
        or strain_.shape[-1] != strain_.shape[-2]
        or tensor.shape != strain_.shape[-2:] * 2
        or not isfinite(refractive_index)
        or refractive_index <= 0
    ):
        raise ValueError(
            "Photoelastic strain/tensor/index are incompatible or nonphysical."
        )
    strain_ = eqx.error_if(
        strain_,
        jnp.any(~jnp.isfinite(strain_) | ~jnp.isfinite(tensor)),
        "Photoelastic strain and tensor must be finite.",
    )
    return -(float(refractive_index) ** 4) * contract(
        "ijkl,...kl->...ij", tensor, strain_, backend="jax"
    )


def thermo_optic_index(
    refractive_index_reference: ArrayLike,
    temperature_k: ArrayLike,
    reference_temperature_k: float,
    thermo_optic_coefficient_k_inv: ArrayLike,
    /,
) -> Array:
    if not isfinite(reference_temperature_k) or reference_temperature_k <= 0:
        raise ValueError(
            "Thermo-optic reference temperature must be finite and positive."
        )
    index, temperature, coefficient = jnp.broadcast_arrays(
        jnp.asarray(refractive_index_reference),
        jnp.asarray(temperature_k),
        jnp.asarray(thermo_optic_coefficient_k_inv),
    )
    index = eqx.error_if(
        index,
        jnp.any(
            ~jnp.isfinite(index)
            | ~jnp.isfinite(temperature)
            | ~jnp.isfinite(coefficient)
            | (index <= 0)
            | (temperature <= 0)
        ),
        "Thermo-optic index/temperature/coefficient must be finite and physical.",
    )
    return index + coefficient * (temperature - float(reference_temperature_k))


def optomechanics_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("optomechanics.maxwell-stress", "time-averaged-isotropic"),
        ("optomechanics.photoelastic", "fourth-order-linear"),
        ("optomechanics.thermo-optic", "linear-temperature"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("energy-consistency", "analytic-control", "public-workflow"),
        )
        for name, formulation in specs
    )


__all__ = [
    "optical_maxwell_stress",
    "optomechanics_candidate_profiles",
    "photoelastic_permittivity_perturbation",
    "thermo_optic_index",
]
