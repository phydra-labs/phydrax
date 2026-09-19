#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Material streams, isothermal flash, and fixed-point recycle workflows."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class MaterialStream:
    molar_flow_mol_s: Array
    composition: Array
    temperature_k: Array
    pressure_pa: Array

    @classmethod
    def create(cls, molar_flow_mol_s, composition, temperature_k, pressure_pa):
        value = cls(
            jnp.asarray(molar_flow_mol_s),
            jnp.asarray(composition),
            jnp.asarray(temperature_k),
            jnp.asarray(pressure_pa),
        )
        if (
            value.composition.ndim != 1
            or not bool(jnp.all(value.composition >= 0.0))
            or not bool(jnp.isclose(jnp.sum(value.composition), 1.0))
        ):
            raise ValueError("Material-stream composition must be a normalized vector.")
        return value


@dataclass(frozen=True, slots=True)
class FlashResult:
    vapor_fraction: Array
    liquid_composition: Array
    vapor_composition: Array
    material_balance_error: Array


def isothermal_flash(
    feed_composition: ArrayLike, equilibrium_ratios: ArrayLike, /, *, iterations: int = 80
) -> FlashResult:
    feed = jnp.asarray(feed_composition)
    ratios = jnp.asarray(equilibrium_ratios)
    if feed.shape != ratios.shape or feed.ndim != 1:
        raise ValueError("Flash feed and equilibrium ratios must align.")

    def residual(fraction):
        return jnp.sum(feed * (ratios - 1.0) / (1.0 + fraction * (ratios - 1.0)))

    def bisect(_, bounds):
        lower, upper = bounds
        middle = 0.5 * (lower + upper)
        value = residual(middle)
        return jnp.where(value > 0.0, middle, lower), jnp.where(
            value > 0.0, upper, middle
        )

    lower, upper = jax.lax.fori_loop(
        0, int(iterations), bisect, (jnp.asarray(0.0), jnp.asarray(1.0))
    )
    fraction = 0.5 * (lower + upper)
    liquid = feed / (1.0 + fraction * (ratios - 1.0))
    vapor = ratios * liquid
    liquid = liquid / jnp.sum(liquid)
    vapor = vapor / jnp.sum(vapor)
    reconstructed = (1.0 - fraction) * liquid + fraction * vapor
    return FlashResult(fraction, liquid, vapor, jnp.linalg.norm(reconstructed - feed))


def fixed_point_recycle(
    update, initial: ArrayLike, /, *, relaxation: float = 1.0, iterations: int = 32
) -> tuple[Array, Array]:
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("Recycle relaxation must lie in (0, 1].")
    value = jnp.asarray(initial)
    for _ in range(int(iterations)):
        candidate = jnp.asarray(update(value))
        value = value + float(relaxation) * (candidate - value)
    return value, jnp.linalg.norm(jnp.asarray(update(value)) - value)


def process_system_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("process-systems.material-stream", "normalized-composition"),
        ("process-systems.isothermal-flash", "rachford-rice-bisection"),
        ("process-systems.recycle", "relaxed-fixed-point"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("mass-balance", "analytic-control", "public-workflow"),
        )
        for name, formulation in specs
    )


__all__ = [
    "FlashResult",
    "MaterialStream",
    "fixed_point_recycle",
    "isothermal_flash",
    "process_system_candidate_profiles",
]
