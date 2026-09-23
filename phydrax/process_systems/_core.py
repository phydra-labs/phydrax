#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Material streams, isothermal flash, and fixed-point recycle workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import equinox as eqx
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
        if value.composition.ndim != 1 or value.composition.size == 0:
            raise ValueError("Material-stream composition must be a nonempty vector.")
        composition_ = eqx.error_if(
            value.composition,
            jnp.any(~jnp.isfinite(value.composition) | (value.composition < 0))
            | ~jnp.isclose(jnp.sum(value.composition), 1.0),
            "Material-stream composition must be finite, nonnegative, and normalized.",
        )
        flow = eqx.error_if(
            value.molar_flow_mol_s,
            jnp.any(~jnp.isfinite(value.molar_flow_mol_s) | (value.molar_flow_mol_s < 0))
            | jnp.any(~jnp.isfinite(value.temperature_k) | (value.temperature_k <= 0))
            | jnp.any(~jnp.isfinite(value.pressure_pa) | (value.pressure_pa <= 0)),
            "Material-stream flow, temperature, and pressure are outside physical bounds.",
        )
        return cls(flow, composition_, value.temperature_k, value.pressure_pa)


@dataclass(frozen=True, slots=True)
class FlashResult:
    vapor_fraction: Array
    liquid_composition: Array
    vapor_composition: Array
    material_balance_error: Array
    phase_code: Array
    converged: Array
    successful: Array


def isothermal_flash(
    feed_composition: ArrayLike,
    equilibrium_ratios: ArrayLike,
    /,
    *,
    iterations: int = 80,
    tolerance: float = 1e-10,
) -> FlashResult:
    feed = jnp.asarray(feed_composition)
    ratios = jnp.asarray(equilibrium_ratios)
    if feed.shape != ratios.shape or feed.ndim != 1:
        raise ValueError("Flash feed and equilibrium ratios must align.")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
        or not isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Flash iteration controls must be positive.")
    feed = eqx.error_if(
        feed,
        jnp.any(~jnp.isfinite(feed) | ~jnp.isfinite(ratios) | (feed < 0) | (ratios <= 0))
        | ~jnp.isclose(jnp.sum(feed), 1.0),
        "Flash feed must be normalized/nonnegative and K values finite/positive.",
    )

    def residual(fraction):
        return jnp.sum(feed * (ratios - 1.0) / (1.0 + fraction * (ratios - 1.0)))

    def bisect(_, bounds):
        lower, upper = bounds
        middle = 0.5 * (lower + upper)
        value = residual(middle)
        return jnp.where(value > 0.0, middle, lower), jnp.where(
            value > 0.0, upper, middle
        )

    residual_zero = residual(jnp.asarray(0.0, dtype=feed.dtype))
    residual_one = residual(jnp.asarray(1.0, dtype=feed.dtype))
    indeterminate = (jnp.abs(residual_zero) <= tolerance) & (
        jnp.abs(residual_one) <= tolerance
    )
    liquid_phase = residual_zero < -tolerance
    vapor_phase = residual_one > tolerance
    lower, upper = jax.lax.fori_loop(
        0,
        iterations,
        bisect,
        (jnp.asarray(0.0, dtype=feed.dtype), jnp.asarray(1.0, dtype=feed.dtype)),
    )
    two_phase_fraction = 0.5 * (lower + upper)
    fraction = jnp.where(
        liquid_phase,
        jnp.asarray(0.0, dtype=feed.dtype),
        jnp.where(vapor_phase, jnp.asarray(1.0, dtype=feed.dtype), two_phase_fraction),
    )
    liquid = feed / (1.0 + fraction * (ratios - 1.0))
    vapor = ratios * liquid
    liquid = liquid / jnp.sum(liquid)
    vapor = vapor / jnp.sum(vapor)
    reconstructed = (1.0 - fraction) * liquid + fraction * vapor
    balance_error = jnp.linalg.norm(reconstructed - feed)
    phase_code = jnp.where(
        indeterminate,
        2,
        jnp.where(liquid_phase, -1, jnp.where(vapor_phase, 1, 0)),
    )
    converged = ~indeterminate & (
        liquid_phase | vapor_phase | (jnp.abs(residual(two_phase_fraction)) <= tolerance)
    )
    successful = (
        converged
        & jnp.all(jnp.isfinite(liquid))
        & jnp.all(jnp.isfinite(vapor))
        & jnp.isfinite(balance_error)
        & (balance_error <= tolerance)
    )
    return FlashResult(
        fraction,
        liquid,
        vapor,
        balance_error,
        phase_code,
        converged,
        successful,
    )


@dataclass(frozen=True, slots=True)
class RecycleResult:
    value: Array
    candidate_value: Array
    residual_norm: Array
    iterations: int
    successful: Array


def fixed_point_recycle(
    update: Callable[[Array], ArrayLike],
    initial: ArrayLike,
    /,
    *,
    relaxation: float = 1.0,
    iterations: int = 32,
    tolerance: float = 1e-10,
) -> RecycleResult:
    if not callable(update):
        raise TypeError("Recycle update must be callable.")
    if (
        not isfinite(relaxation)
        or not 0.0 < relaxation <= 1.0
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
        or not isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Recycle controls must be finite and positive.")
    source = jnp.asarray(initial)
    value = source
    for _ in range(iterations):
        candidate = jnp.asarray(update(value))
        if candidate.shape != source.shape:
            raise ValueError("Recycle update must preserve state shape.")
        value = value + float(relaxation) * (candidate - value)
    candidate = jnp.asarray(update(value))
    residual = jnp.linalg.norm(candidate - value)
    successful = (
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(candidate))
        & jnp.isfinite(residual)
        & (residual <= tolerance)
    )
    accepted = jnp.where(successful, value, source)
    return RecycleResult(accepted, value, residual, iterations, successful)


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
    "RecycleResult",
    "MaterialStream",
    "fixed_point_recycle",
    "isothermal_flash",
    "process_system_candidate_profiles",
]
