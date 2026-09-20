#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from ._strict import StrictModule


def _nonnegative(value: Array, message: str, /) -> Array:
    invalid = jnp.any(value < 0)
    if isinstance(invalid, jax_core.Tracer):
        return eqx.error_if(value, invalid, message)
    if bool(invalid):
        raise ValueError(message)
    return value


class ExactMass(StrictModule):
    """Analytically or structurally exact total measure mass."""

    value: Array

    def __init__(self, value: ArrayLike, /):
        value_ = jnp.asarray(value, dtype=jnp.float64).reshape(())
        self.value = _nonnegative(value_, "Measure mass must be non-negative.")


class EstimatedMass(StrictModule):
    """Numerically estimated mass with explicit uncertainty and provenance."""

    value: Array
    uncertainty: Array
    evaluations: int = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        uncertainty: ArrayLike,
        /,
        *,
        evaluations: int,
        provenance: str,
    ):
        value_ = jnp.asarray(value, dtype=jnp.float64).reshape(())
        uncertainty_ = jnp.asarray(uncertainty, dtype=jnp.float64).reshape(())
        message = "Estimated mass and uncertainty must be non-negative."
        self.value = _nonnegative(value_, message)
        self.uncertainty = _nonnegative(uncertainty_, message)
        if evaluations <= 0:
            raise ValueError("EstimatedMass.evaluations must be positive.")
        if not provenance:
            raise ValueError("EstimatedMass.provenance must be non-empty.")
        self.evaluations = evaluations
        self.provenance = provenance


class UnknownMass(StrictModule):
    """Mass that is intentionally unavailable without numerical estimation."""

    reason: str = eqx.field(static=True)

    def __init__(self, reason: str, /):
        if not reason:
            raise ValueError("UnknownMass.reason must be non-empty.")
        self.reason = reason


Mass: TypeAlias = ExactMass | EstimatedMass | UnknownMass


def product_mass(masses: Sequence[Mass], /) -> Mass:
    """Combine independent factor masses without hiding uncertainty."""
    exact = jnp.asarray(1.0, dtype=jnp.float64)
    estimated_value = jnp.asarray(1.0, dtype=jnp.float64)
    estimated_relative_variance = jnp.asarray(0.0, dtype=jnp.float64)
    evaluations = 0
    provenance: list[str] = []
    for mass in masses:
        if isinstance(mass, UnknownMass):
            return UnknownMass(f"product includes unknown mass: {mass.reason}")
        if isinstance(mass, ExactMass):
            exact = exact * mass.value
            estimated_value = estimated_value * mass.value
            continue
        estimated_value = estimated_value * mass.value
        safe_value = jnp.maximum(mass.value, jnp.finfo(jnp.float64).tiny)
        estimated_relative_variance = (
            estimated_relative_variance + (mass.uncertainty / safe_value) ** 2
        )
        evaluations += mass.evaluations
        provenance.append(mass.provenance)
    if provenance:
        return EstimatedMass(
            estimated_value,
            jnp.abs(estimated_value) * jnp.sqrt(estimated_relative_variance),
            evaluations=evaluations,
            provenance="product(" + ", ".join(provenance) + ")",
        )
    return ExactMass(exact)


def sum_mass(masses: Sequence[Mass], /) -> Mass:
    """Combine disjoint additive masses without treating unknown values as zero."""
    value = jnp.asarray(0.0, dtype=jnp.float64)
    variance = jnp.asarray(0.0, dtype=jnp.float64)
    evaluations = 0
    provenance: list[str] = []
    for mass in masses:
        if isinstance(mass, UnknownMass):
            return UnknownMass(f"sum includes unknown mass: {mass.reason}")
        value = value + mass.value
        if isinstance(mass, EstimatedMass):
            variance = variance + mass.uncertainty**2
            evaluations += mass.evaluations
            provenance.append(mass.provenance)
    if provenance:
        return EstimatedMass(
            value,
            jnp.sqrt(variance),
            evaluations=evaluations,
            provenance="sum(" + ", ".join(provenance) + ")",
        )
    return ExactMass(value)


def scale_mass(mass: Mass, factor: ArrayLike, /, *, provenance: str) -> Mass:
    """Scale a mass by an exact non-negative factor."""
    factor_ = _nonnegative(
        jnp.asarray(factor, dtype=jnp.float64).reshape(()),
        "Mass scale factor must be non-negative.",
    )
    if isinstance(mass, UnknownMass):
        return mass
    if isinstance(mass, EstimatedMass):
        return EstimatedMass(
            factor_ * mass.value,
            factor_ * mass.uncertainty,
            evaluations=mass.evaluations,
            provenance=f"{provenance}({mass.provenance})",
        )
    return ExactMass(factor_ * mass.value)


def known_mass_value(mass: Mass, /, *, operation: str) -> Array:
    """Return a known exact or estimated value without erasing its stored evidence."""
    if isinstance(mass, UnknownMass):
        raise ValueError(f"{operation} requires known mass: {mass.reason}")
    return mass.value


def require_exact_mass(mass: Mass, /, *, operation: str) -> Array:
    """Return an exact mass or reject an operation that requires one."""
    if isinstance(mass, ExactMass):
        return mass.value
    if isinstance(mass, EstimatedMass):
        raise ValueError(
            f"{operation} requires exact mass; received estimate from {mass.provenance!r}."
        )
    raise ValueError(f"{operation} requires exact mass: {mass.reason}")


__all__ = [
    "ExactMass",
    "EstimatedMass",
    "Mass",
    "UnknownMass",
    "known_mass_value",
    "product_mass",
    "require_exact_mass",
    "scale_mass",
    "sum_mass",
]
