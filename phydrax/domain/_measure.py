#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


MeasureKind = Literal[
    "lebesgue",
    "hausdorff",
    "probability",
    "counting",
    "dirac",
    "trajectory",
    "riemannian",
    "external",
]


class ExactMass(StrictModule):
    """Analytically or structurally exact total measure mass."""

    value: Array

    def __init__(self, value: ArrayLike, /):
        value_ = jnp.asarray(value, dtype=float).reshape(())
        if bool(value_ < 0):
            raise ValueError("Measure mass must be non-negative.")
        self.value = value_


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
        value_ = jnp.asarray(value, dtype=float).reshape(())
        uncertainty_ = jnp.asarray(uncertainty, dtype=float).reshape(())
        if bool(value_ < 0) or bool(uncertainty_ < 0):
            raise ValueError("Estimated mass and uncertainty must be non-negative.")
        if int(evaluations) <= 0:
            raise ValueError("EstimatedMass.evaluations must be positive.")
        if not provenance:
            raise ValueError("EstimatedMass.provenance must be non-empty.")
        self.value = value_
        self.uncertainty = uncertainty_
        self.evaluations = int(evaluations)
        self.provenance = str(provenance)


class UnknownMass(StrictModule):
    """Mass that is intentionally unavailable without numerical estimation."""

    reason: str = eqx.field(static=True)

    def __init__(self, reason: str, /):
        if not reason:
            raise ValueError("UnknownMass.reason must be non-empty.")
        self.reason = str(reason)


Mass: TypeAlias = ExactMass | EstimatedMass | UnknownMass


class BaseMeasure(StrictModule):
    """Measure family and its currently known total mass."""

    kind: MeasureKind = eqx.field(static=True)
    mass: Mass
    normalized: bool = eqx.field(static=True)

    def __init__(
        self,
        kind: MeasureKind,
        mass: Mass,
        /,
        *,
        normalized: bool = False,
    ):
        if kind not in (
            "lebesgue",
            "hausdorff",
            "probability",
            "counting",
            "dirac",
            "trajectory",
            "riemannian",
            "external",
        ):
            raise ValueError(f"Unknown measure kind {kind!r}.")
        if not isinstance(mass, (ExactMass, EstimatedMass, UnknownMass)):
            raise TypeError("BaseMeasure.mass must be an explicit Mass descriptor.")
        if normalized and isinstance(mass, ExactMass) and not bool(
            jnp.isclose(mass.value, 1.0)
        ):
            raise ValueError("A normalized exact measure must have unit mass.")
        self.kind = kind
        self.mass = mass
        self.normalized = bool(normalized)


def product_mass(masses: Sequence[Mass], /) -> Mass:
    """Combine independent factor masses without hiding uncertainty."""
    exact = jnp.asarray(1.0, dtype=float)
    estimated_value = jnp.asarray(1.0, dtype=float)
    estimated_relative_variance = jnp.asarray(0.0, dtype=float)
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
        safe_value = jnp.maximum(mass.value, jnp.finfo(float).tiny)
        estimated_relative_variance = estimated_relative_variance + (
            mass.uncertainty / safe_value
        ) ** 2
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
    value = jnp.asarray(0.0, dtype=float)
    variance = jnp.asarray(0.0, dtype=float)
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


def require_exact_mass(mass: Mass, /, *, operation: str) -> Array:
    """Return an exact mass or reject an operation that requires one."""
    if isinstance(mass, ExactMass):
        return mass.value
    if isinstance(mass, EstimatedMass):
        raise ValueError(
            f"{operation} requires exact mass; received estimate from "
            f"{mass.provenance!r}."
        )
    raise ValueError(f"{operation} requires exact mass: {mass.reason}")


__all__ = [
    "BaseMeasure",
    "ExactMass",
    "EstimatedMass",
    "Mass",
    "MeasureKind",
    "UnknownMass",
    "product_mass",
    "require_exact_mass",
    "sum_mass",
]
