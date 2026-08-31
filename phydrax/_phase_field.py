#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import AbstractAttribute, StrictModule
from ._trainable import NonTrainableState


def double_well_free_energy_density(
    value: ArrayLike,
    scale: ArrayLike = 1.0,
    /,
) -> Array:
    """Evaluate the canonical symmetric binary double-well energy density."""

    values = jnp.asarray(value)
    coefficient = jnp.asarray(scale, dtype=values.dtype)
    if coefficient.shape != ():
        raise ValueError("Double-well scale must be scalar.")
    coefficient = eqx.error_if(
        coefficient,
        ~jnp.isfinite(coefficient) | (coefficient <= 0.0),
        "Double-well scale must be finite and positive.",
    )
    return 0.25 * coefficient * (values**2 - 1.0) ** 2


def double_well_chemical_derivative(
    value: ArrayLike,
    scale: ArrayLike = 1.0,
    /,
) -> Array:
    """Evaluate the canonical double-well chemical derivative."""

    values = jnp.asarray(value)
    coefficient = jnp.asarray(scale, dtype=values.dtype)
    if coefficient.shape != ():
        raise ValueError("Double-well scale must be scalar.")
    coefficient = eqx.error_if(
        coefficient,
        ~jnp.isfinite(coefficient) | (coefficient <= 0.0),
        "Double-well scale must be finite and positive.",
    )
    return coefficient * (values**3 - values)


class AbstractBulkFreeEnergy(StrictModule, NonTrainableState):
    free_energy_id: AbstractAttribute[str]

    @abc.abstractmethod
    def density(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError


class DoubleWellFreeEnergy(AbstractBulkFreeEnergy):
    scale: Array
    free_energy_id: str = eqx.field(static=True)

    def __init__(self, scale: ArrayLike = 1.0, /):
        scale_host = np.asarray(scale)
        if scale_host.shape != () or not np.isfinite(scale_host) or scale_host <= 0.0:
            raise ValueError("Double-well scale must be one positive finite scalar.")
        self.scale = jnp.asarray(scale_host)
        self.free_energy_id = canonical_fingerprint(
            {"kind": "double-well-free-energy", "scale": float(scale_host)}
        )

    def density(self, value: ArrayLike, /) -> Array:
        return double_well_free_energy_density(value, self.scale)

    def derivative(self, value: ArrayLike, /) -> Array:
        return double_well_chemical_derivative(value, self.scale)


class BinaryFreeEnergyEvaluation(StrictModule):
    density: Array
    chemical_derivative: Array
    free_energy_id: str = eqx.field(static=True)


def evaluate_binary_free_energy(
    free_energy: AbstractBulkFreeEnergy,
    value: ArrayLike,
    /,
) -> BinaryFreeEnergyEvaluation:
    if not isinstance(free_energy, AbstractBulkFreeEnergy):
        raise TypeError("free_energy must be AbstractBulkFreeEnergy.")
    values = jnp.asarray(value)
    density = jnp.asarray(free_energy.density(values))
    derivative = jnp.asarray(free_energy.derivative(values))
    if density.shape != values.shape or derivative.shape != values.shape:
        raise ValueError("Bulk free energy must preserve field shape.")
    return BinaryFreeEnergyEvaluation(density, derivative, free_energy.free_energy_id)


__all__ = [
    "AbstractBulkFreeEnergy",
    "BinaryFreeEnergyEvaluation",
    "DoubleWellFreeEnergy",
    "double_well_chemical_derivative",
    "double_well_free_energy_density",
    "evaluate_binary_free_energy",
]
