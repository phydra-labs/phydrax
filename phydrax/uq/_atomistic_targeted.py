#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..atomistic._alchemical import PreparedAlchemicalTransformation
from ._posterior import AbstractBijector
from ._targeted_free_energy import (
    AbstractReducedPotential,
    ReducedPotentialEvaluation,
)


class CenterOfMassPreservingBijector(AbstractBijector):
    """Lift an exact internal-coordinate bijector while preserving center of mass."""

    internal: AbstractBijector
    masses: Array
    translation_basis: Array
    internal_basis: Array
    event_shape: tuple[int, int] = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)

    def __init__(
        self,
        internal: AbstractBijector,
        masses: ArrayLike,
        /,
        *,
        chart_id: str | None = None,
    ):
        if not isinstance(internal, AbstractBijector):
            raise TypeError("internal must implement AbstractBijector.")
        mass = np.asarray(masses, dtype=float).reshape((-1,))
        if mass.size < 2 or np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError(
                "Center-of-mass chart requires at least two positive masses."
            )
        atom_count = int(mass.size)
        internal_shape = (3 * atom_count - 3,)
        if (
            internal.forward_shape(internal_shape) != internal_shape
            or internal.inverse_shape(internal_shape) != internal_shape
        ):
            raise ValueError(
                "Internal bijector shape does not match translation-free space."
            )
        translation_atom = np.sqrt(mass) / np.sqrt(np.sum(mass))
        complement = []
        for index in range(atom_count):
            vector = np.zeros((atom_count,), dtype=float)
            vector[index] = 1.0
            vector = vector - np.dot(vector, translation_atom) * translation_atom
            for basis in complement:
                vector = vector - np.dot(vector, basis) * basis
            norm = float(np.sqrt(np.dot(vector, vector)))
            if norm > 128.0 * np.finfo(float).eps:
                complement.append(vector / norm)
            if len(complement) == atom_count - 1:
                break
        if len(complement) != atom_count - 1:
            raise ValueError("Could not construct translation-free coordinate basis.")
        atom_basis = np.stack(complement, axis=1)
        identity = np.eye(3)
        translation_basis = np.kron(translation_atom[:, None], identity)
        internal_basis = np.kron(atom_basis, identity)
        identifier = chart_id or canonical_fingerprint(
            {
                "kind": "center-of-mass-preserving-bijector",
                "internal_type": f"{type(internal).__module__}.{type(internal).__qualname__}",
                "arrays": array_tree_fingerprint(
                    {
                        "masses": mass,
                        "translation_basis": translation_basis,
                        "internal_basis": internal_basis,
                    }
                ),
            }
        )
        self.internal = internal
        self.masses = jnp.asarray(mass)
        self.translation_basis = jnp.asarray(translation_basis)
        self.internal_basis = jnp.asarray(internal_basis)
        self.event_shape = (atom_count, 3)
        self.chart_id = identifier

    def forward_shape(self, raw_shape: tuple[int, ...], /) -> tuple[int, ...]:
        shape = tuple(int(size) for size in raw_shape)
        if shape != self.event_shape:
            raise ValueError(
                f"Expected Cartesian event shape {self.event_shape}; got {shape}."
            )
        return shape

    def inverse_shape(self, physical_shape: tuple[int, ...], /) -> tuple[int, ...]:
        return self.forward_shape(physical_shape)

    def _coordinates(self, positions: Array, /) -> tuple[Array, Array]:
        weighted = (positions * jnp.sqrt(self.masses)[:, None]).reshape((-1,))
        translation = contract("di,d->i", self.translation_basis, weighted)
        internal = contract("di,d->i", self.internal_basis, weighted)
        return translation, internal

    def _positions(self, translation: Array, internal: Array, /) -> Array:
        weighted = contract("di,i->d", self.translation_basis, translation) + contract(
            "di,i->d", self.internal_basis, internal
        )
        return weighted.reshape(self.event_shape) / jnp.sqrt(self.masses)[:, None]

    def forward(self, value: ArrayLike, /) -> Array:
        positions = jnp.asarray(value)
        translation, internal = self._coordinates(positions)
        mapped = self.internal.forward(internal)
        return self._positions(translation, mapped)

    def inverse(self, value: ArrayLike, /) -> Array:
        positions = jnp.asarray(value)
        translation, internal = self._coordinates(positions)
        mapped = self.internal.inverse(internal)
        return self._positions(translation, mapped)

    def forward_log_det_jacobian(self, value: ArrayLike, /) -> Array:
        _, internal = self._coordinates(jnp.asarray(value))
        return self.internal.forward_log_det_jacobian(internal)


class AlchemicalEndpointReducedPotential(AbstractReducedPotential):
    transformation: PreparedAlchemicalTransformation
    lambda_value: float = eqx.field(static=True)
    event_shape: tuple[int, int] = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)

    def __init__(
        self,
        transformation: PreparedAlchemicalTransformation,
        lambda_value: float,
        /,
    ):
        if not isinstance(transformation, PreparedAlchemicalTransformation):
            raise TypeError("transformation must be PreparedAlchemicalTransformation.")
        value = float(lambda_value)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("lambda_value must lie in [0, 1].")
        if not bool(jnp.all(transformation.active_mask)) or bool(
            jnp.any(transformation.dummy_mask)
        ):
            raise ValueError(
                "Targeted endpoint maps require identical active support without dummies."
            )
        self.transformation = transformation
        self.lambda_value = value
        self.event_shape = (transformation.plan.atom_capacity, 3)
        self.potential_id = canonical_fingerprint(
            {
                "kind": "alchemical-endpoint-reduced-potential",
                "transformation": transformation.prepared_id,
                "lambda": value.hex(),
                "beta": transformation.plan.beta.hex(),
            }
        )

    def evaluate(self, value: ArrayLike, /) -> ReducedPotentialEvaluation:
        positions = jnp.asarray(value)
        if positions.shape != self.event_shape:
            raise ValueError(f"value must have event shape {self.event_shape}.")
        evaluation = self.transformation.evaluate(positions, self.lambda_value)
        reduced = self.transformation.plan.beta * evaluation.energy
        valid = evaluation.successful & jnp.isfinite(reduced)
        return ReducedPotentialEvaluation(reduced, valid, self.potential_id)


__all__ = [
    "AlchemicalEndpointReducedPotential",
    "CenterOfMassPreservingBijector",
]
