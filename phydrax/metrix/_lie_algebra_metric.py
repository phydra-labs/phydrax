#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax import ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    LocalBlockFactorization,
    prepare_local_block_factorization,
    solve_local_blocks,
)
from ._lie_group import AbstractLieGroup


class LieAlgebraCoordinateMetric(StrictModule, NonTrainableState):
    """Frobenius metric represented in one Lie group's ``hat`` coordinates."""

    group: AbstractLieGroup
    basis: Array
    gram: Array
    factorization: LocalBlockFactorization
    symmetry_residual: Array
    reconstruction_residual: Array
    valid: Array
    dimension: int = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(self, group: AbstractLieGroup, /):
        if not isinstance(group, AbstractLieGroup):
            raise TypeError("group must implement AbstractLieGroup.")
        if len(group.algebra_shape) != 1:
            raise ValueError("Lie-algebra coordinate metrics require vector coordinates.")
        dimension = group.algebra_shape[0]
        coordinates = jnp.eye(dimension)
        basis = jax.vmap(group.hat)(coordinates)
        gram = jnp.real(ein.contract("aij,bij->ab", jnp.conj(basis), basis))
        factorization = prepare_local_block_factorization(
            gram[None, ...],
            positive_definite=True,
        )
        symmetry_residual = jnp.max(jnp.abs(gram - gram.T))
        factor = factorization.factors[0]
        reconstruction_residual = jnp.max(jnp.abs(factor @ factor.T - gram))
        valid = (
            ~factorization.failed_blocks[0]
            & jnp.all(jnp.isfinite(gram))
            & jnp.isfinite(symmetry_residual)
            & jnp.isfinite(reconstruction_residual)
        )
        if not bool(valid):
            raise ValueError("Lie-algebra coordinate Gram matrix is not finite SPD.")
        self.group = group
        self.basis = basis
        self.gram = gram
        self.factorization = factorization
        self.symmetry_residual = symmetry_residual
        self.reconstruction_residual = reconstruction_residual
        self.valid = valid
        self.dimension = dimension
        self.metric_id = canonical_fingerprint(
            {
                "kind": "lie-algebra-coordinate-metric",
                "group": group.group_id,
                "gram": array_tree_fingerprint(gram),
                "inner_product": "real-frobenius",
            }
        )

    def solve(self, covector: ArrayLike, /) -> Array:
        """Apply the coordinate inverse Gram map to trailing algebra coordinates."""
        value = jnp.asarray(covector)
        if value.ndim < 1 or value.shape[-1] != self.dimension:
            raise ValueError(f"covector must have trailing shape ({self.dimension},).")
        flattened = value.reshape((-1, self.dimension))
        right_hand_side = jnp.swapaxes(flattened, 0, 1)[None, ...]
        solved, failed = solve_local_blocks(self.factorization, right_hand_side)
        solved = jnp.swapaxes(solved[0], 0, 1).reshape(value.shape)
        return eqx.error_if(
            solved,
            failed,
            "Lie-algebra coordinate metric solve failed.",
        )

    def sample_momentum(
        self,
        key: Key[Array, ""],
        leading_shape: Sequence[int],
        /,
        *,
        dtype: object = jnp.float64,
    ) -> Array:
        """Sample covector momentum with covariance equal to the Gram matrix."""
        leading = tuple(int(size) for size in leading_shape)
        if any(size <= 0 for size in leading):
            raise ValueError("leading_shape dimensions must be positive.")
        normal = jr.normal(key, leading + (self.dimension,), dtype=dtype)
        return ein.contract("ab,...b->...a", self.factorization.factors[0], normal)

    def kinetic_energy(self, momentum: ArrayLike, /) -> Array:
        """Return one half of the inverse-metric quadratic form."""
        value = jnp.asarray(momentum)
        velocity = self.solve(value)
        return 0.5 * jnp.sum(value * velocity)


__all__ = ["LieAlgebraCoordinateMetric"]
