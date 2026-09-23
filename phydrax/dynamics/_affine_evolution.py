#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, matrix_exponential_phi_combination_action


class AffineLinearEvolutionResult(StrictModule):
    value: Array
    successful: Array
    finite: Array


class PreparedAffineLinearEvolution(StrictModule, NonTrainableState):
    operator: DenseLinearOperator
    source: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, matrix: ArrayLike, source: ArrayLike, /):
        matrix_ = jnp.asarray(matrix)
        source_ = jnp.asarray(source, dtype=matrix_.dtype)
        if matrix_.ndim != 2 or matrix_.shape[0] != matrix_.shape[1]:
            raise ValueError("Affine evolution matrix must be square.")
        if source_.shape != (matrix_.shape[0],):
            raise ValueError("Affine evolution source has incompatible shape.")
        self.operator = DenseLinearOperator(matrix_)
        self.source = source_
        self.prepared_id = canonical_fingerprint(
            {"kind": "affine-linear-evolution", "dimension": matrix_.shape[0]}
        )

    def step(
        self, state: ArrayLike, duration: ArrayLike, /
    ) -> AffineLinearEvolutionResult:
        state_ = jnp.asarray(state, dtype=self.source.dtype)
        if state_.shape != self.source.shape:
            raise ValueError("Affine evolution state has incompatible shape.")
        time = jnp.asarray(duration, dtype=state_.real.dtype)
        action = matrix_exponential_phi_combination_action(
            self.operator,
            (state_, self.source),
            time,
        )
        value = action.value
        finite = jnp.all(jnp.isfinite(value))
        return AffineLinearEvolutionResult(
            value,
            action.successful & finite,
            finite,
        )


__all__ = ["AffineLinearEvolutionResult", "PreparedAffineLinearEvolution"]
