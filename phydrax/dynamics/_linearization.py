#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    LinearizationPolicy,
    OperatorCapabilities,
    OperatorProperties,
    PreparedLinearization,
)
from ._evolution import AbstractDifferentiableEvolution, EvolutionStep


class EvolutionJacobianAction(AbstractLinearOperator):
    """Matrix-free tangent and cotangent actions for one evolution segment."""

    evolution: AbstractDifferentiableEvolution
    state: Array
    source_coordinate: Array
    target_coordinate: Array
    args: Any
    primal: EvolutionStep
    linearization: PreparedLinearization

    def __init__(
        self,
        evolution: AbstractDifferentiableEvolution,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        /,
        *,
        args: Any = None,
        operator_id: str | None = None,
    ):
        if not isinstance(evolution, AbstractDifferentiableEvolution):
            raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
        state_array = jnp.asarray(state)
        if state_array.shape != evolution.state_layout.shape:
            raise ValueError(
                f"state must have shape {evolution.state_layout.shape}; got {state_array.shape}."
            )
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        if source.shape != () or target.shape != ():
            raise ValueError("Evolution segment coordinates must be scalar.")
        self.evolution = evolution
        self.state = state_array
        self.source_coordinate = source
        self.target_coordinate = target
        self.args = args
        self.primal = evolution.advance(state_array, source, target, args)
        space = ArraySpace(evolution.state_layout.shape, dtype=state_array.dtype)
        self.source = space
        self.target = space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
        )
        self.batch_shape = ()
        self.operator_id = (
            canonical_fingerprint(
                {
                    "kind": "evolution-jacobian",
                    "evolution": type(evolution).__qualname__,
                    "space": space.space_id,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not self.operator_id:
            raise ValueError("operator_id must be non-empty.")

        def pushforward(tangent):
            return evolution.tangent_action(
                state_array,
                tangent,
                source,
                target,
                args,
            ).tangent

        transposed = jax.linear_transpose(pushforward, space.zeros())

        def pullback(cotangent):
            return transposed(cotangent)[0]

        geometry = evolution.state_layout.geometry
        linearization_point = (
            state_array if geometry.trivial else jnp.zeros_like(state_array)
        )
        linearization_primal = (
            self.primal.final_state if geometry.trivial else jnp.zeros_like(state_array)
        )
        self.linearization = PreparedLinearization(
            source=space,
            target=space,
            point=linearization_point,
            primal=linearization_primal,
            pushforward=pushforward,
            pullback=pullback,
            policy=LinearizationPolicy(),
            linearization_id=f"{self.operator_id}:prepared",
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.evolution.state_layout.shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.evolution.state_layout.shape

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.linearization.jvp(vector)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        return self.linearization.vjp(vector)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return jnp.conj(self.transpose_mv(jnp.conj(jnp.asarray(vector))))

    def as_dense(self, /, *, max_dimension: int = 128) -> Array:
        dimension = self.evolution.state_layout.size
        maximum = int(max_dimension)
        if maximum < 1:
            raise ValueError("max_dimension must be positive.")
        if dimension > maximum:
            raise ValueError(
                f"Dense evolution Jacobian dimension {dimension} exceeds {maximum}."
            )
        basis = jnp.eye(dimension, dtype=self.state.dtype).reshape(
            (dimension,) + self.input_shape
        )
        columns = jax.vmap(self.mv)(basis).reshape((dimension, dimension))
        return columns.T

    def _materialize(self, /) -> Array:
        return self.as_dense(max_dimension=self.evolution.state_layout.size)


__all__ = ["EvolutionJacobianAction"]
