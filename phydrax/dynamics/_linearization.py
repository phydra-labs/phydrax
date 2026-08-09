#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._evolution import AbstractDifferentiableEvolution, EvolutionStep


class EvolutionJacobianAction(StrictModule):
    """Matrix-free tangent and cotangent actions for one evolution segment."""

    evolution: AbstractDifferentiableEvolution
    state: Array
    source_coordinate: Array
    target_coordinate: Array
    args: Any
    primal: EvolutionStep

    def __init__(
        self,
        evolution: AbstractDifferentiableEvolution,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        /,
        *,
        args: Any = None,
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

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.evolution.state_layout.shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.evolution.state_layout.shape

    def _local_map(self, local: Array, /) -> Array:
        geometry = self.evolution.state_layout.geometry
        if geometry.trivial:
            return self.evolution.advance(
                local,
                self.source_coordinate,
                self.target_coordinate,
                self.args,
            ).final_state
        perturbed = geometry.retract(self.state, local)
        endpoint = self.evolution.advance(
            perturbed,
            self.source_coordinate,
            self.target_coordinate,
            self.args,
        ).final_state
        return geometry.inverse_retract(self.primal.final_state, endpoint)

    def mv(self, vector: ArrayLike, /) -> Array:
        tangent = jnp.asarray(vector)
        if tangent.shape != self.input_shape:
            raise ValueError(
                f"vector must have shape {self.input_shape}; got {tangent.shape}."
            )
        return self.evolution.tangent_action(
            self.state,
            tangent,
            self.source_coordinate,
            self.target_coordinate,
            self.args,
        ).tangent

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        cotangent = jnp.asarray(vector)
        if cotangent.shape != self.output_shape:
            raise ValueError(
                f"vector must have shape {self.output_shape}; got {cotangent.shape}."
            )
        geometry = self.evolution.state_layout.geometry
        origin = self.state if geometry.trivial else jnp.zeros_like(self.state)
        _, pullback = jax.vjp(self._local_map, origin)
        return pullback(cotangent)[0]

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


__all__ = ["EvolutionJacobianAction"]
