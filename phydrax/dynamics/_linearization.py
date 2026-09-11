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
    JacobianLinearOperator,
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
        linearization = evolution.state_linearization(
            state_array,
            source,
            target,
            args,
        )
        if not isinstance(linearization.source, ArraySpace) or not isinstance(
            linearization.target, ArraySpace
        ):
            raise TypeError(
                "Evolution state linearization must use ArraySpace endpoints."
            )
        if (
            linearization.source.shape != evolution.state_layout.shape
            or linearization.target.shape != evolution.state_layout.shape
        ):
            raise ValueError(
                "Evolution state linearization spaces must match the state layout."
            )
        self.source = linearization.source
        self.target = linearization.target
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
                    "evolution": evolution.evolution_id,
                    "source": linearization.source.space_id,
                    "target": linearization.target.space_id,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not self.operator_id:
            raise ValueError("operator_id must be non-empty.")
        self.linearization = linearization

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


class EvolutionArgumentJacobianAction(AbstractLinearOperator):
    """Matrix-free endpoint derivative with respect to one argument PyTree."""

    evolution: AbstractDifferentiableEvolution
    state: Array
    source_coordinate: Array
    target_coordinate: Array
    args: Any
    primal: Any
    linearization: PreparedLinearization
    operator: JacobianLinearOperator

    def __init__(
        self,
        evolution: AbstractDifferentiableEvolution,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any,
        /,
        *,
        operator_id: str | None = None,
    ):
        if not isinstance(evolution, AbstractDifferentiableEvolution):
            raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
        state_array = jnp.asarray(state)
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        linearization = evolution.argument_linearization(
            state_array,
            source,
            target,
            args,
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "evolution-argument-jacobian",
                    "evolution": evolution.evolution_id,
                    "source": linearization.source.space_id,
                    "target": linearization.target.space_id,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        operator = JacobianLinearOperator(
            linearization,
            operator_id=identifier,
        )
        self.evolution = evolution
        self.state = state_array
        self.source_coordinate = source
        self.target_coordinate = target
        self.args = args
        self.primal = linearization.primal
        self.linearization = linearization
        self.operator = operator
        self.source = operator.source
        self.target = operator.target
        self.properties = operator.properties
        self.capabilities = operator.capabilities
        self.batch_shape = operator.batch_shape
        self.operator_id = operator.operator_id

    def mv(self, vector: Any, /) -> Any:
        return self.operator.mv(vector)

    def transpose_mv(self, vector: Any, /) -> Any:
        return self.operator.transpose_mv(vector)

    def adjoint_mv(self, vector: Any, /) -> Any:
        return self.operator.adjoint_mv(vector)

    def _materialize(self, /) -> Array:
        return self.operator._materialize()


__all__ = ["EvolutionArgumentJacobianAction", "EvolutionJacobianAction"]
