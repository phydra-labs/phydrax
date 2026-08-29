#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractVectorSpace,
    ArraySpace,
    FunctionLinearOperator,
    LinearSolvePolicy,
)


class CoordinateObservation(StrictModule, NonTrainableState):
    """Fixed canonical-coordinate observations with an exact transpose action."""

    state_space: AbstractVectorSpace
    indices: Array
    weights: Array
    operator: FunctionLinearOperator
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_space: AbstractVectorSpace,
        indices: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        observation_id: str = "coordinate-observation",
    ):
        if not isinstance(state_space, AbstractVectorSpace):
            raise TypeError("state_space must be AbstractVectorSpace.")
        indices_ = jnp.asarray(indices, dtype=jnp.int32)
        if indices_.ndim != 1 or indices_.size == 0:
            raise ValueError("Observation indices must be one non-empty rank-1 array.")
        if bool(jnp.any((indices_ < 0) | (indices_ >= state_space.size))):
            raise ValueError("Observation indices are out of state-space bounds.")
        weights_ = jnp.ones(indices_.shape) if weights is None else jnp.asarray(weights)
        if weights_.shape != indices_.shape or bool(jnp.any(~jnp.isfinite(weights_))):
            raise ValueError("Observation weights must be finite and match indices.")
        target = ArraySpace((indices_.size,), dtype=weights_.dtype)

        def apply(state):
            return weights_ * state_space.flatten(state)[indices_]

        def transpose(value):
            coordinates = (
                jnp.zeros((state_space.size,), dtype=jnp.result_type(value, weights_))
                .at[indices_]
                .add(weights_ * value)
            )
            return state_space.unflatten(coordinates)

        identifier = str(observation_id)
        if not identifier:
            raise ValueError("observation_id must be non-empty.")
        resolved_id = canonical_fingerprint(
            {
                "kind": "coordinate-observation",
                "declared_id": identifier,
                "space": state_space.space_id,
                "indices": indices_.tolist(),
                "weight_shape": list(weights_.shape),
            }
        )
        self.state_space = state_space
        self.indices = indices_
        self.weights = weights_
        self.operator = FunctionLinearOperator(
            apply,
            source=state_space,
            target=target,
            transpose_action=transpose,
            operator_id=resolved_id,
            closure_convert=False,
        )
        self.observation_id = resolved_id

    def evaluate(self, state: object, /) -> Array:
        return self.operator.mv(state)

    def transpose(self, value: ArrayLike, /):
        return self.operator.transpose_mv(value)


class FiniteElementLeastSquaresObjective(StrictModule, NonTrainableState):
    observation: CoordinateObservation
    target: Array
    precision: Array
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        observation: CoordinateObservation,
        target: ArrayLike,
        /,
        *,
        precision: ArrayLike = 1.0,
        objective_id: str = "finite-element-least-squares",
    ):
        if not isinstance(observation, CoordinateObservation):
            raise TypeError("observation must be CoordinateObservation.")
        target_ = jnp.asarray(target)
        precision_ = jnp.asarray(precision)
        if target_.shape != observation.weights.shape:
            raise ValueError("Objective target must match observation shape.")
        precision_ = jnp.broadcast_to(precision_, target_.shape)
        if bool(jnp.any(~jnp.isfinite(precision_) | (precision_ <= 0.0))):
            raise ValueError("Objective precision must be positive and finite.")
        identifier = str(objective_id)
        if not identifier:
            raise ValueError("objective_id must be non-empty.")
        self.observation = observation
        self.target = target_
        self.precision = precision_
        self.objective_id = canonical_fingerprint(
            {
                "kind": "finite-element-least-squares",
                "declared_id": identifier,
                "observation": observation.observation_id,
                "shape": list(target_.shape),
            }
        )

    def residual(self, state: object, /) -> Array:
        return self.observation.evaluate(state) - self.target

    def value(self, state: object, /) -> Array:
        residual = self.residual(state)
        return 0.5 * jnp.sum(self.precision * jnp.abs(residual) ** 2)

    def state_gradient(self, state: object, /):
        residual = self.residual(state)
        return self.observation.transpose(self.precision * residual)


class FiniteElementAdjointResult(StrictModule):
    adjoint: object
    state_gradient: object
    objective_value: Array


def solve_finite_element_adjoint(
    compiled_problem,
    solution: object,
    objective: FiniteElementLeastSquaresObjective,
    args: object = None,
    /,
    *,
    linear_policy: LinearSolvePolicy | None = None,
) -> FiniteElementAdjointResult:
    if not isinstance(objective, FiniteElementLeastSquaresObjective):
        raise TypeError("objective must be FiniteElementLeastSquaresObjective.")
    if not objective.observation.state_space.compatible(compiled_problem.state_space):
        raise ValueError("Objective and compiled problem state spaces are incompatible.")
    gradient = objective.state_gradient(solution)
    adjoint = compiled_problem.solve_adjoint(
        solution,
        gradient,
        args,
        linear_policy=linear_policy,
    )
    return FiniteElementAdjointResult(
        adjoint=adjoint,
        state_gradient=gradient,
        objective_value=objective.value(solution),
    )


def finite_element_parameter_gradient(
    adjoint_result: FiniteElementAdjointResult,
    residual_parameter_pullback: Callable,
    parameter: object,
    /,
    *,
    direct_objective_gradient: object = None,
):
    if not isinstance(adjoint_result, FiniteElementAdjointResult):
        raise TypeError("adjoint_result must be FiniteElementAdjointResult.")
    if not callable(residual_parameter_pullback):
        raise TypeError("residual_parameter_pullback must be callable.")
    pulled = residual_parameter_pullback(adjoint_result.adjoint, parameter)
    if direct_objective_gradient is None:
        return -pulled
    return direct_objective_gradient - pulled


__all__ = [
    "CoordinateObservation",
    "FiniteElementAdjointResult",
    "FiniteElementLeastSquaresObjective",
    "finite_element_parameter_gradient",
    "solve_finite_element_adjoint",
]
