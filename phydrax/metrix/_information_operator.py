#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    FunctionLinearOperator,
    LinearSolvePlan,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve,
)


class _DampedInformationAction(StrictModule):
    action: Callable[[Array], Array]
    damping: Array

    def __init__(self, action: Callable[[Array], Array], damping: ArrayLike, /):
        self.action = action
        self.damping = jnp.asarray(damping)

    def __call__(self, vector: Array, /) -> Array:
        return jnp.asarray(self.action(vector)) + self.damping * vector


class InformationMetricOperator(StrictModule):
    """Matrix-free symmetric information metric over one array space."""

    operator: FunctionLinearOperator
    space: ArraySpace
    damping: Array
    metric_id: str

    def __init__(
        self,
        action: Callable[[Array], Array],
        coordinates: ArrayLike,
        /,
        *,
        damping: ArrayLike = 0.0,
        metric_id: str,
    ):
        if not callable(action):
            raise TypeError("action must be callable.")
        point = jnp.asarray(coordinates)
        damping_ = jnp.asarray(damping, dtype=point.real.dtype)
        if damping_.shape != ():
            raise ValueError("damping must be scalar.")
        identifier = str(metric_id)
        if not identifier:
            raise ValueError("metric_id must be non-empty.")
        space = ArraySpace(point.shape, dtype=point.dtype, space_id=f"{identifier}:space")
        damped = _DampedInformationAction(action, damping_)
        self.operator = FunctionLinearOperator(
            damped,
            source=space,
            target=space,
            transpose_action=damped,
            operator_id=f"{identifier}:operator",
        )
        self.space = space
        self.damping = damping_
        self.metric_id = identifier

    @property
    def shape(self) -> tuple[int, ...]:
        return self.space.shape

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.operator.mv(jnp.asarray(vector))

    def solve(
        self,
        cotangent: ArrayLike,
        /,
        *,
        policy: LinearSolvePolicy | LinearSolvePlan | None = None,
        initial_guess: ArrayLike | None = None,
    ) -> LinearSolveResult:
        problem = LinearSystem(
            self.operator,
            problem_id=f"{self.metric_id}:duality",
        )
        return solve(
            problem,
            jnp.asarray(cotangent),
            policy=policy,
            initial_guess=None if initial_guess is None else jnp.asarray(initial_guess),
        )

    def materialize(self, /, *, maximum_size: int = 256) -> Array:
        size = self.space.size
        if size > int(maximum_size):
            raise ValueError("Information metric exceeds bounded materialization size.")
        identity = jnp.eye(size, dtype=self.space.dtype)
        columns = jax.vmap(
            lambda column: self.space.flatten(self.mv(self.space.unflatten(column)))
        )(identity)
        return jnp.swapaxes(columns, -1, -2)


def pulled_back_information_operator(
    parameter_map: Callable[[Array], Array],
    parameters: ArrayLike,
    target_metric: InformationMetricOperator,
    /,
    *,
    damping: ArrayLike = 0.0,
    metric_id: str = "pullback-information",
) -> InformationMetricOperator:
    """Return ``J^T G J`` without materializing ``J`` or ``G``."""
    if not callable(parameter_map):
        raise TypeError("parameter_map must be callable.")
    parameters_ = jnp.asarray(parameters)
    target = jnp.asarray(parameter_map(parameters_))
    if target.shape != target_metric.shape:
        raise ValueError("parameter_map output must match target metric coordinates.")
    _, linearized = jax.linearize(parameter_map, parameters_)

    def action(vector: Array) -> Array:
        target_vector = linearized(vector)
        target_action = target_metric.mv(target_vector)
        return jax.linear_transpose(linearized, parameters_)(target_action)[0]

    return InformationMetricOperator(
        action,
        parameters_,
        damping=damping,
        metric_id=metric_id,
    )


__all__ = ["InformationMetricOperator", "pulled_back_information_operator"]
