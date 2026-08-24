#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
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
    precision: GeometryPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope

    def __init__(
        self,
        action: Callable[[Array], Array],
        coordinates: ArrayLike,
        /,
        *,
        damping: ArrayLike = 0.0,
        metric_id: str,
        precision: GeometryPrecisionPolicy | None = None,
    ):
        if not callable(action):
            raise TypeError("action must be callable.")
        original = jnp.asarray(coordinates)
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
        precision_.validate_coordinates(original)
        point = precision_.compute(original)
        damping_ = precision_.compute(damping)
        if damping_.shape != ():
            raise ValueError("damping must be scalar.")
        identifier = str(metric_id)
        if not identifier:
            raise ValueError("metric_id must be non-empty.")
        space = ArraySpace(point.shape, dtype=point.dtype, space_id=f"{identifier}:space")
        damped = _DampedInformationAction(
            lambda vector: precision_.compute(action(precision_.compute(vector))),
            damping_,
        )
        self.operator = FunctionLinearOperator(
            damped,
            source=space,
            target=space,
            transpose_action=damped,
            operator_id=f"{identifier}:operator",
        )
        self.space = space
        self.damping = damping_
        self.precision = precision_
        self.precision_evidence = precision_.evidence_for(original)
        self.metric_id = identifier

    @property
    def shape(self) -> tuple[int, ...]:
        return self.space.shape

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.operator.mv(self.precision.compute(vector))

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
            self.precision.compute(cotangent),
            policy=policy,
            initial_guess=(
                None if initial_guess is None else self.precision.compute(initial_guess)
            ),
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
    precision: GeometryPrecisionPolicy | None = None,
) -> InformationMetricOperator:
    """Return ``J^T G J`` without materializing ``J`` or ``G``."""
    if not callable(parameter_map):
        raise TypeError("parameter_map must be callable.")
    parameters_original = jnp.asarray(parameters)
    precision_ = target_metric.precision if precision is None else precision
    if not isinstance(precision_, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
    parameters_ = precision_.compute(parameters_original)
    target = precision_.compute(parameter_map(parameters_))
    if target.shape != target_metric.shape:
        raise ValueError("parameter_map output must match target metric coordinates.")
    _, linearized = jax.linearize(parameter_map, parameters_)

    def action(vector: Array) -> Array:
        target_vector = precision_.compute(linearized(precision_.compute(vector)))
        target_action = target_metric.mv(target_vector)
        return precision_.compute(
            jax.linear_transpose(linearized, parameters_)(target_action)[0]
        )

    return InformationMetricOperator(
        action,
        parameters_,
        damping=damping,
        metric_id=metric_id,
        precision=precision_,
    )


__all__ = ["InformationMetricOperator", "pulled_back_information_operator"]
