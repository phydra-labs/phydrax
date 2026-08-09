#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ._parameter_geometry import ParameterGeometry


class ArmijoLineSearch(StrictModule):
    """Static frozen-objective Armijo backtracking policy."""

    initial_rate: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    minimum_rate: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_rate: float = 1.0,
        contraction: float = 0.5,
        sufficient_decrease: float = 1e-4,
        maximum_steps: int = 20,
        minimum_rate: float = 1e-12,
    ):
        initial = float(initial_rate)
        reduction = float(contraction)
        decrease = float(sufficient_decrease)
        steps = int(maximum_steps)
        minimum = float(minimum_rate)
        if not isfinite(initial) or initial <= 0.0:
            raise ValueError("initial_rate must be finite and positive.")
        if not isfinite(reduction) or not 0.0 < reduction < 1.0:
            raise ValueError("contraction must lie strictly between zero and one.")
        if not isfinite(decrease) or not 0.0 < decrease < 1.0:
            raise ValueError(
                "sufficient_decrease must lie strictly between zero and one."
            )
        if steps <= 0:
            raise ValueError("maximum_steps must be positive.")
        if not isfinite(minimum) or minimum <= 0.0 or minimum > initial:
            raise ValueError(
                "minimum_rate must be finite, positive, and no larger than initial_rate."
            )
        self.initial_rate = initial
        self.contraction = reduction
        self.sufficient_decrease = decrease
        self.maximum_steps = steps
        self.minimum_rate = minimum


class ArmijoResult(StrictModule):
    """Result of one frozen-objective retraction line search."""

    parameters: PyTree[Array]
    value: Array
    rate: Array
    evaluations: Array
    accepted: Array
    directional_derivative: Array

    def __init__(
        self,
        *,
        parameters: PyTree[Array],
        value: Array,
        rate: Array,
        evaluations: Array,
        accepted: Array,
        directional_derivative: Array,
    ):
        self.parameters = parameters
        self.value = jnp.asarray(value)
        self.rate = jnp.asarray(rate)
        self.evaluations = jnp.asarray(evaluations)
        self.accepted = jnp.asarray(accepted, dtype=bool)
        self.directional_derivative = jnp.asarray(directional_derivative)


def armijo_backtracking(
    value_function: Callable[[PyTree[Any]], Array],
    parameter_geometry: ParameterGeometry,
    parameters: PyTree[Any],
    value: Array,
    gradient: PyTree[Any],
    direction: PyTree[Any],
    /,
    *,
    policy: ArmijoLineSearch,
) -> ArmijoResult:
    """Search one retraction ray while reusing the caller's frozen objective closure."""

    if not isinstance(parameter_geometry, ParameterGeometry):
        raise TypeError("parameter_geometry must be a ParameterGeometry.")
    if not isinstance(policy, ArmijoLineSearch):
        raise TypeError("policy must be an ArmijoLineSearch.")
    initial_value = jnp.asarray(value)
    if initial_value.shape != ():
        raise ValueError("Armijo value must be scalar.")
    initial_value = eqx.error_if(
        initial_value,
        ~jnp.isfinite(initial_value),
        "Armijo initial value must be finite.",
    )
    directional = parameter_geometry.inner(parameters, gradient, direction)
    scalar_dtype = jnp.result_type(initial_value, directional, float)
    initial_rate = jnp.asarray(policy.initial_rate, dtype=scalar_dtype)
    minimum_rate = jnp.asarray(policy.minimum_rate, dtype=scalar_dtype)
    contraction = jnp.asarray(policy.contraction, dtype=scalar_dtype)
    sufficient_decrease = jnp.asarray(policy.sufficient_decrease, dtype=scalar_dtype)

    def condition(carry):
        iteration, rate, _, _, accepted = carry
        return (
            (iteration < policy.maximum_steps)
            & (~accepted)
            & (rate >= minimum_rate)
            & jnp.isfinite(rate)
        )

    def body(carry):
        iteration, rate, _, _, _ = carry
        tangent_step = jax.tree.map(lambda leaf: rate * leaf, direction)
        candidate = parameter_geometry.retract(parameters, tangent_step)
        candidate_value = jnp.asarray(value_function(candidate)).reshape(())
        finite = jnp.isfinite(candidate_value) & parameter_geometry.contains(candidate)
        armijo_bound = initial_value + sufficient_decrease * rate * directional
        accepted = finite & (directional < 0.0) & (candidate_value <= armijo_bound)
        next_rate = jnp.where(accepted, rate, rate * contraction)
        return iteration + 1, next_rate, candidate, candidate_value, accepted

    initial_carry = (
        jnp.asarray(0, dtype=jnp.int32),
        initial_rate,
        parameters,
        initial_value,
        jnp.asarray(False),
    )
    evaluations, next_rate, candidate, candidate_value, accepted = jax.lax.while_loop(
        condition,
        body,
        initial_carry,
    )
    accepted_rate = jnp.where(accepted, next_rate, jnp.zeros_like(next_rate))
    accepted_parameters = jax.tree.map(
        lambda proposed, original: jnp.where(accepted, proposed, original),
        candidate,
        parameters,
    )
    accepted_value = jnp.where(accepted, candidate_value, initial_value)
    return ArmijoResult(
        parameters=accepted_parameters,
        value=accepted_value,
        rate=accepted_rate,
        evaluations=evaluations,
        accepted=accepted,
        directional_derivative=directional,
    )


__all__ = ["ArmijoLineSearch", "ArmijoResult", "armijo_backtracking"]
