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
from ..._tree_math import (
    tree_allfinite as _tree_allfinite,
    tree_inner as _tree_inner,
    tree_where as _tree_where,
)


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
    """Result of one frozen-objective line search."""

    parameters: PyTree[Array]
    value: Array
    rate: Array
    evaluations: Array
    accepted: Array
    directional_derivative: Array
    finite_candidate_seen: Array

    def __init__(
        self,
        *,
        parameters: PyTree[Array],
        value: Array,
        rate: Array,
        evaluations: Array,
        accepted: Array,
        directional_derivative: Array,
        finite_candidate_seen: Array,
    ):
        self.parameters = parameters
        self.value = jnp.asarray(value)
        self.rate = jnp.asarray(rate)
        self.evaluations = jnp.asarray(evaluations, dtype=jnp.int32)
        self.accepted = jnp.asarray(accepted, dtype=bool)
        self.directional_derivative = jnp.asarray(directional_derivative)
        self.finite_candidate_seen = jnp.asarray(finite_candidate_seen, dtype=bool)


def armijo_backtracking(
    value_function: Callable[[PyTree[Any]], Array],
    parameters: PyTree[Any],
    value: Array,
    direction: PyTree[Any],
    directional_derivative: Array,
    /,
    *,
    step: Callable[[PyTree[Any], PyTree[Any], Array], PyTree[Array]],
    contains: Callable[[PyTree[Any]], Array],
    policy: ArmijoLineSearch,
    maximum_evaluations: Any | None = None,
) -> ArmijoResult:
    """Search one candidate ray while reusing one frozen objective realization."""

    if not callable(value_function):
        raise TypeError("value_function must be callable.")
    if not callable(step):
        raise TypeError("step must be callable.")
    if not callable(contains):
        raise TypeError("contains must be callable.")
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
    directional = jnp.asarray(directional_derivative)
    if directional.shape != ():
        raise ValueError("Armijo directional derivative must be scalar.")
    scalar_dtype = jnp.result_type(initial_value, directional, float)
    initial_rate = jnp.asarray(policy.initial_rate, dtype=scalar_dtype)
    minimum_rate = jnp.asarray(policy.minimum_rate, dtype=scalar_dtype)
    contraction = jnp.asarray(policy.contraction, dtype=scalar_dtype)
    sufficient_decrease = jnp.asarray(policy.sufficient_decrease, dtype=scalar_dtype)
    evaluation_limit = (
        jnp.asarray(policy.maximum_steps, dtype=jnp.int32)
        if maximum_evaluations is None
        else jnp.maximum(
            jnp.asarray(maximum_evaluations, dtype=jnp.int32),
            0,
        )
    )

    def condition(carry):
        iteration, rate, _, _, _, _, accepted, _ = carry
        return (
            (iteration < policy.maximum_steps)
            & (iteration < evaluation_limit)
            & (~accepted)
            & (rate >= minimum_rate)
            & jnp.isfinite(rate)
        )

    def body(carry):
        iteration, rate, base, tangent, _, _, _, finite_seen = carry
        candidate = step(base, tangent, rate)
        candidate_value = jnp.asarray(value_function(candidate)).reshape(())
        finite = jnp.isfinite(candidate_value) & jnp.asarray(
            contains(candidate), dtype=bool
        )
        armijo_bound = initial_value + sufficient_decrease * rate * directional
        accepted = finite & (directional < 0.0) & (candidate_value <= armijo_bound)
        next_rate = jnp.where(accepted, rate, rate * contraction)
        return (
            iteration + 1,
            next_rate,
            base,
            tangent,
            candidate,
            candidate_value,
            accepted,
            finite_seen | finite,
        )

    initial_carry = (
        jnp.asarray(0, dtype=jnp.int32),
        initial_rate,
        parameters,
        direction,
        parameters,
        initial_value,
        jnp.asarray(False),
        jnp.asarray(False),
    )
    (
        evaluations,
        next_rate,
        _,
        _,
        candidate,
        candidate_value,
        accepted,
        finite_seen,
    ) = jax.lax.while_loop(condition, body, initial_carry)
    accepted_rate = jnp.where(accepted, next_rate, jnp.zeros_like(next_rate))
    accepted_parameters = _tree_where(accepted, candidate, parameters)
    accepted_value = jnp.where(accepted, candidate_value, initial_value)
    return ArmijoResult(
        parameters=accepted_parameters,
        value=accepted_value,
        rate=accepted_rate,
        evaluations=evaluations,
        accepted=accepted,
        directional_derivative=directional,
        finite_candidate_seen=finite_seen,
    )


class StrongWolfeLineSearch(StrictModule):
    """Bracket-and-zoom line search enforcing both strong-Wolfe inequalities."""

    initial_rate: float = eqx.field(static=True)
    expansion: float = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    curvature: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    minimum_rate: float = eqx.field(static=True)
    maximum_rate: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_rate: float = 1.0,
        expansion: float = 2.0,
        sufficient_decrease: float = 1e-4,
        curvature: float = 0.9,
        maximum_steps: int = 40,
        minimum_rate: float = 1e-12,
        maximum_rate: float = 1e6,
    ):
        initial = float(initial_rate)
        expansion_ = float(expansion)
        decrease = float(sufficient_decrease)
        curvature_ = float(curvature)
        steps = int(maximum_steps)
        minimum = float(minimum_rate)
        maximum = float(maximum_rate)
        if not isfinite(initial) or initial <= 0.0:
            raise ValueError("initial_rate must be finite and positive.")
        if not isfinite(expansion_) or expansion_ <= 1.0:
            raise ValueError("expansion must be finite and greater than one.")
        if not 0.0 < decrease < curvature_ < 1.0:
            raise ValueError(
                "Strong-Wolfe constants must satisfy 0 < sufficient_decrease "
                "< curvature < 1."
            )
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        if (
            not isfinite(minimum)
            or not isfinite(maximum)
            or minimum <= 0.0
            or maximum < initial
            or minimum > initial
        ):
            raise ValueError(
                "Rate bounds must be finite, positive, and contain initial_rate."
            )
        self.initial_rate = initial
        self.expansion = expansion_
        self.sufficient_decrease = decrease
        self.curvature = curvature_
        self.maximum_steps = steps
        self.minimum_rate = minimum
        self.maximum_rate = maximum


class StrongWolfeResult(StrictModule):
    """Accepted point and explicit evidence for both strong-Wolfe conditions."""

    parameters: PyTree[Array]
    value: Array
    gradient: PyTree[Array]
    rate: Array
    evaluations: Array
    accepted: Array
    initial_directional_derivative: Array
    directional_derivative: Array
    sufficient_decrease_satisfied: Array
    curvature_satisfied: Array
    finite_candidate_seen: Array

    def __init__(
        self,
        *,
        parameters: PyTree[Any],
        value: Any,
        gradient: PyTree[Any],
        rate: Any,
        evaluations: Any,
        accepted: Any,
        initial_directional_derivative: Any,
        directional_derivative: Any,
        sufficient_decrease_satisfied: Any,
        curvature_satisfied: Any,
        finite_candidate_seen: Any,
    ):
        self.parameters = parameters
        self.value = jnp.asarray(value)
        self.gradient = gradient
        self.rate = jnp.asarray(rate)
        self.evaluations = jnp.asarray(evaluations, dtype=jnp.int32)
        self.accepted = jnp.asarray(accepted, dtype=bool)
        self.initial_directional_derivative = jnp.asarray(initial_directional_derivative)
        self.directional_derivative = jnp.asarray(directional_derivative)
        self.sufficient_decrease_satisfied = jnp.asarray(
            sufficient_decrease_satisfied, dtype=bool
        )
        self.curvature_satisfied = jnp.asarray(curvature_satisfied, dtype=bool)
        self.finite_candidate_seen = jnp.asarray(finite_candidate_seen, dtype=bool)


def strong_wolfe_line_search(
    value_and_gradient: Callable[[PyTree[Any]], tuple[Array, PyTree[Array]]],
    parameters: PyTree[Any],
    value: Array,
    gradient: PyTree[Any],
    direction: PyTree[Any],
    /,
    *,
    step: Callable[[PyTree[Any], PyTree[Any], Array], PyTree[Array]],
    contains: Callable[[PyTree[Any]], Array],
    policy: StrongWolfeLineSearch,
    maximum_evaluations: Any | None = None,
) -> StrongWolfeResult:
    """Bracket and bisect a ray until both strong-Wolfe inequalities hold."""

    if not callable(value_and_gradient):
        raise TypeError("value_and_gradient must be callable.")
    if not callable(step):
        raise TypeError("step must be callable.")
    if not callable(contains):
        raise TypeError("contains must be callable.")
    if not isinstance(policy, StrongWolfeLineSearch):
        raise TypeError("policy must be a StrongWolfeLineSearch.")
    initial_value = jnp.asarray(value)
    if initial_value.shape != ():
        raise ValueError("Strong-Wolfe value must be scalar.")
    initial_value = eqx.error_if(
        initial_value,
        ~jnp.isfinite(initial_value),
        "Strong-Wolfe initial value must be finite.",
    )
    initial_directional = jnp.asarray(_tree_inner(gradient, direction))
    if initial_directional.shape != ():
        raise ValueError("Strong-Wolfe directional derivative must be scalar.")
    scalar_dtype = jnp.result_type(initial_value, initial_directional, float)
    zero = jnp.asarray(0.0, dtype=scalar_dtype)
    initial_rate = jnp.asarray(policy.initial_rate, dtype=scalar_dtype)
    minimum_rate = jnp.asarray(policy.minimum_rate, dtype=scalar_dtype)
    maximum_rate = jnp.asarray(policy.maximum_rate, dtype=scalar_dtype)
    expansion = jnp.asarray(policy.expansion, dtype=scalar_dtype)
    decrease = jnp.asarray(policy.sufficient_decrease, dtype=scalar_dtype)
    curvature = jnp.asarray(policy.curvature, dtype=scalar_dtype)
    evaluation_limit = (
        jnp.asarray(policy.maximum_steps, dtype=jnp.int32)
        if maximum_evaluations is None
        else jnp.maximum(
            jnp.asarray(maximum_evaluations, dtype=jnp.int32),
            0,
        )
    )

    def condition(carry):
        iteration, rate, *_, accepted, __, ___, ____ = carry
        return (
            (iteration < policy.maximum_steps)
            & (iteration < evaluation_limit)
            & (~accepted)
            & jnp.isfinite(rate)
            & (rate >= minimum_rate)
            & (rate <= maximum_rate)
        )

    def body(carry):
        (
            iteration,
            rate,
            low_rate,
            high_rate,
            low_value,
            previous_rate,
            previous_value,
            bracketed,
            accepted_parameters,
            accepted_value,
            accepted_gradient,
            accepted_directional,
            finite_seen,
            _,
            _,
            _,
            _,
        ) = carry
        candidate = step(parameters, direction, rate)
        candidate_value, candidate_gradient = value_and_gradient(candidate)
        candidate_value = jnp.asarray(candidate_value).reshape(())
        candidate_directional = jnp.asarray(
            _tree_inner(candidate_gradient, direction)
        ).reshape(())
        finite = (
            jnp.isfinite(candidate_value)
            & jnp.isfinite(candidate_directional)
            & jnp.asarray(contains(candidate), dtype=bool)
            & _tree_allfinite(candidate_gradient)
        )
        armijo = (
            finite
            & (initial_directional < 0.0)
            & (candidate_value <= initial_value + decrease * rate * initial_directional)
        )
        strong_curvature = finite & (
            jnp.abs(candidate_directional) <= curvature * jnp.abs(initial_directional)
        )
        accept = armijo & strong_curvature
        accepted_parameters = _tree_where(accept, candidate, accepted_parameters)
        accepted_gradient = _tree_where(accept, candidate_gradient, accepted_gradient)
        accepted_value = jnp.where(accept, candidate_value, accepted_value)
        accepted_directional = jnp.where(
            accept, candidate_directional, accepted_directional
        )

        value_bracket = (~armijo) | (
            (iteration > 0) & (candidate_value >= previous_value)
        )
        derivative_bracket = finite & (candidate_directional >= 0.0)
        make_bracket = (~bracketed) & (value_bracket | derivative_bracket)

        zoom_high = jnp.where(
            (~armijo) | (candidate_value >= low_value),
            rate,
            jnp.where(
                candidate_directional * (high_rate - low_rate) >= 0.0,
                low_rate,
                high_rate,
            ),
        )
        zoom_updates_low = armijo & (candidate_value < low_value)
        zoom_low = jnp.where(zoom_updates_low, rate, low_rate)
        zoom_low_value = jnp.where(zoom_updates_low, candidate_value, low_value)

        bracket_low = previous_rate
        bracket_high = rate
        bracket_low_value = previous_value
        next_low = jnp.where(
            bracketed,
            zoom_low,
            jnp.where(make_bracket, bracket_low, low_rate),
        )
        next_high = jnp.where(
            bracketed,
            zoom_high,
            jnp.where(make_bracket, bracket_high, high_rate),
        )
        next_low_value = jnp.where(
            bracketed,
            zoom_low_value,
            jnp.where(make_bracket, bracket_low_value, low_value),
        )
        next_bracketed = bracketed | make_bracket
        expanded_rate = jnp.minimum(maximum_rate, rate * expansion)
        zoom_rate = 0.5 * (next_low + next_high)
        next_rate = jnp.where(next_bracketed, zoom_rate, expanded_rate)
        next_rate = jnp.where(accept, rate, next_rate)
        return (
            iteration + 1,
            next_rate,
            next_low,
            next_high,
            next_low_value,
            rate,
            candidate_value,
            next_bracketed,
            accepted_parameters,
            accepted_value,
            accepted_gradient,
            accepted_directional,
            finite_seen | finite,
            accept,
            armijo & accept,
            strong_curvature & accept,
            rate,
        )

    initial_carry = (
        jnp.asarray(0, dtype=jnp.int32),
        initial_rate,
        zero,
        initial_rate,
        initial_value,
        zero,
        initial_value,
        jnp.asarray(False),
        parameters,
        initial_value,
        gradient,
        initial_directional,
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        zero,
    )
    (
        evaluations,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        accepted_parameters,
        accepted_value,
        accepted_gradient,
        accepted_directional,
        finite_seen,
        accepted,
        armijo_satisfied,
        curvature_satisfied,
        accepted_rate,
    ) = jax.lax.while_loop(condition, body, initial_carry)
    rate = jnp.where(accepted, accepted_rate, jnp.zeros_like(accepted_rate))
    return StrongWolfeResult(
        parameters=accepted_parameters,
        value=accepted_value,
        gradient=accepted_gradient,
        rate=rate,
        evaluations=evaluations,
        accepted=accepted,
        initial_directional_derivative=initial_directional,
        directional_derivative=accepted_directional,
        sufficient_decrease_satisfied=armijo_satisfied,
        curvature_satisfied=curvature_satisfied,
        finite_candidate_seen=finite_seen,
    )


__all__ = [
    "ArmijoLineSearch",
    "ArmijoResult",
    "StrongWolfeLineSearch",
    "StrongWolfeResult",
    "armijo_backtracking",
    "strong_wolfe_line_search",
]
