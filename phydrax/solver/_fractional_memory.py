#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._memory import _solution_valid, _time_grid, MemoryEquationSolution


FractionalVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]


class CaputoFractionalProblem(StrictModule):
    """Caputo initial-value problem of real order in ``(0, 2]``.

    For order at most one the initial datum is ``y(t0)``. Orders above one also
    require ``initial_derivative = y'(t0)``. The vector field declares
    ``D_C**order y(t) = f(t, y(t), args)``.
    """

    vector_field: FractionalVectorField
    initial_state: Array
    initial_derivative: Array | None
    order: Array
    t0: Array
    t1: Array
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    order_interval: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: FractionalVectorField,
        initial_state: ArrayLike,
        order: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        initial_derivative: ArrayLike | None = None,
        args: Any = None,
        problem_id: str = "caputo-fractional-problem",
    ):
        if not callable(vector_field):
            raise TypeError("vector_field must be callable.")
        state = jnp.asarray(initial_state)
        state = state.astype(jnp.result_type(state, float))
        state_shape = tuple(int(size) for size in state.shape)
        if not state_shape or any(size <= 0 for size in state_shape):
            raise ValueError("initial_state must have a non-empty positive shape.")
        raw_order = order
        if isinstance(raw_order, (int, float)):
            alpha_value = float(raw_order)
        elif isinstance(raw_order, jax_core.Tracer):
            raise ValueError("CaputoFractionalProblem order must be declared statically.")
        else:
            alpha_value = float(jax.device_get(jnp.asarray(raw_order)))
        alpha = jnp.asarray(raw_order, dtype=float)
        if alpha.shape != ():
            raise ValueError("order must be scalar.")
        if not isfinite(alpha_value) or not 0.0 < alpha_value <= 2.0:
            raise ValueError("order must be finite and lie in (0, 2].")
        interval = 1 if alpha_value <= 1.0 else 2
        if interval == 2:
            if initial_derivative is None:
                raise ValueError("orders above one require initial_derivative.")
            derivative = jnp.asarray(initial_derivative, dtype=state.dtype)
            if derivative.shape != state_shape:
                raise ValueError("initial_derivative must match initial_state shape.")
        else:
            if initial_derivative is not None:
                raise ValueError("initial_derivative is only valid for orders above one.")
            derivative = None
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if start.shape != () or end.shape != ():
            raise ValueError("t0 and t1 must be scalar.")
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start) | ~jnp.isfinite(end) | (end <= start),
            "CaputoFractionalProblem requires finite t1 > t0.",
        )
        value = jnp.asarray(vector_field(start, state, args))
        if value.shape != state_shape:
            raise ValueError("vector_field must preserve initial_state shape.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.vector_field = vector_field
        self.initial_state = state
        self.initial_derivative = derivative
        self.order = alpha
        self.t0 = start
        self.t1 = end
        self.args = args
        self.state_shape = state_shape
        self.order_interval = interval
        self.problem_id = identifier


def solve_caputo_fractional(
    problem: CaputoFractionalProblem,
    /,
    *,
    times: ArrayLike,
) -> MemoryEquationSolution:
    """Solve a Caputo IVP by causal piecewise-constant product integration.

    The exact power-law cell weights support nonuniform grids. The method is explicit,
    JAX-transformable, and uses quadratic work and linear state storage.
    """
    if not isinstance(problem, CaputoFractionalProblem):
        raise TypeError("problem must be a CaputoFractionalProblem.")
    grid = _time_grid(problem.t0, problem.t1, times)
    num_times = int(grid.size)
    states = jnp.zeros(
        (num_times,) + problem.state_shape,
        dtype=problem.initial_state.dtype,
    ).at[0].set(problem.initial_state)
    normalization = jsp.special.gamma(problem.order + 1.0)

    def outer(index, state_buffer):
        target = grid[index]
        base = problem.initial_state
        if problem.order_interval == 2:
            assert problem.initial_derivative is not None
            base = base + (target - problem.t0) * problem.initial_derivative

        def inner(source_index, total):
            def contribute(accumulator):
                left_lag = target - grid[source_index]
                right_lag = target - grid[source_index + 1]
                weight = (
                    jnp.power(left_lag, problem.order)
                    - jnp.power(right_lag, problem.order)
                ) / normalization
                source = grid[source_index]
                value = jnp.asarray(
                    problem.vector_field(
                        source,
                        state_buffer[source_index],
                        problem.args,
                    )
                )
                if value.shape != problem.state_shape:
                    raise ValueError("vector_field changed its declared state shape.")
                return accumulator + weight * value

            return jax.lax.cond(
                source_index < index,
                contribute,
                lambda accumulator: accumulator,
                total,
            )

        memory = jax.lax.fori_loop(
            0,
            num_times - 1,
            inner,
            jnp.zeros(problem.state_shape, dtype=problem.initial_state.dtype),
        )
        return state_buffer.at[index].set(base + memory)

    states = jax.lax.fori_loop(1, num_times, outer, states)
    num_cells = num_times * (num_times - 1) // 2
    return MemoryEquationSolution(
        times=grid,
        states=states,
        valid=_solution_valid(states, ()),
        realization=None,
        state_shape=problem.state_shape,
        solver_name="CaputoProductIntegration",
        solver_id="solver:fractional:caputo-product-integration:v1",
        resolved_method="explicit-piecewise-constant-power-law-convolution",
        stats={
            "num_steps": num_times - 1,
            "num_accepted_steps": num_times - 1,
            "num_rejected_steps": 0,
            "num_memory_cells": num_cells,
        },
        metadata={
            "problem_id": problem.problem_id,
            "fractional_order": problem.order,
            "caputo_order_interval": problem.order_interval,
            "memory_kernel": "power-law",
            "quadrature": "piecewise-constant-product-integration",
            "grid": "nonuniform-supported",
        },
    )


__all__ = [
    "CaputoFractionalProblem",
    "FractionalVectorField",
    "solve_caputo_fractional",
]
