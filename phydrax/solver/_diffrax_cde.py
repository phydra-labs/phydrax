#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..stochastic import AbstractRoughControl
from ._differential import DifferentialProblem, DifferentialSolution
from ._diffrax_backend import solve_diffrax
from ._diffrax_state_packing import DiffraxComplexStatePolicy
from ._driving_path import (
    AbstractDifferentiableDrivingPath,
    CallableDrivingPath,
)
from ._rough import RoughDifferentialProblem


class _ControlledVectorField(eqx.Module):
    problem: RoughDifferentialProblem
    path: AbstractDifferentiableDrivingPath

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        drift = jnp.asarray(self.problem.drift(time, state, args))
        fields = jnp.asarray(self.problem.vector_fields(time, state, args))
        derivative = jnp.asarray(self.path.derivative(time, "right"))
        return drift + jnp.tensordot(fields, derivative, axes=((-1,), (0,)))


class ControlledDifferentialSolution(StrictModule):
    """A Diffrax solution with its differentiable control and lowering provenance."""

    differential_solution: DifferentialSolution
    path: AbstractDifferentiableDrivingPath
    derivative_discontinuities: Array
    derivative_discontinuity_mask: Array
    metadata: frozendict[str, Any]
    path_id: str = eqx.field(static=True)
    path_interpolation: str = eqx.field(static=True)
    control_dimension: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        differential_solution: DifferentialSolution,
        path: AbstractDifferentiableDrivingPath,
        /,
        *,
        problem_id: str,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(differential_solution, DifferentialSolution):
            raise TypeError("differential_solution must be a DifferentialSolution.")
        if not isinstance(path, AbstractDifferentiableDrivingPath):
            raise TypeError("path must be an AbstractDifferentiableDrivingPath.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        breakpoints = jnp.asarray(path.breakpoints, dtype=float)
        breakpoint_mask = jnp.asarray(path.breakpoint_mask, dtype=bool)
        if breakpoints.ndim != 1 or breakpoint_mask.shape != breakpoints.shape:
            raise ValueError(
                "Path derivative-discontinuity arrays must be aligned rank-1 arrays."
            )
        self.differential_solution = differential_solution
        self.path = path
        self.derivative_discontinuities = breakpoints
        self.derivative_discontinuity_mask = breakpoint_mask
        self.metadata = frozendict({} if metadata is None else dict(metadata))
        self.path_id = path.path_id
        self.path_interpolation = type(path).__name__
        self.control_dimension = int(path.value_shape[0])
        self.problem_id = identifier

    @property
    def times(self) -> Array:
        return self.differential_solution.times

    @property
    def states(self) -> Array:
        return self.differential_solution.states

    @property
    def valid(self) -> Array:
        return self.differential_solution.valid

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.differential_solution.sample_shape

    @property
    def num_times(self) -> int:
        return self.differential_solution.num_times

    @property
    def interpolation(self) -> Any | None:
        return self.differential_solution.interpolation

    @property
    def backend_result(self) -> Any:
        return self.differential_solution.backend_result

    @property
    def stats(self) -> frozendict[str, Any]:
        return self.differential_solution.stats

    @property
    def event_mask(self) -> Any:
        return self.differential_solution.event_mask

    @property
    def realization(self) -> Any | None:
        return self.differential_solution.realization

    @property
    def wiener_term_slices(self) -> frozendict[str, tuple[int, int]]:
        return self.differential_solution.wiener_term_slices

    @property
    def solver_name(self) -> str:
        return self.differential_solution.solver_name

    @property
    def solver_id(self) -> str:
        return self.differential_solution.solver_id

    @property
    def resolved_method(self) -> str:
        return self.differential_solution.resolved_method

    @property
    def interpretation(self) -> str:
        return self.differential_solution.interpretation

    @property
    def state_geometry_id(self) -> str | None:
        return self.differential_solution.state_geometry_id

    @property
    def successful(self) -> Array:
        return self.differential_solution.successful

    @property
    def has_dense_interpolation(self) -> bool:
        return self.differential_solution.has_dense_interpolation

    def evaluate(
        self,
        query_times: ArrayLike,
        /,
        *,
        left: bool = True,
    ) -> Array:
        """Evaluate the wrapped dense Diffrax interpolation."""
        return self.differential_solution.evaluate(query_times, left=left)


def _continuity_checked_initial_state(
    initial_state: Array,
    path: AbstractDifferentiableDrivingPath,
    /,
) -> Array:
    if not isinstance(path, CallableDrivingPath) or int(path.breakpoints.size) == 0:
        return initial_state
    breakpoints = jax.lax.stop_gradient(jnp.asarray(path.breakpoints, dtype=float))
    active = jax.lax.stop_gradient(jnp.asarray(path.breakpoint_mask, dtype=bool))
    start, end = path.support
    safe_points = jnp.where(active, breakpoints, 0.5 * (start + end))
    left_values = tuple(jnp.asarray(path.value(point, "left")) for point in safe_points)
    right_values = tuple(jnp.asarray(path.value(point, "right")) for point in safe_points)
    if any(
        value.shape != tuple(path.value_shape) for value in left_values + right_values
    ):
        raise ValueError(
            "Callable driving-path values at breakpoints must return value_shape."
        )
    left = jnp.stack(left_values)
    right = jnp.stack(right_values)
    invalid = active & (
        jnp.any(~jnp.isfinite(left), axis=tuple(range(1, left.ndim)))
        | jnp.any(~jnp.isfinite(right), axis=tuple(range(1, right.ndim)))
        | jnp.any(left != right, axis=tuple(range(1, left.ndim)))
    )
    return eqx.error_if(
        initial_state,
        jnp.any(invalid),
        "Active callable driving-path breakpoint values must have finite matching "
        "left/right limits.",
    )


def _jump_schedule(path: AbstractDifferentiableDrivingPath, /) -> Array:
    breakpoints = jax.lax.stop_gradient(jnp.asarray(path.breakpoints, dtype=float))
    mask = jax.lax.stop_gradient(jnp.asarray(path.breakpoint_mask, dtype=bool))
    start, end = path.support
    active = mask & (breakpoints > start) & (breakpoints < end)
    inactive = jnp.asarray(jnp.inf, dtype=breakpoints.dtype)
    return jnp.sort(jnp.where(active, breakpoints, inactive))


def _discontinuity_controller(
    path: AbstractDifferentiableDrivingPath,
    controller: Any | None,
    /,
    *,
    rtol: float,
    atol: float,
) -> dfx.AbstractStepSizeController | None:
    if int(path.breakpoints.size) == 0:
        return controller
    inner = (
        dfx.PIDController(rtol=float(rtol), atol=float(atol))
        if controller is None
        else controller
    )
    if not isinstance(inner, dfx.AbstractAdaptiveStepSizeController):
        raise TypeError(
            "solve_diffrax_cde requires an adaptive Diffrax stepsize controller so "
            "every path-derivative discontinuity is treated as an exact jump; use "
            "diffrax.PIDController."
        )
    schedule = _jump_schedule(path)
    return dfx.ClipStepSizeController(
        inner,
        step_ts=schedule,
        jump_ts=schedule,
    )


def solve_diffrax_cde(
    problem: RoughDifferentialProblem,
    path: AbstractDifferentiableDrivingPath | AbstractRoughControl,
    /,
    *,
    save_times: ArrayLike,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    event: Any | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense: bool = False,
    max_steps: int | None = 4096,
    throw: bool = False,
    complex_state_policy: DiffraxComplexStatePolicy | None = None,
) -> ControlledDifferentialSolution:
    """Solve a smooth controlled differential equation through ``solve_diffrax``.

    Only differentiable first-level controls belong on this execution path. Finite
    rough controls carry nontrivial second-level information and must instead be
    integrated with :func:`solve_rough_differential`.
    """
    if not isinstance(problem, RoughDifferentialProblem):
        raise TypeError("solve_diffrax_cde requires a RoughDifferentialProblem.")
    if isinstance(path, AbstractRoughControl):
        raise TypeError(
            "solve_diffrax_cde accepts differentiable first-level paths only; pass "
            "rough or second-level controls to solve_rough_differential."
        )
    if not isinstance(path, AbstractDifferentiableDrivingPath):
        raise TypeError(
            "path must be an AbstractDifferentiableDrivingPath; rough controls belong "
            "in solve_rough_differential."
        )
    if tuple(path.value_shape) != (problem.driver_dimension,):
        raise ValueError(
            "Path value_shape must be the one-dimensional rough driver shape "
            f"({problem.driver_dimension},); got {path.value_shape}."
        )

    start, end = path.support
    lowered = DifferentialProblem(
        _ControlledVectorField(problem, path),
        _continuity_checked_initial_state(problem.initial_state, path),
        t0=start,
        t1=end,
        args=problem.args,
        state_geometry=problem.geometry,
    )
    controller = _discontinuity_controller(
        path,
        stepsize_controller,
        rtol=rtol,
        atol=atol,
    )
    solution = solve_diffrax(
        lowered,
        save_times=save_times,
        solver=solver,
        stepsize_controller=controller,
        adjoint=adjoint,
        dt0=dt0,
        event=event,
        rtol=rtol,
        atol=atol,
        dense=dense,
        max_steps=max_steps,
        throw=throw,
        complex_state_policy=complex_state_policy,
    )
    return ControlledDifferentialSolution(
        solution,
        path,
        problem_id=problem.problem_id,
        metadata={
            "lowering": "differentiable-control-to-ode",
            "control_level": 1,
            "driver_dimension": problem.driver_dimension,
            "path_id": path.path_id,
            "path_interpolation": type(path).__name__,
            "derivative_discontinuity_policy": "diffrax-jump-landing",
        },
    )


__all__ = ["ControlledDifferentialSolution", "solve_diffrax_cde"]
