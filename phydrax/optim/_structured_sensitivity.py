#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from ..continuation import ParameterContinuationProblem
from ..linalg import LinearSolvePolicy, MaterializationPolicy
from ._constrained_sensitivity import (
    constrained_solution_jvp,
    constrained_solution_vjp,
    ConstrainedSensitivityResult,
    ConstrainedSensitivityRoute,
)
from ._iterative import Bounds, MinimizationProblem, NonlinearConstraint
from ._nonlinear_constraints import _canonical_constraints, _constraint_layout
from ._structured_nonlinear import (
    PreparedStructuredNonlinearProgram,
    StructuredNonlinearResult,
)


StructuredSensitivityMode: TypeAlias = Literal["fixed-active", "barrier"]


def _generic_problem(
    prepared: PreparedStructuredNonlinearProgram, /
) -> MinimizationProblem:
    program = prepared.program

    @jax.custom_jvp
    def constraints(coordinates: Array, args: Any) -> Array:
        return jnp.asarray(program.constraints(coordinates, args))

    @constraints.defjvp
    def constraints_jvp(
        primals: tuple[Array, Any], tangents: tuple[Array, Any]
    ) -> tuple[Array, Array]:
        coordinates, args = primals
        tangent_coordinates, tangent_args = tangents
        value, argument_action = jax.jvp(
            lambda current_args: jnp.asarray(
                program.constraints(coordinates, current_args)
            ),
            (args,),
            (tangent_args,),
        )
        operator = program.jacobian_plan.operator(coordinates, args)
        return value, operator.mv(tangent_coordinates) + argument_action

    constraint_blocks = (
        (
            NonlinearConstraint(
                constraints,
                lower=prepared.constraint_lower,
                upper=prepared.constraint_upper,
                constraint_id=f"{program.program_id}:structured-sensitivity",
            ),
        )
        if program.num_constraints
        else ()
    )
    return MinimizationProblem(
        lambda coordinates, args: program.objective(coordinates, args),
        bounds=Bounds(prepared.variable_lower, prepared.variable_upper),
        constraints=constraint_blocks,
        problem_id=program.program_id,
    )


def structured_solution_jvp(
    prepared: PreparedStructuredNonlinearProgram,
    result: StructuredNonlinearResult,
    tangent_args: Any,
    /,
    *,
    mode: StructuredSensitivityMode = "fixed-active",
    active_tolerance: float = 1e-7,
    barrier: float = 1e-8,
    linear: LinearSolvePolicy | None = None,
    route: ConstrainedSensitivityRoute = "matrix-free",
    materialization: MaterializationPolicy | None = None,
) -> ConstrainedSensitivityResult:
    """Differentiate a certified structured solution with respect to numeric args."""
    if not isinstance(prepared, PreparedStructuredNonlinearProgram):
        raise TypeError("prepared must be a PreparedStructuredNonlinearProgram.")
    if not isinstance(result, StructuredNonlinearResult):
        raise TypeError("result must be a StructuredNonlinearResult.")
    if (
        result.structure_id != prepared.structure_id
        or result.numeric_binding_id != prepared.numeric_binding_id
    ):
        raise ValueError("Structured result and prepared numeric binding do not match.")
    if not bool(result.successful):
        raise ValueError("Structured sensitivity requires a successful primal solve.")
    return constrained_solution_jvp(
        _generic_problem(prepared),
        result.parameters,
        prepared.args,
        tangent_args,
        mode=mode,
        active_tolerance=active_tolerance,
        barrier=barrier,
        linear=linear,
        route=route,
        materialization=materialization,
    )


def structured_solution_vjp(
    prepared: PreparedStructuredNonlinearProgram,
    result: StructuredNonlinearResult,
    cotangent_coordinates: Array,
    /,
    *,
    mode: StructuredSensitivityMode = "fixed-active",
    active_tolerance: float = 1e-7,
    barrier: float = 1e-8,
    linear: LinearSolvePolicy | None = None,
    route: ConstrainedSensitivityRoute = "matrix-free",
    materialization: MaterializationPolicy | None = None,
) -> ConstrainedSensitivityResult:
    """Pull one coordinate cotangent back to structured numeric args."""
    if not isinstance(prepared, PreparedStructuredNonlinearProgram):
        raise TypeError("prepared must be a PreparedStructuredNonlinearProgram.")
    if not isinstance(result, StructuredNonlinearResult):
        raise TypeError("result must be a StructuredNonlinearResult.")
    if (
        result.structure_id != prepared.structure_id
        or result.numeric_binding_id != prepared.numeric_binding_id
    ):
        raise ValueError("Structured result and prepared numeric binding do not match.")
    if not bool(result.successful):
        raise ValueError("Structured sensitivity requires a successful primal solve.")
    cotangent = prepared.validate_coordinates(cotangent_coordinates)
    return constrained_solution_vjp(
        _generic_problem(prepared),
        result.parameters,
        prepared.args,
        cotangent,
        mode=mode,
        active_tolerance=active_tolerance,
        barrier=barrier,
        linear=linear,
        route=route,
        materialization=materialization,
    )


class StructuredContinuationSeed(StrictModule):
    """Fixed-active KKT continuation problem and its exact corrected seed."""

    problem: ParameterContinuationProblem
    state: Array
    active_mask: Array
    structure_id: str = eqx.field(static=True)


def structured_parameter_continuation(
    prepared: PreparedStructuredNonlinearProgram,
    result: StructuredNonlinearResult,
    args_path: Callable[[Array], Any],
    /,
    *,
    parameter_lower: float = -jnp.inf,
    parameter_upper: float = jnp.inf,
    active_tolerance: float = 1e-7,
    problem_id: str | None = None,
) -> StructuredContinuationSeed:
    """Expose one certified fixed-active structured KKT branch to continuation."""
    if not isinstance(prepared, PreparedStructuredNonlinearProgram):
        raise TypeError("prepared must be a PreparedStructuredNonlinearProgram.")
    if not isinstance(result, StructuredNonlinearResult):
        raise TypeError("result must be a StructuredNonlinearResult.")
    if not callable(args_path):
        raise TypeError("args_path must be callable.")
    if (
        result.structure_id != prepared.structure_id
        or result.numeric_binding_id != prepared.numeric_binding_id
    ):
        raise ValueError("Structured result and prepared numeric binding do not match.")
    if not bool(result.successful):
        raise ValueError("Structured continuation requires a successful primal solve.")
    if np.any(np.asarray(prepared.template.fixed_variable_mask)):
        raise ValueError(
            "Structured continuation requires fixed variables to be eliminated first."
        )
    generic = _generic_problem(prepared)
    coordinates = jnp.asarray(result.parameters)
    layout = _constraint_layout(generic, coordinates, prepared.args)
    warm = result.warm_start
    equality = warm.constraint_multipliers[prepared.program.equality_indices]
    lower_constraints = -warm.constraint_multipliers[prepared.program.lower_indices]
    upper_constraints = warm.constraint_multipliers[prepared.program.upper_indices]
    lower_x_indices = np.flatnonzero(np.isfinite(np.asarray(prepared.variable_lower)))
    upper_x_indices = np.flatnonzero(np.isfinite(np.asarray(prepared.variable_upper)))
    inequality = jnp.concatenate(
        (
            lower_constraints,
            warm.lower_bound_multipliers[jnp.asarray(lower_x_indices)],
            upper_constraints,
            warm.upper_bound_multipliers[jnp.asarray(upper_x_indices)],
        )
    )
    _, inequality_values = _canonical_constraints(
        generic,
        layout,
        coordinates,
        prepared.args,
    )
    active = inequality_values >= -active_tolerance
    inequality = jnp.where(active, inequality, 0.0)
    initial = jnp.concatenate((coordinates, equality, inequality))
    n = prepared.program.num_variables
    me = equality.size

    def residual(state: Array, coordinate: Array, _: Any) -> Array:
        current_args = args_path(coordinate)
        x = state[:n]
        equality_multipliers = state[n : n + me]
        inequality_multipliers = state[n + me :]
        gradient = jax.grad(lambda value: generic.value(value, current_args)[0])(x)
        equality_residual, inequality_residual = _canonical_constraints(
            generic,
            layout,
            x,
            current_args,
        )
        active_multipliers = jnp.where(active, inequality_multipliers, 0.0)
        lower_count = lower_constraints.size
        lower_bound_count = lower_x_indices.size
        upper_count = upper_constraints.size
        lower_dual = active_multipliers[:lower_count]
        lower_bound_dual = active_multipliers[
            lower_count : lower_count + lower_bound_count
        ]
        upper_start = lower_count + lower_bound_count
        upper_dual = active_multipliers[upper_start : upper_start + upper_count]
        upper_bound_dual = active_multipliers[upper_start + upper_count :]
        raw_dual = jnp.zeros((prepared.program.num_constraints,), dtype=x.dtype)
        raw_dual = raw_dual.at[prepared.program.equality_indices].add(
            equality_multipliers
        )
        raw_dual = raw_dual.at[prepared.program.lower_indices].add(-lower_dual)
        raw_dual = raw_dual.at[prepared.program.upper_indices].add(upper_dual)
        operator = prepared.program.jacobian_plan.operator(x, current_args)
        stationarity = gradient + operator.adjoint_mv(raw_dual)
        stationarity = stationarity.at[jnp.asarray(lower_x_indices)].add(
            -lower_bound_dual
        )
        stationarity = stationarity.at[jnp.asarray(upper_x_indices)].add(upper_bound_dual)
        complement = jnp.where(active, inequality_residual, inequality_multipliers)
        return jnp.concatenate((stationarity, equality_residual, complement))

    continuation = ParameterContinuationProblem(
        residual,
        parameter_lower=parameter_lower,
        parameter_upper=parameter_upper,
        problem_id=(
            f"{prepared.program.program_id}:structured-kkt-continuation"
            if problem_id is None
            else problem_id
        ),
    )
    return StructuredContinuationSeed(
        continuation,
        initial,
        active,
        prepared.structure_id,
    )


__all__ = [
    "StructuredContinuationSeed",
    "StructuredSensitivityMode",
    "structured_parameter_continuation",
    "structured_solution_jvp",
    "structured_solution_vjp",
]
