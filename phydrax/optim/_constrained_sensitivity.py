#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._constrained_model import prepare_constrained_model
from ._iterative import MinimizationProblem


ConstrainedSensitivityMode: TypeAlias = Literal["fixed-active", "barrier"]


class ConstrainedSensitivityResult(StrictModule):
    value: PyTree[Array]
    condition_estimate: Array
    active_constraints: Array
    regular: Array
    mode: ConstrainedSensitivityMode = eqx.field(static=True)


def _sensitivity_system(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    mode: ConstrainedSensitivityMode,
    active_tolerance: float,
    barrier: float,
):
    prepared = prepare_constrained_model(problem, parameters, args=args)
    evaluation = prepared.evaluate(parameters, args)
    coordinates = evaluation.coordinates
    equality_count = evaluation.equalities.size
    if mode == "fixed-active":
        active_mask = evaluation.inequality_slacks <= active_tolerance
    else:
        active_mask = jnp.ones_like(evaluation.inequality_slacks, dtype=jnp.bool_)
    lower_count = evaluation.lower_slacks.size
    lower_jacobian = evaluation.constraint_jacobian[prepared.lower_indices]
    upper_jacobian = -evaluation.constraint_jacobian[prepared.upper_indices]
    inequality_jacobian = jnp.concatenate([lower_jacobian, upper_jacobian], axis=0)
    active_jacobian = inequality_jacobian[active_mask]
    equality_jacobian = evaluation.constraint_jacobian[prepared.equality_indices]
    if mode == "fixed-active":
        multiplier_matrix = jnp.concatenate(
            [jnp.conj(equality_jacobian.T), -jnp.conj(active_jacobian.T)],
            axis=1,
        )
        multipliers = jnp.linalg.lstsq(
            multiplier_matrix,
            -evaluation.gradient,
            rcond=None,
        )[0]
        equality_multipliers = multipliers[:equality_count]
        active_multipliers = multipliers[equality_count:]
    else:
        equality_multipliers = jnp.linalg.lstsq(
            jnp.conj(equality_jacobian.T),
            -evaluation.gradient,
            rcond=None,
        )[0]
        active_multipliers = barrier / jnp.maximum(
            evaluation.inequality_slacks[active_mask], 1e-12
        )
    initial = jnp.concatenate([coordinates, equality_multipliers, active_multipliers])

    def residual(combined, current_args):
        x = combined[: coordinates.size]
        equality_dual = combined[coordinates.size : coordinates.size + equality_count]
        inequality_dual = combined[coordinates.size + equality_count :]
        point = prepared.unflatten(x)
        current = prepared.evaluate(point, current_args)
        lower_j = current.constraint_jacobian[prepared.lower_indices]
        upper_j = -current.constraint_jacobian[prepared.upper_indices]
        inequality_j = jnp.concatenate([lower_j, upper_j], axis=0)[active_mask]
        equality_j = current.constraint_jacobian[prepared.equality_indices]
        stationarity = (
            current.gradient
            + jnp.conj(equality_j.T) @ equality_dual
            - jnp.conj(inequality_j.T) @ inequality_dual
        )
        if mode == "fixed-active":
            active_values = current.inequality_slacks[active_mask]
        else:
            active_values = (
                current.inequality_slacks[active_mask] * inequality_dual - barrier
            )
        return jnp.concatenate([stationarity, current.equalities, active_values])

    matrix = jax.jacfwd(lambda value: residual(value, args))(initial)
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    condition = singular_values[0] / jnp.maximum(singular_values[-1], 1e-30)
    regular = jnp.isfinite(condition) & (condition < 1e12)
    return prepared, initial, residual, matrix, condition, regular, active_mask


def constrained_solution_jvp(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    tangent_args: Any,
    /,
    *,
    mode: ConstrainedSensitivityMode = "fixed-active",
    active_tolerance: float = 1e-7,
    barrier: float = 1e-8,
) -> ConstrainedSensitivityResult:
    if mode not in ("fixed-active", "barrier"):
        raise ValueError("Unknown constrained sensitivity mode.")
    prepared, initial, residual, matrix, condition, regular, active_mask = (
        _sensitivity_system(
            problem,
            parameters,
            args,
            mode,
            active_tolerance,
            barrier,
        )
    )
    _, argument_action = jax.jvp(
        lambda current_args: residual(initial, current_args),
        (args,),
        (tangent_args,),
    )
    direction = jnp.linalg.solve(matrix, -argument_action)
    tangent = prepared.unflatten(direction[: prepared.template_coordinates.size])
    tangent = jax.tree.map(
        lambda value: jnp.where(regular, value, jnp.full_like(value, jnp.nan)),
        tangent,
    )
    return ConstrainedSensitivityResult(
        tangent,
        condition,
        jnp.sum(active_mask, dtype=jnp.int32),
        regular,
        mode=mode,
    )


def constrained_solution_vjp(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    cotangent_parameters: PyTree[Any],
    /,
    *,
    mode: ConstrainedSensitivityMode = "fixed-active",
    active_tolerance: float = 1e-7,
    barrier: float = 1e-8,
) -> ConstrainedSensitivityResult:
    prepared, initial, residual, matrix, condition, regular, active_mask = (
        _sensitivity_system(
            problem,
            parameters,
            args,
            mode,
            active_tolerance,
            barrier,
        )
    )
    cotangent, _ = ravel_pytree(cotangent_parameters)
    right = jnp.concatenate(
        [
            cotangent,
            jnp.zeros((matrix.shape[0] - cotangent.size,), dtype=cotangent.dtype),
        ]
    )
    adjoint = jnp.linalg.solve(jnp.conj(matrix.T), right)
    _, pullback = jax.vjp(lambda current_args: residual(initial, current_args), args)
    argument_cotangent = jax.tree.map(jnp.negative, pullback(adjoint)[0])
    argument_cotangent = jax.tree.map(
        lambda value: jnp.where(regular, value, jnp.full_like(value, jnp.nan)),
        argument_cotangent,
    )
    return ConstrainedSensitivityResult(
        argument_cotangent,
        condition,
        jnp.sum(active_mask, dtype=jnp.int32),
        regular,
        mode=mode,
    )


__all__ = [
    "ConstrainedSensitivityMode",
    "ConstrainedSensitivityResult",
    "constrained_solution_jvp",
    "constrained_solution_vjp",
]
