#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._qpax_backend import solve_qpax_implicit, solve_qpax_implicit_primal


QPMethod: TypeAlias = Literal["dense-primal-dual", "qpax-implicit"]
QPDifferentiableMethod: TypeAlias = Literal["dense-active-set", "qpax-implicit"]

QP_SUCCESS = 0
QP_MAX_ITERATIONS = 1
QP_INFEASIBLE = 2
QP_NONFINITE = 3


def _batch_shape(value: Sequence[int], /) -> tuple[int, ...]:
    return tuple(int(size) for size in value)


def _canonical_matrix(
    value: ArrayLike | None,
    /,
    *,
    rows_from: ArrayLike | None,
    variables: int,
    name: str,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    matrix = None if value is None else jnp.asarray(value)
    rhs = None if rows_from is None else jnp.asarray(rows_from)
    if (matrix is not None and jnp.issubdtype(matrix.dtype, jnp.complexfloating)) or (
        rhs is not None and jnp.issubdtype(rhs.dtype, jnp.complexfloating)
    ):
        raise TypeError("QuadraticProgram data must be real-valued.")
    if matrix is None:
        if rhs is not None:
            raise ValueError(f"{name} and its right-hand side must be provided together.")
        return (
            jnp.empty((0, variables), dtype=dtype),
            jnp.empty((0,), dtype=dtype),
        )
    if rhs is None:
        raise ValueError(f"{name} and its right-hand side must be provided together.")
    matrix = matrix.astype(dtype)
    rhs = rhs.astype(dtype)
    if matrix.ndim < 2:
        raise ValueError(
            f"{name} must have shape batch_shape + (constraints, variables)."
        )
    if matrix.shape[-1] != variables:
        raise ValueError(
            f"{name} has {matrix.shape[-1]} variables; expected {variables}."
        )
    if rhs.ndim < 1 or rhs.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"The right-hand side for {name} must end in shape ({matrix.shape[-2]},)."
        )
    return matrix, rhs


class QuadraticProgram(StrictModule):
    r"""A convex quadratic program in canonical equality/inequality form.

    The program is

    .. math::

        \min_x \tfrac12 x^T Q x + q^T x
        \quad\text{subject to}\quad A x=b,\;Gx\leq h.

    Every array is broadcast to ``batch_shape`` followed by its event dimensions.
    Missing constraint families are represented by zero-row arrays, so the problem is
    a transformation-compatible PyTree with one layout for every constraint pattern.
    Convexity requires the symmetric part of ``quadratic`` to be positive semidefinite;
    this mathematical precondition is not changed by hidden jitter or projection.
    """

    quadratic: Array
    linear: Array
    equality_matrix: Array
    equality_rhs: Array
    inequality_matrix: Array
    inequality_rhs: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_equalities: int = eqx.field(static=True)
    num_inequalities: int = eqx.field(static=True)

    def __init__(
        self,
        quadratic: ArrayLike,
        linear: ArrayLike,
        /,
        *,
        equality_matrix: ArrayLike | None = None,
        equality_rhs: ArrayLike | None = None,
        inequality_matrix: ArrayLike | None = None,
        inequality_rhs: ArrayLike | None = None,
    ):
        quadratic_value = jnp.asarray(quadratic)
        linear_value = jnp.asarray(linear)
        if (
            quadratic_value.ndim < 2
            or quadratic_value.shape[-2] != quadratic_value.shape[-1]
        ):
            raise ValueError(
                "quadratic must have shape batch_shape + (variables, variables)."
            )
        variables = int(quadratic_value.shape[-1])
        if variables < 1:
            raise ValueError("QuadraticProgram requires at least one decision variable.")
        if linear_value.ndim < 1 or linear_value.shape[-1] != variables:
            raise ValueError(f"linear must have shape batch_shape + ({variables},).")
        dtype = jnp.result_type(quadratic_value, linear_value, jnp.float32)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("QuadraticProgram data must be real-valued.")
        quadratic_value = quadratic_value.astype(dtype)
        quadratic_value = 0.5 * (quadratic_value + jnp.swapaxes(quadratic_value, -1, -2))
        linear_value = linear_value.astype(dtype)
        equality_matrix_value, equality_rhs_value = _canonical_matrix(
            equality_matrix,
            rows_from=equality_rhs,
            variables=variables,
            name="equality_matrix",
            dtype=dtype,
        )
        inequality_matrix_value, inequality_rhs_value = _canonical_matrix(
            inequality_matrix,
            rows_from=inequality_rhs,
            variables=variables,
            name="inequality_matrix",
            dtype=dtype,
        )
        batch = np.broadcast_shapes(
            quadratic_value.shape[:-2],
            linear_value.shape[:-1],
            equality_matrix_value.shape[:-2],
            equality_rhs_value.shape[:-1],
            inequality_matrix_value.shape[:-2],
            inequality_rhs_value.shape[:-1],
        )
        equalities = int(equality_matrix_value.shape[-2])
        inequalities = int(inequality_matrix_value.shape[-2])
        self.quadratic = jnp.broadcast_to(quadratic_value, batch + (variables, variables))
        self.linear = jnp.broadcast_to(linear_value, batch + (variables,))
        self.equality_matrix = jnp.broadcast_to(
            equality_matrix_value, batch + (equalities, variables)
        )
        self.equality_rhs = jnp.broadcast_to(equality_rhs_value, batch + (equalities,))
        self.inequality_matrix = jnp.broadcast_to(
            inequality_matrix_value, batch + (inequalities, variables)
        )
        self.inequality_rhs = jnp.broadcast_to(
            inequality_rhs_value, batch + (inequalities,)
        )
        self.batch_shape = _batch_shape(batch)
        self.num_variables = variables
        self.num_equalities = equalities
        self.num_inequalities = inequalities


class QuadraticProgramResult(StrictModule):
    """Primal/dual solution, KKT diagnostics, and complete solver provenance."""

    primal: Array
    equality_dual: Array
    inequality_dual: Array
    inequality_slack: Array
    objective: Array
    stationarity_residual: Array
    solver_stationarity_residual: Array
    equality_residual: Array
    inequality_residual: Array
    inequality_violation: Array
    complementarity_residual: Array
    primal_residual_norm: Array
    dual_residual_norm: Array
    solver_dual_residual_norm: Array
    complementarity_gap: Array
    kkt_residual_norm: Array
    iterations: Array
    backend_converged: Array
    valid: Array
    status: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    method: QPMethod = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == QP_SUCCESS)


def _max_abs(value: Array, /) -> Array:
    if value.shape[-1] == 0:
        return jnp.zeros(value.shape[:-1], dtype=value.dtype)
    return jnp.max(jnp.abs(value), axis=-1)


def _min_value(value: Array, /) -> Array:
    if value.shape[-1] == 0:
        return jnp.full(value.shape[:-1], jnp.inf, dtype=value.dtype)
    return jnp.min(value, axis=-1)


def _fraction_to_boundary(value: Array, direction: Array, step_fraction: float) -> Array:
    candidates = jnp.where(direction < 0, -value / direction, jnp.inf)
    return jnp.minimum(
        jnp.asarray(1.0, dtype=value.dtype), step_fraction * jnp.min(candidates)
    )


def _solve_saddle(
    matrix: Array, constraint: Array, primal_rhs: Array, constraint_rhs: Array
) -> tuple[Array, Array]:
    variables = matrix.shape[0]
    equalities = constraint.shape[0]
    zeros = jnp.zeros((equalities, equalities), dtype=matrix.dtype)
    kkt = jnp.block([[matrix, constraint.T], [constraint, zeros]])
    solution, _, _, _ = jnp.linalg.lstsq(
        kkt, jnp.concatenate((primal_rhs, constraint_rhs)), rcond=None
    )
    return solution[:variables], solution[variables:]


def _equality_feasibility(matrix: Array, rhs: Array, tolerance: float) -> Array:
    if matrix.shape[0] == 0:
        return jnp.asarray(True)
    candidate, _, _, _ = jnp.linalg.lstsq(matrix, rhs, rcond=None)
    residual = matrix @ candidate - rhs
    return _max_abs(residual) <= tolerance * (1.0 + _max_abs(rhs))


def _newton_direction(
    quadratic: Array,
    equality_matrix: Array,
    inequality_matrix: Array,
    slack: Array,
    inequality_dual: Array,
    dual_residual: Array,
    equality_residual: Array,
    inequality_residual: Array,
    centering_residual: Array,
    regularization: float,
) -> tuple[Array, Array, Array, Array]:
    variables = quadratic.shape[0]
    equalities = equality_matrix.shape[0]
    inequalities = inequality_matrix.shape[0]
    dtype = quadratic.dtype
    zeros_variables_inequalities = jnp.zeros((variables, inequalities), dtype=dtype)
    zeros_equalities = jnp.zeros((equalities, equalities), dtype=dtype)
    zeros_equalities_inequalities = jnp.zeros((equalities, inequalities), dtype=dtype)
    zeros_inequalities_equalities = jnp.zeros((inequalities, equalities), dtype=dtype)
    zeros_inequalities = jnp.zeros((inequalities, inequalities), dtype=dtype)
    zeros_inequalities_variables = jnp.zeros((inequalities, variables), dtype=dtype)
    regularized = quadratic + regularization * jnp.eye(variables, dtype=dtype)
    kkt = jnp.block(
        [
            [
                regularized,
                equality_matrix.T,
                inequality_matrix.T,
                zeros_variables_inequalities,
            ],
            [
                equality_matrix,
                zeros_equalities,
                zeros_equalities_inequalities,
                zeros_equalities_inequalities,
            ],
            [
                inequality_matrix,
                zeros_inequalities_equalities,
                zeros_inequalities,
                jnp.eye(inequalities, dtype=dtype),
            ],
            [
                zeros_inequalities_variables,
                zeros_inequalities_equalities,
                jnp.diag(slack),
                jnp.diag(inequality_dual),
            ],
        ]
    )
    solution = jnp.linalg.solve(
        kkt,
        -jnp.concatenate(
            (
                dual_residual,
                equality_residual,
                inequality_residual,
                centering_residual,
            )
        ),
    )
    equality_end = variables + equalities
    inequality_end = equality_end + inequalities
    primal_direction = solution[:variables]
    equality_dual_direction = solution[variables:equality_end]
    inequality_dual_direction = solution[equality_end:inequality_end]
    slack_direction = solution[inequality_end:]
    return (
        primal_direction,
        slack_direction,
        inequality_dual_direction,
        equality_dual_direction,
    )


def _farkas_certificate_single(
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    equality_dual: Array,
    inequality_dual: Array,
    tolerance: float,
) -> Array:
    stationarity = (
        equality_matrix.T @ equality_dual + inequality_matrix.T @ inequality_dual
    )
    scale = jnp.maximum(
        1.0,
        jnp.maximum(_max_abs(equality_dual), _max_abs(inequality_dual)),
    )
    certificate_tolerance = jnp.sqrt(
        jnp.asarray(tolerance, dtype=inequality_matrix.dtype)
    )
    value = (equality_rhs @ equality_dual + inequality_rhs @ inequality_dual) / scale
    return (
        (_max_abs(stationarity) / scale <= certificate_tolerance)
        & (value < -certificate_tolerance)
        & (_min_value(inequality_dual) >= -certificate_tolerance)
        & jnp.isfinite(value)
    )


def _dense_constrained_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    *,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    variables = quadratic.shape[0]
    inequalities = inequality_matrix.shape[0]
    equality_candidate, _, _, _ = jnp.linalg.lstsq(
        equality_matrix, equality_rhs, rcond=None
    )
    primal = jnp.where(
        equality_matrix.shape[0] == 0,
        jnp.zeros((variables,), dtype=quadratic.dtype),
        equality_candidate,
    )
    raw_slack = inequality_rhs - inequality_matrix @ primal
    slack = jnp.where(raw_slack > 0, raw_slack, jnp.ones_like(raw_slack))
    inequality_dual = jnp.ones((inequalities,), dtype=quadratic.dtype)
    equality_dual = jnp.zeros((equality_matrix.shape[0],), dtype=quadratic.dtype)
    initial = (
        primal,
        slack,
        inequality_dual,
        equality_dual,
        jnp.asarray(0, dtype=jnp.int32),
    )

    def residuals(primal, slack, inequality_dual, equality_dual):
        dual_residual = (
            quadratic @ primal
            + regularization * primal
            + linear
            + equality_matrix.T @ equality_dual
            + inequality_matrix.T @ inequality_dual
        )
        equality_residual = equality_matrix @ primal - equality_rhs
        inequality_residual = inequality_matrix @ primal + slack - inequality_rhs
        complementarity = slack * inequality_dual
        residual_norm = jnp.maximum(
            jnp.maximum(_max_abs(dual_residual), _max_abs(equality_residual)),
            jnp.maximum(_max_abs(inequality_residual), _max_abs(complementarity)),
        )
        return (
            dual_residual,
            equality_residual,
            inequality_residual,
            complementarity,
            residual_norm,
        )

    def condition(state):
        primal, slack, inequality_dual, equality_dual, iterations = state
        *_, residual_norm = residuals(primal, slack, inequality_dual, equality_dual)
        infeasible = _farkas_certificate_single(
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            equality_dual,
            inequality_dual,
            tolerance,
        )
        return (iterations < max_iterations) & (residual_norm > tolerance) & ~infeasible

    def body(state):
        primal, slack, inequality_dual, equality_dual, iterations = state
        (
            dual_residual,
            equality_residual,
            inequality_residual,
            complementarity,
            _,
        ) = residuals(primal, slack, inequality_dual, equality_dual)
        affine = _newton_direction(
            quadratic,
            equality_matrix,
            inequality_matrix,
            slack,
            inequality_dual,
            dual_residual,
            equality_residual,
            inequality_residual,
            complementarity,
            regularization,
        )
        primal_affine, slack_affine, dual_affine, _ = affine
        alpha_primal_affine = _fraction_to_boundary(slack, slack_affine, 1.0)
        alpha_dual_affine = _fraction_to_boundary(inequality_dual, dual_affine, 1.0)
        mean_complementarity = jnp.mean(complementarity)
        affine_complementarity = jnp.mean(
            (slack + alpha_primal_affine * slack_affine)
            * (inequality_dual + alpha_dual_affine * dual_affine)
        )
        centering = jnp.where(
            mean_complementarity > 0,
            (affine_complementarity / mean_complementarity) ** 3,
            jnp.asarray(0.0, dtype=quadratic.dtype),
        )
        corrected_complementarity = (
            complementarity
            + slack_affine * dual_affine
            - centering * mean_complementarity
        )
        direction = _newton_direction(
            quadratic,
            equality_matrix,
            inequality_matrix,
            slack,
            inequality_dual,
            dual_residual,
            equality_residual,
            inequality_residual,
            corrected_complementarity,
            regularization,
        )
        primal_direction, slack_direction, dual_direction, equality_direction = direction
        alpha_primal = _fraction_to_boundary(slack, slack_direction, step_fraction)
        alpha_dual = _fraction_to_boundary(inequality_dual, dual_direction, step_fraction)
        return (
            primal + alpha_primal * primal_direction,
            slack + alpha_primal * slack_direction,
            inequality_dual + alpha_dual * dual_direction,
            equality_dual + alpha_dual * equality_direction,
            iterations + 1,
        )

    final = jax.lax.while_loop(condition, body, initial)
    primal, slack, inequality_dual, equality_dual, iterations = final
    *_, final_residual_norm = residuals(primal, slack, inequality_dual, equality_dual)
    return (
        primal,
        slack,
        inequality_dual,
        equality_dual,
        final_residual_norm <= tolerance,
        iterations,
    )


def _dense_finite_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    *,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    variables = quadratic.shape[0]
    inequalities = inequality_matrix.shape[0]
    regularized = quadratic + regularization * jnp.eye(variables, dtype=quadratic.dtype)
    if inequalities == 0:
        primal, equality_dual = _solve_saddle(
            regularized,
            equality_matrix,
            -linear,
            equality_rhs,
        )
        direct_residual_norm = jnp.maximum(
            _max_abs(regularized @ primal + linear + equality_matrix.T @ equality_dual),
            _max_abs(equality_matrix @ primal - equality_rhs),
        )
        return (
            primal,
            jnp.empty((0,), dtype=quadratic.dtype),
            jnp.empty((0,), dtype=quadratic.dtype),
            equality_dual,
            jnp.isfinite(direct_residual_norm) & (direct_residual_norm <= tolerance),
            jnp.asarray(0, dtype=jnp.int32),
        )
    return _dense_constrained_single(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )


def _nonfinite_single(
    quadratic: Array,
    equality_matrix: Array,
    inequality_matrix: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    dtype = quadratic.dtype
    return (
        jnp.full((quadratic.shape[0],), jnp.nan, dtype=dtype),
        jnp.full((inequality_matrix.shape[0],), jnp.nan, dtype=dtype),
        jnp.full((inequality_matrix.shape[0],), jnp.nan, dtype=dtype),
        jnp.full((equality_matrix.shape[0],), jnp.nan, dtype=dtype),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _dense_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    *,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    input_finite = jnp.all(
        jnp.stack(
            (
                jnp.all(jnp.isfinite(quadratic)),
                jnp.all(jnp.isfinite(linear)),
                jnp.all(jnp.isfinite(equality_matrix)),
                jnp.all(jnp.isfinite(equality_rhs)),
                jnp.all(jnp.isfinite(inequality_matrix)),
                jnp.all(jnp.isfinite(inequality_rhs)),
            )
        )
    )
    return jax.lax.cond(
        input_finite,
        lambda _: _dense_finite_single(
            quadratic,
            linear,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            tolerance=tolerance,
            max_iterations=max_iterations,
            regularization=regularization,
            step_fraction=step_fraction,
        ),
        lambda _: _nonfinite_single(quadratic, equality_matrix, inequality_matrix),
        operand=None,
    )


def _flatten_problem(
    problem: QuadraticProgram, /
) -> tuple[Array, Array, Array, Array, Array, Array]:
    count = int(np.prod(problem.batch_shape)) if problem.batch_shape else 1
    return (
        problem.quadratic.reshape((count, problem.num_variables, problem.num_variables)),
        problem.linear.reshape((count, problem.num_variables)),
        problem.equality_matrix.reshape(
            (count, problem.num_equalities, problem.num_variables)
        ),
        problem.equality_rhs.reshape((count, problem.num_equalities)),
        problem.inequality_matrix.reshape(
            (count, problem.num_inequalities, problem.num_variables)
        ),
        problem.inequality_rhs.reshape((count, problem.num_inequalities)),
    )


def _solve_dense_arrays(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    *,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    solve_one = partial(
        _dense_single,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )
    return jax.lax.map(
        lambda arrays: solve_one(*arrays),
        (
            quadratic,
            linear,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
        ),
    )


def _diagnostics(
    problem: QuadraticProgram,
    primal: Array,
    slack: Array,
    inequality_dual: Array,
    equality_dual: Array,
    input_backend_converged: Array,
    iterations: Array,
    *,
    method: QPMethod,
    backend: str,
    tolerance: float,
    max_iterations: int,
    regularization: float,
) -> QuadraticProgramResult:
    quadratic = problem.quadratic
    linear = problem.linear
    equality_matrix = problem.equality_matrix
    equality_rhs = problem.equality_rhs
    inequality_matrix = problem.inequality_matrix
    inequality_rhs = problem.inequality_rhs
    objective = 0.5 * oe.contract(
        "...i,...ij,...j->...", primal, quadratic, primal
    ) + oe.contract("...i,...i->...", linear, primal)
    stationarity = (
        oe.contract("...ij,...j->...i", quadratic, primal)
        + linear
        + oe.contract("...ji,...j->...i", equality_matrix, equality_dual)
        + oe.contract("...ji,...j->...i", inequality_matrix, inequality_dual)
    )
    solver_stationarity = stationarity + regularization * primal
    equality_residual = (
        oe.contract("...ij,...j->...i", equality_matrix, primal) - equality_rhs
    )
    inequality_residual = (
        oe.contract("...ij,...j->...i", inequality_matrix, primal)
        + slack
        - inequality_rhs
    )
    inequality_violation = jnp.maximum(
        oe.contract("...ij,...j->...i", inequality_matrix, primal) - inequality_rhs,
        0.0,
    )
    complementarity = slack * inequality_dual
    equality_norm = _max_abs(equality_residual)
    inequality_norm = _max_abs(inequality_residual)
    violation_norm = _max_abs(inequality_violation)
    primal_norm = jnp.maximum(jnp.maximum(equality_norm, inequality_norm), violation_norm)
    dual_norm = _max_abs(stationarity)
    solver_dual_norm = _max_abs(solver_stationarity)
    complementarity_norm = _max_abs(complementarity)
    complementarity_gap = (
        jnp.mean(complementarity, axis=-1)
        if problem.num_inequalities
        else jnp.zeros(problem.batch_shape, dtype=quadratic.dtype)
    )
    kkt_norm = jnp.maximum(
        jnp.maximum(primal_norm, solver_dual_norm), complementarity_norm
    )
    input_finite = (
        jnp.all(jnp.isfinite(quadratic), axis=(-2, -1))
        & jnp.all(jnp.isfinite(linear), axis=-1)
        & jnp.all(jnp.isfinite(equality_matrix), axis=(-2, -1))
        & jnp.all(jnp.isfinite(equality_rhs), axis=-1)
        & jnp.all(jnp.isfinite(inequality_matrix), axis=(-2, -1))
        & jnp.all(jnp.isfinite(inequality_rhs), axis=-1)
    )
    output_finite = (
        jnp.all(jnp.isfinite(primal), axis=-1)
        & jnp.all(jnp.isfinite(slack), axis=-1)
        & jnp.all(jnp.isfinite(inequality_dual), axis=-1)
        & jnp.all(jnp.isfinite(equality_dual), axis=-1)
        & jnp.isfinite(kkt_norm)
    )
    nonnegative = (_min_value(slack) >= -tolerance) & (
        _min_value(inequality_dual) >= -tolerance
    )
    converged = output_finite & nonnegative & (kkt_norm <= tolerance)

    flat_count = int(np.prod(problem.batch_shape)) if problem.batch_shape else 1
    equality_feasible = jax.vmap(
        lambda matrix, rhs: _equality_feasibility(matrix, rhs, tolerance)
    )(
        equality_matrix.reshape(
            (flat_count, problem.num_equalities, problem.num_variables)
        ),
        equality_rhs.reshape((flat_count, problem.num_equalities)),
    ).reshape(problem.batch_shape)
    certificate_stationarity = oe.contract(
        "...ji,...j->...i", equality_matrix, equality_dual
    ) + oe.contract("...ji,...j->...i", inequality_matrix, inequality_dual)
    certificate_scale = jnp.maximum(
        1.0,
        jnp.maximum(_max_abs(equality_dual), _max_abs(inequality_dual)),
    )
    certificate_residual = _max_abs(certificate_stationarity) / certificate_scale
    certificate_value = (
        oe.contract("...i,...i->...", equality_rhs, equality_dual)
        + oe.contract("...i,...i->...", inequality_rhs, inequality_dual)
    ) / certificate_scale
    certificate_tolerance = jnp.sqrt(jnp.asarray(tolerance, dtype=quadratic.dtype))
    farkas_infeasible = (
        (certificate_residual <= certificate_tolerance)
        & (certificate_value < -certificate_tolerance)
        & (_min_value(inequality_dual) >= -certificate_tolerance)
    )
    infeasible = (~equality_feasible) | farkas_infeasible
    status = jnp.where(
        ~input_finite,
        QP_NONFINITE,
        jnp.where(
            converged,
            QP_SUCCESS,
            jnp.where(
                infeasible,
                QP_INFEASIBLE,
                jnp.where(~output_finite, QP_NONFINITE, QP_MAX_ITERATIONS),
            ),
        ),
    ).astype(jnp.int32)
    valid = status == QP_SUCCESS
    return QuadraticProgramResult(
        primal=primal,
        equality_dual=equality_dual,
        inequality_dual=inequality_dual,
        inequality_slack=slack,
        objective=objective,
        stationarity_residual=stationarity,
        solver_stationarity_residual=solver_stationarity,
        equality_residual=equality_residual,
        inequality_residual=inequality_residual,
        inequality_violation=inequality_violation,
        complementarity_residual=complementarity,
        primal_residual_norm=primal_norm,
        dual_residual_norm=dual_norm,
        solver_dual_residual_norm=solver_dual_norm,
        complementarity_gap=complementarity_gap,
        kkt_residual_norm=kkt_norm,
        iterations=iterations,
        backend_converged=jnp.asarray(input_backend_converged, dtype=bool),
        valid=valid,
        status=status,
        batch_shape=problem.batch_shape,
        method=method,
        backend=backend,
        regularization=regularization,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _validate_solver_configuration(
    problem: QuadraticProgram,
    /,
    *,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
    max_dense_dimension: int,
) -> tuple[float, int, float, float, int]:
    if not isinstance(problem, QuadraticProgram):
        raise TypeError("problem must be a QuadraticProgram.")
    tolerance_value = float(tolerance)
    iterations_value = int(max_iterations)
    regularization_value = float(regularization)
    step_value = float(step_fraction)
    dense_limit = int(max_dense_dimension)
    if not np.isfinite(tolerance_value) or tolerance_value <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if iterations_value < 1:
        raise ValueError("max_iterations must be positive.")
    if not np.isfinite(regularization_value) or regularization_value < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    if not np.isfinite(step_value) or not 0 < step_value < 1:
        raise ValueError(
            "step_fraction must be finite and strictly between zero and one."
        )
    if dense_limit < 1:
        raise ValueError("max_dense_dimension must be positive.")
    dimension = (
        problem.num_variables + problem.num_equalities + 2 * problem.num_inequalities
    )
    if dimension > dense_limit:
        raise ValueError(
            f"Dense QP dimension {dimension} exceeds max_dense_dimension={dense_limit}."
        )
    return (
        tolerance_value,
        iterations_value,
        regularization_value,
        step_value,
        dense_limit,
    )


def solve_quadratic_program(
    problem: QuadraticProgram,
    /,
    *,
    method: QPMethod = "dense-primal-dual",
    tolerance: float = 1e-7,
    max_iterations: int = 100,
    regularization: float = 0.0,
    step_fraction: float = 0.995,
    max_dense_dimension: int = 512,
) -> QuadraticProgramResult:
    """Solve a convex QP and return primal/dual variables plus audited KKT data.

    QPax 0.1.4 fixes its fraction-to-boundary multiplier at 0.99, so
    ``method="qpax-implicit"`` rejects non-default ``step_fraction`` requests.
    """

    tolerance, max_iterations, regularization, step_fraction, _ = (
        _validate_solver_configuration(
            problem,
            tolerance=tolerance,
            max_iterations=max_iterations,
            regularization=regularization,
            step_fraction=step_fraction,
            max_dense_dimension=max_dense_dimension,
        )
    )
    arrays = _flatten_problem(problem)
    if method == "dense-primal-dual":
        primal, slack, inequality_dual, equality_dual, backend_converged, iterations = (
            _solve_dense_arrays(
                *arrays,
                tolerance=tolerance,
                max_iterations=max_iterations,
                regularization=regularization,
                step_fraction=step_fraction,
            )
        )
        backend = "phydrax"
    elif method == "qpax-implicit":
        primal, slack, inequality_dual, equality_dual, backend_converged, iterations = (
            solve_qpax_implicit(
                *arrays,
                tolerance=tolerance,
                max_iterations=max_iterations,
                regularization=regularization,
                step_fraction=step_fraction,
            )
        )
        backend = "qpax-0.1.4"
    else:
        raise ValueError("method must be 'dense-primal-dual' or 'qpax-implicit'.")
    primal = primal.reshape(problem.batch_shape + (problem.num_variables,))
    slack = slack.reshape(problem.batch_shape + (problem.num_inequalities,))
    inequality_dual = inequality_dual.reshape(
        problem.batch_shape + (problem.num_inequalities,)
    )
    equality_dual = equality_dual.reshape(problem.batch_shape + (problem.num_equalities,))
    backend_converged = backend_converged.reshape(problem.batch_shape)
    iterations = iterations.reshape(problem.batch_shape)
    return _diagnostics(
        problem,
        primal,
        slack,
        inequality_dual,
        equality_dual,
        backend_converged,
        iterations,
        method=method,
        backend=backend,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
    )


def _active_set_adjoint_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    primal: Array,
    equality_dual: Array,
    inequality_dual: Array,
    slack: Array,
    cotangent: Array,
    valid: Array,
    *,
    regularization: float,
    active_set_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    del linear, equality_rhs, inequality_rhs
    variables = quadratic.shape[0]
    equalities = equality_matrix.shape[0]
    active = (slack <= active_set_tolerance) & (inequality_dual > active_set_tolerance)
    active_scalar = active.astype(quadratic.dtype)
    active_matrix = active_scalar[:, None] * inequality_matrix
    constraint = jnp.concatenate((equality_matrix, active_matrix), axis=0)
    inactive_diagonal = jnp.concatenate(
        (
            jnp.zeros((equalities,), dtype=quadratic.dtype),
            1.0 - active_scalar,
        )
    )
    regularized = quadratic + regularization * jnp.eye(variables, dtype=quadratic.dtype)
    kkt = jnp.block(
        [
            [regularized, constraint.T],
            [constraint, -jnp.diag(inactive_diagonal)],
        ]
    )
    # Degenerate active sets may contain dependent rows even when the primal
    # sensitivity is unique; select the consistent minimum-norm adjoint.
    adjoint = jnp.linalg.lstsq(
        kkt.T,
        jnp.concatenate((cotangent, jnp.zeros_like(inactive_diagonal))),
        rcond=None,
    )[0]
    primal_adjoint = adjoint[:variables]
    constraint_adjoint = adjoint[variables:]
    equality_adjoint = constraint_adjoint[:equalities]
    inequality_adjoint = constraint_adjoint[equalities:] * active_scalar
    active_dual = inequality_dual * active_scalar
    quadratic_gradient = -jnp.outer(primal_adjoint, primal)
    linear_gradient = -primal_adjoint
    equality_matrix_gradient = -(
        jnp.outer(equality_dual, primal_adjoint) + jnp.outer(equality_adjoint, primal)
    )
    equality_rhs_gradient = equality_adjoint
    inequality_matrix_gradient = -(
        jnp.outer(active_dual, primal_adjoint) + jnp.outer(inequality_adjoint, primal)
    )
    inequality_rhs_gradient = inequality_adjoint
    gradients = (
        quadratic_gradient,
        linear_gradient,
        equality_matrix_gradient,
        equality_rhs_gradient,
        inequality_matrix_gradient,
        inequality_rhs_gradient,
    )

    def masked(gradient: Array, /) -> Array:
        return jnp.where(valid, gradient, jnp.full_like(gradient, jnp.nan))

    return (
        masked(quadratic_gradient),
        masked(linear_gradient),
        masked(equality_matrix_gradient),
        masked(equality_rhs_gradient),
        masked(inequality_matrix_gradient),
        masked(inequality_rhs_gradient),
    )


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _dense_primal_implicit(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    max_iterations: int,
    tolerance: float,
    regularization: float,
    step_fraction: float,
    active_set_tolerance: float,
) -> Array:
    primal, _, _, _, _, _ = _solve_dense_arrays(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )
    return primal


def _dense_primal_forward(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    max_iterations: int,
    tolerance: float,
    regularization: float,
    step_fraction: float,
    active_set_tolerance: float,
):
    primal, slack, inequality_dual, equality_dual, _, _ = _solve_dense_arrays(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )
    stationarity = (
        oe.contract("...ij,...j->...i", quadratic, primal)
        + regularization * primal
        + linear
        + oe.contract("...ji,...j->...i", equality_matrix, equality_dual)
        + oe.contract("...ji,...j->...i", inequality_matrix, inequality_dual)
    )
    equality_residual = (
        oe.contract("...ij,...j->...i", equality_matrix, primal) - equality_rhs
    )
    inequality_residual = (
        oe.contract("...ij,...j->...i", inequality_matrix, primal)
        + slack
        - inequality_rhs
    )
    complementarity = slack * inequality_dual
    residual = jnp.maximum(
        jnp.maximum(_max_abs(stationarity), _max_abs(equality_residual)),
        jnp.maximum(_max_abs(inequality_residual), _max_abs(complementarity)),
    )
    valid = (
        jnp.all(jnp.isfinite(primal), axis=-1)
        & (_min_value(slack) >= -tolerance)
        & (_min_value(inequality_dual) >= -tolerance)
        & (residual <= tolerance)
    )
    saved = (
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        primal,
        equality_dual,
        inequality_dual,
        slack,
        valid,
    )
    return primal, saved


def _dense_primal_backward(
    max_iterations: int,
    tolerance: float,
    regularization: float,
    step_fraction: float,
    active_set_tolerance: float,
    saved,
    cotangent: Array,
):
    del max_iterations, tolerance, step_fraction
    (
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        primal,
        equality_dual,
        inequality_dual,
        slack,
        valid,
    ) = saved
    adjoint = partial(
        _active_set_adjoint_single,
        regularization=regularization,
        active_set_tolerance=active_set_tolerance,
    )
    return jax.vmap(adjoint)(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        primal,
        equality_dual,
        inequality_dual,
        slack,
        cotangent,
        valid,
    )


_dense_primal_implicit.defvjp(_dense_primal_forward, _dense_primal_backward)


def solve_quadratic_program_primal(
    problem: QuadraticProgram,
    /,
    *,
    method: QPDifferentiableMethod = "dense-active-set",
    tolerance: float = 1e-7,
    max_iterations: int = 100,
    regularization: float = 0.0,
    step_fraction: float = 0.995,
    active_set_tolerance: float = 1e-5,
    max_dense_dimension: int = 512,
) -> Array:
    """Return a differentiable primal solution with an explicit gradient method.

    ``dense-active-set`` differentiates the locally fixed active-set KKT system.
    ``qpax-implicit`` delegates to QPax's public implicit custom-VJP API. QPax's
    explicit differentiation backend is intentionally not accepted, and QPax
    0.1.4's fixed 0.99 fraction-to-boundary multiplier means non-default
    ``step_fraction`` requests are rejected.
    """

    tolerance, max_iterations, regularization, step_fraction, _ = (
        _validate_solver_configuration(
            problem,
            tolerance=tolerance,
            max_iterations=max_iterations,
            regularization=regularization,
            step_fraction=step_fraction,
            max_dense_dimension=max_dense_dimension,
        )
    )
    active_tolerance = float(active_set_tolerance)
    if not np.isfinite(active_tolerance) or active_tolerance <= 0:
        raise ValueError("active_set_tolerance must be finite and positive.")
    arrays = _flatten_problem(problem)
    if method == "dense-active-set":
        primal = _dense_primal_implicit(
            *arrays,
            max_iterations,
            tolerance,
            regularization,
            step_fraction,
            active_tolerance,
        )
    elif method == "qpax-implicit":
        primal = solve_qpax_implicit_primal(
            *arrays,
            tolerance=tolerance,
            max_iterations=max_iterations,
            regularization=regularization,
            step_fraction=step_fraction,
        )
    else:
        raise ValueError(
            "Differentiable method must be 'dense-active-set' or 'qpax-implicit'; "
            "QPax explicit differentiation is not enabled."
        )
    return primal.reshape(problem.batch_shape + (problem.num_variables,))


__all__ = [
    "QP_INFEASIBLE",
    "QP_MAX_ITERATIONS",
    "QP_NONFINITE",
    "QP_SUCCESS",
    "QPDifferentiableMethod",
    "QPMethod",
    "QuadraticProgram",
    "QuadraticProgramResult",
    "solve_quadratic_program",
    "solve_quadratic_program_primal",
]
