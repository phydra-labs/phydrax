#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Algebraic Riccati equations with explicit solvability diagnostics."""

from __future__ import annotations

from enum import IntEnum
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._lyapunov import (
    continuous_lyapunov_solution,
    discrete_lyapunov_solution,
    LinearMatrixEquationDiagnostics,
    LinearMatrixEquationStatus,
)


class RiccatiStatus(IntEnum):
    """Stable status codes returned by algebraic Riccati solvers."""

    SUCCESS = 0
    UNSTABILIZABLE = 1
    UNDETECTABLE = 2
    UNSTABLE_SOLUTION = 3
    NONFINITE = 4
    NONCONVERGED = 5


class AlgebraicRiccatiDiagnostics(StrictModule):
    """ARE-specific evidence composed with shared matrix-equation diagnostics."""

    equation: LinearMatrixEquationDiagnostics
    control_condition_number: Array
    stabilizable: Array
    detectable: Array
    status: Array

    @property
    def residual_norm(self) -> Array:
        return self.equation.residual_norm

    @property
    def relative_residual(self) -> Array:
        return self.equation.relative_residual

    @property
    def equation_condition_number(self) -> Array:
        return self.equation.condition_number

    @property
    def stable(self) -> Array:
        return self.equation.stable

    @property
    def finite(self) -> Array:
        return self.equation.finite

    @property
    def converged(self) -> Array:
        return self.equation.converged & self.stabilizable & self.detectable

    @property
    def iterations(self) -> Array:
        return self.equation.iterations

    @property
    def method(self) -> str:
        return self.equation.method


class AlgebraicRiccatiResult(StrictModule):
    """A stabilizing algebraic Riccati matrix and its diagnostics."""

    matrix: Array
    diagnostics: AlgebraicRiccatiDiagnostics
    valid: Array
    status: Array


def _matrix(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.ndim < 2 or result.shape[-2] != result.shape[-1]:
        raise ValueError(
            f"{name} must end in equal matrix dimensions; got {result.shape}."
        )
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)),
        f"{name} must contain only finite values.",
    )


def _require_shape(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> Array:
    result = jnp.asarray(value)
    if tuple(result.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)),
        f"{name} must contain only finite values.",
    )


def _require_symmetric(
    value: Array,
    name: str,
    tolerance: float,
    /,
) -> Array:
    asymmetry = jnp.max(jnp.abs(value - jnp.swapaxes(value, -1, -2)))
    return eqx.error_if(
        value,
        asymmetry > tolerance,
        f"{name} must be symmetric within cost_tolerance.",
    )


def _require_positive_semidefinite(
    value: Array,
    name: str,
    tolerance: float,
    /,
) -> Array:
    value = _require_symmetric(value, name, tolerance)
    minimum = jnp.min(jnp.linalg.eigvalsh(value))
    return eqx.error_if(
        value,
        minimum < -tolerance,
        f"{name} must be positive semidefinite; indefinite costs are unsupported.",
    )


def _require_positive_definite(
    value: Array,
    name: str,
    tolerance: float,
    /,
) -> Array:
    value = _require_symmetric(value, name, tolerance)
    minimum = jnp.min(jnp.linalg.eigvalsh(value))
    return eqx.error_if(
        value,
        minimum <= tolerance,
        f"{name} must be positive definite; singular control costs are unsupported.",
    )


def _validate_are_inputs(
    a: ArrayLike,
    b: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    s: ArrayLike | None,
    cost_tolerance: float,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    a_ = _matrix(a, "a")
    n = int(a_.shape[-1])
    case_shape = tuple(a_.shape[:-2])
    b_ = jnp.asarray(b)
    if b_.ndim < 2 or tuple(b_.shape[:-2]) != case_shape or b_.shape[-2] != n:
        raise ValueError(
            "b must have shape case_shape + (state_size, control_size); "
            f"got {b_.shape} for a shape {a_.shape}."
        )
    if jnp.issubdtype(b_.dtype, jnp.complexfloating):
        raise TypeError("b must be real-valued.")
    if not jnp.issubdtype(b_.dtype, jnp.inexact):
        b_ = b_.astype(float)
    b_ = eqx.error_if(b_, jnp.any(~jnp.isfinite(b_)), "b must be finite.")
    m = int(b_.shape[-1])
    q_ = _require_shape(q, case_shape + (n, n), "q")
    r_ = _require_shape(r, case_shape + (m, m), "r")
    if s is None:
        s_ = jnp.zeros(case_shape + (n, m), dtype=jnp.result_type(a_, b_, q_, r_))
    else:
        s_ = _require_shape(s, case_shape + (n, m), "s")
    q_ = _require_positive_semidefinite(q_, "q", cost_tolerance)
    r_ = _require_positive_definite(r_, "r", cost_tolerance)
    stage_hessian = jnp.concatenate(
        (
            jnp.concatenate((q_, s_), axis=-1),
            jnp.concatenate((jnp.swapaxes(s_, -1, -2), r_), axis=-1),
        ),
        axis=-2,
    )
    stage_hessian = _require_positive_semidefinite(
        stage_hessian, "the joint state-control cost", cost_tolerance
    )
    q_ = stage_hessian[..., :n, :n]
    s_ = stage_hessian[..., :n, n:]
    r_ = stage_hessian[..., n:, n:]
    dtype = jnp.result_type(a_, b_, q_, r_, s_)
    return (
        a_.astype(dtype),
        b_.astype(dtype),
        q_.astype(dtype),
        r_.astype(dtype),
        s_.astype(dtype),
    )


def _continuous_residual(
    p: Array,
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    s: Array,
    /,
) -> Array:
    right = jnp.swapaxes(b, -1, -2) @ p + jnp.swapaxes(s, -1, -2)
    return (
        jnp.swapaxes(a, -1, -2) @ p + p @ a - (p @ b + s) @ jnp.linalg.solve(r, right) + q
    )


def _discrete_residual(
    p: Array,
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    s: Array,
    /,
) -> Array:
    pb = p @ b
    control_hessian = r + jnp.swapaxes(b, -1, -2) @ pb
    cross = jnp.swapaxes(a, -1, -2) @ pb + s
    return (
        q
        + jnp.swapaxes(a, -1, -2) @ p @ a
        - cross @ jnp.linalg.solve(control_hessian, jnp.swapaxes(cross, -1, -2))
        - p
    )


def _care_primal(a: Array, b: Array, q: Array, r: Array, s: Array, /) -> Array:
    r_inv_st = jnp.linalg.solve(r, jnp.swapaxes(s, -1, -2))
    a_reduced = a - b @ r_inv_st
    q_reduced = q - s @ r_inv_st
    brb = b @ jnp.linalg.solve(r, jnp.swapaxes(b, -1, -2))
    hamiltonian = jnp.concatenate(
        (
            jnp.concatenate((a_reduced, -brb), axis=-1),
            jnp.concatenate((-q_reduced, -jnp.swapaxes(a_reduced, -1, -2)), axis=-1),
        ),
        axis=-2,
    )
    eigenvalues, eigenvectors = jnp.linalg.eig(hamiltonian)
    n = int(a.shape[-1])
    order = jnp.argsort(jnp.real(eigenvalues), axis=-1)
    columns = jnp.take_along_axis(eigenvectors, order[..., None, :n], axis=-1)
    upper = columns[..., :n, :]
    lower = columns[..., n:, :]
    p_complex = jnp.swapaxes(
        jnp.linalg.solve(jnp.swapaxes(upper, -1, -2), jnp.swapaxes(lower, -1, -2)),
        -1,
        -2,
    )
    p = jnp.real(p_complex)
    return 0.5 * (p + jnp.swapaxes(p, -1, -2))


def _batched_implicit_equation(
    matrix: Array,
    source: Array,
    /,
    *,
    discrete: bool,
) -> Array:
    """Apply the shared matrix-equation primitive across explicit case axes."""
    n = int(matrix.shape[-1])
    case_shape = tuple(source.shape[:-2])
    matrix_flat = matrix.reshape((-1, n, n))
    source_flat = source.reshape((-1, n, n))
    if discrete:
        solved = jax.vmap(
            lambda matrix_, source_: discrete_lyapunov_solution(
                matrix_, source_, max_dimension=n
            )
        )(matrix_flat, source_flat)
    else:
        solved = jax.vmap(
            lambda matrix_, source_: continuous_lyapunov_solution(
                matrix_, source_, max_dimension=n
            )
        )(matrix_flat, source_flat)
    return solved.reshape(case_shape + (n, n))


@jax.custom_jvp
def _care_solution(a: Array, b: Array, q: Array, r: Array, s: Array, /) -> Array:
    return _care_primal(a, b, q, r, s)


@_care_solution.defjvp
def _care_solution_jvp(primals, tangents):
    a, b, q, r, s = primals
    da, db, dq, dr, ds = tangents
    p = _care_solution(a, b, q, r, s)
    parameter_tangent = jax.jvp(
        lambda a_, b_, q_, r_, s_: _continuous_residual(p, a_, b_, q_, r_, s_),
        (a, b, q, r, s),
        (da, db, dq, dr, ds),
    )[1]
    gain = -jnp.linalg.solve(r, jnp.swapaxes(b, -1, -2) @ p + jnp.swapaxes(s, -1, -2))
    closed_loop = a + b @ gain
    dp = _batched_implicit_equation(
        jnp.swapaxes(closed_loop, -1, -2),
        parameter_tangent,
        discrete=False,
    )
    dp = 0.5 * (dp + jnp.swapaxes(dp, -1, -2))
    return p, dp


def _dare_primal(
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    s: Array,
    max_iterations: int,
    tolerance: float,
    /,
) -> tuple[Array, Array]:
    def body(_, carry):
        p, done, count = carry
        residual = _discrete_residual(p, a, b, q, r, s)
        candidate = p + residual
        candidate = 0.5 * (candidate + jnp.swapaxes(candidate, -1, -2))
        delta = jnp.linalg.norm(candidate - p, axis=(-2, -1))
        scale = jnp.linalg.norm(candidate, axis=(-2, -1))
        newly_done = delta <= tolerance * jnp.maximum(jnp.ones_like(scale), scale)
        active = ~done
        p = jnp.where(active[..., None, None], candidate, p)
        count = jnp.where(active, count + 1, count)
        return p, done | newly_done, count

    case_shape = tuple(a.shape[:-2])
    initial = (
        q,
        jnp.zeros(case_shape, dtype=bool),
        jnp.zeros(case_shape, dtype=jnp.int32),
    )
    p, _, count = jax.lax.fori_loop(0, max_iterations, body, initial)
    return p, count


def _solve_discrete_implicit(
    closed_loop: Array,
    right_hand_side: Array,
    /,
) -> Array:
    """Solve the DARE tangent with the shared discrete Lyapunov primitive."""
    return _batched_implicit_equation(
        jnp.swapaxes(closed_loop, -1, -2),
        right_hand_side,
        discrete=True,
    )


@partial(jax.custom_jvp, nondiff_argnums=(5, 6))
def _dare_solution(
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    s: Array,
    max_iterations: int,
    tolerance: float,
    /,
) -> tuple[Array, Array]:
    p, count = _dare_primal(a, b, q, r, s, max_iterations, tolerance)
    return p, count.astype(p.dtype)


@_dare_solution.defjvp
def _dare_solution_jvp(max_iterations, tolerance, primals, tangents):
    a, b, q, r, s = primals
    da, db, dq, dr, ds = tangents
    p, count = _dare_solution(a, b, q, r, s, max_iterations, tolerance)
    parameter_tangent = jax.jvp(
        lambda a_, b_, q_, r_, s_: _discrete_residual(p, a_, b_, q_, r_, s_),
        (a, b, q, r, s),
        (da, db, dq, dr, ds),
    )[1]
    control_hessian = r + jnp.swapaxes(b, -1, -2) @ p @ b
    gain = -jnp.linalg.solve(
        control_hessian,
        jnp.swapaxes(b, -1, -2) @ p @ a + jnp.swapaxes(s, -1, -2),
    )
    closed_loop = a + b @ gain
    dp = _solve_discrete_implicit(closed_loop, parameter_tangent)
    dp = 0.5 * (dp + jnp.swapaxes(dp, -1, -2))
    return (p, count), (dp, jnp.zeros_like(count))


def _pbh_diagnostics(
    a: Array,
    b: Array,
    q: Array,
    *,
    discrete: bool,
    tolerance: float,
) -> tuple[Array, Array]:
    n = int(a.shape[-1])
    eigenvalues = jnp.linalg.eigvals(a)
    identity = jnp.eye(n, dtype=eigenvalues.dtype)
    pencils = eigenvalues[..., :, None, None] * identity - a[..., None, :, :]
    b_columns = jnp.broadcast_to(
        b[..., None, :, :], pencils.shape[:-1] + (b.shape[-1],)
    ).astype(eigenvalues.dtype)
    controllability_pencils = jnp.concatenate((pencils, b_columns), axis=-1)
    q_rows = jnp.broadcast_to(
        q[..., None, :, :], pencils.shape[:-2] + q.shape[-2:]
    ).astype(eigenvalues.dtype)
    observability_pencils = jnp.concatenate((pencils, q_rows), axis=-2)
    controllability_rank = jnp.linalg.matrix_rank(controllability_pencils, tol=tolerance)
    observability_rank = jnp.linalg.matrix_rank(observability_pencils, tol=tolerance)
    if discrete:
        unstable = jnp.abs(eigenvalues) >= 1.0 - tolerance
    else:
        unstable = jnp.real(eigenvalues) >= -tolerance
    stabilizable = jnp.all((~unstable) | (controllability_rank == n), axis=-1)
    detectable = jnp.all((~unstable) | (observability_rank == n), axis=-1)
    return stabilizable, detectable


def _equation_condition(eigenvalues: Array, *, discrete: bool) -> tuple[Array, Array]:
    if discrete:
        separation = jnp.abs(
            1.0 - jnp.conj(eigenvalues)[..., :, None] * eigenvalues[..., None, :]
        )
    else:
        separation = jnp.abs(
            jnp.conj(eigenvalues)[..., :, None] + eigenvalues[..., None, :]
        )
    minimum = jnp.min(separation, axis=(-2, -1))
    maximum = jnp.max(separation, axis=(-2, -1))
    condition = jnp.where(minimum > 0.0, maximum / minimum, jnp.inf)
    return minimum, condition


def _diagnostics(
    p: Array,
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    s: Array,
    *,
    discrete: bool,
    tolerance: float,
    pbh_tolerance: float,
    iterations: Array,
    method: str,
) -> AlgebraicRiccatiDiagnostics:
    if discrete:
        residual = _discrete_residual(p, a, b, q, r, s)
        control_hessian = r + jnp.swapaxes(b, -1, -2) @ p @ b
    else:
        residual = _continuous_residual(p, a, b, q, r, s)
        control_hessian = r
    gain = -jnp.linalg.solve(
        control_hessian,
        jnp.swapaxes(b, -1, -2) @ p @ a + jnp.swapaxes(s, -1, -2)
        if discrete
        else jnp.swapaxes(b, -1, -2) @ p + jnp.swapaxes(s, -1, -2),
    )
    closed_loop = a + b @ gain
    closed_loop_eigenvalues = jnp.linalg.eigvals(closed_loop)
    stable = (
        jnp.all(jnp.abs(closed_loop_eigenvalues) < 1.0, axis=-1)
        if discrete
        else jnp.all(jnp.real(closed_loop_eigenvalues) < 0.0, axis=-1)
    )
    stabilizable, detectable = _pbh_diagnostics(
        a, b, q, discrete=discrete, tolerance=pbh_tolerance
    )
    residual_norm = jnp.linalg.norm(residual, axis=(-2, -1))
    scale = jnp.linalg.norm(q, axis=(-2, -1)) + jnp.linalg.norm(p, axis=(-2, -1))
    relative = jnp.where(scale > 0.0, residual_norm / scale, residual_norm)
    control_condition = jnp.linalg.cond(control_hessian)
    spectral_separation, equation_condition = _equation_condition(
        closed_loop_eigenvalues, discrete=discrete
    )
    finite = (
        jnp.all(jnp.isfinite(p), axis=(-2, -1))
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(control_condition)
        & jnp.isfinite(equation_condition)
    )
    equation_converged = finite & stable & (relative <= tolerance)
    status = jnp.where(
        ~stabilizable,
        int(RiccatiStatus.UNSTABILIZABLE),
        jnp.where(
            ~detectable,
            int(RiccatiStatus.UNDETECTABLE),
            jnp.where(
                ~finite,
                int(RiccatiStatus.NONFINITE),
                jnp.where(
                    ~stable,
                    int(RiccatiStatus.UNSTABLE_SOLUTION),
                    jnp.where(
                        relative > tolerance,
                        int(RiccatiStatus.NONCONVERGED),
                        int(RiccatiStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    equation_status = jnp.where(
        ~finite,
        int(LinearMatrixEquationStatus.NONFINITE),
        jnp.where(
            ~stable,
            int(LinearMatrixEquationStatus.UNSTABLE_SYSTEM),
            jnp.where(
                relative > tolerance,
                int(LinearMatrixEquationStatus.RESIDUAL_TOLERANCE_NOT_MET),
                int(LinearMatrixEquationStatus.CONVERGED),
            ),
        ),
    ).astype(jnp.int32)
    equation = LinearMatrixEquationDiagnostics(
        residual_norm=residual_norm,
        relative_residual=relative,
        condition_number=equation_condition,
        spectral_separation=spectral_separation,
        stable=stable,
        finite=finite,
        converged=equation_converged,
        status=equation_status,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        method=method,
        system_type="discrete" if discrete else "continuous",
    )
    return AlgebraicRiccatiDiagnostics(
        equation=equation,
        control_condition_number=control_condition,
        stabilizable=stabilizable,
        detectable=detectable,
        status=status,
    )


def solve_continuous_are(
    a: ArrayLike,
    b: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    /,
    *,
    s: ArrayLike | None = None,
    tolerance: float = 1e-9,
    pbh_tolerance: float = 1e-9,
    cost_tolerance: float = 1e-10,
) -> AlgebraicRiccatiResult:
    """Solve the continuous-time stabilizing algebraic Riccati equation.

    The cost convention is ``xᵀQx / 2 + uᵀRu / 2 + xᵀSu``. The
    Hamiltonian invariant-subspace solve is JAX-native; its derivative is the
    exact implicit Sylvester equation at the returned stabilizing solution.
    """
    if tolerance <= 0.0 or pbh_tolerance <= 0.0 or cost_tolerance < 0.0:
        raise ValueError(
            "Riccati tolerances must be positive (cost_tolerance may be zero)."
        )
    a_, b_, q_, r_, s_ = _validate_are_inputs(a, b, q, r, s, cost_tolerance)
    p = _care_solution(a_, b_, q_, r_, s_)
    diagnostics = _diagnostics(
        p,
        a_,
        b_,
        q_,
        r_,
        s_,
        discrete=False,
        tolerance=tolerance,
        pbh_tolerance=pbh_tolerance,
        iterations=jnp.zeros(a_.shape[:-2], dtype=jnp.int32),
        method="hamiltonian-invariant-subspace+implicit-sylvester-gradient",
    )
    return AlgebraicRiccatiResult(
        matrix=p,
        diagnostics=diagnostics,
        valid=diagnostics.converged,
        status=diagnostics.status,
    )


def solve_discrete_are(
    a: ArrayLike,
    b: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    /,
    *,
    s: ArrayLike | None = None,
    tolerance: float = 1e-9,
    pbh_tolerance: float = 1e-9,
    cost_tolerance: float = 1e-10,
    max_iterations: int = 512,
) -> AlgebraicRiccatiResult:
    """Solve the discrete-time stabilizing algebraic Riccati equation.

    The canonical primal is the monotone Riccati iteration. Gradients solve the
    exact implicit discrete Lyapunov equation and do not differentiate through
    the iteration count.
    """
    if tolerance <= 0.0 or pbh_tolerance <= 0.0 or cost_tolerance < 0.0:
        raise ValueError(
            "Riccati tolerances must be positive (cost_tolerance may be zero)."
        )
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    a_, b_, q_, r_, s_ = _validate_are_inputs(a, b, q, r, s, cost_tolerance)
    p, iterations_float = _dare_solution(a_, b_, q_, r_, s_, max_iterations, tolerance)
    diagnostics = _diagnostics(
        p,
        a_,
        b_,
        q_,
        r_,
        s_,
        discrete=True,
        tolerance=tolerance,
        pbh_tolerance=pbh_tolerance,
        iterations=iterations_float.astype(jnp.int32),
        method="sequential-fixed-point+implicit-discrete-lyapunov-gradient",
    )
    return AlgebraicRiccatiResult(
        matrix=p,
        diagnostics=diagnostics,
        valid=diagnostics.converged,
        status=diagnostics.status,
    )
