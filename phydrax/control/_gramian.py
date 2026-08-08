#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..solver._matrix_functions import matrix_exponential_action, MatrixFunctionPolicy
from ._lyapunov import (
    finite_continuous_lyapunov,
    finite_discrete_lyapunov,
    LinearMatrixEquationDiagnostics,
    LinearMatrixEquationStatus,
    solve_continuous_lyapunov,
    solve_discrete_lyapunov,
)


GramianKind: TypeAlias = Literal["controllability", "observability"]
GramianSystemType: TypeAlias = Literal["continuous", "discrete"]


class GramianDiagnostics(StrictModule):
    """Equation and positive-semidefinite diagnostics for a dense Gramian."""

    equation: LinearMatrixEquationDiagnostics
    minimum_eigenvalue: Array
    maximum_eigenvalue: Array
    gramian_condition_number: Array
    rank: Array
    rank_threshold: Array
    positive_semidefinite: Array
    singular: Array
    horizon: Array
    kind: GramianKind = eqx.field(static=True)
    finite_horizon: bool = eqx.field(static=True)

    @property
    def residual_norm(self) -> Array:
        return self.equation.residual_norm

    @property
    def relative_residual(self) -> Array:
        return self.equation.relative_residual

    @property
    def condition_number(self) -> Array:
        return self.equation.condition_number

    @property
    def stable(self) -> Array:
        return self.equation.stable

    @property
    def finite(self) -> Array:
        return self.equation.finite

    @property
    def converged(self) -> Array:
        return self.equation.converged

    @property
    def status(self) -> Array:
        return self.equation.status

    @property
    def method(self) -> str:
        return self.equation.method


class GramianResult(StrictModule):
    """A dense controllability or observability Gramian and diagnostics."""

    value: Array
    diagnostics: GramianDiagnostics


class GramianActionDiagnostics(StrictModule):
    """Work and approximation diagnostics for one matrix-free Gramian action."""

    quadrature_error_estimate: Array
    relative_quadrature_error: Array
    condition_number: Array
    finite: Array
    converged: Array
    status: Array
    horizon: Array
    terms: Array
    quadrature_order: int = eqx.field(static=True)
    krylov_dimension: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    kind: GramianKind = eqx.field(static=True)
    system_type: GramianSystemType = eqx.field(static=True)


class GramianActionResult(StrictModule):
    """The value and diagnostics of a matrix-free Gramian-vector product."""

    value: Array
    diagnostics: GramianActionDiagnostics


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _system_matrix(matrix: ArrayLike, /, *, max_dimension: int) -> tuple[Array, int]:
    dimension_limit = int(max_dimension)
    if dimension_limit <= 0:
        raise ValueError("max_dimension must be positive.")
    matrix_array = _inexact(matrix)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError(f"matrix must be square; got shape {matrix_array.shape}.")
    dimension = int(matrix_array.shape[0])
    if dimension == 0:
        raise ValueError("matrix must have positive dimension.")
    if dimension > dimension_limit:
        raise ValueError(
            f"Dense Gramian dimension {dimension} exceeds "
            f"max_dimension={dimension_limit}."
        )
    return matrix_array, dimension


def _controllability_source(
    matrix: ArrayLike,
    input_matrix: ArrayLike,
    /,
    *,
    max_dimension: int,
) -> tuple[Array, Array]:
    matrix_array, dimension = _system_matrix(matrix, max_dimension=max_dimension)
    input_array = _inexact(input_matrix)
    if input_array.ndim != 2 or input_array.shape[0] != dimension:
        raise ValueError(
            "input_matrix must have shape (state_dimension, input_dimension); "
            f"got {input_array.shape}."
        )
    dtype = jnp.result_type(matrix_array.dtype, input_array.dtype)
    matrix_array = matrix_array.astype(dtype)
    input_array = input_array.astype(dtype)
    return matrix_array, input_array @ jnp.conj(input_array.T)


def _observability_source(
    matrix: ArrayLike,
    output_matrix: ArrayLike,
    /,
    *,
    max_dimension: int,
) -> tuple[Array, Array]:
    matrix_array, dimension = _system_matrix(matrix, max_dimension=max_dimension)
    output_array = _inexact(output_matrix)
    if output_array.ndim != 2 or output_array.shape[1] != dimension:
        raise ValueError(
            "output_matrix must have shape (output_dimension, state_dimension); "
            f"got {output_array.shape}."
        )
    dtype = jnp.result_type(matrix_array.dtype, output_array.dtype)
    matrix_array = matrix_array.astype(dtype)
    output_array = output_array.astype(dtype)
    return jnp.conj(matrix_array.T), jnp.conj(output_array.T) @ output_array


def _infinite_horizon_equation(
    diagnostics: LinearMatrixEquationDiagnostics,
    matrix: Array,
    source: Array,
    /,
    *,
    system_type: GramianSystemType,
    stability_tolerance: float,
) -> LinearMatrixEquationDiagnostics:
    if system_type == "continuous":
        boundary_measure = jnp.max(jnp.real(jnp.linalg.eigvals(matrix)))
        marginal = (~diagnostics.stable) & (
            boundary_measure <= float(stability_tolerance)
        )
    else:
        boundary_measure = jnp.max(jnp.abs(jnp.linalg.eigvals(matrix)))
        marginal = (~diagnostics.stable) & (
            boundary_measure <= 1.0 + float(stability_tolerance)
        )
    inputs_finite = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(source))
    acceptable = diagnostics.converged & diagnostics.stable
    status = jnp.where(
        diagnostics.stable,
        diagnostics.status,
        int(LinearMatrixEquationStatus.UNSTABLE_SYSTEM),
    )
    status = jnp.where(
        marginal,
        int(LinearMatrixEquationStatus.MARGINAL_SYSTEM),
        status,
    )
    status = jnp.where(
        inputs_finite,
        status,
        int(LinearMatrixEquationStatus.NONFINITE),
    )
    return LinearMatrixEquationDiagnostics(
        residual_norm=diagnostics.residual_norm,
        relative_residual=diagnostics.relative_residual,
        condition_number=diagnostics.condition_number,
        spectral_separation=diagnostics.spectral_separation,
        stable=diagnostics.stable,
        finite=diagnostics.finite,
        converged=acceptable,
        status=jnp.asarray(status, dtype=jnp.int32),
        iterations=diagnostics.iterations,
        method=diagnostics.method,
        system_type=diagnostics.system_type,
    )


def _gramian_diagnostics(
    value: Array,
    equation: LinearMatrixEquationDiagnostics,
    /,
    *,
    horizon: Array,
    kind: GramianKind,
    finite_horizon: bool,
    psd_tolerance: float,
    rank_tolerance: float | None,
) -> GramianDiagnostics:
    psd_threshold = float(psd_tolerance)
    if psd_threshold < 0.0:
        raise ValueError("psd_tolerance must be non-negative.")
    if rank_tolerance is not None and float(rank_tolerance) < 0.0:
        raise ValueError("rank_tolerance must be non-negative or None.")
    hermitian = 0.5 * (value + jnp.conj(value.T))
    eigenvalues = jnp.linalg.eigvalsh(hermitian)
    minimum = jnp.min(eigenvalues)
    maximum = jnp.max(eigenvalues)
    scale = jnp.maximum(jnp.asarray(1.0, dtype=maximum.dtype), jnp.abs(maximum))
    if rank_tolerance is None:
        rank_threshold = jnp.finfo(value.real.dtype).eps * value.shape[0] * scale
    else:
        rank_threshold = jnp.asarray(rank_tolerance, dtype=maximum.dtype)
    rank = jnp.sum(eigenvalues > rank_threshold, dtype=jnp.int32)
    singular = rank < value.shape[0]
    gramian_condition = jnp.where(
        minimum > rank_threshold,
        maximum / minimum,
        jnp.inf,
    )
    positive_semidefinite = minimum >= -psd_threshold * scale
    return GramianDiagnostics(
        equation=equation,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        gramian_condition_number=gramian_condition,
        rank=rank,
        rank_threshold=rank_threshold,
        positive_semidefinite=positive_semidefinite,
        singular=singular,
        horizon=horizon,
        kind=kind,
        finite_horizon=finite_horizon,
    )


def _gramian_result(
    value: Array,
    equation: LinearMatrixEquationDiagnostics,
    /,
    *,
    horizon: Array,
    kind: GramianKind,
    finite_horizon: bool,
    psd_tolerance: float,
    rank_tolerance: float | None,
) -> GramianResult:
    return GramianResult(
        value=value,
        diagnostics=_gramian_diagnostics(
            value,
            equation,
            horizon=horizon,
            kind=kind,
            finite_horizon=finite_horizon,
            psd_tolerance=psd_tolerance,
            rank_tolerance=rank_tolerance,
        ),
    )


def continuous_controllability_gramian(
    matrix: ArrayLike,
    input_matrix: ArrayLike,
    /,
    *,
    horizon: ArrayLike | None = None,
    method: Literal["schur", "cayley"] = "schur",
    max_dimension: int = 128,
    tolerance: float = 1e-6,
    stability_tolerance: float = 1e-7,
    separation_tolerance: float = 1e-10,
    max_iterations: int = 32,
    cayley_shift: ArrayLike | None = None,
    psd_tolerance: float = 1e-7,
    rank_tolerance: float | None = None,
) -> GramianResult:
    r"""Return ``∫ exp(A t) B Bᴴ exp(Aᴴ t) dt``.

    ``horizon=None`` denotes the infinite integral and is successful only for a
    Hurwitz matrix. A finite horizon is valid for stable, marginal, and unstable
    matrices.
    """
    matrix_array, source = _controllability_source(
        matrix, input_matrix, max_dimension=max_dimension
    )
    if horizon is None:
        result = solve_continuous_lyapunov(
            matrix_array,
            source,
            method=method,
            max_dimension=max_dimension,
            tolerance=tolerance,
            stability_tolerance=stability_tolerance,
            separation_tolerance=separation_tolerance,
            max_iterations=max_iterations,
            cayley_shift=cayley_shift,
        )
        equation = _infinite_horizon_equation(
            result.diagnostics,
            matrix_array,
            source,
            system_type="continuous",
            stability_tolerance=stability_tolerance,
        )
        horizon_value = jnp.asarray(jnp.inf, dtype=matrix_array.real.dtype)
        finite_horizon = False
    else:
        result = finite_continuous_lyapunov(
            matrix_array,
            source,
            horizon,
            max_dimension=max_dimension,
            tolerance=tolerance,
            stability_tolerance=stability_tolerance,
        )
        equation = result.diagnostics
        horizon_value = jnp.asarray(horizon, dtype=matrix_array.real.dtype).reshape(())
        finite_horizon = True
    return _gramian_result(
        result.value,
        equation,
        horizon=horizon_value,
        kind="controllability",
        finite_horizon=finite_horizon,
        psd_tolerance=psd_tolerance,
        rank_tolerance=rank_tolerance,
    )


def continuous_observability_gramian(
    matrix: ArrayLike,
    output_matrix: ArrayLike,
    /,
    *,
    horizon: ArrayLike | None = None,
    method: Literal["schur", "cayley"] = "schur",
    max_dimension: int = 128,
    tolerance: float = 1e-6,
    stability_tolerance: float = 1e-7,
    separation_tolerance: float = 1e-10,
    max_iterations: int = 32,
    cayley_shift: ArrayLike | None = None,
    psd_tolerance: float = 1e-7,
    rank_tolerance: float | None = None,
) -> GramianResult:
    r"""Return ``∫ exp(Aᴴ t) Cᴴ C exp(A t) dt``."""
    generator, source = _observability_source(
        matrix, output_matrix, max_dimension=max_dimension
    )
    if horizon is None:
        result = solve_continuous_lyapunov(
            generator,
            source,
            method=method,
            max_dimension=max_dimension,
            tolerance=tolerance,
            stability_tolerance=stability_tolerance,
            separation_tolerance=separation_tolerance,
            max_iterations=max_iterations,
            cayley_shift=cayley_shift,
        )
        equation = _infinite_horizon_equation(
            result.diagnostics,
            generator,
            source,
            system_type="continuous",
            stability_tolerance=stability_tolerance,
        )
        horizon_value = jnp.asarray(jnp.inf, dtype=generator.real.dtype)
        finite_horizon = False
    else:
        result = finite_continuous_lyapunov(
            generator,
            source,
            horizon,
            max_dimension=max_dimension,
            tolerance=tolerance,
            stability_tolerance=stability_tolerance,
        )
        equation = result.diagnostics
        horizon_value = jnp.asarray(horizon, dtype=generator.real.dtype).reshape(())
        finite_horizon = True
    return _gramian_result(
        result.value,
        equation,
        horizon=horizon_value,
        kind="observability",
        finite_horizon=finite_horizon,
        psd_tolerance=psd_tolerance,
        rank_tolerance=rank_tolerance,
    )


def discrete_controllability_gramian(
    matrix: ArrayLike,
    input_matrix: ArrayLike,
    /,
    *,
    steps: int | None = None,
    method: Literal["schur", "doubling"] = "schur",
    max_dimension: int = 128,
    tolerance: float = 1e-6,
    stability_tolerance: float = 1e-7,
    separation_tolerance: float = 1e-10,
    max_iterations: int = 32,
    psd_tolerance: float = 1e-7,
    rank_tolerance: float | None = None,
) -> GramianResult:
    r"""Return ``Σ Aᵏ B Bᴴ (Aᴴ)ᵏ`` for finite or infinite horizon."""
    matrix_array, source = _controllability_source(
        matrix, input_matrix, max_dimension=max_dimension
    )
    if steps is None:
        result = solve_discrete_lyapunov(
            matrix_array,
            source,
            method=method,
            max_dimension=max_dimension,
            tolerance=tolerance,
            stability_tolerance=stability_tolerance,
            separation_tolerance=separation_tolerance,
            max_iterations=max_iterations,
        )
        equation = _infinite_horizon_equation(
            result.diagnostics,
            matrix_array,
            source,
            system_type="discrete",
            stability_tolerance=stability_tolerance,
        )
        horizon_value = jnp.asarray(jnp.inf, dtype=matrix_array.real.dtype)
        finite_horizon = False
    else:
        result = finite_discrete_lyapunov(
            matrix_array,
            source,
            steps,
            max_dimension=max_dimension,
            tolerance=tolerance,
            stability_tolerance=stability_tolerance,
        )
        equation = result.diagnostics
        horizon_value = jnp.asarray(int(steps), dtype=matrix_array.real.dtype)
        finite_horizon = True
    return _gramian_result(
        result.value,
        equation,
        horizon=horizon_value,
        kind="controllability",
        finite_horizon=finite_horizon,
        psd_tolerance=psd_tolerance,
        rank_tolerance=rank_tolerance,
    )


def discrete_observability_gramian(
    matrix: ArrayLike,
    output_matrix: ArrayLike,
    /,
    *,
    steps: int | None = None,
    method: Literal["schur", "doubling"] = "schur",
    max_dimension: int = 128,
    tolerance: float = 1e-6,
    stability_tolerance: float = 1e-7,
    separation_tolerance: float = 1e-10,
    max_iterations: int = 32,
    psd_tolerance: float = 1e-7,
    rank_tolerance: float | None = None,
) -> GramianResult:
    r"""Return ``Σ (Aᴴ)ᵏ Cᴴ C Aᵏ`` for finite or infinite horizon."""
    generator, source = _observability_source(
        matrix, output_matrix, max_dimension=max_dimension
    )
    if steps is None:
        result = solve_discrete_lyapunov(
            generator,
            source,
            method=method,
            max_dimension=max_dimension,
            tolerance=tolerance,
            stability_tolerance=stability_tolerance,
            separation_tolerance=separation_tolerance,
            max_iterations=max_iterations,
        )
        equation = _infinite_horizon_equation(
            result.diagnostics,
            generator,
            source,
            system_type="discrete",
            stability_tolerance=stability_tolerance,
        )
        horizon_value = jnp.asarray(jnp.inf, dtype=generator.real.dtype)
        finite_horizon = False
    else:
        result = finite_discrete_lyapunov(
            generator,
            source,
            steps,
            max_dimension=max_dimension,
            tolerance=tolerance,
            stability_tolerance=stability_tolerance,
        )
        equation = result.diagnostics
        horizon_value = jnp.asarray(int(steps), dtype=generator.real.dtype)
        finite_horizon = True
    return _gramian_result(
        result.value,
        equation,
        horizon=horizon_value,
        kind="observability",
        finite_horizon=finite_horizon,
        psd_tolerance=psd_tolerance,
        rank_tolerance=rank_tolerance,
    )


def _checked_action(
    action: Callable[[Array], ArrayLike],
    value: Array,
    /,
    *,
    owner: str,
    expected_shape: tuple[int, ...] | None = None,
) -> Array:
    if not callable(action):
        raise TypeError(f"{owner} must be callable.")
    result = _inexact(action(value))
    if expected_shape is not None and result.shape != expected_shape:
        raise ValueError(
            f"{owner} must return shape {expected_shape}; got {result.shape}."
        )
    return result


def _complex_arnoldi_exponential_action(
    action: Callable[[Array], ArrayLike],
    vector: Array,
    time: Array,
    /,
    *,
    policy: MatrixFunctionPolicy,
) -> Array:
    if policy.method not in ("auto", "arnoldi"):
        raise ValueError(
            "Complex Gramian actions require an Arnoldi matrix-function policy."
        )
    size = int(vector.size)
    iterations = min(policy.num_matvecs, size)
    norm = jnp.linalg.norm(vector)
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    basis = jnp.zeros((size, iterations + 1), dtype=vector.dtype)
    basis = basis.at[:, 0].set(vector / safe_norm)
    hessenberg = jnp.zeros((iterations + 1, iterations), dtype=vector.dtype)

    def body(index, carry):
        vectors, projected = carry
        image = _checked_action(
            action,
            vectors[:, index],
            owner="generator action",
            expected_shape=vector.shape,
        )
        coefficients = jnp.conj(vectors.T) @ image
        residual = image - vectors @ coefficients
        if policy.reorthogonalization == "full":
            correction = jnp.conj(vectors.T) @ residual
            coefficients = coefficients + correction
            residual = residual - vectors @ correction
        residual_norm = jnp.linalg.norm(residual)
        projected = projected.at[:, index].set(coefficients)
        projected = projected.at[index + 1, index].set(residual_norm)
        next_vector = jnp.where(
            residual_norm > jnp.finfo(vector.real.dtype).eps,
            residual / jnp.where(residual_norm > 0.0, residual_norm, 1.0),
            jnp.zeros_like(residual),
        )
        vectors = vectors.at[:, index + 1].set(next_vector)
        return vectors, projected

    basis, hessenberg = jax.lax.fori_loop(0, iterations, body, (basis, hessenberg))
    exponential = jsp.linalg.expm(time * hessenberg[:iterations, :iterations])
    approximation = safe_norm * (basis[:, :iterations] @ exponential[:, 0])
    return jnp.where(norm > 0.0, approximation, jnp.zeros_like(approximation))


def _exponential_action(
    action: Callable[[Array], ArrayLike],
    vector: Array,
    time: Array,
    /,
    *,
    policy: MatrixFunctionPolicy,
) -> Array:
    if jnp.issubdtype(vector.dtype, jnp.complexfloating):
        return _complex_arnoldi_exponential_action(action, vector, time, policy=policy)
    return matrix_exponential_action(action, vector, time, policy=policy)


def _continuous_action_quadrature(
    generator: Callable[[Array], ArrayLike],
    adjoint_generator: Callable[[Array], ArrayLike],
    source: Callable[[Array], ArrayLike],
    vector: Array,
    horizon: Array,
    /,
    *,
    order: int,
    policy: MatrixFunctionPolicy,
) -> Array:
    nodes_host, weights_host = np.polynomial.legendre.leggauss(order)
    nodes = jnp.asarray(nodes_host, dtype=horizon.dtype)
    weights = jnp.asarray(weights_host, dtype=horizon.dtype)
    result = jnp.zeros_like(vector)
    for index in range(order):
        time = 0.5 * horizon * (nodes[index] + 1.0)
        propagated_adjoint = _exponential_action(
            adjoint_generator, vector, time, policy=policy
        )
        sourced = _checked_action(
            source,
            propagated_adjoint,
            owner="source action",
            expected_shape=vector.shape,
        )
        propagated = _exponential_action(generator, sourced, time, policy=policy)
        result = result + weights[index] * propagated
    return 0.5 * horizon * result


def _continuous_gramian_action(
    generator: Callable[[Array], ArrayLike],
    adjoint_generator: Callable[[Array], ArrayLike],
    source: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    horizon: ArrayLike,
    /,
    *,
    kind: GramianKind,
    quadrature_order: int,
    krylov_dimension: int,
    action_tolerance: float,
    policy: MatrixFunctionPolicy | None,
) -> GramianActionResult:
    if not callable(generator) or not callable(adjoint_generator):
        raise TypeError("generator and adjoint_generator must be callable.")
    value = _inexact(vector)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("vector must be a non-empty rank-one array.")
    duration = jnp.asarray(horizon, dtype=value.real.dtype).reshape(())
    duration = eqx.error_if(
        duration,
        (~jnp.isfinite(duration)) | (duration < 0.0),
        "horizon must be finite and non-negative.",
    )
    order = int(quadrature_order)
    krylov_count = int(krylov_dimension)
    if order < 2:
        raise ValueError("quadrature_order must be at least 2.")
    if krylov_count <= 0:
        raise ValueError("krylov_dimension must be positive.")
    accuracy = float(action_tolerance)
    if accuracy < 0.0:
        raise ValueError("action_tolerance must be non-negative.")
    selected_policy = (
        MatrixFunctionPolicy("arnoldi", num_matvecs=krylov_count)
        if policy is None
        else policy
    )
    if not isinstance(selected_policy, MatrixFunctionPolicy):
        raise TypeError("policy must be a MatrixFunctionPolicy or None.")
    approximation = _continuous_action_quadrature(
        generator,
        adjoint_generator,
        source,
        value,
        duration,
        order=order,
        policy=selected_policy,
    )
    comparison_order = max(1, order // 2)
    comparison = _continuous_action_quadrature(
        generator,
        adjoint_generator,
        source,
        value,
        duration,
        order=comparison_order,
        policy=selected_policy,
    )
    error = jnp.linalg.norm(approximation - comparison)
    norm = jnp.linalg.norm(approximation)
    relative_error = jnp.where(norm > 0.0, error / norm, error)
    finite = jnp.all(jnp.isfinite(approximation)) & jnp.isfinite(error)
    krylov_complete = (
        selected_policy.method in ("auto", "arnoldi")
        and selected_policy.num_matvecs >= value.size
    )
    converged = finite & (relative_error <= accuracy) & krylov_complete
    status = jnp.where(
        converged,
        int(LinearMatrixEquationStatus.CONVERGED),
        int(LinearMatrixEquationStatus.RESIDUAL_TOLERANCE_NOT_MET),
    )
    status = jnp.where(
        finite,
        status,
        int(LinearMatrixEquationStatus.NONFINITE),
    )
    return GramianActionResult(
        value=approximation,
        diagnostics=GramianActionDiagnostics(
            quadrature_error_estimate=error,
            relative_quadrature_error=relative_error,
            condition_number=jnp.asarray(jnp.nan, dtype=value.real.dtype),
            finite=finite,
            converged=converged,
            status=jnp.asarray(status, dtype=jnp.int32),
            horizon=duration,
            terms=jnp.asarray(order, dtype=jnp.int32),
            quadrature_order=order,
            krylov_dimension=selected_policy.num_matvecs,
            method="gauss-legendre-krylov-action",
            kind=kind,
            system_type="continuous",
        ),
    )


def continuous_controllability_gramian_action(
    generator: Callable[[Array], ArrayLike],
    adjoint_generator: Callable[[Array], ArrayLike],
    input_action: Callable[[Array], ArrayLike],
    adjoint_input_action: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    horizon: ArrayLike,
    /,
    *,
    quadrature_order: int = 16,
    krylov_dimension: int = 32,
    action_tolerance: float = 1e-6,
    policy: MatrixFunctionPolicy | None = None,
) -> GramianActionResult:
    """Apply a finite continuous controllability Gramian using only actions."""
    value = _inexact(vector)

    def source(state):
        control = _checked_action(
            adjoint_input_action, state, owner="adjoint_input_action"
        )
        return _checked_action(
            input_action,
            control,
            owner="input_action",
            expected_shape=value.shape,
        )

    return _continuous_gramian_action(
        generator,
        adjoint_generator,
        source,
        value,
        horizon,
        kind="controllability",
        quadrature_order=quadrature_order,
        krylov_dimension=krylov_dimension,
        action_tolerance=action_tolerance,
        policy=policy,
    )


def continuous_observability_gramian_action(
    generator: Callable[[Array], ArrayLike],
    adjoint_generator: Callable[[Array], ArrayLike],
    output_action: Callable[[Array], ArrayLike],
    adjoint_output_action: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    horizon: ArrayLike,
    /,
    *,
    quadrature_order: int = 16,
    krylov_dimension: int = 32,
    action_tolerance: float = 1e-6,
    policy: MatrixFunctionPolicy | None = None,
) -> GramianActionResult:
    """Apply a finite continuous observability Gramian using only actions."""
    value = _inexact(vector)

    def source(state):
        output = _checked_action(output_action, state, owner="output_action")
        return _checked_action(
            adjoint_output_action,
            output,
            owner="adjoint_output_action",
            expected_shape=value.shape,
        )

    return _continuous_gramian_action(
        adjoint_generator,
        generator,
        source,
        value,
        horizon,
        kind="observability",
        quadrature_order=quadrature_order,
        krylov_dimension=krylov_dimension,
        policy=policy,
        action_tolerance=action_tolerance,
    )


def _scanned_discrete_action(
    generator: Callable[[Array], ArrayLike],
    adjoint_generator: Callable[[Array], ArrayLike],
    source: Callable[[Array], ArrayLike],
    vector: Array,
    depth: int,
    /,
) -> Array:
    def advance_adjoint(current, _):
        following = _checked_action(
            adjoint_generator,
            current,
            owner="adjoint_generator",
            expected_shape=vector.shape,
        )
        return following, current

    _, adjoint_values = jax.lax.scan(
        advance_adjoint,
        vector,
        xs=None,
        length=depth,
    )

    def accumulate(current, adjoint_value):
        propagated = _checked_action(
            generator,
            current,
            owner="generator",
            expected_shape=vector.shape,
        )
        sourced = _checked_action(
            source,
            adjoint_value,
            owner="source action",
            expected_shape=vector.shape,
        )
        return sourced + propagated, None

    result, _ = jax.lax.scan(
        accumulate,
        jnp.zeros_like(vector),
        adjoint_values,
        reverse=True,
    )
    return result


def _discrete_gramian_action(
    generator: Callable[[Array], ArrayLike],
    adjoint_generator: Callable[[Array], ArrayLike],
    source: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    steps: int,
    /,
    *,
    kind: GramianKind,
) -> GramianActionResult:
    if not callable(generator) or not callable(adjoint_generator):
        raise TypeError("generator and adjoint_generator must be callable.")
    value = _inexact(vector)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("vector must be a non-empty rank-one array.")
    step_count = int(steps)
    if step_count < 0:
        raise ValueError("steps must be non-negative.")
    result = _scanned_discrete_action(
        generator,
        adjoint_generator,
        source,
        value,
        step_count,
    )
    finite = jnp.all(jnp.isfinite(result))
    status = jnp.where(
        finite,
        int(LinearMatrixEquationStatus.CONVERGED),
        int(LinearMatrixEquationStatus.NONFINITE),
    )
    zero = jnp.asarray(0.0, dtype=value.real.dtype)
    return GramianActionResult(
        value=result,
        diagnostics=GramianActionDiagnostics(
            quadrature_error_estimate=zero,
            relative_quadrature_error=zero,
            condition_number=jnp.asarray(jnp.nan, dtype=value.real.dtype),
            finite=finite,
            converged=finite,
            status=jnp.asarray(status, dtype=jnp.int32),
            horizon=jnp.asarray(step_count, dtype=value.real.dtype),
            terms=jnp.asarray(step_count, dtype=jnp.int32),
            quadrature_order=0,
            krylov_dimension=0,
            method="finite-scanned-action",
            kind=kind,
            system_type="discrete",
        ),
    )


def discrete_controllability_gramian_action(
    generator: Callable[[Array], ArrayLike],
    adjoint_generator: Callable[[Array], ArrayLike],
    input_action: Callable[[Array], ArrayLike],
    adjoint_input_action: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    steps: int,
    /,
) -> GramianActionResult:
    """Apply a finite discrete controllability Gramian using only actions."""
    value = _inexact(vector)

    def source(state):
        control = _checked_action(
            adjoint_input_action, state, owner="adjoint_input_action"
        )
        return _checked_action(
            input_action,
            control,
            owner="input_action",
            expected_shape=value.shape,
        )

    return _discrete_gramian_action(
        generator,
        adjoint_generator,
        source,
        value,
        steps,
        kind="controllability",
    )


def discrete_observability_gramian_action(
    generator: Callable[[Array], ArrayLike],
    adjoint_generator: Callable[[Array], ArrayLike],
    output_action: Callable[[Array], ArrayLike],
    adjoint_output_action: Callable[[Array], ArrayLike],
    vector: ArrayLike,
    steps: int,
    /,
) -> GramianActionResult:
    """Apply a finite discrete observability Gramian using only actions."""
    value = _inexact(vector)

    def source(state):
        output = _checked_action(output_action, state, owner="output_action")
        return _checked_action(
            adjoint_output_action,
            output,
            owner="adjoint_output_action",
            expected_shape=value.shape,
        )

    return _discrete_gramian_action(
        adjoint_generator,
        generator,
        source,
        value,
        steps,
        kind="observability",
    )


__all__ = [
    "GramianActionDiagnostics",
    "GramianActionResult",
    "GramianDiagnostics",
    "GramianKind",
    "GramianResult",
    "GramianSystemType",
    "continuous_controllability_gramian",
    "continuous_controllability_gramian_action",
    "continuous_observability_gramian",
    "continuous_observability_gramian_action",
    "discrete_controllability_gramian",
    "discrete_controllability_gramian_action",
    "discrete_observability_gramian",
    "discrete_observability_gramian_action",
]
