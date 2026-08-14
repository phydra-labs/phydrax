#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    prepare,
    solve,
    TolerancePolicy,
)


LyapunovMethod: TypeAlias = Literal["schur", "cayley", "doubling", "gmres"]
LyapunovSystemType: TypeAlias = Literal["continuous", "discrete"]


class LinearMatrixEquationStatus(IntEnum):
    """Stable status codes shared by diagnosed linear matrix equations."""

    CONVERGED = 0
    RESIDUAL_TOLERANCE_NOT_MET = 1
    SINGULAR_EQUATION = 2
    NONFINITE = 3
    UNSTABLE_SYSTEM = 4
    MARGINAL_SYSTEM = 5


_STATUS_MESSAGES = {
    LinearMatrixEquationStatus.CONVERGED: "converged",
    LinearMatrixEquationStatus.RESIDUAL_TOLERANCE_NOT_MET: (
        "residual tolerance was not met"
    ),
    LinearMatrixEquationStatus.SINGULAR_EQUATION: (
        "the Lyapunov operator is singular or below the requested separation"
    ),
    LinearMatrixEquationStatus.NONFINITE: "an input or computed quantity is non-finite",
    LinearMatrixEquationStatus.UNSTABLE_SYSTEM: (
        "the infinite-horizon system is unstable"
    ),
    LinearMatrixEquationStatus.MARGINAL_SYSTEM: (
        "the infinite-horizon system is marginally stable"
    ),
}


def linear_matrix_equation_status_message(
    status: int | LinearMatrixEquationStatus,
    /,
) -> str:
    """Return the stable human-readable description of a matrix-equation status."""
    return _STATUS_MESSAGES[LinearMatrixEquationStatus(int(status))]


class LinearMatrixEquationDiagnostics(StrictModule):
    """Residual, conditioning, stability, and method diagnostics.

    ``condition_number`` is the eigenvalue-separation condition estimate. It is
    the exact 2-norm condition number of the Lyapunov operator when the system
    matrix is normal; for nonnormal matrices it intentionally does not conceal
    eigenvector conditioning.
    """

    residual_norm: Array
    relative_residual: Array
    condition_number: Array
    spectral_separation: Array
    stable: Array
    finite: Array
    converged: Array
    status: Array
    iterations: Array
    method: str = eqx.field(static=True)
    system_type: LyapunovSystemType = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LinearMatrixEquationStatus.CONVERGED)


class LyapunovResult(StrictModule):
    """A Lyapunov solution and diagnostics for the equation actually solved."""

    value: Array
    diagnostics: LinearMatrixEquationDiagnostics


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _matrix_inputs(
    matrix: ArrayLike,
    source: ArrayLike,
    /,
    *,
    max_dimension: int,
) -> tuple[Array, Array, int]:
    dimension_limit = int(max_dimension)
    if dimension_limit <= 0:
        raise ValueError("max_dimension must be positive.")
    matrix_array = _inexact(matrix)
    source_array = _inexact(source)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError(f"matrix must be square; got shape {matrix_array.shape}.")
    dimension = int(matrix_array.shape[0])
    if dimension == 0:
        raise ValueError("matrix must have positive dimension.")
    if dimension > dimension_limit:
        raise ValueError(
            f"Dense Lyapunov dimension {dimension} exceeds "
            f"max_dimension={dimension_limit}."
        )
    if source_array.shape != (dimension, dimension):
        raise ValueError(
            f"source must have shape {(dimension, dimension)}; got {source_array.shape}."
        )
    dtype = jnp.result_type(matrix_array.dtype, source_array.dtype)
    return matrix_array.astype(dtype), source_array.astype(dtype), dimension


def _validate_solver_scalars(
    tolerance: float,
    stability_tolerance: float,
    separation_tolerance: float,
) -> tuple[float, float, float]:
    residual_tolerance = float(tolerance)
    stability_threshold = float(stability_tolerance)
    separation_threshold = float(separation_tolerance)
    if residual_tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    if stability_threshold < 0.0:
        raise ValueError("stability_tolerance must be non-negative.")
    if separation_threshold < 0.0:
        raise ValueError("separation_tolerance must be non-negative.")
    return residual_tolerance, stability_threshold, separation_threshold


def _continuous_schur_solution(matrix: Array, source: Array, /) -> Array:
    adjoint = jnp.conj(matrix.T)

    def equation(value):
        return matrix @ value + value @ adjoint

    def solve(_, right_hand_side):
        return jsp.linalg.solve_sylvester(
            matrix,
            adjoint,
            right_hand_side,
            method="schur",
        )

    def transpose_solve(_, right_hand_side):
        return jsp.linalg.solve_sylvester(
            matrix.T,
            jnp.conj(matrix),
            right_hand_side,
            method="schur",
        )

    return jax.lax.custom_linear_solve(
        equation,
        -source,
        solve=solve,
        transpose_solve=transpose_solve,
    )


def _discrete_schur_impl(matrix: Array, source: Array, /) -> Array:
    """Solve X - A X Aᴴ = Q through a Cayley-transformed Sylvester equation."""
    size = matrix.shape[0]
    identity = jnp.eye(size, dtype=matrix.dtype)
    plus = matrix + identity
    prepared = prepare(
        LinearSystem(DenseLinearOperator(plus)),
        LinearSolvePolicy(DenseLU()),
    )
    initial = solve(
        prepared,
        jnp.concatenate((matrix - identity, source), axis=1),
    ).value
    cayley = initial[:, :size]
    left_source = initial[:, size:]
    transformed_source = 2.0 * jnp.conj(solve(prepared, jnp.conj(left_source.T)).value.T)
    return jsp.linalg.solve_sylvester(
        cayley,
        jnp.conj(cayley.T),
        -transformed_source,
        method="schur",
    )


def _discrete_schur_solution(matrix: Array, source: Array, /) -> Array:
    adjoint = jnp.conj(matrix.T)

    def equation(value):
        return value - matrix @ value @ adjoint

    def solve(_, right_hand_side):
        return _discrete_schur_impl(matrix, right_hand_side)

    def transpose_solve(_, right_hand_side):
        return _discrete_schur_impl(matrix.T, right_hand_side)

    return jax.lax.custom_linear_solve(
        equation,
        source,
        solve=solve,
        transpose_solve=transpose_solve,
    )


def _discrete_doubling_solution(
    matrix: Array,
    source: Array,
    /,
    *,
    max_iterations: int,
) -> Array:
    def body(_, carry):
        power, solution = carry
        solution = solution + power @ solution @ jnp.conj(power.T)
        return power @ power, solution

    _, solution = jax.lax.fori_loop(
        0,
        max_iterations,
        body,
        (matrix, source),
    )
    return solution


def continuous_lyapunov_solution(
    matrix: ArrayLike,
    source: ArrayLike,
    /,
    *,
    method: Literal["schur", "cayley"] = "schur",
    max_dimension: int = 128,
    max_iterations: int = 32,
    cayley_shift: ArrayLike | None = None,
) -> Array:
    r"""Return ``X`` solving ``A X + X Aᴴ + Q = 0``.

    The Schur path is a JAX-native Bartels--Stewart solve with an analytic JVP.
    The Cayley path maps a stable continuous equation to a discrete equation and
    applies a fixed-count doubling iteration. Neither path materializes a
    Kronecker matrix.
    """
    matrix_array, source_array, dimension = _matrix_inputs(
        matrix, source, max_dimension=max_dimension
    )
    if method == "schur":
        return _continuous_schur_solution(matrix_array, source_array)
    if method != "cayley":
        raise ValueError("method must be 'schur' or 'cayley'.")
    iteration_count = int(max_iterations)
    if iteration_count <= 0:
        raise ValueError("max_iterations must be positive.")
    if cayley_shift is None:
        shift = jnp.maximum(
            jnp.asarray(1.0, dtype=matrix_array.real.dtype),
            jnp.linalg.norm(matrix_array, ord=1),
        )
    else:
        shift = jnp.asarray(cayley_shift, dtype=matrix_array.real.dtype).reshape(())
        shift = eqx.error_if(
            shift,
            (~jnp.isfinite(shift)) | (shift <= 0.0),
            "cayley_shift must be finite and positive.",
        )
    identity = jnp.eye(dimension, dtype=matrix_array.dtype)
    denominator = shift * identity - matrix_array
    prepared = prepare(
        LinearSystem(DenseLinearOperator(denominator)),
        LinearSolvePolicy(DenseLU()),
    )
    transformed = solve(
        prepared,
        jnp.concatenate(
            (shift * identity + matrix_array, 2.0 * shift * source_array),
            axis=1,
        ),
    ).value
    discrete_matrix = transformed[:, :dimension]
    left_source = transformed[:, dimension:]
    discrete_source = jnp.conj(solve(prepared, jnp.conj(left_source.T)).value.T)
    return _discrete_doubling_solution(
        discrete_matrix,
        discrete_source,
        max_iterations=iteration_count,
    )


def discrete_lyapunov_solution(
    matrix: ArrayLike,
    source: ArrayLike,
    /,
    *,
    method: Literal["schur", "doubling"] = "schur",
    max_dimension: int = 128,
    max_iterations: int = 32,
) -> Array:
    r"""Return ``X`` solving ``X - A X Aᴴ = Q``.

    ``method="schur"`` uses a bilinear transformation and a JAX-native
    Bartels--Stewart solve with an analytic JVP. ``method="doubling"`` evaluates
    the stable series by a fixed-count quadratic doubling recurrence. No
    regularization or fallback is applied.
    """
    matrix_array, source_array, _ = _matrix_inputs(
        matrix, source, max_dimension=max_dimension
    )
    if method == "schur":
        return _discrete_schur_solution(matrix_array, source_array)
    if method != "doubling":
        raise ValueError("method must be 'schur' or 'doubling'.")
    iteration_count = int(max_iterations)
    if iteration_count <= 0:
        raise ValueError("max_iterations must be positive.")
    return _discrete_doubling_solution(
        matrix_array,
        source_array,
        max_iterations=iteration_count,
    )


def _spectral_diagnostics(
    matrix: Array,
    /,
    *,
    system_type: LyapunovSystemType,
    stability_tolerance: float,
) -> tuple[Array, Array, Array]:
    eigenvalues = jnp.linalg.eigvals(matrix)
    if system_type == "continuous":
        spectral_abscissa = jnp.max(jnp.real(eigenvalues))
        stable = spectral_abscissa < -stability_tolerance
        separation_values = jnp.abs(eigenvalues[:, None] + jnp.conj(eigenvalues[None, :]))
    else:
        spectral_radius = jnp.max(jnp.abs(eigenvalues))
        stable = spectral_radius < 1.0 - stability_tolerance
        separation_values = jnp.abs(
            1.0 - eigenvalues[:, None] * jnp.conj(eigenvalues[None, :])
        )
    separation = jnp.min(separation_values)
    largest = jnp.max(separation_values)
    condition = jnp.where(separation > 0.0, largest / separation, jnp.inf)
    return stable, separation, condition


def _matrix_equation_diagnostics(
    matrix: Array,
    source: Array,
    solution: Array,
    /,
    *,
    system_type: LyapunovSystemType,
    method: str,
    tolerance: float,
    stability_tolerance: float,
    separation_tolerance: float,
    iterations: int,
    require_unique: bool = True,
) -> LinearMatrixEquationDiagnostics:
    adjoint = jnp.conj(matrix.T)
    if system_type == "continuous":
        left = matrix @ solution
        right = solution @ adjoint
        residual = left + right + source
    else:
        propagated = matrix @ solution @ adjoint
        residual = solution - propagated - source
        left = solution
        right = propagated
    residual_norm = jnp.linalg.norm(residual)
    scale = jnp.linalg.norm(left) + jnp.linalg.norm(right) + jnp.linalg.norm(source)
    relative_residual = jnp.where(scale > 0.0, residual_norm / scale, residual_norm)
    stable, separation, condition = _spectral_diagnostics(
        matrix,
        system_type=system_type,
        stability_tolerance=stability_tolerance,
    )
    input_finite = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(source))
    finite = input_finite & jnp.all(jnp.isfinite(solution)) & jnp.isfinite(residual_norm)
    singular = require_unique & (separation <= separation_tolerance)
    converged = finite & (~singular) & (relative_residual <= tolerance)
    status = jnp.where(
        converged,
        int(LinearMatrixEquationStatus.CONVERGED),
        int(LinearMatrixEquationStatus.RESIDUAL_TOLERANCE_NOT_MET),
    )
    status = jnp.where(
        singular,
        int(LinearMatrixEquationStatus.SINGULAR_EQUATION),
        status,
    )
    status = jnp.where(
        finite,
        status,
        int(LinearMatrixEquationStatus.NONFINITE),
    )
    status = jnp.where(
        input_finite & singular,
        int(LinearMatrixEquationStatus.SINGULAR_EQUATION),
        status,
    )
    return LinearMatrixEquationDiagnostics(
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        condition_number=condition,
        spectral_separation=separation,
        stable=stable,
        finite=finite,
        converged=converged,
        status=jnp.asarray(status, dtype=jnp.int32),
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        method=method,
        system_type=system_type,
    )


def solve_continuous_lyapunov(
    matrix: ArrayLike,
    source: ArrayLike,
    /,
    *,
    method: Literal["schur", "cayley"] = "schur",
    max_dimension: int = 128,
    tolerance: float = 1e-6,
    stability_tolerance: float = 1e-7,
    separation_tolerance: float = 1e-10,
    max_iterations: int = 32,
    cayley_shift: ArrayLike | None = None,
) -> LyapunovResult:
    r"""Solve ``A X + X Aᴴ + Q = 0`` with explicit diagnostics."""
    tolerance_, stability_tolerance_, separation_tolerance_ = _validate_solver_scalars(
        tolerance, stability_tolerance, separation_tolerance
    )
    matrix_array, source_array, _ = _matrix_inputs(
        matrix, source, max_dimension=max_dimension
    )
    solution = continuous_lyapunov_solution(
        matrix_array,
        source_array,
        method=method,
        max_dimension=max_dimension,
        max_iterations=max_iterations,
        cayley_shift=cayley_shift,
    )
    iterations = 1 if method == "schur" else int(max_iterations)
    return LyapunovResult(
        value=solution,
        diagnostics=_matrix_equation_diagnostics(
            matrix_array,
            source_array,
            solution,
            system_type="continuous",
            method="bartels-stewart" if method == "schur" else "cayley-doubling",
            tolerance=tolerance_,
            stability_tolerance=stability_tolerance_,
            separation_tolerance=separation_tolerance_,
            iterations=iterations,
        ),
    )


def solve_discrete_lyapunov(
    matrix: ArrayLike,
    source: ArrayLike,
    /,
    *,
    method: Literal["schur", "doubling"] = "schur",
    max_dimension: int = 128,
    tolerance: float = 1e-6,
    stability_tolerance: float = 1e-7,
    separation_tolerance: float = 1e-10,
    max_iterations: int = 32,
) -> LyapunovResult:
    r"""Solve ``X - A X Aᴴ = Q`` with explicit diagnostics."""
    tolerance_, stability_tolerance_, separation_tolerance_ = _validate_solver_scalars(
        tolerance, stability_tolerance, separation_tolerance
    )
    matrix_array, source_array, _ = _matrix_inputs(
        matrix, source, max_dimension=max_dimension
    )
    solution = discrete_lyapunov_solution(
        matrix_array,
        source_array,
        method=method,
        max_dimension=max_dimension,
        max_iterations=max_iterations,
    )
    iterations = 1 if method == "schur" else int(max_iterations)
    return LyapunovResult(
        value=solution,
        diagnostics=_matrix_equation_diagnostics(
            matrix_array,
            source_array,
            solution,
            system_type="discrete",
            method="bilinear-bartels-stewart" if method == "schur" else "doubling",
            tolerance=tolerance_,
            stability_tolerance=stability_tolerance_,
            separation_tolerance=separation_tolerance_,
            iterations=iterations,
        ),
    )


def finite_continuous_lyapunov(
    matrix: ArrayLike,
    source: ArrayLike,
    horizon: ArrayLike,
    /,
    *,
    max_dimension: int = 128,
    tolerance: float = 1e-6,
    stability_tolerance: float = 1e-7,
) -> LyapunovResult:
    r"""Evaluate ``∫₀ᵀ exp(A t) Q exp(Aᴴ t) dt`` by a block exponential."""
    tolerance_, stability_tolerance_, _ = _validate_solver_scalars(
        tolerance, stability_tolerance, 0.0
    )
    matrix_array, source_array, dimension = _matrix_inputs(
        matrix, source, max_dimension=max_dimension
    )
    duration = jnp.asarray(horizon, dtype=matrix_array.real.dtype).reshape(())
    duration = eqx.error_if(
        duration,
        (~jnp.isfinite(duration)) | (duration < 0.0),
        "horizon must be finite and non-negative.",
    )
    zero = jnp.zeros_like(matrix_array)
    block = jnp.concatenate(
        (
            jnp.concatenate((matrix_array, source_array), axis=1),
            jnp.concatenate((zero, -jnp.conj(matrix_array.T)), axis=1),
        ),
        axis=0,
    )
    exponential = jsp.linalg.expm(duration * block)
    transition = exponential[:dimension, :dimension]
    cross = exponential[:dimension, dimension:]
    terminal_adjoint = jnp.conj(transition.T)
    solution = cross @ terminal_adjoint
    terminal_source = transition @ source_array @ terminal_adjoint
    effective_source = source_array - terminal_source
    return LyapunovResult(
        value=solution,
        diagnostics=_matrix_equation_diagnostics(
            matrix_array,
            effective_source,
            solution,
            system_type="continuous",
            method="block-exponential",
            tolerance=tolerance_,
            stability_tolerance=stability_tolerance_,
            separation_tolerance=0.0,
            iterations=1,
            require_unique=False,
        ),
    )


def finite_discrete_lyapunov(
    matrix: ArrayLike,
    source: ArrayLike,
    steps: int,
    /,
    *,
    max_dimension: int = 128,
    tolerance: float = 1e-6,
    stability_tolerance: float = 1e-7,
) -> LyapunovResult:
    r"""Evaluate ``Σₖ₌₀ᴺ⁻¹ Aᵏ Q (Aᴴ)ᵏ`` without a Kronecker matrix."""
    tolerance_, stability_tolerance_, _ = _validate_solver_scalars(
        tolerance, stability_tolerance, 0.0
    )
    step_count = int(steps)
    if step_count < 0:
        raise ValueError("steps must be non-negative.")
    matrix_array, source_array, dimension = _matrix_inputs(
        matrix, source, max_dimension=max_dimension
    )
    identity = jnp.eye(dimension, dtype=matrix_array.dtype)

    def body(_, carry):
        solution, power = carry
        solution = source_array + matrix_array @ solution @ jnp.conj(matrix_array.T)
        return solution, matrix_array @ power

    solution, terminal_power = jax.lax.fori_loop(
        0,
        step_count,
        body,
        (jnp.zeros_like(source_array), identity),
    )
    terminal_source = terminal_power @ source_array @ jnp.conj(terminal_power.T)
    effective_source = source_array - terminal_source
    return LyapunovResult(
        value=solution,
        diagnostics=_matrix_equation_diagnostics(
            matrix_array,
            effective_source,
            solution,
            system_type="discrete",
            method="finite-sum",
            tolerance=tolerance_,
            stability_tolerance=stability_tolerance_,
            separation_tolerance=0.0,
            iterations=step_count,
            require_unique=False,
        ),
    )


def _left_action(action: Callable[[Array], ArrayLike], matrix: Array, /) -> Array:
    def apply(column):
        result = jnp.asarray(action(column), dtype=matrix.dtype)
        if result.shape != column.shape:
            raise ValueError(
                f"operator action must preserve shape {column.shape}; got {result.shape}."
            )
        return result

    return jax.vmap(apply, in_axes=1, out_axes=1)(matrix)


def _right_adjoint_action(
    action: Callable[[Array], ArrayLike], matrix: Array, /
) -> Array:
    left_on_adjoint = _left_action(action, jnp.conj(matrix.T))
    return jnp.conj(left_on_adjoint.T)


def _krylov_lyapunov(
    operator: Callable[[Array], ArrayLike],
    source: ArrayLike,
    /,
    *,
    system_type: LyapunovSystemType,
    tolerance: float,
    absolute_tolerance: float,
    restart: int,
    max_steps: int,
) -> LyapunovResult:
    if not callable(operator):
        raise TypeError("operator must be callable.")
    source_array = _inexact(source)
    if source_array.ndim != 2 or source_array.shape[0] != source_array.shape[1]:
        raise ValueError(f"source must be square; got shape {source_array.shape}.")
    relative_tolerance = float(tolerance)
    absolute_tolerance_ = float(absolute_tolerance)
    if relative_tolerance < 0.0 or absolute_tolerance_ < 0.0:
        raise ValueError("Krylov tolerances must be non-negative.")
    restart_count = int(restart)
    step_count = int(max_steps)
    if restart_count <= 0 or step_count <= 0:
        raise ValueError("restart and max_steps must be positive.")

    def equation(value):
        left = _left_action(operator, value)
        if system_type == "continuous":
            return left + _right_adjoint_action(operator, value)
        return value - _right_adjoint_action(operator, left)

    right_hand_side = -source_array if system_type == "continuous" else source_array
    space = ArraySpace(source_array.shape, dtype=source_array.dtype)
    linear_result = solve(
        LinearSystem(
            FunctionLinearOperator(
                equation,
                source=space,
                target=space,
                operator_id=f"{system_type}-lyapunov-operator",
            )
        ),
        right_hand_side,
        policy=LinearSolvePolicy(
            GMRES(
                restart=restart_count,
                stagnation_iterations=restart_count,
            ),
            tolerance=TolerancePolicy(
                relative=relative_tolerance,
                absolute=absolute_tolerance_,
                max_steps=step_count,
            ),
        ),
    )
    solution = linear_result.value
    residual = equation(solution) - right_hand_side
    residual_norm = jnp.linalg.norm(residual)
    scale = jnp.linalg.norm(right_hand_side)
    relative_residual = jnp.where(scale > 0.0, residual_norm / scale, residual_norm)
    finite = jnp.all(jnp.isfinite(solution)) & jnp.isfinite(residual_norm)
    threshold = jnp.maximum(absolute_tolerance_, relative_tolerance * scale)
    converged = finite & (residual_norm <= threshold)
    status = jnp.where(
        converged,
        int(LinearMatrixEquationStatus.CONVERGED),
        int(LinearMatrixEquationStatus.RESIDUAL_TOLERANCE_NOT_MET),
    )
    status = jnp.where(finite, status, int(LinearMatrixEquationStatus.NONFINITE))
    diagnostics = LinearMatrixEquationDiagnostics(
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        condition_number=jnp.asarray(jnp.nan, dtype=source_array.real.dtype),
        spectral_separation=jnp.asarray(jnp.nan, dtype=source_array.real.dtype),
        stable=jnp.asarray(False),
        finite=finite,
        converged=converged,
        status=jnp.asarray(status, dtype=jnp.int32),
        iterations=linear_result.diagnostics.iterations,
        method="gmres-operator-action",
        system_type=system_type,
    )
    return LyapunovResult(value=solution, diagnostics=diagnostics)


def solve_continuous_lyapunov_krylov(
    operator: Callable[[Array], ArrayLike],
    source: ArrayLike,
    /,
    *,
    tolerance: float = 1e-6,
    absolute_tolerance: float = 0.0,
    restart: int = 20,
    max_steps: int = 20,
) -> LyapunovResult:
    r"""Solve ``A X + X Aᴴ + Q = 0`` from an ``A`` action using GMRES."""
    return _krylov_lyapunov(
        operator,
        source,
        system_type="continuous",
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        restart=restart,
        max_steps=max_steps,
    )


def solve_discrete_lyapunov_krylov(
    operator: Callable[[Array], ArrayLike],
    source: ArrayLike,
    /,
    *,
    tolerance: float = 1e-6,
    absolute_tolerance: float = 0.0,
    restart: int = 20,
    max_steps: int = 20,
) -> LyapunovResult:
    r"""Solve ``X - A X Aᴴ = Q`` from an ``A`` action using GMRES."""
    return _krylov_lyapunov(
        operator,
        source,
        system_type="discrete",
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        restart=restart,
        max_steps=max_steps,
    )


__all__ = [
    "LinearMatrixEquationDiagnostics",
    "LinearMatrixEquationStatus",
    "LyapunovMethod",
    "LyapunovResult",
    "LyapunovSystemType",
    "continuous_lyapunov_solution",
    "discrete_lyapunov_solution",
    "finite_continuous_lyapunov",
    "finite_discrete_lyapunov",
    "linear_matrix_equation_status_message",
    "solve_continuous_lyapunov",
    "solve_continuous_lyapunov_krylov",
    "solve_discrete_lyapunov",
    "solve_discrete_lyapunov_krylov",
]
