#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp

import phydrax as phx
from benchmarks._runtime import capture_environment, measure_repeated, synchronize


la = phx.linalg
eig = la.eigen


def _measure(
    operation: Callable[[], Any], /, *, repeats: int
) -> tuple[Any, float, float]:
    result, distribution = measure_repeated(
        operation,
        warmup=1,
        repeats=repeats,
    )
    values = 1_000.0 * jnp.asarray(distribution.samples_seconds)
    return result, float(jnp.mean(values)), float(jnp.std(values))


def _prepare(operation: Callable[[], Any], /) -> tuple[Any, float]:
    started = time.perf_counter()
    result = operation()
    synchronize(result)
    return result, 1e3 * (time.perf_counter() - started)


def _properties() -> la.OperatorProperties:
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
        },
    )


def _base_problem(size: int, key: jax.Array, /) -> tuple[jax.Array, Any, jax.Array]:
    diagonal = jnp.linspace(2.5, 4.0, size, dtype=jnp.float64)
    matrix = (
        jnp.diag(diagonal)
        - 0.35 * jnp.eye(size, k=1, dtype=jnp.float64)
        - 0.35 * jnp.eye(size, k=-1, dtype=jnp.float64)
    )
    operator = la.DenseLinearOperator(
        matrix,
        properties=_properties(),
        operator_id=f"advanced-benchmark-spd-{size}",
    )
    rhs = jr.normal(key, (size,), dtype=jnp.float64)
    return matrix, operator, rhs


def _krylov_reuse_benchmark(
    matrix: jax.Array,
    operator: Any,
    rhs: jax.Array,
    /,
    *,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    size = matrix.shape[0]
    projection_policy = la.KrylovProjectionPolicy(
        "lanczos",
        max_dimension=size,
    )
    projection, preparation_ms = _prepare(
        lambda: la.prepare_krylov_projection(operator, rhs, projection_policy)
    )
    action_policy = la.MatrixFunctionPolicy(
        "lanczos",
        max_dimension=size,
        error_tolerance=1e-11,
    )
    reused_action = jax.jit(
        lambda: la.matrix_exponential_action(
            operator,
            rhs,
            0.1,
            policy=action_policy,
            decomposition=projection,
        )
    )
    fresh_action = jax.jit(
        lambda: la.matrix_exponential_action(
            operator,
            rhs,
            0.1,
            policy=action_policy,
        )
    )
    reused, reused_ms, reused_std = _measure(reused_action, repeats=repeats)
    fresh, fresh_ms, fresh_std = _measure(fresh_action, repeats=repeats)
    reference = jsp.linalg.expm(0.1 * matrix) @ rhs
    error = float(jnp.max(jnp.abs(reused.value - reference)))
    agreement = float(jnp.max(jnp.abs(reused.value - fresh.value)))
    passed = bool(reused.converged) and error < 1e-8 and agreement < 1e-10
    return {
        "dimension": size,
        "preparation_ms": preparation_ms,
        "projection_matvec_count": int(projection.decomposition.matvec_count),
        "reused_action_ms": reused_ms,
        "reused_action_std_ms": reused_std,
        "fresh_action_ms": fresh_ms,
        "fresh_action_std_ms": fresh_std,
        "speedup_excluding_preparation": fresh_ms / max(reused_ms, 1e-12),
        "maximum_absolute_error": error,
        "fresh_reused_difference": agreement,
        "status": int(reused.breakdown_status),
        "passed": passed,
    }, passed


def _shifted_and_rational_benchmark(
    matrix: jax.Array,
    operator: Any,
    rhs: jax.Array,
    /,
    *,
    shift_count: int,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    size = matrix.shape[0]
    shifts = jnp.linspace(5.0, 8.0, shift_count, dtype=jnp.float64)
    family = la.ShiftedLinearSystemFamily(operator, shifts, family_id="advanced-shifts")
    policy = la.ShiftedSolvePolicy(
        "lanczos",
        max_dimension=size,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-12,
    )
    prepared, preparation_ms = _prepare(
        lambda: la.prepare_shifted_solve(family, rhs, policy)
    )
    shifted_action = jax.jit(lambda: la.solve_shifted(prepared))
    shifted, shifted_ms, shifted_std = _measure(shifted_action, repeats=repeats)
    dense_shifted_action = jax.jit(
        lambda: jax.vmap(
            lambda shift: jnp.linalg.solve(shift * jnp.eye(size) - matrix, rhs)
        )(shifts)
    )
    dense_shifted, dense_shifted_ms, dense_shifted_std = _measure(
        dense_shifted_action, repeats=repeats
    )
    shifted_error = float(jnp.max(jnp.abs(shifted.value - dense_shifted)))

    residues = jnp.linspace(0.25, 0.75, shift_count, dtype=jnp.float64)
    rational = la.PartialFractionRationalFunction(
        shifts,
        residues,
        polynomial_coefficients=jnp.asarray([0.1, -0.025], dtype=jnp.float64),
        function_id="advanced-rational",
    )
    rational_policy = la.RationalFunctionPolicy(shifted=policy)
    prepared_rational, rational_preparation_ms = _prepare(
        lambda: la.prepare_rational_function_action(
            operator,
            rhs,
            rational,
            rational_policy,
        )
    )
    rational_action = jax.jit(lambda: la.rational_function_action(prepared_rational))
    rational_result, rational_ms, rational_std = _measure(
        rational_action, repeats=repeats
    )
    dense_rational = (
        0.1 * rhs
        - 0.025 * (matrix @ rhs)
        + jnp.sum(residues[:, None] * dense_shifted, axis=0)
    )
    rational_error = float(jnp.max(jnp.abs(rational_result.value - dense_rational)))
    shifted_success = bool(jnp.all(shifted.status == int(la.ShiftedSolveStatus.SUCCESS)))
    rational_success = int(rational_result.status) == int(
        la.RationalFunctionStatus.SUCCESS
    )
    passed = (
        shifted_success
        and rational_success
        and shifted_error < 1e-8
        and rational_error < 1e-8
    )
    return {
        "dimension": size,
        "shift_count": shift_count,
        "shared_preparation_ms": preparation_ms,
        "shared_solve_ms": shifted_ms,
        "shared_solve_std_ms": shifted_std,
        "independent_dense_solve_ms": dense_shifted_ms,
        "independent_dense_solve_std_ms": dense_shifted_std,
        "shifted_speedup": dense_shifted_ms / max(shifted_ms, 1e-12),
        "shifted_maximum_absolute_error": shifted_error,
        "shifted_statuses": [int(value) for value in shifted.status.tolist()],
        "rational_preparation_ms": rational_preparation_ms,
        "rational_action_ms": rational_ms,
        "rational_action_std_ms": rational_std,
        "rational_maximum_absolute_error": rational_error,
        "rational_status": int(rational_result.status),
        "shared_basis_rank": int(jnp.max(shifted.diagnostics.rank)),
        "shared_basis_matvec_count": int(jnp.max(shifted.diagnostics.setup_matvec_count)),
        "passed": passed,
    }, passed


def _matrix_equation_benchmark(
    size: int,
    /,
    *,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    diagonal = -jnp.linspace(1.0, 2.0, size, dtype=jnp.float64)
    matrix = jnp.diag(diagonal) + 0.2 * jnp.eye(size, k=1, dtype=jnp.float64)
    right_hand_side = jnp.eye(size, dtype=jnp.float64)
    problem = la.continuous_lyapunov_equation(
        matrix,
        right_hand_side,
        problem_id=f"advanced-lyapunov-{size}",
    )
    policy = la.MatrixEquationPolicy(
        linear=la.LinearSolvePolicy(la.DenseLU()),
    )
    prepared, preparation_ms = _prepare(
        lambda: la.prepare_matrix_equation(problem, policy)
    )
    solve_action = jax.jit(lambda: la.solve_matrix_equation(prepared))
    result, solve_ms, solve_std = _measure(solve_action, repeats=repeats)
    dense_action = jax.jit(
        lambda: jsp.linalg.solve_sylvester(
            matrix,
            matrix.T,
            -right_hand_side,
        )
    )
    dense, dense_ms, dense_std = _measure(dense_action, repeats=repeats)
    error = float(jnp.max(jnp.abs(result.value - dense)))
    passed = int(result.status) == int(la.MatrixEquationStatus.SUCCESS) and error < 1e-9
    return {
        "dimension": size,
        "linearized_dimension": size * size,
        "preparation_ms": preparation_ms,
        "prepared_solve_ms": solve_ms,
        "prepared_solve_std_ms": solve_std,
        "dense_sylvester_ms": dense_ms,
        "dense_sylvester_std_ms": dense_std,
        "maximum_absolute_error": error,
        "relative_residual": float(result.diagnostics.relative_residual),
        "self_adjoint_error": float(result.diagnostics.self_adjoint_error),
        "status": int(result.status),
        "passed": passed,
    }, passed


def _spectral_subspace_benchmark(
    size: int,
    key: jax.Array,
    /,
    *,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    selected_dimension = size // 2
    eigenvalues = jnp.concatenate(
        (
            -jnp.linspace(2.0, 0.5, selected_dimension, dtype=jnp.float64),
            jnp.linspace(0.75, 2.25, size - selected_dimension, dtype=jnp.float64),
        )
    )
    nonnormal = 0.15 * jnp.triu(
        jr.normal(key, (size, size), dtype=jnp.float64),
        k=1,
    )
    matrix = jnp.diag(eigenvalues) + nonnormal
    problem = eig.SchurEigenproblem(
        la.DenseLinearOperator(matrix, operator_id=f"advanced-subspace-{size}"),
        problem_id=f"advanced-subspace-{size}",
    )
    selection = eig.SpectralSelection.real_below(
        0.0,
        expected_dimension=selected_dimension,
    )
    prepared, preparation_ms = _prepare(
        lambda: eig.prepare_spectral_subspace(problem, selection)
    )
    perturbation = 0.05 * jr.normal(key, (size, size), dtype=jnp.float64)
    derivative_action = jax.jit(
        lambda: eig.spectral_projector_derivative(prepared, perturbation)
    )
    derivative, derivative_ms, derivative_std = _measure(
        derivative_action, repeats=repeats
    )
    step = 1e-5

    def prepared_projector(value: jax.Array) -> jax.Array:
        shifted_problem = eig.SchurEigenproblem(
            la.DenseLinearOperator(
                value,
                operator_id=f"advanced-subspace-{size}",
            ),
            problem_id=f"advanced-subspace-{size}",
        )
        return eig.prepare_spectral_subspace(shifted_problem, selection).projector

    finite_difference = (
        prepared_projector(matrix + step * perturbation)
        - prepared_projector(matrix - step * perturbation)
    ) / (2.0 * step)
    derivative_error = float(jnp.max(jnp.abs(derivative.value - finite_difference)))
    projector_residual = float(
        jnp.linalg.norm(prepared.projector @ prepared.projector - prepared.projector)
    )
    passed = (
        int(prepared.status) == int(eig.SpectralSubspaceStatus.SUCCESS)
        and int(derivative.status) == int(eig.SpectralProjectorDerivativeStatus.SUCCESS)
        and derivative_error < 1e-7
        and projector_residual < 1e-10
    )
    return {
        "dimension": size,
        "selected_dimension": int(prepared.selected_dimension),
        "preparation_ms": preparation_ms,
        "projector_derivative_ms": derivative_ms,
        "projector_derivative_std_ms": derivative_std,
        "projector_norm": float(prepared.diagnostics.projector_norm),
        "sylvester_separation": float(prepared.diagnostics.sylvester_separation),
        "projector_idempotence_residual": projector_residual,
        "derivative_maximum_absolute_error": derivative_error,
        "derivative_commutator_residual": float(
            derivative.diagnostics.commutator_residual_norm
        ),
        "derivative_tangent_residual": float(
            derivative.diagnostics.tangent_residual_norm
        ),
        "status": int(prepared.status),
        "derivative_status": int(derivative.status),
        "passed": passed,
    }, passed


def _low_rank_benchmark(
    size: int,
    key: jax.Array,
    /,
    *,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    rank = min(4, max(1, size // 4))
    base_matrix = jnp.diag(jnp.linspace(2.0, 4.0, size, dtype=jnp.float64))
    base = la.DenseLinearOperator(
        base_matrix,
        properties=_properties(),
        operator_id=f"advanced-low-rank-base-{size}",
    )
    left_key, right_key, rhs_key = jr.split(key, 3)
    left = 0.1 * jr.normal(left_key, (size, rank), dtype=jnp.float64)
    right = 0.1 * jr.normal(right_key, (size, rank), dtype=jnp.float64)
    core = jnp.diag(jnp.linspace(0.5, 1.0, rank, dtype=jnp.float64))
    operator = la.BasePlusLowRankLinearOperator(
        base,
        left,
        right,
        core,
        operator_id=f"advanced-low-rank-{size}",
    )
    policy = la.LowRankSolvePolicy(
        la.LinearSolvePolicy(la.DenseLU()),
        base_nonsingularity="certified",
    )
    prepared, preparation_ms = _prepare(
        lambda: la.prepare_low_rank_solve(operator, policy)
    )
    rhs = jr.normal(rhs_key, (size,), dtype=jnp.float64)
    solve_action = jax.jit(lambda: la.solve_low_rank(prepared, rhs))
    result, solve_ms, solve_std = _measure(solve_action, repeats=repeats)
    dense_matrix = base_matrix + left @ core @ right.T
    dense_action = jax.jit(lambda: jnp.linalg.solve(dense_matrix, rhs))
    dense, dense_ms, dense_std = _measure(dense_action, repeats=repeats)
    error = float(jnp.max(jnp.abs(result.value - dense)))
    passed = int(result.status) == int(la.LowRankSolveStatus.SUCCESS) and error < 1e-10
    return {
        "dimension": size,
        "rank": rank,
        "preparation_ms": preparation_ms,
        "woodbury_solve_ms": solve_ms,
        "woodbury_solve_std_ms": solve_std,
        "dense_solve_ms": dense_ms,
        "dense_solve_std_ms": dense_std,
        "maximum_absolute_error": error,
        "correction_condition": float(result.diagnostics.correction_condition),
        "status": int(result.status),
        "passed": passed,
    }, passed


def _resilience_benchmark(
    size: int,
    /,
    *,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    scales = jnp.logspace(-6.0, 6.0, size, dtype=jnp.float64)
    matrix = jnp.diag(scales)
    matrix = matrix + 1e-3 * jnp.outer(jnp.sqrt(scales), jnp.sqrt(scales))
    operator = la.DenseLinearOperator(
        matrix,
        properties=_properties(),
        operator_id=f"advanced-resilient-{size}",
    )
    problem = la.LinearSystem(operator, problem_id=f"advanced-resilient-{size}")
    policy = la.ResilientSolvePolicy(
        la.LinearSolvePolicy(la.DenseLU()),
        equilibration=la.EquilibrationPolicy(
            "symmetric-ruiz",
            diagnose_condition=True,
        ),
        refinement=la.RefinementPolicy(max_steps=3),
        failure=la.FailurePolicy("status"),
    )
    prepared, preparation_ms = _prepare(
        lambda: la.prepare_resilient_solve(problem, policy)
    )
    rhs = jnp.linspace(-1.0, 1.0, size, dtype=jnp.float64)
    solve_action = jax.jit(lambda: la.solve_resilient(prepared, rhs))
    result, solve_ms, solve_std = _measure(solve_action, repeats=repeats)
    reference = jnp.linalg.solve(matrix, rhs)
    relative_error = float(
        jnp.linalg.norm(result.value - reference) / jnp.linalg.norm(reference)
    )
    passed = (
        int(result.status) == int(la.ResilientSolveStatus.SUCCESS)
        and relative_error < 1e-9
    )
    return {
        "dimension": size,
        "preparation_ms": preparation_ms,
        "solve_ms": solve_ms,
        "solve_std_ms": solve_std,
        "condition_before": float(prepared.condition_before),
        "condition_after": float(prepared.condition_after),
        "condition_reduction": float(
            prepared.condition_before / prepared.condition_after
        ),
        "refinement_steps": int(result.diagnostics.refinement_steps),
        "relative_residual": float(result.diagnostics.relative_residual),
        "relative_solution_error": relative_error,
        "status": int(result.status),
        "passed": passed,
    }, passed


def _differentiable_spectral_benchmark(
    size: int,
    key: jax.Array,
    /,
    *,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    dimension = max(4, min(size, 12))
    selected_dimension = dimension // 2
    lower = jnp.linspace(1.0, 2.0, selected_dimension, dtype=jnp.float64)
    upper = jnp.linspace(
        4.0,
        6.0,
        dimension - selected_dimension,
        dtype=jnp.float64,
    )
    diagonal = jnp.concatenate((lower, upper))
    matrices = jnp.stack((jnp.diag(diagonal), jnp.diag(diagonal + 0.1)))
    random = jr.normal(key, matrices.shape, dtype=jnp.float64)
    perturbations = 0.05 * (random + jnp.swapaxes(random, -1, -2))
    properties = la.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    )
    problem = eig.Eigenproblem(la.DenseLinearOperator(matrices, properties=properties))
    prepared, preparation_ms = _prepare(
        lambda: eig.prepare_self_adjoint_spectrum(problem)
    )
    selection = eig.SpectralSelection.real_below(
        3.0,
        expected_dimension=selected_dimension,
    )
    subspace = eig.self_adjoint_spectral_subspace(prepared, selection)
    derivative_action = jax.jit(
        lambda: eig.self_adjoint_spectral_projector_derivative(
            prepared,
            selection,
            perturbations,
        )
    )
    derivative, derivative_ms, derivative_std = _measure(
        derivative_action,
        repeats=repeats,
    )

    function_policy = eig.SelfAdjointSpectralOperatorPolicy(differentiation="frechet")

    def squared_operator(current: jax.Array, /) -> jax.Array:
        current_problem = eig.Eigenproblem(
            la.DenseLinearOperator(current, properties=properties)
        )
        return eig.self_adjoint_spectral_operator(
            current_problem,
            eig.PolynomialSpectralFunction(jnp.asarray([0.0, 0.0, 1.0])),
            policy=function_policy,
        ).operator

    spectral_jvp_action = jax.jit(
        lambda: jax.jvp(
            squared_operator,
            (matrices,),
            (perturbations,),
        )
    )
    (squared, squared_tangent), spectral_ms, spectral_std = _measure(
        spectral_jvp_action,
        repeats=repeats,
    )
    expected_squared = matrices @ matrices
    expected_tangent = matrices @ perturbations + perturbations @ matrices
    value_error = float(jnp.max(jnp.abs(squared - expected_squared)))
    tangent_error = float(jnp.max(jnp.abs(squared_tangent - expected_tangent)))

    def coefficient_loss(coefficients: jax.Array, /) -> jax.Array:
        result = eig.self_adjoint_spectral_operator(
            prepared,
            eig.PolynomialSpectralFunction(coefficients),
            policy=function_policy,
        )
        return jnp.sum(jnp.square(jnp.abs(result.operator)))

    coefficients = jnp.asarray([0.2, -0.1, 0.05])
    training_step = jax.jit(jax.value_and_grad(coefficient_loss))
    (loss, gradient), training_ms, training_std = _measure(
        lambda: training_step(coefficients),
        repeats=repeats,
    )
    maximum_derivative_residual = float(jnp.max(derivative.diagnostics.relative_residual))
    passed = bool(
        jnp.all(subspace.successful)
        & jnp.all(derivative.successful)
        & jnp.isfinite(loss)
        & jnp.all(jnp.isfinite(gradient))
        & (maximum_derivative_residual < 1e-10)
        & (value_error < 1e-12)
        & (tangent_error < 1e-10)
    )
    return {
        "batch_size": int(matrices.shape[0]),
        "dimension": dimension,
        "selected_dimension": selected_dimension,
        "preparation_ms": preparation_ms,
        "projector_derivative_ms": derivative_ms,
        "projector_derivative_std_ms": derivative_std,
        "spectral_jvp_ms": spectral_ms,
        "spectral_jvp_std_ms": spectral_std,
        "training_step_ms": training_ms,
        "training_step_std_ms": training_std,
        "maximum_derivative_relative_residual": maximum_derivative_residual,
        "squared_operator_error": value_error,
        "squared_operator_tangent_error": tangent_error,
        "loss": float(loss),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "retained_storage_bytes": int(prepared.plan.cost.retained_bytes),
        "passed": passed,
    }, passed


def _adaptive_spectral_benchmark(
    matrix: jax.Array,
    operator: Any,
    key: jax.Array,
    /,
    *,
    repeats: int,
) -> tuple[dict[str, Any], bool]:
    size = matrix.shape[0]
    policy = la.AdaptiveStochasticPolicy(
        min_probes=4,
        max_probes=12,
        batch_size=2,
        max_dimension=min(size, 16),
        relative_tolerance=0.1,
        absolute_tolerance=1e-8,
    )
    trace_action = jax.jit(
        lambda: la.adaptive_stochastic_trace(operator, key=key, policy=policy)
    )
    estimate, estimate_ms, estimate_std = _measure(trace_action, repeats=repeats)
    exact = jnp.trace(matrix)
    relative_error = float(jnp.abs(estimate.estimate - exact) / jnp.abs(exact))
    passed = bool(estimate.finite) and relative_error < 0.25
    return {
        "dimension": size,
        "estimate_ms": estimate_ms,
        "estimate_std_ms": estimate_std,
        "estimate": float(estimate.estimate),
        "exact_trace": float(exact),
        "relative_error": relative_error,
        "statistical_error": float(estimate.standard_error),
        "projection_error": float(estimate.numerical_error_estimate),
        "num_probes": int(estimate.num_probes),
        "matvec_count": int(estimate.matvec_count),
        "converged": bool(estimate.converged),
        "stopped_early": bool(estimate.stopped_early),
        "passed": passed,
    }, passed


def run_benchmarks(
    *,
    size: int = 32,
    shift_count: int = 8,
    repeats: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Benchmark reusable, structured, spectral, and resilient linear algebra APIs."""
    if size < 4:
        raise ValueError("size must be at least four.")
    if shift_count < 1:
        raise ValueError("shift_count must be positive.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")

    keys = jr.split(jr.key(seed), 6)
    matrix, operator, rhs = _base_problem(size, keys[0])
    matrix_equation_size = max(2, min(10, int(size**0.5) + 1))
    subspace_size = max(4, min(14, size))

    records: dict[str, Any] = {}
    passed: list[bool] = []
    records["krylov_reuse"], status = _krylov_reuse_benchmark(
        matrix, operator, rhs, repeats=repeats
    )
    passed.append(status)
    records["shifted_and_rational"], status = _shifted_and_rational_benchmark(
        matrix,
        operator,
        rhs,
        shift_count=shift_count,
        repeats=repeats,
    )
    passed.append(status)
    records["matrix_equation"], status = _matrix_equation_benchmark(
        matrix_equation_size,
        repeats=repeats,
    )
    passed.append(status)
    records["spectral_subspace"], status = _spectral_subspace_benchmark(
        subspace_size,
        keys[1],
        repeats=repeats,
    )
    passed.append(status)
    records["low_rank"], status = _low_rank_benchmark(
        size,
        keys[2],
        repeats=repeats,
    )
    passed.append(status)
    records["resilience"], status = _resilience_benchmark(
        size,
        repeats=repeats,
    )
    passed.append(status)
    records["adaptive_spectral"], status = _adaptive_spectral_benchmark(
        matrix,
        operator,
        keys[3],
        repeats=repeats,
    )
    passed.append(status)
    records["differentiable_spectral"], status = _differentiable_spectral_benchmark(
        size,
        keys[4],
        repeats=repeats,
    )
    passed.append(status)

    return {
        "configuration": {
            "size": size,
            "shift_count": shift_count,
            "repeats": repeats,
            "seed": seed,
        },
        "environment": capture_environment().to_dict(),
        "benchmarks": records,
        "passed": all(passed),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark advanced Phydrax linear algebra lifecycles."
    )
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--shift-count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small single-repeat correctness and execution smoke benchmark.",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = run_benchmarks(
        size=6 if arguments.smoke else arguments.size,
        shift_count=3 if arguments.smoke else arguments.shift_count,
        repeats=1 if arguments.smoke else arguments.repeats,
        seed=arguments.seed,
    )
    if arguments.smoke and not report["passed"]:
        raise RuntimeError("At least one advanced linear algebra smoke benchmark failed.")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
