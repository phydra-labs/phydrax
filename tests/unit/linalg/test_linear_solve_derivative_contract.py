from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

import phydrax as phx
from phydrax import (
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
)
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator
from phydrax.sparse._linear import _SparseStoragePlan


type DifferentiationMode = Literal["mathematical", "rhs-only", "algorithmic", "none"]

RHS = DerivativeSurface.SOLVER_ARGUMENT
OPERATOR = DerivativeSurface.PHYSICAL_PARAMETER
MATRIX = jnp.asarray([[4.0, 1.0, 0.3], [1.0, 3.0, 0.2], [0.3, 0.2, 5.0]])
VECTOR = jnp.asarray([1.0, 2.0, 3.0])


@dataclass(frozen=True, slots=True)
class DerivativeCase:
    case_id: str
    mode: DifferentiationMode
    route: DerivativeRoute
    surfaces: tuple[DerivativeSurface, ...]
    conditions: tuple[str, ...]


CASES = (
    DerivativeCase(
        "mathematical",
        "mathematical",
        DerivativeRoute.IMPLICIT,
        (RHS, OPERATOR),
        ("solve-converged",),
    ),
    DerivativeCase(
        "rhs-only",
        "rhs-only",
        DerivativeRoute.IMPLICIT,
        (RHS,),
        ("solve-converged",),
    ),
    DerivativeCase(
        "algorithmic",
        "algorithmic",
        DerivativeRoute.UNROLLED,
        (RHS, OPERATOR),
        ("decisions-frozen",),
    ),
    DerivativeCase("none", "none", DerivativeRoute.STOPPED, (), ()),
)


def _solve(
    mode: DifferentiationMode,
    rhs: jax.Array = VECTOR,
    *,
    matrix: jax.Array = MATRIX,
    failing: bool = False,
    rhs_layout: phx.linalg.RHSLayout | None = None,
) -> phx.linalg.LinearSolveResult:
    method = phx.linalg.FGMRES(restart=1) if failing else None
    tolerance = (
        phx.linalg.TolerancePolicy(relative=1e-14, absolute=0.0, max_steps=1)
        if failing
        else None
    )
    policy = phx.linalg.LinearSolvePolicy(
        method,
        tolerance=tolerance,
        differentiation=phx.linalg.DifferentiationPolicy(mode),
    )
    problem = phx.linalg.LinearSystem(phx.linalg.DenseLinearOperator(matrix))
    return phx.linalg.solve(problem, rhs, policy=policy, rhs_layout=rhs_layout)


def test_linear_solve_derivative_contract_scenario_1() -> None:
    for case in CASES:
        result = _solve(case.mode)
        contract = result.derivative_contract
        assert bool(result.successful), case.case_id
        assert contract.route is case.route, case.case_id
        assert contract.supported_surfaces == tuple(
            sorted(case.surfaces, key=list(DerivativeSurface).index)
        ), case.case_id
        assert all(
            contract.level(surface) is GradientLevel.SMOOTH for surface in case.surfaces
        ), case.case_id
        assert contract.conditions == case.conditions, case.case_id
    rhs_only = _solve("rhs-only").derivative_contract
    assert rhs_only.admit(DifferentiationRequest({RHS})).supported
    operator_admission = rhs_only.admit(DifferentiationRequest({OPERATOR}))
    assert not operator_admission.supported
    assert operator_admission.level(OPERATOR) is GradientLevel.NONE

    unrolled = _solve("algorithmic", failing=True)
    stopped = _solve("none")
    assert not bool(unrolled.successful)
    assert bool(unrolled.derivative_valid)
    assert bool(stopped.successful)
    assert not bool(stopped.derivative_valid)
    failed = _solve("mathematical", failing=True)
    converged = _solve("mathematical")
    assert not bool(failed.successful)
    assert not bool(failed.derivative_valid)
    assert bool(converged.derivative_valid)

    gradient = jax.grad(
        lambda rhs: jnp.sum(_solve("mathematical", rhs, failing=True).value)
    )(VECTOR)
    assert not bool(jnp.all(jnp.isfinite(gradient)))

    stacked = jnp.stack([VECTOR, jnp.zeros(3)], axis=-1)
    batched = _solve(
        "mathematical",
        stacked,
        failing=True,
        rhs_layout=phx.linalg.RHSLayout((2,)),
    )
    assert batched.derivative_valid.tolist() == batched.successful.tolist()
    assert batched.derivative_valid.tolist() == [False, True]


def _saddle_point_operator(scale: jax.Array) -> jax.Array:
    """Indefinite ``[[scale A, B^T], [B, 0]]`` larger than a GMRES restart cycle."""
    rng = np.random.default_rng(7)
    primal, dual = 60, 30
    factor = rng.standard_normal((primal, primal))
    spd = factor @ factor.T / primal + np.eye(primal)
    coupling = rng.standard_normal((dual, primal))
    upper = jnp.concatenate((scale * jnp.asarray(spd), jnp.asarray(coupling.T)), axis=1)
    lower = jnp.concatenate(
        (jnp.asarray(coupling), jnp.zeros((dual, dual), dtype=jnp.float64)), axis=1
    )
    return jnp.concatenate((upper, lower), axis=0)


def _spd_operator(scale: jax.Array) -> jax.Array:
    rng = np.random.default_rng(11)
    factor = rng.standard_normal((90, 90))
    return scale * jnp.asarray(factor @ factor.T / 90.0) + jnp.eye(90)


_CERTIFIED_SPD = phx.linalg.OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    evidence={
        "self_adjoint": "construction",
        "positive_definite": "construction",
        "positive_semidefinite": "construction",
    },
)


@pytest.mark.parametrize(
    ("method", "operator", "properties"),
    (
        (phx.linalg.DenseLU(), _saddle_point_operator, None),
        (phx.linalg.DenseCholesky(), _spd_operator, _CERTIFIED_SPD),
    ),
    ids=("lu-saddle-point", "cholesky-spd"),
)
def test_primal_factor_route_derivatives_are_exact_factored_solves(
    method: phx.linalg.AbstractLinearMethod,
    operator: Callable[[jax.Array], jax.Array],
    properties: phx.linalg.OperatorProperties | None,
) -> None:
    # Host reference: x(s) = K(s)^{-1} b, dx/ds = -K^{-1} (dK/ds) x.
    rhs = jnp.cos(jnp.arange(90, dtype=jnp.float64))
    scale = jnp.asarray(1.7)
    policy = phx.linalg.LinearSolvePolicy(
        method,
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(route="primal-factors"),
    )

    def state(value: jax.Array) -> jax.Array:
        system = phx.linalg.LinearSystem(
            phx.linalg.DenseLinearOperator(operator(value), properties=properties)
        )
        return phx.linalg.solve(system, rhs, policy=policy).value

    matrix = np.asarray(operator(scale))
    derivative = np.asarray(jax.jacfwd(operator)(scale))
    solution = np.linalg.solve(matrix, np.asarray(rhs))
    reference = -np.linalg.solve(matrix, derivative @ solution)
    cotangent = np.sin(np.arange(90, dtype=np.float64))
    _, tangent = jax.jvp(state, (scale,), (jnp.asarray(1.0),))
    gradient = jax.grad(lambda value: jnp.dot(jnp.asarray(cotangent), state(value)))(
        scale
    )

    np.testing.assert_allclose(tangent, reference, rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(gradient, cotangent @ reference, rtol=1.0e-9)


def test_nonsymmetric_sparse_factor_preconditioned_cotangents_match_dense() -> None:
    la = phx.linalg
    size = 128
    diagonal = np.arange(size, dtype=np.int32)
    sources = np.concatenate((diagonal, diagonal[:-1], diagonal[1:]))
    targets = np.concatenate((diagonal, diagonal[1:], diagonal[:-1]))
    relation = EdgeRelation(sources, targets, source_size=size, target_size=size)
    storage_plan = _SparseStoragePlan(relation)
    space = la.ArraySpace((size,), dtype=jnp.float64)
    # Upwind advection-diffusion: a nonsymmetric, weakly dominant M-matrix.
    coefficients = jnp.asarray(
        np.concatenate(
            (np.full(size, 2.001), np.full(size - 1, -1.8), np.full(size - 1, -0.2))
        )
    )
    rhs = jnp.cos(jnp.arange(size, dtype=jnp.float64) / 10.0)
    weights = jnp.sin(jnp.arange(size, dtype=jnp.float64) / 13.0)
    policy = la.LinearSolvePolicy(
        la.GMRES(restart=4),
        tolerance=la.TolerancePolicy(relative=1e-11, absolute=1e-12, max_steps=8),
        differentiation=la.DifferentiationPolicy("mathematical"),
        derivative_solve=la.LinearDerivativeSolvePolicy(
            relative_tolerance=1e-11, absolute_tolerance=1e-12, maximum_steps=8
        ),
        preconditioning=la.PreconditioningPolicy(
            la.SparseFactorizationPreconditionerBuilder(
                la.SparseFactorizationPolicy("lu", ordering="approximate-minimum-degree")
            )
        ),
    )

    def observable(values: jax.Array, right: jax.Array) -> jax.Array:
        operator = SparseCoordinateOperator(
            relation,
            values,
            source=space,
            target=space,
            operator_id="upwind-advection-diffusion",
            storage_plan=storage_plan,
        )
        result = la.solve(la.LinearSystem(operator), right, policy=policy)
        return jnp.vdot(weights, result.value)

    value, (coefficient_gradient, rhs_gradient) = jax.jit(
        jax.value_and_grad(observable, argnums=(0, 1))
    )(coefficients, rhs)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[targets, sources] = np.asarray(coefficients)
    solution = np.linalg.solve(matrix, np.asarray(rhs))
    multiplier = np.linalg.solve(matrix.T, np.asarray(weights))
    expected_coefficients = -multiplier[targets] * solution[sources]

    assert np.all(np.isfinite(coefficient_gradient))
    assert np.all(np.isfinite(rhs_gradient))
    np.testing.assert_allclose(value, np.asarray(weights) @ solution, rtol=1e-9)
    np.testing.assert_allclose(
        coefficient_gradient, expected_coefficients, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(rhs_gradient, multiplier, rtol=1e-9, atol=1e-10)


def _well_conditioned_system() -> phx.linalg.LinearSystem:
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((20, 20)) + 20.0 * np.eye(20)
    return phx.linalg.LinearSystem(phx.linalg.DenseLinearOperator(jnp.asarray(matrix)))


def _primal_factor_policy(
    failure: Literal["status", "error"],
    /,
) -> phx.linalg.LinearSolvePolicy:
    # Zero tolerances demand an exactly vanishing derivative residual, which a
    # floating-point factored solve of a dense random operator cannot deliver.
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
            route="primal-factors",
        ),
        failure=phx.linalg.FailurePolicy(failure),
    )


def test_primal_factor_route_status_mode_poisons_only_the_failed_derivative() -> None:
    system = _well_conditioned_system()
    policy = _primal_factor_policy("status")
    rhs = jnp.cos(jnp.arange(20, dtype=jnp.float64))
    direction = jnp.sin(jnp.arange(20, dtype=jnp.float64))

    result = phx.linalg.solve(system, rhs, policy=policy)
    _, tangent = jax.jvp(
        lambda value: phx.linalg.solve(system, value, policy=policy).value,
        (rhs,),
        (direction,),
    )

    assert result.successful
    assert jnp.all(jnp.isfinite(result.value))
    assert jnp.all(jnp.isnan(tangent))


def test_primal_factor_route_error_mode_raises_on_failed_derivative() -> None:
    system = _well_conditioned_system()
    policy = _primal_factor_policy("error")
    rhs = jnp.cos(jnp.arange(20, dtype=jnp.float64))

    with pytest.raises(Exception, match="Implicit linear derivative solve failed"):
        gradient = jax.grad(
            lambda value: jnp.sum(phx.linalg.solve(system, value, policy=policy).value)
        )(rhs)
        jax.block_until_ready(gradient)


_SQUARE = phx.linalg.LinearSystem(phx.linalg.DenseLinearOperator(MATRIX))
_LEAST_SQUARES = phx.linalg.LeastSquaresProblem(phx.linalg.DenseLinearOperator(MATRIX))


@pytest.mark.parametrize(
    ("problem", "method"),
    (
        (_SQUARE, phx.linalg.GMRES()),
        (_LEAST_SQUARES, phx.linalg.DenseQR()),
        (_LEAST_SQUARES, phx.linalg.DenseSVD()),
    ),
    ids=("iterative", "dense-qr", "dense-svd"),
)
def test_primal_factor_route_refuses_plans_without_square_direct_factors(
    problem: phx.linalg.LinearSystem | phx.linalg.LeastSquaresProblem,
    method: phx.linalg.AbstractLinearMethod,
) -> None:
    policy = phx.linalg.LinearSolvePolicy(
        method,
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(route="primal-factors"),
    )

    with pytest.raises(ValueError, match="route='primal-factors'"):
        phx.linalg.plan(problem, policy)


@pytest.mark.parametrize(
    "construct",
    (
        lambda: phx.linalg.LinearDerivativeSolvePolicy(
            route="primal-factors", maximum_steps=4
        ),
        lambda: phx.linalg.LinearDerivativeSolvePolicy(
            route="primal-factors", require_nullspace=True
        ),
    ),
    ids=("maximum-steps", "require-nullspace"),
)
def test_primal_factor_route_refuses_krylov_only_options(
    construct: Callable[[], phx.linalg.LinearDerivativeSolvePolicy],
) -> None:
    with pytest.raises(ValueError, match="route='primal-factors'"):
        construct()


@pytest.mark.parametrize(
    "method",
    [phx.linalg.DenseLU(), phx.linalg.PCG(), phx.linalg.MINRES()],
    ids=["dense-lu", "pcg", "minres"],
)
def test_complex_native_dense_solve_has_strict_dtype_jit_parameter_derivative(
    method: phx.linalg.AbstractLinearMethod,
) -> None:
    """Complex implicit-root Krylov tangents retain real norm evidence."""
    space = phx.linalg.ArraySpace(
        (5,), dtype=jnp.complex128, space_id="complex-native:space"
    )
    rhs = jnp.asarray([1 + 2j, -2 + 1j, 3 - 4j, 2j, 5 + 1j], dtype=jnp.complex128)

    def mapped(diagonal: jax.Array) -> jax.Array:
        operator = phx.linalg.DenseLinearOperator(
            jnp.diag(diagonal.astype(jnp.complex128)),
            source=space,
            target=space,
            operator_id="complex-native:diagonal",
            properties=phx.linalg.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
        )
        result = phx.linalg.solve(
            phx.linalg.LinearSystem(operator, problem_id="complex-native:problem"),
            rhs,
            policy=phx.linalg.LinearSolvePolicy(method),
        )
        return space.flatten(result.value)

    diagonal = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    with jax.numpy_dtype_promotion("strict"):
        np.testing.assert_allclose(
            jax.jit(mapped)(diagonal),
            rhs / diagonal.astype(jnp.complex128),
            atol=1e-11,
        )
        expected = jnp.diag(-rhs / (diagonal**2).astype(jnp.complex128))
        np.testing.assert_allclose(
            jax.jit(jax.jacfwd(mapped))(diagonal), expected, atol=1e-10
        )


def test_complex_multi_rhs_pcg_implicit_parameter_gradient_matches_dense_solve() -> None:
    """A Hermitian, non-real operator needs columnwise complex transpose solves."""
    space = phx.linalg.ArraySpace(
        (3,), dtype=jnp.complex128, space_id="complex-multi-rhs:space"
    )
    base = jnp.asarray(
        [
            [1.2 + 0.3j, -0.4 + 0.7j, 0.2 - 0.5j],
            [0.6 - 0.8j, 1.5 + 0.2j, -0.3 + 0.4j],
            [-0.2 + 0.6j, 0.5 - 0.1j, 1.1 - 0.4j],
        ],
        dtype=jnp.complex128,
    )
    direction = jnp.asarray(
        [
            [0.2 - 0.1j, 0.3 + 0.4j, -0.1 + 0.2j],
            [-0.4 + 0.2j, 0.1 - 0.3j, 0.2 + 0.1j],
            [0.3 + 0.1j, -0.2 + 0.5j, 0.4 - 0.2j],
        ],
        dtype=jnp.complex128,
    )
    rhs = jnp.asarray(
        [
            [1.0 + 0.7j, -0.4 + 1.2j],
            [0.3 - 1.1j, 1.4 + 0.2j],
            [-0.8 + 0.5j, 0.6 - 0.9j],
        ],
        dtype=jnp.complex128,
    )
    probe = jnp.asarray(
        [
            [0.2 + 0.9j, 1.1 - 0.3j],
            [-0.7 + 0.4j, 0.5 + 0.8j],
            [1.3 - 0.6j, -0.2 + 0.7j],
        ],
        dtype=jnp.complex128,
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-12, absolute=1e-14, max_steps=12
        ),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
            relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_steps=12
        ),
    )

    def matrix(theta: jax.Array) -> jax.Array:
        factor = base + theta * direction
        return factor.conj().T @ factor + 0.8 * jnp.eye(3, dtype=jnp.complex128)

    def solve(theta: jax.Array) -> phx.linalg.LinearSolveResult:
        # B^H B + cI with c > 0 certifies Euclidean Hermitian positive definiteness.
        operator = phx.linalg.DenseLinearOperator(
            matrix(theta),
            source=space,
            target=space,
            operator_id="complex-multi-rhs:gram",
            properties=_CERTIFIED_SPD,
        )
        problem = phx.linalg.LinearSystem(
            operator, problem_id="complex-multi-rhs:problem"
        )
        return phx.linalg.solve_many(phx.linalg.prepare(problem, policy), rhs)

    def objective(theta: jax.Array) -> jax.Array:
        return jnp.real(jnp.vdot(probe, solve(theta).value))

    def reference(theta: jax.Array) -> jax.Array:
        return jnp.real(jnp.vdot(probe, jnp.linalg.solve(matrix(theta), rhs)))

    theta = jnp.asarray(0.65, dtype=jnp.float64)
    result = jax.jit(solve)(theta)
    assert result.value.shape == (3, 2)
    assert result.successful.tolist() == [True, True]
    assert result.derivative_valid.tolist() == [True, True]
    assert result.derivative_contract.route is DerivativeRoute.IMPLICIT
    assert result.derivative_contract.level(OPERATOR) is GradientLevel.SMOOTH
    value, gradient = jax.jit(jax.value_and_grad(objective))(theta)
    expected_value, expected_gradient = jax.jit(jax.value_and_grad(reference))(theta)
    np.testing.assert_allclose(value, expected_value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-10, atol=1e-12)


def test_krylov_derivative_solves_reuse_the_prepared_primal_preconditioner() -> None:
    """An ill-conditioned SPD solve meets its derivative contract in one step.

    The 1-D Dirichlet Laplacian has condition number ~6e2 at n=40, so a
    one-step unpreconditioned derivative GMRES cannot meet 1e-10; the exact
    dense-inverse primal accelerator makes the tangent and cotangent one-step
    solves. The independent reference is the dense solve with A and Aᵀ.
    """
    count = 40
    laplacian = 2.0 * np.eye(count) - np.eye(count, k=1) - np.eye(count, k=-1)
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(restart=count),
        tolerance=phx.linalg.TolerancePolicy(relative=1e-12, absolute=0.0),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
            relative_tolerance=1e-10, absolute_tolerance=0.0, maximum_steps=1
        ),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        failure=phx.linalg.FailurePolicy("status"),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.DenseInversePreconditionerBuilder()
        ),
    )

    def state(scale: jax.Array, rhs: jax.Array) -> jax.Array:
        system = phx.linalg.LinearSystem(
            phx.linalg.DenseLinearOperator(scale * jnp.asarray(laplacian))
        )
        return phx.linalg.solve(system, rhs, policy=policy).value

    scale = jnp.asarray(1.5)
    rhs = jnp.cos(jnp.arange(count, dtype=jnp.float64))
    direction = jnp.sin(jnp.arange(count, dtype=jnp.float64))
    cotangent = jnp.linspace(-1.0, 1.0, count)
    matrix = 1.5 * laplacian
    value = np.linalg.solve(matrix, np.asarray(rhs))
    reference_tangent = np.linalg.solve(
        matrix, np.asarray(direction) - 0.5 * laplacian @ value
    )
    adjoint = np.linalg.solve(matrix.T, np.asarray(cotangent))

    _, tangent = jax.jvp(state, (scale, rhs), (jnp.asarray(0.5), direction))
    scale_gradient, rhs_gradient = jax.grad(
        lambda s, b: jnp.dot(cotangent, state(s, b)), argnums=(0, 1)
    )(scale, rhs)

    np.testing.assert_allclose(tangent, reference_tangent, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(rhs_gradient, adjoint, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(scale_gradient, -adjoint @ laplacian @ value, rtol=1e-8)


@pytest.mark.parametrize(
    ("batch_shape", "rhs_count", "zero_rhs"),
    [
        ((), 1, False),
        ((1,), 1, False),
        ((), 3, False),
        ((2,), 1, False),
        ((), 1, True),
    ],
    ids=[
        "single-column",
        "singleton-batch",
        "multiple-columns",
        "multiple-batches",
        "zero-rhs",
    ],
)
def test_native_krylov_mathematical_rhs_shapes_preserve_tangent_and_adjoint(
    batch_shape: tuple[int, ...], rhs_count: int, zero_rhs: bool
) -> None:
    """Batched direct primals retain the native implicit Krylov derivative route."""
    base = np.broadcast_to(
        np.asarray([[3.0, 0.2], [0.2, 2.0]], dtype=np.float64),
        batch_shape + (2, 2),
    )
    shape = batch_shape + (2, rhs_count)
    host_rhs = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 0.5
    if zero_rhs:
        host_rhs = np.zeros(shape, dtype=np.float64)
    rhs = jnp.asarray(host_rhs)
    direction = jnp.sin(jnp.arange(rhs.size, dtype=jnp.float64)).reshape(shape)
    cotangent = jnp.cos(jnp.arange(rhs.size, dtype=jnp.float64)).reshape(shape)
    scale = jnp.asarray(1.2, dtype=jnp.float64)
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        tolerance=phx.linalg.TolerancePolicy(relative=1e-11, absolute=1e-12),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
            route="krylov",
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
            maximum_steps=4,
        ),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
    )

    def result(
        current_scale: jax.Array, current_rhs: jax.Array
    ) -> phx.linalg.LinearSolveResult:
        operator = phx.linalg.DenseLinearOperator(
            current_scale * jnp.asarray(base), operator_id="derivative-rhs-shape-contract"
        )
        return phx.linalg.solve(
            phx.linalg.LinearSystem(operator), current_rhs, policy=policy
        )

    def state(current_scale: jax.Array, current_rhs: jax.Array) -> jax.Array:
        return result(current_scale, current_rhs).value

    primal = result(scale, rhs)
    _, tangent = jax.jvp(
        state, (scale, rhs), (jnp.asarray(0.2, dtype=jnp.float64), direction)
    )
    _, pullback = jax.vjp(state, scale, rhs)
    scale_gradient, rhs_gradient = pullback(cotangent)
    matrix = 1.2 * base
    expected = np.linalg.solve(matrix, host_rhs)
    expected_tangent = np.linalg.solve(
        matrix, np.asarray(direction) - 0.2 * (base @ expected)
    )
    expected_adjoint = np.linalg.solve(np.swapaxes(matrix, -1, -2), np.asarray(cotangent))
    assert bool(jnp.all(primal.successful))
    assert bool(jnp.all(primal.derivative_valid))
    np.testing.assert_allclose(primal.value, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(rhs_gradient, expected_adjoint, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        scale_gradient,
        -np.sum(expected_adjoint * (base @ expected)),
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.strict_jax
def test_pcg_implicit_derivative_preserves_spd_and_independent_work_limit() -> None:
    """SPD derivatives converge without restarted-GMRES stagnation."""
    n = 96
    host = (
        np.diag(np.full(n, 2.0))
        + np.diag(np.full(n - 1, -1.0), 1)
        + np.diag(np.full(n - 1, -1.0), -1)
    )
    matrix = jnp.asarray(host, dtype=jnp.float64)
    rhs = jnp.asarray(np.linspace(0.2, 1.0, n), dtype=jnp.float64)
    cotangent = jnp.asarray(np.linspace(1.0, 0.2, n), dtype=jnp.float64)
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={"self_adjoint": "construction", "positive_definite": "construction"},
    )
    operator = phx.linalg.DenseLinearOperator(matrix, properties=properties)

    def objective(value: jax.Array, maximum_steps: int) -> jax.Array:
        policy = phx.linalg.LinearSolvePolicy(
            phx.linalg.PCG(),
            derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
                maximum_steps=maximum_steps
            ),
        )
        result = phx.linalg.solve(phx.linalg.LinearSystem(operator), value, policy=policy)
        return jnp.dot(result.value, cotangent)

    gradient = jax.grad(lambda value: objective(value, n))(rhs)
    np.testing.assert_allclose(
        gradient, np.linalg.solve(host, np.asarray(cotangent)), rtol=1e-7, atol=1e-8
    )
    failed = jax.grad(lambda value: objective(value, 1))(rhs)
    assert np.all(np.isnan(failed))


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
@pytest.mark.parametrize("route", ["krylov", "primal-factors", "pcg"])
def test_default_derivative_accuracy_tracks_coordinate_dtype(
    dtype: DTypeLike,
    route: Literal["krylov", "primal-factors", "pcg"],
) -> None:
    la = phx.linalg
    with jax.enable_x64():
        matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=dtype)
        rhs = jnp.asarray([1.0, -2.0], dtype=dtype)
        direction = jnp.eye(2, dtype=dtype) * 0.1
        forcing_direction = jnp.ones(2, dtype=dtype) * 0.2
        policy = la.LinearSolvePolicy(
            la.PCG() if route == "pcg" else la.DenseLU(),
            # Only the independent derivative policy is under test; the primal
            # PCG must not fail a sub-roundoff float32 stopping requirement.
            tolerance=la.TolerancePolicy(relative=1e-6) if route == "pcg" else None,
            derivative_solve=la.LinearDerivativeSolvePolicy(
                route="primal-factors" if route == "primal-factors" else "krylov"
            ),
        )

        def solve(values: jax.Array, forcing: jax.Array) -> la.LinearSolveResult:
            system = la.LinearSystem(
                la.DenseLinearOperator(values, properties=_CERTIFIED_SPD)
            )
            return la.solve(system, forcing, policy=policy)

        def value(values: jax.Array, forcing: jax.Array) -> jax.Array:
            return solve(values, forcing).value

        result = solve(matrix, rhs)
        expected_relative = max(1e-8, 10.0 * float(jnp.finfo(rhs.real.dtype).eps))
        assert result.successful
        assert result.provenance.derivative_relative_tolerance == expected_relative
        assert result.provenance.derivative_absolute_tolerance == 1e-10
        _, tangent = jax.jvp(
            jax.jit(value),
            (matrix, rhs),
            (direction, forcing_direction),
        )
        expected_value = jnp.linalg.solve(matrix, rhs)
        expected_tangent = jnp.linalg.solve(
            matrix, forcing_direction - direction @ expected_value
        )
        _, pullback = jax.vjp(value, matrix, rhs)
        _, reference_pullback = jax.vjp(jnp.linalg.solve, matrix, rhs)
        accuracy = 1e-5 if rhs.real.dtype == jnp.float32 else 1e-10
        np.testing.assert_allclose(
            tangent, expected_tangent, rtol=accuracy, atol=accuracy
        )
        for actual, expected in zip(
            pullback(jnp.ones_like(rhs)),
            reference_pullback(jnp.ones_like(rhs)),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=accuracy, atol=accuracy)
        system = la.LinearSystem(
            la.DenseLinearOperator(matrix, properties=_CERTIFIED_SPD)
        )
        _, primal_evidence = la.solve_checked(
            system,
            rhs,
            policy=policy,
            check_policy=la.LinearSolveCheckPolicy(relative_tolerance=accuracy),
        )
        adjoint_result, adjoint_evidence = la.solve_adjoint_checked(
            system,
            rhs,
            policy=policy,
            primal_evidence=primal_evidence,
        )
        assert adjoint_evidence.valid
        assert (
            adjoint_result.provenance.derivative_relative_tolerance == expected_relative
        )
        assert adjoint_result.provenance.derivative_absolute_tolerance == 1e-10


@pytest.mark.parametrize(
    ("relative", "absolute"),
    [(None, 0.0), (1e-12, None), (0.0, 0.0)],
)
def test_explicit_derivative_tolerances_remain_verbatim(
    relative: float | None,
    absolute: float | None,
) -> None:
    la = phx.linalg
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)
    rhs = jnp.ones(2, dtype=jnp.float32)
    system = la.LinearSystem(la.DenseLinearOperator(matrix))
    requested = la.LinearDerivativeSolvePolicy(
        relative_tolerance=relative, absolute_tolerance=absolute
    )
    result = la.solve(
        system,
        rhs,
        policy=la.LinearSolvePolicy(la.DenseLU(), derivative_solve=requested),
    )
    assert requested.relative_tolerance == relative
    assert requested.absolute_tolerance == absolute
    assert result.provenance.derivative_relative_tolerance == (
        10.0 * float(jnp.finfo(jnp.float32).eps) if relative is None else relative
    )
    assert result.provenance.derivative_absolute_tolerance == (
        1e-10 if absolute is None else absolute
    )
    if relative is not None:
        policy = la.LinearSolvePolicy(
            la.DenseLU(),
            derivative_solve=requested,
            failure=la.FailurePolicy("error"),
        )
        with pytest.raises(Exception, match="independent residual contract"):
            derivative = jax.jvp(
                lambda forcing: la.solve(system, forcing, policy=policy).value,
                (rhs,),
                (rhs,),
            )[1]
            jax.block_until_ready(derivative)


def test_dtype_resolved_default_has_distinct_plan_identity() -> None:
    la = phx.linalg
    system = la.LinearSystem(la.DenseLinearOperator(jnp.eye(2, dtype=jnp.float64)))
    automatic = la.plan(system, la.LinearSolvePolicy(la.DenseLU()))
    explicit = la.plan(
        system,
        la.LinearSolvePolicy(
            la.DenseLU(),
            derivative_solve=la.LinearDerivativeSolvePolicy(
                relative_tolerance=1e-8, absolute_tolerance=1e-10
            ),
        ),
    )
    assert automatic.plan_id != explicit.plan_id


def test_pcg_adjoint_does_not_certify_an_unknown_definiteness_claim() -> None:
    la = phx.linalg
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    operator = la.DenseLinearOperator(
        matrix,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"self_adjoint": "construction"},
        ),
    )
    with pytest.raises(ValueError, match="certified positive definiteness"):
        la.solve_adjoint(
            la.LinearSystem(operator),
            jnp.ones(2, dtype=matrix.dtype),
            policy=la.LinearSolvePolicy(la.PCG()),
        )
