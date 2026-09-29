from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import (
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
)


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
