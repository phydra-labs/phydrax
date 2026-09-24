import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax import (
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
)


RHS = DerivativeSurface.SOLVER_ARGUMENT
OPERATOR = DerivativeSurface.PHYSICAL_PARAMETER
MATRIX = jnp.asarray([[4.0, 1.0, 0.3], [1.0, 3.0, 0.2], [0.3, 0.2, 5.0]])
VECTOR = jnp.asarray([1.0, 2.0, 3.0])


def _solve(mode, rhs=VECTOR, *, matrix=MATRIX, failing=False, rhs_layout=None):
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


@pytest.mark.parametrize(
    ("mode", "route", "surfaces", "conditions"),
    (
        ("mathematical", DerivativeRoute.IMPLICIT, (RHS, OPERATOR), ("solve-converged",)),
        ("rhs-only", DerivativeRoute.IMPLICIT, (RHS,), ("solve-converged",)),
        ("algorithmic", DerivativeRoute.UNROLLED, (RHS, OPERATOR), ("decisions-frozen",)),
        ("none", DerivativeRoute.STOPPED, (), ()),
    ),
)
def test_linear_solve_reports_the_contract_of_its_differentiation_mode(
    mode, route, surfaces, conditions
):
    result = _solve(mode)
    contract = result.derivative_contract

    assert bool(result.successful)
    assert contract.route is route
    assert contract.supported_surfaces == tuple(
        sorted(surfaces, key=list(DerivativeSurface).index)
    )
    assert all(contract.level(surface) is GradientLevel.SMOOTH for surface in surfaces)
    assert contract.conditions == conditions


def test_rhs_only_solves_stop_operator_derivatives():
    contract = _solve("rhs-only").derivative_contract

    assert contract.admit(DifferentiationRequest({RHS})).supported
    admission = contract.admit(DifferentiationRequest({OPERATOR}))
    assert not admission.supported
    assert admission.level(OPERATOR) is GradientLevel.NONE


def test_failed_mathematical_solve_reports_an_invalid_poisoned_derivative():
    failed = _solve("mathematical", failing=True)
    converged = _solve("mathematical")

    assert not bool(failed.successful)
    assert not bool(failed.derivative_valid)
    assert bool(converged.derivative_valid)
    gradient = jax.grad(
        lambda rhs: jnp.sum(_solve("mathematical", rhs, failing=True).value)
    )(VECTOR)
    assert not bool(jnp.all(jnp.isfinite(gradient)))


def test_derivative_validity_is_reported_per_right_hand_side():
    stacked = jnp.stack([VECTOR, jnp.zeros(3)], axis=-1)
    result = _solve(
        "mathematical",
        stacked,
        failing=True,
        rhs_layout=phx.linalg.RHSLayout((2,)),
    )

    assert result.derivative_valid.tolist() == result.successful.tolist()
    assert result.derivative_valid.tolist() == [False, True]


def test_unrolled_and_stopped_contracts_do_not_depend_on_convergence():
    unrolled = _solve("algorithmic", failing=True)
    stopped = _solve("none")

    assert not bool(unrolled.successful)
    assert bool(unrolled.derivative_valid)
    assert bool(stopped.successful)
    assert not bool(stopped.derivative_valid)
