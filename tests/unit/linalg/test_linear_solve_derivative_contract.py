from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

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
