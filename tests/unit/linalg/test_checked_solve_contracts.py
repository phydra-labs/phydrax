#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax.numpy as jnp
import pytest

import phydrax.linalg as la
from phydrax.linalg import (
    LinearDerivativeSolvePolicy,
    LinearSolveCheckPolicy,
    solve_adjoint_checked,
    solve_checked,
    StabilityLowerBound,
)


def _positive_definite_properties() -> la.OperatorProperties:
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def test_checked_contracts() -> None:
    matrix = jnp.asarray(
        [[3.0 + 0.5j, -1.0j], [2.0, 4.0 - 0.25j]],
        dtype=jnp.complex128,
    )
    operator = la.DenseLinearOperator(matrix, operator_id="checked-complex-system")
    problem = la.LinearSystem(operator)
    stability = StabilityLowerBound(operator, 0.1, evidence="verified")
    solve_policy = la.LinearSolvePolicy(la.DenseLU())
    rhs = jnp.asarray([1.0 - 2.0j, 0.5 + 1.0j])

    primal, primal_evidence = solve_checked(
        problem,
        rhs,
        policy=solve_policy,
        check_policy=LinearSolveCheckPolicy(stability_lower_bound=stability),
    )
    adjoint_rhs = jnp.asarray([-0.5 + 0.25j, 2.0 - 1.0j])
    derivative, derivative_evidence = solve_adjoint_checked(
        problem,
        adjoint_rhs,
        primal_evidence=primal_evidence,
        policy=solve_policy,
        check_policy=LinearDerivativeSolvePolicy(stability_lower_bound=stability),
    )

    assert bool(primal.successful)
    assert bool(primal_evidence.valid)
    assert primal_evidence.stability_checked
    assert jnp.allclose(primal.value, jnp.linalg.solve(matrix, rhs))
    assert jnp.allclose(primal_evidence.true_residual_norm, 0.0, atol=1e-12)
    assert bool(derivative.successful)
    assert bool(derivative_evidence.valid)
    assert derivative_evidence.kind == "adjoint"
    assert jnp.allclose(
        derivative.value,
        jnp.linalg.solve(jnp.conj(matrix.T), adjoint_rhs),
    )
    assert jnp.allclose(derivative_evidence.true_residual_norm, 0.0, atol=1e-12)
    space = la.ArraySpace((3,), dtype=jnp.float64)
    matrix = jnp.asarray([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id="checked-nullspace-system",
    )
    kernel = la.LinearSubspace(space, jnp.ones((3, 1)))
    certificate = la.KernelCertificate(
        operator,
        kernel,
        complete=True,
        evidence="verified",
    )
    problem = la.LinearSystem(
        operator,
        nullspace_policy=la.NullspacePolicy(certificate=certificate),
    )

    result, evidence = solve_checked(
        problem,
        jnp.asarray([1.0, 0.0, -1.0]),
        check_policy=LinearSolveCheckPolicy(require_nullspace=True),
    )

    assert bool(result.successful)
    assert evidence.nullspace_checked
    assert bool(evidence.nullspace_ok)
    assert bool(evidence.valid)
    assert jnp.allclose(evidence.compatibility_residual, 0.0, atol=1e-12)
    assert jnp.allclose(evidence.gauge_residual, 0.0, atol=1e-12)
    matrix = jnp.diag(jnp.asarray([1.0, 4.0]))
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_definite_properties(),
        operator_id="checked-forward-error",
    )
    rhs = jnp.ones((2,))
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(
            relative=0.0,
            absolute=0.0,
            max_steps=1,
        ),
        differentiation=la.DifferentiationPolicy("none"),
    )
    result, evidence = solve_checked(
        la.LinearSystem(operator),
        rhs,
        policy=policy,
        check_policy=LinearSolveCheckPolicy(
            stability_lower_bound=StabilityLowerBound(
                operator,
                1.0,
                evidence="verified",
            )
        ),
    )
    exact = jnp.linalg.solve(matrix, rhs)
    actual_error = jnp.linalg.norm(result.value - exact)

    assert evidence.forward_error_bound_available
    assert evidence.forward_error_bound_certified
    assert jnp.allclose(
        evidence.forward_error_upper_bound,
        evidence.true_residual_norm / evidence.stability_lower_bound,
    )
    assert actual_error <= evidence.forward_error_upper_bound + 1.0e-15

    _, asserted = solve_checked(
        la.LinearSystem(operator),
        rhs,
        policy=la.LinearSolvePolicy(la.DenseLU()),
        check_policy=LinearSolveCheckPolicy(
            stability_lower_bound=StabilityLowerBound(
                operator,
                1.0,
                evidence="asserted",
            )
        ),
    )
    assert asserted.forward_error_bound_available
    assert not asserted.forward_error_bound_certified


def test_checked_solve_contracts_scenario_1() -> None:
    diagonal = jnp.diag(jnp.asarray([1.0, 2.0, 4.0, 8.0]))
    space = la.ArraySpace((4,), dtype=diagonal.dtype)
    iterative_operator = la.FunctionLinearOperator(
        lambda vector: diagonal @ vector,
        source=space,
        target=space,
        properties=_positive_definite_properties(),
        operator_id="checked-limited-pcg",
    )
    prepared = la.prepare(
        la.LinearSystem(iterative_operator),
        la.LinearSolvePolicy(
            la.PCG(),
            tolerance=la.TolerancePolicy(
                relative=1e-5,
                absolute=1e-7,
                max_steps=4,
            ),
            differentiation=la.DifferentiationPolicy("none"),
        ),
    )
    limited, limited_evidence = solve_checked(
        prepared,
        jnp.ones((4,)),
        control=la.LinearSolveControl(
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
            maximum_steps=1,
        ),
    )

    assert limited.status == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    assert not bool(limited_evidence.status_ok)
    assert not bool(limited_evidence.converged)
    assert not bool(limited_evidence.valid)

    dense_operator = la.DenseLinearOperator(jnp.eye(2), operator_id="checked-finite")
    nonfinite, nonfinite_evidence = solve_checked(
        la.LinearSystem(dense_operator),
        jnp.asarray([jnp.inf, 1.0]),
        policy=la.LinearSolvePolicy(la.DenseLU()),
    )

    assert nonfinite.status == int(la.LinearSolveStatus.NONFINITE_INPUT)
    assert not bool(nonfinite_evidence.finite)
    assert not bool(nonfinite_evidence.valid)
    matrix = jnp.asarray([[2.0, 0.5], [-1.0, 3.0]])
    operator = la.DenseLinearOperator(matrix, operator_id="primal-evidence-system")
    problem = la.LinearSystem(operator)
    unrelated = la.DenseLinearOperator(jnp.eye(2), operator_id="unrelated-system")
    mismatched_bound = StabilityLowerBound(unrelated, 1.0, evidence="verified")
    solve_policy = la.LinearSolvePolicy(la.DenseLU())

    primal, primal_evidence = solve_checked(
        problem,
        jnp.asarray([1.0, -2.0]),
        policy=solve_policy,
        check_policy=LinearSolveCheckPolicy(stability_lower_bound=mismatched_bound),
    )
    adjoint, adjoint_evidence = solve_adjoint_checked(
        problem,
        jnp.asarray([0.25, 1.5]),
        primal_evidence=primal_evidence,
        policy=solve_policy,
    )

    assert bool(primal.successful)
    assert not bool(primal_evidence.stability_ok)
    assert not bool(primal_evidence.valid)
    assert bool(adjoint.successful)
    assert not bool(adjoint_evidence.primal_valid)
    assert not bool(adjoint_evidence.valid)
    assert jnp.allclose(
        adjoint.value,
        jnp.linalg.solve(matrix.T, jnp.asarray([0.25, 1.5])),
    )
    operator = la.DiagonalLinearOperator(
        jnp.asarray([1.0, 2.0]),
        operator_id="spectral-interval-identity",
    )
    first = la.SpectralInterval(operator, 0.5, 2.5, evidence="verified")
    second = la.SpectralInterval(operator, 0.75, 2.5, evidence="verified")

    assert first.structure_id == second.structure_id
    assert first.certificate_id != second.certificate_id


def test_checked_solve_contracts_scenario_2() -> None:
    indefinite = la.hermitian_sqrt(
        jnp.asarray([[-1.0]]),
        tolerance=1.0e-8,
    )
    semidefinite = la.hermitian_sqrt(
        jnp.asarray([[0.0, 0.0], [0.0, 4.0]]),
        tolerance=1.0e-8,
    )

    assert not bool(indefinite.valid)
    assert bool(semidefinite.valid)
    assert jnp.allclose(
        semidefinite.value @ semidefinite.value,
        semidefinite.spectrum.reconstruct(),
    )
    with pytest.raises(TypeError, match="operands must be real"):
        la.contract_block_scaled(
            "i,i->",
            jnp.asarray([1.0 + 2.0j]),
            jnp.asarray([3.0]),
        )
