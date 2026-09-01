#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax.linalg as la
from phydrax.linalg._certificates import StabilityLowerBound
from phydrax.linalg._policies import (
    LinearDerivativeSolvePolicy,
    LinearSolveCheckPolicy,
)
from phydrax.linalg._runtime import solve_adjoint_checked, solve_checked


def _positive_definite_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def test_block_actions_preserve_every_column_and_transpose_adjoint_duality():
    source_weights = jnp.asarray([2.0, 3.0, 5.0])
    target_weights = jnp.asarray([7.0, 11.0])
    source = la.ArraySpace(
        (3,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(source_weights),
    )
    target = la.ArraySpace(
        (2,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(target_weights),
    )
    matrix = jnp.asarray([[1.0 + 2.0j, -3.0j, 0.5], [2.0, -1.0 + 1.0j, 4.0j]])
    operator = la.DenseLinearOperator(matrix, source=source, target=target)
    source_block = jnp.asarray(
        [
            [1.0 + 0.5j, -2.0j, 3.0],
            [0.25, 1.0 - 1.0j, -4.0j],
            [-2.0, 0.5j, 1.5 + 2.0j],
        ]
    )
    target_block = jnp.asarray([[0.5 - 1.0j, 2.0, -3.0j], [1.5j, -0.25 + 0.5j, 4.0]])
    adjoint_matrix = (
        jnp.reciprocal(source_weights)[:, None]
        * jnp.conj(matrix.T)
        * target_weights[None, :]
    )

    image = operator.mv_block(source_block)
    transpose_image = operator.transpose_mv_block(target_block)
    adjoint_image = operator.adjoint_mv_block(target_block)

    assert operator.supports_fused_block_action
    assert image.shape == (2, 3)
    assert transpose_image.shape == (3, 3)
    assert adjoint_image.shape == (3, 3)
    assert jnp.allclose(image, matrix @ source_block)
    assert jnp.allclose(transpose_image, matrix.T @ target_block)
    assert jnp.allclose(adjoint_image, adjoint_matrix @ target_block)
    assert jnp.allclose(
        jnp.sum(image * target_block, axis=0),
        jnp.sum(source_block * transpose_image, axis=0),
    )
    assert jnp.allclose(
        jnp.sum(jnp.conj(image) * target_weights[:, None] * target_block, axis=0),
        jnp.sum(
            jnp.conj(source_block) * source_weights[:, None] * adjoint_image,
            axis=0,
        ),
    )


def test_default_block_action_is_explicitly_nonfused_and_column_complete():
    matrix = jnp.asarray([[2.0, -1.0], [0.5, 3.0]])
    space = la.ArraySpace((2,), dtype=matrix.dtype)
    operator = la.FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: matrix.T @ vector,
    )
    block = jnp.asarray([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])

    assert not operator.supports_fused_block_action
    assert jnp.allclose(operator.mv_block(block), matrix @ block)
    assert jnp.allclose(operator.transpose_mv_block(block), matrix.T @ block)
    assert jnp.allclose(operator.adjoint_mv_block(block), matrix.T @ block)

    diagonal = la.DiagonalLinearOperator(jnp.asarray([2.0, -3.0]))
    assert diagonal.supports_fused_block_action
    assert jnp.array_equal(
        diagonal.mv_block(block),
        jnp.asarray([2.0, -3.0])[:, None] * block,
    )


def test_checked_primal_and_adjoint_use_true_declared_operator_residuals():
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


def test_checked_nullspace_evidence_uses_compatibility_and_gauge_projections():
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


def test_nonfinite_and_nonconverged_solves_cannot_produce_valid_evidence():
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


def test_invalid_primal_evidence_invalidates_successful_adjoint_evidence():
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
