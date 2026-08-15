#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest
import scipy.linalg as spla

import phydrax as phx


la = phx.linalg
eigen = la.eigen


def _self_adjoint_properties(*, positive_definite=False):
    evidence = {"self_adjoint": "construction"}
    if positive_definite:
        evidence.update(
            {
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            }
        )
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=positive_definite,
        evidence=evidence,
    )


def test_lobpcg_standard_eigenpairs_are_jittable_and_resource_checked():
    diagonal = jnp.asarray([1.0, 2.0, 4.0, 8.0])
    operator = la.DiagonalLinearOperator(
        diagonal,
        properties=_self_adjoint_properties(),
    )
    problem = eigen.Eigenproblem(operator)
    policy = eigen.EigenSolvePolicy(
        eigen.LOBPCG(block_dimension=2),
        count=2,
        max_steps=30,
        initial_basis=jnp.asarray(
            [
                [1.0, 0.2],
                [0.3, 1.0],
                [0.2, -0.1],
                [-0.1, 0.3],
            ]
        ),
        tolerance=eigen.EigenTolerancePolicy(
            relative=1e-9,
            absolute=1e-11,
            orthogonality=1e-8,
        ),
    )
    prepared = eigen.prepare_eigensolve(problem, policy)
    result = eigen.eigensolve(prepared)
    compiled_values = jax.jit(lambda: eigen.eigensolve(prepared).eigenvalues)()

    assert bool(result.successful)
    assert jnp.allclose(result.eigenvalues, diagonal[:2], rtol=1e-7, atol=1e-8)
    assert jnp.allclose(compiled_values, diagonal[:2], rtol=1e-7, atol=1e-8)
    assert jnp.all(result.residual_norms[result.mode_mask] < 1e-7)
    assert result.provenance.method == "lobpcg"

    with pytest.raises(ValueError, match="[Kk]rylov|resource"):
        eigen.plan_eigensolve(
            problem,
            eigen.EigenSolvePolicy(
                eigen.LOBPCG(block_dimension=2),
                count=2,
                resources=eigen.EigenResourcePolicy(krylov_basis_bytes=1),
            ),
        )


def test_generalized_eigenproblem_honors_metric_and_constraint_subspace():
    space = la.ArraySpace((3,), dtype=jnp.float64)
    operator = la.DiagonalLinearOperator(
        jnp.asarray([2.0, 6.0, 12.0]),
        space=space,
        properties=_self_adjoint_properties(),
    )
    metric = la.DiagonalLinearOperator(
        jnp.asarray([1.0, 2.0, 3.0]),
        space=space,
        properties=_self_adjoint_properties(positive_definite=True),
    )
    constraints = la.LinearSubspace(
        space,
        jnp.asarray([[1.0], [0.0], [0.0]]),
        orthonormal=True,
    )
    problem = eigen.GeneralizedEigenproblem(
        operator,
        metric,
        constraints=constraints,
    )
    result = eigen.eigensolve(
        problem,
        policy=eigen.EigenSolvePolicy(
            eigen.LOBPCG(block_dimension=2),
            count=2,
            initial_basis=jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        ),
    )
    vectors = jnp.asarray(result.eigenvectors)

    assert bool(result.successful)
    assert jnp.allclose(result.eigenvalues, jnp.asarray([3.0, 4.0]), atol=1e-9)
    assert jnp.allclose(vectors[0], 0.0, atol=1e-9)
    assert jnp.allclose(
        vectors.T @ (metric.diagonal[:, None] * vectors),
        jnp.eye(2),
    )


def test_restarted_lanczos_supports_magnitude_targets_and_refresh():
    diagonal = jnp.asarray([-5.0, 1.0, 2.0, 4.0])
    operator = la.DiagonalLinearOperator(
        diagonal,
        properties=_self_adjoint_properties(),
        operator_id="refreshable-eigen-operator",
    )
    problem = eigen.Eigenproblem(operator, problem_id="refreshable-eigenproblem")
    policy = eigen.EigenSolvePolicy(
        eigen.RestartedLanczos(subspace_dimension=4, restart_dimension=2),
        count=2,
        which="largest-magnitude",
        max_steps=12,
        key=jax.random.key(7),
    )
    prepared = eigen.prepare_eigensolve(problem, policy)
    result = eigen.eigensolve(prepared)

    assert bool(result.successful)
    assert jnp.allclose(jnp.sort(result.eigenvalues), jnp.asarray([-5.0, 4.0]))

    updated_diagonal = jnp.asarray([-6.0, 1.0, 2.0, 4.5])
    updated = eigen.Eigenproblem(
        la.DiagonalLinearOperator(
            updated_diagonal,
            properties=_self_adjoint_properties(),
            operator_id=operator.operator_id,
        ),
        problem_id=problem.problem_id,
    )
    refreshed = eigen.refresh_eigensolve(prepared, updated)
    refreshed_result = eigen.eigensolve(refreshed)

    assert refreshed.numeric_version == prepared.numeric_version + 1
    assert jnp.allclose(
        jnp.sort(refreshed_result.eigenvalues),
        jnp.asarray([-6.0, 4.5]),
        atol=1e-8,
    )


def test_isolated_eigenvalue_gradient_uses_mathematical_derivative():
    properties = _self_adjoint_properties()
    policy = eigen.EigenSolvePolicy(
        eigen.LOBPCG(block_dimension=2),
        count=1,
        initial_basis=jnp.eye(2),
        differentiation="eigenvalues",
    )

    def smallest_eigenvalue(coefficient):
        operator = la.DiagonalLinearOperator(
            jnp.stack((coefficient, jnp.asarray(3.0))),
            properties=properties,
        )
        return eigen.eigensolve(
            eigen.Eigenproblem(operator),
            policy=policy,
        ).eigenvalues[0]

    assert jnp.allclose(jax.grad(smallest_eigenvalue)(1.25), 1.0, atol=1e-8)


def test_lobpcg_repairs_rank_deficient_initial_basis_deterministically():
    operator = la.DiagonalLinearOperator(
        jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        properties=_self_adjoint_properties(),
    )
    repeated = jnp.asarray(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    policy = eigen.EigenSolvePolicy(
        eigen.LOBPCG(block_dimension=2),
        count=2,
        max_steps=40,
        initial_basis=repeated,
        key=jax.random.key(7),
    )

    first = eigen.eigensolve(eigen.Eigenproblem(operator), policy=policy)
    second = eigen.eigensolve(eigen.Eigenproblem(operator), policy=policy)

    assert first.successful
    assert first.diagnostics.initial_rank == 2
    assert jnp.allclose(first.eigenvalues, jnp.asarray([1.0, 2.0]), atol=1e-8)
    assert jnp.allclose(first.eigenvalues, second.eigenvalues, atol=0.0)
    assert jnp.allclose(first.eigenvectors, second.eigenvectors, atol=0.0)


def test_lobpcg_uses_preconditioner_and_reports_partial_convergence():
    properties = _self_adjoint_properties()
    diagonal = jnp.asarray([1.0, 2.0, 4.0, 8.0])
    operator = la.DiagonalLinearOperator(diagonal, properties=properties)
    preconditioned = eigen.eigensolve(
        eigen.Eigenproblem(operator),
        policy=eigen.EigenSolvePolicy(
            eigen.LOBPCG(block_dimension=2),
            count=1,
            max_steps=20,
            initial_basis=jnp.asarray(
                [
                    [0.2, 0.1],
                    [1.0, 0.3],
                    [0.4, 1.0],
                    [0.2, -0.2],
                ]
            ),
            preconditioning=la.PreconditioningPolicy(
                la.DiagonalPreconditioner(1.0 / diagonal)
            ),
        ),
    )

    assert preconditioned.successful
    assert preconditioned.preconditioner_apply_count > 0
    assert jnp.allclose(preconditioned.eigenvalues, jnp.asarray([1.0]), atol=1e-8)

    partial_operator = la.DiagonalLinearOperator(
        jnp.asarray([1.0, 2.0, 4.0, 8.0, 16.0]),
        properties=properties,
    )
    partial = eigen.eigensolve(
        eigen.Eigenproblem(partial_operator),
        policy=eigen.EigenSolvePolicy(
            eigen.LOBPCG(block_dimension=2),
            count=2,
            max_steps=1,
            initial_basis=jnp.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
            tolerance=eigen.EigenTolerancePolicy(
                relative=1e-14,
                absolute=1e-14,
                orthogonality=1e-12,
            ),
        ),
    )

    assert partial.status == int(eigen.EigenSolveStatus.PARTIAL_CONVERGENCE)
    assert jnp.array_equal(partial.converged, jnp.asarray([True, False]))
    assert partial.iterations == 1


def test_dense_eigh_auto_full_spectrum_is_jittable_and_refreshable():
    matrix = jnp.asarray([[2.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 4.0]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_self_adjoint_properties(),
        operator_id="dense-eigh-refreshable",
    )
    problem = eigen.Eigenproblem(operator, problem_id="dense-eigh-problem")
    policy = eigen.EigenSolvePolicy(count=3)
    plan = eigen.plan_eigensolve(problem, policy)
    prepared = eigen.prepare_eigensolve(problem, plan)
    result = eigen.eigensolve(prepared)
    compiled = jax.jit(eigen.eigensolve)(prepared)

    assert isinstance(plan.selected_method, eigen.DenseEigh)
    assert bool(result.successful)
    assert jnp.allclose(result.eigenvalues, jnp.linalg.eigvalsh(matrix))
    assert jnp.allclose(compiled.eigenvalues, result.eigenvalues)
    assert jnp.max(result.residual_norms) < 1e-12
    assert float(result.orthogonality_error) < 1e-12

    changed = la.DenseLinearOperator(
        2.0 * matrix,
        properties=_self_adjoint_properties(),
        operator_id=operator.operator_id,
    )
    refreshed = eigen.refresh_eigensolve(
        prepared,
        eigen.Eigenproblem(changed, problem_id=problem.problem_id),
    )
    refreshed_result = eigen.eigensolve(refreshed)
    assert refreshed.numeric_version == 1
    assert jnp.allclose(refreshed_result.eigenvalues, 2.0 * result.eigenvalues)

    with pytest.raises(ValueError, match="materialization limit"):
        eigen.plan_eigensolve(
            problem,
            eigen.EigenSolvePolicy(
                eigen.DenseEigh(),
                count=3,
                materialization=la.MaterializationPolicy(
                    max_entries=8,
                    max_bytes=1024,
                ),
            ),
        )


def test_dense_generalized_eigh_respects_non_euclidean_pairing():
    pairing_weights = jnp.asarray([2.0, 3.0, 4.0])
    space = la.ArraySpace(
        (3,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(pairing_weights),
    )
    paired_operator = jnp.asarray([[4.0, 1.0, 0.0], [1.0, 5.0, 0.5], [0.0, 0.5, 6.0]])
    paired_metric = jnp.asarray([[3.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 4.0]])
    operator = la.DenseLinearOperator(
        paired_operator / pairing_weights[:, None],
        source=space,
        target=space,
        properties=_self_adjoint_properties(),
    )
    metric = la.DenseLinearOperator(
        paired_metric / pairing_weights[:, None],
        source=space,
        target=space,
        properties=_self_adjoint_properties(positive_definite=True),
    )
    problem = eigen.GeneralizedEigenproblem(operator, metric)
    result = eigen.eigensolve(
        problem,
        policy=eigen.EigenSolvePolicy(eigen.DenseEigh(), count=3),
    )
    vectors = jnp.asarray(result.eigenvectors)

    assert bool(result.successful)
    assert jnp.allclose(
        result.eigenvalues,
        spla.eigh(pairing_weights[:, None] * operator.matrix, paired_metric)[0],
    )
    assert jnp.allclose(
        vectors.T @ paired_metric @ vectors,
        jnp.eye(3),
        atol=1e-12,
    )
    assert jnp.max(result.residual_norms) < 1e-12


def test_dense_eigenvalue_derivatives_require_isolated_modes():
    properties = _self_adjoint_properties()
    policy = eigen.EigenSolvePolicy(
        eigen.DenseEigh(),
        count=3,
        differentiation="eigenvalues",
    )

    def spectral_sum(diagonal):
        problem = eigen.Eigenproblem(
            la.DiagonalLinearOperator(diagonal, properties=properties)
        )
        return jnp.sum(eigen.eigensolve(problem, policy=policy).eigenvalues)

    diagonal = jnp.asarray([1.0, 2.0, 4.0])
    assert jnp.allclose(jax.jit(jax.grad(spectral_sum))(diagonal), jnp.ones(3))

    repeated = eigen.eigensolve(
        eigen.Eigenproblem(
            la.DiagonalLinearOperator(
                jnp.asarray([1.0, 1.0, 4.0]),
                properties=properties,
            )
        ),
        policy=policy,
    )
    assert repeated.status == int(eigen.EigenSolveStatus.DIFFERENTIATION_REJECTED)
