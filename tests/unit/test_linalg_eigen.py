#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as spla

import phydrax as phx


la = phx.linalg
eigen = la.eigen


def _self_adjoint_properties(*, positive_definite: Any = False) -> Any:
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
        # ty: ignore[invalid-argument-type]
        evidence=evidence,
    )


def test_lobpcg_contracts() -> None:
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


def test_linalg_eigen_scenario_1() -> None:
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
    expected = spla.eigh(
        np.asarray(paired_operator, dtype=np.float64),
        np.asarray(paired_metric, dtype=np.float64),
        eigvals_only=True,
    )

    assert bool(result.successful)
    assert jnp.allclose(result.eigenvalues, expected)
    assert jnp.allclose(
        vectors.T @ paired_metric @ vectors,
        jnp.eye(3),
        atol=1e-12,
    )
    assert jnp.max(result.residual_norms) < 1e-12


def test_isolated_eigenvalue_gradient_uses_mathematical_derivative() -> None:
    properties = _self_adjoint_properties()
    policy = eigen.EigenSolvePolicy(
        eigen.LOBPCG(block_dimension=2),
        count=1,
        initial_basis=jnp.eye(2),
        differentiation="eigenvalues",
    )

    def smallest_eigenvalue(coefficient: Any) -> Any:
        operator = la.DiagonalLinearOperator(
            jnp.stack((coefficient, jnp.asarray(3.0))),
            properties=properties,
        )
        return eigen.eigensolve(
            eigen.Eigenproblem(operator),
            policy=policy,
        ).eigenvalues[0]

    assert jnp.allclose(jax.grad(smallest_eigenvalue)(1.25), 1.0, atol=1e-8)


def test_matrix_free_eigenvalue_gradient_supports_closure_converted_operator() -> None:
    properties = _self_adjoint_properties()
    policy = eigen.EigenSolvePolicy(
        eigen.LOBPCG(block_dimension=2),
        count=1,
        initial_basis=jnp.eye(2),
        differentiation="eigenvalues",
    )

    def smallest_eigenvalue(coefficient: Any) -> Any:
        diagonal = jnp.stack((coefficient, jnp.asarray(3.0)))
        space = la.ArraySpace((2,), dtype=diagonal.dtype)
        operator = la.FunctionLinearOperator(
            lambda vector: diagonal * vector,
            source=space,
            target=space,
            properties=properties,
        )
        return eigen.eigensolve(
            eigen.Eigenproblem(operator),
            policy=policy,
        ).eigenvalues[0]

    gradient = jax.jit(jax.grad(smallest_eigenvalue))(jnp.asarray(1.25))

    assert jnp.allclose(gradient, 1.0, atol=1e-8)


def test_dense_eigenvalue_derivatives_require_isolated_modes() -> None:
    properties = _self_adjoint_properties()
    policy = eigen.EigenSolvePolicy(
        eigen.DenseEigh(),
        count=3,
        differentiation="eigenvalues",
    )

    def spectral_sum(diagonal: Any) -> Any:
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


def test_generalized_restart_preserves_metric_images_and_deflates_dependent_directions() -> (
    None
):
    """A singular PSD stiffness and healthy noncommuting mass keep bounded Ritz values."""
    size = 19
    random = np.random.default_rng(1701)
    stiffness_basis, _ = np.linalg.qr(random.normal(size=(size, size)))
    mass_basis, _ = np.linalg.qr(random.normal(size=(size, size)))
    stiffness = (
        stiffness_basis
        @ np.diag(np.concatenate((np.zeros(7), np.linspace(1.0, 12.0, 12))))
        @ stiffness_basis.T
    )
    mass = mass_basis @ np.diag(np.linspace(0.04, 0.3, size)) @ mass_basis.T
    space = la.ArraySpace(
        (size,), dtype=jnp.float64, space_id="generalized-restart:coordinates"
    )
    operator = la.DenseLinearOperator(
        jnp.asarray(stiffness),
        source=space,
        target=space,
        properties=_self_adjoint_properties(),
        operator_id="generalized-restart:stiffness",
    )
    metric = la.DenseLinearOperator(
        jnp.asarray(mass),
        source=space,
        target=space,
        properties=_self_adjoint_properties(positive_definite=True),
        operator_id="generalized-restart:mass",
    )
    problem = eigen.GeneralizedEigenproblem(
        operator, metric, problem_id="generalized-restart:pencil"
    )
    policy = eigen.EigenSolvePolicy(
        eigen.RestartedLanczos(subspace_dimension=size),
        count=1,
        which="largest-algebraic",
        max_steps=100,
        key=jax.random.key(0),
        tolerance=eigen.EigenTolerancePolicy(
            relative=1e-8, absolute=1e-10, orthogonality=1e-7
        ),
    )
    result = eigen.eigensolve(problem, policy=policy)
    oracle_stiffness: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        stiffness, dtype=np.float64
    ).reshape((size, size))
    oracle_mass: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        mass, dtype=np.float64
    ).reshape((size, size))
    expected = spla.eigvalsh(oracle_stiffness, oracle_mass)[-1]
    assert bool(result.successful)
    np.testing.assert_allclose(result.eigenvalues, [expected], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        result.eigenvectors.T @ mass @ result.eigenvectors,
        np.ones((1, 1)),
        atol=1e-7,
    )
    assert bool(result.residual_norms[0] < 1e-6)


def test_lobpcg_zero_mode_keeps_true_operator_and_metric_images() -> None:
    """Small residual/direction differences must not amplify cached image roundoff."""
    size = 64
    indices = np.arange(size, dtype=np.int64)
    gradient = np.zeros((size, size), dtype=np.float64)
    gradient[indices, indices] = -1.0
    gradient[indices, (indices + 1) % size] = 1.0
    stiffness = 1000.0 * gradient.T @ np.diag(np.linspace(1.0, 2.0, size)) @ gradient
    weights = np.linspace(1.0, 1.5, size)
    space = la.ArraySpace((size,), dtype=jnp.float64, space_id="lobpcg-zero:coordinates")
    operator = la.DenseLinearOperator(
        jnp.asarray(stiffness),
        source=space,
        target=space,
        properties=_self_adjoint_properties(),
        operator_id="lobpcg-zero:stiffness",
    )
    metric = la.DiagonalLinearOperator(
        jnp.asarray(weights),
        space=space,
        properties=_self_adjoint_properties(positive_definite=True),
        operator_id="lobpcg-zero:mass",
    )
    preconditioner = la.DiagonalPreconditioner(
        jnp.asarray(np.diag(stiffness)),
        space=space,
        positive_definite=True,
    )
    policy = eigen.EigenSolvePolicy(
        eigen.LOBPCG(block_dimension=3),
        count=3,
        max_steps=1000,
        key=jax.random.key(0),
        tolerance=eigen.EigenTolerancePolicy(
            relative=1e-9, absolute=1e-9, orthogonality=1e-9
        ),
        preconditioning=la.PreconditioningPolicy(preconditioner),
    )
    result = eigen.eigensolve(
        eigen.GeneralizedEigenproblem(operator, metric), policy=policy
    )
    assert bool(result.successful)
    basis = np.asarray(result.eigenvectors)
    np.testing.assert_allclose(result.eigenvalues[0], 0, atol=1e-9)
    assert np.linalg.norm(stiffness @ basis[:, 0]) < 1e-8
    expected = np.ones((size,), dtype=np.float64) / np.sqrt(np.sum(weights))
    np.testing.assert_allclose(
        np.outer(basis[:, 0], basis[:, 0]), np.outer(expected, expected), atol=1e-8
    )
