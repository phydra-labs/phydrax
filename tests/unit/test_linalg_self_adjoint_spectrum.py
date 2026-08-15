#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
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


def _standard_problem(matrix):
    return eigen.Eigenproblem(
        la.DenseLinearOperator(
            matrix,
            properties=_self_adjoint_properties(),
        )
    )


def _generalized_problem(operator_matrix, metric_matrix, *, space=None):
    return eigen.GeneralizedEigenproblem(
        la.DenseLinearOperator(
            operator_matrix,
            source=space,
            target=space,
            properties=_self_adjoint_properties(),
        ),
        la.DenseLinearOperator(
            metric_matrix,
            source=space,
            target=space,
            properties=_self_adjoint_properties(positive_definite=True),
        ),
    )


def test_self_adjoint_spectrum_reuses_dense_state_and_refreshes_numeric_values():
    matrix = jnp.asarray(
        [
            [2.0, 0.4, 0.0, 0.0],
            [0.4, 3.0, 0.3, 0.0],
            [0.0, 0.3, 5.0, 0.2],
            [0.0, 0.0, 0.2, 8.0],
        ]
    )
    problem = _standard_problem(matrix)
    plan = eigen.plan_self_adjoint_spectrum(problem)
    prepared = eigen.prepare_self_adjoint_spectrum(problem, plan)
    result = eigen.self_adjoint_spectrum(prepared)
    compiled = jax.jit(eigen.self_adjoint_spectrum)(prepared)

    assert bool(result.successful)
    assert jnp.allclose(result.eigenvalues, jnp.linalg.eigvalsh(matrix), atol=1e-12)
    assert jnp.allclose(compiled.eigenvalues, result.eigenvalues, atol=1e-12)
    assert jnp.allclose(result.inverse_basis @ result.eigenvectors, jnp.eye(4), atol=1e-12)
    assert prepared.problem.dimension == 4
    assert result.provenance.plan_id == plan.plan_id
    assert plan.cost.retained_bytes > 0

    changed = matrix.at[0, 0].set(2.5)
    refreshed = eigen.refresh_self_adjoint_spectrum(
        prepared,
        _standard_problem(changed),
    )
    assert refreshed.eigen_prepared.numeric_version == prepared.eigen_prepared.numeric_version + 1
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert not jnp.allclose(refreshed.eigenvalues, prepared.eigenvalues)

    incompatible = la.ArraySpace((5,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="symbolic plan"):
        eigen.refresh_self_adjoint_spectrum(
            prepared,
            eigen.Eigenproblem(
                la.DiagonalLinearOperator(
                    jnp.arange(1.0, 6.0),
                    space=incompatible,
                    properties=_self_adjoint_properties(),
                )
            ),
        )


def test_generalized_spectrum_honors_declared_pairing_and_metric():
    pairing_weights = jnp.asarray([2.0, 3.0, 5.0, 7.0])
    space = la.ArraySpace(
        (4,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(pairing_weights),
    )
    paired_operator = jnp.asarray(
        [
            [4.0, 0.4, 0.0, 0.0],
            [0.4, 7.0, 0.5, 0.0],
            [0.0, 0.5, 11.0, 0.2],
            [0.0, 0.0, 0.2, 17.0],
        ]
    )
    paired_metric = jnp.asarray(
        [
            [2.0, 0.1, 0.0, 0.0],
            [0.1, 3.0, 0.2, 0.0],
            [0.0, 0.2, 4.0, 0.1],
            [0.0, 0.0, 0.1, 5.0],
        ]
    )
    problem = _generalized_problem(
        paired_operator / pairing_weights[:, None],
        paired_metric / pairing_weights[:, None],
        space=space,
    )
    spectrum = eigen.self_adjoint_spectrum(problem)
    reference_values, reference_vectors = spla.eigh(paired_operator, paired_metric)

    assert bool(spectrum.successful)
    assert jnp.allclose(spectrum.eigenvalues, reference_values, atol=1e-12)
    assert jnp.allclose(
        jnp.conj(spectrum.eigenvectors.T) @ paired_metric @ spectrum.eigenvectors,
        jnp.eye(4),
        atol=1e-12,
    )
    assert jnp.allclose(
        spectrum.inverse_basis,
        jnp.conj(spectrum.eigenvectors.T) @ paired_metric,
        atol=1e-12,
    )
    assert jnp.allclose(
        jnp.abs(jnp.conj(reference_vectors.T) @ paired_metric @ spectrum.eigenvectors),
        jnp.eye(4),
        atol=1e-12,
    )


def test_spectrum_planning_rejects_constraints_and_resource_overflow():
    diagonal = jnp.asarray([1.0, 2.0, 4.0, 8.0])
    operator = la.DiagonalLinearOperator(
        diagonal,
        properties=_self_adjoint_properties(),
    )
    problem = eigen.Eigenproblem(operator)
    constrained = eigen.Eigenproblem(
        operator,
        constraints=la.LinearSubspace(
            operator.source,
            jnp.asarray([[1.0], [0.0], [0.0], [0.0]]),
            orthonormal=True,
        ),
    )

    with pytest.raises(ValueError, match="constraints"):
        eigen.plan_self_adjoint_spectrum(constrained)
    with pytest.raises(ValueError, match="retained estimate"):
        eigen.plan_self_adjoint_spectrum(
            problem,
            eigen.SelfAdjointSpectrumPolicy(max_retained_bytes=1),
        )


def test_projector_is_basis_invariant_for_repeated_internal_eigenvalues():
    matrix = jnp.diag(jnp.asarray([1.0, 1.0, 4.0, 7.0]))
    spectrum = eigen.prepare_self_adjoint_spectrum(_standard_problem(matrix))
    selection = eigen.SpectralSelection.real_below(
        2.0,
        expected_dimension=2,
    )
    subspace = eigen.self_adjoint_spectral_subspace(spectrum, selection)
    angle = 0.73
    rotation = jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ]
    )
    rotated_vectors = spectrum.eigenvectors.at[:, :2].set(
        spectrum.eigenvectors[:, :2] @ rotation
    )
    rotated_inverse = spectrum.inverse_basis.at[:2, :].set(
        rotation.T @ spectrum.inverse_basis[:2, :]
    )
    rotated_spectrum = eqx.tree_at(
        lambda current: (current.eigenvectors, current.inverse_basis),
        spectrum,
        (rotated_vectors, rotated_inverse),
    )
    rotated = eigen.self_adjoint_spectral_subspace(rotated_spectrum, selection)

    assert bool(subspace.successful)
    assert jnp.allclose(subspace.projector, jnp.diag(jnp.asarray([1.0, 1.0, 0.0, 0.0])))
    assert jnp.allclose(rotated.projector, subspace.projector, atol=1e-12)
    assert subspace.diagnostics.idempotence_error < 1e-12


def test_projector_derivatives_match_explicit_kernel_forward_reverse_and_finite_difference():
    matrix = jnp.asarray(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 1.0, -0.2, 0.0],
            [0.1, -0.2, 4.0, 0.3],
            [0.0, 0.0, 0.3, 7.0],
        ]
    )
    perturbation = jnp.asarray(
        [
            [0.1, -0.2, 0.3, 0.0],
            [-0.2, -0.1, 0.2, 0.1],
            [0.3, 0.2, 0.4, -0.1],
            [0.0, 0.1, -0.1, -0.2],
        ]
    )
    selection = eigen.SpectralSelection.real_below(2.0, expected_dimension=2)
    policy = eigen.SelfAdjointSpectralSubspacePolicy(differentiation="projector")

    def projector(current):
        return eigen.self_adjoint_spectral_subspace(
            _standard_problem(current),
            selection,
            policy=policy,
        ).projector

    primal, tangent = jax.jit(jax.jvp, static_argnums=0)(
        projector,
        (matrix,),
        (perturbation,),
    )
    prepared = eigen.prepare_self_adjoint_spectrum(_standard_problem(matrix))
    explicit = eigen.self_adjoint_spectral_projector_derivative(
        prepared,
        selection,
        perturbation,
    )
    step = 1e-5
    finite_difference = (
        projector(matrix + step * perturbation)
        - projector(matrix - step * perturbation)
    ) / (2 * step)
    cotangent = jnp.asarray(
        [
            [0.2, -0.1, 0.3, 0.0],
            [-0.1, 0.4, -0.2, 0.1],
            [0.3, -0.2, -0.2, 0.2],
            [0.0, 0.1, 0.2, 0.1],
        ]
    )
    reverse = jax.jit(jax.grad(lambda value: jnp.sum(projector(value) * cotangent)))(
        matrix
    )

    assert jnp.all(jnp.isfinite(primal))
    assert jnp.allclose(tangent, explicit.projector, rtol=1e-8, atol=1e-9)
    assert jnp.allclose(tangent, finite_difference, rtol=3e-6, atol=3e-7)
    assert jnp.allclose(
        jnp.sum(reverse * perturbation),
        jnp.sum(cotangent * tangent),
        rtol=1e-9,
        atol=1e-10,
    )
    assert bool(explicit.successful)
    assert explicit.diagnostics.relative_residual < 1e-12


def test_complex_hermitian_projector_derivative_is_cluster_safe_and_matches_finite_difference():
    matrix = jnp.asarray(
        [
            [1.0 + 0.0j, 0.0, 0.0 + 0.1j, 0.0],
            [0.0, 1.0 + 0.0j, 0.2 + 0.0j, 0.0],
            [0.0 - 0.1j, 0.2 + 0.0j, 4.0 + 0.0j, 0.0 + 0.1j],
            [0.0, 0.0, 0.0 - 0.1j, 7.0 + 0.0j],
        ]
    )
    perturbation = jnp.asarray(
        [
            [0.1 + 0.0j, 0.2 + 0.1j, 0.3 - 0.2j, 0.0],
            [0.2 - 0.1j, -0.1 + 0.0j, 0.2 + 0.1j, 0.0 + 0.1j],
            [0.3 + 0.2j, 0.2 - 0.1j, 0.4 + 0.0j, -0.1 + 0.1j],
            [0.0, 0.0 - 0.1j, -0.1 - 0.1j, -0.2 + 0.0j],
        ]
    )
    selection = eigen.SpectralSelection.real_below(2.0, expected_dimension=2)
    policy = eigen.SelfAdjointSpectralSubspacePolicy(differentiation="projector")

    def projector(current):
        return eigen.self_adjoint_spectral_subspace(
            _standard_problem(current),
            selection,
            policy=policy,
        ).projector

    _, tangent = jax.jvp(projector, (matrix,), (perturbation,))
    step = 1e-5
    finite_difference = (
        projector(matrix + step * perturbation)
        - projector(matrix - step * perturbation)
    ) / (2 * step)

    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.allclose(tangent, jnp.conj(tangent.T), atol=1e-12)
    assert jnp.allclose(tangent, finite_difference, rtol=3e-6, atol=3e-7)


def test_generalized_projector_and_density_derivatives_include_metric_perturbations():
    operator = jnp.diag(jnp.asarray([1.0, 4.0, 12.0, 28.0]))
    metric = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    operator_perturbation = jnp.asarray(
        [
            [0.1, 0.2, 0.0, 0.0],
            [0.2, -0.1, 0.3, 0.0],
            [0.0, 0.3, 0.2, 0.1],
            [0.0, 0.0, 0.1, -0.2],
        ]
    )
    metric_perturbation = jnp.asarray(
        [
            [0.02, -0.01, 0.0, 0.0],
            [-0.01, 0.03, 0.02, 0.0],
            [0.0, 0.02, -0.01, 0.01],
            [0.0, 0.0, 0.01, 0.02],
        ]
    )
    selection = eigen.SpectralSelection.real_below(3.0, expected_dimension=2)
    policy = eigen.SelfAdjointSpectralSubspacePolicy(differentiation="projector")

    def outputs(current_operator, current_metric):
        result = eigen.self_adjoint_spectral_subspace(
            _generalized_problem(current_operator, current_metric),
            selection,
            policy=policy,
        )
        return result.projector, result.density_kernel

    _, tangent = jax.jvp(
        outputs,
        (operator, metric),
        (operator_perturbation, metric_perturbation),
    )
    prepared = eigen.prepare_self_adjoint_spectrum(
        _generalized_problem(operator, metric)
    )
    explicit = eigen.self_adjoint_spectral_projector_derivative(
        prepared,
        selection,
        operator_perturbation,
        metric_perturbation,
    )
    step = 1e-5
    plus = outputs(
        operator + step * operator_perturbation,
        metric + step * metric_perturbation,
    )
    minus = outputs(
        operator - step * operator_perturbation,
        metric - step * metric_perturbation,
    )
    finite_difference = tuple(
        (upper - lower) / (2 * step) for upper, lower in zip(plus, minus, strict=True)
    )

    assert jnp.allclose(tangent[0], explicit.projector, rtol=1e-8, atol=1e-9)
    assert jnp.allclose(tangent[1], explicit.density_kernel, rtol=1e-8, atol=1e-9)
    assert jnp.allclose(tangent[0], finite_difference[0], rtol=5e-6, atol=5e-7)
    assert jnp.allclose(tangent[1], finite_difference[1], rtol=5e-6, atol=5e-7)
    assert explicit.diagnostics.density_identity_residual_norm < 1e-12


def test_subspace_reports_dimension_boundary_and_external_gap_failures():
    spectrum = eigen.prepare_self_adjoint_spectrum(
        _standard_problem(jnp.diag(jnp.asarray([1.0, 1.0, 4.0, 7.0])))
    )
    mismatch = eigen.self_adjoint_spectral_subspace(
        spectrum,
        eigen.SpectralSelection.real_below(2.0, expected_dimension=1),
    )
    boundary = eigen.self_adjoint_spectral_subspace(
        spectrum,
        eigen.SpectralSelection.real_below(1.0, expected_dimension=2, inclusive=True),
    )
    split_cluster = eigen.self_adjoint_spectral_subspace(
        spectrum,
        eigen.SpectralSelection.real_below(1.0, expected_dimension=1, inclusive=True),
    )

    assert mismatch.status == int(
        eigen.SelfAdjointSpectralSubspaceStatus.SELECTION_DIMENSION_MISMATCH
    )
    assert boundary.status == int(
        eigen.SelfAdjointSpectralSubspaceStatus.BOUNDARY_UNRESOLVED
    )
    assert split_cluster.status in (
        int(eigen.SelfAdjointSpectralSubspaceStatus.SELECTION_DIMENSION_MISMATCH),
        int(eigen.SelfAdjointSpectralSubspaceStatus.BOUNDARY_UNRESOLVED),
        int(eigen.SelfAdjointSpectralSubspaceStatus.CLUSTER_NOT_ISOLATED),
    )


def test_batched_dense_eigen_lifecycle_handles_standard_generalized_and_complex_cases():
    standard_matrices = jnp.asarray(
        [
            [[1.0, 0.2, 0.0], [0.2, 3.0, 0.1], [0.0, 0.1, 6.0]],
            [[2.0, -0.1, 0.3], [-0.1, 4.0, 0.2], [0.3, 0.2, 7.0]],
        ]
    )
    standard_problem = eigen.Eigenproblem(
        la.DenseLinearOperator(
            standard_matrices,
            properties=_self_adjoint_properties(),
        )
    )
    policy = eigen.EigenSolvePolicy(eigen.DenseEigh(), count=3)
    result = eigen.eigensolve(standard_problem, policy=policy)
    compiled = jax.jit(lambda problem: eigen.eigensolve(problem, policy=policy))(
        standard_problem
    )
    reference = jax.vmap(jnp.linalg.eigvalsh)(standard_matrices)

    assert result.eigenvalues.shape == (2, 3)
    assert result.eigenvectors.shape == (2, 3, 3)
    assert result.status.shape == (2,)
    assert jnp.all(result.successful)
    assert jnp.allclose(result.eigenvalues, reference, atol=1e-12)
    assert jnp.allclose(compiled.eigenvalues, reference, atol=1e-12)

    complex_operator = jnp.asarray(
        [
            [
                [2.0 + 0.0j, 0.2 + 0.1j, 0.0 + 0.0j],
                [0.2 - 0.1j, 4.0 + 0.0j, -0.1 + 0.2j],
                [0.0 + 0.0j, -0.1 - 0.2j, 7.0 + 0.0j],
            ],
            [
                [3.0 + 0.0j, -0.2 + 0.1j, 0.1 + 0.0j],
                [-0.2 - 0.1j, 5.0 + 0.0j, 0.0 + 0.2j],
                [0.1 + 0.0j, 0.0 - 0.2j, 8.0 + 0.0j],
            ],
        ]
    )
    complex_metric = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        ],
        dtype=jnp.complex128,
    )
    generalized = _generalized_problem(complex_operator, complex_metric)
    generalized_result = eigen.eigensolve(generalized, policy=policy)
    vectors = generalized_result.eigenvectors
    residual = (
        complex_operator @ vectors
        - (complex_metric @ vectors)
        * generalized_result.eigenvalues[..., None, :]
    )
    metric_gram = (
        jnp.conj(jnp.swapaxes(vectors, -1, -2))
        @ complex_metric
        @ vectors
    )

    assert jnp.all(generalized_result.successful)
    assert jnp.max(jnp.abs(residual)) < 1e-12
    assert jnp.allclose(
        metric_gram,
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        atol=1e-12,
    )

    batched_plan = eigen.plan_eigensolve(standard_problem, policy)
    unbatched_plan = eigen.plan_eigensolve(
        _standard_problem(standard_matrices[0]),
        policy,
    )
    batched_cost = next(value for value in batched_plan.candidates if value.accepted)
    unbatched_cost = next(value for value in unbatched_plan.candidates if value.accepted)
    assert batched_cost.storage_bytes == 2 * unbatched_cost.storage_bytes
    with pytest.raises(ValueError, match="LOBPCG|batch"):
        eigen.plan_eigensolve(
            standard_problem,
            eigen.EigenSolvePolicy(
                eigen.LOBPCG(block_dimension=3),
                count=2,
            ),
        )
    with pytest.raises(ValueError, match="batch shape"):
        _generalized_problem(
            standard_matrices,
            jnp.broadcast_to(jnp.eye(3), (3, 3, 3)),
        )


def test_batched_spectral_subspaces_have_fixed_shapes_mixed_status_and_exact_derivatives():
    matrices = jnp.stack(
        (
            jnp.diag(jnp.asarray([1.0, 2.0, 4.0])),
            jnp.diag(jnp.asarray([1.0, 3.0, 5.0])),
        )
    )
    perturbations = jnp.asarray(
        [
            [[0.1, 0.2, 0.3], [0.2, -0.1, 0.1], [0.3, 0.1, 0.2]],
            [[0.2, 0.1, -0.2], [0.1, 0.3, 0.2], [-0.2, 0.2, -0.1]],
        ]
    )
    problem = eigen.Eigenproblem(
        la.DenseLinearOperator(
            matrices,
            properties=_self_adjoint_properties(),
        )
    )
    spectrum = eigen.prepare_self_adjoint_spectrum(problem)
    mixed = eigen.self_adjoint_spectral_subspace(
        spectrum,
        eigen.SpectralSelection.real_below(2.5, expected_dimension=2),
    )

    assert mixed.projector.shape == (2, 3, 3)
    assert mixed.selected_eigenvalues.shape == (2, 2)
    assert jnp.array_equal(
        mixed.status,
        jnp.asarray(
            [
                int(eigen.SelfAdjointSpectralSubspaceStatus.SUCCESS),
                int(
                    eigen.SelfAdjointSpectralSubspaceStatus.SELECTION_DIMENSION_MISMATCH
                ),
            ]
        ),
    )

    selection = eigen.SpectralSelection.real_below(3.5, expected_dimension=2)
    derivative_policy = eigen.SelfAdjointSpectralSubspacePolicy(
        differentiation="projector"
    )

    def batched_projector(current):
        current_problem = eigen.Eigenproblem(
            la.DenseLinearOperator(
                current,
                properties=_self_adjoint_properties(),
            )
        )
        return eigen.self_adjoint_spectral_subspace(
            current_problem,
            selection,
            policy=derivative_policy,
        ).projector

    _, tangent = jax.jit(
        lambda value, direction: jax.jvp(
            batched_projector,
            (value,),
            (direction,),
        )
    )(matrices, perturbations)
    explicit = eigen.self_adjoint_spectral_projector_derivative(
        spectrum,
        selection,
        perturbations,
    )

    assert tangent.shape == (2, 3, 3)
    assert jnp.all(explicit.successful)
    assert jnp.allclose(tangent, explicit.projector, rtol=1e-9, atol=1e-10)
    assert jnp.all(explicit.diagnostics.relative_residual < 1e-12)
