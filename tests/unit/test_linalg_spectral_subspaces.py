#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg
eig = la.eigen


def _problem(matrix, *, operator_id="spectral-cluster", problem_id="spectral-cluster"):
    return eig.SchurEigenproblem(
        la.DenseLinearOperator(jnp.asarray(matrix), operator_id=operator_id),
        problem_id=problem_id,
    )


def _nonnormal_matrix():
    return jnp.asarray(
        [
            [-2.0, 8.0, 1.0],
            [0.0, -1.0, 3.0],
            [0.0, 0.0, 4.0],
        ]
    )


def test_spectral_selection_masks_half_planes_and_disks():
    eigenvalues = jnp.asarray([-2.0 + 0.5j, -0.25, 1.0 + 1.0j, 3.0])

    assert jnp.array_equal(
        eig.SpectralSelection.real_below(0.0).mask(eigenvalues),
        jnp.asarray([True, True, False, False]),
    )
    assert jnp.array_equal(
        eig.SpectralSelection.real_above(0.5).mask(eigenvalues),
        jnp.asarray([False, False, True, True]),
    )
    assert jnp.array_equal(
        eig.SpectralSelection.disk(0.0, 1.5).mask(eigenvalues),
        jnp.asarray([False, True, True, False]),
    )
    assert jnp.array_equal(
        eig.SpectralSelection.exterior_disk(0.0, 1.5).mask(eigenvalues),
        jnp.asarray([True, False, False, True]),
    )


def test_nonnormal_riesz_projector_commutes_and_is_not_orthogonal_projector():
    matrix = _nonnormal_matrix()
    selection = eig.SpectralSelection.real_below(0.0, expected_dimension=2)
    prepared = eig.prepare_spectral_subspace(_problem(matrix), selection)
    subspace = eig.spectral_subspace(prepared)

    assert subspace.status == int(eig.SpectralSubspaceStatus.SUCCESS)
    assert subspace.successful
    assert jnp.allclose(subspace.projector @ subspace.projector, subspace.projector)
    assert jnp.allclose(matrix @ subspace.projector, subspace.projector @ matrix)
    assert not jnp.allclose(subspace.projector, subspace.orthogonal_projector)
    assert jnp.allclose(
        subspace.projector,
        subspace.basis @ jnp.conj(subspace.left_dual_basis.T),
    )
    assert jnp.allclose(
        jnp.conj(subspace.left_dual_basis.T) @ subspace.basis,
        jnp.eye(2),
    )
    assert prepared.diagnostics.projector_norm > 1.5
    assert prepared.diagnostics.separation_exact
    vector = jnp.asarray([1.0, -2.0, 0.5])
    assert jnp.allclose(subspace.project_coordinates(vector), subspace.projector @ vector)


def test_normal_spectral_projector_equals_orthogonal_projector():
    matrix = jnp.asarray([[1.0, 2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 0.0, 4.0]])
    selection = eig.SpectralSelection.real_below(0.0, expected_dimension=1)
    subspace = eig.spectral_subspace(_problem(matrix), selection)

    assert subspace.status == int(eig.SpectralSubspaceStatus.SUCCESS)
    assert jnp.allclose(
        subspace.projector,
        subspace.orthogonal_projector,
        rtol=1e-11,
        atol=1e-12,
    )
    assert subspace.diagnostics.projector_norm == pytest.approx(1.0)


def test_projector_derivative_matches_centered_finite_difference_and_jit():
    matrix = _nonnormal_matrix()
    perturbation = jnp.asarray([[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [0.05, 0.2, -0.3]])
    selection = eig.SpectralSelection.real_below(0.0, expected_dimension=2)
    prepared = eig.prepare_spectral_subspace(_problem(matrix), selection)
    derivative = jax.jit(eig.spectral_projector_derivative)(prepared, perturbation)
    step = 1e-5
    positive = eig.prepare_spectral_subspace(
        _problem(matrix + step * perturbation),
        selection,
    )
    negative = eig.prepare_spectral_subspace(
        _problem(matrix - step * perturbation),
        selection,
    )
    finite_difference = (positive.projector - negative.projector) / (2.0 * step)

    assert derivative.status == int(eig.SpectralProjectorDerivativeStatus.SUCCESS)
    assert derivative.successful
    assert jnp.allclose(derivative.value, finite_difference, rtol=1e-7, atol=1e-8)
    assert derivative.diagnostics.commutator_residual_norm < 1e-12
    assert derivative.diagnostics.tangent_residual_norm < 1e-12


def test_projector_derivative_is_linear_and_satisfies_differentiated_identities():
    matrix = _nonnormal_matrix()
    selection = eig.SpectralSelection.real_below(0.0, expected_dimension=2)
    prepared = eig.prepare_spectral_subspace(_problem(matrix), selection)
    first = jnp.asarray([[0.2, 0.0, 0.1], [0.0, -0.3, 0.2], [0.1, 0.0, 0.4]])
    second = jnp.asarray([[0.0, -0.2, 0.3], [0.5, 0.1, 0.0], [-0.1, 0.2, 0.0]])
    combined = eig.spectral_projector_derivative(prepared, 1.5 * first - 0.25 * second)
    first_derivative = eig.spectral_projector_derivative(prepared, first)
    second_derivative = eig.spectral_projector_derivative(prepared, second)
    expected = 1.5 * first_derivative.value - 0.25 * second_derivative.value
    projector = prepared.projector
    derivative = combined.value
    perturbation = 1.5 * first - 0.25 * second

    assert jnp.allclose(derivative, expected, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(
        matrix @ derivative - derivative @ matrix,
        projector @ perturbation - perturbation @ projector,
        rtol=1e-11,
        atol=1e-12,
    )
    assert jnp.allclose(
        projector @ derivative + derivative @ projector,
        derivative,
        rtol=1e-11,
        atol=1e-12,
    )


def test_projector_derivative_accepts_a_matching_operator_perturbation():
    matrix = _nonnormal_matrix()
    selection = eig.SpectralSelection.real_below(0.0, expected_dimension=2)
    prepared = eig.prepare_spectral_subspace(_problem(matrix), selection)
    perturbation = jnp.asarray([[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [0.05, 0.2, -0.3]])
    array_result = eig.spectral_projector_derivative(prepared, perturbation)
    operator_result = eig.spectral_projector_derivative(
        prepared,
        la.DenseLinearOperator(perturbation, operator_id="projector-perturbation"),
    )

    assert jnp.allclose(array_result.value, operator_result.value)


def test_spectral_refresh_preserves_dimension_and_rejects_crossings():
    matrix = _nonnormal_matrix()
    selection = eig.SpectralSelection.real_below(0.0)
    prepared = eig.prepare_spectral_subspace(_problem(matrix), selection)
    updated_matrix = matrix.at[0, 0].set(-2.5).at[2, 2].set(3.5)
    refreshed = eig.refresh_spectral_subspace(prepared, _problem(updated_matrix))

    assert refreshed.selected_dimension == prepared.selected_dimension
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    assert refreshed.refresh_count == 1

    crossed_matrix = matrix.at[1, 1].set(0.5)
    with pytest.raises(ValueError, match="changed the selected dimension"):
        eig.refresh_spectral_subspace(prepared, _problem(crossed_matrix))


def test_selection_boundary_expected_dimension_and_resource_limits_are_explicit():
    matrix = jnp.diag(jnp.asarray([-1.0, 0.0, 2.0]))
    problem = _problem(matrix)
    boundary = eig.SpectralSelection.real_below(0.0, boundary_tolerance=1e-6)
    with pytest.raises(ValueError, match="protected"):
        eig.prepare_spectral_subspace(problem, boundary)

    wrong_dimension = eig.SpectralSelection.real_below(
        1.0,
        expected_dimension=1,
    )
    with pytest.raises(ValueError, match="expected dimension"):
        eig.prepare_spectral_subspace(problem, wrong_dimension)

    policy = eig.SpectralSubspacePolicy(
        resources=eig.SpectralSubspaceResourcePolicy(max_dimension=2)
    )
    with pytest.raises(ValueError, match="exceeds limit"):
        eig.plan_spectral_subspace(problem, eig.SpectralSelection.real_below(1.0), policy)


def test_exact_sylvester_separation_budget_and_condition_status_are_visible():
    matrix = _nonnormal_matrix()
    problem = _problem(matrix)
    selection = eig.SpectralSelection.real_below(0.0, expected_dimension=2)
    approximate_policy = eig.SpectralSubspacePolicy(
        resources=eig.SpectralSubspaceResourcePolicy(max_separation_entries=3)
    )
    approximate = eig.prepare_spectral_subspace(
        problem,
        selection,
        approximate_policy,
    )
    assert not approximate.diagnostics.separation_exact
    assert jnp.isnan(approximate.diagnostics.sylvester_separation)

    required_policy = eig.SpectralSubspacePolicy(
        resources=eig.SpectralSubspaceResourcePolicy(max_separation_entries=3),
        require_exact_separation=True,
    )
    with pytest.raises(ValueError, match="Exact Sylvester separation"):
        eig.prepare_spectral_subspace(problem, selection, required_policy)

    conditioned_policy = eig.SpectralSubspacePolicy(max_projector_norm=1.1)
    conditioned = eig.prepare_spectral_subspace(problem, selection, conditioned_policy)
    assert conditioned.status == int(eig.SpectralSubspaceStatus.ILL_CONDITIONED)
    derivative = eig.spectral_projector_derivative(conditioned, jnp.eye(3))
    assert derivative.status == int(eig.SpectralProjectorDerivativeStatus.SOURCE_FAILURE)
