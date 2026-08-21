#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.linalg.eigen import (
    CayleyTransform,
    DenseSchurQZ,
    general_eigensolve,
    general_eigenvalue_derivative,
    general_invariant_projector_derivative,
    GeneralEigenproblem,
    GeneralEigenResourcePolicy,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
    GeneralEigenSolveStatus,
    GeneralEigenTolerancePolicy,
    plan_general_eigensolve,
    prepare_general_eigensolve,
    refresh_general_eigensolve,
    RestartedArnoldi,
    ShiftInvertTransform,
)


la = phx.linalg


def _dense(matrix, *, space=None):
    return la.DenseLinearOperator(jnp.asarray(matrix), source=space, target=space)


class _MatrixFreeOperator(la.AbstractLinearOperator):
    matrix: jax.Array

    def __init__(self, matrix, /, *, properties=None, operator_id):
        matrix_ = jnp.asarray(matrix)
        space = la.ArraySpace((matrix_.shape[0],), dtype=matrix_.dtype)
        self.source = space
        self.target = space
        self.matrix = matrix_
        self.properties = la.OperatorProperties() if properties is None else properties
        self.capabilities = la.OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def mv(self, vector):
        return self.matrix @ self.source.validate(vector)

    def transpose_mv(self, vector):
        return self.matrix.T @ self.target.validate(vector)

    def adjoint_mv(self, vector):
        return jnp.conj(self.matrix.T) @ self.target.validate(vector)

    def _materialize(self):
        raise AssertionError("native Arnoldi must not materialize this operator")


def test_dense_standard_and_generalized_eigenpairs_are_complex_and_paired():
    standard_matrix = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    standard = general_eigensolve(GeneralEigenproblem(_dense(standard_matrix)))

    assert bool(standard.successful)
    assert jnp.allclose(
        jnp.sort(jnp.imag(standard.eigenvalues)),
        jnp.asarray([-1.0, 1.0]),
        atol=1e-10,
    )
    assert standard.right_eigenvector_coordinates.shape == (2, 2)
    assert standard.left_eigenvector_coordinates.shape == (2, 2)
    assert jnp.all(standard.diagnostics.right_relative_residuals < 1e-10)
    assert jnp.all(standard.diagnostics.left_relative_residuals < 1e-10)
    assert jnp.allclose(
        standard.diagnostics.pairing_matrix,
        jnp.eye(2),
        atol=1e-9,
    )

    mass_matrix = jnp.diag(jnp.asarray([1.0, 2.0]))
    pencil_matrix = jnp.asarray([[0.0, -1.0], [2.0, 0.0]])
    generalized = general_eigensolve(
        GeneralEigenproblem(_dense(pencil_matrix), _dense(mass_matrix))
    )
    right = generalized.right_eigenvector_coordinates
    left = generalized.left_eigenvector_coordinates

    assert bool(generalized.successful)
    assert jnp.allclose(
        jnp.sort(jnp.imag(generalized.eigenvalues)),
        jnp.asarray([-1.0, 1.0]),
        atol=1e-10,
    )
    assert jnp.allclose(
        pencil_matrix @ right,
        (mass_matrix @ right) * generalized.eigenvalues[None, :],
        atol=1e-9,
    )
    assert jnp.allclose(
        jnp.conj(pencil_matrix.T) @ left,
        (jnp.conj(mass_matrix.T) @ left) * jnp.conj(generalized.eigenvalues)[None, :],
        atol=1e-9,
    )
    assert jnp.allclose(
        generalized.diagnostics.pairing_matrix,
        jnp.eye(2),
        atol=1e-9,
    )


def test_dense_singular_pencil_reports_homogeneous_finite_and_infinite_modes():
    matrix = _dense(jnp.diag(jnp.asarray([2.0, 3.0])))
    singular_mass = _dense(jnp.diag(jnp.asarray([1.0, 0.0])))
    problem = GeneralEigenproblem(matrix, singular_mass)
    prepared = prepare_general_eigensolve(problem)
    result = general_eigensolve(prepared)

    assert bool(result.successful)
    assert int(result.diagnostics.mass_rank) == 1
    assert bool(result.diagnostics.mass_singular)
    assert int(jnp.sum(result.finite_mask)) == 1
    assert int(jnp.sum(result.infinite_mask)) == 1
    assert not bool(jnp.any(result.indeterminate_mask))
    finite_value = result.eigenvalues[result.finite_mask][0]
    assert jnp.allclose(finite_value, 2.0)
    assert jnp.isinf(jnp.abs(result.eigenvalues[result.infinite_mask][0]))
    assert jnp.allclose(result.beta[result.infinite_mask], 0.0)

    with pytest.raises(ValueError, match="singular"):
        prepare_general_eigensolve(
            problem,
            GeneralEigenSolvePolicy(DenseSchurQZ(), singular_mass="error"),
        )


def test_restarted_arnoldi_shift_invert_and_cayley_target_interior_modes():
    values = jnp.asarray([0.0, 2.0, 5.0, 8.0, 11.0])
    problem = GeneralEigenproblem(
        _MatrixFreeOperator(
            jnp.diag(values),
            operator_id="matrix-free-interior-spectrum",
        )
    )
    selection = GeneralEigenSelection.closest(5.2, 1)

    shift_invert = general_eigensolve(
        problem,
        policy=GeneralEigenSolvePolicy(
            RestartedArnoldi(subspace_dimension=4),
            transform=ShiftInvertTransform(5.2),
            selection=selection,
            max_steps=100,
        ),
    )
    cayley = general_eigensolve(
        problem,
        policy=GeneralEigenSolvePolicy(
            RestartedArnoldi(subspace_dimension=4),
            transform=CayleyTransform(5.2),
            selection=selection,
            max_steps=100,
        ),
    )

    assert bool(shift_invert.successful)
    assert bool(cayley.successful)
    assert jnp.allclose(shift_invert.eigenvalues, jnp.asarray([5.0]), atol=1e-8)
    assert jnp.allclose(cayley.eigenvalues, jnp.asarray([5.0]), atol=1e-8)
    assert shift_invert.provenance.backend == "phydrax-native-restarted-arnoldi"
    assert not shift_invert.provenance.host_only
    assert shift_invert.provenance.transform == "shift-invert"
    assert cayley.provenance.transform == "cayley"
    assert shift_invert.diagnostics.pairing_matrix.shape == (1, 1)
    assert jnp.allclose(shift_invert.diagnostics.pairing_matrix, 1.0, atol=1e-7)


def test_matrix_free_generalized_arnoldi_uses_certified_mass_and_full_pairing():
    dimension = 6
    mass_matrix = jnp.diag(jnp.linspace(1.0, 2.0, dimension))
    reduced = jnp.diag(jnp.arange(1.0, dimension + 1.0))
    reduced = reduced.at[jnp.arange(dimension - 1), jnp.arange(1, dimension)].set(0.2)
    pencil_matrix = mass_matrix @ reduced
    mass_properties = la.OperatorProperties(
        rank=dimension,
        evidence={"rank": "construction"},
    )
    problem = GeneralEigenproblem(
        _MatrixFreeOperator(
            pencil_matrix,
            operator_id="matrix-free-generalized-operator",
        ),
        _MatrixFreeOperator(
            mass_matrix,
            properties=mass_properties,
            operator_id="matrix-free-certified-mass",
        ),
    )
    result = general_eigensolve(
        problem,
        policy=GeneralEigenSolvePolicy(
            RestartedArnoldi(subspace_dimension=6),
            selection=GeneralEigenSelection("largest-magnitude", count=2),
            max_steps=60,
        ),
    )

    assert bool(result.successful)
    assert jnp.allclose(
        jnp.sort(jnp.real(result.eigenvalues)),
        jnp.asarray([5.0, 6.0]),
        atol=1e-6,
    )
    assert jnp.all(result.diagnostics.right_relative_residuals < 1e-6)
    assert jnp.all(result.diagnostics.left_relative_residuals < 1e-6)
    assert jnp.allclose(
        result.diagnostics.pairing_matrix,
        jnp.eye(2),
        atol=1e-6,
    )


def test_public_native_eigensolve_is_jittable_without_materialization():
    values = jnp.asarray([1.0, 2.0, 4.0, 7.0, 11.0])
    problem = GeneralEigenproblem(
        _MatrixFreeOperator(
            jnp.diag(values),
            operator_id="jitted-matrix-free-arnoldi",
        )
    )
    prepared = prepare_general_eigensolve(
        problem,
        GeneralEigenSolvePolicy(
            RestartedArnoldi(subspace_dimension=4),
            selection=GeneralEigenSelection("largest-magnitude", count=1),
            max_steps=20,
        ),
    )
    compiled = eqx.filter_jit(general_eigensolve)
    result = compiled(prepared)

    assert bool(result.successful)
    assert result.eigenvalues.shape == (1,)
    assert result.right_eigenvector_coordinates.shape == (5, 1)
    assert int(result.diagnostics.arnoldi_action_count) > 0
    assert int(result.diagnostics.available_count) == 1
    assert jnp.all(result.diagnostics.converged_mask)


def test_targeted_native_simple_derivative_is_matrix_free_and_jittable():
    problem = GeneralEigenproblem(
        _MatrixFreeOperator(
            jnp.diag(jnp.asarray([1.0, 2.0, 4.0, 7.0, 11.0])),
            operator_id="targeted-derivative-operator",
        )
    )
    prepared = prepare_general_eigensolve(
        problem,
        GeneralEigenSolvePolicy(
            RestartedArnoldi(subspace_dimension=4),
            selection=GeneralEigenSelection("largest-magnitude", count=1),
            max_steps=20,
        ),
    )
    result = general_eigensolve(prepared)
    perturbation = _MatrixFreeOperator(
        jnp.diag(jnp.asarray([0.5, 1.0, 2.0, 3.0, 5.0])),
        operator_id="targeted-derivative-perturbation",
    )
    derivative = general_eigenvalue_derivative(
        prepared,
        result,
        perturbation,
        0,
    )
    compiled = eqx.filter_jit(
        lambda prepared_, result_, perturbation_: general_eigenvalue_derivative(
            prepared_,
            result_,
            perturbation_,
            0,
        )
    )
    compiled_derivative = compiled(prepared, result, perturbation)

    assert bool(derivative.successful)
    assert bool(compiled_derivative.successful)
    assert jnp.allclose(derivative.scalar_derivative, 5.0, atol=1e-7)
    assert jnp.allclose(compiled_derivative.scalar_derivative, 5.0, atol=1e-7)
    assert (
        derivative.provenance.method == "matrix-free paired simple-mode pencil quotient"
    )


def test_native_partial_convergence_and_failure_policy_are_residual_driven():
    values = jnp.asarray([1.0, 2.0, 4.0, 7.0, 11.0])
    problem = GeneralEigenproblem(
        _MatrixFreeOperator(
            jnp.diag(values),
            operator_id="partial-matrix-free-arnoldi",
        )
    )
    partial_policy = GeneralEigenSolvePolicy(
        RestartedArnoldi(subspace_dimension=3),
        selection=GeneralEigenSelection("largest-magnitude", count=1),
        max_steps=2,
        tolerance=GeneralEigenTolerancePolicy(relative=0.0, absolute=0.0),
    )
    result = general_eigensolve(problem, policy=partial_policy)

    assert int(result.status) == int(GeneralEigenSolveStatus.PARTIAL_CONVERGENCE)
    assert int(result.diagnostics.available_count) == 0
    assert int(result.diagnostics.converged_count) == 0
    assert not jnp.any(result.diagnostics.converged_mask)
    assert int(result.diagnostics.arnoldi_action_count) <= 2 * partial_policy.max_steps

    error_policy = GeneralEigenSolvePolicy(
        RestartedArnoldi(subspace_dimension=3),
        selection=GeneralEigenSelection("largest-magnitude", count=1),
        max_steps=2,
        tolerance=GeneralEigenTolerancePolicy(relative=0.0, absolute=0.0),
        failure=la.FailurePolicy("error"),
    )
    with pytest.raises(RuntimeError, match="numerical contract"):
        failed = general_eigensolve(problem, policy=error_policy)
        jax.block_until_ready(failed.eigenvalues)


def test_native_plan_rejects_retained_workspace_and_matvec_budget_overruns():
    problem = GeneralEigenproblem(
        _MatrixFreeOperator(
            jnp.diag(jnp.asarray([1.0, 2.0, 4.0, 7.0, 11.0])),
            operator_id="budgeted-matrix-free-arnoldi",
        )
    )
    method = RestartedArnoldi(subspace_dimension=4)
    selection = GeneralEigenSelection("largest-magnitude", count=1)
    baseline = plan_general_eigensolve(
        problem,
        GeneralEigenSolvePolicy(
            method,
            selection=selection,
            max_steps=20,
        ),
    )
    limits = (
        (
            "krylov_basis_bytes",
            baseline.cost.krylov_basis_bytes,
            "Krylov basis",
        ),
        ("workspace_bytes", baseline.cost.workspace_bytes, "workspace"),
        ("operator_matvecs", baseline.cost.operator_matvecs, "operator matvec"),
    )
    for field, required, message in limits:
        resources = GeneralEigenResourcePolicy(**{field: required - 1})
        with pytest.raises(ValueError, match=message):
            plan_general_eigensolve(
                problem,
                GeneralEigenSolvePolicy(
                    method,
                    selection=selection,
                    max_steps=20,
                    resources=resources,
                ),
            )


def test_general_eigen_preparation_refresh_preserves_symbolic_identity():
    first = GeneralEigenproblem(
        _dense(jnp.asarray([[1.0, 1.0], [0.0, 2.0]])),
        problem_id="refreshable-general-eigen",
    )
    prepared = prepare_general_eigensolve(first)
    second = GeneralEigenproblem(
        _dense(jnp.asarray([[1.5, 1.0], [0.0, 2.5]])),
        problem_id="refreshable-general-eigen",
    )
    refreshed = refresh_general_eigensolve(prepared, second)

    assert refreshed.prepared_id == prepared.prepared_id
    assert int(refreshed.refresh_count) == 1
    assert int(refreshed.numeric_version) == 1
    assert jnp.allclose(
        jnp.sort(jnp.real(general_eigensolve(refreshed).eigenvalues)),
        jnp.asarray([1.5, 2.5]),
    )


def test_repeated_cluster_derivative_returns_basis_invariant_projected_data():
    matrix = jnp.diag(jnp.asarray([2.0, 2.0, 5.0]))
    perturbation = jnp.asarray(
        [
            [1.0, 2.0, 0.5],
            [2.0, 3.0, -0.25],
            [0.75, 0.5, 4.0],
        ]
    )
    prepared = prepare_general_eigensolve(GeneralEigenproblem(_dense(matrix)))
    result = general_eigensolve(prepared)
    derivative = general_eigenvalue_derivative(
        prepared,
        result,
        perturbation,
        (0, 1),
    )
    rotation = jnp.asarray([[0.6, -0.8], [0.8, 0.6]], dtype=jnp.complex128)
    rotated_right = result.right_eigenvector_coordinates.at[:, :2].set(
        result.right_eigenvector_coordinates[:, :2] @ rotation
    )
    rotated_left = result.left_eigenvector_coordinates.at[:, :2].set(
        result.left_eigenvector_coordinates[:, :2] @ rotation
    )
    rotated_result = eqx.tree_at(
        lambda value: (
            value.right_eigenvector_coordinates,
            value.left_eigenvector_coordinates,
        ),
        result,
        (rotated_right, rotated_left),
    )
    rotated_derivative = general_eigenvalue_derivative(
        prepared,
        rotated_result,
        perturbation,
        (0, 1),
    )
    simple_derivative = general_eigenvalue_derivative(
        prepared,
        result,
        perturbation,
        2,
    )
    invalid_scalar = general_eigenvalue_derivative(
        prepared,
        result,
        perturbation,
        0,
    )
    expected = jnp.linalg.eigvalsh(perturbation[:2, :2])

    assert bool(derivative.successful)
    assert derivative.scalar_derivative is None
    assert not derivative.provenance.within_cluster_denominators
    assert jnp.allclose(
        jnp.sort(jnp.real(derivative.projected_eigenvalue_derivatives)),
        expected,
        atol=1e-9,
    )
    assert jnp.allclose(
        jnp.sort(jnp.real(rotated_derivative.projected_eigenvalue_derivatives)),
        expected,
        atol=1e-9,
    )
    assert jnp.allclose(derivative.trace_derivative, jnp.trace(perturbation[:2, :2]))
    assert jnp.allclose(simple_derivative.scalar_derivative, 4.0)
    assert not bool(invalid_scalar.successful)
    assert invalid_scalar.scalar_derivative is None


def test_generalized_simple_eigenvalue_derivative_includes_mass_perturbation():
    matrix = jnp.diag(jnp.asarray([2.0, 6.0]))
    mass = jnp.diag(jnp.asarray([1.0, 2.0]))
    prepared = prepare_general_eigensolve(
        GeneralEigenproblem(_dense(matrix), _dense(mass))
    )
    result = general_eigensolve(prepared)
    derivative = general_eigenvalue_derivative(
        prepared,
        result,
        jnp.diag(jnp.asarray([0.0, 4.0])),
        1,
        mass_perturbation=jnp.diag(jnp.asarray([0.0, 1.0])),
    )

    assert bool(derivative.successful)
    assert jnp.allclose(derivative.scalar_derivative, 0.5)


def test_cluster_projector_derivative_uses_only_external_spectral_gaps():
    matrix = jnp.diag(jnp.asarray([2.0, 2.0, 5.0]))
    perturbation = jnp.asarray(
        [
            [0.0, 1.0, 3.0],
            [2.0, 0.0, -6.0],
            [9.0, 12.0, 0.0],
        ]
    )
    prepared = prepare_general_eigensolve(GeneralEigenproblem(_dense(matrix)))
    result = general_eigensolve(prepared)
    derivative = general_invariant_projector_derivative(
        prepared,
        result,
        perturbation,
        (0, 1),
    )
    expected = jnp.asarray(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 2.0],
            [-3.0, -4.0, 0.0],
        ],
        dtype=jnp.complex128,
    )

    assert bool(derivative.successful)
    assert derivative.external_denominators.shape == (2, 1)
    assert jnp.allclose(derivative.external_denominators, -3.0)
    assert not derivative.provenance.used_internal_denominators
    assert (
        derivative.provenance.denominator_scope == "selected cluster to complement only"
    )
    assert jnp.allclose(derivative.value, expected, atol=1e-9)


def test_general_eigenvectors_preserve_complexified_pytree_space_structure():
    structure = {
        "position": jax.ShapeDtypeStruct((1,), jnp.float64),
        "velocity": jax.ShapeDtypeStruct((1,), jnp.float64),
    }
    space = la.PyTreeSpace(structure)
    problem = GeneralEigenproblem(
        _dense(jnp.asarray([[0.0, -1.0], [1.0, 0.0]]), space=space)
    )
    result = general_eigensolve(problem)

    assert bool(result.successful)
    assert set(result.right_eigenvectors) == {"position", "velocity"}
    assert result.right_eigenvectors["position"].shape == (1, 2)
    assert result.left_eigenvectors["velocity"].shape == (1, 2)
    assert jnp.issubdtype(
        result.right_eigenvectors["position"].dtype, jnp.complexfloating
    )
