#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _materialize(operator):
    return la.materialize(
        operator,
        la.MaterializationPolicy(max_entries=10_000, max_bytes=1_000_000),
    )


def test_dct_diagonal_operator_actions_properties_and_structured_solve():
    spectrum = jnp.asarray([1.0, 2.0, 4.0, 8.0])
    operator = la.TransformDiagonalLinearOperator(
        spectrum,
        transform="dct",
        spectral_property="positive-definite",
    )
    matrix = _materialize(operator)
    vector = jnp.asarray([1.0, -2.0, 0.5, 3.0])
    problem = la.LinearSystem(operator)
    prepared = la.prepare(problem)

    result = la.solve(prepared, vector)
    compiled = jax.jit(la.solve)(prepared, vector)

    assert operator.properties.certifies("self_adjoint")
    assert operator.properties.certifies("positive_definite")
    assert operator.properties.rank == 4
    assert jnp.allclose(operator.mv(vector), matrix @ vector)
    assert jnp.allclose(operator.transpose_mv(vector), matrix.T @ vector)
    assert jnp.allclose(operator.adjoint_mv(vector), matrix.T @ vector)
    assert prepared.plan.backend == "jax-structured"
    assert prepared.plan.method == "structured-direct"
    assert result.successful
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, vector), atol=1e-12)
    assert jnp.allclose(compiled.value, result.value, atol=1e-12)
    cost = la.estimate_operator_action_cost(operator)
    assert cost.exact
    assert cost.operation_class == "transform-diagonal-action"


def test_fft_diagonal_operator_has_correct_complex_transpose_and_adjoint():
    spectrum = jnp.asarray([1.0 + 0.5j, 2.0 - 1.0j, 3.0 + 0.25j])
    operator = la.TransformDiagonalLinearOperator(
        spectrum,
        transform="fft",
        nonsingular=True,
    )
    matrix = _materialize(operator)
    vector = jnp.asarray([1.0 + 2.0j, -1.0 + 0.5j, 3.0 - 2.0j])

    assert not operator.properties.certifies("self_adjoint")
    assert jnp.allclose(operator.mv(vector), matrix @ vector, atol=1e-12)
    assert jnp.allclose(operator.transpose_mv(vector), matrix.T @ vector, atol=1e-12)
    assert jnp.allclose(
        operator.adjoint_mv(vector),
        jnp.conj(matrix.T) @ vector,
        atol=1e-12,
    )
    result = la.solve(la.LinearSystem(operator), vector)
    assert result.successful
    assert jnp.allclose(operator.mv(result.value), vector, atol=1e-12)


def test_transform_diagonal_supports_multiaxis_event_shapes():
    spectrum = jnp.arange(1.0, 7.0).reshape((2, 3))
    space = la.ArraySpace((2, 3), dtype=jnp.float64)
    operator = la.TransformDiagonalLinearOperator(
        spectrum,
        space=space,
        transform="dct",
        axes=(0, 1),
        spectral_property="positive-definite",
    )
    vector = jnp.arange(6.0).reshape((2, 3))
    transformed = operator.to_transform_coordinates(vector)

    assert jnp.allclose(operator.from_transform_coordinates(transformed), vector)
    assert jnp.allclose(
        operator.mv(vector).reshape((-1,)),
        _materialize(operator) @ vector.reshape((-1,)),
        atol=1e-12,
    )


def test_transform_diagonal_singular_status_and_property_validation():
    operator = la.TransformDiagonalLinearOperator(
        jnp.asarray([1.0, 0.0, 2.0]),
        transform="dct",
    )
    policy = la.LinearSolvePolicy(
        la.StructuredDirect(),
        failure=la.FailurePolicy("status"),
    )
    result = la.solve(
        la.LinearSystem(operator),
        jnp.ones(3),
        policy=policy,
    )
    assert result.status == int(la.LinearSolveStatus.SINGULAR)
    with pytest.raises(ValueError, match="violates"):
        la.TransformDiagonalLinearOperator(
            jnp.asarray([1.0, -1.0, 2.0]),
            transform="dct",
            spectral_property="positive-definite",
        )
    with pytest.raises(TypeError, match="complex coordinates"):
        la.TransformDiagonalLinearOperator(
            jnp.ones(3),
            transform="fft",
        )


def test_transform_diagonal_structured_solve_differentiates_through_spectrum():
    rhs = jnp.asarray([1.0, -2.0, 3.0])

    def specialized(spectrum):
        operator = la.TransformDiagonalLinearOperator(
            spectrum,
            transform="dct",
            spectral_property="positive-definite",
        )
        policy = la.LinearSolvePolicy(
            la.StructuredDirect(),
            differentiation=la.DifferentiationPolicy("mathematical"),
        )
        return jnp.sum(la.solve(la.LinearSystem(operator), rhs, policy=policy).value)

    spectrum = jnp.asarray([1.5, 2.0, 4.0])
    actual = jax.jit(jax.grad(specialized))(spectrum)

    def spectral_formula(values):
        operator = la.TransformDiagonalLinearOperator(
            jax.lax.stop_gradient(values),
            transform="dct",
            spectral_property="positive-definite",
        )
        transformed_rhs = operator.to_transform_coordinates(rhs)
        return jnp.sum(operator.from_transform_coordinates(transformed_rhs / values))

    expected = jax.grad(spectral_formula)(spectrum)
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    ("matrix", "candidate", "expected_type"),
    [
        (
            jnp.diag(jnp.asarray([1.0, 2.0, 3.0])),
            "diagonal",
            la.DiagonalLinearOperator,
        ),
        (
            jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            "permutation",
            la.PermutationLinearOperator,
        ),
        (
            jnp.asarray([[2.0, 1.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]]),
            "tridiagonal",
            la.TridiagonalLinearOperator,
        ),
        (
            jnp.asarray([[2.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 6.0, 7.0]]),
            "triangular",
            la.TriangularLinearOperator,
        ),
        (
            jnp.asarray(
                [
                    [2.0, 1.0, 0.0, 0.0],
                    [3.0, 4.0, 5.0, 0.0],
                    [6.0, 7.0, 8.0, 9.0],
                    [0.0, 10.0, 11.0, 12.0],
                ]
            ),
            "banded",
            la.BandedLinearOperator,
        ),
    ],
)
def test_structure_compiler_recovers_exact_native_structures(
    matrix, candidate, expected_type
):
    policy = la.StructureCompilationPolicy(
        candidates=(candidate,),
        max_bandwidth=3,
        fallback="error",
    )
    compiled = la.compile_linear_structure(matrix, policy)

    assert compiled.structure == candidate
    assert compiled.exact
    assert isinstance(compiled.operator, expected_type)
    assert compiled.discarded_norm == 0
    assert jnp.array_equal(_materialize(compiled.operator), matrix)


def test_structure_compiler_requires_explicit_approximation_consent():
    matrix = jnp.asarray([[2.0, 1e-8], [0.0, 3.0]])
    with pytest.raises(ValueError, match="require allow_approximation"):
        la.StructureCompilationPolicy(absolute_tolerance=1e-7)

    exact = la.compile_linear_structure(
        matrix,
        la.StructureCompilationPolicy(
            candidates=("diagonal",),
            fallback="dense",
        ),
    )
    approximate = la.compile_linear_structure(
        matrix,
        la.StructureCompilationPolicy(
            candidates=("diagonal",),
            absolute_tolerance=1e-7,
            allow_approximation=True,
            fallback="error",
        ),
    )

    assert exact.structure == "dense"
    assert exact.exact
    assert approximate.structure == "diagonal"
    assert not approximate.exact
    assert approximate.discarded_norm > 0
    assert jnp.allclose(
        _materialize(approximate.operator),
        jnp.diag(jnp.diag(matrix)),
    )


def test_structure_compiler_detects_dct_and_fft_diagonalization():
    dct_operator = la.TransformDiagonalLinearOperator(
        jnp.asarray([1.0, 2.0, 4.0, 8.0]),
        transform="dct",
        spectral_property="positive-definite",
    )
    dct_policy = la.StructureCompilationPolicy(
        candidates=("dct-diagonal",),
        absolute_tolerance=1e-12,
        allow_approximation=True,
        fallback="error",
    )
    dct_compiled = la.compile_linear_structure(_materialize(dct_operator), dct_policy)

    fft_operator = la.TransformDiagonalLinearOperator(
        jnp.asarray([1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 0.5j, 4.0 + 0.25j]),
        transform="fft",
        nonsingular=True,
    )
    fft_policy = la.StructureCompilationPolicy(
        candidates=("fft-diagonal",),
        absolute_tolerance=1e-12,
        allow_approximation=True,
        fallback="error",
    )
    fft_compiled = la.compile_linear_structure(_materialize(fft_operator), fft_policy)

    assert dct_compiled.structure == "dct-diagonal"
    assert fft_compiled.structure == "fft-diagonal"
    assert isinstance(dct_compiled.operator, la.TransformDiagonalLinearOperator)
    assert isinstance(fft_compiled.operator, la.TransformDiagonalLinearOperator)
    assert jnp.allclose(
        _materialize(dct_compiled.operator),
        _materialize(dct_operator),
        atol=1e-12,
    )
    assert jnp.allclose(
        _materialize(fft_compiled.operator),
        _materialize(fft_operator),
        atol=1e-12,
    )


def test_structure_refresh_preserves_identity_and_rejects_structural_drift():
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))
    policy = la.StructureCompilationPolicy(
        candidates=("diagonal",),
        fallback="dense",
    )
    compiled = la.compile_linear_structure(matrix, policy)
    refreshed_matrix = jnp.diag(jnp.asarray([2.0, 3.0, 4.0]))
    refreshed = la.refresh_linear_structure(compiled, refreshed_matrix)

    assert refreshed.structure == "diagonal"
    assert refreshed.compiler_id == compiled.compiler_id
    assert refreshed.operator.operator_id == compiled.operator.operator_id
    assert refreshed.numeric_version == 1
    assert jnp.array_equal(_materialize(refreshed.operator), refreshed_matrix)

    drifted = refreshed_matrix.at[0, 1].set(1.0)
    with pytest.raises(ValueError, match="no longer satisfy"):
        la.refresh_linear_structure(refreshed, drifted)
    recompiled = la.refresh_linear_structure(refreshed, drifted, recompile=True)
    assert recompiled.structure == "dense"
    assert recompiled.numeric_version == 2
