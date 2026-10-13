from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax.linalg as la
from phydrax.linalg._constraint_operators import ConstraintOperatorPlan


class _SingleColumnSumOperator(la.AbstractLinearOperator):
    """A scalar-action provider with a one-column block capacity."""

    delegate: la.FunctionLinearOperator
    source: la.ArraySpace
    target: la.ArraySpace

    def __init__(self) -> None:
        self.source = la.ArraySpace((16,), dtype=jnp.float64)
        self.target = la.ArraySpace((1,), dtype=jnp.float64)
        self.delegate = la.FunctionLinearOperator(
            lambda vector: jnp.reshape(jnp.sum(vector), (1,)),
            source=self.source,
            target=self.target,
        )
        self.properties = self.delegate.properties
        self.capabilities = la.OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.batch_shape = ()
        self.operator_id = "single-column-sum"

    def mv(self, vector: Array, /) -> Array:
        return jnp.asarray(self.delegate.mv(vector))

    def transpose_mv(self, vector: Array, /) -> Array:
        return jnp.asarray(self.delegate.transpose_mv(vector))

    def adjoint_mv(self, vector: Array, /) -> Array:
        return jnp.asarray(self.delegate.adjoint_mv(vector))

    def mv_block(self, vectors: ArrayLike, /) -> Array:
        coordinates = jnp.asarray(vectors)
        if coordinates.shape[-1] > 1:
            raise ValueError("Provider block capacity exceeded")
        return self.delegate.mv_block(coordinates)

    def _materialize(self, /) -> Array:
        raise ValueError("Provider does not materialize")


@pytest.mark.strict_jax
def test_matrix_free_constraint_preparation_respects_single_column_capacity() -> None:
    operator = _SingleColumnSumOperator()
    prepared = ConstraintOperatorPlan(
        operator,
        materialization=la.MaterializationPolicy(max_entries=16, max_bytes=128),
    ).prepare()
    target = jnp.asarray([2.0], dtype=jnp.float64)
    lift = prepared.strict_right_inverse(target)
    np.testing.assert_allclose(lift, np.full((16,), 0.125), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(prepared.apply(lift), target, rtol=1e-12, atol=1e-12)
    assert prepared.evidence.operator_kind == "matrix-free"
    assert prepared.evidence.setup_matvec_count == 16
    assert prepared.evidence.operator_matrix_bytes == 128


def test_constraint_preparation_refuses_finite_failed_native_solve() -> None:
    matrix = jnp.asarray([[1.0, 1.0], [1.0, 1.0 + 1e-12]], dtype=jnp.float64)
    operator = la.DenseLinearOperator(matrix)
    factorization = la.factorize(operator, la.FactorizationPolicy("svd"))
    native = factorization.solve(
        jnp.eye(2, dtype=jnp.float64), rhs_layout=la.RHSLayout((2,))
    )
    assert bool(jnp.all(jnp.isfinite(native.value)))
    assert not bool(jnp.all(native.successful))
    with pytest.raises(
        RuntimeError, match="block solve failed: status=.*normal_residual_norm"
    ):
        ConstraintOperatorPlan(operator).prepare()


@pytest.mark.parametrize("shape", [(7, 3), (3, 7)])
@pytest.mark.parametrize("complex_values", [False, True])
def test_blocked_constraint_certification_matches_dense_reference(
    shape: tuple[int, int], complex_values: bool
) -> None:
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=shape)
    if complex_values:
        matrix = matrix + 1j * rng.normal(size=shape)
    rows, columns = shape
    workspace = 2 * (3 * rows + 2 * columns) * matrix.dtype.itemsize
    prepared = la.prepare_constraint_operator(
        la.DenseLinearOperator(jnp.asarray(matrix)),
        require_full_row_rank=rows <= columns,
        materialization=la.MaterializationPolicy(max_workspace_bytes=workspace),
    )
    right = np.linalg.pinv(matrix)
    projector = matrix @ right
    evidence = prepared.evidence
    np.testing.assert_allclose(prepared.right_inverse, right, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        evidence.generalized_right_inverse_residual_norm,
        np.linalg.norm(projector @ matrix - matrix),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        evidence.strict_right_inverse_residual_norm,
        np.linalg.norm(projector - np.eye(rows)),
        atol=1e-12,
    )
    expected_compatibility_tolerance = (
        1024
        * np.finfo(matrix.real.dtype).eps
        * max(shape)
        * max(np.linalg.norm(projector), 1)
    )
    np.testing.assert_allclose(
        evidence.compatibility_tolerance, expected_compatibility_tolerance, rtol=1e-12
    )
    basis = np.asarray(prepared.nullspace_basis)
    np.testing.assert_allclose(
        matrix @ basis, np.zeros((rows, prepared.nullity)), atol=1e-12
    )
    np.testing.assert_allclose(
        basis.conj().T @ right, np.zeros((prepared.nullity, rows)), atol=1e-12
    )
    compatible = matrix @ np.arange(columns, dtype=np.float64)
    np.testing.assert_allclose(
        prepared.minimum_norm_lift(jnp.asarray(compatible)),
        right @ compatible,
        atol=1e-12,
    )
    if rows > columns:
        with pytest.raises(ValueError, match="full-row-rank"):
            prepared.strict_right_inverse(jnp.asarray(compatible))
        with pytest.raises(ValueError, match="target dimension"):
            la.prepare_constraint_operator(la.DenseLinearOperator(jnp.asarray(matrix)))


def test_constraint_certification_refuses_single_column_workspace() -> None:
    matrix = jnp.asarray(np.random.default_rng(23).normal(size=(7, 3)))
    with pytest.raises(ValueError, match="cannot fit one column"):
        la.prepare_constraint_operator(
            la.DenseLinearOperator(matrix),
            require_full_row_rank=False,
            materialization=la.MaterializationPolicy(max_workspace_bytes=215),
        )


def test_duplicate_generalized_constraints_match_dense_minimum_norm_reference() -> None:
    matrix = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    operator = la.DenseLinearOperator(jnp.asarray(matrix))
    prepared = la.prepare_constraint_operator(operator, require_full_row_rank=False)
    reference = np.linalg.pinv(matrix)
    np.testing.assert_allclose(prepared.right_inverse, reference, atol=1e-12)
    assert prepared.rank == 1
    assert prepared.nullity == 3
    assert not prepared.evidence.full_row_rank
    assert (
        prepared.evidence.generalized_right_inverse_residual_norm
        <= prepared.evidence.residual_tolerance
    )
    residual = matrix @ np.asarray(prepared.right_inverse) - np.eye(2, dtype=np.float64)
    np.testing.assert_allclose(matrix.T @ residual, 0.0, atol=1e-12)
    compatible = jnp.asarray([2.0, 2.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        prepared.minimum_norm_lift(compatible), reference @ compatible, atol=1e-12
    )
    assert bool(prepared.is_compatible(compatible))
    assert not bool(prepared.is_compatible(jnp.asarray([1.0, -1.0], dtype=jnp.float64)))
    with pytest.raises(ValueError, match="full-row-rank"):
        prepared.strict_right_inverse(compatible)
    with pytest.raises(ValueError, match="full row rank"):
        la.prepare_constraint_operator(operator)
