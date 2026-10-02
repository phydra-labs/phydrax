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
