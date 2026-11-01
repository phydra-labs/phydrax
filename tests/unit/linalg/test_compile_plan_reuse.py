#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.linalg import _operators, _pairings, _runtime
from phydrax.sparse import _linear


la = phx.linalg


def test_pairing_shape_validation_reuses_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    eqx.clear_caches()
    calls = []
    original = _pairings.EuclideanPairing.riesz

    def riesz(self: _pairings.EuclideanPairing, vector: Any) -> Any:
        calls.append(None)
        return original(self, vector)

    monkeypatch.setattr(_pairings.EuclideanPairing, "riesz", riesz)
    first = la.ArraySpace((7,), dtype=jnp.float64)
    count = len(calls)
    second = la.ArraySpace((7,), dtype=jnp.float64)
    assert count > 0
    assert len(calls) == count
    assert first.space_id == second.space_id


def test_diagonal_shape_and_execution_share_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eqx.clear_caches()
    calls = []
    original = _operators.DenseLinearOperator._assemble_diagonal

    def diagonal(self: _operators.DenseLinearOperator) -> jax.Array:
        calls.append(None)
        return original(self)

    monkeypatch.setattr(_operators.DenseLinearOperator, "_assemble_diagonal", diagonal)
    operator = la.DenseLinearOperator(jnp.asarray([[2.0, 1.0], [1.0, 3.0]]))
    np.testing.assert_array_equal(la.assemble_diagonal(operator), [2.0, 3.0])
    assert len(calls) == 1
    changed = eqx.tree_at(lambda op: op.matrix, operator, operator.matrix * 2)
    np.testing.assert_array_equal(la.assemble_diagonal(changed), [4.0, 6.0])
    assert len(calls) == 1


def test_prepared_inverse_and_custom_root_reuse_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eqx.clear_caches()
    calls = []
    original = _runtime.custom_root

    def root(*args: Any, **kwargs: Any) -> Any:
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(_runtime, "custom_root", root)
    matrix = jnp.asarray([[3.0, 0.5], [0.5, 2.0]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    prepared = la.prepare(la.LinearSystem(operator), la.LinearSolvePolicy(la.DenseLU()))
    rhs = jnp.asarray([1.0, -2.0])
    np.testing.assert_allclose(
        la.solve(prepared, rhs).value, jnp.linalg.solve(matrix, rhs)
    )
    count = len(calls)
    np.testing.assert_allclose(
        la.solve(prepared, rhs * 2).value, jnp.linalg.solve(matrix, rhs * 2)
    )
    assert count > 0
    assert len(calls) == count
    pairing = la.OperatorPairing(operator, prepared_inverse=prepared)
    np.testing.assert_allclose(pairing.inverse_riesz(rhs), jnp.linalg.solve(matrix, rhs))
    count = len(calls)
    np.testing.assert_allclose(
        pairing.inverse_riesz(rhs * 2), jnp.linalg.solve(matrix, rhs * 2)
    )
    assert len(calls) == count
    _, tangent = jax.jvp(pairing.inverse_riesz, (rhs,), (rhs,))
    _, transpose = jax.vjp(pairing.inverse_riesz, rhs)
    np.testing.assert_allclose(tangent, jnp.linalg.solve(matrix, rhs))
    np.testing.assert_allclose(transpose(rhs)[0], jnp.linalg.solve(matrix.T, rhs))


def test_eager_route_accumulation_reuses_trace_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax.clear_caches()
    calls = []
    original = _linear._accumulate_slot

    def accumulate(
        slot: jax.Array, carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        calls.append(None)
        return original(slot, carry)

    monkeypatch.setattr(_linear, "_accumulate_slot", accumulate)
    products = jnp.asarray([[1e16, 2.0], [1.0, 3.0], [-1e16, 4.0]])
    np.testing.assert_array_equal(_linear._sum_route_products(products), [0.0, 9.0])
    count = len(calls)
    np.testing.assert_array_equal(_linear._sum_route_products(products * 2), [0.0, 18.0])
    assert count > 0
    assert len(calls) == count
