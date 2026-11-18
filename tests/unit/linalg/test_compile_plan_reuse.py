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
from phydrax.linalg import _pairings
from phydrax.sparse import _linear


la = phx.linalg


def test_equal_array_spaces_share_one_identity() -> None:
    first = la.ArraySpace((7,), dtype=jnp.float64)
    second = la.ArraySpace((7,), dtype=jnp.float64)
    assert first.space_id == second.space_id


def test_diagonal_assembly_follows_changed_matrix_leaves() -> None:
    operator = la.DenseLinearOperator(jnp.asarray([[2.0, 1.0], [1.0, 3.0]]))
    np.testing.assert_array_equal(la.assemble_diagonal(operator), [2.0, 3.0])
    # A second operator of the same structure must not see the first one's values.
    changed = eqx.tree_at(lambda op: op.matrix, operator, operator.matrix * 2)
    np.testing.assert_array_equal(la.assemble_diagonal(changed), [4.0, 6.0])


def test_prepared_inverse_serves_repeated_right_hand_sides_and_derivatives() -> None:
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
    np.testing.assert_allclose(
        la.solve(prepared, rhs * 2).value, jnp.linalg.solve(matrix, rhs * 2)
    )
    pairing = la.OperatorPairing(operator, prepared_inverse=prepared)
    np.testing.assert_allclose(pairing.inverse_riesz(rhs), jnp.linalg.solve(matrix, rhs))
    np.testing.assert_allclose(
        pairing.inverse_riesz(rhs * 2), jnp.linalg.solve(matrix, rhs * 2)
    )
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


@pytest.mark.parametrize("action", ("riesz", "inverse"))
@pytest.mark.parametrize("defect", ("shape", "dtype", "pytree"))
def test_space_refuses_invalid_pairing_before_downstream_actions(
    monkeypatch: pytest.MonkeyPatch, action: str, defect: str
) -> None:
    eqx.clear_caches()
    calls: list[str] = []

    def malformed(vector: Any) -> Any:
        if defect == "shape":
            return vector[:-1]
        if defect == "dtype":
            return vector.astype(jnp.float32)
        return (vector,)

    def riesz(self: _pairings.EuclideanPairing, vector: Any) -> Any:
        calls.append("riesz")
        return malformed(vector) if action == "riesz" else vector

    def inverse(self: _pairings.EuclideanPairing, vector: Any) -> Any:
        calls.append("inverse")
        if action == "riesz":
            raise AssertionError("An invalid Riesz result reached the inverse.")
        return malformed(vector)

    def inner(self: _pairings.EuclideanPairing, left: Any, right: Any) -> Any:
        calls.append("inner")
        raise AssertionError("An invalid pairing action reached the inner product.")

    monkeypatch.setattr(_pairings.EuclideanPairing, "riesz", riesz)
    monkeypatch.setattr(_pairings.EuclideanPairing, "inverse_riesz", inverse)
    monkeypatch.setattr(_pairings.EuclideanPairing, "inner", inner)
    name = "Riesz" if action == "riesz" else "inverse-Riesz"
    with pytest.raises(
        TypeError, match=f"Pairing {name} output must match the vector-space structure."
    ):
        la.ArraySpace((3,), dtype=jnp.float64, pairing=la.EuclideanPairing())
    assert calls == (["riesz"] if action == "riesz" else ["riesz", "inverse"])
