"""Independent dense references for the workspace-linear Lobatto action."""

from __future__ import annotations

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.discretization.collocation import ChebyshevDifferentiationAction


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))


def _dense_second_reference(nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Squaring the float64 dense D amplifies row-roundoff at the endpoints.
    # Evaluate that same independent dense matrix action at 80-digit precision.
    count = nodes.size
    with localcontext() as context:
        context.prec = 80
        x = [Decimal.from_float(float(value)) for value in nodes]
        c = [
            Decimal(-1 if index % 2 else 1) * (2 if index in (0, count - 1) else 1)
            for index in range(count)
        ]
        matrix: list[list[Decimal]] = []
        for row in range(count):
            entries = [
                c[row] / c[column] / (x[row] - x[column]) if row != column else Decimal(0)
                for column in range(count)
            ]
            entries[row] = -sum(entries, Decimal(0))
            matrix.append(entries)
        first = [
            [
                sum(
                    (
                        matrix[row][column]
                        * Decimal.from_float(float(values[column, field]))
                        for column in range(count)
                    ),
                    Decimal(0),
                )
                for field in range(values.shape[1])
            ]
            for row in range(count)
        ]
        second = [
            [
                float(
                    sum(
                        (
                            matrix[row][column] * first[column][field]
                            for column in range(count)
                        ),
                        Decimal(0),
                    )
                )
                for field in range(values.shape[1])
            ]
            for row in range(count)
        ]
    return np.asarray(second, dtype=np.float64)


@pytest.mark.parametrize("count", [17, 33, 65])
@pytest.mark.parametrize("bounds", [(-1.0, 1.0), (-2.0, 3.0)])
def test_action_matches_dense_derivatives(
    count: int, bounds: tuple[float, float]
) -> None:
    plan = ChebyshevDifferentiationAction(count, lower=bounds[0], upper=bounds[1])
    nodes = np.asarray(plan.nodes)
    coefficients = np.asarray((-1.0) ** np.arange(count), dtype=np.float64)
    coefficients[[0, -1]] *= 2.0
    difference = nodes[:, None] - nodes[None, :]
    matrix = (coefficients[:, None] / coefficients[None, :]) / (
        difference + np.eye(count)
    )
    matrix -= np.diag(np.sum(matrix, axis=1))
    values = np.stack((np.sin(nodes), nodes**4), axis=1)
    np.testing.assert_allclose(
        plan.derivative(values), matrix @ values, rtol=1e-10, atol=2e-11
    )
    np.testing.assert_allclose(
        plan.derivative(values, 2),
        _dense_second_reference(nodes, values),
        rtol=1e-10,
        atol=1e-10,
    )
    tangent = jnp.asarray(values)
    _, actual = jax.jvp(plan.derivative, (tangent,), (tangent,))
    np.testing.assert_allclose(actual, matrix @ values, rtol=1e-10, atol=2e-11)
    np.testing.assert_allclose(plan.derivative(jnp.ones(count)), 0.0, atol=0.0)
    assert sum(leaf.nbytes for leaf in jax.tree.leaves(plan)) == 16 * count


def test_action_rejects_invalid_count_and_values() -> None:
    with pytest.raises(ValueError):
        ChebyshevDifferentiationAction(2)
    with pytest.raises(ValueError):
        ChebyshevDifferentiationAction(17).derivative(np.zeros(16))
