"""White-box resource accounting for prepared approximation interval graphs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.geometry._interval_enclosure import PreparedIntervalFunction
from phydrax.geometry.brep._curve_approximation_bounds import _graph_storage


def test_interval_graph_storage_counts_nested_conditional_live_buffers() -> None:
    def branch(point: Array) -> Array:
        return jnp.sum(jnp.broadcast_to(point, (1, 128)), axis=1)

    def reversed_branch(point: Array) -> Array:
        return jnp.sum(-jnp.broadcast_to(point, (1, 128)), axis=1)

    def program(point: Array) -> Array:
        return jax.lax.cond(point[0, 0] > 0.0, branch, reversed_branch, point)

    graph = jax.make_jaxpr(program)(jax.ShapeDtypeStruct((1, 1), np.float64))
    prepared = PreparedIntervalFunction(graph, 1, (), 1)
    retained, workspace = _graph_storage(prepared)
    assert retained == 0
    # The nested broadcast is 128 binary64 values; the interpreter's declared
    # twenty-buffer allowance alone must cover this branch workspace.
    assert workspace >= 20 * 128 * np.dtype(np.float64).itemsize


def test_interval_graph_storage_deduplicates_shared_coefficient_bounds() -> None:
    def program(point: Array) -> Array:
        return point

    graph = jax.make_jaxpr(program)(jax.ShapeDtypeStruct((1, 1), np.float64))
    coefficients = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    prepared = PreparedIntervalFunction(
        graph, 1, (1,), 1, ((coefficients, coefficients, coefficients),)
    )
    retained, workspace = _graph_storage(prepared)
    assert retained == coefficients.nbytes
    assert workspace > 0
