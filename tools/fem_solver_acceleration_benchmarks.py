#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from time import perf_counter

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


def _average_time(action, value, iterations=50):
    compiled = eqx.filter_jit(action)
    compiled(value).block_until_ready()
    start = perf_counter()
    for _ in range(iterations):
        compiled(value).block_until_ready()
    return (perf_counter() - start) / iterations


def run():
    points = 8
    derivative = jnp.asarray(
        [
            [
                0.0 if row == column else (-1.0) ** (row + column) / (row - column)
                for column in range(points)
            ]
            for row in range(points)
        ]
    )
    elements = 16
    local_size = points * points
    gathers = jnp.arange(elements * local_size, dtype=jnp.int32).reshape(
        (elements, local_size)
    )
    metric = jnp.ones((elements, points, points, 3))
    mass = jnp.ones((elements, points, points))
    operator = phx.equations.fem.CollocatedTensorProductOperator(
        derivative,
        metric,
        mass,
        gathers,
        elements * local_size,
    )
    value = jnp.linspace(0.0, 1.0, elements * local_size)
    collocated_time = _average_time(operator.mv, value)

    space = phx.linalg.ArraySpace((2,))
    dense = phx.linalg.DenseLinearOperator(
        jnp.asarray([[2.0, 0.0], [0.0, 3.0]]), source=space, target=space
    )
    history = phx.linalg.LinearSolveHistory.empty(
        dense,
        phx.linalg.LinearSolveHistoryPolicy("projection", capacity=3),
        "benchmark-family",
    )
    history = history.update(dense, jnp.asarray([1.0, 0.0]), time=0.0)
    history = history.update(dense, jnp.asarray([0.0, 1.0]), time=1.0)
    guess, diagnostics = history.initial_guess(dense.mv(jnp.asarray([2.0, 3.0])))

    return {
        "collocated_apply_seconds": collocated_time,
        "collocated_dofs": int(value.size),
        "history_effective_dimension": int(diagnostics.effective_dimension),
        "history_projection_residual": float(diagnostics.projection_residual_norm),
        "history_guess_norm": float(jnp.linalg.norm(guess)),
    }


if __name__ == "__main__":
    print(run())
