#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def run():
    marker = phx.solver.RelaxedHPMarking(2, 0.2)
    stable_ids = jnp.asarray(((0, 1), (0, 2), (0, 3)))

    def objective(scale):
        indicators = scale * jnp.asarray((1.0, 3.0, 2.0))
        weights = marker.weights(indicators, jnp.ones((3,), dtype=bool))
        return jnp.sum(weights * indicators)

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(0.5))
    selected = marker.safe_project(
        jnp.asarray((1.0, 3.0, 2.0)),
        jnp.ones((3,), dtype=bool),
        stable_ids,
    )
    return {
        "relaxed_objective": float(value),
        "gradient": float(gradient),
        "safe_selected": tuple(bool(value) for value in selected),
    }


if __name__ == "__main__":
    print(run())
