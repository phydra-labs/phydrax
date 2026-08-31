#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    velocity = jnp.asarray(((0.2, 0.0), (0.2, 0.0)))
    metric = phx.equations.fem.ALEMetricState(
        coordinates,
        velocity,
        jnp.ones((2,)),
        jnp.asarray((0.1, -0.1)),
        jnp.asarray((-0.1, 0.1)),
    )
    time = phx.equations.fem.LocalTimeSteppingPlan(
        jnp.asarray((0.1, 0.05)), jnp.asarray((0, 1))
    )
    return {
        "temporal_gcl_defect": float(jnp.max(jnp.abs(metric.temporal_gcl_defect))),
        "local_steps": tuple(float(value) for value in time.level_steps),
    }


if __name__ == "__main__":
    print(run())
