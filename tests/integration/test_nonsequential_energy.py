#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.optics.geometric._interface import OpticalRayState
from phydrax.optics.geometric._nonsequential import (
    NonSequentialOpticsPlan,
    NonSequentialSurfaceTable,
    prepare_nonsequential_optics,
    trace_nonsequential_optics,
)


def test_finite_dielectric_reflection_tree_closes_launched_power():
    vertices = jnp.asarray(
        [
            [-50.0, -50.0, 0.0],
            [50.0, -50.0, 0.0],
            [50.0, 50.0, 0.0],
            [-50.0, 50.0, 0.0],
            [-50.0, -50.0, 1.0],
            [50.0, -50.0, 1.0],
            [50.0, 50.0, 1.0],
            [-50.0, 50.0, 1.0],
        ]
    )
    triangles = jnp.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=jnp.int32)
    surfaces = NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([0, 0, 1, 1]),
        jnp.asarray([1, 1, 0, 0]),
        jnp.asarray([1.0, 1.5]),
        surface_ids=jnp.asarray([0, 0, 1, 1]),
    )
    prepared = prepare_nonsequential_optics(
        NonSequentialOpticsPlan(
            surfaces,
            maximum_interactions=8,
            branch_capacity=32,
            power_tolerance=0.0,
        )
    )
    rays = OpticalRayState(
        jnp.asarray([[0.0, 0.0, -2.0], [1.0, 0.5, -2.0]]),
        jnp.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        jnp.ones((2,)),
    )
    result = trace_nonsequential_optics(
        prepared, rays, jnp.asarray([1.0, 0.4]), jnp.zeros((2,), dtype=jnp.int32)
    )

    np.testing.assert_allclose(result.power_ledger_residual, 0.0, atol=3e-6)
    accounted = (
        result.absorbed_power
        + result.detected_power.sum(axis=-1)
        + result.escaped_power
        + result.discarded_power
        + result.ambiguous_power
        + result.truncated_power
        + result.live_power
    )
    np.testing.assert_allclose(accounted, result.launched_power, atol=3e-6)
    assert bool(jnp.all(result.truncated_power >= 0.0))
    assert bool(jnp.all(result.live_power >= 0.0))
