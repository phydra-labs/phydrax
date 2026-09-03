from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _geometry():
    prepared = phx.discretization.SurfelSetPlan(
        jnp.asarray((5,)),
        jnp.asarray(((0.0, 0.0, 0.0),)),
        jnp.asarray((1.0,)),
    ).prepare()
    return phx.discretization.SurfelGeometryPlan(prepared).materialize(
        prepared.reference_position,
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray(([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]],)),
    )


def test_surfel_footprint_evaluates_tangent_coordinates_and_plane_distance() -> None:
    geometry = _geometry()
    plan = phx.discretization.SurfelFootprintPlan(3)
    points = jnp.asarray(((0.0, 0.0, 0.5), (2.0, 0.0, -0.25), (2.1, 0.0, 0.0)))
    result = plan.evaluate(geometry, points, jnp.zeros((3,), dtype=jnp.int32))
    np.testing.assert_allclose(result.signed_normal_distance, [0.5, -0.25, 0.0])
    np.testing.assert_allclose(result.tangent_coordinates[:2], [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(result.inside, [True, True, False])
    assert result.kernel_weight[0] == 1.0
    assert result.kernel_weight[1] == 0.0


def test_surfel_footprint_query_gradient_is_finite() -> None:
    geometry = _geometry()
    plan = phx.discretization.SurfelFootprintPlan(3)

    def objective(point):
        result = plan.evaluate(
            geometry, point[None, :], jnp.asarray((0,), dtype=jnp.int32)
        )
        return result.signed_normal_distance[0] + result.normalized_radius_squared[0]

    gradient = jax.grad(objective)(jnp.asarray((0.5, 0.25, 0.2)))
    np.testing.assert_allclose(gradient, [0.25, 0.5, 1.0], rtol=1e-12, atol=1e-12)
