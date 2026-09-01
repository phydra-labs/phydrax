#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def test_normal_gauge_projection_has_expected_sphere_radius_derivative():
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="sphere",
    ).compile()
    anchors = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        anchors,
        0.25,
        source_id="sphere-anchors",
    )
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))

    def coordinates(radius):
        state = geometry.state.replace_at(radius_index, radius)
        return plan.realize(state).proposed_points

    points, tangent = jax.jvp(
        coordinates,
        (jnp.asarray(1.1),),
        (jnp.asarray(1.0),),
    )
    result = eqx.filter_jit(plan.realize)(geometry.state.replace_at(radius_index, 1.1))
    assert jnp.allclose(points, 1.1 * anchors, atol=1.0e-7)
    assert jnp.allclose(tangent, anchors, atol=1.0e-6)
    assert bool(result.accepted)
    assert result.evidence.root_residual <= plan.policy.root_tolerance


def test_projection_rejects_trust_region_expiry_with_finite_fallback():
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="sphere",
    ).compile()
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        jnp.asarray([[1.0, 0.0, 0.0]]),
        0.1,
        source_id="sphere-anchor",
    )
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))
    state = geometry.state.replace_at(radius_index, jnp.asarray(1.5))

    result = eqx.filter_jit(plan.realize)(state)

    assert not bool(result.accepted)
    assert bool(result.refresh_required)
    assert jnp.all(jnp.isfinite(result.proposed_points))
    assert jnp.array_equal(result.points, plan.anchors)
