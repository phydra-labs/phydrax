#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.units import METER


def test_projection_reconstruction_preserves_retained_inputs_and_binding() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.asarray([0.1, 0.2]),
        source_id="sphere-anchors",
        plan_id="retained-projection",
    )

    restored = phx.geometry.ImplicitPointProjectionPlan.from_stored_fields(
        plan, geometry=geometry
    )

    assert restored is not plan
    assert restored.kernel is geometry.kernel
    assert restored.schema is geometry.schema
    assert restored.tolerance is geometry.tolerance
    assert restored.policy is plan.policy
    assert restored.source_id == plan.source_id
    assert restored.plan_id == plan.plan_id
    assert jnp.array_equal(restored.anchors, plan.anchors)
    assert jnp.array_equal(restored.trust_radii, plan.trust_radii)
    assert eqx.tree_equal(restored.realize(geometry.state), plan.realize(geometry.state))


def test_projection_reconstruction_revalidates_stored_trust_radii() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        jnp.asarray([[1.0, 0.0, 0.0]]),
        0.1,
        source_id="sphere-anchor",
    )
    invalid = eqx.tree_at(lambda value: value.trust_radii, plan, jnp.asarray([-0.1]))

    with pytest.raises(ValueError, match="trust_radii must be positive"):
        phx.geometry.ImplicitPointProjectionPlan.from_stored_fields(
            invalid, geometry=geometry
        )


def test_projection_reconstruction_validates_anchors_against_supplied_geometry() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        jnp.asarray([[1.0, 0.0, 0.0]]),
        0.1,
        source_id="sphere-anchor",
    )
    other_geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 2.0, feature_id="sphere"
    ).compile(coordinate_unit=METER)

    with pytest.raises(ValueError, match="discovery zero set"):
        phx.geometry.ImplicitPointProjectionPlan.from_stored_fields(
            plan, geometry=other_geometry
        )
