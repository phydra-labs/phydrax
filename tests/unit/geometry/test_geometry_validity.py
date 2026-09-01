#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_parameter_bounds_and_superquadric_conditions_are_executable():
    sphere = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="sphere",
    ).compile()
    radius = sphere.schema.index(phx.geometry.ParameterId("sphere", "radius"))
    invalid_sphere = sphere.state.replace_at(radius, jnp.asarray(-1.0))

    assert bool(sphere.validity().accepted)
    assert not bool(sphere.validity(invalid_sphere).accepted)
    assert int(sphere.validity(invalid_sphere).disposition) == int(
        phx.geometry.GeometryValidityDisposition.INVALID
    )

    superquadric = (
        phx.geometry.Superquadric(
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            first_blockiness=2.5,
            second_blockiness=3.0,
            feature_id="sq",
        )
        .translated((1.0, 0.0, 0.0))
        .compile()
    )
    orientation = superquadric.schema.index(phx.geometry.ParameterId("sq", "orientation"))
    invalid_orientation = superquadric.state.replace_at(
        orientation,
        jnp.zeros((4,)),
    )

    assert bool(superquadric.validity().accepted)
    assert not bool(superquadric.validity(invalid_orientation).accepted)


def test_geometry_validity_is_jittable_and_keeps_fixed_evidence_shape():
    compiled = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="body",
    ).compile()
    radius = compiled.schema.index(phx.geometry.ParameterId("body", "radius"))

    def evaluate(value):
        state = compiled.state.replace_at(radius, value)
        evidence = compiled.validity(state)
        return evidence.accepted, evidence.margins

    accepted, margins = jax.jit(evaluate)(jnp.asarray(2.0))
    rejected, rejected_margins = jax.jit(evaluate)(jnp.asarray(-0.5))

    assert bool(accepted)
    assert not bool(rejected)
    assert margins.shape == rejected_margins.shape
    assert jnp.all(jnp.isfinite(margins))
