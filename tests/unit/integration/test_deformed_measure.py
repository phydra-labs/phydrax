#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.integration._deformed_measure import DeformedMeasurePlan


def test_volume_measure_tracks_the_dynamic_jacobian():
    plan = DeformedMeasurePlan("volume", jnp.asarray((0.25, 0.75)))
    deformation = jnp.asarray(
        (
            ((2.0, 0.0), (0.0, 3.0)),
            ((1.5, 0.2), (0.0, 2.0)),
        )
    )
    state = plan.evaluate(deformation)
    np.testing.assert_allclose(state.jacobian, jnp.asarray((6.0, 3.0)))
    np.testing.assert_allclose(state.current_measure, jnp.asarray((1.5, 2.25)))
    assert bool(state.valid)
    derivative = jax.grad(
        lambda stretch: jnp.sum(
            plan.evaluate(
                stretch * jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
            ).current_measure
        )
    )(jnp.asarray(2.0))
    np.testing.assert_allclose(derivative, 4.0)


def test_surface_measure_uses_nanson_and_preserves_oriented_normals():
    normals = jnp.asarray(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    plan = DeformedMeasurePlan(
        "surface",
        jnp.asarray((2.0, 3.0)),
        reference_normal=normals,
    )
    deformation = jnp.asarray(
        (
            ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
            ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        )
    )
    state = plan.evaluate(deformation)
    np.testing.assert_allclose(state.measure_ratio, jnp.asarray((12.0, 8.0)))
    np.testing.assert_allclose(state.current_measure, jnp.asarray((24.0, 24.0)))
    np.testing.assert_allclose(state.current_normal, normals)
    np.testing.assert_allclose(state.normal("reference"), normals)
    assert bool(state.valid)


def test_deformed_measure_refuses_invalid_reference_data_and_inversion():
    with pytest.raises(ValueError, match="strictly positive"):
        DeformedMeasurePlan("volume", jnp.asarray((1.0, 0.0)))
    with pytest.raises(ValueError, match="oriented reference normal"):
        DeformedMeasurePlan("surface", jnp.asarray((1.0,)))

    plan = DeformedMeasurePlan("volume", jnp.asarray(1.0))
    inverted = plan.evaluate(jnp.asarray(((-1.0, 0.0), (0.0, 1.0))))
    assert not bool(inverted.valid)
    assert jnp.isnan(inverted.current_measure)
